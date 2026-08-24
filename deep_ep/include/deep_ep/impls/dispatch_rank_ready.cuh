#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

// Build the ordinary rank-major and expert-major prefix sums after this
// destination rank has received every source lane. This kernel intentionally
// waits on destination-local payload doorbells instead of dispatch tag1.
template <int kNumRanks, int kNumExperts, int kExpertAlignment,
          int64_t kNumTimeoutCycles, int kNumThreads = 32>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_rank_ready_prefix_impl(void* workspace,
                                int* psum_num_recv_tokens_per_rank,
                                int* psum_num_recv_tokens_per_expert,
                                int* num_unaligned_recv_tokens_per_expert,
                                int* cumulative_local_expert_recv_stats,
                                const int destination_rank_idx,
                                const uint64_t generation) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid expert/rank shape");
    EP_STATIC_ASSERT(kNumRanks <= 32, "Rank-ready prefix supports at most one warp of ranks");
    EP_STATIC_ASSERT(kNumExpertsPerRank <= layout::WorkspaceLayout::kNumMaxExpertsPerRank,
                     "Too many local experts");

    if (threadIdx.x != 0)
        return;

    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);

    // One thread performs all acquires so visibility is carried directly into
    // the count reads below; the work is only O(ranks * local_experts).
    for (int source_rank_idx = 0; source_rank_idx < kNumRanks; ++ source_rank_idx) {
        const auto* lane = workspace_layout.get_streaming_lane_control_ptr(source_rank_idx);
        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
            if (streaming::acquire_payload_ready(lane, generation))
                return true;
            if (is_last_check) {
                printf("DeepEP rank-ready timeout, dst: %d, src: %d, expected generation: %llu, observed: %llu\n",
                       destination_rank_idx, source_rank_idx,
                       static_cast<unsigned long long>(generation),
                       static_cast<unsigned long long>(ptx::ld_acquire_sys(&lane->payload_done_seq)));
            }
            return false;
        });
    }

    int rank_end = 0;
    for (int source_rank_idx = 0; source_rank_idx < kNumRanks; ++ source_rank_idx) {
        const auto count = *workspace_layout.get_streaming_lane_token_count_ptr(source_rank_idx);
        EP_DEVICE_ASSERT(count >= 0);
        rank_end += static_cast<int>(count);
        psum_num_recv_tokens_per_rank[source_rank_idx] = rank_end;
    }

    int expert_start = 0;
    for (int local_expert_idx = 0; local_expert_idx < kNumExpertsPerRank; ++ local_expert_idx) {
        int count = 0;
        for (int source_rank_idx = 0; source_rank_idx < kNumRanks; ++ source_rank_idx) {
            const auto lane_count = *workspace_layout.get_streaming_lane_expert_count_ptr(
                source_rank_idx, local_expert_idx);
            EP_DEVICE_ASSERT(lane_count >= 0);
            count += static_cast<int>(lane_count);
        }
        if (num_unaligned_recv_tokens_per_expert != nullptr)
            num_unaligned_recv_tokens_per_expert[local_expert_idx] = count;
        if (cumulative_local_expert_recv_stats != nullptr)
            atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, count);
        psum_num_recv_tokens_per_expert[local_expert_idx] = expert_start;
        expert_start += math::align(count, kExpertAlignment);
    }
}

}  // namespace deep_ep::elastic
