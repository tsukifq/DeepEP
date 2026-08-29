#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

// One CTA owns one source lane and packs that lane independently of every
// other source. kExpertAlignment=1 preserves the M1B diagnostic layout;
// production passes the SM90 DeepGEMM alignment so a lane can be consumed
// directly through the existing M-grouped psum API.
template <int kNumWarps,
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int64_t kNumTimeoutCycles,
          int kExpertAlignment = 1,
          int kNumThreads = kNumWarps * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_streaming_copy_impl(void* buffer, void* workspace,
                             void* packed_x, sf_pack_t* packed_sf,
                             float* packed_topk_weights,
                             int* packed_src_metadata,
                             int* packed_lane_control,
                             const int packed_sf_token_stride,
                             const int packed_sf_hidden_stride,
                             const int destination_rank_idx,
                             const uint64_t generation) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    constexpr int kRawLaneRouteCapacity =
        kNumMaxTokensPerRank * (kNumTopk < kNumExpertsPerRank ? kNumTopk : kNumExpertsPerRank);
    constexpr int kUnalignedLaneRouteCapacity = kRawLaneRouteCapacity +
        (kNumExpertsPerRank - 1) * (kExpertAlignment - 1);
    constexpr int kLaneRouteCapacity =
        (kUnalignedLaneRouteCapacity + kExpertAlignment - 1) /
        kExpertAlignment * kExpertAlignment;
    constexpr int kMetadataStride = 2 + kNumTopk;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid expert/rank shape");
    EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
    EP_STATIC_ASSERT(kExpertAlignment > 0, "Invalid expert alignment");
    EP_STATIC_ASSERT(kNumExpertsPerRank <= layout::WorkspaceLayout::kNumMaxExpertsPerRank,
                     "Too many local experts");

    const auto source_rank_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    EP_DEVICE_ASSERT(source_rank_idx < kNumRanks);

    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    auto* lane_control = workspace_layout.get_streaming_lane_control_ptr(source_rank_idx);

    __shared__ int lane_num_tokens;
    __shared__ int lane_num_routes;
    __shared__ int lane_expert_cursor[kNumExpertsPerRank];

    // Each CTA waits only for its own source. A slow lower-numbered source
    // therefore cannot block a ready higher-numbered lane.
    if (thread_idx == 0) {
        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
            if (streaming::acquire_payload_ready(lane_control, generation))
                return true;
            if (is_last_check) {
                printf("DeepEP streaming payload timeout, dst: %d, src: %d, expected generation: %llu, observed: %llu\n",
                       destination_rank_idx, source_rank_idx,
                       static_cast<unsigned long long>(generation),
                       static_cast<unsigned long long>(ptx::ld_acquire_sys(&lane_control->payload_done_seq)));
            }
            return false;
        });

        const auto num_tokens = *workspace_layout.get_streaming_lane_token_count_ptr(source_rank_idx);
        EP_DEVICE_ASSERT(0 <= num_tokens and num_tokens <= kNumMaxTokensPerRank);
        lane_num_tokens = static_cast<int>(num_tokens);

        int64_t packed_end = 0;
        int64_t raw_routes = 0;
        for (int expert_idx = 0; expert_idx < kNumExpertsPerRank; ++ expert_idx) {
            const auto count = *workspace_layout.get_streaming_lane_expert_count_ptr(
                source_rank_idx, expert_idx);
            EP_DEVICE_ASSERT(count >= 0);
            const auto packed_start = math::align(
                packed_end, static_cast<int64_t>(kExpertAlignment));
            packed_end = packed_start + count;
            raw_routes += count;
            EP_DEVICE_ASSERT(packed_end <= kLaneRouteCapacity);
            packed_lane_control[
                source_rank_idx * (kNumExpertsPerRank + 2) + 2 + expert_idx] =
                static_cast<int>(packed_end);
            lane_expert_cursor[expert_idx] = static_cast<int>(packed_start);
        }
        EP_DEVICE_ASSERT(raw_routes <= kRawLaneRouteCapacity);
        lane_num_routes = static_cast<int>(raw_routes);
        packed_lane_control[source_rank_idx * (kNumExpertsPerRank + 2) + 0] =
            lane_num_tokens;
        packed_lane_control[source_rank_idx * (kNumExpertsPerRank + 2) + 1] =
            lane_num_routes;
        lane_control->generation = generation;
        // Workspace mirrors are diagnostic only. Downstream consumers own
        // generation-private count and psum sidecars.
        lane_control->num_unique_tokens = lane_num_tokens;
        lane_control->num_routes = lane_num_routes;
        ptx::st_relaxed_sys(
            &lane_control->state, static_cast<uint32_t>(streaming::LaneState::kPacking));
    }
    __syncthreads();

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto token_layout = layout::TokenLayout(
        kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);

    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    const int expert_start_idx = destination_rank_idx * kNumExpertsPerRank;
    const int expert_end_idx = expert_start_idx + kNumExpertsPerRank;
    for (int token_idx = warp_idx; token_idx < lane_num_tokens; token_idx += kNumWarps) {
        const auto buffer_token = scaleup_buffer.get_rank_buffer(source_rank_idx)
            .get_token_buffer(token_idx);

        // The per-warp TMA staging slot can be reused only after its previous
        // output copies have completed.
        ptx::tma_store_wait();
        __syncwarp();
        if (ptx::elect_one_sync()) {
            ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                             mbarrier_ptr, tma_buffer.get_num_bytes<false>());
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, tma_buffer.get_num_bytes<false>());
        }
        __syncwarp();

        int dst_expert_idx = lane_idx < kNumTopk ?
            buffer_token.get_topk_idx_ptr()[lane_idx] : -1;
        __syncwarp();
        const bool in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;
        const auto master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));
        const int local_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;
        EP_DEVICE_ASSERT(ptx::deduplicate(local_expert_idx, lane_idx) or local_expert_idx == -1);

        int dst_tensor_idx = -1;
        if (in_range) {
            const int lane_row = atomicAdd_block(lane_expert_cursor + local_expert_idx, 1);
            EP_DEVICE_ASSERT(lane_row < kLaneRouteCapacity);
            dst_tensor_idx = source_rank_idx * kLaneRouteCapacity + lane_row;
        }
        __syncwarp();

        if (ptx::elect_one_sync())
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        __syncwarp();

        if (dst_tensor_idx >= 0) {
            ptx::tma_store_1d(
                math::advance_ptr(packed_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();

        if constexpr (kNumSFPacks > 0) {
            constexpr int kNumFullIters = kNumSFPacks / 32;
            constexpr bool kHasLastIter = kNumSFPacks % 32 != 0;
            const bool do_last_iter = kHasLastIter and kNumFullIters * 32 + lane_idx < kNumSFPacks;
            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");

            const auto smem_src_ptr = tma_buffer.get_sf_ptr();
            sf_pack_t reg_src[kNumFullIters + 1];
            #pragma unroll
            for (int k = 0; k < kNumFullIters; ++ k)
                reg_src[k] = smem_src_ptr[k * 32 + lane_idx];
            if (do_last_iter)
                reg_src[kNumFullIters] = smem_src_ptr[kNumFullIters * 32 + lane_idx];

            auto mask = ptx::gather(dst_tensor_idx >= 0);
            while (mask) {
                const int valid_lane_idx = __ffs(mask) - 1;
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(
                    packed_sf,
                    ptx::exchange(dst_tensor_idx, valid_lane_idx) *
                        (static_cast<int64_t>(packed_sf_token_stride) * sizeof(sf_pack_t)));
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k)
                    gmem_dst[(k * 32 + lane_idx) * static_cast<int64_t>(packed_sf_hidden_stride)] = reg_src[k];
                if (do_last_iter)
                    gmem_dst[(kNumFullIters * 32 + lane_idx) *
                        static_cast<int64_t>(packed_sf_hidden_stride)] = reg_src[kNumFullIters];
                mask ^= 1 << valid_lane_idx;
            }
        }

        if (packed_topk_weights != nullptr and dst_tensor_idx >= 0)
            packed_topk_weights[dst_tensor_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        __syncwarp();

        const int metadata_idx = source_rank_idx * kNumMaxTokensPerRank + token_idx;
        if (ptx::elect_one_sync()) {
            packed_src_metadata[metadata_idx * kMetadataStride + 0] =
                *tma_buffer.get_src_token_global_idx_ptr();
            packed_src_metadata[metadata_idx * kMetadataStride + 1] =
                source_rank_idx * kNumTopk + master_src_topk_idx;
        }
        if (lane_idx < kNumTopk)
            packed_src_metadata[metadata_idx * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;
        __syncwarp();
    }

    ptx::tma_store_wait();
    __syncwarp();
    __syncthreads();
    if (thread_idx == 0) {
        for (int expert_idx = 0; expert_idx < kNumExpertsPerRank; ++ expert_idx) {
            const auto expected_end = packed_lane_control[
                source_rank_idx * (kNumExpertsPerRank + 2) + 2 + expert_idx];
            if (lane_expert_cursor[expert_idx] != expected_end) {
                printf("DeepEP streaming pack count mismatch, dst: %d, src: %d, generation: %llu, expert: %d, actual end: %d, expected end: %d, tokens: %d, routes: %d\n",
                       destination_rank_idx, source_rank_idx,
                       static_cast<unsigned long long>(generation), expert_idx,
                       lane_expert_cursor[expert_idx],
                       expected_end,
                       lane_num_tokens, lane_num_routes);
            }
            EP_DEVICE_ASSERT(lane_expert_cursor[expert_idx] == expected_end);
        }
        ptx::fence_acq_rel_sys();
        streaming::publish_pack_ready(lane_control, generation);
        // The copy CTA is now done with the remote single-slot ingress. The
        // consumer must observe pack_done on its lane stream before queuing
        // the ACK that waits on this marker.
        streaming::publish_ingress_consumed(lane_control, generation);
    }
}

}  // namespace deep_ep::elastic
