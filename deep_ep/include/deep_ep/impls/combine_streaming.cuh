#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

// Functional single-CTA return path for one ready (source, destination)
// route batch. It has no all-rank barrier and publishes only the source's
// destination-indexed return doorbell. The performance path will shard this
// work into expert tiles and use a device queue.
template <int kNumWarps,
          int kNumRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_streaming_return_impl(
    nv_bfloat16* lane_output,
    int* lane_src_metadata,
    const int lane_base_row,
    const int lane_capacity,
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* buffer, void* workspace,
    const int source_rank_idx,
    const int destination_rank_idx,
    const uint64_t generation) {
    EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
    EP_STATIC_ASSERT(kNumRanks <= kNumTopk, "Streaming return requires rank layout");
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    EP_DEVICE_ASSERT(blockIdx.x == 0);
    EP_DEVICE_ASSERT(0 <= source_rank_idx and source_rank_idx < kNumRanks);
    EP_DEVICE_ASSERT(0 <= destination_rank_idx and destination_rank_idx < kNumRanks);
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    const auto recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);
    const auto* lane_control =
        workspace_layout.get_streaming_lane_control_ptr(source_rank_idx);
    const int num_source_tokens = lane_control->num_unique_tokens;
    const int num_source_routes = lane_control->num_routes;
    EP_DEVICE_ASSERT(0 <= num_source_tokens and
                     num_source_tokens <= kNumMaxTokensPerRank);
    EP_DEVICE_ASSERT(num_source_routes >= 0);

    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<1, kNumQPs, kNumWarps>(0, warp_idx);
    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    constexpr int kMetadataStride = 2 + kNumTopk;
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);
    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
    for (int token_idx = warp_idx;
         token_idx < num_source_tokens;
         token_idx += kNumWarps) {
        const auto metadata = lane_src_metadata + token_idx * kMetadataStride;
        const int source_token_idx = __ldg(metadata) % kNumMaxTokensPerRank;
        int stored_row = lane_idx < kNumTopk ? __ldg(metadata + 2 + lane_idx) : -1;
        const bool valid =
            lane_base_row <= stored_row and stored_row < lane_base_row + lane_capacity;
        const auto valid_mask = ptx::gather(valid);
        if (not valid_mask)
            continue;

        int route_rows[kNumTopk];
        compute_topk_slots(
            route_rows, valid_mask,
            [=](const int& idx) {
                return ptx::exchange(stored_row, idx) - lane_base_row;
            });
        combine_reduce<kHiddenVec, kUnrollFactor, kNumTopk>(
            lane_idx, route_rows,
            static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
            [=](const int& row) {
                return math::advance_ptr<combine_vec_t>(
                    lane_output,
                    row * static_cast<int64_t>(kNumHiddenBytes));
            },
            [=]() {
                ptx::tma_store_wait();
                __syncwarp();
            });
        ptx::tma_store_fence();
        __syncwarp();

        auto remote_token = recv_buffer.get_rank_buffer(destination_rank_idx)
            .get_token_buffer(source_token_idx);
        remote_token.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(
            remote_token.get_base_ptr(), source_rank_idx));
        EP_DEVICE_ASSERT(remote_token.get_base_ptr() != nullptr);
        if (ptx::elect_one_sync()) {
            ptx::tma_store_1d(
                remote_token.get_base_ptr(), tma_buffer.get_base_ptr(),
                kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();
    }

    ptx::tma_store_wait();
    __syncthreads();
    if (thread_idx == 0) {
        ptx::fence_acq_rel_sys();
        auto* local_control = workspace_layout.get_streaming_return_control_ptr(
            destination_rank_idx);
        auto* remote_control = gin.get_sym_ptr<ncclTeamTagLsa>(
            local_control, source_rank_idx);
        EP_DEVICE_ASSERT(remote_control != nullptr);
        ptx::st_relaxed_sys(&remote_control->generation, generation);
        streaming::publish_return_ready(remote_control, generation, num_source_routes);
    }
}

// Source-local reduction waits for this source's destination returns only.
// One CTA is intentional in the functional milestone: source_done_seq can be
// released without a grid-wide dependency. Expert-tile scheduling replaces
// this kernel before performance evaluation.
template <int kNumWarps,
          int kNumRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_streaming_reduce_impl(
    nv_bfloat16* combined_x,
    topk_idx_t* combined_topk_idx,
    void* buffer, void* workspace,
    const int num_combined_tokens,
    const uint64_t generation) {
    EP_STATIC_ASSERT(kNumRanks <= kNumTopk, "Streaming reduce requires rank layout");
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    EP_DEVICE_ASSERT(blockIdx.x == 0);

    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);
    if (thread_idx == 0) {
        for (int destination = 0; destination < kNumRanks; ++ destination) {
            const auto* control =
                workspace_layout.get_streaming_return_control_ptr(destination);
            while (not streaming::acquire_return_ready(control, generation)) {}
        }
    }
    __syncthreads();

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto comm_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto comm_buffer = layout::BufferLayout<false>(
        comm_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
    const auto output_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    const auto output_buffer = layout::BufferLayout<false>(
        output_layout, 1, num_combined_tokens, combined_x);
    const auto tma_buffer = layout::BufferLayout<false>(
        output_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);
    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    for (int token_idx = warp_idx;
         token_idx < num_combined_tokens;
         token_idx += kNumWarps) {
        int destination = -1;
        if (lane_idx < kNumTopk) {
            const int expert = static_cast<int>(
                combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            destination = expert >= 0 ? expert / kNumExpertsPerRank : -1;
        }
        __syncwarp();
        const auto valid_mask = ptx::gather(
            ptx::deduplicate(destination, lane_idx) and destination >= 0);
        int destination_slots[kNumRanks];
        compute_topk_slots(
            destination_slots, valid_mask,
            [=](const int& idx) {
                return ptx::exchange(destination, idx);
            });
        combine_reduce<kHiddenVec, kUnrollFactor, kNumRanks>(
            lane_idx, destination_slots,
            static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
            [=](const int& destination_rank) {
                return static_cast<combine_vec_t*>(
                    comm_buffer.get_rank_buffer(destination_rank)
                        .get_token_buffer(token_idx).get_base_ptr());
            },
            [=]() {
                ptx::tma_store_wait();
                __syncwarp();
            });
        ptx::tma_store_fence();
        __syncwarp();
        if (ptx::elect_one_sync()) {
            ptx::tma_store_1d(
                output_buffer.get_token_buffer(token_idx).get_base_ptr(),
                tma_buffer.get_base_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();
    }

    ptx::tma_store_wait();
    __syncthreads();
    if (thread_idx == 0) {
        for (int destination = 0; destination < kNumRanks; ++ destination) {
            auto* control =
                workspace_layout.get_streaming_return_control_ptr(destination);
            streaming::publish_combine_done(control, generation);
            streaming::publish_return_ack(control, generation);
        }
        auto* layer_control = workspace_layout.get_streaming_layer_control_ptr();
        streaming::publish_source_done(layer_control, generation);
        ptx::st_release_sys(&layer_control->drain_done_seq, generation);
    }
}

}  // namespace deep_ep::elastic
