#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

template <typename vec_t>
__device__ __forceinline__
vec_t scale_bf16_vector(vec_t value, const float weight) {
    constexpr int kNumPairs = sizeof(vec_t) / sizeof(nv_bfloat162);
    auto* pairs = reinterpret_cast<nv_bfloat162*>(&value);
    #pragma unroll
    for (int i = 0; i < kNumPairs; ++ i) {
        auto values = __bfloat1622float2(pairs[i]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        values = __fmul2_rn(values, {weight, weight});
#else
        values.x *= weight;
        values.y *= weight;
#endif
        pairs[i] = __float22bfloat162_rn(values);
    }
    return value;
}

// Match the existing scatter -> return numerical boundary while loading
// directly from rank-merged W2 rows: each route is weighted and rounded to
// BF16 before routes for one source token are accumulated.
template <int kHiddenVec, int kUnrollFactor, int kNumExpectedTopk,
          int kNumValidTopk, typename vec_t,
          typename get_src_buffer_ptr_func_t, typename wait_buffer_func_t>
__device__ __forceinline__
void combine_reduce_weighted(
    const int lane_idx,
    int (&topk_slot_idx)[kNumValidTopk],
    float (&topk_weights)[kNumValidTopk],
    vec_t* dst_buffer_ptr,
    const get_src_buffer_ptr_func_t& get_src_buffer_ptr_func,
    const wait_buffer_func_t& wait_buffer_func) {
    constexpr int kNumElemsPerVec = sizeof(vec_t) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kNumElemsPerVec % 2 == 0, "Invalid number of elements");
    EP_STATIC_ASSERT(kHiddenVec % (kUnrollFactor * 32) == 0, "Invalid unrolling");
    EP_STATIC_ASSERT(kNumValidTopk > 0, "Invalid top-k");
    const bool enable_hadd_bypass =
        kNumValidTopk <= 2 or topk_slot_idx[2] < 0;

    if (enable_hadd_bypass) {
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            const auto slot_0 = topk_slot_idx[0];
            const auto src_base_ptr_0 = get_src_buffer_ptr_func(slot_0);
            vec_t values_0[kUnrollFactor] = {};
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                values_0[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_0 +
                        (i * (kUnrollFactor * 32) + j * 32 + lane_idx),
                    slot_0);
                values_0[j] = scale_bf16_vector(values_0[j], topk_weights[0]);
            }

            const auto slot_1 = kNumValidTopk == 1 ? -1 : topk_slot_idx[1];
            const auto src_base_ptr_1 = get_src_buffer_ptr_func(slot_1);
            vec_t values_1[kUnrollFactor] = {};
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                values_1[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_1 +
                        (i * (kUnrollFactor * 32) + j * 32 + lane_idx),
                    slot_1);
                values_1[j] = scale_bf16_vector(values_1[j], topk_weights[1]);
            }

            if (i == 0)
                wait_buffer_func();
            auto* bf162_view_0 = reinterpret_cast<nv_bfloat162*>(values_0);
            const auto* bf162_view_1 = reinterpret_cast<nv_bfloat162*>(values_1);
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    bf162_view_0[j * (kNumElemsPerVec / 2) + l] +=
                        bf162_view_1[j * (kNumElemsPerVec / 2) + l];
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] =
                    values_0[j];
            }
        }
    } else {
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            float2 reduced[kUnrollFactor * kNumElemsPerVec / 2] = {};
            #pragma unroll
            for (int k = 0; k < kNumValidTopk; ++ k) {
                if (k >= kNumExpectedTopk and topk_slot_idx[k] < 0)
                    break;
                const auto src_base_ptr =
                    get_src_buffer_ptr_func(topk_slot_idx[k]);
                vec_t values[kUnrollFactor] = {};
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j) {
                    values[j] = ptx::ldg_with_gez_pred(
                        src_base_ptr +
                            (i * (kUnrollFactor * 32) + j * 32 + lane_idx),
                        topk_slot_idx[k]);
                    values[j] = scale_bf16_vector(values[j], topk_weights[k]);
                }
                const auto* bf162_view =
                    reinterpret_cast<const nv_bfloat162*>(values);
                #pragma unroll
                for (int j = 0;
                     j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    ptx::accumulate(reduced[j], bf162_view[j]);
            }
            if (i == 0)
                wait_buffer_func();
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                vec_t casted_value;
                auto* bf162_view =
                    reinterpret_cast<nv_bfloat162*>(&casted_value);
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    bf162_view[l] = __float22bfloat162_rn(
                        reduced[j * (kNumElemsPerVec / 2) + l]);
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] =
                    casted_value;
            }
        }
    }
}

// Return path for one ready (source, destination) route batch. Token rows are
// sharded across a small grid, without an all-rank barrier. The last CTA
// publishes only the source's destination-indexed return doorbell.
template <bool kMergedOutput,
          bool kApplyRouteWeights,
          int kNumBlocks,
          int kNumWarps,
          int kNumRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_streaming_return_impl(
    nv_bfloat16* lane_output,
    nv_bfloat16* merged_output,
    int* lane_to_merged,
    float* lane_route_weights,
    int* lane_src_metadata,
    const int* lane_counts,
    const int lane_base_row,
    const int lane_capacity,
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* buffer, void* workspace,
    const int source_rank_idx,
    const int destination_rank_idx,
    const uint64_t generation) {
    EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    const bool batched_sources = source_rank_idx < 0;
    const int source_rank = batched_sources
        ? static_cast<int>(blockIdx.x) / kNumBlocks
        : source_rank_idx;
    const int return_block_idx = batched_sources
        ? static_cast<int>(blockIdx.x) % kNumBlocks
        : static_cast<int>(blockIdx.x);
    if constexpr (kMergedOutput)
        EP_DEVICE_ASSERT(batched_sources);
    EP_DEVICE_ASSERT(blockIdx.x <
                     (batched_sources ? kNumRanks * kNumBlocks : kNumBlocks));
    EP_DEVICE_ASSERT(0 <= source_rank and source_rank < kNumRanks);
    EP_DEVICE_ASSERT(0 <= destination_rank_idx and destination_rank_idx < kNumRanks);
    constexpr int kMetadataStride = 2 + kNumTopk;
    constexpr int kLaneControlStride = 2 + kNumExperts / kNumRanks;
    if (batched_sources) {
        if constexpr (not kMergedOutput)
            lane_output += source_rank * lane_capacity * kHidden;
        lane_src_metadata +=
            source_rank * kNumMaxTokensPerRank * kMetadataStride;
        lane_counts += source_rank * kLaneControlStride;
    }
    const int effective_lane_base_row =
        batched_sources ? source_rank * lane_capacity : lane_base_row;
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    const auto recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);
    // Counts are a generation-owned copy sidecar. The source may already be
    // writing its next single-slot ingress and LaneControl at this point.
    const int num_source_tokens = lane_counts[0];
    const int num_source_routes = lane_counts[1];
    EP_DEVICE_ASSERT(0 <= num_source_tokens and
                     num_source_tokens <= kNumMaxTokensPerRank);
    EP_DEVICE_ASSERT(num_source_routes >= 0);

    // The return buffer is single-slot per (source, destination). Block zero
    // waits for this source's previous reduction acknowledgement, initializes
    // destination-local grid completion, then releases the remaining CTAs.
    auto* launch_control =
        workspace_layout.get_streaming_return_launch_control_ptr(source_rank);
    if (thread_idx == 0) {
        if (return_block_idx == 0) {
            if (generation > 1) {
                const auto* ack_control =
                    workspace_layout.get_streaming_return_control_ptr(source_rank);
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    if (streaming::acquire_return_ack(ack_control, generation - 1))
                        return true;
                    if (is_last_check) {
                        printf("DeepEP streaming return reuse timeout, dst: %d, src: %d, expected generation: %llu, observed: %llu\n",
                               destination_rank_idx, source_rank,
                               static_cast<unsigned long long>(generation - 1),
                               static_cast<unsigned long long>(
                                   ptx::ld_acquire_sys(&ack_control->ack_seq)));
                    }
                    return false;
                });
            }
            ptx::st_relaxed_sys(&launch_control->completed_blocks, 0u);
            ptx::st_release_sys(&launch_control->generation, generation);
        } else {
            while (ptx::ld_acquire_sys(&launch_control->generation) != generation) {}
        }
    }
    __syncthreads();

    // Give each concurrent source/block CTA a distinct virtual SM/QP whenever
    // enough QPs exist. get_qp_mode safely falls back to GPU sharing otherwise.
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumRanks * kNumBlocks, kNumQPs, kNumWarps>(
            source_rank * kNumBlocks + return_block_idx, warp_idx);
    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // A zero-route lane must still publish its generation so the source-side
    // reduce can make progress.  Avoid scanning every source token and
    // setting up the TMA/NVLink data path when there is nothing to return.
    if (num_source_routes == 0) {
        if (thread_idx == 0 and return_block_idx == 0) {
            ptx::fence_acq_rel_sys();
            auto* local_control =
                workspace_layout.get_streaming_return_control_ptr(
                    destination_rank_idx);
            auto* remote_control = gin.get_sym_ptr<ncclTeamTagLsa>(
                local_control, source_rank);
            EP_DEVICE_ASSERT(remote_control != nullptr);
            ptx::st_relaxed_sys(&remote_control->generation, generation);
            streaming::publish_return_ready(remote_control, generation, 0);
        }
        return;
    }

    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);
    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
    for (int token_idx = return_block_idx * kNumWarps + warp_idx;
         token_idx < num_source_tokens;
         token_idx += kNumBlocks * kNumWarps) {
        const auto metadata = lane_src_metadata + token_idx * kMetadataStride;
        const int source_token_idx = __ldg(metadata) % kNumMaxTokensPerRank;
        int stored_row = lane_idx < kNumTopk ? __ldg(metadata + 2 + lane_idx) : -1;
        const bool valid =
            effective_lane_base_row <= stored_row and
            stored_row < effective_lane_base_row + lane_capacity;
        const auto valid_mask = ptx::gather(valid);
        if (not valid_mask)
            continue;

        int route_rows[kNumTopk];
        if constexpr (kMergedOutput) {
            auto remaining = valid_mask;
            #pragma unroll
            for (int k = 0; k < kNumTopk; ++ k) {
                const int lowest_idx = __ffs(remaining) - 1;
                const int lane_row =
                    ptx::exchange(stored_row, lowest_idx) -
                    effective_lane_base_row;
                remaining &= remaining - 1;
                route_rows[k] = lowest_idx >= 0
                    ? __ldg(lane_to_merged +
                            source_rank * lane_capacity + lane_row)
                    : -1;
            }
            if constexpr (kApplyRouteWeights) {
                float route_weights[kNumTopk];
                auto weight_mask = valid_mask;
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k) {
                    const int lowest_idx = __ffs(weight_mask) - 1;
                    const int lane_row =
                        ptx::exchange(stored_row, lowest_idx) -
                        effective_lane_base_row;
                    weight_mask &= weight_mask - 1;
                    route_weights[k] = lowest_idx >= 0
                        ? __ldg(lane_route_weights +
                                source_rank * lane_capacity + lane_row)
                        : 0.0f;
                }
                combine_reduce_weighted<
                    kHiddenVec, kUnrollFactor, kNumTopk, kNumTopk>(
                    lane_idx, route_rows, route_weights,
                    static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    [=](const int& row) {
                        return math::advance_ptr<combine_vec_t>(
                            merged_output,
                            row * static_cast<int64_t>(kNumHiddenBytes));
                    },
                    [=]() {
                        ptx::tma_store_wait();
                        __syncwarp();
                    });
            } else {
                combine_reduce<kHiddenVec, kUnrollFactor, kNumTopk>(
                    lane_idx, route_rows,
                    static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    [=](const int& row) {
                        return math::advance_ptr<combine_vec_t>(
                            merged_output,
                            row * static_cast<int64_t>(kNumHiddenBytes));
                    },
                    [=]() { ptx::tma_store_wait(); __syncwarp(); });
            }
        } else {
            compute_topk_slots(
                route_rows, valid_mask,
                [=](const int& idx) {
                    return ptx::exchange(stored_row, idx) -
                        effective_lane_base_row;
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
        }
        ptx::tma_store_fence();
        __syncwarp();

        auto remote_token = recv_buffer.get_rank_buffer(destination_rank_idx)
            .get_token_buffer(source_token_idx);
        remote_token.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(
            remote_token.get_base_ptr(), source_rank));
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

    __shared__ bool is_last_block;
    if (thread_idx == 0) {
        ptx::fence_acq_rel_sys();
        is_last_block =
            atomicAdd(&launch_control->completed_blocks, 1u) + 1u == kNumBlocks;
    }
    __syncthreads();
    if (not is_last_block)
        return;

    if (thread_idx == 0) {
        ptx::fence_acq_rel_sys();
        auto* local_control = workspace_layout.get_streaming_return_control_ptr(
            destination_rank_idx);
        auto* remote_control = gin.get_sym_ptr<ncclTeamTagLsa>(
            local_control, source_rank);
        EP_DEVICE_ASSERT(remote_control != nullptr);
        ptx::st_relaxed_sys(&remote_control->generation, generation);
        streaming::publish_return_ready(remote_control, generation, num_source_routes);
    }
}

// Hybrid combine readiness gate. Keep asynchronous per-destination returns,
// but do not start the reduction grid until every owner has published its
// return slot. A single CTA performs the remote polling and initializes the
// generation-local completion state; the following reduce-only launch is
// ordered after this kernel on the same CUDA stream.
template <int kNumRanks, int kNumExperts>
__global__ void __launch_bounds__(32, 1)
combine_streaming_wait_ready_impl(
    void* workspace,
    const uint64_t generation) {
    if (threadIdx.x != 0)
        return;

    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);
    for (int destination = 0; destination < kNumRanks; ++ destination) {
        const auto* control =
            workspace_layout.get_streaming_return_control_ptr(destination);
        while (not streaming::acquire_return_ready(control, generation)) {}
    }
    auto* layer_control = workspace_layout.get_streaming_layer_control_ptr();
    ptx::st_relaxed_sys(&layer_control->reduce_completed_blocks, 0u);
    ptx::st_release_sys(&layer_control->reduce_generation, generation);
}

// Source-local reduction consumes this source's destination return slots.
// The legacy variant waits and initializes completion state in block zero;
// the hybrid variant is launched after combine_streaming_wait_ready_impl.
// Token rows are sharded over a small grid. A generation-tagged completion
// counter lets the last CTA publish source completion without a cooperative
// launch or a second kernel.
template <bool kWaitForReturns,
          int kNumBlocks,
          int kNumWarps,
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
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* buffer, void* workspace,
    const int source_rank_idx,
    const int num_combined_tokens,
    const uint64_t generation) {
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    EP_DEVICE_ASSERT(blockIdx.x < kNumBlocks);
    EP_DEVICE_ASSERT(0 <= source_rank_idx and source_rank_idx < kNumRanks);

    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);
    auto* layer_control = workspace_layout.get_streaming_layer_control_ptr();
    if constexpr (kWaitForReturns) {
        if (thread_idx == 0) {
            if (blockIdx.x == 0) {
                for (int destination = 0; destination < kNumRanks; ++ destination) {
                    const auto* control =
                        workspace_layout.get_streaming_return_control_ptr(destination);
                    while (not streaming::acquire_return_ready(control, generation)) {}
                }
                ptx::st_relaxed_sys(&layer_control->reduce_completed_blocks, 0u);
                ptx::st_release_sys(&layer_control->reduce_generation, generation);
            } else {
                while (ptx::ld_acquire_sys(&layer_control->reduce_generation) !=
                       generation) {}
            }
        }
        __syncthreads();
    }

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
    for (int token_idx = static_cast<int>(blockIdx.x) * kNumWarps + warp_idx;
         token_idx < num_combined_tokens;
         token_idx += kNumBlocks * kNumWarps) {
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

    __shared__ bool is_last_block;
    if (thread_idx == 0) {
        ptx::fence_acq_rel_sys();
        is_last_block =
            atomicAdd(&layer_control->reduce_completed_blocks, 1u) + 1u ==
            kNumBlocks;
    }
    __syncthreads();
    if (not is_last_block)
        return;

    if (thread_idx == 0) {
        for (int destination = 0; destination < kNumRanks; ++ destination) {
            auto* control =
                workspace_layout.get_streaming_return_control_ptr(destination);
            streaming::publish_combine_done(control, generation);
        }
        streaming::publish_source_done(layer_control, generation);
        ptx::st_release_sys(&layer_control->drain_done_seq, generation);
    }
    __syncthreads();

    // Let each destination independently reuse the return slot that it owns
    // for this source.  The reduction above has consumed every returned row.
    if (thread_idx < kNumRanks) {
        const auto gin = handle::NCCLGin(
            nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
        auto* local_ack =
            workspace_layout.get_streaming_return_control_ptr(source_rank_idx);
        auto* remote_ack = gin.get_sym_ptr<ncclTeamTagLsa>(
            local_ack, thread_idx);
        EP_DEVICE_ASSERT(remote_ack != nullptr);
        ptx::fence_acq_rel_sys();
        ptx::st_release_sys(&remote_ack->ack_seq, generation);
    }
}

}  // namespace deep_ep::elastic
