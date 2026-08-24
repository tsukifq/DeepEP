#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

// A destination publishes one acknowledgement to every source after all of
// its lane consumers have drained.  The acknowledgement is written into the
// source rank's symmetric workspace at the destination-indexed ack slot.
template <int kNumRanks, int kNumExperts, int kNumThreads = 32>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_streaming_ack_impl(const ncclDevComm_t nccl_dev_comm,
                            const ncclWindow_t nccl_window,
                            void* workspace,
                            const int destination_rank_idx,
                            const uint64_t generation) {
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads,
                     "Streaming ack supports at most one warp of ranks");
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0,
                     "Invalid expert/rank shape");

    const auto source_rank_idx = static_cast<int>(threadIdx.x);
    if (source_rank_idx >= kNumRanks)
        return;

    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);
    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    auto* local_ack_slot =
        workspace_layout.get_streaming_lane_control_ptr(destination_rank_idx);
    auto* remote_ack_slot = gin.get_sym_ptr<ncclTeamTagLsa>(
        local_ack_slot, source_rank_idx);
    EP_DEVICE_ASSERT(remote_ack_slot != nullptr);

    // The kernel is queued after every local lane consumer.  Order all of
    // those reads before allowing the source to reuse its remote ingress slot.
    ptx::fence_acq_rel_sys();
    ptx::st_release_sys(&remote_ack_slot->ack_seq, generation);
}

// A source may start its next generation only after every destination has
// stopped reading the source's previous single-slot ingress payload.  This is
// source-local backpressure, not an all-rank dispatch-entry barrier: another
// source whose destinations have acknowledged it can proceed independently.
template <int kNumRanks, int kNumExperts, int64_t kNumTimeoutCycles,
          int kNumThreads = 32>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_streaming_reuse_wait_impl(void* workspace,
                                   const int source_rank_idx,
                                   const uint64_t previous_generation) {
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads,
                     "Streaming reuse wait supports at most one warp of ranks");
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0,
                     "Invalid expert/rank shape");

    const auto destination_rank_idx = static_cast<int>(threadIdx.x);
    if (destination_rank_idx >= kNumRanks)
        return;

    const auto workspace_layout = layout::WorkspaceLayout(
        workspace, 1, kNumRanks, kNumExperts);
    const auto* ack_slot =
        workspace_layout.get_streaming_lane_control_ptr(destination_rank_idx);
    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
        if (streaming::acquire_ack(ack_slot, previous_generation))
            return true;
        if (is_last_check) {
            printf("DeepEP streaming reuse timeout, src: %d, dst: %d, expected generation: %llu, observed: %llu\n",
                   source_rank_idx, destination_rank_idx,
                   static_cast<unsigned long long>(previous_generation),
                   static_cast<unsigned long long>(
                       ptx::ld_acquire_sys(&ack_slot->ack_seq)));
        }
        return false;
    });
}

}  // namespace deep_ep::elastic
