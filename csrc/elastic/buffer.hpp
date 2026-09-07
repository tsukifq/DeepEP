#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>
#include <pybind11/functional.h>

#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/compiled.cuh>

#include "../kernels/backend/api.cuh"
#include "../kernels/elastic/api.hpp"
#include "../utils/event.hpp"
#include "utils.hpp"

namespace deep_ep::elastic {

class ElasticBuffer {
    // Buffer bytes = GPU buffer + CPU buffer (excludes workspace)
    // Memory layout: [[[Workspace] GPU buffer] CPU buffer]
    int64_t num_buffer_bytes;
    int64_t num_gpu_buffer_bytes;
    int64_t num_cpu_buffer_bytes;
    void* buffer;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // Workspace
    // NOTES: for all workspace, we must keep them as zeros
    void *workspace;
    void *host_workspace, *mapped_host_workspace;
    std::shared_ptr<layout::WorkspaceLayout> workspace_layout_wo_expert;

    // CUDA streams
    at::cuda::CUDAStream comm_stream;
    // Preserve the original single-stream rank-ready/merged-epilogue path.
    at::cuda::CUDAStream streaming_copy_stream;
    // Packed streaming ingress owns one fixed FIFO stream per source rank.
    std::vector<at::cuda::CUDAStream> streaming_lane_copy_streams;
    mutable std::optional<torch::Tensor> latest_streaming_packed_x;
    mutable std::optional<torch::Tensor> latest_streaming_packed_sf;
    mutable std::optional<torch::Tensor> latest_streaming_packed_topk_weights;
    mutable std::optional<torch::Tensor> latest_streaming_packed_src_metadata;
    mutable std::optional<torch::Tensor> latest_streaming_packed_lane_control;
    mutable std::optional<EventHandle> streaming_consumer_event;
    mutable std::optional<EventHandle> streaming_lane_drain_event;
    mutable uint64_t streaming_generation = 0;
    mutable uint64_t latest_streaming_generation = 0;
    mutable int latest_streaming_num_ranks = 0;
    mutable int latest_streaming_num_local_experts = 0;
    mutable int latest_streaming_lane_route_capacity = 0;
    mutable void* latest_streaming_return_buffer = nullptr;
    mutable std::vector<bool> streaming_lane_return_submitted;
    mutable std::vector<bool> streaming_lane_ingress_released;
    mutable std::vector<std::optional<EventHandle>> streaming_lane_release_events;
    mutable std::vector<std::optional<EventHandle>> streaming_lane_return_events;
    mutable bool streaming_copy_used = false;
    mutable bool streaming_lane_copy_used = false;
    mutable bool streaming_lane_view_outstanding = false;
    mutable bool streaming_generation_released_per_lane = false;

    // Whether to use hybrid mode (scale-out with scale-up)
    bool allow_hybrid_mode;

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;

    // Whether to prefer overlapping communication with compute (use more SMs and channels if false)
    bool prefer_overlap_with_compute;

    // Timeout settings
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;

    // NCCL context
    std::shared_ptr<nccl::NCCLSymmetricMemoryContext> nccl_context;

    // Some EP hybrid mode settings
    static constexpr int kNumMaxChannelsPerSM = 8;
    static constexpr int kNumMaxSMs = 160;
    static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;

    // Some Engram storage settings
    int num_engram_entries = 0, engram_hidden = 0;
    std::optional<torch::Tensor> engram_sf;

    // PP settings
    int prev_rank_idx = 0, next_rank_idx = 0;
    int64_t num_max_pp_tensor_bytes = 0;
    int num_max_pp_inflight_tensors = 0;

    // AGRS session settings
    int64_t num_max_agrs_session_bytes = 0;
    int num_max_agrs_per_session = 0;
    int agrs_session_idx = 0;
    bool agrs_in_session = false;

    // AGRS in-session settings
    int64_t agrs_buffer_offset = 0;
    int agrs_buffer_slot_idx = 0;

public:
    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& nccl_comm, const symmetric::cpu_comm_t& cpu_comm,
                  const int64_t& num_buffer_bytes, const int64_t& num_cpu_buffer_bytes,
                  const bool& allow_hybrid_mode,
                  const bool& allow_multiple_reduction,
                  const bool& prefer_overlap_with_compute,
                  const int& sl_idx, const int& num_allocated_qps,
                  const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
                  const bool& explicitly_destroy):
        num_buffer_bytes(num_buffer_bytes),
        num_cpu_buffer_bytes(num_cpu_buffer_bytes),
        explicitly_destroy(explicitly_destroy),
        comm_stream(get_global_comm_stream()),
        streaming_copy_stream(at::cuda::getStreamFromPool(false)),
        allow_hybrid_mode(allow_hybrid_mode),
        allow_multiple_reduction(allow_multiple_reduction),
        prefer_overlap_with_compute(prefer_overlap_with_compute) {
        // Check buffer bytes alignment (2 MB)
        EP_HOST_ASSERT(num_buffer_bytes > 0 and num_buffer_bytes % symmetric::kNumAlignmentBytes == 0);
        EP_HOST_ASSERT(num_cpu_buffer_bytes >= 0 and num_cpu_buffer_bytes % symmetric::kNumAlignmentBytes == 0);
        EP_HOST_ASSERT(num_cpu_buffer_bytes <= num_buffer_bytes);
        num_gpu_buffer_bytes = num_buffer_bytes - num_cpu_buffer_bytes;

        // Workspace is aligned to 2 MB so that it sits cleanly at the front of the GPU segment
        const auto num_workspace_bytes = math::align<int64_t>(
            layout::WorkspaceLayout::get_num_bytes(), symmetric::kNumAlignmentBytes);

        // Create NCCL symmetric memory context
        // Symmetric memory layout: [[[Workspace] GPU buffer] CPU buffer]
        // sym.num_bytes = workspace + buffer, sym.num_cpu_bytes = CPU buffer
        const auto num_sym_bytes = num_workspace_bytes + num_buffer_bytes;
        this->nccl_context = std::make_shared<nccl::NCCLSymmetricMemoryContext>(
            nccl_comm, cpu_comm, num_ranks, rank_idx,
            num_sym_bytes, num_cpu_buffer_bytes,
            allow_hybrid_mode, sl_idx, num_allocated_qps);

        // Source lanes own disjoint payload and doorbell slices, so keep their
        // ingress producers concurrent. A diagnostic fallback can serialize them
        // without changing the protocol if a platform-specific driver issue recurs.
        streaming_lane_copy_streams.reserve(nccl_context->num_ranks);
        const auto serialize_lane_copies =
            get_env<int>("EP_STREAMING_SERIALIZE_LANE_COPIES", 0) != 0;
        const auto shared_lane_copy_stream = at::cuda::getStreamFromPool(false);
        for (int source_rank_idx = 0;
             source_rank_idx < nccl_context->num_ranks;
             ++ source_rank_idx) {
            streaming_lane_copy_streams.emplace_back(
                serialize_lane_copies ? shared_lane_copy_stream
                                      : at::cuda::getStreamFromPool(false));
        }

        // Verify the symmetric memory layout matches our expectations
        EP_HOST_ASSERT(num_workspace_bytes + num_gpu_buffer_bytes == nccl_context->num_gpu_bytes);
        EP_HOST_ASSERT(num_cpu_buffer_bytes == nccl_context->num_cpu_bytes);

        // Timeout
        this->num_cpu_timeout_secs = num_cpu_timeout_secs;
        this->num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs);
        this->num_gpu_timeout_cycles *= jit::device_runtime->get_clock_rate();

        // Assign workspaces and buffers
        workspace = this->nccl_context->mapped_window_ptr;
        workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(
            workspace, nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks, 0);
        buffer = static_cast<uint8_t*>(workspace) + num_workspace_bytes;
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, num_workspace_bytes));

        // Allocate host workspaces
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
        std::memset(host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());

        // We should call a barrier at the end
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
    }

    ~ElasticBuffer() noexcept(false) {
        if (not explicitly_destroy)
            destroy();

        if (not destroyed) {
            printf("`destroy()` is not called before DeepEP elastic buffer destruction, which can leak resources.\n");
            fflush(stdout);
        }
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(not streaming_lane_view_outstanding and
                       "Call release_streaming_lane_view() before destroy()");

        // The experimental copy consumers run on fixed per-source streams
        // and may outlive the legacy dispatch return event. Drain every one
        // before freeing symmetric workspace or ingress buffers.
        if (streaming_lane_drain_event.has_value())
            stream_wait(comm_stream, streaming_lane_drain_event.value());
        if (streaming_copy_used)
            stream_wait(comm_stream, streaming_copy_stream);
        if (streaming_lane_copy_used) {
            for (const auto& source_stream : streaming_lane_copy_streams)
                stream_wait(comm_stream, source_stream);
        }

        // Finish all works on all GPUs
        barrier(true, true);

        // Deallocate host workspaces
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));

        // Destroy NCCL context
        nccl_context->finalize();

        // Cannot use anymore
        destroyed = true;
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;
    }

    uint64_t get_streaming_generation() const {
        return streaming_generation;
    }

    int get_streaming_lane_protocol_version() const {
        // Version 2 makes lane control generation-owned and permits ingress
        // release after the matching pack doorbell has been observed.
        return 2;
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, torch::Tensor,
               torch::Tensor, torch::Tensor, uint64_t>
    get_streaming_lane_view() const {
        EP_HOST_ASSERT(streaming_lane_view_outstanding and
                       latest_streaming_packed_x.has_value() and
                       latest_streaming_packed_src_metadata.has_value() and
                       latest_streaming_packed_lane_control.has_value());
        const auto num_ranks = latest_streaming_num_ranks;
        const auto num_local_experts = latest_streaming_num_local_experts;
        const auto lane_capacity = latest_streaming_lane_route_capacity;
        const auto workspace_layout = layout::WorkspaceLayout(
            workspace, 1, num_ranks, num_ranks * num_local_experts);
        auto pack_done_seq = torch::from_blob(
            &workspace_layout.get_streaming_lane_control_ptr(0)->pack_done_seq,
            {num_ranks},
            {static_cast<int64_t>(sizeof(streaming::LaneControl) / sizeof(uint64_t))},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        auto lane_sf = std::optional<torch::Tensor>();
        if (latest_streaming_packed_sf.has_value()) {
            const auto& packed_sf = latest_streaming_packed_sf.value();
            EP_HOST_ASSERT(packed_sf.dim() == 3 and
                           packed_sf.size(0) == num_ranks and
                           packed_sf.size(1) == lane_capacity);
            lane_sf = packed_sf;
        }
        return {
            latest_streaming_packed_x->view(
                {num_ranks, lane_capacity, latest_streaming_packed_x->size(1)}),
            lane_sf,
            latest_streaming_packed_topk_weights.has_value() ?
                std::make_optional(latest_streaming_packed_topk_weights->view(
                    {num_ranks, lane_capacity})) : std::nullopt,
            latest_streaming_packed_src_metadata.value(),
            latest_streaming_packed_lane_control->slice(
                1, 2, 2 + num_local_experts),
            pack_done_seq,
            latest_streaming_generation};
    }

    void release_streaming_lane_view() const {
        EP_HOST_ASSERT(streaming_lane_view_outstanding);
        EP_HOST_ASSERT(streaming_lane_ingress_released.size() ==
                       static_cast<size_t>(latest_streaming_num_ranks));
        EP_HOST_ASSERT(streaming_lane_release_events.size() ==
                       static_cast<size_t>(latest_streaming_num_ranks));
        EP_HOST_ASSERT(streaming_lane_return_events.size() ==
                       static_cast<size_t>(latest_streaming_num_ranks));
        uint32_t unreleased_source_mask = 0;
        for (int source_rank_idx = 0;
             source_rank_idx < latest_streaming_num_ranks;
             ++ source_rank_idx) {
            if (not streaming_lane_ingress_released[source_rank_idx])
                unreleased_source_mask |= uint32_t{1} << source_rank_idx;
        }
        const auto stream = at::cuda::getCurrentCUDAStream();
        if (unreleased_source_mask != 0) {
            // Backward-compatible whole-view release: the caller's stream is
            // responsible for ordering all remaining consumers before this
            // masked ACK.  Lanes released explicitly are not ACKed twice.
            launch_dispatch_streaming_ack(
                nccl_context->dev_comm, nccl_context->window, workspace,
                nccl_context->scaleup_rank_idx, latest_streaming_generation,
                latest_streaming_num_ranks,
                latest_streaming_num_ranks * latest_streaming_num_local_experts,
                stream, unreleased_source_mask);
            streaming_consumer_event = EventHandle(stream);
            streaming_generation_released_per_lane = false;
        } else {
            // Every source owns an independent device-side reuse dependency;
            // require every generation-owned return to be submitted before
            // making the view reusable, then avoid recreating a
            // destination-wide join on the comm stream.
            EP_HOST_ASSERT(not streaming_consumer_event.has_value());
            for (int source_rank_idx = 0;
                 source_rank_idx < latest_streaming_num_ranks;
                 ++ source_rank_idx) {
                EP_HOST_ASSERT(
                    streaming_lane_return_submitted[source_rank_idx] and
                    streaming_lane_return_events[source_rank_idx].has_value() and
                    "All per-lane returns must be submitted before strict view finalization");
            }
            streaming_generation_released_per_lane = true;
        }
        // Aggregate the per-lane ACK completions only onto this lifetime
        // stream.  The next dispatch does not consume this dependency.
        for (const auto& release_event : streaming_lane_release_events) {
            if (release_event.has_value())
                stream_wait(stream, release_event.value());
        }
        for (const auto& return_event : streaming_lane_return_events) {
            if (return_event.has_value())
                stream_wait(stream, return_event.value());
        }
        // This event is a lifetime fence, not a next-generation dependency.
        // Strict per-lane dispatch intentionally ignores it, while destroy()
        // must retain a path to every outstanding ACK/return kernel.  Chain
        // successive drain streams before replacing the retained event.
        if (streaming_lane_drain_event.has_value())
            stream_wait(stream, streaming_lane_drain_event.value());
        streaming_lane_drain_event = EventHandle(stream);
        streaming_lane_view_outstanding = false;
    }

    void release_streaming_lane(
        const int& source_rank_idx,
        const uint64_t& generation) const {
        EP_HOST_ASSERT(streaming_lane_view_outstanding and
                       "No streaming lane view is outstanding");
        EP_HOST_ASSERT(generation == latest_streaming_generation and
                       "Streaming lane release generation mismatch");
        EP_HOST_ASSERT(0 <= source_rank_idx and
                       source_rank_idx < latest_streaming_num_ranks);
        EP_HOST_ASSERT(streaming_lane_return_submitted.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks) and
                       streaming_lane_ingress_released.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks) and
                       streaming_lane_release_events.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks));
        EP_HOST_ASSERT(not streaming_lane_ingress_released[source_rank_idx] and
                       "Streaming lane was already released");

        const auto stream = at::cuda::getCurrentCUDAStream();
        launch_dispatch_streaming_lane_ack(
            nccl_context->dev_comm, nccl_context->window, workspace,
            nccl_context->scaleup_rank_idx, source_rank_idx, generation,
            latest_streaming_num_ranks,
            latest_streaming_num_ranks * latest_streaming_num_local_experts,
            num_gpu_timeout_cycles, stream);
        streaming_lane_release_events[source_rank_idx] = EventHandle(stream);
        streaming_lane_ingress_released[source_rank_idx] = true;
    }

    void streaming_combine_return(
        torch::Tensor lane_output,
        torch::Tensor lane_src_metadata,
        const int& source_rank_idx,
        const uint64_t& generation) const {
        EP_HOST_ASSERT(get_env<int>("EP_EXPERIMENTAL_STREAMING_LAYER") != 0 and
                       "Streaming layer return is disabled");
        EP_HOST_ASSERT(streaming_lane_view_outstanding and
                       latest_streaming_packed_x.has_value() and
                       latest_streaming_packed_lane_control.has_value());
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1 and
                       nccl_context->is_scaleup_nvlink and
                       "Streaming layer return requires one NVLink scale-up domain");
        EP_HOST_ASSERT(generation == latest_streaming_generation);
        EP_HOST_ASSERT(latest_streaming_return_buffer != nullptr);
        EP_HOST_ASSERT(0 <= source_rank_idx and
                       source_rank_idx < latest_streaming_num_ranks);
        EP_HOST_ASSERT(streaming_lane_return_submitted.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks) and
                       streaming_lane_ingress_released.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks) and
                       streaming_lane_return_events.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks));
        EP_HOST_ASSERT(not streaming_lane_return_submitted[source_rank_idx] and
                       "Streaming lane return was already submitted");
        EP_HOST_ASSERT(lane_output.dim() == 2 and lane_output.is_cuda() and
                       lane_output.is_contiguous() and
                       lane_output.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(lane_output.size(0) == latest_streaming_lane_route_capacity and
                       lane_output.size(1) == latest_streaming_packed_x->size(1));
        EP_HOST_ASSERT(lane_src_metadata.dim() == 2 and
                       lane_src_metadata.is_cuda() and
                       lane_src_metadata.is_contiguous() and
                       lane_src_metadata.scalar_type() == torch::kInt);
        const auto lane_control_stride = 2 + latest_streaming_num_local_experts;
        const auto lane_counts =
            latest_streaming_packed_lane_control->data_ptr<int>() +
            source_rank_idx * lane_control_stride;
        const int num_topk = lane_src_metadata.size(1) - 2;
        const int num_max_tokens_per_rank = lane_src_metadata.size(0);
        EP_HOST_ASSERT(num_topk > 0 and num_max_tokens_per_rank > 0);

        const auto stream = at::cuda::getCurrentCUDAStream();
        launch_streaming_combine_return(
            lane_output.data_ptr(), lane_src_metadata.data_ptr<int>(),
            lane_counts,
            source_rank_idx * latest_streaming_lane_route_capacity,
            latest_streaming_lane_route_capacity,
            nccl_context->dev_comm, nccl_context->window,
            latest_streaming_return_buffer, workspace,
            source_rank_idx, nccl_context->scaleup_rank_idx,
            generation,
            latest_streaming_num_ranks, lane_output.size(1),
            num_max_tokens_per_rank,
            latest_streaming_num_ranks * latest_streaming_num_local_experts,
            num_topk, nccl_context->num_allocated_qps,
            num_gpu_timeout_cycles,
            false,
            stream);
        lane_output.record_stream(stream);
        lane_src_metadata.record_stream(stream);
        latest_streaming_packed_lane_control->record_stream(stream);
        streaming_lane_return_events[source_rank_idx] = EventHandle(stream);
        streaming_lane_return_submitted[source_rank_idx] = true;
    }

    void streaming_combine_return_many(
        torch::Tensor lane_output,
        torch::Tensor src_metadata,
        const uint64_t& generation) const {
        EP_HOST_ASSERT(get_env<int>("EP_EXPERIMENTAL_STREAMING_LAYER") != 0 and
                       "Streaming layer return is disabled");
        EP_HOST_ASSERT(streaming_lane_view_outstanding and
                       latest_streaming_packed_x.has_value() and
                       latest_streaming_packed_lane_control.has_value());
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1 and
                       nccl_context->is_scaleup_nvlink and
                       "Streaming layer return requires one NVLink scale-up domain");
        EP_HOST_ASSERT(generation == latest_streaming_generation);
        EP_HOST_ASSERT(latest_streaming_return_buffer != nullptr);
        EP_HOST_ASSERT(streaming_lane_return_submitted.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks) and
                       streaming_lane_ingress_released.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks) and
                       streaming_lane_return_events.size() ==
                           static_cast<size_t>(latest_streaming_num_ranks));
        EP_HOST_ASSERT(std::none_of(streaming_lane_return_submitted.begin(),
                                   streaming_lane_return_submitted.end(),
                                   [](bool submitted) { return submitted; }) and
                       "A streaming lane return was already submitted");
        EP_HOST_ASSERT(lane_output.dim() == 3 and lane_output.is_cuda() and
                       lane_output.is_contiguous() and
                       lane_output.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(lane_output.size(0) == latest_streaming_num_ranks and
                       lane_output.size(1) == latest_streaming_lane_route_capacity and
                       lane_output.size(2) == latest_streaming_packed_x->size(1));
        EP_HOST_ASSERT(src_metadata.dim() == 2 and src_metadata.is_cuda() and
                       src_metadata.is_contiguous() and
                       src_metadata.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(src_metadata.size(0) % latest_streaming_num_ranks == 0);
        const int num_topk = src_metadata.size(1) - 2;
        const int num_max_tokens_per_rank =
            src_metadata.size(0) / latest_streaming_num_ranks;
        EP_HOST_ASSERT(num_topk > 0 and num_max_tokens_per_rank > 0);

        const auto stream = at::cuda::getCurrentCUDAStream();
        launch_streaming_combine_return(
            lane_output.data_ptr(), src_metadata.data_ptr<int>(),
            latest_streaming_packed_lane_control->data_ptr<int>(),
            0, latest_streaming_lane_route_capacity,
            nccl_context->dev_comm, nccl_context->window,
            latest_streaming_return_buffer, workspace,
            -1, nccl_context->scaleup_rank_idx, generation,
            latest_streaming_num_ranks, lane_output.size(2),
            num_max_tokens_per_rank,
            latest_streaming_num_ranks * latest_streaming_num_local_experts,
            num_topk, nccl_context->num_allocated_qps,
            num_gpu_timeout_cycles, true, stream);
        lane_output.record_stream(stream);
        src_metadata.record_stream(stream);
        latest_streaming_packed_lane_control->record_stream(stream);
        for (int source_rank_idx = 0;
             source_rank_idx < latest_streaming_num_ranks;
             ++source_rank_idx) {
            streaming_lane_return_events[source_rank_idx] = EventHandle(stream);
            streaming_lane_return_submitted[source_rank_idx] = true;
        }
    }

    std::tuple<torch::Tensor, EventHandle> streaming_combine_reduce(
        torch::Tensor topk_idx,
        const uint64_t& generation) const {
        EP_HOST_ASSERT(get_env<int>("EP_EXPERIMENTAL_STREAMING_LAYER") != 0 and
                       "Streaming layer reduce is disabled");
        EP_HOST_ASSERT(streaming_lane_view_outstanding and
                       latest_streaming_packed_x.has_value());
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1 and
                       nccl_context->is_scaleup_nvlink);
        EP_HOST_ASSERT(generation == latest_streaming_generation);
        EP_HOST_ASSERT(latest_streaming_return_buffer != nullptr);
        EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_cuda() and
                       topk_idx.is_contiguous() and
                       topk_idx.scalar_type() ==
                           c10::CppTypeToScalarType<topk_idx_t>::value);
        const int num_combined_tokens = topk_idx.size(0);
        const int num_topk = topk_idx.size(1);
        const int hidden = latest_streaming_packed_x->size(1);
        const int num_max_tokens_per_rank =
            latest_streaming_packed_src_metadata->size(0) /
            latest_streaming_num_ranks;
        // Expert GEMM and weighted return always produce BF16, regardless of
        // whether dispatch transported BF16 or FP8 activations.
        auto combined_x = torch::empty(
            {num_combined_tokens, hidden},
            latest_streaming_packed_x->options().dtype(torch::kBFloat16));
        const auto stream = at::cuda::getCurrentCUDAStream();
        launch_streaming_combine_reduce(
            combined_x.data_ptr(), topk_idx.data_ptr<topk_idx_t>(),
            nccl_context->dev_comm, nccl_context->window,
            latest_streaming_return_buffer, workspace,
            nccl_context->scaleup_rank_idx,
            num_combined_tokens, generation,
            latest_streaming_num_ranks, hidden,
            num_max_tokens_per_rank,
            latest_streaming_num_ranks * latest_streaming_num_local_experts,
            num_topk, stream);
        topk_idx.record_stream(stream);
        combined_x.record_stream(stream);
        return {combined_x, EventHandle(stream)};
    }

    std::tuple<int, int> get_physical_domain_size() const {
        return {nccl_context->num_rdma_ranks, nccl_context->num_nvl_ranks};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        return {nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks};
    }

    // ReSharper disable once CppMemberFunctionMayBeStatic
    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync, const bool& sequential = true) const {
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto stream = use_comm_stream ? comm_stream : compute_stream;
        if (use_comm_stream)
            stream_wait(comm_stream, compute_stream);

        // Wait all streams to finish on this GPU
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Launch GPU barrier
        launch_barrier(nccl_context->dev_comm, nccl_context->window,
                       workspace,
                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                       num_gpu_timeout_cycles,
                       nccl_context->is_scaleup_nvlink,
                       sequential,
                       stream);

        // Let CPU wait
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Compute stream should also wait for the barrier
        if (use_comm_stream)
            stream_wait(compute_stream, comm_stream);
    }

    void engram_write(const torch::Tensor& storage, const std::optional<torch::Tensor>& sf) {
        // Ensure previous fetch are finished
        barrier(false, true);

        const auto compute_stream = at::cuda::getCurrentCUDAStream();

        // Check storage
        const auto [num_entries, hidden] = get_shape<2>(storage);
        EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16 or
                       storage.scalar_type() == torch::kFloat8_e4m3fn);
        EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());
        num_engram_entries = num_entries, engram_hidden = hidden;

        // Store globally-replicated FP8 scaling factors for local gather during fetch.
        if (sf.has_value())
            EP_HOST_ASSERT(sf->dim() == 2 and sf->is_cuda() and sf->is_contiguous() and
                           sf->element_size() == sizeof(sf_pack_t));
        engram_sf = sf;

        // Write storage to CPU segment at back of buffer
        EP_HOST_ASSERT(storage.nbytes() <= num_cpu_buffer_bytes and "Engram storage exceeds CPU buffer size");
        const auto cpu_write_offset = allow_hybrid_mode
            ? static_cast<int64_t>(nccl_context->scaleup_rank_idx) * num_cpu_buffer_bytes : 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            math::advance_ptr(buffer, num_gpu_buffer_bytes + cpu_write_offset),
            storage.data_ptr(), storage.nbytes(),
            cudaMemcpyDeviceToDevice, compute_stream));

        // Ensure data is visible for all ranks
        barrier(false, true);
    }

    std::function<std::tuple<torch::Tensor, std::optional<torch::Tensor>>()>
    engram_fetch(const torch::Tensor& indices, int num_qps, const bool& use_tma_aligned_col_major_sf) const {
        const auto use_fp8 = engram_sf.has_value();
        const auto fetched_dtype = use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
        const int elem_size = use_fp8 ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        const auto [num_tokens, num_entries_per_token] = get_shape<2>(indices);
        EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(indices.is_cuda() and indices.is_contiguous());
        EP_HOST_ASSERT(num_tokens * num_entries_per_token * engram_hidden * elem_size <= num_gpu_buffer_bytes);

        // Calculate a QP count
        if (num_qps == 0)
            num_qps = nccl_context->num_allocated_qps;

        // Return tensor from the raw buffer: each token's entries are concatenated along hidden
        EP_HOST_ASSERT(num_engram_entries > 0);
        const auto fetched = torch::from_blob(
            buffer,
            {num_tokens, engram_hidden * num_entries_per_token},
            torch::TensorOptions().dtype(fetched_dtype).device(torch::kCUDA)
        );

        auto fetched_sf = std::optional<torch::Tensor>();
        void* sf_table_ptr = nullptr;
        void* fetched_sf_ptr = nullptr;
        int num_sf_packs = 0, sf_token_stride = 0, sf_hidden_stride = 0;
        if (use_fp8) {
            num_sf_packs = static_cast<int>(engram_sf->size(1));
            if (use_tma_aligned_col_major_sf) {
                // TMA-aligned column-major layout for the next GEMM input
                sf_token_stride = 1, sf_hidden_stride = math::align(num_tokens, kNumAlignedSFPacks);
            } else {
                sf_token_stride = num_entries_per_token * num_sf_packs, sf_hidden_stride = 1;
            }
            fetched_sf = torch::empty_strided({num_tokens, num_entries_per_token * num_sf_packs},
                                              {sf_token_stride, sf_hidden_stride}, engram_sf->options());
            sf_table_ptr = engram_sf->data_ptr();
            fetched_sf_ptr = fetched_sf->data_ptr();
        }

        // Last issued Gin requests
        const auto num_gin_ranks = allow_hybrid_mode
            ? nccl_context->num_scaleout_ranks
            : nccl_context->num_ranks;
        const auto last_gin_requests = torch::empty(
            {num_gin_ranks * num_qps, sizeof(ncclGinRequest_t)},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );

        // Launch the fetch kernel
        launch_engram_fetch(
            nccl_context->dev_comm, nccl_context->window,
            math::advance_ptr(buffer, num_gpu_buffer_bytes),
            fetched.data_ptr(),
            indices.data_ptr<int>(),
            static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
            sf_table_ptr, fetched_sf_ptr, sf_token_stride, sf_hidden_stride,
            num_engram_entries,
            engram_hidden, elem_size, num_sf_packs,
            num_entries_per_token,
            num_tokens,
            nccl_context->num_scaleout_ranks,
            nccl_context->num_scaleup_ranks,
            num_cpu_buffer_bytes,
            num_qps,
            allow_hybrid_mode,
            at::cuda::getCurrentCUDAStream()
        );

        return [=, this]() -> std::tuple<torch::Tensor, std::optional<torch::Tensor>> {
            // Wait for all RDMA gets to complete
            launch_engram_fetch_wait(
                static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
                nccl_context->dev_comm,
                nccl_context->window,
                nccl_context->num_scaleout_ranks,
                nccl_context->num_scaleup_ranks,
                num_qps,
                allow_hybrid_mode,
                at::cuda::getCurrentCUDAStream()
            );
            return {fetched, fetched_sf};
        };
    }

    void pp_set_config(const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors) {
        // Flush previous operations
        barrier(false, true);

        EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);
        EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <= num_buffer_bytes);
        this->prev_rank_idx = (nccl_context->rank_idx + nccl_context->num_ranks - 1) % nccl_context->num_ranks;
        this->next_rank_idx = (nccl_context->rank_idx + 1) % nccl_context->num_ranks;
        this->num_max_pp_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, 32);
        this->num_max_pp_inflight_tensors = num_max_inflight_tensors;
    }

    void pp_send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or dst_rank_idx == next_rank_idx);

        launch_pp_send(
            nccl_context->dev_comm, nccl_context->window,
            x.data_ptr(), x.nbytes(),
            buffer, workspace,
            nccl_context->rank_idx, dst_rank_idx, nccl_context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
            num_gpu_timeout_cycles,
            jit::device_runtime->get_num_smem_bytes(),
            at::cuda::getCurrentCUDAStream()
        );
    }

    void pp_recv(const torch::Tensor& x, const int& src_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(src_rank_idx == prev_rank_idx or src_rank_idx == next_rank_idx);

        launch_pp_recv(
            nccl_context->dev_comm, nccl_context->window,
            x.data_ptr(), x.nbytes(),
            buffer, workspace,
            nccl_context->rank_idx, src_rank_idx, nccl_context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
            num_gpu_timeout_cycles,
            jit::device_runtime->get_num_smem_bytes(),
            at::cuda::getCurrentCUDAStream()
        );
    }

    void agrs_set_config(const int64_t& num_max_session_bytes,
                         const int& new_num_max_agrs_per_session) {
        // Flush previous operations
        barrier(true, true);

        EP_HOST_ASSERT(nccl_context->num_ranks > 1);
        EP_HOST_ASSERT(num_max_session_bytes > 0 and new_num_max_agrs_per_session > 0);
        EP_HOST_ASSERT(num_max_session_bytes <= num_buffer_bytes);
        EP_HOST_ASSERT(new_num_max_agrs_per_session <= layout::WorkspaceLayout::kNumMaxInflightAGRS);
        EP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks);
        this->num_max_agrs_session_bytes = math::align<int64_t>(num_max_session_bytes, 32);
        this->num_max_agrs_per_session = new_num_max_agrs_per_session;
    }

    void create_agrs_session() {
        EP_HOST_ASSERT(not agrs_in_session);
        agrs_in_session = true;
        agrs_buffer_offset = 0;
        agrs_buffer_slot_idx = 0;
        agrs_session_idx += 1;
    }

    void destroy_agrs_session() {
        // Must be in a session
        EP_HOST_ASSERT(agrs_in_session);
        agrs_in_session = false;

        // Wait compute stream
        stream_wait(comm_stream, at::cuda::getCurrentCUDAStream());

        // Notify that the buffer is now available & Wait for the buffer to be ready
        // NOTES: self-wait is guaranteed by in-stream order
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
            write_ptrs[i] = static_cast<int*>(
                nccl_context->get_sym_ptr(workspace_layout_wo_expert->get_agrs_session_signal_ptr(nccl_context->rank_idx), dst_rank_idx));
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_session_signal_ptr(dst_rank_idx);
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, agrs_session_idx);
    }

    std::vector<torch::Tensor> agrs_get_inplace_tensor(const std::vector<int64_t>& num_bytes_list) const {
        EP_HOST_ASSERT(num_bytes_list.size() >= 1);
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);

        std::vector<torch::Tensor> out;
        out.reserve(num_bytes_list.size());
        int64_t offset = agrs_buffer_offset;
        for (const auto& num_bytes: num_bytes_list) {
            EP_HOST_ASSERT(offset + num_bytes * nccl_context->num_ranks <= num_max_agrs_session_bytes and
                           agrs_buffer_slot_idx < num_max_agrs_per_session and
                           "Not enough session buffer size. Did you forget to flush session?");
            out.push_back(torch::from_blob(math::advance_ptr(buffer, offset + num_bytes * nccl_context->rank_idx),
                          {num_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)));
            offset += math::align<int64_t>(num_bytes * nccl_context->num_ranks, 32);
        }
        return out;
    }

    std::pair<std::vector<torch::Tensor>, std::function<void()>>
    all_gather(const std::vector<torch::Tensor>& tensors) {
        const int num_tensors = tensors.size();
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);
        EP_HOST_ASSERT(num_tensors >= 1);

        int num_copies = 0;
        std::vector<int64_t> offset(num_tensors);
        for (int i = 0; i < num_tensors; ++ i) {
            const auto& x = tensors[i];
            EP_HOST_ASSERT(x.is_contiguous());

            const auto x_offset = math::ptr_diff(x.data_ptr(), buffer);
            const bool is_inplace = 0 <= x_offset and x_offset < num_max_agrs_session_bytes;
            offset[i] = agrs_buffer_offset;
            num_copies += nccl_context->num_ranks - is_inplace;
            agrs_buffer_offset += math::align<int64_t>(x.nbytes() * nccl_context->num_ranks, 32);
            EP_HOST_ASSERT(not is_inplace or x.data_ptr() == math::advance_ptr(buffer, offset[i] + x.nbytes() * nccl_context->rank_idx));
        }
        EP_HOST_ASSERT(agrs_buffer_offset <= num_max_agrs_session_bytes and
                       agrs_buffer_slot_idx < num_max_agrs_per_session and
                       "Not enough session buffer size. Did you forget to flush session?");

        // Wait compute stream
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        stream_wait(comm_stream, compute_stream);

        // Send data to all ranks
        std::vector<size_t> sizes(num_copies);
        std::vector<void*> dst_ptrs(num_copies), src_ptrs(num_copies);
        int count = 0;
        for (int i = 0; i < nccl_context->num_ranks; ++ i) {
            for (int j = 0; j < num_tensors; ++ j) {
                const auto& x = tensors[j];
                const auto dst_rank_idx = (nccl_context->rank_idx + i) % nccl_context->num_ranks;
                void* src_ptr = x.data_ptr();
                void* dst_ptr =
                    nccl_context->get_sym_ptr(math::advance_ptr(buffer, offset[j] + x.nbytes() * nccl_context->rank_idx), dst_rank_idx);
                if (src_ptr != dst_ptr) {
                    src_ptrs[count] = src_ptr;
                    dst_ptrs[count] = dst_ptr;
                    sizes[count] = x.nbytes();
                    count += 1;
                }
            }
        }
        cudaMemcpyAttributes attrs = {
            .srcAccessOrder = cudaMemcpySrcAccessOrderStream,
            .flags = cudaMemcpyFlagPreferOverlapWithCompute
        };
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, comm_stream));
#else
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, nullptr, comm_stream));
#endif

        // Wait for data from other ranks
        const int current_session = agrs_session_idx;
        const int slot_idx = agrs_buffer_slot_idx;
        agrs_buffer_slot_idx += 1;
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
            write_ptrs[i] = nccl_context->get_sym_ptr(
                workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, nccl_context->rank_idx), dst_rank_idx);
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, dst_rank_idx);
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, current_session);

        // Build output tensors eagerly
        std::vector<torch::Tensor> out(num_tensors);
        for (int i = 0; i < num_tensors; ++ i) {
            auto shape = tensors[i].sizes().vec();
            shape.insert(shape.begin(), nccl_context->num_ranks);
            out[i] = torch::from_blob(math::advance_ptr(buffer, offset[i]), shape, tensors[i].options());
        }

        // Return tensors and a handle to wait for data arrival
        const auto event = EventHandle(comm_stream);
        auto handle = [=, this]() {
            EP_HOST_ASSERT(compute_stream == at::cuda::getCurrentCUDAStream());
            EP_HOST_ASSERT(agrs_in_session and current_session == this->agrs_session_idx);
            stream_wait(compute_stream, event);
        };
        return {std::move(out), std::move(handle)};
    }

    torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                    const bool& allocate_on_comm_stream,
                                                    const bool& async_with_compute_stream) const {
        // Allocate all tensors on communication stream if set
        // NOTES: do not allocate tensors upfront!
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(comm_stream);

        // Assertion for safety
        // `previous_event` implicitly means the overlapping computation kernels are launched first,
        // in order not to use the memory on the compute stream, we must allocate on the communication stream.
        // If you launch the communication kernels firstly, then `previous_event` must be unnecessary.
        if (previous_event.has_value())
            EP_HOST_ASSERT(allocate_on_comm_stream);

        // Wait previous tasks to finish
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }
        return compute_stream;
    }

    void stream_control_before_epilogue(const std::optional<EventHandle>& previous_event_before_epilogue) const {
        if (previous_event_before_epilogue.has_value())
            stream_wait(comm_stream, previous_event_before_epilogue.value());
    }

    std::optional<EventHandle> stream_control_epilogue_on_stream(
                                                       const std::vector<std::optional<torch::Tensor>>& tensors,
                                                       const at::cuda::CUDAStream& compute_stream,
                                                       const at::cuda::CUDAStream& completion_stream,
                                                       const bool& also_record_comm_stream,
                                                       const bool& allocate_on_comm_stream,
                                                       const bool& async_with_compute_stream) const {
        // Ensure memory access safety between two streams
        std::optional<EventHandle> event;
        if (async_with_compute_stream) {
            event = EventHandle(completion_stream);

            // NOTES: this environment only applies to V2 APIs
            if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
                event->tensors_to_record = tensors;
            } else {
                for (auto& t: tensors) if (t.has_value()) {
                    t->record_stream(compute_stream);
                    t->record_stream(completion_stream);
                    if (also_record_comm_stream and comm_stream.id() != completion_stream.id())
                        t->record_stream(comm_stream);
                }
            }
        } else {
            stream_wait(compute_stream, completion_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // The CUDA event marking the finishing
        return event;
    }

    std::optional<EventHandle> stream_control_epilogue(const std::vector<std::optional<torch::Tensor>>& tensors,
                                                       const at::cuda::CUDAStream& compute_stream,
                                                       const bool& allocate_on_comm_stream,
                                                       const bool& async_with_compute_stream) const {
        return stream_control_epilogue_on_stream(
            tensors, compute_stream, comm_stream, false,
            allocate_on_comm_stream, async_with_compute_stream);
    }

    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& hidden, const int& num_sf_packs, const int& num_topk,
                                            const int& elem_size,
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                            const bool& is_scaleup_nvlink) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct dispatch
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_ranks, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid dispatch
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, 1, num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleout_ranks,
                /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const bool& is_scaleup_nvlink,
                                           const bool& allow_multiple_reduction) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct combine
            // Streaming return addresses one slot per destination rank.  The
            // regular multiple-reduction layout needs only min(ranks, top-k),
            // which is too small for EP8 models whose router top-k is smaller.
            const auto num_tokens_in_layout =
                get_env<int>("EP_EXPERIMENTAL_STREAMING_LAYER") != 0
                    ? num_ranks
                    : (allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk);
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : num_ranks,
                // For single reduction cases, the maximum number of received tokens is
                // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
                num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid combine
            const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
            const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, allow_multiple_reduction ? 1 : num_topk,
                /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
                num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t calculate_buffer_size(const int64_t& nccl_comm,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch,
                                         const bool& allow_hybrid_mode,
                                         const bool& allow_multiple_reduction) {
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

        // The worst case SF bytes must be less than the main part
        EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

        // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Topology
        const auto [num_rdma_ranks, num_nvl_ranks] = nccl::get_physical_domain_size(nccl_comm);
        const auto [num_scaleout_ranks, num_scaleup_ranks] = nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
        const auto is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

        // Dispatch size
        const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0; // An approximation for number of SF packs
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink);

        // Combine layout
        const auto num_combine_bytes = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink, allow_multiple_reduction);

        // The regular path serializes dispatch and combine and can overlay the
        // same region.  Streaming layers overlap them, so reserve disjoint
        // symmetric regions with an aligned boundary between the layouts.
        if (get_env<int>("EP_EXPERIMENTAL_STREAMING_LANES") != 0 and
            get_env<int>("EP_EXPERIMENTAL_STREAMING_LAYER") != 0) {
            return math::align(
                math::align(num_dispatch_bytes, symmetric::kNumAlignmentBytes) +
                    num_combine_bytes,
                symmetric::kNumAlignmentBytes);
        }
        return math::align(
            std::max(num_dispatch_bytes, num_combine_bytes),
            symmetric::kNumAlignmentBytes);
    }

    static symmetric::cpu_handle_t create_cpu_handle(const int64_t& num_cpu_bytes) {
        EP_HOST_ASSERT(num_cpu_bytes > 0 and num_cpu_bytes % symmetric::kNumAlignmentBytes == 0);
        return symmetric::HybridElasticSymmetricMemory::create_cpu_handle(num_cpu_bytes);
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               int, int,
               std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const std::optional<torch::Tensor>& cached_channel_linked_list,
             const int& num_max_tokens_per_rank,
             const int& num_experts, const int& expert_alignment,
             const int& num_sms, const int& num_qps,
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf) const {
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);

        // Zero padding only makes sense with expand mode
        EP_HOST_ASSERT(not do_zero_padding or do_expand);

        // Cached mode must have responding handles
        const bool cached_mode = cached_num_recv_tokens.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(cached_num_recv_tokens.has_value());
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
            EP_HOST_ASSERT(cached_num_expanded_tokens.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_scaleup_rank.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());

            // Hybrid kernels require more
            if (nccl_context->num_scaleout_ranks > 1) {
                EP_HOST_ASSERT(cached_token_metadata_at_forward.has_value());
                EP_HOST_ASSERT(cached_channel_linked_list.has_value());
            }
        }

        // Check data tensor
        const auto [num_tokens, hidden] = get_shape<2>(x);
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());
        const auto num_local_experts = num_experts / nccl_context->num_ranks;
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        // Check SF stuffs
        int num_sf_packs = 0;
        void* sf_ptr = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        if (sf.has_value()) {
            // SF must be FP32 or packed UE8M0x4
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
            EP_HOST_ASSERT(num_tokens == num_tokens_);
            EP_HOST_ASSERT(sf->is_cuda());
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            num_sf_packs = num_sf_packs_;
            sf_ptr = sf->data_ptr();
            sf_token_stride = sf->stride(0);
            sf_hidden_stride = sf->stride(1);
        }

        // Check top-k stuffs
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);
        EP_HOST_ASSERT(num_tokens == num_tokens_);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__);
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // The number of received tokens per expert
        // This is useful for expanding mode
        EP_HOST_ASSERT(num_experts % nccl_context->num_ranks == 0);
        auto psum_num_recv_tokens_per_expert = cached_psum_num_recv_tokens_per_expert.value_or(torch::Tensor());
        if (cached_mode) {
            const auto& [num_local_experts_] = get_shape<1>(psum_num_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_cuda() and psum_num_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
            psum_num_recv_tokens_per_expert = torch::empty(
                {num_local_experts + 1}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // The unaligned (actual) number of received tokens per expert
        // Written by the dispatch kernel's notify warps, used by the epilogue for zero padding
        auto num_unaligned_recv_tokens_per_expert = cached_num_unaligned_recv_tokens_per_expert.value_or(torch::Tensor());
        int* num_unaligned_recv_tokens_per_expert_ptr = nullptr;
        if (cached_mode) {
            const auto& [num_local_experts_] = get_shape<1>(num_unaligned_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.is_cuda() and num_unaligned_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            num_unaligned_recv_tokens_per_expert = torch::empty(
                {num_local_experts}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }
        num_unaligned_recv_tokens_per_expert_ptr = num_unaligned_recv_tokens_per_expert.data_ptr<int>();

        // The prefix sum tensor of number of received tokens from each rank
        // Will also be used in combine as the dispatch handle
        auto psum_num_recv_tokens_per_scaleup_rank = cached_psum_num_recv_tokens_per_scaleup_rank.value_or(torch::Tensor());
        if (cached_mode) {
            const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
            EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        } else {
            psum_num_recv_tokens_per_scaleup_rank = torch::empty(
                {nccl_context->num_scaleup_ranks}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // Decide number of channels by shared memory consumption
        // Only for hybrid version
        int num_channels_per_sm = 1, num_channels = 1;
        const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
        if (nccl_context->num_scaleout_ranks > 1) {
            const auto dispatch_token_layout = get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk);
            const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
            EP_HOST_ASSERT(num_sms <= kNumMaxSMs);
            num_channels_per_sm = std::min<int>(
                (num_smem_bytes - get_num_notify_smem_bytes(nccl_context->num_ranks, num_experts)) / dispatch_token_layout.get_num_bytes<true>(),
                32 - kNumNotifyWarps);
            num_channels_per_sm = std::min<int>(
                num_smem_bytes / combine_token_layout.get_num_bytes<true>(),
                num_channels_per_sm);
            num_channels_per_sm = std::min<int>(
                /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
            if (not prefer_overlap_with_compute)
                num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
            num_channels = num_sms * num_channels_per_sm;
            if (get_env<int>("EP_BUFFER_DEBUG"))
                printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);
        }

        // Non-hybrid mode handles
        auto dst_buffer_slot_idx = cached_dst_buffer_slot_idx.value_or(torch::Tensor());
        if (nccl_context->num_scaleout_ranks == 1) {
            if (cached_mode) {
                const auto [num_tokens__, num_topk_] = get_shape<2>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                // Allocate a new tensor
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            }
        }

        // Hybrid mode handles
        std::optional<torch::Tensor> token_metadata_at_forward, channel_linked_list;
        int *token_metadata_at_forward_ptr = nullptr, *channel_linked_list_ptr = nullptr;
        if (nccl_context->num_scaleout_ranks > 1) {
            // The token destination slot idx during forward
            // `[i, j, k, l]` means: from channel i from scale-out peer k, the j-th token's index in the l-th rank buffer
            // NOTES: Used primarily for cached mode
            // TODO: May make it a linked list to remove the redundant info in `token_metadata_at_forward`
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
            if (cached_mode) {
                const auto [num_channels_, num_scaleout_ranks_, num_max_tokens_per_channel_, num_topk_] =
                    get_shape<4>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_channels == num_channels_ and nccl_context->num_scaleout_ranks == num_scaleout_ranks_ and
                               num_max_tokens_per_channel == num_max_tokens_per_channel_ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                dst_buffer_slot_idx = torch::empty(
                    {num_channels, nccl_context->num_scaleout_ranks, num_max_tokens_per_channel, num_topk},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }

            // The token metadata during forward
            // `[i, j]` means: in channel i, the j-th forwarded token's metadata
            // Info contains:
            //   - Scaleout rank index and source token index in the original rank (0)
            //   - Whether the token is the last one in the chunk (1)
            //   - cached top-k scaleup peer indices (top-k)
            //   - each selections' destination slot indices (top-k)
            const auto num_max_forwarded_tokens = nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1;
            const auto num_forward_metadata_dims = 2 + num_topk * 2;
            if (cached_mode) {
                token_metadata_at_forward = cached_token_metadata_at_forward;
                const auto [num_channels_, num_max_forwarded_tokens_, num_forward_metadata_dims_] = get_shape<3>(token_metadata_at_forward.value());
                EP_HOST_ASSERT(num_channels == num_channels_ and num_max_forwarded_tokens == num_max_forwarded_tokens_
                               and num_forward_metadata_dims == num_forward_metadata_dims_);
                EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
                EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);
            } else {
                token_metadata_at_forward = torch::empty(
                    {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();

            // Per-scaleup-peer-per-channel linked list
            // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
            if (cached_mode) {
                channel_linked_list = cached_channel_linked_list;
                const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
                channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
                EP_HOST_ASSERT(num_channels == num_channels__);
                EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
                EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
                EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
                EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
            } else {
                channel_linked_list = torch::empty(
                    // Index 0 of the list means the starting item
                    {num_channels,
                    nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1,
                    nccl_context->num_scaleup_ranks},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
        }

        // Clone `topk_idx` for saving in the handle (to prevent users' modification)
        auto copied_topk_idx = std::optional<torch::Tensor>();
        topk_idx_t* copied_topk_idx_ptr = nullptr;
        if (do_handle_copy and not cached_mode) {
            copied_topk_idx = torch::empty_like(topk_idx);
            copied_topk_idx_ptr = copied_topk_idx->data_ptr<topk_idx_t>();
        }

        // Check buffer size
        EP_HOST_ASSERT(get_dispatch_buffer_size(
                       num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, x.element_size(),
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                       nccl_context->is_scaleup_nvlink) <= num_buffer_bytes);

        // Ready and clean host workspace for this round
        const auto host_workspace_layout = layout::WorkspaceLayout(
            host_workspace,
            nccl_context->num_scaleout_ranks,
            nccl_context->num_scaleup_ranks,
            num_experts);
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), nccl_context->num_scaleup_ranks, 0);
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Do dispatch into the buffers (with SM limitation)
        EP_HOST_ASSERT(num_sms <= jit::device_runtime->get_num_sms());
        const bool export_streaming_lanes = get_env<int>("EP_EXPERIMENTAL_STREAMING_LANES") != 0;
        const bool run_streaming_shadow = get_env<int>("EP_EXPERIMENTAL_STREAMING_COPY_SHADOW") != 0;
        const bool run_rank_ready = get_env<int>("EP_EXPERIMENTAL_RANK_READY") != 0;
        const bool instrument_streaming_bulk =
            get_env<int>("EP_EXPERIMENTAL_STREAMING_INSTRUMENTED_BULK") != 0;
        const bool request_streaming_signals =
            get_env<int>("EP_EXPERIMENTAL_STREAMING_SIGNALS") != 0;
        EP_HOST_ASSERT(static_cast<int>(export_streaming_lanes) +
                       static_cast<int>(run_streaming_shadow) +
                       static_cast<int>(run_rank_ready) +
                       static_cast<int>(instrument_streaming_bulk) <= 1);
        const bool run_streaming_copy = export_streaming_lanes or run_streaming_shadow;
        // Preserve the original signal-only switch as a safe bulk arm. It
        // publishes the control plane, but still materializes and consumes
        // the ordinary merged layout.
        const bool run_signal_only =
            request_streaming_signals and not run_streaming_copy and
            not run_rank_ready and not instrument_streaming_bulk;
        const bool emit_streaming_signals =
            run_streaming_copy or run_rank_ready or instrument_streaming_bulk or
            run_signal_only;
        // Shadow/instrumented arms publish the generation control plane while
        // retaining the upstream barrier, count exchange, and merged epilogue.
        // Only a real lane consumer or rank-ready replacement may bypass it.
        const bool bypass_legacy_completion =
            export_streaming_lanes or run_rank_ready;
        EP_HOST_ASSERT(not emit_streaming_signals or
                       (nccl_context->num_scaleout_ranks == 1 and
                        nccl_context->is_scaleup_nvlink and not cached_mode));
        EP_HOST_ASSERT(not run_streaming_copy or
                       (do_expand and not do_cpu_sync and not do_zero_padding));
        EP_HOST_ASSERT(not run_rank_ready or
                       (do_expand and not do_cpu_sync and not do_zero_padding and
                        not cached_mode and nccl_context->num_scaleout_ranks == 1));
        EP_HOST_ASSERT(not run_rank_ready or get_env<int>("EP_AVOID_RECORD_STREAM", 0) == 0);
        const uint64_t current_streaming_generation =
            emit_streaming_signals ? ++ streaming_generation : 0;
        if (export_streaming_lanes and
            get_env<int>("EP_EXPERIMENTAL_STREAMING_LAYER") != 0) {
            const auto dispatch_region_bytes = math::align(
                get_dispatch_buffer_size(
                    num_max_tokens_per_rank, hidden, num_sf_packs, num_topk,
                    x.element_size(), nccl_context->num_scaleout_ranks,
                    nccl_context->num_scaleup_ranks,
                    nccl_context->is_scaleup_nvlink),
                symmetric::kNumAlignmentBytes);
            const auto return_region_bytes = get_combine_buffer_size(
                num_max_tokens_per_rank, hidden, num_topk,
                nccl_context->num_scaleout_ranks,
                nccl_context->num_scaleup_ranks,
                nccl_context->is_scaleup_nvlink,
                allow_multiple_reduction);
            EP_HOST_ASSERT(dispatch_region_bytes + return_region_bytes <=
                           num_gpu_buffer_bytes and
                           "Streaming dispatch and combine require disjoint symmetric regions");
            latest_streaming_return_buffer =
                math::advance_ptr(buffer, dispatch_region_bytes);
        } else {
            latest_streaming_return_buffer = nullptr;
        }
        if (emit_streaming_signals) {
            EP_HOST_ASSERT(not streaming_lane_view_outstanding and
                           "Previous streaming lane view was not released");
            if (not streaming_generation_released_per_lane) {
                if (streaming_consumer_event.has_value()) {
                    stream_wait(comm_stream, streaming_consumer_event.value());
                    streaming_consumer_event.reset();
                }
                // Legacy/shadow release still has destination-wide lifetime
                // semantics.  Preserve its local joins exactly.
                if (streaming_copy_used)
                    stream_wait(comm_stream, streaming_copy_stream);
                if (streaming_lane_copy_used) {
                    for (const auto& source_stream : streaming_lane_copy_streams)
                        stream_wait(comm_stream, source_stream);
                }
            } else {
                // Per-source ACKs are consumed by the source-side reuse wait
                // below.  Joining this destination's unrelated lane streams
                // here would recreate an implicit rank barrier.
                EP_HOST_ASSERT(not streaming_consumer_event.has_value());
                // The caller can enqueue the next model layer before the
                // current stream's source-reduce wait has retired.  On SM100,
                // allowing that next dispatch to reuse the shared symmetric
                // ingress while lane GEMMs/returns are still outstanding can
                // race the device-side reuse protocol.  Preserve asynchronous
                // work within a generation, but fence generation reuse on the
                // accumulated lane lifetime event.
                if (streaming_lane_drain_event.has_value())
                    stream_wait(comm_stream, streaming_lane_drain_event.value());
            }
            streaming_generation_released_per_lane = false;
        }
        if (run_rank_ready or run_streaming_shadow) {
            streaming_copy_used = true;
        }
        if (run_streaming_copy)
            streaming_lane_copy_used = true;
        if (emit_streaming_signals and current_streaming_generation > 1) {
            launch_dispatch_streaming_reuse_wait(
                workspace, nccl_context->scaleup_rank_idx,
                current_streaming_generation - 1,
                nccl_context->num_ranks, num_experts,
                num_gpu_timeout_cycles, comm_stream);
        }
        launch_dispatch(x.data_ptr(), sf_ptr,
                        topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,
                        copied_topk_idx_ptr,
                        run_rank_ready ? nullptr : cumulative_local_expert_recv_stats_ptr,
                        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),
                        num_unaligned_recv_tokens_per_expert_ptr,
                        dst_buffer_slot_idx.data_ptr<int>(),
                        token_metadata_at_forward_ptr,
                        num_tokens, num_max_tokens_per_rank,
                        hidden, x.element_size(),
                        num_sf_packs, sf_token_stride, sf_hidden_stride,
                        num_experts, num_topk, expert_alignment,
                        nccl_context->dev_comm, nccl_context->window,
                        buffer,
                        workspace, mapped_host_workspace,
                        nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                        nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                        nccl_context->is_scaleup_nvlink,
                        num_sms, num_channels_per_sm,
                        num_smem_bytes,
                        num_qps, num_gpu_timeout_cycles,
                        cached_mode, emit_streaming_signals,
                        bypass_legacy_completion,
                        current_streaming_generation, do_cpu_sync,
                        comm_stream);

        // Received token counters
        int num_recv_tokens = 0, num_expanded_tokens = 0;
        int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;
        std::vector<int> num_recv_tokens_per_expert_list;

        // Assign these values according to modes
        if (cached_mode) {
            // Cached mode
            EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
            num_recv_tokens = cached_num_recv_tokens.value();
            num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
            num_expanded_tokens = cached_num_expanded_tokens.value();
        } else if (do_cpu_sync) {
            // Non-cached mode with sync
            const auto start_cpu_time = std::chrono::high_resolution_clock::now();
            while (true) {
                bool ready = true;

                // Read number of received tokens from each scaleup rank
                while (counter_scaleup_rank_idx < nccl_context->num_scaleup_ranks and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_rank_count_ptr<false>()[counter_scaleup_rank_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens += count;
                        ++ counter_scaleup_rank_idx;
                    }
                }

                // Read expert counts
                while (counter_local_expert_idx < num_local_experts and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_expert_count_ptr<false>()[counter_local_expert_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens_per_expert_list.push_back(count);
                        num_expanded_tokens += count;
                        ++ counter_local_expert_idx;
                    }
                }

                // Ready and do next steps
                const auto get_buffer_info = [&]() {
                    std::stringstream ss;
                    ss << "CPU side received count (scaleup: " << nccl_context->scaleup_rank_idx << "): ";
                    for (int i = 0; i < nccl_context->num_scaleup_ranks + num_local_experts; ++ i) {
                        ss << host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i];
                        ss << (i == nccl_context->num_scaleup_ranks - 1 ? " # ": " ");
                    }
                    return ss.str();
                };
                if (ready) {
                    if (get_env<int>("EP_BUFFER_DEBUG"))
                        printf("%s\n", get_buffer_info().c_str());
                    break;
                }

                // Timeout checks
                const auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > num_cpu_timeout_secs)
                    throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
            }
        } else {
            // Non-cached mode without CPU sync, allocate with the worst case
            num_recv_tokens = num_max_tokens_per_rank * nccl_context->num_ranks;
            num_expanded_tokens = nccl_context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_local_experts);
            num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
            num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
        }

        // Allocate received tensors
        // `recv_src_metadata` includes source token indices and buffer slot indices
        const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        auto recv_x = torch::empty({num_allocated_tokens, hidden}, x.options());
        auto recv_sf = std::optional<torch::Tensor>();
        auto recv_topk_idx = std::optional<torch::Tensor>();
        auto recv_topk_weights = std::optional<torch::Tensor>();
        auto recv_src_metadata = cached_mode ?
            cached_recv_src_metadata.value() :
            torch::empty({num_recv_tokens, num_topk + 2},
                         torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

        // Optional tensors
        void* recv_sf_ptr = nullptr;
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
        if (sf.has_value()) {
            if (not use_tma_aligned_col_major_sf) {
                recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;
            } else {
                // TMA-aligned layout for the next GEMM input
                recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks);
            }
            recv_sf = torch::empty_strided({num_allocated_tokens, num_sf_packs},
                                           {recv_sf_token_stride, recv_sf_hidden_stride},
                                           sf->options());
            recv_sf_ptr = recv_sf->data_ptr();
        }
        if (not do_expand) {
            recv_topk_idx = torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        }
        if (topk_weights.has_value()) {
            recv_topk_weights = do_expand ?
                torch::empty({num_allocated_tokens}, topk_weights->options()) :
                torch::empty({num_allocated_tokens, num_topk}, topk_weights->options());
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }

        // M1C sidecar output uses DeepGEMM-compatible source-lane psum layout.
        // EP_EXPERIMENTAL_STREAMING_LANES exports it to the grouped-GEMM
        // consumer; the shadow-only mode retains it as a diagnostic.
        auto streaming_packed_x = std::optional<torch::Tensor>();
        auto streaming_packed_sf = std::optional<torch::Tensor>();
        auto streaming_packed_topk_weights = std::optional<torch::Tensor>();
        auto streaming_packed_src_metadata = std::optional<torch::Tensor>();
        auto streaming_packed_lane_control = std::optional<torch::Tensor>();
        if (run_streaming_copy) {
            const int64_t raw_lane_route_capacity =
                static_cast<int64_t>(num_max_tokens_per_rank) * std::min(num_topk, num_local_experts);
            const int64_t unaligned_lane_route_capacity = raw_lane_route_capacity +
                static_cast<int64_t>(num_local_experts - 1) *
                    (streaming::kExpertAlignmentRows - 1);
            const int64_t lane_route_capacity = math::align(
                unaligned_lane_route_capacity,
                static_cast<int64_t>(streaming::kExpertAlignmentRows));
            const int64_t num_streaming_packed_tokens =
                nccl_context->num_ranks * lane_route_capacity;
            EP_HOST_ASSERT(
                num_streaming_packed_tokens <= std::numeric_limits<int>::max());
            streaming_packed_x = torch::empty(
                {num_streaming_packed_tokens, hidden}, x.options());

            int streaming_sf_lane_stride = 0;
            int streaming_sf_token_stride = 0, streaming_sf_hidden_stride = 0;
            if (sf.has_value()) {
                if (not use_tma_aligned_col_major_sf) {
                    streaming_sf_token_stride = num_sf_packs;
                    streaming_sf_hidden_stride = 1;
                    streaming_sf_lane_stride =
                        lane_route_capacity * num_sf_packs;
                } else {
                    streaming_sf_token_stride = 1;
                    streaming_sf_hidden_stride = math::align(
                        static_cast<int>(lane_route_capacity), kNumAlignedSFPacks);
                    streaming_sf_lane_stride =
                        streaming_sf_hidden_stride * num_sf_packs;
                }
                streaming_packed_sf = torch::empty_strided(
                    {nccl_context->num_ranks, lane_route_capacity, num_sf_packs},
                    {streaming_sf_lane_stride, streaming_sf_token_stride,
                     streaming_sf_hidden_stride},
                    sf->options());
            }
            if (topk_weights.has_value()) {
                streaming_packed_topk_weights = torch::empty(
                    {num_streaming_packed_tokens}, topk_weights->options());
            }
            streaming_packed_src_metadata = torch::empty(
                {static_cast<int64_t>(nccl_context->num_ranks) * num_max_tokens_per_rank,
                 num_topk + 2},
                torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            // Generation-owned downstream control. This must not alias the
            // single-slot workspace because ingress may be acknowledged as
            // soon as packing completes.
            streaming_packed_lane_control = torch::empty(
                {nccl_context->num_ranks, 2 + num_local_experts},
                torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

            // Small dispatches do not amortize a wide CTA, while full DSV4
            // prefill chunks are limited by the source-lane copy. The local
            // token count is available without reading remote metadata and is
            // a stable proxy for selecting the copy width on every rank.
            const int configured_streaming_copy_warps =
                get_env<int>("EP_STREAMING_DISPATCH_COPY_WARPS", 0);
            const int streaming_copy_warp_min_tokens =
                get_env<int>("EP_STREAMING_DISPATCH_COPY_WARP_MIN_TOKENS", 1152);
            EP_HOST_ASSERT(configured_streaming_copy_warps >= 0 and
                           streaming_copy_warp_min_tokens > 0);
            const int streaming_copy_warps = configured_streaming_copy_warps != 0
                ? configured_streaming_copy_warps
                : (num_tokens >= streaming_copy_warp_min_tokens ? 16 : 2);

            launch_dispatch_streaming_copy(
                buffer, workspace,
                streaming_packed_x->mutable_data_ptr(),
                streaming_packed_sf.has_value() ? streaming_packed_sf->mutable_data_ptr() : nullptr,
                get_data_ptr<float>(streaming_packed_topk_weights),
                streaming_packed_src_metadata->data_ptr<int>(),
                streaming_packed_lane_control->data_ptr<int>(),
                streaming_sf_lane_stride,
                streaming_sf_token_stride, streaming_sf_hidden_stride,
                nccl_context->scaleup_rank_idx,
                current_streaming_generation,
                nccl_context->num_ranks,
                num_hidden_bytes, num_sf_packs,
                num_max_tokens_per_rank,
                num_experts, num_topk,
                jit::device_runtime->get_num_smem_bytes(),
                num_gpu_timeout_cycles,
                streaming_copy_warps,
                streaming_lane_copy_streams);

            // Every source stream touches a disjoint slice of every packed
            // allocation. Record all producers so the allocator cannot reuse
            // storage until the slowest source kernel has retired.
            for (const auto& source_stream : streaming_lane_copy_streams) {
                streaming_packed_x->record_stream(source_stream);
                if (streaming_packed_sf.has_value())
                    streaming_packed_sf->record_stream(source_stream);
                if (streaming_packed_topk_weights.has_value())
                    streaming_packed_topk_weights->record_stream(source_stream);
                streaming_packed_src_metadata->record_stream(source_stream);
                streaming_packed_lane_control->record_stream(source_stream);
            }

            if (export_streaming_lanes) {
                latest_streaming_packed_x = streaming_packed_x;
                latest_streaming_packed_sf = streaming_packed_sf;
                latest_streaming_packed_topk_weights = streaming_packed_topk_weights;
                latest_streaming_packed_src_metadata = streaming_packed_src_metadata;
                latest_streaming_packed_lane_control = streaming_packed_lane_control;
                latest_streaming_generation = current_streaming_generation;
                latest_streaming_num_ranks = nccl_context->num_ranks;
                latest_streaming_num_local_experts = num_local_experts;
                latest_streaming_lane_route_capacity = lane_route_capacity;
                streaming_lane_return_submitted.assign(
                    nccl_context->num_ranks, false);
                streaming_lane_ingress_released.assign(
                    nccl_context->num_ranks, false);
                streaming_lane_release_events.clear();
                streaming_lane_release_events.resize(nccl_context->num_ranks);
                streaming_lane_return_events.clear();
                streaming_lane_return_events.resize(nccl_context->num_ranks);
                streaming_lane_view_outstanding = true;
            }
        }

        // The rank-ready path owns separate counters: dispatch notify warps
        // may still be blocked in the legacy all-rank count exchange while
        // the destination-local prefix kernel consumes lane-local counts.
        auto bulk_psum_num_recv_tokens_per_scaleup_rank = std::optional<torch::Tensor>();
        auto bulk_psum_num_recv_tokens_per_expert = std::optional<torch::Tensor>();
        auto bulk_num_unaligned_recv_tokens_per_expert = std::optional<torch::Tensor>();
        if (run_rank_ready) {
            bulk_psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank;
            bulk_psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert;
            bulk_num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert;
            psum_num_recv_tokens_per_scaleup_rank = torch::empty(
                {nccl_context->num_scaleup_ranks},
                at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            psum_num_recv_tokens_per_expert = torch::empty(
                {num_local_experts},
                at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            num_unaligned_recv_tokens_per_expert = torch::empty(
                {num_local_experts},
                at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            num_unaligned_recv_tokens_per_expert_ptr =
                num_unaligned_recv_tokens_per_expert.data_ptr<int>();
        } else if (not cached_mode) {
            if (do_expand) {
                // Slice the exclusive part and do atomic additions into inclusive
                psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
            } else {
                // Slice the inclusive part (and will not be used in the epilogue)
                psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
            }
        }
        EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) == num_local_experts);

        // Legacy and rank-ready modes still materialize the ordinary merged
        // receive layout. The exported lane path consumes its independently
        // released sidecar and must not queue this unused bulk epilogue.
        const auto completion_stream = run_rank_ready ? streaming_copy_stream : comm_stream;
        if (not export_streaming_lanes) {
            if (previous_event_before_epilogue.has_value())
                stream_wait(completion_stream, previous_event_before_epilogue.value());
            if (run_rank_ready) {
                launch_dispatch_rank_ready_prefix(
                    workspace,
                    psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                    psum_num_recv_tokens_per_expert.data_ptr<int>(),
                    num_unaligned_recv_tokens_per_expert_ptr,
                    cumulative_local_expert_recv_stats_ptr,
                    nccl_context->scaleup_rank_idx,
                    current_streaming_generation,
                    nccl_context->num_ranks,
                    num_experts,
                    expert_alignment,
                    num_gpu_timeout_cycles,
                    completion_stream);
            }
            launch_dispatch_copy_epilogue(buffer, workspace,
                                          psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                                          psum_num_recv_tokens_per_expert.data_ptr<int>(),
                                          recv_x.data_ptr(), recv_sf_ptr,
                                          recv_topk_idx_ptr, recv_topk_weights_ptr,
                                          recv_src_metadata.data_ptr<int>(),
                                          channel_linked_list_ptr,
                                          num_unaligned_recv_tokens_per_expert_ptr,
                                          num_recv_tokens, num_max_tokens_per_rank,
                                          num_hidden_bytes,
                                          num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,
                                          num_experts, num_topk, expert_alignment,
                                          nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                                          nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                          jit::device_runtime->get_num_sms(),
                                          jit::device_runtime->get_num_smem_bytes(),
                                          num_channels,
                                          do_expand, cached_mode,
                                          do_zero_padding,
                                          not run_rank_ready,
                                          completion_stream);
            if (instrument_streaming_bulk or run_signal_only or run_rank_ready) {
                // These modes consume the ordinary merged layout instead of
                // exporting a lane view. Acknowledge the generation after the
                // merged epilogue so a later streaming arm can safely reuse
                // the same single-slot ingress workspace.
                launch_dispatch_streaming_ack(
                    nccl_context->dev_comm, nccl_context->window, workspace,
                    nccl_context->scaleup_rank_idx,
                    current_streaming_generation,
                    nccl_context->num_ranks, num_experts,
                    completion_stream);
            } else if (run_streaming_shadow) {
                // Shadow mode has two local readers: the ordinary merged
                // epilogue and every sidecar source copy. Aggregate them on
                // the original auxiliary stream before publishing the
                // destination acknowledgement used by the next epoch.
                stream_wait(streaming_copy_stream, completion_stream);
                for (const auto& source_stream : streaming_lane_copy_streams)
                    stream_wait(streaming_copy_stream, source_stream);
                launch_dispatch_streaming_ack(
                    nccl_context->dev_comm, nccl_context->window, workspace,
                    nccl_context->scaleup_rank_idx,
                    current_streaming_generation,
                    nccl_context->num_ranks, num_experts,
                    streaming_copy_stream);
            }
        }

        // Stream control
        const auto event = stream_control_epilogue_on_stream(
            {x, sf, topk_idx, topk_weights,
             recv_x, recv_sf, recv_topk_idx, recv_topk_weights,
             cumulative_local_expert_recv_stats,
             copied_topk_idx,
             psum_num_recv_tokens_per_scaleup_rank,
             psum_num_recv_tokens_per_expert,
             num_unaligned_recv_tokens_per_expert,
             recv_src_metadata,
             dst_buffer_slot_idx,
             token_metadata_at_forward,
             channel_linked_list,
             bulk_psum_num_recv_tokens_per_scaleup_rank,
             bulk_psum_num_recv_tokens_per_expert,
             bulk_num_unaligned_recv_tokens_per_expert},
            compute_stream,
            completion_stream,
            run_rank_ready,
            allocate_on_comm_stream, async_with_compute_stream);

        return {recv_x, recv_sf,
                recv_topk_idx, recv_topk_weights,
                copied_topk_idx,
                num_recv_tokens, num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                token_metadata_at_forward,
                channel_linked_list,
                event};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const std::optional<torch::Tensor>& token_metadata_at_forward,
            const std::optional<torch::Tensor>& channel_linked_list,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms, const int& num_qps,
            const std::optional<EventHandle>& previous_event,
            const std::optional<EventHandle>& previous_event_before_epilogue,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout) const {
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);

        // Check data
        const auto [num_tokens, hidden] = get_shape<2>(x);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);

        // Check tensors at dispatch
        const auto [num_combined_tokens, num_topk] = get_shape<2>(combined_topk_idx);
        const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
        EP_HOST_ASSERT(combined_topk_idx.is_cuda() and combined_topk_idx.is_contiguous());
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);

        // Check metadata
        // For reduction mode, `num_tokens_` means the number of unexpanded tokens
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

        // Check optional tensors
        if (topk_weights.has_value()) {
            if (use_expanded_layout) {
                const auto [num_tokens__] = get_shape<1>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__);
            } else {
                const auto [num_tokens__, num_topk__] = get_shape<2>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);
            }
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
        }

        const auto bias_opts = std::vector({bias_0, bias_1});
        void* bias_ptrs[2] = {nullptr, nullptr};
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                bias_ptrs[i] = bias.data_ptr();
            }
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // Check buffer size
        EP_HOST_ASSERT(get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk,
                                               nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                               nccl_context->is_scaleup_nvlink, allow_multiple_reduction) <= num_buffer_bytes);

        // Optional configs and metadata for hybrid combine
        int num_channels = 1;
        int* token_metadata_at_forward_ptr = nullptr;
        int* channel_linked_list_ptr = nullptr;
        if (nccl_context->num_scaleout_ranks > 1) {
            // The token metadata during forward
            const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward.value());
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);
            num_channels = num_channels_;
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();
            EP_HOST_ASSERT(d1 == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
            EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
            EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);

            // Per-scaleup-peer-per-channel linked list
            const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
            EP_HOST_ASSERT(num_channels == num_channels__);
            EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
            EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
            EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
        }

        // Push data into remote buffers
        // NOTES: we don't use `num_hidden_bytes` due to enable later quantization possibility
        const bool bypass_combine_entry_barrier =
            get_env<int>("EP_EXPERIMENTAL_ASYNC_COMBINE_ENTRY") != 0;
        EP_HOST_ASSERT((not bypass_combine_entry_barrier or
                        (get_env<int>("EP_EXPERIMENTAL_RANK_READY") != 0 and
                         nccl_context->num_scaleout_ranks == 1 and
                         nccl_context->is_scaleup_nvlink)) and
                       "Asynchronous combine entry is inference-only and requires rank-ready dispatch in a single NVLink scale-up domain");
        const auto reduce_buffer = launch_combine(
            x.data_ptr(),
            topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            token_metadata_at_forward_ptr,
            channel_linked_list_ptr,
            nccl_context->dev_comm, nccl_context->window,
            buffer, workspace,
            num_reduced_tokens, num_max_tokens_per_rank,
            hidden, num_experts, num_topk,
            num_qps, num_gpu_timeout_cycles,
            nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
            nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
            nccl_context->is_scaleup_nvlink,
            num_sms, jit::device_runtime->get_num_smem_bytes(),
            num_channels,
            use_expanded_layout, allow_multiple_reduction,
            bypass_combine_entry_barrier,
            comm_stream);

        // Allocate output tensors
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        auto combined_topk_weights = std::optional<torch::Tensor>();
        float* combined_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
        }

        // Combine pushed data
        stream_control_before_epilogue(previous_event_before_epilogue);
        launch_combine_reduce_epilogue(combined_x.data_ptr(),
                                       combined_topk_weights_ptr,
                                       combined_topk_idx.data_ptr<topk_idx_t>(),
                                       num_combined_tokens, num_max_tokens_per_rank,
                                       hidden,
                                       num_experts, num_topk,
                                       reduce_buffer,
                                       bias_ptrs[0], bias_ptrs[1],
                                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                                       jit::device_runtime->get_num_sms(),
                                       jit::device_runtime->get_num_smem_bytes(),
                                       use_expanded_layout, allow_multiple_reduction,
                                       comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, topk_weights, bias_0, bias_1,
             src_metadata,
             combined_topk_idx,
             combined_x, combined_topk_weights,
             psum_num_recv_tokens_per_scaleup_rank,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);
        return {combined_x, combined_topk_weights, event};
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<int, int, int64_t, symmetric::cpu_comm_t, int64_t, int64_t, bool, bool, bool, int, int, int, int, bool>())
        .def("destroy", &ElasticBuffer::destroy)
        .def("get_comm_stream", &ElasticBuffer::get_comm_stream)
        .def("get_streaming_generation", &ElasticBuffer::get_streaming_generation)
        .def("get_streaming_lane_protocol_version", &ElasticBuffer::get_streaming_lane_protocol_version)
        .def("get_streaming_lane_view", &ElasticBuffer::get_streaming_lane_view)
        .def("release_streaming_lane", &ElasticBuffer::release_streaming_lane)
        .def("release_streaming_lane_view", &ElasticBuffer::release_streaming_lane_view)
        .def("streaming_combine_return", &ElasticBuffer::streaming_combine_return)
        .def("streaming_combine_return_many", &ElasticBuffer::streaming_combine_return_many)
        .def("streaming_combine_reduce", &ElasticBuffer::streaming_combine_reduce)
        .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)
        .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)
        .def("barrier", &ElasticBuffer::barrier)
        .def("engram_write", &ElasticBuffer::engram_write)
        .def("engram_fetch", &ElasticBuffer::engram_fetch)
        .def("pp_set_config", &ElasticBuffer::pp_set_config)
        .def("pp_send", &ElasticBuffer::pp_send)
        .def("pp_recv", &ElasticBuffer::pp_recv)
        .def("create_agrs_session", &ElasticBuffer::create_agrs_session)
        .def("destroy_agrs_session", &ElasticBuffer::destroy_agrs_session)
        .def("agrs_set_config", &ElasticBuffer::agrs_set_config)
        .def("agrs_get_inplace_tensor", &ElasticBuffer::agrs_get_inplace_tensor)
        .def("all_gather", &ElasticBuffer::all_gather)
        .def("dispatch", &ElasticBuffer::dispatch)
        .def("combine", &ElasticBuffer::combine);
    m.def("create_cpu_handle", &ElasticBuffer::create_cpu_handle);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
    m.def("get_elastic_buffer_alignment", [=]() {
        return symmetric::kNumAlignmentBytes;
    });

    // NCCL communicator handle
    m.def("get_local_nccl_unique_id", &nccl::get_local_unique_id);
    m.def("create_nccl_comm", &nccl::create_nccl_comm);
    m.def("destroy_nccl_comm", &nccl::destroy_nccl_comm);

    // Communication domain utilities
    m.def("get_physical_domain_size", &nccl::get_physical_domain_size);
    m.def("get_logical_domain_size", &nccl::get_logical_domain_size);
}

}  // namespace deep_ep
