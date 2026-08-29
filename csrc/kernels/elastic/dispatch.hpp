#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class DispatchRuntime final : public jit::LaunchRuntime<DispatchRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        bool do_cpu_sync;
        bool reuse_slot_indices;
        bool emit_streaming_signals;
        bool bypass_legacy_completion;
        int num_notify_warps;
        int num_dispatch_warps; // For hybrid dispatch
        int num_scaleout_warps, num_forward_warps; // For direct dispatch
        int num_scaleout_ranks, num_scaleup_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts, num_topk, expert_alignment;
        int num_qps;
        int64_t num_timeout_cycles;

        // Parameters
        void* x; sf_pack_t* sf; topk_idx_t* topk_idx; float* topk_weights;
        topk_idx_t* copied_topk_idx;
        int* cumulative_local_expert_recv_stats;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        int* num_unaligned_recv_tokens_per_expert;
        int* dst_buffer_slot_idx;
        int* token_metadata_at_forward;
        int num_tokens;
        int sf_token_stride, sf_hidden_stride;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace; void* mapped_host_workspace;
        int scaleout_rank_idx, scaleup_rank_idx;
        uint64_t streaming_generation;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        std::string header_name, func_name;
        if (args.num_scaleout_ranks == 1) {
            header_name = "dispatch";
            func_name = fmt::format("dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                args.is_scaleup_nvlink,
                args.do_cpu_sync,
                args.reuse_slot_indices,
                args.emit_streaming_signals,
                args.bypass_legacy_completion,
                args.launch_args.grid_dim.first,
                args.num_notify_warps, args.num_dispatch_warps,
                args.num_scaleup_ranks,
                args.num_hidden_bytes, args.num_sf_packs,
                args.num_max_tokens_per_rank,
                args.num_experts, args.num_topk, args.expert_alignment,
                args.num_qps, args.num_timeout_cycles);
        } else {
            header_name = "hybrid_dispatch";
            func_name = fmt::format("hybrid_dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                args.do_cpu_sync,
                args.reuse_slot_indices,
                args.launch_args.grid_dim.first,
                args.num_notify_warps, args.num_scaleout_warps, args.num_forward_warps,
                args.num_scaleout_ranks, args.num_scaleup_ranks,
                args.num_hidden_bytes, args.num_sf_packs,
                args.num_max_tokens_per_rank,
                args.num_experts, args.num_topk, args.expert_alignment,
                args.num_qps, args.num_timeout_cycles);
        }

        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.num_scaleout_ranks == 1) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,
                args.copied_topk_idx,
                args.cumulative_local_expert_recv_stats,
                args.psum_num_recv_tokens_per_scaleup_rank,
                args.psum_num_recv_tokens_per_expert,
                args.num_unaligned_recv_tokens_per_expert,
                args.dst_buffer_slot_idx,
                args.num_tokens,
                args.sf_token_stride, args.sf_hidden_stride,
                args.nccl_dev_comm, args.nccl_window,
                args.buffer,
                args.workspace, args.mapped_host_workspace,
                args.scaleup_rank_idx,
                args.streaming_generation));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,
                args.copied_topk_idx,
                args.cumulative_local_expert_recv_stats,
                args.psum_num_recv_tokens_per_scaleup_rank,
                args.psum_num_recv_tokens_per_expert,
                args.num_unaligned_recv_tokens_per_expert,
                args.dst_buffer_slot_idx,
                args.token_metadata_at_forward,
                args.num_tokens,
                args.sf_token_stride, args.sf_hidden_stride,
                args.nccl_dev_comm, args.nccl_window,
                args.buffer,
                args.workspace, args.mapped_host_workspace,
                args.scaleout_rank_idx, args.scaleup_rank_idx
            ));
        }
    }
};

class DispatchStreamingPublishRuntime final
    : public jit::LaunchRuntime<DispatchStreamingPublishRuntime> {
public:
    struct Args {
        int num_ranks, num_experts;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* workspace;
        int source_rank_idx;
        uint64_t generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_streaming_publish_impl<{}, {}>);
}}
)", args.num_ranks, args.num_experts);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window, args.workspace,
            args.source_rank_idx, args.generation));
    }
};

static void launch_dispatch_streaming_publish(
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* workspace,
    const int& source_rank_idx,
    const uint64_t& generation,
    const int& num_ranks,
    const int& num_experts,
    const at::cuda::CUDAStream& stream) {
    const DispatchStreamingPublishRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_experts = num_experts,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .workspace = workspace,
        .source_rank_idx = source_rank_idx,
        .generation = generation,
        .launch_args = jit::LaunchArgs(1, 32, 0)};
    const auto code = DispatchStreamingPublishRuntime::generate(args);
    const auto runtime = jit::compiler->build(
        "dispatch_streaming_publish", code);
    DispatchStreamingPublishRuntime::launch(runtime, args, stream);
}

class DispatchStreamingAckRuntime final
    : public jit::LaunchRuntime<DispatchStreamingAckRuntime> {
public:
    struct Args {
        int num_ranks, num_experts;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* workspace;
        int destination_rank_idx;
        uint64_t generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_streaming_reuse.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_streaming_ack_impl<{}, {}>);
}}
)", args.num_ranks, args.num_experts);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window, args.workspace,
            args.destination_rank_idx, args.generation));
    }
};

static void launch_dispatch_streaming_ack(
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* workspace,
    const int& destination_rank_idx,
    const uint64_t& generation,
    const int& num_ranks,
    const int& num_experts,
    const at::cuda::CUDAStream& stream) {
    const DispatchStreamingAckRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_experts = num_experts,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .workspace = workspace,
        .destination_rank_idx = destination_rank_idx,
        .generation = generation,
        .launch_args = jit::LaunchArgs(1, 32, 0)};
    const auto runtime = jit::compiler->build(
        "dispatch_streaming_ack",
        DispatchStreamingAckRuntime::generate(args));
    DispatchStreamingAckRuntime::launch(runtime, args, stream);
}

class DispatchStreamingReuseWaitRuntime final
    : public jit::LaunchRuntime<DispatchStreamingReuseWaitRuntime> {
public:
    struct Args {
        int num_ranks, num_experts;
        int64_t num_timeout_cycles;
        void* workspace;
        int source_rank_idx;
        uint64_t previous_generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_streaming_reuse.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_streaming_reuse_wait_impl<{}, {}, {}>);
}}
)", args.num_ranks, args.num_experts, args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config, args.workspace,
            args.source_rank_idx, args.previous_generation));
    }
};

static void launch_dispatch_streaming_reuse_wait(
    void* workspace,
    const int& source_rank_idx,
    const uint64_t& previous_generation,
    const int& num_ranks,
    const int& num_experts,
    const int64_t& num_timeout_cycles,
    const at::cuda::CUDAStream& stream) {
    const DispatchStreamingReuseWaitRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_experts = num_experts,
        .num_timeout_cycles = num_timeout_cycles,
        .workspace = workspace,
        .source_rank_idx = source_rank_idx,
        .previous_generation = previous_generation,
        .launch_args = jit::LaunchArgs(1, 32, 0)};
    const auto runtime = jit::compiler->build(
        "dispatch_streaming_reuse_wait",
        DispatchStreamingReuseWaitRuntime::generate(args));
    DispatchStreamingReuseWaitRuntime::launch(runtime, args, stream);
}

constexpr int kNumNotifyWarps = 4;

static int get_num_notify_smem_bytes(const int& num_ranks, const int& num_experts) {
    return math::align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int);
}

static layout::TokenLayout get_dispatch_token_layout(
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
}

static void launch_dispatch(void* x, void* sf,
                            topk_idx_t* topk_idx, float* topk_weights,
                            topk_idx_t* copied_topk_idx,
                            int* cumulative_local_expert_recv_stats,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* psum_num_recv_tokens_per_expert,
                            int* num_unaligned_recv_tokens_per_expert,
                            int* dst_buffer_slot_idx,
                            int* token_metadata_at_forward,
                            const int& num_tokens, const int& num_max_tokens_per_rank,
                            const int& hidden, const int& elem_size,
                            const int& num_sf_packs, const int& sf_token_stride, const int& sf_hidden_stride,
                            const int& num_experts, const int& num_topk, const int& expert_alignment,
                            const jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer,
                            void* workspace, void* mapped_host_workspace,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_channels_per_sm,
                            const int& num_smem_bytes,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const bool& cached_mode,
                            const bool& emit_streaming_signals,
                            const bool& bypass_legacy_completion,
                            const uint64_t& streaming_generation,
                            const bool& do_cpu_sync,
                            const at::cuda::CUDAStream& stream) {
    // Cached mode does not support expert token counting
    if (cached_mode)
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats == nullptr);

    // Utils
    const auto num_ranks = num_scaleout_ranks * num_scaleup_ranks;

    // Notify warps
    // TODO: why don't we use 4 notify warps?
    const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps;
    const bool reuse_slot_indices = cached_mode;
    const int num_notify_smem_bytes = cached_mode ? 0 : get_num_notify_smem_bytes(num_ranks, num_experts);
    EP_HOST_ASSERT(num_notify_warps % 4 == 0);

    // Other warps
    int num_dispatch_warps = 0;
    int num_scaleout_warps = 0, num_forward_warps = 0;
    int num_threads = 0;

    // Maximize shared memory utilization
    if (num_scaleout_ranks == 1) {
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);
        num_dispatch_warps = std::min<int>(
            (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(), 32 - num_notify_warps);
        num_threads = (num_notify_warps + num_dispatch_warps) * 32;
    } else {
        // Hybrid kernels
        num_scaleout_warps = num_channels_per_sm;
        num_forward_warps = num_channels_per_sm;
        num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;
    }

    // Generate, build and launch
    const DispatchRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .do_cpu_sync = do_cpu_sync,
        .reuse_slot_indices = reuse_slot_indices,
        .emit_streaming_signals = emit_streaming_signals,
        .bypass_legacy_completion = bypass_legacy_completion,
        .num_notify_warps = num_notify_warps,
        .num_dispatch_warps = num_dispatch_warps,
        .num_scaleout_warps = num_scaleout_warps, .num_forward_warps = num_forward_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .num_hidden_bytes = hidden * elem_size, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk, .expert_alignment = expert_alignment,
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        .x = x, .sf = static_cast<sf_pack_t*>(sf), .topk_idx = topk_idx, .topk_weights = topk_weights,
        .copied_topk_idx = copied_topk_idx,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert,
        .dst_buffer_slot_idx = dst_buffer_slot_idx,
        .token_metadata_at_forward = token_metadata_at_forward,
        .num_tokens = num_tokens,
        .sf_token_stride = sf_token_stride, .sf_hidden_stride = sf_hidden_stride,
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        .buffer = buffer,
        .workspace = workspace, .mapped_host_workspace = mapped_host_workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .streaming_generation = streaming_generation,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)};
    const auto code = DispatchRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch", code);
    DispatchRuntime::launch(runtime, args, stream);
    if (emit_streaming_signals) {
        EP_HOST_ASSERT(num_scaleout_ranks == 1 and is_scaleup_nvlink and
                       not cached_mode);
        launch_dispatch_streaming_publish(
            nccl_dev_comm, nccl_window, workspace,
            scaleup_rank_idx, streaming_generation,
            num_scaleup_ranks, num_experts, stream);
    }
}

class DispatchStreamingCopyRuntime final : public jit::LaunchRuntime<DispatchStreamingCopyRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_warps, num_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;
        int expert_alignment;
        int64_t num_timeout_cycles;

        // Parameters
        void *buffer, *workspace;
        void *packed_x, *packed_sf;
        float* packed_topk_weights;
        int* packed_src_metadata;
        int packed_sf_token_stride, packed_sf_hidden_stride;
        int destination_rank_idx;
        uint64_t generation;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_streaming_copy.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_streaming_copy_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.num_warps, args.num_ranks,
                           args.num_hidden_bytes, args.num_sf_packs,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk,
                           args.num_timeout_cycles, args.expert_alignment);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.buffer, args.workspace,
            args.packed_x, args.packed_sf,
            args.packed_topk_weights,
            args.packed_src_metadata,
            args.packed_sf_token_stride, args.packed_sf_hidden_stride,
            args.destination_rank_idx, args.generation));
    }
};

static void launch_dispatch_streaming_copy(
    void* buffer, void* workspace,
    void* packed_x, void* packed_sf,
    float* packed_topk_weights,
    int* packed_src_metadata,
    const int& packed_sf_token_stride, const int& packed_sf_hidden_stride,
    const int& destination_rank_idx,
    const uint64_t& generation,
    const int& num_ranks,
    const int& num_hidden_bytes, const int& num_sf_packs,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const int& num_smem_bytes,
    const int64_t& num_timeout_cycles,
    const at::cuda::CUDAStream& stream) {
    const auto token_layout = layout::TokenLayout(
        num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    const auto num_warps = std::min(
        num_smem_bytes / token_layout.get_num_bytes<true>(), 2);
    EP_HOST_ASSERT(num_warps > 0);
    const auto streaming_smem_bytes = num_warps * token_layout.get_num_bytes<true>();

    const DispatchStreamingCopyRuntime::Args args = {
        .num_warps = num_warps, .num_ranks = num_ranks,
        .num_hidden_bytes = num_hidden_bytes, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .expert_alignment = streaming::kExpertAlignmentRows,
        .num_timeout_cycles = num_timeout_cycles,
        .buffer = buffer, .workspace = workspace,
        .packed_x = packed_x, .packed_sf = packed_sf,
        .packed_topk_weights = packed_topk_weights,
        .packed_src_metadata = packed_src_metadata,
        .packed_sf_token_stride = packed_sf_token_stride,
        .packed_sf_hidden_stride = packed_sf_hidden_stride,
        .destination_rank_idx = destination_rank_idx,
        .generation = generation,
        .launch_args = jit::LaunchArgs(
            num_ranks, num_warps * 32, streaming_smem_bytes)};
    const auto code = DispatchStreamingCopyRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch_streaming_copy", code);
    DispatchStreamingCopyRuntime::launch(runtime, args, stream);
}

class DispatchRankReadyPrefixRuntime final : public jit::LaunchRuntime<DispatchRankReadyPrefixRuntime> {
public:
    struct Args {
        int num_ranks, num_experts, expert_alignment;
        int64_t num_timeout_cycles;
        void* workspace;
        int* psum_num_recv_tokens_per_rank;
        int* psum_num_recv_tokens_per_expert;
        int* num_unaligned_recv_tokens_per_expert;
        int* cumulative_local_expert_recv_stats;
        int destination_rank_idx;
        uint64_t generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_rank_ready.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_rank_ready_prefix_impl<{}, {}, {}, {}>);
}}
)",
                           args.num_ranks, args.num_experts,
                           args.expert_alignment, args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.workspace,
            args.psum_num_recv_tokens_per_rank,
            args.psum_num_recv_tokens_per_expert,
            args.num_unaligned_recv_tokens_per_expert,
            args.cumulative_local_expert_recv_stats,
            args.destination_rank_idx,
            args.generation));
    }
};

static void launch_dispatch_rank_ready_prefix(
    void* workspace,
    int* psum_num_recv_tokens_per_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* cumulative_local_expert_recv_stats,
    const int& destination_rank_idx,
    const uint64_t& generation,
    const int& num_ranks,
    const int& num_experts,
    const int& expert_alignment,
    const int64_t& num_timeout_cycles,
    const at::cuda::CUDAStream& stream) {
    const DispatchRankReadyPrefixRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_experts = num_experts,
        .expert_alignment = expert_alignment,
        .num_timeout_cycles = num_timeout_cycles,
        .workspace = workspace,
        .psum_num_recv_tokens_per_rank = psum_num_recv_tokens_per_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats,
        .destination_rank_idx = destination_rank_idx,
        .generation = generation,
        .launch_args = jit::LaunchArgs(1, 32, 0)};
    const auto code = DispatchRankReadyPrefixRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch_rank_ready_prefix", code);
    DispatchRankReadyPrefixRuntime::launch(runtime, args, stream);
}

class DispatchCopyEpilogueRuntime final : public jit::LaunchRuntime<DispatchCopyEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool wait_for_dispatch, do_expand, cached_mode, do_zero_padding;
        int num_channels;
        int num_warps;
        int num_scaleout_ranks, num_scaleup_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts, num_topk, expert_alignment;

        // Parameters
        void *buffer, *workspace;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        void* recv_x; void* recv_sf;
        topk_idx_t* recv_topk_idx; float* recv_topk_weights;
        int* recv_src_metadata;
        int* channel_linked_list;
        int* num_unaligned_recv_tokens_per_expert;
        int num_recv_tokens;
        int recv_sf_token_stride, recv_sf_hidden_stride;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_copy_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.wait_for_dispatch,
                           args.do_expand, args.cached_mode, args.do_zero_padding,
                           args.launch_args.grid_dim.first, args.num_channels, args.num_warps,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.num_hidden_bytes, args.num_sf_packs,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk, args.expert_alignment);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.buffer, args.workspace,
                                                 args.psum_num_recv_tokens_per_scaleup_rank,
                                                 args.psum_num_recv_tokens_per_expert,
                                                 args.recv_x, args.recv_sf, args.recv_topk_idx, args.recv_topk_weights,
                                                 args.recv_src_metadata,
                                                 args.channel_linked_list,
                                                 args.num_unaligned_recv_tokens_per_expert,
                                                 args.num_recv_tokens,
                                                 args.recv_sf_token_stride, args.recv_sf_hidden_stride,
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

static void launch_dispatch_copy_epilogue(void* buffer, void* workspace,
                                          int* psum_num_recv_tokens_per_scaleup_rank,
                                          int* psum_num_recv_tokens_per_expert,
                                          void* recv_x, void* recv_sf,
                                          topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                                          int* recv_src_metadata,
                                          int* channel_linked_list,
                                          int* num_unaligned_recv_tokens_per_expert,
                                          const int& num_recv_tokens, const int& num_max_tokens_per_rank,
                                          const int& num_hidden_bytes,
                                          const int& num_sf_packs, const int& recv_sf_token_stride, const int& recv_sf_hidden_stride,
                                          const int& num_experts, const int& num_topk, const int& expert_alignment,
                                          const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                          const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                          const int& num_sms, const int& num_smem_bytes,
                                          const int& num_channels,
                                          const bool& do_expand, const bool& cached_mode,
                                          const bool& do_zero_padding,
                                          const bool& wait_for_dispatch,
                                          const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    const auto token_layout = layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    const auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const DispatchCopyEpilogueRuntime::Args args = {
        .wait_for_dispatch = wait_for_dispatch,
        .do_expand = do_expand, .cached_mode = cached_mode, .do_zero_padding = do_zero_padding,
        .num_channels = num_channels, .num_warps = num_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .num_hidden_bytes = num_hidden_bytes, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk, .expert_alignment = expert_alignment,
        .buffer = buffer, .workspace = workspace,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .recv_x = recv_x, .recv_sf = recv_sf,
        .recv_topk_idx = recv_topk_idx, .recv_topk_weights = recv_topk_weights,
        .recv_src_metadata = recv_src_metadata,
        .channel_linked_list = channel_linked_list,
        .num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert,
        .num_recv_tokens = num_recv_tokens,
        .recv_sf_token_stride = recv_sf_token_stride, .recv_sf_hidden_stride = recv_sf_hidden_stride,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)};
    const auto code = DispatchCopyEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch_copy_epilogue", code);
    DispatchCopyEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
