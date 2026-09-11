#pragma once

#include <nccl.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class CombineRuntime final : public jit::LaunchRuntime<CombineRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        bool bypass_entry_barrier;
        bool use_expanded_layout, allow_multiple_reduction;
        int num_scaleup_warps, num_forward_warps;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts;
        int num_topk;
        int num_qps;
        int64_t num_timeout_cycles;

        // Parameters
        nv_bfloat16* x;
        float* topk_weights;
        int* src_metadata;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* token_metadata_at_forward;
        int* channel_linked_list;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;
        int scaleout_rank_idx, scaleup_rank_idx;
        int num_reduced_tokens;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        std::string header_name, func_name;
        if (args.num_scaleout_ranks == 1) {
            header_name = "combine";
            func_name = fmt::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.is_scaleup_nvlink,
                                    args.bypass_entry_barrier,
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.launch_args.num_threads / 32,
                                    args.num_scaleup_ranks * args.num_scaleout_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps, args.num_timeout_cycles);
        } else {
            EP_HOST_ASSERT(not args.bypass_entry_barrier and
                           "Asynchronous combine entry does not support hybrid mode");
            header_name = "hybrid_combine";
            func_name = fmt::format("hybrid_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.num_scaleup_warps, args.num_forward_warps,
                                    args.num_scaleout_ranks, args.num_scaleup_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps,
                                    args.num_timeout_cycles);
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
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata, args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleup_rank_idx,
                                                     args.num_reduced_tokens));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata,
                                                     args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.token_metadata_at_forward,
                                                     args.channel_linked_list,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleout_rank_idx, args.scaleup_rank_idx,
                                                     args.num_reduced_tokens));
        }
    }
};

static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            int* src_metadata,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* token_metadata_at_forward,
                            int* channel_linked_list,
                            const jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer, void* workspace,
                            const int& num_reduced_tokens, const int& num_max_tokens_per_rank,
                            const int& hidden,
                            const int& num_experts, const int& num_topk,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_smem_bytes,
                            const int& num_channels,
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                            const bool& bypass_entry_barrier,
                            const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);

    // Decide warps
    int num_scaleup_warps = 0, num_forward_warps = 0;
    if (num_scaleout_ranks > 1) {
        EP_HOST_ASSERT(num_channels % num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        EP_HOST_ASSERT(num_channels / num_sms <= 16);

        num_scaleup_warps = num_forward_warps = num_channels / num_sms;
        num_warps = num_scaleup_warps + num_forward_warps;
        EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                       "Invalid combine SM count, please try to match your dispatch config");
    }

    // Generate, build and launch
    const auto num_threads = num_warps * 32;
    const CombineRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .bypass_entry_barrier = bypass_entry_barrier,
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .num_scaleup_warps = num_scaleup_warps, .num_forward_warps = num_forward_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        .x = static_cast<nv_bfloat16*>(x),
        .topk_weights = static_cast<float*>(topk_weights),
        .src_metadata = src_metadata,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .token_metadata_at_forward = token_metadata_at_forward,
        .channel_linked_list = channel_linked_list,
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        .buffer = buffer, .workspace = workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .num_reduced_tokens = num_reduced_tokens,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)
    };
    const auto code = CombineRuntime::generate(args);
    const auto runtime = jit::compiler->build("combine", code);
    CombineRuntime::launch(runtime, args, stream);

    // Return the buffer to be reduced
    if (num_scaleout_ranks == 1)
        return buffer;

    // For hybrid mode, we have to skip the scale-up buffer
    const bool is_scaleup_buffer_rank_layout =
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, 
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
        num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    return scaleup_buffer.get_buffer_end_ptr();
}

class CombineReduceEpilogueRuntime final : public jit::LaunchRuntime<CombineReduceEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool use_expanded_layout, allow_multiple_reduction;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;

        // Parameters
        nv_bfloat16* combined_x;
        float* combined_topk_weights;
        topk_idx_t* combined_topk_idx;
        void* reduce_buffer;
        void* bias_0;
        void* bias_1;
        int num_combined_tokens;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/combine_reduce_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                        args.use_expanded_layout, args.allow_multiple_reduction,
                           args.launch_args.grid_dim.first,
                           args.launch_args.num_threads / 32,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.hidden,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.combined_x,
                                                 args.combined_topk_weights,
                                                 args.combined_topk_idx,
                                                 args.reduce_buffer,
                                                 args.bias_0, args.bias_1,
                                                 args.num_combined_tokens,
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights,
                                           topk_idx_t* combined_topk_idx,
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,
                                           const int& hidden,
                                           const int& num_experts, const int& num_topk,
                                           void* reduce_buffer,
                                           void* bias_0, void* bias_1,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                           const int& num_sms, const int& num_smem_bytes,
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                                           const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const CombineReduceEpilogueRuntime::Args args = {
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        .combined_topk_weights = combined_topk_weights,
        .combined_topk_idx = combined_topk_idx,
        .reduce_buffer = reduce_buffer,
        .bias_0 = bias_0, .bias_1 = bias_1,
        .num_combined_tokens = num_combined_tokens,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)
    };
    const auto code = CombineReduceEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("combine_reduce_epilogue", code);
    CombineReduceEpilogueRuntime::launch(runtime, args, stream);
}

class StreamingCombineReturnRuntime final :
    public jit::LaunchRuntime<StreamingCombineReturnRuntime> {
public:
    struct Args {
        bool merged_output_layout;
        bool apply_route_weights;
        int num_blocks, num_warps, num_ranks, hidden, num_max_tokens_per_rank;
        int num_experts, num_topk, num_qps;
        int64_t num_timeout_cycles;
        nv_bfloat16* lane_output;
        nv_bfloat16* merged_output;
        int* lane_to_merged;
        float* lane_route_weights;
        int* lane_src_metadata;
        const int* lane_counts;
        int lane_base_row, lane_capacity;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;
        int source_rank_idx, destination_rank_idx;
        uint64_t generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/combine_streaming.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_streaming_return_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
 )",         args.merged_output_layout,
             args.apply_route_weights,
             args.num_blocks, args.num_warps, args.num_ranks, args.hidden,
             args.num_max_tokens_per_rank, args.num_experts, args.num_topk,
             args.num_qps, args.num_timeout_cycles);
    }

    static void launch_impl(
        const jit::KernelHandle& kernel,
        const jit::LaunchConfigHandle& config,
        Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.lane_output, args.merged_output, args.lane_to_merged,
            args.lane_route_weights, args.lane_src_metadata, args.lane_counts,
            args.lane_base_row, args.lane_capacity,
            args.nccl_dev_comm, args.nccl_window,
            args.buffer, args.workspace,
            args.source_rank_idx, args.destination_rank_idx,
            args.generation));
    }
};

static void launch_streaming_combine_return(
    void* lane_output, int* lane_src_metadata, const int* lane_counts,
    void* merged_output, int* lane_to_merged,
    float* lane_route_weights,
    const bool& merged_output_layout,
    const bool& apply_route_weights,
    const int& lane_base_row, const int& lane_capacity,
    const jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
    void* buffer, void* workspace,
    const int& source_rank_idx, const int& destination_rank_idx,
    const uint64_t& generation,
    const int& num_ranks, const int& hidden,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk, const int& num_qps,
    const int64_t& num_timeout_cycles,
    const bool& return_all_sources,
    const at::cuda::CUDAStream& stream) {
    const int default_num_blocks =
        num_qps >= 4 * num_ranks ? 4 :
        (num_qps >= 2 * num_ranks ? 2 : 1);
    const int num_blocks = get_env<int>(
        "EP_STREAMING_COMBINE_RETURN_BLOCKS", default_num_blocks);
    EP_HOST_ASSERT(1 <= num_blocks and num_blocks <= 8);
    constexpr int kNumWarps = 4;
    const auto token_layout = layout::TokenLayout(
        hidden * sizeof(nv_bfloat16), 0, num_topk, false);
    const auto num_smem_bytes =
        kNumWarps * token_layout.get_num_bytes<true>();
    EP_HOST_ASSERT(not return_all_sources or source_rank_idx == -1);
    const int grid_blocks =
        num_blocks * (return_all_sources ? num_ranks : 1);
    const StreamingCombineReturnRuntime::Args args = {
        .merged_output_layout = merged_output_layout,
        .apply_route_weights = apply_route_weights,
        .num_blocks = num_blocks,
        .num_warps = kNumWarps,
        .num_ranks = num_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_qps = num_qps,
        .num_timeout_cycles = num_timeout_cycles,
        .lane_output = static_cast<nv_bfloat16*>(lane_output),
        .merged_output = static_cast<nv_bfloat16*>(merged_output),
        .lane_to_merged = lane_to_merged,
        .lane_route_weights = lane_route_weights,
        .lane_src_metadata = lane_src_metadata,
        .lane_counts = lane_counts,
        .lane_base_row = lane_base_row,
        .lane_capacity = lane_capacity,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .buffer = buffer,
        .workspace = workspace,
        .source_rank_idx = source_rank_idx,
        .destination_rank_idx = destination_rank_idx,
        .generation = generation,
        .launch_args = jit::LaunchArgs(
            grid_blocks, kNumWarps * 32, num_smem_bytes),
    };
    const auto runtime = jit::compiler->build(
        merged_output_layout
            ? (apply_route_weights
                   ? "streaming_combine_return_merged"
                   : "streaming_combine_return_merged_preweighted")
            : "streaming_combine_return",
        StreamingCombineReturnRuntime::generate(args));
    StreamingCombineReturnRuntime::launch(runtime, args, stream);
}

class StreamingCombineWaitReadyRuntime final :
    public jit::LaunchRuntime<StreamingCombineWaitReadyRuntime> {
public:
    struct Args {
        int num_ranks, num_experts;
        void* workspace;
        uint64_t generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/combine_streaming.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_streaming_wait_ready_impl<{}, {}>);
}}
)",          args.num_ranks, args.num_experts);
    }

    static void launch_impl(
        const jit::KernelHandle& kernel,
        const jit::LaunchConfigHandle& config,
        Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config, args.workspace, args.generation));
    }
};

static void launch_streaming_combine_wait_ready(
    void* workspace,
    const uint64_t& generation,
    const int& num_ranks,
    const int& num_experts,
    const at::cuda::CUDAStream& stream) {
    const StreamingCombineWaitReadyRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_experts = num_experts,
        .workspace = workspace,
        .generation = generation,
        .launch_args = jit::LaunchArgs(1, 32, 0),
    };
    const auto runtime = jit::compiler->build(
        "streaming_combine_wait_ready",
        StreamingCombineWaitReadyRuntime::generate(args));
    StreamingCombineWaitReadyRuntime::launch(runtime, args, stream);
}

class StreamingCombineReduceRuntime final :
    public jit::LaunchRuntime<StreamingCombineReduceRuntime> {
public:
    struct Args {
        bool wait_for_returns;
        int num_blocks, num_warps, num_ranks, hidden, num_max_tokens_per_rank;
        int num_experts, num_topk;
        nv_bfloat16* combined_x;
        topk_idx_t* combined_topk_idx;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;
        int source_rank_idx;
        int num_combined_tokens;
        uint64_t generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/combine_streaming.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_streaming_reduce_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",          args.wait_for_returns,
             args.num_blocks, args.num_warps, args.num_ranks, args.hidden,
             args.num_max_tokens_per_rank, args.num_experts, args.num_topk);
    }

    static void launch_impl(
        const jit::KernelHandle& kernel,
        const jit::LaunchConfigHandle& config,
        Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.combined_x, args.combined_topk_idx,
            args.nccl_dev_comm, args.nccl_window,
            args.buffer, args.workspace,
            args.source_rank_idx,
            args.num_combined_tokens, args.generation));
    }
};

static void launch_streaming_combine_reduce(
    void* combined_x, topk_idx_t* combined_topk_idx,
    const jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
    void* buffer, void* workspace,
    const int& source_rank_idx,
    const int& num_combined_tokens, const uint64_t& generation,
    const int& num_ranks, const int& hidden,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const at::cuda::CUDAStream& stream) {
    const bool sync_reduce =
        get_env<int>("EP_EXPERIMENTAL_STREAMING_SYNC_REDUCE", 0) != 0;
    const int num_blocks = get_env<int>(
        "EP_STREAMING_COMBINE_REDUCE_BLOCKS", 4);
    EP_HOST_ASSERT(1 <= num_blocks and num_blocks <= 16);
    constexpr int kNumWarps = 4;
    const auto output_layout = layout::TokenLayout(
        hidden * sizeof(nv_bfloat16), 0, 0, false);
    const auto num_smem_bytes =
        kNumWarps * output_layout.get_num_bytes<false>();
    if (sync_reduce) {
        launch_streaming_combine_wait_ready(
            workspace, generation, num_ranks, num_experts, stream);
    }

    const StreamingCombineReduceRuntime::Args args = {
        .wait_for_returns = not sync_reduce,
        .num_blocks = num_blocks,
        .num_warps = kNumWarps,
        .num_ranks = num_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        .combined_topk_idx = combined_topk_idx,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .buffer = buffer,
        .workspace = workspace,
        .source_rank_idx = source_rank_idx,
        .num_combined_tokens = num_combined_tokens,
        .generation = generation,
        .launch_args = jit::LaunchArgs(
            num_blocks, kNumWarps * 32, num_smem_bytes),
    };
    const auto runtime = jit::compiler->build(
        sync_reduce ? "streaming_combine_reduce_ready" :
                      "streaming_combine_reduce",
        StreamingCombineReduceRuntime::generate(args));
    StreamingCombineReduceRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
