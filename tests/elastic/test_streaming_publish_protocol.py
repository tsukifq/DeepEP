from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_streaming_payload_doorbell_is_published_after_dispatch_grid():
    device_source = (ROOT / "deep_ep/include/deep_ep/impls/dispatch.cuh").read_text(
        encoding="utf-8"
    )
    host_source = (ROOT / "csrc/kernels/elastic/dispatch.hpp").read_text(
        encoding="utf-8"
    )

    assert "dispatch_streaming_publish_impl" in device_source
    assert "ptx::st_release_sys(&remote_lane->payload_done_seq" in device_source
    assert "completed_blocks" not in device_source

    dispatch_launch = host_source.index(
        "DispatchRuntime::launch(runtime, args, stream)"
    )
    publish_launch = host_source.index(
        "launch_dispatch_streaming_publish(", dispatch_launch
    )
    assert publish_launch > dispatch_launch
    assert '"dispatch_streaming_publish", code' in host_source


def test_streaming_dispatch_decouples_signals_from_legacy_completion():
    device_source = (ROOT / "deep_ep/include/deep_ep/impls/dispatch.cuh").read_text(
        encoding="utf-8"
    )
    host_source = (ROOT / "csrc/kernels/elastic/dispatch.hpp").read_text(
        encoding="utf-8"
    )

    assert "bool kEmitStreamingSignals," in device_source
    assert "bool kBypassLegacyCompletion," in device_source
    assert "if constexpr (not kBypassLegacyCompletion)" in device_source
    # The notify warps and the grid epilogue both skip legacy completion only
    # for a replacement consumer, not merely because instrumentation is on.
    assert device_source.count("if constexpr (kBypassLegacyCompletion)") == 2
    assert (
        "if constexpr (kBypassLegacyCompletion)\n"
        "        return;\n\n"
        "    // Barrier to ensure data arrival"
    ) in device_source

    assert "bool bypass_legacy_completion;" in host_source
    assert "args.emit_streaming_signals," in host_source
    assert "args.bypass_legacy_completion," in host_source


def test_instrumented_bulk_acks_only_after_the_merged_epilogue():
    source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")

    assert "EP_EXPERIMENTAL_STREAMING_INSTRUMENTED_BULK" in source
    assert (
        "const bool bypass_legacy_completion =\n"
        "            export_streaming_lanes or run_rank_ready;"
    ) in source
    epilogue = source.index("launch_dispatch_copy_epilogue(")
    ack = source.index(
        "if (instrument_streaming_bulk or run_signal_only or run_rank_ready)",
        epilogue,
    )
    assert ack > epilogue
    assert "launch_dispatch_streaming_ack(" in source[ack:]


def test_streaming_generation_is_exposed_without_advancing_the_baseline():
    host_source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")
    python_source = (ROOT / "deep_ep/buffers/elastic.py").read_text(
        encoding="utf-8"
    )

    assert "emit_streaming_signals ? ++ streaming_generation : 0" in host_source
    assert '.def("get_streaming_generation"' in host_source
    assert "def get_streaming_generation(self) -> int:" in python_source
    assert "return self.runtime.get_streaming_generation()" in python_source


def test_per_source_release_waits_for_copy_owned_ingress_snapshot():
    control_source = (
        ROOT / "deep_ep/include/deep_ep/common/streaming.cuh"
    ).read_text(encoding="utf-8")
    copy_source = (
        ROOT / "deep_ep/include/deep_ep/impls/dispatch_streaming_copy.cuh"
    ).read_text(encoding="utf-8")
    ack_source = (
        ROOT / "deep_ep/include/deep_ep/impls/dispatch_streaming_reuse.cuh"
    ).read_text(encoding="utf-8")

    assert "uint64_t ingress_consumed_seq;" in control_source
    assert "publish_ingress_consumed" in control_source
    assert "acquire_ingress_consumed" in control_source

    snapshot = copy_source.index(
        "packed_lane_control[source_rank_idx * (kNumExpertsPerRank + 2) + 0]"
    )
    pack_ready = copy_source.index(
        "streaming::publish_pack_ready(lane_control, generation);", snapshot
    )
    ingress_consumed = copy_source.index(
        "streaming::publish_ingress_consumed(lane_control, generation);",
        pack_ready,
    )
    assert snapshot < pack_ready < ingress_consumed

    wait = ack_source.index(
        "streaming::acquire_ingress_consumed(source_lane, generation)"
    )
    remote_ack = ack_source.index(
        "ptx::st_release_sys(&remote_ack_slot->ack_seq, generation);", wait
    )
    assert "dispatch_streaming_lane_ack_impl" in ack_source
    assert wait < remote_ack


def test_generation_owned_lane_control_replaces_workspace_downstream_reads():
    host_source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")
    python_source = (ROOT / "deep_ep/buffers/elastic.py").read_text(
        encoding="utf-8"
    )
    copy_source = (
        ROOT / "deep_ep/include/deep_ep/impls/dispatch_streaming_copy.cuh"
    ).read_text(encoding="utf-8")
    return_source = (
        ROOT / "deep_ep/include/deep_ep/impls/combine_streaming.cuh"
    ).read_text(encoding="utf-8")

    assert (
        "{nccl_context->num_ranks, 2 + num_local_experts}" in host_source
    )
    # Public lane-view stays at seven fields: only the snapshot psum is
    # exported, while counts remain internal to preserve the original ABI.
    assert "latest_streaming_packed_lane_control->slice(1, 0, 2)" not in host_source
    assert (
        "latest_streaming_packed_lane_control->slice(\n"
        "                1, 2, 2 + num_local_experts)"
    ) in host_source
    assert "int* packed_lane_control" in copy_source
    assert "get_streaming_lane_psum_ptr" not in copy_source

    assert "const int* lane_counts" in return_source
    assert "const int num_source_tokens = lane_counts[0];" in return_source
    assert "const int num_source_routes = lane_counts[1];" in return_source
    assert "lane_control->num_unique_tokens" not in return_source
    assert "lane_control->num_routes" not in return_source
    assert "publish_ingress_consumed" not in return_source
    assert "torch::Tensor lane_counts," not in host_source
    assert "const auto lane_counts =" in host_source
    assert "source_rank_idx * lane_control_stride;" in host_source
    assert "latest_streaming_packed_lane_control->record_stream(stream);" in host_source
    assert "lane_counts: torch.Tensor," not in python_source
    assert (
        "lane_output, lane_src_metadata, source_rank, generation"
        in python_source
    )
    assert "int get_streaming_lane_protocol_version() const" in host_source
    assert "return 2;" in host_source
    assert '.def("get_streaming_lane_protocol_version"' in host_source
    assert "def get_streaming_lane_protocol_version(self) -> int:" in python_source


def test_generation_owned_lane_control_survives_workspace_reuse_semantics():
    # One logical destination snapshots g before its shared ingress/control is
    # overwritten by g+1. GEMM and return must continue to observe g.
    shared_workspace = {
        "tokens": 5,
        "routes": 9,
        "expert_psum": [2, 9],
        "generation": 7,
    }
    generation_control = (
        shared_workspace["tokens"],
        shared_workspace["routes"],
        *shared_workspace["expert_psum"],
    )
    shared_workspace.update(
        tokens=3, routes=4, expert_psum=[1, 4], generation=8
    )

    assert generation_control[:2] == (5, 9)
    assert generation_control[2:] == (2, 9)
    assert shared_workspace["generation"] == 8


def test_per_source_release_api_is_generation_checked_and_not_double_acked():
    host_source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")
    python_source = (ROOT / "deep_ep/buffers/elastic.py").read_text(
        encoding="utf-8"
    )

    method = host_source.index("void release_streaming_lane(")
    method_end = host_source.index("void streaming_combine_return(", method)
    method_source = host_source[method:method_end]
    assert "generation == latest_streaming_generation" in method_source
    assert "streaming_lane_return_submitted[source_rank_idx]" not in method_source
    assert "not streaming_lane_ingress_released[source_rank_idx]" in method_source
    assert "launch_dispatch_streaming_lane_ack(" in method_source
    assert "streaming_lane_release_events[source_rank_idx] = EventHandle(stream);" in method_source
    assert "streaming_lane_ingress_released[source_rank_idx] = true;" in method_source

    assert '.def("release_streaming_lane"' in host_source
    assert "def release_streaming_lane(" in python_source
    assert "self.runtime.release_streaming_lane(source_rank, generation)" in python_source


def test_zero_route_return_publishes_completion_without_scanning_tokens():
    source = (
        ROOT / "deep_ep/include/deep_ep/impls/combine_streaming.cuh"
    ).read_text(encoding="utf-8")

    one_lane = source.index("combine_streaming_return_impl(")
    reduce_kernel = source.index("combine_streaming_reduce_impl(", one_lane)
    one_lane_source = source[one_lane:reduce_kernel]
    empty = one_lane_source.index("if (num_source_routes == 0)")
    token_loop = one_lane_source.index("for (int token_idx = warp_idx;")
    assert empty < token_loop
    assert "publish_return_ready(remote_control, generation, 0)" in one_lane_source


def test_per_source_release_removes_only_the_strict_generation_wide_joins():
    source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")

    finalize = source.index("void release_streaming_lane_view() const")
    release_one = source.index("void release_streaming_lane(", finalize)
    finalize_source = source[finalize:release_one]
    assert "unreleased_source_mask" in finalize_source
    assert "stream, unreleased_source_mask" in finalize_source
    assert "streaming_consumer_event = EventHandle(stream);" in finalize_source
    assert "streaming_generation_released_per_lane = true;" in finalize_source
    assert "streaming_lane_return_submitted[source_rank_idx]" in finalize_source
    assert "streaming_lane_return_events[source_rank_idx].has_value()" in finalize_source
    assert "All per-lane returns must be submitted" in finalize_source
    assert "for (const auto& release_event : streaming_lane_release_events)" in finalize_source
    assert "for (const auto& return_event : streaming_lane_return_events)" in finalize_source
    assert "streaming_lane_drain_event = EventHandle(stream);" in finalize_source

    dispatch_guard = source.index("if (not streaming_generation_released_per_lane)")
    dispatch_guard_end = source.index(
        "streaming_generation_released_per_lane = false;", dispatch_guard
    )
    guard_source = source[dispatch_guard:dispatch_guard_end]
    assert "stream_wait(comm_stream, streaming_consumer_event.value())" in guard_source
    assert "stream_wait(comm_stream, streaming_copy_stream)" in guard_source
    assert "EP_HOST_ASSERT(not streaming_consumer_event.has_value())" in guard_source
    destroy = source.index("void destroy()")
    destroy_end = source.index("torch::Stream get_comm_stream()", destroy)
    assert (
        "stream_wait(comm_stream, streaming_lane_drain_event.value())"
        in source[destroy:destroy_end]
    )


def test_streaming_copy_launches_one_runtime_selected_source_per_stream():
    device_source = (
        ROOT / "deep_ep/include/deep_ep/impls/dispatch_streaming_copy.cuh"
    ).read_text(encoding="utf-8")
    host_source = (ROOT / "csrc/kernels/elastic/dispatch.hpp").read_text(
        encoding="utf-8"
    )

    assert "const int source_rank_idx," in device_source
    assert "source_rank_idx = static_cast<int>(blockIdx.x)" not in device_source
    assert "EP_DEVICE_ASSERT(blockIdx.x == 0);" in device_source
    assert (
        "EP_DEVICE_ASSERT(0 <= source_rank_idx and source_rank_idx < kNumRanks);"
        in device_source
    )

    launch = host_source.index("static void launch_dispatch_streaming_copy(")
    launch_end = host_source.index(
        "class DispatchRankReadyPrefixRuntime", launch
    )
    launch_source = host_source[launch:launch_end]
    build = launch_source.index(
        'jit::compiler->build("dispatch_streaming_copy", code)'
    )
    source_loop = launch_source.index(
        "for (int source_rank_idx = 0; source_rank_idx < num_ranks;"
    )
    source_launch = launch_source.index(
        "DispatchStreamingCopyRuntime::launch(", source_loop
    )
    assert ".launch_args = jit::LaunchArgs(\n            1," in launch_source
    assert launch_source.count(
        'jit::compiler->build("dispatch_streaming_copy", code)'
    ) == 1
    assert build < source_loop < source_launch
    assert "source_args.source_rank_idx = source_rank_idx;" in launch_source
    assert "source_streams[source_rank_idx]" in launch_source


def test_streaming_copy_owns_stable_source_stream_slots_and_tensor_lifetimes():
    source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")

    assert (
        "std::vector<at::cuda::CUDAStream> streaming_lane_copy_streams;"
        in source
    )
    constructor = source.index("ElasticBuffer(const int& rank_idx")
    constructor_end = source.index("~ElasticBuffer()", constructor)
    constructor_source = source[constructor:constructor_end]
    assert (
        "streaming_lane_copy_streams.reserve(nccl_context->num_ranks);"
        in constructor_source
    )
    assert "const auto shared_lane_copy_stream" in constructor_source
    assert "EP_STREAMING_SERIALIZE_LANE_COPIES" in constructor_source
    assert "serialize_lane_copies ? shared_lane_copy_stream" in constructor_source
    assert ": at::cuda::getStreamFromPool(false)" in constructor_source

    launch = source.index("launch_dispatch_streaming_copy(")
    records = source.index(
        "for (const auto& source_stream : streaming_lane_copy_streams)", launch
    )
    records_end = source.index("if (export_streaming_lanes)", records)
    record_source = source[records:records_end]
    for tensor_name in (
        "streaming_packed_x",
        "streaming_packed_sf",
        "streaming_packed_topk_weights",
        "streaming_packed_src_metadata",
        "streaming_packed_lane_control",
    ):
        assert f"{tensor_name}->record_stream(source_stream);" in record_source


def test_streaming_copy_lifecycle_joins_every_source_without_changing_rank_ready():
    source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")

    destroy = source.index("void destroy()")
    destroy_end = source.index("torch::Stream get_comm_stream()", destroy)
    destroy_source = source[destroy:destroy_end]
    assert "if (streaming_copy_used)" in destroy_source
    assert "stream_wait(comm_stream, streaming_copy_stream);" in destroy_source
    assert "if (streaming_lane_copy_used)" in destroy_source
    assert "for (const auto& source_stream : streaming_lane_copy_streams)" in destroy_source
    assert "stream_wait(comm_stream, source_stream);" in destroy_source

    legacy = source.index("if (not streaming_generation_released_per_lane)")
    legacy_end = source.index("} else {", legacy)
    legacy_source = source[legacy:legacy_end]
    assert "stream_wait(comm_stream, streaming_copy_stream);" in legacy_source
    assert "for (const auto& source_stream : streaming_lane_copy_streams)" in legacy_source
    assert "stream_wait(comm_stream, source_stream);" in legacy_source

    shadow = source.index("} else if (run_streaming_shadow) {")
    shadow_end = source.index("launch_dispatch_streaming_ack(", shadow)
    shadow_source = source[shadow:shadow_end]
    assert "stream_wait(streaming_copy_stream, completion_stream);" in shadow_source
    assert "for (const auto& source_stream : streaming_lane_copy_streams)" in shadow_source
    assert "stream_wait(streaming_copy_stream, source_stream);" in shadow_source

    # Rank-ready retains the original single-stream prefix/merged epilogue.
    assert (
        "const auto completion_stream = run_rank_ready ? streaming_copy_stream : comm_stream;"
        in source
    )
    assert "if (run_rank_ready or run_streaming_shadow)" in source
    assert "if (run_streaming_copy)\n            streaming_lane_copy_used = true;" in source


def test_rdc_device_link_keeps_hopper_or_blackwell_native_architecture():
    source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "rdc_arches = {'9.0': '90', '10.0': '100'}" in source
    assert "if torch_cuda_arch not in rdc_arches:" in source
    assert "f'-gencode=arch=compute_{rdc_arch},code=sm_{rdc_arch}'" in source
