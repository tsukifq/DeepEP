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


def test_per_source_release_waits_for_the_return_kernels_final_ingress_read():
    control_source = (
        ROOT / "deep_ep/include/deep_ep/common/streaming.cuh"
    ).read_text(encoding="utf-8")
    return_source = (
        ROOT / "deep_ep/include/deep_ep/impls/combine_streaming.cuh"
    ).read_text(encoding="utf-8")
    ack_source = (
        ROOT / "deep_ep/include/deep_ep/impls/dispatch_streaming_reuse.cuh"
    ).read_text(encoding="utf-8")

    assert "uint64_t ingress_consumed_seq;" in control_source
    assert "publish_ingress_consumed" in control_source
    assert "acquire_ingress_consumed" in control_source

    count_read = return_source.index(
        "const int num_source_tokens = lane_control->num_unique_tokens;"
    )
    return_publish = return_source.index(
        "streaming::publish_return_ready", count_read
    )
    ingress_consumed = return_source.index(
        "streaming::publish_ingress_consumed(lane_control, generation);",
        return_publish,
    )
    assert count_read < return_publish < ingress_consumed

    wait = ack_source.index(
        "streaming::acquire_ingress_consumed(source_lane, generation)"
    )
    remote_ack = ack_source.index(
        "ptx::st_release_sys(&remote_ack_slot->ack_seq, generation);", wait
    )
    assert "dispatch_streaming_lane_ack_impl" in ack_source
    assert wait < remote_ack


def test_per_source_release_api_is_generation_checked_and_not_double_acked():
    host_source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")
    python_source = (ROOT / "deep_ep/buffers/elastic.py").read_text(
        encoding="utf-8"
    )

    method = host_source.index("void release_streaming_lane(")
    method_end = host_source.index("void streaming_combine_return(", method)
    method_source = host_source[method:method_end]
    assert "generation == latest_streaming_generation" in method_source
    assert "streaming_lane_return_submitted[source_rank_idx]" in method_source
    assert "not streaming_lane_ingress_released[source_rank_idx]" in method_source
    assert "launch_dispatch_streaming_lane_ack(" in method_source
    assert "streaming_lane_release_events[source_rank_idx] = EventHandle(stream);" in method_source
    assert "streaming_lane_ingress_released[source_rank_idx] = true;" in method_source

    assert '.def("release_streaming_lane"' in host_source
    assert "def release_streaming_lane(" in python_source
    assert "self.runtime.release_streaming_lane(source_rank, generation)" in python_source


def test_per_source_release_removes_only_the_strict_generation_wide_joins():
    source = (ROOT / "csrc/elastic/buffer.hpp").read_text(encoding="utf-8")

    finalize = source.index("void release_streaming_lane_view() const")
    release_one = source.index("void release_streaming_lane(", finalize)
    finalize_source = source[finalize:release_one]
    assert "unreleased_source_mask" in finalize_source
    assert "stream, unreleased_source_mask" in finalize_source
    assert "streaming_consumer_event = EventHandle(stream);" in finalize_source
    assert "streaming_generation_released_per_lane = true;" in finalize_source
    assert "for (const auto& release_event : streaming_lane_release_events)" in finalize_source
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


def test_sm90_rdc_device_link_keeps_the_target_architecture():
    source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "nvcc_dlink.extend(['-gencode=arch=compute_90,code=sm_90'])" in source
