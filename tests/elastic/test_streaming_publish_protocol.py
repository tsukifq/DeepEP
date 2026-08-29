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


def test_sm90_rdc_device_link_keeps_the_target_architecture():
    source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "nvcc_dlink.extend(['-gencode=arch=compute_90,code=sm_90'])" in source
