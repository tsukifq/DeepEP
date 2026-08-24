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


def test_streaming_dispatch_keeps_the_legacy_barrier_out_of_specialization():
    source = (ROOT / "deep_ep/include/deep_ep/impls/dispatch.cuh").read_text(
        encoding="utf-8"
    )

    assert "if constexpr (not kEmitStreamingSignals)" in source
    assert "if constexpr (kEmitStreamingSignals)\n        return;" in source
