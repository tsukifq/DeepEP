#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic::streaming {

// SM90 DeepGEMM M-grouped psum layout aligns each expert segment start to
// 128 rows. Keep the allocator and pack kernel on one shared contract.
constexpr int kExpertAlignmentRows = 128;

enum class LaneState : uint32_t {
    kFree = 0,
    kReserved = 1,
    kDescReady = 2,
    kPayloadWriting = 3,
    kPayloadReady = 4,
    kPacking = 5,
    kGemmReady = 6,
    kGemmRunning = 7,
    kGemmDone = 8,
    kAcked = 9,
};

enum class LaneError : uint32_t {
    kNone = 0,
    kGenerationMismatch = 1,
    kCapacityOverflow = 2,
    kTimeout = 3,
};

// One cache line per source lane. Sequence fields carry the epoch generation,
// rather than a reusable boolean, so consumers cannot accept an ABA-stale
// notification after a ring slot wraps.
struct alignas(64) LaneControl {
    uint64_t generation;
    uint64_t payload_done_seq;
    uint64_t pack_done_seq;
    uint64_t gemm_done_seq;
    uint64_t ack_seq;

    uint32_t num_unique_tokens;
    uint32_t num_routes;
    uint32_t state;
    uint32_t error;
};

static_assert(sizeof(LaneControl) == 64, "LaneControl must occupy one cache line");
static_assert(alignof(LaneControl) == 64, "LaneControl must be cache-line aligned");

struct alignas(16) DispatchSync {
    uint32_t completed_blocks;
    uint32_t counts_ready;
    uint64_t reserved;
};

static_assert(sizeof(DispatchSync) == 16, "Invalid DispatchSync layout");

// Return/combine progress is separate from ingress LaneControl. A serving
// stream may leave the layer after source_done_seq, while the epoch allocator
// waits for every return ack before reusing symmetric storage.
enum class ReturnState : uint32_t {
    kWaitGemm = 0,
    kReturning = 1,
    kReturnReady = 2,
    kCombining = 3,
    kCombined = 4,
    kAcked = 5,
};

struct alignas(64) ReturnControl {
    uint64_t generation;
    uint64_t return_done_seq;
    uint64_t combine_done_seq;
    uint64_t ack_seq;

    uint32_t expected_routes;
    uint32_t completed_routes;
    uint32_t state;
    uint32_t error;

    uint64_t reserved[2];
};

static_assert(sizeof(ReturnControl) == 64, "ReturnControl must occupy one cache line");
static_assert(alignof(ReturnControl) == 64, "ReturnControl must be cache-line aligned");

struct alignas(64) LayerControl {
    uint64_t generation;
    uint64_t attention_done_seq;
    uint64_t source_done_seq;
    uint64_t drain_done_seq;

    uint32_t expected_returns;
    uint32_t completed_returns;
    uint32_t acknowledged_returns;
    uint32_t error;

    uint64_t reserved[2];
};

static_assert(sizeof(LayerControl) == 64, "LayerControl must occupy one cache line");
static_assert(alignof(LayerControl) == 64, "LayerControl must be cache-line aligned");

#ifdef __CUDACC__
__forceinline__ __device__ void publish_payload_ready(
    LaneControl* lane, const uint64_t& generation) {
    ptx::st_relaxed_sys(&lane->state, static_cast<uint32_t>(LaneState::kPayloadReady));
    ptx::st_release_sys(&lane->payload_done_seq, generation);
}

__forceinline__ __device__ bool acquire_payload_ready(
    const LaneControl* lane, const uint64_t& expected_generation) {
    return ptx::ld_acquire_sys(&lane->payload_done_seq) == expected_generation;
}

__forceinline__ __device__ void publish_pack_ready(
    LaneControl* lane, const uint64_t& generation) {
    ptx::st_relaxed_sys(&lane->state, static_cast<uint32_t>(LaneState::kGemmReady));
    ptx::st_release_sys(&lane->pack_done_seq, generation);
}

__forceinline__ __device__ bool acquire_pack_ready(
    const LaneControl* lane, const uint64_t& expected_generation) {
    return ptx::ld_acquire_sys(&lane->pack_done_seq) == expected_generation;
}

__forceinline__ __device__ void publish_gemm_done(
    LaneControl* lane, const uint64_t& generation) {
    ptx::st_relaxed_sys(&lane->state, static_cast<uint32_t>(LaneState::kGemmDone));
    ptx::st_release_sys(&lane->gemm_done_seq, generation);
}

__forceinline__ __device__ bool acquire_gemm_done(
    const LaneControl* lane, const uint64_t& expected_generation) {
    return ptx::ld_acquire_sys(&lane->gemm_done_seq) == expected_generation;
}

__forceinline__ __device__ void publish_ack(
    LaneControl* lane, const uint64_t& generation) {
    ptx::st_relaxed_sys(&lane->state, static_cast<uint32_t>(LaneState::kAcked));
    ptx::st_release_sys(&lane->ack_seq, generation);
}

__forceinline__ __device__ bool acquire_ack(
    const LaneControl* lane, const uint64_t& expected_generation) {
    return ptx::ld_acquire_sys(&lane->ack_seq) == expected_generation;
}

__forceinline__ __device__ void publish_return_ready(
    ReturnControl* control, const uint64_t& generation,
    const uint32_t& completed_routes) {
    ptx::st_relaxed_sys(&control->completed_routes, completed_routes);
    ptx::st_relaxed_sys(
        &control->state, static_cast<uint32_t>(ReturnState::kReturnReady));
    ptx::st_release_sys(&control->return_done_seq, generation);
}

__forceinline__ __device__ bool acquire_return_ready(
    const ReturnControl* control, const uint64_t& expected_generation) {
    return ptx::ld_acquire_sys(&control->return_done_seq) == expected_generation;
}

__forceinline__ __device__ void publish_combine_done(
    ReturnControl* control, const uint64_t& generation) {
    ptx::st_relaxed_sys(
        &control->state, static_cast<uint32_t>(ReturnState::kCombined));
    ptx::st_release_sys(&control->combine_done_seq, generation);
}

__forceinline__ __device__ void publish_source_done(
    LayerControl* control, const uint64_t& generation) {
    ptx::st_release_sys(&control->source_done_seq, generation);
}

__forceinline__ __device__ bool acquire_source_done(
    const LayerControl* control, const uint64_t& expected_generation) {
    return ptx::ld_acquire_sys(&control->source_done_seq) == expected_generation;
}

__forceinline__ __device__ void publish_return_ack(
    ReturnControl* control, const uint64_t& generation) {
    ptx::st_relaxed_sys(
        &control->state, static_cast<uint32_t>(ReturnState::kAcked));
    ptx::st_release_sys(&control->ack_seq, generation);
}

__forceinline__ __device__ bool acquire_return_ack(
    const ReturnControl* control, const uint64_t& expected_generation) {
    return ptx::ld_acquire_sys(&control->ack_seq) == expected_generation;
}
#endif

}  // namespace deep_ep::elastic::streaming
