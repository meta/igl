/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <igl/webgpu/Common.h>

namespace igl::webgpu {

class WebGPUContext;

/// @brief A timestamp query set and its asynchronous readback, shared by ITimer and
/// ITimestampQueries.
///
/// A command buffer that writes queries resolves them before it is finished (encodeResolve()) and
/// maps the readback buffer once it is submitted (mapAfterSubmit()). Results arrive through
/// WebGPU's event processing; poll() processes events and reports whether a resolve has completed,
/// whose timestamps stay readable until a newer one completes. Every resolve uses its own readback
/// buffer, so a query set can be resolved again while an earlier readback is still pending.
///
/// Not thread-safe: like the rest of the device, it is used from one thread at a time, and the
/// getters process WebGPU events on the calling thread.
class TimestampQuerySet final {
 public:
  [[nodiscard]] static std::unique_ptr<TimestampQuerySet> create(WebGPUContext& ctx,
                                                                 uint32_t count,
                                                                 Result* IGL_NULLABLE outResult);
  ~TimestampQuerySet();

  TimestampQuerySet(const TimestampQuerySet&) = delete;
  TimestampQuerySet& operator=(const TimestampQuerySet&) = delete;
  TimestampQuerySet(TimestampQuerySet&&) = delete;
  TimestampQuerySet& operator=(TimestampQuerySet&&) = delete;

  [[nodiscard]] WGPUQuerySet IGL_NULLABLE getQuerySet() const noexcept {
    return querySet_.get();
  }
  [[nodiscard]] uint32_t getCount() const noexcept;

  /// Encodes resolving queries [0, count) into a new readback buffer.
  void encodeResolve(WGPUCommandEncoder IGL_NONNULL encoder, uint32_t count);
  /// Maps the readback buffer of the last encodeResolve(); call after its command buffer was
  /// submitted.
  void mapAfterSubmit();
  /// Starts a new cycle of queries; the newest completed result stays available until a newer one
  /// completes. If a new query set cannot be created, getQuerySet() is null for the cycle and all
  /// results are discarded.
  void restart();
  /// Starts a new cycle and discards pending and completed results.
  void reset();
  /// Processes WebGPU events; true once a mapped resolve has completed. A resolve whose mapping
  /// failed completes with no timestamps.
  [[nodiscard]] bool poll() const;
  /// Like poll(), but true only once every mapped resolve has completed, so the timestamps are
  /// those of the last mapped one.
  [[nodiscard]] bool pollAll() const;
  /// Timestamps in nanoseconds of the newest completed resolve; queries not written in its cycle
  /// are 0.
  [[nodiscard]] std::span<const uint64_t> getTimestamps() const;

 private:
  struct Readback;

  TimestampQuerySet(WebGPUContext& ctx, uint32_t count, Handle<WGPUQuerySet> querySet);
  void renewQuerySet();
  /// Makes the newest completed readback in mapped_ the result and drops it and older readbacks.
  void promoteCompleted() const;

  WebGPUContext& ctx_;
  const uint32_t count_;
  Handle<WGPUQuerySet> querySet_;
  Handle<WGPUBuffer> resolveBuffer_;
  std::shared_ptr<Readback> encoded_;
  /// Readbacks in flight, oldest first.
  mutable std::vector<std::shared_ptr<Readback>> mapped_;
  mutable std::shared_ptr<Readback> completed_;
};

[[nodiscard]] inline uint32_t TimestampQuerySet::getCount() const noexcept {
  return count_;
}

} // namespace igl::webgpu
