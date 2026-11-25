/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <shell/shared/platform/Platform.h>
#include <shell/shared/renderSession/BenchmarkTracker.h>

namespace igl::shell {
struct AppParams;
struct ShellParams;

class RenderSession {
 public:
  explicit RenderSession(std::shared_ptr<Platform> platform);
  virtual ~RenderSession() noexcept = default;

  // Enable move operations (unique_ptr requires explicit declaration)
  RenderSession(RenderSession&&) noexcept = default;
  RenderSession& operator=(RenderSession&&) noexcept = default;

  // Disable copy operations
  RenderSession(const RenderSession&) = delete;
  RenderSession& operator=(const RenderSession&) = delete;

  virtual void initialize() noexcept {}
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  virtual void update([[maybe_unused]] SurfaceTextures surfaceTextures) noexcept {}

  /// @brief Wrapper around update() that automatically handles benchmark timing
  /// Platform code should call this instead of update() directly when benchmark
  /// tracking is desired. This method:
  /// 1. Measures the time taken by update()
  /// 2. Records the frame time for benchmarking
  /// 3. Checks for periodic reporting
  /// 4. Checks for benchmark expiration and sets exitRequested if needed
  void runUpdate(SurfaceTextures surfaceTextures) noexcept;
  virtual void teardown() noexcept {}

  void updateDisplayScale(float scale) noexcept;

  [[nodiscard]] float pixelsPerPoint() const noexcept;
  void setPixelsPerPoint(float scale) noexcept;

  virtual void setShellParams(const ShellParams& shellParams) noexcept {
    shellParams_ = &shellParams;
  }

  /// @brief Params provided to the session by the host
  /// @remark Params may vary each frame.
  [[nodiscard]] const ShellParams& shellParams() const noexcept;

  /// @brief Params provided by the session to the host.
  /// @remark Params may vary each frame.
  [[nodiscard]] const AppParams& appParams() const noexcept;

  void setCurrentQuadLayer(size_t layer) noexcept {
    currentQuadLayer_ = layer;
  }

  /// return the number of seconds since the last call
  float getDeltaSeconds() noexcept;

  static double getSeconds() noexcept;

  virtual ICommandQueue* getCommandQueue() noexcept {
    return commandQueue_.get();
  }

  std::shared_ptr<IFramebuffer> getFramebuffer() noexcept {
    return framebuffer_;
  }

  /// @brief Called by the platform when the surface is resized or recreated. Sessions that keep
  /// surface-sized framebuffers in their own members override this to release them too.
  virtual void releaseFramebuffer() {
    framebuffer_ = nullptr;
  }

  void setPreferredClearColor(const Color& color) noexcept;
  Color getPreferredClearColor() noexcept;

  /// @brief Initializes the benchmark tracker based on shell params
  /// Call this after shellParams_ is set to enable benchmark tracking
  void initBenchmarkTracker() noexcept;

  /// @brief Records a frame's render time for benchmarking
  /// @param renderTimeMs The time in milliseconds for the update call
  void recordBenchmarkFrame(double renderTimeMs) noexcept;

  /// @brief Checks and handles periodic benchmark reporting
  /// Logs stats every minute if benchmark mode is enabled
  void checkBenchmarkPeriodicReport() noexcept;

  /// @brief Checks if the benchmark has expired and should trigger app exit
  /// @return true if benchmark duration exceeded
  [[nodiscard]] bool isBenchmarkExpired() const noexcept;

  /// @brief Generates and logs the final benchmark report
  /// @param wasTimeout true if the benchmark ended due to timeout
  void logFinalBenchmarkReport(bool wasTimeout) noexcept;

  /// @brief Gets the benchmark tracker (may be nullptr if not in benchmark mode)
  [[nodiscard]] BenchmarkTracker* getBenchmarkTracker() noexcept {
    return benchmarkTracker_.get();
  }

  /// @brief Returns mutable reference to AppParams (for benchmark to request exit)
  [[nodiscard]] AppParams& appParamsRef() noexcept;

 protected:
  Platform& getPlatform() noexcept;
  [[nodiscard]] const Platform& getPlatform() const noexcept;

  [[nodiscard]] const std::shared_ptr<Platform>& platform() const noexcept;

  /// @brief Structured submission tracing tag for the current frame, for backends that support
  /// submission-level tracing. Encodes a shell workflow id in the high 16 bits and the frame
  /// number (>= 1) in the low 48 bits. Pass into CommandBufferDesc::submitTag when creating a
  /// command buffer so driver-side trace events can be correlated back to the shell frame. A
  /// backend that does not support submission tracing ignores the tag.
  [[nodiscard]] uint64_t currentFrameSubmitTag() const noexcept;

  std::shared_ptr<IFramebuffer> framebuffer_;
  std::shared_ptr<ICommandQueue> commandQueue_;
  size_t currentQuadLayer_ = 0;
  double lastTime_ = getSeconds();

 private:
  std::shared_ptr<Platform> platform_;
  std::shared_ptr<AppParams> appParams_;
  std::optional<Color> preferredClearColor_;
  const ShellParams* shellParams_ = nullptr;
  std::unique_ptr<BenchmarkTracker> benchmarkTracker_;
  uint32_t frameCount_ = 0;
  bool benchmarkExpiredLogged_ = false;
  bool loggedMissingParams_ = false;
  bool frozen_ = false;
  double prevFrameStartTime_ = 0.0;
};

} // namespace igl::shell
