/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <memory>
#include <shell/shared/platform/FrameDividerRateBackend.h>
#include <shell/shared/platform/android/PlatformAndroid.h>
#include <shell/shared/renderSession/IRenderSessionFactory.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/IGL.h>

namespace igl::samples {

class TinyRenderer final {
 public:
  /// Releases the native window reference this renderer still owns.
  ~TinyRenderer();

  /// The two display rates come from the Java display APIs, which are the only source for
  /// them on Android — there is no NDK query. `displayCurrentRefreshRateHz` is the mode the
  /// panel is in right now and is what every cadence is divided from;
  /// `displayMaxRefreshRateHz` is the highest mode it supports. Zero for either means the
  /// caller could not find out, and the seam then refuses rather than guessing.
  void init(AAssetManager* mgr,
            ANativeWindow* nativeWindow,
            shell::IRenderSessionFactory& factor,
            BackendVersion backendVersion,
            TextureFormat swapchainColorTextureFormat,
            ColorSpace swapchainColorSpace,
            const std::vector<std::string>& args = {},
            float displayCurrentRefreshRateHz = 0.0f,
            float displayMaxRefreshRateHz = 0.0f);

  /// Replaces what this renderer believes the display is doing and, unless asked not
  /// to, reinstalls the presentation-rate backend against it, which re-applies the last
  /// request at the new rate. Call it whenever the platform hands over a fresh reading —
  /// a repeat init() on a renderer that is being reused, or a surface change — because a
  /// divisor derived from a stale rate paces at the wrong cadence, and a first reading of
  /// zero would otherwise disable the seam for the renderer's whole life.
  ///
  /// Pass `reinstallBackend = false` when the caller reinstalls itself right after against
  /// a window this call does not own yet (surfaceChanged adopts inside onSurfacesChanged):
  /// reinstalling here would install against the outgoing window, against the
  /// adopt-before-reinstall contract below.
  void setDisplayRates(float currentHz, float maxHz, bool reinstallBackend = true);
  void recreateSwapchain(ANativeWindow* nativeWindow, bool createSurface); // only for Vulkan

  /// @brief Renders a frame
  /// @param displayScale The display scale factor
  /// @return true if the application should exit (e.g., benchmark timeout)
  bool render(float displayScale);

  void onSurfacesChanged(ANativeWindow* nativeWindow, int width, int height);

  /// Consumes one reference — the one `ANativeWindow_fromSurface()` just returned — and drops
  /// whichever was held before it, so a renderer owns exactly one reference to the window of
  /// the Surface currently driving it. Passing the window already held releases the caller's
  /// redundant reference rather than accumulating it, because that call acquires on every
  /// invocation and returns the same pointer for the same Surface.
  ///
  /// Call this before anything that reinstalls the presentation-rate backend. The install
  /// reads the window to ask the compositor for the panel's best mode, so reinstalling first
  /// asks on behalf of a Surface that is no longer the one being drawn to.
  void adoptNativeWindow(ANativeWindow* nativeWindow);

  /// Drops this renderer's window reference if, and only if, it is the one named. Called from
  /// the surface-destruction path of whichever thread owns the renderer, so the reference does
  /// not outlive the Surface it counts — these renderers live for the process, so nothing else
  /// would ever release it. Matching on the window rather than on the active backend is what
  /// keeps a backend switch from tearing down the wrong renderer's window.
  void releaseNativeWindowIfHeld(ANativeWindow* nativeWindow);

  void touchEvent(bool isDown, float x, float y, float dx, float dy);
  void setClearColorValue(float r, float g, float b, float a);

  [[nodiscard]] bool isHeadless() const noexcept {
    return shellParams_.isHeadless;
  }

  [[nodiscard]] bool isHdrDisplay() const;

  [[nodiscard]] ColorSpace requestedSwapchainColorSpace() const noexcept {
    return requestedSwapchainColorSpace_;
  }

  [[nodiscard]] const char* IGL_NULLABLE hdrDisplayUnsupportedReason() const noexcept {
    return hdrDisplayUnsupportedReason_;
  }

  [[nodiscard]] const BackendVersion& backendVersion() const noexcept {
    return backendVersion_;
  }

 private:
  BackendVersion backendVersion_;
  std::shared_ptr<igl::shell::PlatformAndroid> platform_;
  std::unique_ptr<igl::shell::RenderSession> session_;

  shell::ShellParams shellParams_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  TextureFormat swapchainColorTextureFormat_ = TextureFormat::RGBA_UNorm8;
  ANativeWindow* nativeWindow_ = nullptr;
  ColorSpace requestedSwapchainColorSpace_ = ColorSpace::SRGBNonlinear;
  const char* IGL_NULLABLE hdrDisplayUnsupportedReason_ = "renderer initialization failed";
  /// Shared with the installed backend's rates provider, so an update here is visible to a
  /// backend that is already running rather than only to the next one. Written and read on
  /// the shell's render thread, which is where every JNI entry point that touches it and
  /// every rate request both land.
  std::shared_ptr<igl::shell::DisplayRates> displayRates_ =
      std::make_shared<igl::shell::DisplayRates>();

  // Offscreen textures for headless rendering
  std::shared_ptr<ITexture> offscreenColorTexture_;
  std::shared_ptr<ITexture> offscreenDepthTexture_;

  // Multiview stereo present resources (Vulkan only, used with --force-multiview)
  std::shared_ptr<ITexture> multiviewColor_;
  std::shared_ptr<ITexture> multiviewDepth_;
  std::shared_ptr<ITexture> swapchainColor_;
  std::shared_ptr<ICommandQueue> presentQueue_;
  std::shared_ptr<IRenderPipelineState> presentPipeline_;
  std::shared_ptr<ISamplerState> presentSampler_;
#if IGL_BACKEND_VULKAN
  bool stereoPresentInitialized_ = false;
#endif

  void initStereoPresent(IDevice& device);
  void stereoPresent(IDevice& device);

  /// Rebuilds the Vulkan swapchain around the window already held. Split out of
  /// `recreateSwapchain()` so the resize path, which is re-using that window rather than
  /// being handed a new reference to it, does not have to fake one to satisfy the adopt.
  void rebuildSwapchain(bool createSurface);

  /// Points the platform's presentation-rate seam at whichever lever this backend has: an
  /// EGL swap-interval divider for GLES; the native window's frame rate for Vulkan, advisory
  /// only and with no divider, so capped rungs report unsupported rather than pretend.
  /// Re-run whenever the window is replaced, because the old backend was holding a rate on
  /// a window that is gone; installing a new one re-applies the last request against it.
  void installPresentationRateBackend();
};

} // namespace igl::samples
