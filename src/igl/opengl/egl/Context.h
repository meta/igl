/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define EGL_EGLEXT_PROTOTYPES

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <igl/opengl/IContext.h>

namespace igl::opengl::egl {

class Context final : public IContext {
 public:
  /// Creates a shared context, with matching format based on an existing context.
  static std::unique_ptr<Context> createShareContext(Context& existingContext,
                                                     EGLContext newContext,
                                                     EGLSurface readSurface,
                                                     EGLSurface drawSurface,
                                                     Result* outResult);

  /// Creates a shared context, matching format based on the current context.
  std::unique_ptr<IContext> createShareContext(Result* outResult) override;

  /// Create a new context for default display. This constructor makes the assumption that the EGL
  /// surfaces to be associated with this context are already present and set to current.
  explicit Context(EGLNativeWindowType window);
  explicit Context(BackendVersion backendVersion, EGLNativeWindowType window);
  Context& operator=(Context& other) = delete;
  /// Create a new offscreen context.
  Context(size_t width, size_t height);
  /// Create a new context applicable for a specific display/context/read surface/draw surface.
  /// @param ownsContext If true, this means that constructed Context owns the EGL context that is
  /// passed in and it will destroy the EGL context in its destructor. If false, it's the caller's
  /// responsibility to ensure the EGL context is destroyed.
  /// @param ownsSurfaces If true, this means that constructed Context owns the EGL surfaces that
  /// are passed in and it will destroy the EGL surfaces in its destructor. If false, it's the
  /// caller's responsibility to ensure the EGL surfaces are destroyed.
  Context(EGLDisplay display,
          EGLContext context,
          EGLSurface readSurface,
          EGLSurface drawSurface,
          EGLConfig config = nullptr,
          bool ownsContext = false,
          bool ownsSurfaces = false);
  /// Create a new offscreen context, in the same sharegroup as 'sharedContext'. Dimensions are
  /// also inferred from 'sharedContext'.
  Context(const Context& sharedContext);
  ~Context() override;

  void setCurrent() override;
  void clearCurrentContext() const override;
  bool isCurrentContext() const override;
  bool isCurrentSharegroup() const override;
  void present(std::shared_ptr<ITexture> surface) const override;

  void setPresentationTime(long long presentationTimeNs);
  void updateSurfaces(EGLSurface readSurface, EGLSurface drawSurface);
  void updateSurface(NativeWindowType window);

  EGLSurface createSurface(NativeWindowType window);

  EGLContext get() const;
  EGLDisplay getDisplay() const;
  EGLSurface getReadSurface() const;
  EGLSurface getDrawSurface() const;
  std::pair<EGLint, EGLint> getDrawSurfaceDimensions(Result* outResult) const;
  EGLConfig getConfig() const;

  /// Mark this context as belonging to a sharegroup with another context.
  void markSharegroup(Context& context);

  /// Add an external EGL context handle to this context's sharegroup
  /// so that isCurrentSharegroup() recognizes it.
  bool markSharegroup(EGLContext sharedContextHandle);

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  void imageTargetTexture(EGLImageKHR eglImage, GLenum target) const;
  EGLImageKHR createImageFromAndroidHardwareBuffer(AHardwareBuffer* hwb) const;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

  bool eglSupportssRGB() override {
    if (eglSupportssRGB_.has_value()) {
      return eglSupportssRGB_.value();
    }
    const char* extensionName = "EGL_KHR_gl_colorspace";
    // Get the list of supported EGL extensions
    const char* extensions = eglQueryString(getDisplay(), EGL_EXTENSIONS);
    IGL_LOG_DEBUG("eglQueryString: %s\n", extensions);
    const std::string strExtensions(extensions);
    if (strExtensions.find(extensionName) != std::string::npos) {
      eglSupportssRGB_.emplace(true);
      return eglSupportssRGB_.value();
    }
    eglSupportssRGB_.emplace(false);
    return eglSupportssRGB_.value();
  }

 private:
  std::optional<bool> eglSupportssRGB_;

 private:
  // Since EGLContext does not expose a Share Group, membership is recorded here: every context in
  // the group, including the one that owns this record. The record is shared by the contexts of a
  // group, grows from whichever thread creates or marks a member, and is read by GL calls on every
  // thread that has a member current, so it is synchronized.
  class Sharegroup final {
   public:
    void add(EGLContext context);
    [[nodiscard]] bool contains(EGLContext context) const;

   private:
    mutable std::mutex mutex_;
    std::vector<EGLContext> contexts_;
  };

  Context(BackendVersion backendVersion,
          EGLContext shareContext,
          std::shared_ptr<Sharegroup> sharegroup,
          bool offscreen,
          EGLNativeWindowType window,
          std::pair<EGLint, EGLint> dimensions);

  bool contextOwned_ = false;
  bool surfacesOwned_ = false;
  FOLLY_PUSH_WARNING
  FOLLY_GNU_DISABLE_WARNING("-Wzero-as-null-pointer-constant")
  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLContext context_ = EGL_NO_CONTEXT;
  EGLSurface surface_ = EGL_NO_SURFACE;
  EGLSurface readSurface_ = EGL_NO_SURFACE;
  EGLSurface drawSurface_ = EGL_NO_SURFACE;
  EGLConfig config_ = EGL_NO_CONFIG_KHR;
  FOLLY_POP_WARNING

  // Assigned before this context is reachable from another thread and never replaced afterwards;
  // only the record it points to changes. createShareContext() swaps the record its wrapping
  // constructor made for the existing context's before it returns the new context.
  std::shared_ptr<Sharegroup> sharegroup_;

  static constexpr BackendVersion kDefaultEGLBackendVersion = {
      .flavor = BackendFlavor::OpenGL_ES,
      .majorVersion = 2,
  };
  BackendVersion backendVersion_ = kDefaultEGLBackendVersion;
};

} // namespace igl::opengl::egl
