/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/Config.h>

// clang-format off
#if IGL_PLATFORM_WINDOWS
  #include <igl/opengl/Device.h>
  #include <igl/opengl/wgl/Context.h>
#elif IGL_PLATFORM_LINUX
  #include <igl/opengl/Device.h>
  #include <igl/opengl/glx/Context.h>
#endif
// clang-format on

namespace igl::shell::util {

#if IGL_PLATFORM_WINDOWS
struct WGLDevice final : public opengl::Device {
  explicit WGLDevice(std::unique_ptr<opengl::IContext> context) :
    Device(std::move(context)), platformDevice_(*this) {}

  const opengl::PlatformDevice& getPlatformDevice() const noexcept override {
    return platformDevice_;
  }

  void endScope() override {
    Device::endScope();
    if (!verifyScope()) {
      wglMakeCurrent(nullptr, nullptr);
    }
  }

  opengl::PlatformDevice platformDevice_;
};
#elif IGL_PLATFORM_LINUX
struct GLXDevice final : public opengl::Device {
  explicit GLXDevice(std::unique_ptr<opengl::glx::Context> context) :
    Device(std::move(context)), platformDevice(*this) {}

  [[nodiscard]] const opengl::PlatformDevice& getPlatformDevice() const noexcept override {
    return platformDevice;
  }

  void endScope() override {
    Device::endScope();
    if (!verifyScope()) {
      getContext().clearCurrentContext();
    }
  }

  opengl::PlatformDevice platformDevice;
};
#endif
} // namespace igl::shell::util
