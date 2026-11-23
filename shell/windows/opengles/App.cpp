/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <igl/Core.h>
#include <igl/IGL.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/PlatformDevice.h>
#include <igl/opengl/Version.h>
#if IGL_ANGLE
#include <shell/shared/platform/win/PlatformWin.h>
#include <igl/opengl/egl/Context.h>
#include <igl/opengl/egl/Device.h>
#include <igl/opengl/egl/PlatformDevice.h>
#endif // IGL_ANGLE
#include <GLFW/glfw3.h>
#include <memory>
#include <shell/windows/common/GlfwShell.h>
#include <shell/shared/platform/Platform.h>

using namespace igl;
namespace igl::shell {
namespace {
class OpenGlEsShell final : public GlfwShell {
  SurfaceTextures createSurfaceTextures() noexcept final;
  std::shared_ptr<Platform> createPlatform() noexcept final;

  void willCreateWindow() noexcept final;
  void didCreateWindow() noexcept final;

  void willTick() noexcept final;
};

class EGLDevice final : public ::igl::opengl::Device {
 public:
  // NOLINTNEXTLINE(clang-diagnostic-unused-member-function)
  explicit EGLDevice(std::unique_ptr<::igl::opengl::IContext> context) :
    Device(std::move(context)), platformDevice(*this) {
    {
    }
  }

  [[nodiscard]] const opengl::PlatformDevice& getPlatformDevice() const noexcept override {
    return platformDevice;
  }

  ::igl::opengl::PlatformDevice platformDevice;
};

void OpenGlEsShell::willCreateWindow() noexcept {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
  glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, sessionConfig().backendVersion.majorVersion);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, sessionConfig().backendVersion.minorVersion);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
}

void OpenGlEsShell::didCreateWindow() noexcept {
  [[maybe_unused]] int result = glfwGetWindowAttrib(window(), GLFW_CLIENT_API);

  glfwMakeContextCurrent(window());
  glfwSwapInterval(1);

  IGL_LOG_INFO("Renderer: %s\n", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
  IGL_LOG_INFO("Version: %s\n", reinterpret_cast<const char*>(glGetString(GL_VERSION)));
  IGL_LOG_INFO("WindowAttrib: 0x%x\n", result);
}

SurfaceTextures OpenGlEsShell::createSurfaceTextures() noexcept {
#if IGL_ANGLE
  auto& device = platform().getDevice();
  if (IGL_DEBUG_VERIFY(device.getBackendType() == igl::BackendType::OpenGL)) {
    auto platformDevice = device.getPlatformDevice<igl::opengl::egl::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice != nullptr);
    if (IGL_DEBUG_VERIFY(platformDevice)) {
      auto color = platformDevice->createTextureFromNativeDrawable(nullptr);
      auto depth =
          platformDevice->createTextureFromNativeDepth(igl::TextureFormat::Z_UNorm24, nullptr);
      return igl::SurfaceTextures{.color = std::move(color), .depth = std::move(depth)};
    }
  }
#endif // IGL_ANGLE
  return SurfaceTextures{};
}

std::shared_ptr<Platform> OpenGlEsShell::createPlatform() noexcept {
#if IGL_ANGLE
  auto glesDevice = std::make_unique</*EGLDevice*/ igl::opengl::egl::Device>(
      std::make_unique<::igl::opengl::egl::Context>(glfwGetEGLDisplay(),
                                                    glfwGetEGLContext(window()),
                                                    glfwGetEGLSurface(window()),
                                                    glfwGetEGLSurface(window())));

  return std::make_shared<igl::shell::PlatformWin>(std::move(glesDevice));
#endif // IGL_ANGLE
  return nullptr;
}

void OpenGlEsShell::willTick() noexcept {
  glfwMakeContextCurrent(window());
}

} // namespace
} // namespace igl::shell

int main(int argc, char* argv[]) {
  igl::shell::OpenGlEsShell shell;

  uint32_t majorVersion = 3;
  uint32_t minorVersion = 1;
  if (argc == 2) {
    std::tie(majorVersion, minorVersion) = igl::opengl::parseVersionString(argv[1]);
  }

  igl::shell::RenderSessionWindowConfig suggestedWindowConfig = {
      .width = 1024,
      .height = 768,
      .windowMode = shell::WindowMode::Window,
  };
  igl::shell::RenderSessionConfig suggestedConfig = {
      .displayName =
          "OpenGL ES " + std::to_string(majorVersion) + "." + std::to_string(minorVersion),
      .backendVersion = {.flavor = BackendFlavor::OpenGL_ES,
                         .majorVersion = static_cast<uint8_t>(majorVersion),
                         .minorVersion = static_cast<uint8_t>(minorVersion)},
      .swapchainColorTextureFormat = TextureFormat::RGBA_UNorm8,
  };

  if (!shell.initialize(argc, argv, suggestedWindowConfig, suggestedConfig)) {
    shell.teardown();
    return -1;
  }

  shell.run();
  shell.teardown();

  return 0;
}
