/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/windows/common/GlfwShell.h>

#include <cmath>
#include <cstdlib>
#include <thread>
#include <shell/shared/input/InputDispatcher.h>
#include <shell/shared/renderSession/AppParams.h>
#include <shell/shared/renderSession/DefaultRenderSessionFactory.h>
#include <shell/shared/renderSession/IRenderSessionFactory.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <shell/shared/renderSession/ScreenshotTestRenderSessionHelper.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/Framebuffer.h>

namespace igl::shell {
namespace {
void glfwErrorHandler(int error, const char* description) {
  IGL_LOG_ERROR("GLFW Error: %s\n", description);
}

MouseButton getIGLMouseButton(int button) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    return igl::shell::MouseButton::Left;
  }

  if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    return igl::shell::MouseButton::Right;
  }

  return igl::shell::MouseButton::Middle;
}
} // namespace

GlfwShell::GlfwShell() : window_(nullptr, &glfwDestroyWindow) {}

ShellParams& GlfwShell::shellParams() noexcept {
  return shellParams_;
}

const ShellParams& GlfwShell::shellParams() const noexcept {
  return shellParams_;
}

GLFWwindow* GlfwShell::window() noexcept {
  return window_.get();
}

const GLFWwindow* GlfwShell::window() const noexcept {
  return window_.get();
}

Platform& GlfwShell::platform() noexcept {
  return *platform_;
}

const Platform& GlfwShell::platform() const noexcept {
  return *platform_;
}

const RenderSessionWindowConfig& GlfwShell::windowConfig() const noexcept {
  return windowConfig_;
}

const RenderSessionConfig& GlfwShell::sessionConfig() const noexcept {
  return sessionConfig_;
}

bool GlfwShell::createWindow() noexcept {
  if (shellParams_.isHeadless) {
    return true;
  }

  glfwSetErrorCallback(glfwErrorHandler);

  if (!glfwInit()) {
    IGL_LOG_ERROR("glfwInit failed");
    return false;
  }

  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  glfwWindowHint(GLFW_DECORATED,
                 windowConfig_.windowMode == shell::WindowMode::Window ? GLFW_TRUE : GLFW_FALSE);

  int posX = 0;
  int posY = 0;
  int width = windowConfig_.width;
  int height = windowConfig_.height;

  GLFWmonitor* IGL_NULLABLE monitor = glfwGetPrimaryMonitor();

  if (windowConfig_.windowMode == WindowMode::Fullscreen) {
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    width = mode->width;
    height = mode->height;
  } else if (windowConfig_.windowMode == WindowMode::MaximizedWindow) {
    // render full screen without overlapping the task bar
    glfwGetMonitorWorkarea(monitor, &posX, &posY, &width, &height);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    monitor = nullptr;
  } else {
    monitor = nullptr;
  }

  willCreateWindow();

  GLFWwindow* windowHandle =
      // @fb-only
      glfwCreateWindow(width, height, sessionConfig_.displayName.c_str(), monitor, nullptr);
  if (!windowHandle) {
    return false;
  }
  window_.reset(windowHandle);

  glfwSetWindowUserPointer(windowHandle, this);
  if (windowConfig_.windowMode == WindowMode::MaximizedWindow) {
    glfwSetWindowPos(windowHandle, posX, posY);
  }

  glfwGetFramebufferSize(windowHandle, &width, &height);
  shellParams_.viewportSize.x = width;
  shellParams_.viewportSize.y = height;
  glfwSetCursorPosCallback(windowHandle, [](GLFWwindow* window, double xpos, double ypos) {
    auto* shell = static_cast<GlfwShell*>(glfwGetWindowUserPointer(window));
    shell->platform_->getInputDispatcher().queueEvent(
        MouseMotionEvent(static_cast<float>(xpos), static_cast<float>(ypos), 0, 0));
  });

  glfwSetScrollCallback(windowHandle, [](GLFWwindow* window, double xoffset, double yoffset) {
    auto* shell = static_cast<GlfwShell*>(glfwGetWindowUserPointer(window));
    shell->platform_->getInputDispatcher().queueEvent(
        MouseWheelEvent(static_cast<float>(xoffset), static_cast<float>(yoffset)));
  });

  glfwSetMouseButtonCallback(
      windowHandle, [](GLFWwindow* window, int button, int action, int mods) {
        auto* shell = static_cast<GlfwShell*>(glfwGetWindowUserPointer(window));
        MouseButton iglButton = getIGLMouseButton(button);
        double xpos = NAN, ypos = NAN;
        glfwGetCursorPos(window, &xpos, &ypos);
        shell->platform_->getInputDispatcher().queueEvent(MouseButtonEvent(
            iglButton, action == GLFW_PRESS, static_cast<float>(xpos), static_cast<float>(ypos)));
      });

  glfwSetKeyCallback(windowHandle, [](GLFWwindow* window, int key, int, int action, int mods) {
    auto* shell = static_cast<GlfwShell*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    uint32_t modifiers = 0;
    if (mods & GLFW_MOD_SHIFT) {
      modifiers |= kKeyEventModifierShift;
    }
    if (mods & GLFW_MOD_CONTROL) {
      modifiers |= kKeyEventModifierControl;
    }
    if (mods & GLFW_MOD_ALT) {
      modifiers |= kKeyEventModifierOption;
    }
    if (mods & GLFW_MOD_CAPS_LOCK) {
      modifiers |= kKeyEventModifierCapsLock;
    }
    if (mods & GLFW_MOD_NUM_LOCK) {
      modifiers |= kKeyEventModifierNumLock;
    }
    shell->platform_->getInputDispatcher().queueEvent(
        KeyEvent(action == GLFW_PRESS, key, modifiers));
    shell->platform_->getInputDispatcher().queueEvent(key <= 256 ? CharEvent{static_cast<char>(key)}
                                                                 : CharEvent{});
  });

  glfwSetCharCallback(windowHandle, [](GLFWwindow* window, unsigned int codepoint) {
    auto* shell = static_cast<GlfwShell*>(glfwGetWindowUserPointer(window));

    shell->platform_->getInputDispatcher().queueEvent(
        CharEvent{.character = static_cast<char>(codepoint)});
  });

  didCreateWindow();

  return windowHandle != nullptr;
}

bool GlfwShell::initialize(int argc,
                           char* argv[],
                           RenderSessionWindowConfig suggestedWindowConfig,
                           const RenderSessionConfig& suggestedSessionConfig) noexcept {
  igl::shell::Platform::initializeCommandLineArgs(argc, argv);

  // Use the comprehensive parser
  auto args = shell::convertArgvToParams(argc, argv);
  shell::parseShellParams(args, shellParams_);

  // Apply viewport size if specified
  suggestedWindowConfig.width = static_cast<int>(shellParams_.viewportSize.x);
  suggestedWindowConfig.height = static_cast<int>(shellParams_.viewportSize.y);
  suggestedWindowConfig.windowMode = WindowMode::Window;

  auto factory = igl::shell::createDefaultRenderSessionFactory();

  windowConfig_ =
      factory->requestedWindowConfig(igl::shell::ShellType::Windows, suggestedWindowConfig);
  const auto requestedConfigs =
      factory->requestedSessionConfigs(igl::shell::ShellType::Windows, {suggestedSessionConfig});
  if (IGL_DEBUG_VERIFY_NOT(requestedConfigs.size() != 1) ||
      IGL_DEBUG_VERIFY_NOT(suggestedSessionConfig.backendVersion.flavor !=
                           requestedConfigs[0].backendVersion.flavor)) {
    return false;
  }

  sessionConfig_ = requestedConfigs[0];

  if (!createWindow()) {
    return false;
  }

  platform_ = createPlatform();
  if (!platform_) {
    return false;
  }
  session_ = factory->createRenderSession(platform_);
  if (IGL_DEBUG_VERIFY_NOT(!session_)) {
    return false;
  }

  session_->setShellParams(shellParams_);
  session_->initialize();

  return true;
}

void GlfwShell::run() noexcept {
  uint64_t frameNumber = 0;
  bool frozen = false;
  while ((!window_ || !glfwWindowShouldClose(window_.get())) &&
         !session_->appParams().exitRequested) {
    if (frozen) {
      if (window_) {
        glfwPollEvents();
      }
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
      continue;
    }

    willTick();
    auto surfaceTextures = createSurfaceTextures();
    IGL_DEBUG_ASSERT(surfaceTextures.color != nullptr && surfaceTextures.depth != nullptr);

    std::shared_ptr<ITexture> colorTexture = surfaceTextures.color;

    platform_->getInputDispatcher().processEvents();

    const auto& params = session_->shellParams();
    if (params.freezeAtFrame != ~0u && frameNumber >= params.freezeAtFrame) {
      frozen = true;
      IGL_LOG_INFO("[IGL Shell] Frozen at frame %u\n", params.freezeAtFrame);
      continue;
    }

    const double startTime = RenderSession::getSeconds();
    session_->update(std::move(surfaceTextures));
    postUpdate();

    if (params.fpsThrottleMs > 0) {
      const double endTime = RenderSession::getSeconds();
      const double frameTimeMs = (endTime - startTime) * 1000.0;
      const double targetMs = params.fpsThrottleRandom
                                  // NOLINTNEXTLINE(cert-msc50-cpp, facebook-hte-BadCall-rand)
                                  ? static_cast<double>(1 + (std::rand() % params.fpsThrottleMs))
                                  : static_cast<double>(params.fpsThrottleMs);
      if (frameTimeMs < targetMs) {
        // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(targetMs - frameTimeMs)));
      }
    }

    if (window_) {
      glfwPollEvents();
    }
    if (frameNumber == session_->shellParams().screenshotNumber) {
      IGL_LOG_INFO("\nWe are running screenshot test - breaking after %u frame\n",
                   static_cast<uint32_t>(frameNumber));
      if (!session_->shellParams().screenshotFileName.empty()) {
        saveFrameBufferToPng(session_->shellParams().screenshotFileName.c_str(),
                             platform_->getDevice().createFramebuffer(
                                 {.colorAttachments = {{.texture = colorTexture}}}, nullptr),
                             *platform_);
      }
      break;
    }
    frameNumber++;
  }
}

void GlfwShell::teardown() noexcept {
  // Explicitly destroy all objects before exiting in order to make sure that
  // whatever else global destructors may there, will be called after these. One
  // example is a graphics resource tracker in the client code, which otherwise
  // would not be guaranteed to be called after the graphics resources release.
  session_.reset();
  platform_.reset();
  window_.reset();

  if (!shellParams_.isHeadless) {
    glfwTerminate();
  }
}

} // namespace igl::shell
