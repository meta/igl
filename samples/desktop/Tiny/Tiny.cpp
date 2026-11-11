/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <igl/Config.h>

#if defined(_XLESS_GLFW_)
// do nothing
#elif IGL_PLATFORM_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#elif IGL_PLATFORM_APPLE
#define GLFW_EXPOSE_NATIVE_COCOA
#elif IGL_PLATFORM_LINUX
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#else
#error Unsupported OS
#endif

// NOLINTNEXTLINE(facebook-unused-include-check)
#include <GLFW/glfw3native.h>
#include <cstdio>
#include <cstring>
#include <stb/stb_image_write.h>
#include <igl/IGL.h>

#define USE_OPENGL_BACKEND 0

#if IGL_BACKEND_OPENGL && !IGL_BACKEND_VULKAN
// no IGL/Vulkan was compiled in, switch to IGL/OpenGL
#undef USE_OPENGL_BACKEND
#define USE_OPENGL_BACKEND 1
#endif

// NOLINTBEGIN(facebook-unused-include-check)
// clang-format off
#if USE_OPENGL_BACKEND
  #if IGL_PLATFORM_WINDOWS
    #include <igl/opengl/wgl/Context.h>
    #include <igl/opengl/wgl/Device.h>
    #include <igl/opengl/wgl/HWDevice.h>
    #include <igl/opengl/wgl/PlatformDevice.h>
  #elif IGL_PLATFORM_LINUX
    #include <igl/opengl/glx/Context.h>
    #include <igl/opengl/glx/Device.h>
    #include <igl/opengl/glx/HWDevice.h>
    #include <igl/opengl/glx/PlatformDevice.h>
  #endif
#else
  #include <igl/vulkan/Common.h>
  #include <igl/vulkan/Device.h>
  #include <igl/vulkan/HWDevice.h>
  #include <igl/vulkan/PlatformDevice.h>
  #include <igl/vulkan/VulkanContext.h>
#endif // USE_OPENGL_BACKEND
// clang-format on
// NOLINTEND(facebook-unused-include-check)

#define ENABLE_MULTIPLE_COLOR_ATTACHMENTS 0

#if ENABLE_MULTIPLE_COLOR_ATTACHMENTS
static const uint32_t kNumColorAttachments = 4;
#else
static const uint32_t kNumColorAttachments = 1;
#endif

// NOLINTBEGIN(facebook-unused-include-check)
#if defined(__cpp_lib_format) && !IGL_PLATFORM_APPLE
#include <format>
#define IGL_FORMAT std::format
#else
#include <fmt/format.h>
#define IGL_FORMAT fmt::format
#endif // __cpp_lib_format
// NOLINTEND(facebook-unused-include-check)

// NOLINTNEXTLINE(facebook-static-object-destructor-check,misc-use-anonymous-namespace)
static std::string codeVS = R"(
#version 460
layout (location=0) out vec3 color;
const vec2 pos[3] = vec2[3](
	vec2(-0.6, -0.4),
	vec2( 0.6, -0.4),
	vec2( 0.0,  0.6)
);
const vec3 col[3] = vec3[3](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0)
);
void main() {
	gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
	color = col[gl_VertexIndex];
}
)";

#if ENABLE_MULTIPLE_COLOR_ATTACHMENTS
const char* codeFS = R"(
#version 460
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor;
layout (location=1) out vec4 out_FragColor1;

void main() {
	out_FragColor = vec4(color, 1.0);
	out_FragColor1 = vec4(1.0, 0.0, 0.0, 1.0);
};
)";
#else
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
const static char* codeFS = R"(
#version 460
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor;
void main() {
	out_FragColor = vec4(color, 1.0);
};
)";
#endif

using namespace igl;

// NOLINTBEGIN(facebook-static-object-destructor-check,misc-use-anonymous-namespace)
static int width = 1024;
static int height = 768;

static std::unique_ptr<IDevice> device;
static std::shared_ptr<ICommandQueue> commandQueue;
static RenderPassDesc renderPass;
static std::shared_ptr<IFramebuffer> framebuffer;
// NOLINTNEXTLINE(facebook-static-object-destructor-check)
static std::shared_ptr<IRenderPipelineState> renderPipelineStateTriangle;
// NOLINTEND(facebook-static-object-destructor-check,misc-use-anonymous-namespace)

static GLFWwindow* FOLLY_NULLABLE initIGL(bool isHeadless, bool enableVulkanValidationLayers) {
  if (!glfwInit()) {
    std::printf("glfwInit() failed");
    return nullptr;
  }

#if USE_OPENGL_BACKEND
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, true);
  glfwWindowHint(GLFW_DOUBLEBUFFER, true);
  glfwWindowHint(GLFW_SRGB_CAPABLE, true);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
#else
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

#if USE_OPENGL_BACKEND
  const char* title = "OpenGL Triangle";
#else
  const char* title = "Vulkan Triangle";
#endif

  GLFWwindow* window = isHeadless ? nullptr
                                  : glfwCreateWindow(width, height, title, nullptr, nullptr);

  if (window) {
    glfwSetErrorCallback([](int error, const char* description) {
      std::printf("GLFW Error (%i): %s\n", error, description);
    });

    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int, int action, int) {
      if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
    });

    // @lint-ignore CLANGTIDY
    glfwSetWindowSizeCallback(window, [](GLFWwindow* /*window*/, int w, int h) {
      width = w;
      height = h;
      std::printf("Window resized! width=%d, height=%d\n", width, height);
#if !USE_OPENGL_BACKEND
      auto* vulkanDevice = static_cast<vulkan::Device*>(device.get());
      auto& ctx = vulkanDevice->getVulkanContext();
      ctx.initSwapchain(width, height);
#endif
    });

    glfwGetWindowSize(window, &width, &height);
  }

  // create a device
  {
#if USE_OPENGL_BACKEND
#if IGL_PLATFORM_WINDOWS
    auto ctx = std::make_unique<igl::opengl::wgl::Context>(GetDC(glfwGetWin32Window(window)),
                                                           glfwGetWGLContext(window));
    device = std::make_unique<igl::opengl::wgl::Device>(std::move(ctx));
#elif IGL_PLATFORM_LINUX
    auto ctx = std::make_unique<igl::opengl::glx::Context>(
        nullptr,
        glfwGetX11Display(),
        (igl::opengl::glx::GLXDrawable)glfwGetX11Window(window),
        (igl::opengl::glx::GLXContext)glfwGetGLXContext(window));

    device = std::make_unique<igl::opengl::glx::Device>(std::move(ctx));
#endif
#else
    const igl::vulkan::VulkanContextConfig cfg{
        .terminateOnValidationError = false,
        .enableValidation = enableVulkanValidationLayers,
        .headless = isHeadless,
    };
#ifdef _WIN32
    auto ctx =
        vulkan::HWDevice::createContext(cfg, window ? (void*)glfwGetWin32Window(window) : nullptr);
#elif IGL_PLATFORM_APPLE
    auto ctx =
        vulkan::HWDevice::createContext(cfg, window ? (void*)glfwGetCocoaWindow(window) : nullptr);
#elif defined(_XLESS_GLFW_)
    auto ctx = vulkan::HWDevice::createContext(cfg, nullptr, nullptr);
#elif IGL_PLATFORM_LINUX
    auto ctx = vulkan::HWDevice::createContext(
        cfg, window ? (void*)glfwGetX11Window(window) : nullptr, (void*)glfwGetX11Display());
#else
#error Unsupported OS
#endif

    if (!ctx) {
      IGL_LOG_ERROR("Vulkan 1.1 or newer is required.\n");
      return nullptr;
    }

    std::vector<HWDeviceDesc> devices =
        vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::DiscreteGpu), nullptr);
    if (devices.empty()) {
      devices = vulkan::HWDevice::queryDevices(
          *ctx, HWDeviceQueryDesc(HWDeviceType::IntegratedGpu), nullptr);
    }
    if (devices.empty() || cfg.headless) {
      // LavaPipe etc
      devices = vulkan::HWDevice::queryDevices(
          *ctx, HWDeviceQueryDesc(HWDeviceType::SoftwareGpu), nullptr);
    }
    if (devices.empty()) {
      IGL_LOG_ERROR("No suitable Vulkan device.\n");
      return nullptr;
    }

    device = vulkan::HWDevice::create(
        std::move(ctx), devices[0], static_cast<uint32_t>(width), static_cast<uint32_t>(height));
#endif
    if (!device) {
      IGL_LOG_ERROR("Failed to create a device.\n");
      return nullptr;
    }
  }

  commandQueue = device->createCommandQueue({}, nullptr);

  renderPass.colorAttachments.resize(kNumColorAttachments);

  // first color attachment
  for (uint32_t i = 0; i < kNumColorAttachments; ++i) {
    // Generate sparse color attachments by skipping alternate slots
    if (i & 0x1) {
      continue;
    }
    renderPass.colorAttachments[i] = {
        .loadAction = LoadAction::Clear,
        .storeAction = StoreAction::Store,
        .clearColor = {1.0f, 1.0f, 1.0f, 1.0f},
    };
  }
  renderPass.depthAttachment.loadAction = LoadAction::DontCare;

  return window;
}

static void createRenderPipeline() {
  if (renderPipelineStateTriangle) {
    return;
  }

  IGL_DEBUG_ASSERT(framebuffer);

  RenderPipelineDesc desc;

  desc.targetDesc.colorAttachments.resize(kNumColorAttachments);

  for (uint32_t i = 0; i < kNumColorAttachments; ++i) {
    // color attachment can be sparse
    if (framebuffer->getColorAttachment(i)) {
      desc.targetDesc.colorAttachments[i].textureFormat =
          framebuffer->getColorAttachment(i)->getFormat();
    }
  }

  if (framebuffer->getDepthAttachment()) {
    desc.targetDesc.depthAttachmentFormat = framebuffer->getDepthAttachment()->getFormat();
  }

#if USE_OPENGL_BACKEND
  codeVS = std::regex_replace(codeVS, std::regex("gl_VertexIndex"), "gl_VertexID");
#endif

  desc.shaderStages = ShaderStagesCreator::fromModuleStringInput(
      *device, codeVS.c_str(), "main", "", codeFS, "main", "", nullptr);
  renderPipelineStateTriangle = device->createRenderPipeline(desc, nullptr);
  IGL_DEBUG_ASSERT(renderPipelineStateTriangle);
}

static std::shared_ptr<ITexture> getNativeDrawable() {
  Result ret;
  std::shared_ptr<ITexture> drawable;
#if USE_OPENGL_BACKEND
#if IGL_PLATFORM_WINDOWS
  const auto& platformDevice = device->getPlatformDevice<opengl::wgl::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDrawable(&ret);
#elif IGL_PLATFORM_LINUX
  const auto& platformDevice = device->getPlatformDevice<opengl::glx::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDrawable(width, height, &ret);
#endif
#else
  const auto& platformDevice = device->getPlatformDevice<igl::vulkan::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDrawable(&ret);
#endif
  IGL_DEBUG_ASSERT(ret.isOk(), ret.message.c_str());
  return drawable;
}

static void createFramebuffer(const std::shared_ptr<ITexture>& nativeDrawable) {
  FramebufferDesc framebufferDesc;
  framebufferDesc.colorAttachments[0].texture = nativeDrawable;

  for (uint32_t i = 1; i < kNumColorAttachments; ++i) {
    // Generate sparse color attachments by skipping alternate slots
    if (i & 0x1) {
      continue;
    }
    const TextureDesc desc = TextureDesc::new2D(
        nativeDrawable->getFormat(),
        nativeDrawable->getDimensions().width,
        nativeDrawable->getDimensions().height,
        TextureDesc::TextureUsageBits::Attachment | TextureDesc::TextureUsageBits::Sampled,
        IGL_FORMAT("{}C{}", framebufferDesc.debugName.c_str(), i - 1).c_str());

    framebufferDesc.colorAttachments[i].texture = device->createTexture(desc, nullptr);
  }
  framebuffer = device->createFramebuffer(framebufferDesc, nullptr);
  IGL_DEBUG_ASSERT(framebuffer);
}

static void render(const std::shared_ptr<ITexture>& nativeDrawable) {
  if (!nativeDrawable) {
    return;
  }

  const auto size = framebuffer->getColorAttachment(0)->getSize();
  if (size.width != width || size.height != height) {
    createFramebuffer(nativeDrawable);
  } else {
    framebuffer->updateDrawable(nativeDrawable);
  }

  // Command buffers (1-N per thread): create, submit and forget
  const std::shared_ptr<ICommandBuffer> buffer = commandQueue->createCommandBuffer({}, nullptr);

  const igl::Viewport viewport = {.x = 0.0f,
                                  .y = 0.0f,
                                  .width = static_cast<float>(width),
                                  .height = static_cast<float>(height),
                                  .minDepth = 0.0f,
                                  .maxDepth = +1.0f};
  const igl::ScissorRect scissor = {.x = 0,
                                    .y = 0,
                                    .width = static_cast<uint32_t>(width),
                                    .height = static_cast<uint32_t>(height)};

  // This will clear the framebuffer
  const auto commands = buffer->createRenderCommandEncoder(renderPass, framebuffer);

  commands->bindRenderPipelineState(renderPipelineStateTriangle);
  commands->bindViewport(viewport);
  commands->bindScissorRect(scissor);
  commands->pushDebugGroupLabel("Render Triangle", igl::Color(1, 0, 0));
  commands->draw(3);
  commands->popDebugGroupLabel();
  commands->endEncoding();

  buffer->present(nativeDrawable);

  commandQueue->submit(*buffer);
}

int main(int argc, char* argv[]) {
  bool isHeadless = false;
  bool enableVulkanValidationLayers = true;

  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--headless")) {
      isHeadless = true;
    } else if (!std::strcmp(argv[i], "--disable-vulkan-validation-layers")) {
      enableVulkanValidationLayers = false;
    }
  }

  GLFWwindow* window = initIGL(isHeadless, enableVulkanValidationLayers);
  if (!device) {
    glfwTerminate();
    return 1;
  }

  createFramebuffer(getNativeDrawable());
  createRenderPipeline();

  // Main loop
  while (!window || !glfwWindowShouldClose(window)) {
    render(getNativeDrawable());
    if (window) {
      glfwPollEvents();
    } else {
      std::printf("We are running headless - breaking after 1 frame\n");
      std::shared_ptr<ITexture> texture = framebuffer->getColorAttachment(0);
      const Dimensions dim = texture->getDimensions();
      std::vector<uint8_t> pixelsRGBA(dim.width * dim.height * 4);
      std::vector<uint8_t> pixelsRGB(dim.width * dim.height * 3);
      framebuffer->copyBytesColorAttachment(*commandQueue,
                                            0,
                                            pixelsRGBA.data(),
                                            TextureRangeDesc::new2D(0, 0, dim.width, dim.height));
      if (texture->getFormat() == igl::TextureFormat::BGRA_UNorm8 ||
          texture->getFormat() == igl::TextureFormat::BGRA_SRGB) {
        // swap R-B
        for (uint32_t i = 0; i < pixelsRGBA.size(); i += 4) {
          std::swap(pixelsRGBA[i + 0], pixelsRGBA[i + 2]);
        }
      }
      // convert to RGB
      for (uint32_t i = 0; i < pixelsRGB.size() / 3; i++) {
        pixelsRGB[3 * i + 0] = pixelsRGBA[4 * i + 0];
        pixelsRGB[3 * i + 1] = pixelsRGBA[4 * i + 1];
        pixelsRGB[3 * i + 2] = pixelsRGBA[4 * i + 2];
      }
      const char* fileName = "Tiny.png";
      IGLLog(IGLLogInfo, "Writing screenshot to: '%s'\n", fileName);
      stbi_flip_vertically_on_write(1);
      stbi_write_png(fileName,
                     static_cast<int>(dim.width),
                     static_cast<int>(dim.height),
                     3,
                     pixelsRGB.data(),
                     0);
      break;
    }
  }

  // destroy all the Vulkan stuff before closing the window
  renderPipelineStateTriangle = nullptr;
  framebuffer = nullptr;
  device.reset(nullptr);

  // glfwDestroyWindow() is documented to do nothing when passed NULL, so a nullable `window` here
  // is safe; this is a false positive.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
