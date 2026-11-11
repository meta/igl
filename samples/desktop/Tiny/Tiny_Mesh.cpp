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

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES // NOLINT(bugprone-reserved-identifier)
#endif // _USE_MATH_DEFINES
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>

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
#include <glm/ext.hpp>
#include <glm/gtc/random.hpp>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <igl/FPSCounter.h>
#include <igl/IGL.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/VulkanContext.h>

#define TINY_TEST_USE_DEPTH_BUFFER 1
#define TINY_TEST_USE_ASYNC_SCREENSHOTS 1

#define USE_OPENGL_BACKEND 0

#if IGL_BACKEND_OPENGL && !IGL_BACKEND_VULKAN
// no IGL/Vulkan was compiled in, switch to IGL/OpenGL
#undef USE_OPENGL_BACKEND
#define USE_OPENGL_BACKEND 1
#endif

constexpr uint32_t kNumCubes = 16;

#if IGL_WITH_IGLU
#include <IGLU/imgui/Session.h>

namespace {

// NOLINTNEXTLINE(facebook-static-object-destructor-check)
std::unique_ptr<iglu::imgui::Session> imguiSession_;

// NOLINTNEXTLINE(facebook-static-object-destructor-check)
igl::shell::InputDispatcher inputDispatcher_;

} // namespace
#endif // IGL_WITH_IGLU

namespace {

const char* codeVS = R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec3 col;
layout (location=2) in vec2 st;
layout (location=0) out vec3 color;
layout (location=1) out vec2 uv;

layout (set = 1, binding = 0, std140) uniform UniformsPerFrame {
  mat4 proj;
  mat4 view;
} perFrame;

layout (set = 1, binding = 1, std140) uniform UniformsPerObject {
  mat4 model;
} perObject;

void main() {
  mat4 proj = perFrame.proj;
  mat4 view = perFrame.view;
  mat4 model = perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
  color = col;
  uv = st;
}
)";

const char* codeFS = R"(
layout (location=0) in vec3 color;
layout (location=1) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout (set = 0, binding = 0) uniform sampler2D uTex0;
layout (set = 0, binding = 1) uniform sampler2D uTex1;

void main() {
  vec4 t0 = texture(uTex0, 2.0 * uv);
  vec4 t1 = texture(uTex1,  uv);
  out_FragColor = vec4(color * (t0.rgb + t1.rgb), 1.0);
};
)";

using namespace igl;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

vec3 axis_[kNumCubes];

GLFWwindow* window = nullptr;
int width_ = 1024;
int height_ = 768;
igl::FPSCounter fps_;
bool saveScreenshot_ = false;
[[maybe_unused]] igl::SubmitHandle screenshotSubmitHandle = {};

constexpr uint32_t kNumBufferedFrames = 3;

// NOLINTNEXTLINE(facebook-static-object-destructor-check)
std::unique_ptr<IDevice> device;
// NOLINTNEXTLINE(facebook-static-object-destructor-check)
std::shared_ptr<ICommandQueue> commandQueue_;
// NOLINTBEGIN(facebook-static-object-destructor-check)
RenderPassDesc renderPass_;
FramebufferDesc framebufferDesc_;
std::shared_ptr<IFramebuffer> framebuffer;
std::shared_ptr<IRenderPipelineState> renderPipelineState_Mesh_;
std::shared_ptr<IBuffer> vb0_, ib0_; // buffers for vertices and indices
std::shared_ptr<IBuffer> screenCopy_;
std::vector<std::shared_ptr<IBuffer>> ubPerFrame_, ubPerObject;
// NOLINTEND(facebook-static-object-destructor-check)
std::shared_ptr<IVertexInputState> vertexInput0_;
std::shared_ptr<IDepthStencilState> depthStencilState_;
std::shared_ptr<ITexture> texture0_, texture1_;
std::shared_ptr<ISamplerState> sampler;

struct VertexPosUvw {
  vec3 position;
  vec3 color;
  vec2 uv;
};

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
};
struct UniformsPerObject {
  mat4 model;
};

const float kHalf = 1.0f;

// UV-mapped cube with indices: 24 vertices, 36 indices
const VertexPosUvw vertexData0[] = {
    // top
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.0, 0.0, 1.0}, .uv = {0, 0}}, // 0
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.0, 1.0}, .uv = {1, 0}}, // 1
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 2
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.0, 1.0, 1.0}, .uv = {0, 1}}, // 3
    // bottom
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 4
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.0, 1.0, 0.0}, .uv = {0, 1}}, // 5
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.0}, .uv = {1, 1}}, // 6
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.0, 0.0}, .uv = {1, 0}}, // 7
    // left
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.0}, .uv = {1, 0}}, // 8
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.0, 1.0, 0.0}, .uv = {0, 0}}, // 9
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.0, 1.0, 1.0}, .uv = {0, 1}}, // 10
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 11
    // right
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 12
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.0, 0.0}, .uv = {1, 0}}, // 13
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.0, 1.0}, .uv = {1, 1}}, // 14
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.0, 0.0, 1.0}, .uv = {0, 1}}, // 15
    // front
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.0, 0.0}, .uv = {0, 0}}, // 16
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.0}, .uv = {1, 0}}, // 17
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 18
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.0, 1.0}, .uv = {0, 1}}, // 19
    // back
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.0, 1.0, 0.0}, .uv = {1, 0}}, // 20
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 21
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.0, 0.0, 1.0}, .uv = {0, 1}}, // 22
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.0, 1.0, 1.0}, .uv = {1, 1}}, // 23
};

const uint16_t indexData[] = {0,  1,  2,  2,  3,  0,  4,  5,  6,  6,  7,  4,
                              8,  9,  10, 10, 11, 8,  12, 13, 14, 14, 15, 12,
                              16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};

UniformsPerFrame perFrame;
UniformsPerObject perObject[kNumCubes];

GLFWwindow* FOLLY_NULLABLE initIGL(bool isHeadless, bool enableVulkanValidationLayers) {
  if (!glfwInit()) {
    std::printf("glfwInit() failed");
    return nullptr;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  GLFWwindow* newWindow =
      isHeadless ? nullptr : glfwCreateWindow(1280, 1024, "Vulkan Mesh", nullptr, nullptr);

  if (!isHeadless && !newWindow) {
    glfwTerminate();
    return nullptr;
  }

  if (newWindow) {
    glfwSetErrorCallback([](int error, const char* description) {
      std::printf("GLFW Error (%i): %s\n", error, description);
    });

    glfwSetKeyCallback(newWindow, [](GLFWwindow* window, int key, int, int action, int) {
      if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
      if (key == GLFW_KEY_T && action == GLFW_PRESS) {
        texture1_.reset();
      }
      if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        saveScreenshot_ = true;
      }
    });

    // @lint-ignore CLANGTIDY
    glfwSetWindowSizeCallback(newWindow, [](GLFWwindow* /*window*/, int width, int height) {
      std::printf("Window resized! width=%d, height=%d\n", width, height);
      width_ = width;
      height_ = height;
#if !USE_OPENGL_BACKEND
      auto* vulkanDevice = static_cast<vulkan::Device*>(device.get());
      auto& ctx = vulkanDevice->getVulkanContext();
      ctx.initSwapchain(width_, height_);
#endif
    });

#if IGL_WITH_IGLU
    glfwSetCursorPosCallback(newWindow, [](auto* window, double x, double y) {
      inputDispatcher_.queueEvent(igl::shell::MouseMotionEvent(x, y, 0, 0));
    });
    glfwSetMouseButtonCallback(newWindow, [](auto* window, int button, int action, int mods) {
      double xpos = 0.0, ypos = 0.0;
      glfwGetCursorPos(window, &xpos, &ypos);
      using igl::shell::MouseButton;
      const MouseButton iglButton =
          (button == GLFW_MOUSE_BUTTON_LEFT)
              ? MouseButton::Left
              : (button == GLFW_MOUSE_BUTTON_RIGHT ? MouseButton::Right : MouseButton::Middle);
      inputDispatcher_.queueEvent(igl::shell::MouseButtonEvent(
          iglButton, action == GLFW_PRESS, static_cast<float>(xpos), static_cast<float>(ypos)));
    });
#endif // IGL_WITH_IGLU

    glfwGetWindowSize(newWindow, &width_, &height_);
  }

  // create a device
  {
    const igl::vulkan::VulkanContextConfig cfg = {
        .terminateOnValidationError = true,
        .enableValidation = enableVulkanValidationLayers,
        .headless = isHeadless,
    };
#ifdef _WIN32
    auto ctx = vulkan::HWDevice::createContext(
        cfg, newWindow ? (void*)glfwGetWin32Window(newWindow) : nullptr);
#elif IGL_PLATFORM_APPLE
    auto ctx = vulkan::HWDevice::createContext(
        cfg, newWindow ? (void*)glfwGetCocoaWindow(newWindow) : nullptr);
#elif defined(_XLESS_GLFW_)
    auto ctx = vulkan::HWDevice::createContext(cfg, nullptr, nullptr);
#elif IGL_PLATFORM_LINUX
    auto ctx =
        vulkan::HWDevice::createContext(cfg,
                                        newWindow ? (void*)glfwGetX11Window(newWindow) : nullptr,
                                        newWindow ? (void*)glfwGetX11Display() : nullptr);
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
        std::move(ctx), devices[0], static_cast<uint32_t>(width_), static_cast<uint32_t>(height_));
    if (!device) {
      IGL_LOG_ERROR("Failed to create a device.\n");
      return nullptr;
    }
  }

  // Vertex buffer, Index buffer and Vertex Input. Buffers are allocated in GPU memory.
  vb0_ = device->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                         .data = vertexData0,
                                         .length = sizeof(vertexData0),
                                         .storage = ResourceStorage::Private,
                                         .hint = 0,
                                         .debugName = "Buffer: vertex"},
                              nullptr);
  ib0_ = device->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                         .data = indexData,
                                         .length = sizeof(indexData),
                                         .storage = ResourceStorage::Private,
                                         .hint = 0,
                                         .debugName = "Buffer: index"},
                              nullptr);
  screenCopy_ = device->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Storage,
                                                .data = nullptr,
                                                .length = width_ * height_ * sizeof(uint32_t),
                                                .storage = ResourceStorage::Shared,
                                                .hint = 0,
                                                .debugName = "Buffer: screen copy"},
                                     nullptr);

  // create Uniform buffers to store uniforms for 2 objects
  for (uint32_t i = 0; i != kNumBufferedFrames; i++) {
    ubPerFrame_.push_back(
        device->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                                        .data = &perFrame,
                                        .length = sizeof(UniformsPerFrame),
                                        .storage = ResourceStorage::Shared,
                                        .hint = 0,
                                        .debugName = "Buffer: uniforms (per frame)"},
                             nullptr));
    ubPerObject.push_back(
        device->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                                        .data = perObject,
                                        .length = kNumCubes * sizeof(UniformsPerObject),
                                        .storage = ResourceStorage::Shared,
                                        .hint = 0,
                                        .debugName = "Buffer: uniforms (per object)"},
                             nullptr));
  }

  {
    const VertexInputStateDesc desc = {
        .numAttributes = 3,
        .attributes = {{.bufferIndex = 0,
                        .format = VertexAttributeFormat::Float3,
                        .offset = offsetof(VertexPosUvw, position),
                        .location = 0},
                       {.bufferIndex = 0,
                        .format = VertexAttributeFormat::Float3,
                        .offset = offsetof(VertexPosUvw, color),
                        .location = 1},
                       {.bufferIndex = 0,
                        .format = VertexAttributeFormat::Float2,
                        .offset = offsetof(VertexPosUvw, uv),
                        .location = 2}},
        .numInputBindings = 1,
        .inputBindings = {{.stride = sizeof(VertexPosUvw)}},
    };
    vertexInput0_ = device->createVertexInputState(desc, nullptr);
  }

  {
    const DepthStencilStateDesc desc{
        .compareFunction = igl::CompareFunction::Less,
        .isDepthWriteEnabled = true,
    };
    depthStencilState_ = device->createDepthStencilState(desc, nullptr);
  }

  {
    const uint32_t texWidth = 256;
    const uint32_t texHeight = 256;
    const TextureDesc desc = TextureDesc::new2D(igl::TextureFormat::BGRA_UNorm8,
                                                texWidth,
                                                texHeight,
                                                TextureDesc::TextureUsageBits::Sampled,
                                                "XOR pattern");
    texture0_ = device->createTexture(desc, nullptr);
    std::vector<uint32_t> pixels(texWidth * texHeight);
    for (uint32_t y = 0; y != texHeight; y++) {
      for (uint32_t x = 0; x != texWidth; x++) {
        // create a XOR pattern
        pixels[y * texWidth + x] = 0xFF000000 + ((x ^ y) << 16) + ((x ^ y) << 8) + (x ^ y);
      }
    }
    texture0_->upload(TextureRangeDesc::new2D(0, 0, texWidth, texHeight), pixels.data());
  }
  {
    using namespace std::filesystem;
    path dir = current_path();
    // find IGLU somewhere above our current directory
    // @fb-only
    const char* contentFolder = "third-party/content/src/";
    // @fb-only
    while (dir != current_path().root_path() && !exists(dir / path(contentFolder))) {
      dir = dir.parent_path();
    }
    int32_t texWidth = 0;
    int32_t texHeight = 0;
    int32_t channels = 0;
    uint8_t* pixels = stbi_load(
        (dir / path(contentFolder) / path("bistro/BuildingTextures/wood_polished_01_diff.png"))
            .string()
            .c_str(),
        &texWidth,
        &texHeight,
        &channels,
        4);
    IGL_DEBUG_ASSERT(pixels,
                     "Cannot load textures. Run `deploy_content.py` before running this app.");
    const TextureDesc desc = TextureDesc::new2D(igl::TextureFormat::RGBA_UNorm8,
                                                texWidth,
                                                texHeight,
                                                TextureDesc::TextureUsageBits::Sampled,
                                                "wood_polished_01_diff.png");
    texture1_ = device->createTexture(desc, nullptr);
    texture1_->upload(TextureRangeDesc::new2D(0, 0, texWidth, texHeight), pixels);
    stbi_image_free(pixels);
  }
  {
    igl::SamplerStateDesc desc = igl::SamplerStateDesc::newLinear();
    desc.addressModeU = igl::SamplerAddressMode::Repeat;
    desc.addressModeV = igl::SamplerAddressMode::Repeat;
    desc.debugName = "Sampler: linear";
    sampler = device->createSamplerState(desc, nullptr);
  }

  // Command queue: backed by different types of GPU HW queues
  CommandQueueDesc desc{};
  commandQueue_ = device->createCommandQueue(desc, nullptr);

  renderPass_.colorAttachments.push_back({
      .loadAction = LoadAction::Clear,
      .storeAction = StoreAction::Store,
      .clearColor = {1.0f, 0.0f, 0.0f, 1.0f},
  });
#if TINY_TEST_USE_DEPTH_BUFFER
  renderPass_.depthAttachment = {
      .loadAction = LoadAction::Clear,
      .storeAction = StoreAction::Store, // save it so we can display it via ImGui
      .clearDepth = 1.0,
  };
#else
  renderPass_.depthAttachment = {
      .loadAction = LoadAction::DontCare,
  };
#endif // TINY_TEST_USE_DEPTH_BUFFER

  // initialize random rotation axes for all cubes
  for (auto& axi : axis_) {
    axi = glm::sphericalRand(1.0f);
  }

  return newWindow;
}

void createRenderPipeline() {
  if (renderPipelineState_Mesh_) {
    return;
  }

  IGL_DEBUG_ASSERT(framebuffer);

  RenderPipelineDesc desc;

  desc.targetDesc.colorAttachments.resize(1);
  desc.targetDesc.colorAttachments[0].textureFormat =
      framebuffer->getColorAttachment(0)->getFormat();

  if (framebuffer->getDepthAttachment()) {
    desc.targetDesc.depthAttachmentFormat = framebuffer->getDepthAttachment()->getFormat();
  }

  desc.vertexInputState = vertexInput0_;
  desc.shaderStages = ShaderStagesCreator::fromModuleStringInput(
      *device, codeVS, "main", "", codeFS, "main", "", nullptr);

#if !TINY_TEST_USE_DEPTH_BUFFER
  desc.cullMode = igl::CullMode::Back;
#endif // TINY_TEST_USE_DEPTH_BUFFER

  desc.frontFaceWinding = igl::WindingMode::Clockwise;
  desc.debugName = igl::genNameHandle("Pipeline: mesh");
  renderPipelineState_Mesh_ = device->createRenderPipeline(desc, nullptr);
}

std::shared_ptr<ITexture> getVulkanNativeDrawable() {
  const auto& vkPlatformDevice = device->getPlatformDevice<igl::vulkan::PlatformDevice>();

  IGL_DEBUG_ASSERT(vkPlatformDevice != nullptr);

  Result ret;
  std::shared_ptr<ITexture> drawable = vkPlatformDevice->createTextureFromNativeDrawable(&ret);

  IGL_DEBUG_ASSERT(ret.isOk());
  return drawable;
}

std::shared_ptr<ITexture> getVulkanNativeDepth() {
  const auto& vkPlatformDevice = device->getPlatformDevice<igl::vulkan::PlatformDevice>();

  IGL_DEBUG_ASSERT(vkPlatformDevice != nullptr);

  Result ret;
  std::shared_ptr<ITexture> drawable =
      vkPlatformDevice->createTextureFromNativeDepth(width_, height_, &ret);

  IGL_DEBUG_ASSERT(ret.isOk());
  return drawable;
}

void createFramebuffer(const std::shared_ptr<ITexture>& nativeDrawable) {
  framebufferDesc_.colorAttachments[0].texture = nativeDrawable;

#if TINY_TEST_USE_DEPTH_BUFFER
  framebufferDesc_.depthAttachment.texture = getVulkanNativeDepth();
#endif // TINY_TEST_USE_DEPTH_BUFFER

  framebuffer = device->createFramebuffer(framebufferDesc_, nullptr);
  IGL_DEBUG_ASSERT(framebuffer);
}

void render(const std::shared_ptr<ITexture>& nativeDrawable, uint32_t frameIndex) {
  IGL_PROFILER_FUNCTION();

  if (!nativeDrawable) {
    return;
  }

#if IGL_WITH_IGLU
  imguiSession_->beginFrame(framebufferDesc_, 1.0f);

  ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::Image(ImTextureID(texture1_.get()), ImVec2(512, 512));
  ImGui::End();

  inputDispatcher_.processEvents();
#endif // IGL_WITH_IGLU

  const auto size = framebuffer->getColorAttachment(0)->getSize();
  if (size.width != width_ || size.height != height_) {
    createFramebuffer(nativeDrawable);
  } else {
    framebuffer->updateDrawable(nativeDrawable);
  }

  // from igl/shell/renderSessions/Textured3DCubeSession.cpp
  const float fov = float(45.0f * (M_PI / 180.0f));
  const float aspectRatio = static_cast<float>(width_) / static_cast<float>(height_);
  perFrame.proj = glm::perspectiveLH(fov, aspectRatio, 0.1f, 500.0f);
  // place a "camera" behind the cubes, the distance depends on the total number of cubes
  perFrame.view =
      glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, std::sqrt(kNumCubes / 16) * 20.0f * kHalf));
  ubPerFrame_[frameIndex]->upload(&perFrame, igl::BufferRange(sizeof(perFrame)));

  // rotate cubes around random axes
  for (uint32_t i = 0; i != kNumCubes; i++) {
    const float direction = std::pow(-1, static_cast<float>(i + 1));
    const uint32_t cubesInLine = static_cast<uint32_t>(std::sqrt(kNumCubes));
    const vec3 offset = vec3(-1.5f * std::sqrt(kNumCubes) + 4.0f * (i % cubesInLine),
                             // NOLINTNEXTLINE(bugprone-integer-division)
                             -1.5f * std::sqrt(kNumCubes) + 4.0f * (i / cubesInLine),
                             0);
    perObject[i].model = glm::rotate(glm::translate(mat4(1.0f), offset),
                                     direction * static_cast<float>(glfwGetTime()),
                                     axis_[i]);
  }

  ubPerObject[frameIndex]->upload(&perObject, igl::BufferRange(sizeof(perObject)));

  // Command buffers (1-N per thread): create, submit and forget
  std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer({}, nullptr);

  const igl::Viewport viewport = {.x = 0.0f,
                                  .y = 0.0f,
                                  .width = static_cast<float>(width_),
                                  .height = static_cast<float>(height_),
                                  .minDepth = 0.0f,
                                  .maxDepth = +1.0f};
  const igl::ScissorRect scissor = {.x = 0,
                                    .y = 0,
                                    .width = static_cast<uint32_t>(width_),
                                    .height = static_cast<uint32_t>(height_)};

  // This will clear the framebuffer
  const auto commands = buffer->createRenderCommandEncoder(renderPass_, framebuffer);

  commands->bindRenderPipelineState(renderPipelineState_Mesh_);
  commands->bindViewport(viewport);
  commands->bindScissorRect(scissor);
  commands->pushDebugGroupLabel("Render Mesh", igl::Color(1, 0, 0));
  commands->bindVertexBuffer(0, *vb0_);
  commands->bindDepthStencilState(depthStencilState_);
  commands->bindBuffer(0, ubPerFrame_[frameIndex].get());
  commands->bindTexture(0, igl::BindTarget::kFragment, texture0_.get());
  commands->bindTexture(1, igl::BindTarget::kFragment, texture1_.get());
  commands->bindSamplerState(0, igl::BindTarget::kFragment, sampler.get());
  commands->bindSamplerState(1, igl::BindTarget::kFragment, sampler.get());
  // Draw 2 cubes: we use uniform buffer to update matrices
  commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
  for (uint32_t i = 0; i != kNumCubes; i++) {
    commands->bindBuffer(1, ubPerObject[frameIndex].get(), i * sizeof(UniformsPerObject));
    commands->drawIndexed(3u * 6u * 2u);
  }
  commands->popDebugGroupLabel();
#if IGL_WITH_IGLU
  imguiSession_->drawFPS(fps_.getAverageFPS());
  imguiSession_->endFrame(*device, *commands);
#endif // IGL_WITH_IGLU
  commands->endEncoding();
  if (saveScreenshot_) {
    buffer->copyTextureToBuffer(*nativeDrawable, *screenCopy_, 0);
  }
  buffer->present(nativeDrawable);

  auto submitHandle = commandQueue_->submit(*buffer);

  if (auto* pd = device->getPlatformDevice<igl::vulkan::PlatformDevice>(); pd && saveScreenshot_) {
    saveScreenshot_ = false;
#if TINY_TEST_USE_ASYNC_SCREENSHOTS
    pd->deferredTask(std::packaged_task<void()>([]() {
                       void* data =
                           screenCopy_->map(BufferRange(screenCopy_->getSizeInBytes()), nullptr);
                       stbi_write_bmp("screenshot.bmp", width_, height_, 4, data);
                       screenCopy_->unmap();
                       IGL_LOG_INFO("Screenshot saved.\n");
                     }),
                     submitHandle);
#else
    // store the submit handle from which we want to capture a screenshot
    screenshotSubmitHandle = submitHandle;
  } else if (screenshotSubmitHandle && // we poll the submit handle every frame until it's ready
             pd->waitOnSubmitHandle(screenshotSubmitHandle, 0)) {
    void* data = screenCopy_->map(BufferRange(screenCopy_->getSizeInBytes()), nullptr);
    stbi_write_bmp("screenshot.bmp", width_, height_, 4, data);
    screenCopy_->unmap();
    screenshotSubmitHandle = {};
    IGL_LOG_INFO("Screenshot saved.\n");
#endif // TINY_TEST_USE_ASYNC_SCREENSHOTS
  }
}

} // namespace

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

  window = initIGL(isHeadless, enableVulkanValidationLayers);
  if (!device) {
    glfwTerminate();
    return 1;
  }

  createFramebuffer(getVulkanNativeDrawable());
  createRenderPipeline();

#if IGL_WITH_IGLU
  imguiSession_ = std::make_unique<iglu::imgui::Session>(*device, inputDispatcher_);
#endif // IGL_WITH_IGLU

  double prevTime = glfwGetTime();

  uint32_t frameIndex = 0;

  // Main loop
  while (!window || !glfwWindowShouldClose(window)) {
    const double newTime = glfwGetTime();
    fps_.updateFPS(newTime - prevTime);
    prevTime = newTime;
    render(getVulkanNativeDrawable(), frameIndex);
    frameIndex = (frameIndex + 1) % kNumBufferedFrames;
    if (window) {
      glfwPollEvents();
    } else {
      std::printf("We are running headless - breaking after 1 frame\n");
      std::shared_ptr<ITexture> texture = framebuffer->getColorAttachment(0);
      const Dimensions dim = texture->getDimensions();
      std::vector<uint8_t> pixelsRGBA(dim.width * dim.height * 4);
      std::vector<uint8_t> pixelsRGB(dim.width * dim.height * 3);
      framebuffer->copyBytesColorAttachment(*commandQueue_,
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
      const char* fileName = "TinyMesh.png";
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

#if IGL_WITH_IGLU
  imguiSession_ = nullptr;
#endif // IGL_WITH_IGLU

  // destroy all the Vulkan stuff before closing the window
  vb0_ = nullptr;
  ib0_ = nullptr;
  screenCopy_ = nullptr;
  ubPerFrame_.clear();
  ubPerObject.clear();
  renderPipelineState_Mesh_ = nullptr;
  texture0_ = nullptr;
  texture1_ = nullptr;
  sampler = nullptr;
  framebufferDesc_ = {};
  framebuffer = nullptr;
  device.reset(nullptr);

  // glfwDestroyWindow() is documented to do nothing when passed NULL, so a nullable `window` here
  // is safe; this is a false positive.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
