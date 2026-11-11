/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <memory>
#include <shell/windows/common/GlfwShell.h>
#include <shell/shared/platform/win/PlatformWin.h>
#include <igl/Core.h>
#include <igl/IGL.h>
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/VulkanContext.h>

using namespace igl;
namespace igl::shell {
namespace {

// Fullscreen triangle vertex shader for stereo side-by-side present
const char* kStereoPresentVS = R"(
#version 460
layout(location = 0) out vec2 uv;
out gl_PerVertex { vec4 gl_Position; };
void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
)";

// Fragment shader: left half = layer 0, right half = layer 1
const char* kStereoPresentFS = R"(
#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2DArray stereoTex;
void main() {
    float layer = uv.x < 0.5 ? 0.0 : 1.0;
    vec2 texCoord = vec2(uv.x < 0.5 ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
    outColor = texture(stereoTex, vec3(texCoord, layer));
}
)";

class VulkanShell final : public GlfwShell {
  SurfaceTextures createSurfaceTextures() noexcept final;
  std::shared_ptr<Platform> createPlatform() noexcept final;

  void willCreateWindow() noexcept final;
  void postUpdate() noexcept final;

  // Stereo present resources (initialized lazily on first multiview frame)
  void initStereoPresent(IDevice& device);

  std::shared_ptr<ITexture> offscreenColor_;
  std::shared_ptr<ITexture> offscreenDepth_;
  std::shared_ptr<ITexture> swapchainColor_;
  std::shared_ptr<ICommandQueue> presentQueue_;
  std::shared_ptr<IRenderPipelineState> presentPipeline_;
  std::shared_ptr<ISamplerState> presentSampler_;
  bool stereoPresentInitialized_ = false;
};

void VulkanShell::willCreateWindow() noexcept {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
}

std::shared_ptr<Platform> VulkanShell::createPlatform() noexcept {
  igl::vulkan::VulkanContextConfig cfg = {
#if defined(_MSC_VER) && !IGL_DEBUG
      .enableValidation = false,
#else
      .enableValidation = shellParams().enableVulkanValidationLayers,
#endif
      .requestedSwapChainTextureFormat = sessionConfig().swapchainColorTextureFormat,
      .headless = shellParams().isHeadless,
  };
  auto ctx = vulkan::HWDevice::createContext(
      cfg,
#if defined(_WIN32)
      window() ? (void*)glfwGetWin32Window(window()) : nullptr // NOLINT(performance-no-int-to-ptr)
#else
      window() ? (void*)glfwGetX11Window(window()) // NOLINT(performance-no-int-to-ptr)
               : nullptr,
      (void*)glfwGetX11Display() // NOLINT(performance-no-int-to-ptr)
#endif
  );
  if (!ctx) {
    IGL_LOG_ERROR("Vulkan 1.1 or newer is required.\n");
    return nullptr;
  }

  // Prioritize discrete GPUs. If not found, use any that is available.
  std::vector<HWDeviceDesc> devices =
      vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::DiscreteGpu), nullptr);
  if (devices.empty()) {
    devices = vulkan::HWDevice::queryDevices(
        *ctx, HWDeviceQueryDesc(HWDeviceType::IntegratedGpu), nullptr);
  }
  if (devices.empty()) {
    // Lavapipe etc
    devices =
        vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::SoftwareGpu), nullptr);
  }
  if (devices.empty()) {
    IGL_LOG_ERROR("Could not find Vulkan device with requested capabilities.\n");
    return nullptr;
  }

  auto vulkanDevice = vulkan::HWDevice::create(std::move(ctx),
                                               devices[0],
                                               static_cast<uint32_t>(shellParams().viewportSize.x),
                                               static_cast<uint32_t>(shellParams().viewportSize.y));
  if (!vulkanDevice) {
    IGL_LOG_ERROR("Failed to create a Vulkan device.\n");
    return nullptr;
  }

  // Verify multiview support — if unsupported, report and run as usual
  if (shellParams().forceMultiview && !vulkanDevice->hasFeature(DeviceFeatures::Multiview)) {
    IGL_LOG_ERROR(
        "[IGL Shell] --force-multiview: multiview is not supported by this device. "
        "Running in normal mono mode.\n");
    auto& params = const_cast<ShellParams&>(shellParams());
    params.forceMultiview = false;
    params.renderMode = RenderMode::Mono;
    params.viewParams.clear();
    params.shouldPresent = true;
  }

  return std::make_shared<PlatformWin>(std::move(vulkanDevice));
}

void VulkanShell::initStereoPresent(IDevice& device) {
  if (stereoPresentInitialized_) {
    return;
  }

  // Create command queue for stereo present pass
  presentQueue_ = device.createCommandQueue({}, nullptr);

  // Create sampler
  const SamplerStateDesc samplerDesc = {
      .minFilter = SamplerMinMagFilter::Linear,
      .magFilter = SamplerMinMagFilter::Linear,
      .addressModeU = SamplerAddressMode::Clamp,
      .addressModeV = SamplerAddressMode::Clamp,
  };
  presentSampler_ = device.createSamplerState(samplerDesc, nullptr);

  // Create shaders
  auto shaderStages = igl::ShaderStagesCreator::fromModuleStringInput(
      device, kStereoPresentVS, "main", "", kStereoPresentFS, "main", "", nullptr);

  // Create render pipeline (no depth, no vertex input — fullscreen triangle via gl_VertexIndex)
  const RenderPipelineDesc pipelineDesc = {
      .shaderStages = std::move(shaderStages),
      .targetDesc = {.colorAttachments = {{.textureFormat = swapchainColor_->getFormat()}}},
  };
  presentPipeline_ = device.createRenderPipeline(pipelineDesc, nullptr);

  stereoPresentInitialized_ = true;
  IGL_LOG_INFO("[IGL Shell] Stereo present pipeline initialized for --force-multiview\n");
}

SurfaceTextures VulkanShell::createSurfaceTextures() noexcept {
  IGL_PROFILER_FUNCTION();
  auto& device = platform().getDevice();
  const auto& vkPlatformDevice = device.getPlatformDevice<igl::vulkan::PlatformDevice>();

  IGL_DEBUG_ASSERT(vkPlatformDevice != nullptr);

  Result ret;
  auto color = vkPlatformDevice->createTextureFromNativeDrawable(&ret);
  IGL_DEBUG_ASSERT(ret.isOk());
  auto depth = vkPlatformDevice->createTextureFromNativeDepth(
      shellParams().viewportSize.x, shellParams().viewportSize.y, &ret);
  IGL_DEBUG_ASSERT(ret.isOk());

  if (!shellParams().forceMultiview || shellParams().isHeadless) {
    return SurfaceTextures{.color = std::move(color), .depth = std::move(depth)};
  }

  // --- Force multiview: create offscreen 2-layer array textures ---
  // Suppress session present — our stereo present pass handles the swapchain
  const_cast<ShellParams&>(shellParams()).shouldPresent = false;

  swapchainColor_ = std::move(color);

  const auto dimensions = swapchainColor_->getDimensions();
  const TextureFormat colorFormat = swapchainColor_->getFormat();
  const TextureFormat depthFormat = depth->getFormat();

  // Create offscreen 2-layer textures once (reuse across frames)
  if (!offscreenColor_ || offscreenColor_->getDimensions() != dimensions) {
    const TextureDesc colorDesc = {
        .width = dimensions.width,
        .height = dimensions.height,
        .depth = 1,
        .numLayers = 2,
        .numSamples = 1,
        .usage = TextureDesc::TextureUsageBits::Attachment | TextureDesc::TextureUsageBits::Sampled,
        .numMipLevels = 1,
        .type = TextureType::TwoDArray,
        .format = colorFormat,
        .storage = ResourceStorage::Private,
    };
    offscreenColor_ = device.createTexture(colorDesc, &ret);
    IGL_DEBUG_ASSERT(ret.isOk(), "Failed to create offscreen multiview color texture");

    const TextureDesc depthDesc = {
        .width = dimensions.width,
        .height = dimensions.height,
        .depth = 1,
        .numLayers = 2,
        .numSamples = 1,
        .usage = TextureDesc::TextureUsageBits::Attachment,
        .numMipLevels = 1,
        .type = TextureType::TwoDArray,
        .format = depthFormat,
        .storage = ResourceStorage::Private,
    };
    offscreenDepth_ = device.createTexture(depthDesc, &ret);
    IGL_DEBUG_ASSERT(ret.isOk(), "Failed to create offscreen multiview depth texture");

    // Reset stereo present pipeline when textures change (format might differ)
    stereoPresentInitialized_ = false;

    IGL_LOG_INFO("[IGL Shell] Created multiview offscreen textures: %ux%u, 2 layers\n",
                 dimensions.width,
                 dimensions.height);
  }

  return SurfaceTextures{.color = offscreenColor_, .depth = offscreenDepth_};
}

void VulkanShell::postUpdate() noexcept {
  if (!shellParams().forceMultiview || !swapchainColor_ || !offscreenColor_) {
    return;
  }

  auto& device = platform().getDevice();

  if (!stereoPresentInitialized_) {
    initStereoPresent(device);
  }

  // Framebuffer must be recreated each frame because the swapchain image changes
  const FramebufferDesc fbDesc = {
      .colorAttachments = {{.texture = swapchainColor_}},
  };
  Result ret;
  const auto framebuffer = device.createFramebuffer(fbDesc, &ret);
  if (!ret.isOk()) {
    IGL_LOG_ERROR("Failed to create stereo present framebuffer\n");
    return;
  }

  // Render stereo side-by-side to swapchain
  auto cmdBuf = presentQueue_->createCommandBuffer({}, nullptr);

  const RenderPassDesc renderPass = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
      }},
  };

  auto encoder = cmdBuf->createRenderCommandEncoder(renderPass, framebuffer);

  // Set viewport to match swapchain dimensions exactly
  const auto swapDims = swapchainColor_->getDimensions();
  const Viewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(swapDims.width),
      .height = static_cast<float>(swapDims.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  encoder->bindViewport(viewport);
  const ScissorRect scissor = {
      .x = 0,
      .y = 0,
      .width = swapDims.width,
      .height = swapDims.height,
  };
  encoder->bindScissorRect(scissor);

  encoder->bindRenderPipelineState(presentPipeline_);
  encoder->bindTexture(0, BindTarget::kFragment, offscreenColor_.get());
  encoder->bindSamplerState(0, BindTarget::kFragment, presentSampler_.get());
  encoder->draw(3);
  encoder->endEncoding();

  // Mark swapchain for present — this triggers vkQueuePresentKHR on submit
  cmdBuf->present(swapchainColor_);

  presentQueue_->submit(*cmdBuf);
}

} // namespace

} // namespace igl::shell

int main(int argc, char* argv[]) {
  igl::shell::VulkanShell shell;

  igl::shell::RenderSessionWindowConfig suggestedWindowConfig = {
      .width = 1024,
      .height = 768,
      .windowMode = shell::WindowMode::MaximizedWindow,
  };
  igl::shell::RenderSessionConfig suggestedConfig = {
      .displayName = "Vulkan 1.1",
      .backendVersion = {.flavor = BackendFlavor::Vulkan, .majorVersion = 1, .minorVersion = 1},
      .swapchainColorTextureFormat = TextureFormat::BGRA_SRGB,
  };

  if (!shell.initialize(argc, argv, suggestedWindowConfig, suggestedConfig)) {
    shell.teardown();
    return -1;
  }

  shell.run();
  shell.teardown();

  return 0;
}
