/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/PlatformDevice.h>

#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanSwapchain.h>

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
#include <android/hardware_buffer.h>
#include <igl/vulkan/android/NativeHWBuffer.h>
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

namespace igl::vulkan {

PlatformDevice::PlatformDevice(Device& device) : device_(device) {}

std::shared_ptr<ITexture> PlatformDevice::createTextureFromNativeDepth(uint32_t width,
                                                                       uint32_t height,
                                                                       Result* IGL_NULLABLE
                                                                           outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  const auto& ctx = device_.getVulkanContext();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  const auto& swapChain = ctx.swapchain_;

  if (!ctx.hasSwapchain()) {
    nativeDepthTexture_ = nullptr;
    Result::setResult(outResult, Result::Code::Ok);
    return nullptr;
  };

  std::shared_ptr<VulkanTexture> vkTex = swapChain->getCurrentDepthTexture();

  if (!IGL_DEBUG_VERIFY(vkTex)) {
    Result::setResult(outResult, Result::Code::InvalidOperation, "Swapchain has no valid texture");
    return nullptr;
  }

  IGL_DEBUG_ASSERT(vkTex->image.imageFormat_ != VK_FORMAT_UNDEFINED, "Invalid image format");

  const auto iglFormat = vkFormatToTextureFormat(vkTex->image.imageFormat_);
  if (!IGL_DEBUG_VERIFY(iglFormat != TextureFormat::Invalid)) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Invalid surface depth format");
    return nullptr;
  }

  // allocate new drawable textures if its null or mismatches in size or format
  if (!nativeDepthTexture_ || width != nativeDepthTexture_->getDimensions().width ||
      height != nativeDepthTexture_->getDimensions().height ||
      iglFormat != nativeDepthTexture_->getFormat()) {
    const TextureDesc desc = TextureDesc::new2D(iglFormat,
                                                width,
                                                height,
                                                TextureDesc::TextureUsageBits::Attachment |
                                                    TextureDesc::TextureUsageBits::Sampled,
                                                "SwapChain Texture");
    nativeDepthTexture_ = std::make_shared<Texture>(device_, std::move(vkTex), desc);
  }

  Result::setResult(outResult, Result::Code::Ok);

  return nativeDepthTexture_;
}

std::shared_ptr<ITexture> PlatformDevice::createTextureFromNativeDrawable(
    Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  const auto& ctx = device_.getVulkanContext();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  if (!ctx.hasSwapchain()) {
    nativeDrawableTextures_.clear();
    Result::setResult(outResult, Result::Code::Ok);
    return nullptr;
  };

  const auto& swapChain = ctx.swapchain_;

  auto vkTex = swapChain->getCurrentVulkanTexture();

  if (!vkTex) {
    Result::setResult(outResult, Result::Code::InvalidOperation, "Swapchain has no valid texture");
    return nullptr;
  }

  IGL_DEBUG_ASSERT(vkTex->image.imageFormat_ != VK_FORMAT_UNDEFINED, "Invalid image format");

  const TextureFormat iglFormat = vkFormatToTextureFormat(vkTex->image.imageFormat_);
  if (!IGL_DEBUG_VERIFY(iglFormat != TextureFormat::Invalid)) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Invalid surface color format");
    return nullptr;
  }

  const size_t width = static_cast<size_t>(swapChain->getWidth());
  const size_t height = static_cast<size_t>(swapChain->getHeight());
  const uint32_t currentImageIndex = swapChain->getCurrentImageIndex();

  // resize nativeDrawableTextures_ pushing null pointers
  // null pointers will be allocated later as needed
  if (currentImageIndex >= nativeDrawableTextures_.size()) {
    nativeDrawableTextures_.resize(static_cast<size_t>(currentImageIndex) + 1, nullptr);
  }

  const auto result = nativeDrawableTextures_[currentImageIndex];

  // allocate new drawable textures if its null or mismatches in size or format
  if (!result || width != result->getDimensions().width ||
      height != result->getDimensions().height || iglFormat != result->getFormat()) {
    // The swapchain VkImage may carry VK_IMAGE_USAGE_STORAGE_BIT when the
    // surface + format support it (see VulkanSwapchain::chooseUsageFlags(), e.g.
    // a UNORM swapchain that exposes VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT).
    // Advertise Storage on the igl texture so consumers can bind the drawable as
    // a storage image -- e.g. a compute shader that imageStore()s directly to the
    // swapchain instead of round-tripping through a resolve pass.
    //
    // Hold Storage back for sRGB drawables even if the VkImage reports it: a
    // compute imageStore() to an sRGB image is not consistent across drivers
    // (most bypass the sRGB encode -- SPIR-V rgba8 stores are UNORM -- while
    // MoltenVK applies it), so exposing it would be a cross-platform trap. UNORM
    // swapchains (the recommended path) are unaffected.
    TextureDesc::TextureUsage usage = TextureDesc::TextureUsageBits::Attachment;
    if ((vkTex->image.usageFlags_ & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
        !TextureFormatProperties::fromTextureFormat(iglFormat).isSRGB()) {
      usage |= TextureDesc::TextureUsageBits::Storage;
    }
    // NOLINTBEGIN(clang-diagnostic-shorten-64-to-32)
    const TextureDesc desc =
        TextureDesc::new2D(iglFormat, width, height, usage, "SwapChain Texture");
    // NOLINTEND(clang-diagnostic-shorten-64-to-32)
    nativeDrawableTextures_[currentImageIndex] =
        std::make_shared<Texture>(device_, std::move(vkTex), desc);
  }

  Result::setResult(outResult, Result::Code::Ok);

  return nativeDrawableTextures_[currentImageIndex];
}

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
/// returns a android::NativeHWTextureBuffer on platforms supporting it
/// this texture allows CPU and GPU to both read/write memory
std::shared_ptr<ITexture> PlatformDevice::createTextureWithSharedMemory(const TextureDesc& desc,
                                                                        Result* IGL_NULLABLE
                                                                            outResult) const {
  Result subResult;

  auto texture = std::make_shared<android::NativeHWTextureBuffer>(device_, desc.format);
  subResult = texture->createHWBuffer(desc, false, false);
  Result::setResult(outResult, subResult.code, subResult.message);
  if (!subResult.isOk()) {
    return nullptr;
  }

  return std::move(texture);
}

std::shared_ptr<ITexture> PlatformDevice::createTextureWithSharedMemory(
    struct AHardwareBuffer* buffer,
    Result* IGL_NULLABLE outResult) const {
  if (!buffer) {
    IGL_LOG_ERROR("PlatformDevice::createTextureWithSharedMemory(): null AHardwareBuffer");
    Result::setResult(outResult, Result::Code::ArgumentNull, "AHardwareBuffer is null");
    return nullptr;
  }

  Result subResult;

  AHardwareBuffer_Desc hwbDesc{};
  AHardwareBuffer_describe(buffer, &hwbDesc);

  // Vendor-specific YCbCr AHB formats are valid Vulkan imports but may not map to IGL formats.
  // Use YUV_NV12 only to satisfy Texture's non-invalid format assert before import overwrites
  // desc_.
  TextureFormat textureFormat = igl::android::getIglFormat(hwbDesc.format);
  if (textureFormat == TextureFormat::Invalid) {
    textureFormat = TextureFormat::YUV_NV12;
  }

  auto texture = std::make_shared<android::NativeHWTextureBuffer>(device_, textureFormat);
  subResult = texture->createWithHWBuffer(buffer);
  Result::setResult(outResult, subResult.code, subResult.message);
  if (!subResult.isOk()) {
    return nullptr;
  }

  return std::move(texture);
}
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

VkFence PlatformDevice::getVkFenceFromSubmitHandle(SubmitHandle handle) const {
  return device_.getVulkanContext().getVkFenceFromSubmitHandle(handle);
}

bool PlatformDevice::waitOnSubmitHandle(SubmitHandle handle, uint64_t timeoutNanoseconds) const {
  if (handle == 0) {
    IGL_LOG_ERROR("Invalid submit handle passed to waitOnSubmitHandle");
    return false;
  }

  const auto& ctx = device_.getVulkanContext();
  const auto& immediateCommands = ctx.immediate_;

  return immediateCommands->wait(VulkanImmediateCommands::SubmitHandle(handle),
                                 timeoutNanoseconds) != VK_TIMEOUT;
}

void PlatformDevice::deferredTask(std::packaged_task<void()>&& task, SubmitHandle handle) const {
  if (!handle) {
    IGL_LOG_ERROR("Invalid submit handle passed to PlatformDevice::deferredTask()");
    return;
  }

  device_.getVulkanContext().deferredTask(std::move(task),
                                          VulkanImmediateCommands::SubmitHandle(handle));
}

int PlatformDevice::getFenceFdFromSubmitHandle(SubmitHandle handle) const {
  return device_.getVulkanContext().getFenceFdFromSubmitHandle(handle);
}

bool PlatformDevice::supportsReusableCommandBuffers() const {
  return device_.getVulkanContext().supportsReusableCommandBuffers();
}

std::shared_ptr<ICommandBuffer> PlatformDevice::createReusableCommandBuffer(
    const ReusableCommandBufferDesc& desc,
    Result* IGL_NULLABLE outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  VulkanContext& ctx = device_.getVulkanContext();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  if (!ctx.supportsReusableCommandBuffers()) {
    Result::setResult(outResult,
                      Result::Code::Unsupported,
                      "Reusable command buffers need descriptor sets that nothing rewrites");
    return nullptr;
  }

  VkCommandPool commandPool = VK_NULL_HANDLE;
  const VkResult result = ivkCreateCommandPool(&ctx.vf_,
                                               ctx.getVkDevice(),
                                               VkCommandPoolCreateFlags{0},
                                               ctx.deviceQueues_.graphicsQueueFamilyIndex,
                                               &commandPool);
  if (result != VK_SUCCESS) {
    setResultFrom(outResult, result);
    return nullptr;
  }
  // the command pool exists once vkCreateCommandPool() succeeds
  // @fb-only
  ivkSetDebugObjectName(&ctx.vf_,
                        ctx.getVkDevice(),
                        VK_OBJECT_TYPE_COMMAND_POOL,
                        (uint64_t)commandPool,
                        IGL_FORMAT("Command Pool: {}", desc.debugName).c_str());

  Result::setOk(outResult);
  return std::make_shared<CommandBuffer>(ctx, desc, commandPool);
}

bool PlatformDevice::isRecorded(const ICommandBuffer& commandBuffer) const {
  const auto& vkCommandBuffer = static_cast<const CommandBuffer&>(commandBuffer);
  return vkCommandBuffer.belongsTo(device_.getVulkanContext()) && vkCommandBuffer.isRecorded();
}

void PlatformDevice::executeCommands(ICommandBuffer& commandBuffer,
                                     ICommandBuffer& reusable,
                                     Result* IGL_NULLABLE outResult) const {
  const VulkanContext& ctx = device_.getVulkanContext();
  auto& vkCommandBuffer = static_cast<CommandBuffer&>(commandBuffer);
  if (!vkCommandBuffer.belongsTo(ctx) || !static_cast<CommandBuffer&>(reusable).belongsTo(ctx)) {
    Result::setResult(
        outResult, Result::Code::ArgumentInvalid, "The command buffer belongs to another device");
    return;
  }
  vkCommandBuffer.executeCommands(reusable, outResult);
}

} // namespace igl::vulkan
