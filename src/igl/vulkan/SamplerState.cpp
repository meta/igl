/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/SamplerState.h>

#include <algorithm>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

namespace {
VkFilter samplerMinMagFilterToVkFilter(igl::SamplerMinMagFilter filter) {
  switch (filter) {
  case igl::SamplerMinMagFilter::Nearest:
    return VK_FILTER_NEAREST;
  case igl::SamplerMinMagFilter::Linear:
    return VK_FILTER_LINEAR;
  }
  IGL_DEBUG_ABORT("SamplerMinMagFilter value not handled: %d", static_cast<int>(filter));
  return VK_FILTER_LINEAR;
}

VkSamplerMipmapMode samplerMipFilterToVkSamplerMipmapMode(igl::SamplerMipFilter filter) {
  switch (filter) {
  case igl::SamplerMipFilter::Disabled:
  case igl::SamplerMipFilter::Nearest:
    return VK_SAMPLER_MIPMAP_MODE_NEAREST;
  case igl::SamplerMipFilter::Linear:
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
  IGL_DEBUG_ABORT("SamplerMipFilter value not handled: %d", static_cast<int>(filter));
  return VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode samplerAddressModeToVkSamplerAddressMode(igl::SamplerAddressMode mode) {
  switch (mode) {
  case igl::SamplerAddressMode::Repeat:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case igl::SamplerAddressMode::Clamp:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case igl::SamplerAddressMode::MirrorRepeat:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case igl::SamplerAddressMode::ClampToBorder:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  }
  IGL_DEBUG_ABORT("SamplerAddressMode value not handled: %d", static_cast<int>(mode));
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkSamplerCreateInfo samplerStateDescToVkSamplerCreateInfo(const igl::SamplerStateDesc& desc,
                                                          const VkPhysicalDeviceLimits& limits) {
  IGL_DEBUG_ASSERT(desc.mipLodMax >= desc.mipLodMin,
                   "mipLodMax (%d) must be greater than or equal to mipLodMin (%d)",
                   static_cast<int>(desc.mipLodMax),
                   static_cast<int>(desc.mipLodMin));

  VkSamplerCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .magFilter = samplerMinMagFilterToVkFilter(desc.magFilter),
      .minFilter = samplerMinMagFilterToVkFilter(desc.minFilter),
      .mipmapMode = samplerMipFilterToVkSamplerMipmapMode(desc.mipFilter),
      .addressModeU = samplerAddressModeToVkSamplerAddressMode(desc.addressModeU),
      .addressModeV = samplerAddressModeToVkSamplerAddressMode(desc.addressModeV),
      .addressModeW = samplerAddressModeToVkSamplerAddressMode(desc.addressModeW),
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_FALSE,
      .maxAnisotropy = 0.0f,
      .compareEnable = desc.depthCompareEnabled ? VK_TRUE : VK_FALSE,
      .compareOp = desc.depthCompareEnabled
                       ? igl::vulkan::compareFunctionToVkCompareOp(desc.depthCompareFunction)
                       : VK_COMPARE_OP_ALWAYS,
      .minLod = static_cast<float>(desc.mipLodMin),
      .maxLod = desc.mipFilter == igl::SamplerMipFilter::Disabled
                    ? 0.0f
                    : static_cast<float>(desc.mipLodMax),
      .borderColor = (desc.addressModeU == igl::SamplerAddressMode::ClampToBorder ||
                      desc.addressModeV == igl::SamplerAddressMode::ClampToBorder ||
                      desc.addressModeW == igl::SamplerAddressMode::ClampToBorder)
                         ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                         : VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE,
  };

  if (desc.maxAnisotropic > 1) {
    const bool isAnisotropicFilteringSupported = limits.maxSamplerAnisotropy > 1;
    IGL_DEBUG_ASSERT(isAnisotropicFilteringSupported,
                     "Anisotropic filtering is not supported by the device.");
    ci.anisotropyEnable = isAnisotropicFilteringSupported ? VK_TRUE : VK_FALSE;

    if (limits.maxSamplerAnisotropy < desc.maxAnisotropic) {
      IGL_LOG_INFO(
          "Supplied sampler anisotropic value greater than max supported by the device, setting to "
          "%.0f",
          static_cast<double>(limits.maxSamplerAnisotropy));
    }
    ci.maxAnisotropy = std::min(static_cast<float>(limits.maxSamplerAnisotropy),
                                static_cast<float>(desc.maxAnisotropic));
  }

  return ci;
}

} // namespace

namespace igl::vulkan {

SamplerState::SamplerState(Device& device) : device_(device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
}

Result SamplerState::create(const SamplerStateDesc& desc) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  const VulkanContext& ctx = device_.getVulkanContext();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  desc_ = desc;

  Result result;
  sampler_ =
      Holder<SamplerHandle>(&device_,
                            ctx.createSampler(samplerStateDescToVkSamplerCreateInfo(
                                                  desc, ctx.getVkPhysicalDeviceProperties().limits),
                                              textureFormatToVkFormat(desc.yuvFormat),
                                              &result,
                                              desc_.debugName.c_str()));

  if (!IGL_DEBUG_VERIFY(result.isOk())) {
    return result;
  }

  return sampler_.valid() ? Result()
                          : Result(Result::Code::InvalidOperation, "Cannot create VulkanSampler");
}

uint32_t SamplerState::getSamplerId() const {
  const VulkanSampler* sampler = device_.getVulkanContext().samplers_.get(sampler_);
  return sampler ? sampler->samplerId : 0;
}

bool SamplerState::isYUV() const noexcept {
  return desc_.yuvFormat != TextureFormat::Invalid;
}

} // namespace igl::vulkan
