/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanTextureAccessor.h"

#include "ITextureAccessor.h"

#include <igl/Macros.h>
#if IGL_BACKEND_VULKAN
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanTexture.h>
#endif

#if defined(IGL_CMAKE_BUILD)
#include <igl/IGLSafeC.h>
#else

#include <cstddef>
#endif

namespace iglu::textureaccessor {

VulkanTextureAccessor::VulkanTextureAccessor(std::shared_ptr<igl::ITexture> texture) :
  ITextureAccessor(std::move(texture)) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  assignTexture(texture_);
}

void VulkanTextureAccessor::assignTexture(std::shared_ptr<igl::ITexture> texture) {
  IGL_PROFILER_FUNCTION();
  if (!texture) {
    return;
  }
#if IGL_BACKEND_VULKAN
  auto* vkTexture = static_cast<igl::vulkan::Texture*>(texture.get());
  const igl::vulkan::VulkanImage& vkImage = vkTexture->getVulkanTexture().image;
  vkImage_ = vkImage.getVkImage();
  ctx_ = vkImage.ctx_;
  vkImageAspectFlags_ = vkTexture->getVulkanTexture().imageView_.getVkImageAspectFlags();
  const auto textureFormatProperties =
      igl::TextureFormatProperties::fromTextureFormat(texture->getFormat());
  numBytesRequired_ =
      static_cast<size_t>(textureFormatProperties.getBytesPerRow(texture->getSize().width)) *
      textureFormatProperties.getRows(texture->getFullRange());

  textureWidth_ = texture->getSize().width;
  textureHeight_ = texture->getSize().height;
  vkImageFormat_ = vkImage.imageFormat_;
  vkImageLayout_ = vkImage.imageLayout_;
  bytesPerRow_ = textureFormatProperties.getBytesPerRow(texture->getSize().width);
#endif
  texture_ = std::move(texture);
}

void VulkanTextureAccessor::requestBytes(igl::ICommandQueue& /*commandQueue*/,
                                         std::shared_ptr<igl::ITexture> texture) {
  IGL_PROFILER_FUNCTION();
  status_ = RequestStatus::InProgress;

  if (texture != nullptr) {
    assignTexture(texture);
  }

  if (numBytesRequired_ != latestBytesRead_.size()) {
    latestBytesRead_.resize(numBytesRequired_);
  }

  IGL_DEBUG_ASSERT(texture_ != nullptr, "texture_ is nullptr");

  status_ = RequestStatus::Ready;
}

size_t VulkanTextureAccessor::copyBytes(unsigned char* ptr, size_t length) {
  IGL_PROFILER_FUNCTION();
  if (length < numBytesRequired_) {
    return 0;
  }
#if IGL_BACKEND_VULKAN
  static_cast<void>(ctx_->stagingDevice_->getImageData2D(
      vkImage_,
      0,
      0,
      VkRect2D{.offset = VkOffset2D{.x = 0, .y = 0},
               .extent = VkExtent2D{.width = textureWidth_, .height = textureHeight_}},
      igl::TextureFormatProperties::fromTextureFormat(texture_->getFormat()),
      vkImageFormat_,
      vkImageLayout_,
      VK_IMAGE_ASPECT_COLOR_BIT,
      ptr,
      static_cast<uint32_t>(bytesPerRow_),
      false));
#endif
  return numBytesRequired_;
}

RequestStatus VulkanTextureAccessor::getRequestStatus() {
  return status_;
}

std::vector<unsigned char>& VulkanTextureAccessor::getBytes() {
  IGL_PROFILER_FUNCTION();
  copyBytes(latestBytesRead_.data(), latestBytesRead_.size());
  return latestBytesRead_;
}

} // namespace iglu::textureaccessor
