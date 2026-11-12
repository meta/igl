/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/ResourcesBinder.h>

#include <type_traits>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/PipelineState.h>
#include <igl/vulkan/SamplerState.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanTexture.h>

namespace igl::vulkan {

static_assert(std::is_trivially_copyable_v<BindingsBuffers>);
static_assert(std::is_trivially_copyable_v<BindingsTextures>);
static_assert(std::is_trivially_copyable_v<BindingsStorageImages>);

ResourcesBinder::ResourcesBinder(const CommandBuffer* commandBuffer,
                                 VulkanContext& ctx,
                                 VkPipelineBindPoint bindPoint) :
  ctx_(ctx),
  cmdBuffer_(commandBuffer ? commandBuffer->getVkCommandBuffer() : VK_NULL_HANDLE),
  bindPoint_(bindPoint),
  nextSubmitHandle_(commandBuffer ? commandBuffer->getNextSubmitHandle()
                                  : VulkanImmediateCommands::SubmitHandle{}),
  arenas_(commandBuffer ? commandBuffer->getDescriptorPoolsArenas() : nullptr) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx_);
}

void ResourcesBinder::bindBuffer(uint32_t index,
                                 Buffer* buffer,
                                 size_t bufferOffset,
                                 size_t bufferSize) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(index < IGL_UNIFORM_BLOCKS_BINDING_MAX)) {
    IGL_DEBUG_ABORT("Buffer index should not exceed kMaxBindingSlots");
    return;
  }

  const bool isUniformBuffer =
      ((buffer->getBufferType() & BufferDesc::BufferTypeBits::Uniform) != 0);

  IGL_DEBUG_ASSERT(isUniformBuffer ||
                       ((buffer->getBufferType() & BufferDesc::BufferTypeBits::Storage) != 0),
                   "The buffer must be a uniform or storage buffer");
  if (bufferOffset) {
    const auto& limits = ctx_.getVkPhysicalDeviceProperties().limits;
    const uint32_t alignment =
        static_cast<uint32_t>(isUniformBuffer ? limits.minUniformBufferOffsetAlignment
                                              : limits.minStorageBufferOffsetAlignment);
    if (!IGL_DEBUG_VERIFY((alignment == 0) || (bufferOffset % alignment == 0))) {
      IGL_LOG_ERROR("`bufferOffset = %u` must be a multiple of `VkPhysicalDeviceLimits::%s = %u`",
                    static_cast<uint32_t>(bufferOffset),
                    isUniformBuffer ? "minUniformBufferOffsetAlignment"
                                    : "minStorageBufferOffsetAlignment",
                    alignment);
      return;
    }
  }

  const VkBuffer buf = buffer ? buffer->getVkBuffer() : ctx_.dummyUniformBuffer_->getVkBuffer();
  VkDescriptorBufferInfo& slot = bindingsBuffers_.buffers[index];

  if (slot.buffer != buf || slot.offset != bufferOffset) {
    slot = {
        .buffer = buf,
        .offset = bufferOffset,
        // When no explicit size is given, bind the rest of the buffer starting at `bufferOffset`.
        // Using the full buffer size here would make `offset + range` exceed the buffer when a
        // non-zero offset is used (e.g. binding a sub-range of a uniform buffer).
        .range = bufferSize ? bufferSize
                 : buffer   ? (buffer->getSizeInBytes() - bufferOffset)
                            : VK_WHOLE_SIZE,
    };
    if (ctx_.features().has_VK_KHR_buffer_device_address) {
      bindingsBuffers_.addresses[index] = buffer ? buffer->gpuAddress(bufferOffset)
                                                 : ctx_.dummyUniformBuffer_->getVkDeviceAddress();
    }
    isDirtyFlags_ |= DirtyFlagBits_Buffers;
  }
}

void ResourcesBinder::bindSamplerState(uint32_t index, SamplerState* samplerState) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(index < IGL_TEXTURE_SAMPLERS_MAX)) {
    IGL_DEBUG_ABORT("Invalid sampler index");
    return;
  }

  const VulkanSampler* newSampler = samplerState ? ctx_.samplers_.get(samplerState->sampler_)
                                                 : nullptr;

  const VkSampler sampler = newSampler ? newSampler->vkSampler : VK_NULL_HANDLE;

  if (bindingsTextures_.samplers[index] != sampler) {
    bindingsTextures_.samplers[index] = sampler;
    isDirtyFlags_ |= DirtyFlagBits_Textures;
  }
}

void ResourcesBinder::bindTexture(uint32_t index, Texture* tex) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(index < IGL_TEXTURE_SAMPLERS_MAX)) {
    IGL_DEBUG_ABORT("Invalid texture index");
    return;
  }

  if (tex) {
    const bool isSampled = tex ? (tex->getUsage() & TextureDesc::TextureUsageBits::Sampled) > 0
                               : false;
    const bool isStorage = tex ? (tex->getUsage() & TextureDesc::TextureUsageBits::Storage) > 0
                               : false;

    if (!IGL_DEBUG_VERIFY(isSampled || isStorage)) {
      IGL_DEBUG_ABORT(
          "Did you forget to specify TextureUsageBits::Sampled or "
          "TextureUsageBits::Storage on your texture? `Sampled` is used for sampling; "
          "`Storage` is used for load/store operations");
    }
  }

  const VulkanTexture* newTexture = tex ? &tex->getVulkanTexture() : nullptr;

#if IGL_DEBUG_ABORT_ENABLED
  if (newTexture) {
    const VulkanImage& img = newTexture->image;
    IGL_DEBUG_ASSERT(img.samples_ == VK_SAMPLE_COUNT_1_BIT,
                     "Multisampled images cannot be sampled in shaders");
    if (bindPoint_ == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      // If you trip this assert, then you are likely using an IGL texture
      // that was not rendered to by IGL. If that's the case, then make sure
      // the underlying image is transitioned to
      // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
      IGL_DEBUG_ASSERT(img.imageLayout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
      IGL_DEBUG_ASSERT(img.imageLayout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
                       img.imageLayout_ == VK_IMAGE_LAYOUT_GENERAL);
    }
  }
#endif // IGL_DEBUG_ABORT_ENABLED

  // multisampled images cannot be directly accessed from shaders
  const bool isTextureAvailable =
      (newTexture != nullptr) &&
      ((newTexture->image.samples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT);
  const bool isSampledImage = isTextureAvailable && newTexture->image.isSampledImage();

  const VkImageView imageView = isSampledImage ? newTexture->imageView_.vkImageView
                                               : VK_NULL_HANDLE;

  if (bindingsTextures_.textures[index] != imageView) {
    bindingsTextures_.textures[index] = imageView;
    isDirtyFlags_ |= DirtyFlagBits_Textures;
  }
}

void ResourcesBinder::bindStorageImage(uint32_t index, Texture* tex) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(index < IGL_STORAGE_IMAGES_MAX)) {
    IGL_DEBUG_ABORT("Invalid storage image index");
    return;
  }

  const bool isStorage = tex ? (tex->getUsage() & TextureDesc::TextureUsageBits::Storage) > 0
                             : false;

  if (tex) {
    if (!IGL_DEBUG_VERIFY(isStorage)) {
      IGL_DEBUG_ABORT("Did you forget to specify TextureUsageBits::Storage on your texture?");
    }
  }

  const VulkanTexture* newTexture = tex ? &tex->getVulkanTexture() : nullptr;

#if IGL_DEBUG_ABORT_ENABLED
  if (newTexture) {
    const VulkanImage& img = newTexture->image;
    IGL_DEBUG_ASSERT(img.samples_ == VK_SAMPLE_COUNT_1_BIT,
                     "Multisampled images cannot be sampled in shaders");
    // If you trip this assert, then you are likely using an IGL texture
    // that was not rendered to by IGL. If that's the case, then make sure
    // the underlying image is transitioned to
    // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    IGL_DEBUG_ASSERT(img.imageLayout_ == VK_IMAGE_LAYOUT_GENERAL);
  }
#endif // IGL_DEBUG_ABORT_ENABLED

  // multisampled images cannot be directly accessed from shaders
  const bool isTextureAvailable =
      (newTexture != nullptr) &&
      ((newTexture->image.samples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT);
  const bool isStorageImage = isTextureAvailable && newTexture->image.isStorageImage();

  const VkImageView imageView = isStorageImage ? newTexture->imageView_.vkImageView
                                               : VK_NULL_HANDLE;

  if (bindingsStorageImages_.images[index] != imageView) {
    bindingsStorageImages_.images[index] = imageView;
    isDirtyFlags_ |= DirtyFlagBits_StorageImages;
  }
}

void ResourcesBinder::bindInputAttachment(uint32_t index,
                                          Texture* IGL_NULLABLE texture,
                                          uint32_t mipLevel,
                                          uint32_t layer,
                                          FramebufferMode mode) {
  IGL_PROFILER_FUNCTION();

  if (!IGL_DEBUG_VERIFY(index < IGL_COLOR_ATTACHMENTS_MAX)) {
    IGL_DEBUG_ABORT("Invalid input attachment index");
    return;
  }
  if (!IGL_DEBUG_VERIFY(isGraphics(),
                        "Input attachments can be bound only to graphics pipelines")) {
    return;
  }

  VkImageView imageView = VK_NULL_HANDLE;
  if (texture) {
    const bool isInputAttachment =
        (texture->getUsage() & TextureDesc::TextureUsageBits::InputAttachment) != 0;
    if (IGL_DEBUG_VERIFY(
            isInputAttachment,
            "Did you forget to specify TextureUsageBits::InputAttachment on your texture?")) {
      IGL_DEBUG_ASSERT(mipLevel < texture->getNumMipLevels() && layer < texture->getNumVkLayers(),
                       "Invalid mip level or layer of an input attachment");
      imageView = texture->getVkImageViewForFramebuffer(mipLevel, layer, mode);
    }
  }

  if (bindingsStorageImages_.inputAttachments[index] != imageView) {
    bindingsStorageImages_.inputAttachments[index] = imageView;
    isDirtyFlags_ |= DirtyFlagBits_StorageImages;
  }
}

void ResourcesBinder::updateBindings(VkPipelineLayout layout, const PipelineState& state) {
  ctx_.features().has_VK_EXT_descriptor_buffer ? updateBindingsByDescriptorBuffer(layout, state)
                                               : updateBindingsByDescriptorSet(layout, state);
}

void ResourcesBinder::updateBindingsByDescriptorBuffer(VkPipelineLayout layout,
                                                       const PipelineState& state) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_UPDATE);

  IGL_DEBUG_ASSERT(layout != VK_NULL_HANDLE);

  if ((isDirtyFlags_ & DirtyFlagBits_Textures) != 0) {
    ctx_.updateBindingsTexturesByDescriptorBuffer(cmdBuffer_,
                                                  layout,
                                                  bindPoint_,
                                                  nextSubmitHandle_,
                                                  bindingsTextures_,
                                                  *state.dslCombinedImageSamplers,
                                                  state.info);
  }
  if ((isDirtyFlags_ & DirtyFlagBits_Buffers) != 0) {
    ctx_.updateBindingsBuffersByDescriptorBuffer(cmdBuffer_,
                                                 layout,
                                                 bindPoint_,
                                                 nextSubmitHandle_,
                                                 bindingsBuffers_,
                                                 *state.dslBuffers,
                                                 state.info);
  }
  if ((isDirtyFlags_ & DirtyFlagBits_StorageImages) != 0) {
    ctx_.updateBindingsStorageImagesByDescriptorBuffer(cmdBuffer_,
                                                       layout,
                                                       bindPoint_,
                                                       nextSubmitHandle_,
                                                       bindingsStorageImages_,
                                                       *state.dslStorageImages,
                                                       state.info);
  }

  isDirtyFlags_ = 0;
}

void ResourcesBinder::updateBindingsByDescriptorSet(VkPipelineLayout layout,
                                                    const PipelineState& state) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_UPDATE);

  IGL_DEBUG_ASSERT(layout != VK_NULL_HANDLE);

  if ((isDirtyFlags_ & DirtyFlagBits_Textures) != 0) {
    ctx_.updateBindingsTextures(cmdBuffer_,
                                layout,
                                bindPoint_,
                                nextSubmitHandle_,
                                arenas_,
                                bindingsTextures_,
                                *state.dslCombinedImageSamplers,
                                state.info);
  }
  if ((isDirtyFlags_ & DirtyFlagBits_Buffers) != 0) {
    ctx_.updateBindingsBuffers(cmdBuffer_,
                               layout,
                               bindPoint_,
                               nextSubmitHandle_,
                               arenas_,
                               bindingsBuffers_,
                               *state.dslBuffers,
                               state.info);
  }
  if ((isDirtyFlags_ & DirtyFlagBits_StorageImages) != 0) {
    ctx_.updateBindingsStorageImages(cmdBuffer_,
                                     layout,
                                     bindPoint_,
                                     nextSubmitHandle_,
                                     arenas_,
                                     bindingsStorageImages_,
                                     *state.dslStorageImages,
                                     state.info);
  }

  isDirtyFlags_ = 0;
}

void ResourcesBinder::bindPipeline(VkPipeline pipeline, const util::SpvModuleInfo* info) {
  IGL_PROFILER_FUNCTION();

  if (!info) {
    lastInfo_ = nullptr;
  }

  if (lastPipelineBound_ == pipeline) {
    return;
  }

  const bool keepDescriptors = keepDescriptorsOnNextBind_ && info == lastInfo_;
  if (info && !keepDescriptors) {
    // a new pipeline might want a new descriptors configuration
    if (!info->buffers.empty()) {
      isDirtyFlags_ |= DirtyFlagBits_Buffers;
    }
    if (!info->textures.empty()) {
      isDirtyFlags_ |= DirtyFlagBits_Textures;
    }
    if (!info->images.empty() || !info->inputAttachments.empty()) {
      isDirtyFlags_ |= DirtyFlagBits_StorageImages;
    }
  }
  keepDescriptorsOnNextBind_ = false;

  lastPipelineBound_ = pipeline;
  lastInfo_ = info;

  if (pipeline != VK_NULL_HANDLE) {
#if IGL_VULKAN_PRINT_COMMANDS
    IGL_LOG_INFO("%p vkCmdBindPipeline(%s, %p)\n",
                 cmdBuffer_,
                 bindPoint_ == VK_PIPELINE_BIND_POINT_GRAPHICS ? "GRAPHICS" : "COMPUTE",
                 pipeline);
#endif // IGL_VULKAN_PRINT_COMMANDS
    ctx_.vf_.vkCmdBindPipeline(cmdBuffer_, bindPoint_, pipeline);
  }
}

void ResourcesBinder::invalidateBoundPipeline() {
  IGL_PROFILER_FUNCTION();

  lastPipelineBound_ = VK_NULL_HANDLE;
  keepDescriptorsOnNextBind_ = true;
}

void ResourcesBinder::invalidateAll() {
  IGL_PROFILER_FUNCTION();

  lastPipelineBound_ = VK_NULL_HANDLE;
  // As after invalidateBoundPipeline(), the next pipeline marks resource types dirty only if its
  // shaders differ from the ones of the last pipeline. A caller that clears the dirty flag of a
  // type can then tell from the flag that the type needs a new descriptor set.
  keepDescriptorsOnNextBind_ = true;
  isDirtyFlags_ = DirtyFlagBits_Textures | DirtyFlagBits_Buffers | DirtyFlagBits_StorageImages;
}

} // namespace igl::vulkan
