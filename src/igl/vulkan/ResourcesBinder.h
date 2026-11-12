/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <type_traits>
#include <igl/Common.h>
#include <igl/Framebuffer.h>
#include <igl/Texture.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/VulkanImmediateCommands.h>

namespace igl::vulkan {

namespace util {
struct SpvModuleInfo;
} // namespace util

class Buffer;
struct DescriptorPoolsArenas;
class PipelineState;
class SamplerState;
class Texture;

struct BindingsBuffers {
  VkDescriptorBufferInfo buffers[IGL_UNIFORM_BLOCKS_BINDING_MAX] = {};
  VkDeviceAddress addresses[IGL_UNIFORM_BLOCKS_BINDING_MAX] = {};
};

struct BindingsTextures {
  VkImageView textures[IGL_TEXTURE_SAMPLERS_MAX] = {};
  VkSampler samplers[IGL_TEXTURE_SAMPLERS_MAX] = {};
};

struct BindingsStorageImages {
  VkImageView images[IGL_STORAGE_IMAGES_MAX] = {};
  VkImageView inputAttachments[IGL_COLOR_ATTACHMENTS_MAX] = {};
};

static_assert(std::is_trivially_copyable_v<BindingsBuffers>);
static_assert(std::is_trivially_copyable_v<BindingsTextures>);
static_assert(std::is_trivially_copyable_v<BindingsStorageImages>);

/** @brief Stores uniform and storage buffer bindings, as well as bindings for textures and sampler
 * states for Vulkan. This class maintains vectors for each type of shader resource available in IGL
 * and records the association between binding locations (indices) and the Vulkan objects, while
 * performing specific checks for each type of resource when they are bound. The associations
 * between indices and resources is kept locally and does not affect the GPU until
 * `updateBindings()` is called. This class also records which resource types need to be updated
 * when `updateBindings()` is called and provides a convenience function to update the descriptor
 * sets on the context for all resource types. It only performs the update for a resource type that
 * has been modified after the last call to update the bindings. An instance of this class is bound
 * to one bind point only (VkPipelineBindPoint), which is VK_PIPELINE_BIND_POINT_GRAPHICS by
 * default.
 */
class ResourcesBinder final {
 public:
  ResourcesBinder(const CommandBuffer* commandBuffer,
                  VulkanContext& ctx,
                  VkPipelineBindPoint bindPoint);

  /// @brief Binds a uniform buffer with an offset to index equal to `index`
  void bindBuffer(uint32_t index, Buffer* buffer, size_t bufferOffset, size_t bufferSize);

  /// @brief Binds a sampler state to index equal to `index`
  void bindSamplerState(uint32_t index, SamplerState* samplerState);

  /// @brief Binds a texture to index equal to `index`
  void bindTexture(uint32_t index, Texture* tex);
  void bindStorageImage(uint32_t index, Texture* tex);

  /// @brief Binds a texture that a fragment shader reads as input attachment `index` (a
  /// `subpassInput` declared with `input_attachment_index = index`). The descriptor references the
  /// image view of `mipLevel` and `layer` that the framebuffers of `mode` use, because the render
  /// pass has to read the attachment of the framebuffer: a 2D array view of the layers from
  /// `layer` on for FramebufferMode::Stereo, where the shader reads the layer of the view that it
  /// renders. A null texture unbinds the input attachment.
  void bindInputAttachment(uint32_t index,
                           Texture* IGL_NULLABLE texture,
                           uint32_t mipLevel,
                           uint32_t layer,
                           FramebufferMode mode = FramebufferMode::Mono);

  /// @brief Convenience function that updates all bindings in the context for all resource types
  /// that have been modified since the last time this function was called
  void updateBindings(VkPipelineLayout layout, const PipelineState& state);

  /// @brief If the pipeline passed in as a parameter is different than the last pipeline bound
  /// through this class, binds it and cache it as the last pipeline bound. Does nothing otherwise.
  /// The descriptor sets are updated at the next draw or dispatch for a new pipeline, unless
  /// `invalidateBoundPipeline()` asks to keep them. A null `info` unbinds the last pipeline.
  void bindPipeline(VkPipeline pipeline, const util::SpvModuleInfo* info);

  /// @brief Makes the next `bindPipeline()` bind its pipeline even if it is the one that was bound
  /// last. If the shaders of the next pipeline are the ones of the pipeline that was bound last,
  /// like the pipelines that a render pipeline state has for the subpasses of a render pass, the
  /// descriptor sets that are bound stay valid and are not updated, including the sets of bind
  /// groups.
  void invalidateBoundPipeline();

  /// @brief Forgets everything that this binder bound to the command buffer. The next
  /// `bindPipeline()` binds its pipeline even if it is the one that was bound last, and every
  /// resource type is dirty, so the next `updateBindings()` writes and binds all the descriptor
  /// sets. A caller that binds the descriptor set of a resource type itself, like the encoder does
  /// for bind groups, clears the dirty flag of that type afterwards. The binder then leaves that
  /// set alone, unless a resource of the type changes or a pipeline with other shaders needs it. A
  /// multiview render pass needs this at every subpass, because it leaves all the state of the
  /// command buffer undefined.
  void invalidateAll();

 private:
  void updateBindingsByDescriptorSet(VkPipelineLayout layout, const PipelineState& state);
  void updateBindingsByDescriptorBuffer(VkPipelineLayout layout, const PipelineState& state);

 private:
  friend class VulkanContext;
  friend class RenderCommandEncoder;

  [[nodiscard]] bool isGraphics() const {
    return bindPoint_ == VK_PIPELINE_BIND_POINT_GRAPHICS;
  }

  /*
   * @brief Bitwise flags for dirty descriptor sets (per each supported resource type)
   */
  enum DirtyFlagBits : uint8_t { // NOLINT
    DirtyFlagBits_Textures = 1 << 0, // NOLINT
    DirtyFlagBits_Buffers = 1 << 1, // NOLINT
    DirtyFlagBits_StorageImages = 1 << 2, // NOLINT
  };

 private:
  VulkanContext& ctx_;
  VkCommandBuffer cmdBuffer_ = VK_NULL_HANDLE;
  VkPipeline lastPipelineBound_ = VK_NULL_HANDLE;
  // the shaders of the last pipeline bound, and whether the next pipeline keeps the descriptor sets
  // if it has the same ones: see invalidateBoundPipeline()
  const util::SpvModuleInfo* lastInfo_ = nullptr;
  bool keepDescriptorsOnNextBind_ = false;
  uint32_t isDirtyFlags_ =
      DirtyFlagBits_Textures | DirtyFlagBits_Buffers | DirtyFlagBits_StorageImages;
  BindingsTextures bindingsTextures_;
  BindingsBuffers bindingsBuffers_;
  BindingsStorageImages bindingsStorageImages_;
  VkPipelineBindPoint bindPoint_ = VK_PIPELINE_BIND_POINT_GRAPHICS;
  VulkanImmediateCommands::SubmitHandle nextSubmitHandle_ = {};
  // the arenas of the descriptor sets: the context's if null, otherwise a reusable command buffer's
  DescriptorPoolsArenas* IGL_NULLABLE arenas_ = nullptr;
};

} // namespace igl::vulkan
