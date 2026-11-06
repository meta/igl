/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <memory>
#include <vector>
#include <igl/Buffer.h>
#include <igl/CommandEncoder.h>
#include <igl/Common.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/RenderPipelineState.h>
#include <igl/vulkan/ResourcesBinder.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanImmediateCommands.h>

namespace igl::vulkan {

/// @brief This class implements the igl::IRenderCommandEncoder interface for Vulkan
class RenderCommandEncoder : public IRenderCommandEncoder {
 public:
  static std::unique_ptr<RenderCommandEncoder> create(
      const std::shared_ptr<CommandBuffer>& commandBuffer,
      VulkanContext& ctx,
      const RenderPassDesc& renderPass,
      const std::shared_ptr<IFramebuffer>& framebuffer,
      const Dependencies& dependencies,
      Result* outResult);

  RenderCommandEncoder(const RenderCommandEncoder&) = delete;
  RenderCommandEncoder(RenderCommandEncoder&&) = delete;
  RenderCommandEncoder& operator=(const RenderCommandEncoder&) = delete;
  RenderCommandEncoder& operator=(RenderCommandEncoder&&) = delete;

  ~RenderCommandEncoder() override {
    IGL_DEBUG_ASSERT(!isEncoding_); // did you forget to call endEncoding()?
    endEncoding();
  }

  /// @brief Ends encoding for render commands and transitions the layouts of all images bound to
  /// this encoder back to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`. Also transitions all
  /// dependent textures to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`. If `nextSubpass()` was not
  /// called for every subpass, it asserts in debug builds and advances to the last subpass first.
  void endEncoding() override;

  /// @brief Advances to the next subpass of the render pass. The pipelines that are used from now
  /// on are the variants of the render pipeline states for the new subpass, and the input
  /// attachments are the ones that the new subpass reads. All other bound state is kept. A
  /// multiview render pass leaves the state of the command buffer undefined at the start of a
  /// subpass, so the next draw call records the pipeline, the descriptor sets, the vertex and index
  /// buffers, the push constants and the dynamic state again.
  void nextSubpass() override;

  void pushDebugGroupLabel(const char* label, const Color& color) const override;
  void insertDebugEventLabel(const char* label, const Color& color) const override;
  void popDebugGroupLabel() const override;

  /// @brief Sets the viewport size specified in `viewport`. This function flips the viewport in the
  /// y-direction but retains the same winding as in OpenGL
  void bindViewport(const Viewport& viewport) override;
  void bindScissorRect(const ScissorRect& rect) override;

  void bindRenderPipelineState(const std::shared_ptr<IRenderPipelineState>& pipelineState) override;
  void bindDepthStencilState(const std::shared_ptr<IDepthStencilState>& depthStencilState) override;

  void bindBuffer(uint32_t index,
                  uint8_t target,
                  IBuffer* buffer,
                  size_t bufferOffset,
                  size_t bufferSize) override;
  void bindBuffer(uint32_t index, IBuffer* buffer, size_t bufferOffset, size_t bufferSize) override;
  void bindVertexBuffer(uint32_t index,
                        IBuffer& buffer,
                        size_t bufferOffset,
                        size_t attributeStride) override;
  void bindIndexBuffer(IBuffer& buffer, IndexFormat format, size_t bufferOffset) override;

  /// @brief Not implemented
  void bindBytes(size_t index, uint8_t target, const void* data, size_t length) override;

  /// @brief Binds push constants pointed by `data` with `length` bytes starting at `offset`.
  /// `length` must be a multiple of 4.
  void bindPushConstants(const void* data, size_t length, size_t offset) override;
  void bindSamplerState(size_t index, uint8_t target, ISamplerState* samplerState) override;

  void bindTexture(size_t index, uint8_t target, ITexture* texture) override;
  void bindTexture(size_t index, ITexture* texture) override;

  /// @brief This is only for backends that MUST use single uniforms in some situations. Do not
  /// implement!
  void bindUniform(const UniformDesc& uniformDesc, const void* data) override;

  void bindBindGroup(BindGroupTextureHandle handle) override;
  void bindBindGroup(BindGroupBufferHandle handle,
                     uint32_t numDynamicOffsets = 0,
                     const uint32_t* dynamicOffsets = nullptr) override;

  void draw(size_t vertexCount,
            uint32_t instanceCount,
            uint32_t firstVertex,
            uint32_t baseInstance) override;
  void drawIndexed(size_t indexCount,
                   uint32_t instanceCount,
                   uint32_t firstIndex,
                   int32_t vertexOffset,
                   uint32_t baseInstance) override;
  void drawMeshTasks(const Dimensions& threadgroupsPerGrid,
                     const Dimensions& threadsPerTaskThreadgroup,
                     const Dimensions& threadsPerMeshThreadgroup) override;
  void multiDrawIndirect(IBuffer& indirectBuffer,
                         size_t indirectBufferOffset,
                         uint32_t drawCount,
                         uint32_t stride = 0) override;
  void multiDrawIndexedIndirect(IBuffer& indirectBuffer,
                                size_t indirectBufferOffset,
                                uint32_t drawCount,
                                uint32_t stride = 0) override;

  void setStencilReferenceValue(uint32_t value) override;
  void setBlendColor(const Color& color) override;
  void setCullMode(CullMode cullMode) override;
  void setDepthBias(float depthBias, float slopeScale, float clamp) override;
  void setFrontFacingWinding(WindingMode frontFaceWinding) override;

  [[nodiscard]] VkCommandBuffer getVkCommandBuffer() const {
    return cmdBuffer_;
  }

  ResourcesBinder& binder() {
    return binder_;
  }

  /// @brief Enables or disables the draw call count. If enabled, it will increment the draw call,
  /// otherwise it won't. This is used to disable the draw call count when we are doing auxiliary
  /// draw calls.
  bool setDrawCallCountEnabled(bool value);

  void blitColorImage(const VulkanImage& srcImage,
                      const VulkanImage& destImage,
                      const TextureRangeDesc& srcRange,
                      const TextureRangeDesc& destRange);

 private:
  RenderCommandEncoder(const std::shared_ptr<CommandBuffer>& commandBuffer, VulkanContext& ctx);

  void applyPipelineRasterizationDynamicState();

  /// @brief Ensures that the vertex buffers are bound by performing checks. If the function doesn't
  /// assert at some point, the vertex buffer(s) is bound correctly.
  void ensureVertexBuffers();
  bool flushDynamicState();
  void flushDynamicDepthStencilState();

  void initialize(const RenderPassDesc& renderPass,
                  const std::shared_ptr<IFramebuffer>& framebuffer,
                  const Dependencies& dependencies,
                  Result& outResult);
  void processDependencies(const Dependencies& dependencies);

  /// @brief Binds the color attachments that the current subpass reads as input attachments, and
  /// unbinds the other ones
  void bindSubpassInputAttachments();

  /// @brief The state that the encoder recorded into the command buffer, which the command buffer
  /// loses at the start of every subpass of a multiview render pass. Only the encoders of the
  /// multiview render passes with subpasses keep it.
  struct ReplayState {
    struct VertexBuffer {
      VkBuffer buffer = VK_NULL_HANDLE;
      VkDeviceSize offset = 0;
      VkDeviceSize size = 0; // used with `stride`
      VkDeviceSize stride = 0; // 0 if the vertex buffer was bound without a stride
    };

    VkViewport viewport = {};
    VkRect2D scissor = {};

    std::array<VertexBuffer, IGL_BUFFER_BINDINGS_MAX> vertexBuffers = {};
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceSize indexBufferOffset = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;

    // The bytes that were pushed since the render pass began: [pushConstantsBegin,
    // pushConstantsEnd) of `pushConstants`, which has as many bytes as
    // VkPhysicalDeviceLimits::maxPushConstantsSize. The bytes that no push wrote are 0.
    std::vector<uint8_t> pushConstants;
    uint32_t pushConstantsBegin = 0;
    uint32_t pushConstantsEnd = 0;
    // The push-constant stages of the pipeline layout that the push constants were last replayed
    // for in the current subpass, or 0. IGL pipeline layouts with the same stages have the same
    // push-constant range, so they are compatible for push constants.
    VkShaderStageFlags replayedPushConstantStages = 0;

    // The dynamic state that was set, if its `has` flag is set
    bool hasCullMode = false;
    VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
    bool hasFrontFace = false;
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool hasStencilMasks = false;
    // the front face at index 0, the back face at index 1
    std::array<uint32_t, 2> stencilCompareMasks = {};
    std::array<uint32_t, 2> stencilWriteMasks = {};
    bool hasStencilReference = false;
    uint32_t stencilReference = 0;
    bool hasBlendConstants = false;
    std::array<float, 4> blendConstants = {};
    bool hasDepthBias = false;
    float depthBias = 0.0f;
    float depthBiasClamp = 0.0f;
    float depthBiasSlopeFactor = 0.0f;

    // The descriptor set of a bind group. It is the raw descriptor set and not the handle of the
    // bind group, because the app can destroy the bind group before the next subpass. That frees
    // the handle, but not the descriptor set: its pool is destroyed after the command buffer
    // completes.
    struct BoundBindGroup {
      VkDescriptorSet dset = VK_NULL_HANDLE; // null if the binder provides the descriptor set
      uint32_t usageMask = 0; // the bindings that the descriptor set has
    };

    // The bind groups that provide the descriptor sets that are bound now, and the dynamic offsets
    // of the buffer one
    BoundBindGroup boundTextureGroup;
    BoundBindGroup boundBufferGroup;
    uint32_t numBoundDynamicOffsets = 0;
    std::array<uint32_t, IGL_UNIFORM_BLOCKS_BINDING_MAX> boundDynamicOffsets = {};
  };

  /// @brief Makes the next draw call record again the state that the command buffer lost at the
  /// start of a subpass of a multiview render pass: the binder forgets the bound pipeline and
  /// descriptor sets, and replayState() records the rest.
  void invalidateStateForSubpass();

  /// @brief Records the state in `replay_` again, except for the pipeline and the descriptor sets
  void replayState();

  /// @brief Pushes the push constants in `replay_` again with the layout of the bound pipeline,
  /// unless that layout has no push constants, or the current subpass already replayed them for a
  /// layout that is compatible for push constants. Needs the pipeline to be bound.
  void replayPushConstants();

  /// @brief Binds the descriptor sets of the bind groups that provided them in the last subpass,
  /// unless the caller binds new bind groups, the binder provides them, or the bound pipeline
  /// needs other resources than the bind groups have. Needs the pipeline to be bound.
  void replayBindGroups();

#if IGL_DEBUG_ABORT_ENABLED
  /// @brief Asserts that the bound render pipeline state fits the current subpass
  void validatePipelineForSubpass() const;
#endif // IGL_DEBUG_ABORT_ENABLED

 private:
  VulkanContext& ctx_;
  VkCommandBuffer cmdBuffer_ = VK_NULL_HANDLE;
  bool isEncoding_ = false;
  bool hasDepthAttachment_ = false;
  std::shared_ptr<IFramebuffer> framebuffer_;

  ResourcesBinder binder_;

  RenderPipelineDynamicState dynamicState_;

  // The subpasses of the render pass: the first `numSubpasses_` entries are valid. A render pass
  // without subpasses has 1 subpass that does not read any input attachment.
  std::array<RenderPassDesc::SubpassDesc, IGL_SUBPASSES_MAX> subpasses_ = {};
  uint32_t numSubpasses_ = 1;
  uint32_t currentSubpass_ = 0;
  uint32_t numColorAttachments_ = 0;
  // The mip level, the layer and the mode of the attachments of the framebuffer
  uint32_t framebufferMipLevel_ = 0;
  uint32_t framebufferLayer_ = 0;
  FramebufferMode framebufferMode_ = FramebufferMode::Mono;

  // Null unless the render pass is a multiview one with subpasses
  std::unique_ptr<ReplayState> replay_;
  // Whether the next draw call has to record the state in `replay_` again
  bool isReplayPending_ = false;

  /* Used to increment the draw call count. Should either be 0 or 1
   *  0: When draw call count is disabled during auxiliary draw calls
   *  1: All other times */
  uint32_t drawCallCountEnabled_ = 1u;

  bool isVertexBufferBound_[IGL_BUFFER_BINDINGS_MAX] = {};

  Dependencies dependencies_ = {};

  const RenderPipelineState* rps_ = nullptr;
  BindGroupTextureHandle pendingBindGroupTexture_ = {};
  BindGroupBufferHandle pendingBindGroupBuffer_ = {};
  uint32_t numDynamicOffsets_ = 0;
  uint32_t dynamicOffsets_[IGL_UNIFORM_BLOCKS_BINDING_MAX] = {};

  std::shared_ptr<ITimestampQueries> timestampQueries_;
  uint32_t timestampQuerySlotIndex_ = 0;
};

} // namespace igl::vulkan
