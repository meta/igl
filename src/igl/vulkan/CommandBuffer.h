/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <utility>
#include <vector>
#include <igl/CommandBuffer.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanImmediateCommands.h>

namespace igl::vulkan {

struct DescriptorPoolsArenas;
struct ReusableCommandBufferDesc;
class VulkanContext;
class VulkanImage;

/// @brief The layouts that images had before the recording of a slot of a reusable command buffer
/// first changed them (see VulkanContext::journalImageLayout()). The reusable command buffer puts
/// them back if the slot never runs, except those that changed outside of the recording since.
struct ImageLayoutJournal {
  struct Layout {
    const VulkanImage* image = nullptr;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    // set once the layout of the image changes outside of the recording
    bool isChangedElsewhere = false;
  };
  // the command buffer that records the slot now
  VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
  std::vector<Layout> layouts;
};

/// @brief This class implements the igl::ICommandBuffer interface for Vulkan
class CommandBuffer final : public ICommandBuffer,
                            public std::enable_shared_from_this<CommandBuffer> {
 public:
  /// @brief Constructs a CommandBuffer object, acquires a
  /// `VulkanImmediateCommands::CommandBufferWrapper` from the context's VulkanImmediateCommands
  /// object, and stores the CommandBufferDesc structure used to construct the underlying command
  /// buffer.
  CommandBuffer(VulkanContext& ctx, CommandBufferDesc desc);

  /// @brief Constructs a reusable command buffer (see
  /// PlatformDevice::createReusableCommandBuffer()), which records every slot into command buffers
  /// that it allocates from `commandPool`, which it owns: one primary command buffer per slot, or,
  /// if `desc.executeInCommandBuffer` is set, secondary command buffers for the commands outside of
  /// render passes and for the commands inside each render pass. A slot is an index of
  /// VulkanContext::currentSyncIndex().
  CommandBuffer(VulkanContext& ctx,
                const ReusableCommandBufferDesc& desc,
                VkCommandPool commandPool);
  ~CommandBuffer() override;
  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;
  CommandBuffer(CommandBuffer&&) = delete;
  CommandBuffer& operator=(CommandBuffer&&) = delete;

  /// @brief Creates a ComputeCommandEncoder
  std::unique_ptr<IComputeCommandEncoder> createComputeCommandEncoder() override;

  /** @brief Creates a RenderCommandEncoder
   * Before creating a RenderCommandEncoder, this function transitions all images referenced by the
   * `dependencies` parameter to a shader read only layout. It also transitions all images
   * referenced by the `framebuffer` parameter to their optimal layout: color and resolve
   * attachments are transitioned to color attachment optimal layouts; the depth/stencil is
   * transitioned to depth/stencil attachment optimal layout. Returns
   * a RenderCommandEncoder object.
   */
  std::unique_ptr<IRenderCommandEncoder> createRenderCommandEncoder(
      const RenderPassDesc& renderPass,
      const std::shared_ptr<IFramebuffer>& framebuffer,
      const Dependencies& dependencies,
      Result* outResult) override;

  /** @brief Caches the texture passed in to the function for presentation later. If the texture
   * belongs to a swapchain, this function
   * transitions the texture to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR layout. Otherwise it transitions the
   * texture to the VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL layout if the texture's samples is
   * equal to 1 (it's non multi-sampled).
   */
  void present(const std::shared_ptr<ITexture>& surface) const override;

  void pushDebugGroupLabel(const char* label, const Color& color) const override;

  void popDebugGroupLabel() const override;

  void copyBuffer(IBuffer& src,
                  IBuffer& dst,
                  uint64_t srcOffset,
                  uint64_t dstOffset,
                  uint64_t size) override;
  void fillBuffer(IBuffer& buffer, const BufferRange& range, uint8_t value) override;
  /// @brief Records the write with vkCmdUpdateBuffer() (in chunks of up to 64 KB), between
  /// barriers against all earlier and later commands. The buffer must not be a Ring buffer.
  void updateBuffer(IBuffer& buffer, const BufferRange& range, const void* data) override;
  void copyTextureToBuffer(ITexture& src,
                           IBuffer& dst,
                           uint64_t dstOffset,
                           uint32_t level,
                           uint32_t layer) override;

  /// @brief Waits until the command buffer has been executed by the device. For a reusable command
  /// buffer, waits for the last submission that runs one of its recordings.
  void waitUntilCompleted() override;

  /// @brief Not implemented
  void waitUntilScheduled() override;

  /// @brief Returns true if this is a reusable command buffer whose current slot holds a recording
  /// (see PlatformDevice::isRecorded())
  [[nodiscard]] bool isRecorded() const;

  /// @brief Runs the current slot of `reusable` from this command buffer (see
  /// PlatformDevice::executeCommands()): begins the render passes of the slot in this command
  /// buffer, and executes the secondary command buffers of the slot before, inside and after them.
  /// Until this command buffer is submitted, it owns the recordings of `reusable`, and no other
  /// command buffer can execute the slot; if it is destroyed first, the slot never runs from it.
  /// `reusable` must be a Vulkan command buffer, since a command buffer of another backend cannot
  /// be detected; one of another Vulkan device fails with Result::Code::ArgumentInvalid.
  void executeCommands(ICommandBuffer& reusable, Result* IGL_NULLABLE outResult);

  [[nodiscard]] bool isReusable() const {
    return reusable_ != nullptr;
  }

  [[nodiscard]] bool belongsTo(const VulkanContext& ctx) const noexcept {
    return &ctx_ == &ctx;
  }

  /// @brief Returns the command buffer that records the next commands. A reusable command buffer
  /// begins to record its current slot, unless the slot holds a recording: then it returns
  /// VK_NULL_HANDLE, as it does once the recording of the slot failed. The recording fails if
  /// another slot is the current one.
  [[nodiscard]] VkCommandBuffer IGL_NULLABLE getVkCommandBuffer() const {
    return wrapper_ ? wrapper_->cmdBuf : getReusableVkCommandBuffer();
  }

  [[nodiscard]] VulkanImmediateCommands::SubmitHandle getNextSubmitHandle() const {
    return wrapper_ ? wrapper_->handle : handle_;
  }

  /// @brief Returns the descriptor pool arenas of the slot that a reusable command buffer records,
  /// and null otherwise
  [[nodiscard]] DescriptorPoolsArenas* IGL_NULLABLE getDescriptorPoolsArenas() const {
    return reusable_ && reusable_->recordingSlot != kNoSlot
               ? reusable_->arenas[reusable_->recordingSlot].get()
               : nullptr;
  }

  /// @brief Begins the render pass `bi` and returns the command buffer that records its commands:
  /// a new secondary command buffer if this reusable command buffer runs from another command
  /// buffer, which begins the render pass when it executes the slot, and this one otherwise.
  /// Returns VK_NULL_HANDLE if the secondary command buffer cannot be recorded.
  VkCommandBuffer IGL_NULLABLE beginRenderPass(const VkRenderPassBeginInfo& bi);

  /// @brief Ends the render pass and returns the command buffer that records the next commands, or
  /// VK_NULL_HANDLE if the recording of the slot failed, or if a new secondary command buffer
  /// cannot be recorded
  VkCommandBuffer IGL_NULLABLE endRenderPass();

  [[nodiscard]] bool isFromSwapchain() const {
    return isFromSwapchain_;
  }

  [[nodiscard]] const std::shared_ptr<IFramebuffer>& getFramebuffer() const;

  [[nodiscard]] const std::shared_ptr<ITexture>& getPresentedSurface() const;

 private:
  friend class CommandQueue;
  friend class Texture;

  static constexpr uint32_t kNoSlot = UINT32_MAX;

  // A render pass of a slot of a reusable command buffer that runs from another command buffer
  struct ReusableRenderPass {
    VkRenderPassBeginInfo beginInfo = {};
    std::array<VkClearValue, 2 * IGL_COLOR_ATTACHMENTS_MAX + 2> clearValues = {};
    // the secondary command buffer that records the commands of the render pass
    VkCommandBuffer commands = VK_NULL_HANDLE;
  };

  struct ReusableSlot {
    // The primary command buffer of the slot, or the secondary command buffers that record the
    // commands before the first render pass, between two render passes and after the last one
    std::vector<VkCommandBuffer> commands;
    std::vector<ReusableRenderPass> renderPasses;
    // the draws that ICommandBuffer::incrementCurrentDrawCount() counted while the slot recorded,
    // which every run adds to the draw statistics of the queue
    uint32_t numDraws = 0;
    bool isRecorded = false;
    // Why the recording failed: a command buffer of the recording could not be allocated, begun or
    // ended, or another slot was the current one when the recording was encoded into or run
    Result recordingFailure;
    // the last submission that ran the slot
    VulkanImmediateCommands::SubmitHandle lastRun = {};
    // the command buffer that executes the slot and has not been submitted yet
    const CommandBuffer* IGL_NULLABLE executor = nullptr;
  };

  // The recordings of a reusable command buffer. The reusable command buffer owns them with every
  // command buffer that executes one of its slots, until that command buffer is submitted or
  // destroyed. The destructor destroys their command buffers and descriptor sets once the last
  // submission that runs one of them completes.
  struct Reusable {
    Reusable(VulkanContext& ctx, VkCommandPool commandPool, bool executeInCommandBuffer);
    ~Reusable();
    Reusable(const Reusable&) = delete;
    Reusable& operator=(const Reusable&) = delete;
    Reusable(Reusable&&) = delete;
    Reusable& operator=(Reusable&&) = delete;

    VulkanContext& ctx;
    const VkCommandPool commandPool;
    const bool executeInCommandBuffer;
    std::vector<ReusableSlot> slots;
    uint32_t recordingSlot = kNoSlot;
    // getCurrentDrawCount() of the reusable command buffer when `recordingSlot` began to record
    uint32_t recordingDrawCount = 0;
    // the descriptor pool arenas of every slot, whose descriptor sets live as long as its recording
    std::vector<std::shared_ptr<DescriptorPoolsArenas>> arenas;
    // the image layouts that the recording of `recordingSlot` changed, which VulkanContext journals
    // while the slot records
    ImageLayoutJournal journal;
    // set while a render pass of the recording of `recordingSlot` is being encoded
    bool isInRenderPass = false;
    // the last submission that runs one of the recordings
    VulkanImmediateCommands::SubmitHandle lastSubmission = {};
  };

  // A slot of a reusable command buffer that this command buffer executes
  struct ExecutedSlot {
    std::shared_ptr<Reusable> reusable;
    uint32_t slot = 0;
  };

  /// @brief getVkCommandBuffer() of a reusable command buffer
  [[nodiscard]] VkCommandBuffer IGL_NULLABLE getReusableVkCommandBuffer() const;

  /// @brief Allocates and begins a command buffer for the slot that is being recorded: a primary
  /// command buffer, or a secondary one that continues `renderPass` if it is not null, and that
  /// records commands outside of render passes otherwise.
  [[nodiscard]] VkCommandBuffer IGL_NULLABLE
  beginSlotCommandBuffer(VkRenderPass renderPass, VkFramebuffer framebuffer) const;

  /// @brief Ends the command buffer that records the slot, and begins the next one as
  /// beginSlotCommandBuffer() does. On failure, fails the recording of the slot and returns
  /// VK_NULL_HANDLE.
  VkCommandBuffer IGL_NULLABLE nextSlotCommandBuffer(VkRenderPass renderPass,
                                                     VkFramebuffer framebuffer);

  /// @brief Fails the recording of the slot with `failure`, unless it has failed already: the slot
  /// never runs, and the image layouts that the recording changed are put back
  void failRecording(Result failure) const;

  /// @brief Stops journaling the image layouts that the recording of the slot changes
  void stopImageLayoutJournal() const;

  /// @brief Frees the command buffers and the descriptor sets of a slot that has never run, and
  /// empties the slot
  void discardSlot(uint32_t slotIndex) const;

  /// @brief Returns the slot of this reusable command buffer that runs now, or null if it holds no
  /// recording. Ends the recording of the slot if it is being recorded, and waits until the last
  /// run of the slot completes. Fails, and returns null, while a render pass of the recording is
  /// being encoded, if the recording failed, which discards it, if a command buffer that has not
  /// been submitted executes the slot, or if the last run does not complete within
  /// VulkanContextConfig::fenceTimeoutNanoseconds.
  ReusableSlot* IGL_NULLABLE prepareRun(Result& outResult);

  /// @brief Hands the slots that this command buffer executes over to `submission`, which runs
  /// them, or, if it is empty, gives them back without running them. Then releases their
  /// recordings.
  void releaseExecutedSlots(VulkanImmediateCommands::SubmitHandle submission);

  VulkanContext& ctx_;
  // null for a reusable command buffer
  const VulkanImmediateCommands::CommandBufferWrapper* IGL_NULLABLE wrapper_ = nullptr;
  // the command buffer that records the next commands; null for a reusable command buffer that
  // does not record a slot
  mutable VkCommandBuffer cmdBuf_ = VK_NULL_HANDLE;
  // A reusable command buffer has no submission of its own. Every descriptor set of its arenas is
  // allocated with this constant handle, so the arenas never recycle a pool.
  VulkanImmediateCommands::SubmitHandle handle_ = {};
  // null unless the command buffer is reusable
  std::shared_ptr<Reusable> reusable_;
  // the slots of reusable command buffers that this command buffer executes
  std::vector<ExecutedSlot> executedSlots_;
  // the draws of the slots that this command buffer executes, which the queue counts when it
  // submits this command buffer
  uint32_t executedDrawCount_ = 0;
  // set once the queue submits or discards the command buffer
  bool isSubmitted_ = false;
  // set once executeCommands() succeeds: such a command buffer can be destroyed without being
  // submitted
  bool executesSlots_ = false;
  // the count of the command buffers that the queue that created this one has left to submit
  std::shared_ptr<int> numBuffersLeftToSubmit_;
  // was present() called with a swapchain image?
  mutable bool isFromSwapchain_ = false;

  std::shared_ptr<IFramebuffer> framebuffer_;
  mutable std::shared_ptr<ITexture> presentedSurface_;

  VulkanImmediateCommands::SubmitHandle lastSubmitHandle_ = {};

  // The slot of the Timer in `desc.timer` that measures this command buffer
  uint32_t timerSlot_ = 0;
};

} // namespace igl::vulkan
