/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string_view>
#include <igl/CommandBuffer.h>

namespace iglu::command_buffer_allocator {

/**
 * ICommandBuffer Allocator is am interface to allocate Command Buffers and submit them all at once.
 */
class ICommandBufferAllocator {
 public:
  ICommandBufferAllocator() = default;
  virtual ~ICommandBufferAllocator() = default;

  struct CommandBufferScopeConfig {
    bool present = false;
    std::shared_ptr<igl::ITexture> presentTexture = nullptr;
    bool waitUntilScheduled = false;
    bool waitUntilCompleted = false;
    // Non-owning; consumed when the command buffer is created. Must reference storage that
    // outlives the enclosing commandBufferScope() call. Empty means no debug name.
    std::string_view debugName;
  };

  struct CommandBufferScope {
   public:
    ~CommandBufferScope() noexcept;

    static void* operator new(size_t) = delete; // stack allocation only

    [[nodiscard]] igl::ICommandBuffer& commandBuffer() noexcept;
    [[nodiscard]] const igl::ICommandBuffer& commandBuffer() const noexcept;

   private:
    friend class ICommandBufferAllocator;
    CommandBufferScope(ICommandBufferAllocator& allocator,
                       igl::ICommandBuffer& commandBuffer,
                       bool shouldFinalizeCommandBuffer) noexcept;

    ICommandBufferAllocator& allocator_;
    igl::ICommandBuffer& commandBuffer_;
    bool shouldFinalizeCommandBuffer_ = false;
  };
  virtual void createCommandBuffer(std::string_view debugName) noexcept = 0;
  [[nodiscard]] virtual CommandBufferScope commandBufferScope() noexcept = 0;
  [[nodiscard]] virtual CommandBufferScope commandBufferScope(
      ICommandBufferAllocator& allocator,
      igl::ICommandBuffer& commandBuffer,
      bool shouldFinalizeCommandBuffer) noexcept;
  [[nodiscard]] virtual CommandBufferScope commandBufferScope(
      CommandBufferScopeConfig config) noexcept = 0;
  virtual void finalizeCommandBuffer() noexcept = 0;
};

[[nodiscard]] inline ICommandBufferAllocator::CommandBufferScope
ICommandBufferAllocator::commandBufferScope(ICommandBufferAllocator& allocator,
                                            igl::ICommandBuffer& commandBuffer,
                                            bool shouldFinalizeCommandBuffer) noexcept {
  return {allocator, commandBuffer, shouldFinalizeCommandBuffer};
}

} // namespace iglu::command_buffer_allocator
