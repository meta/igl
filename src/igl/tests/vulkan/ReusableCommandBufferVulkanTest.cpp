/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <optional>
#include <string_view>
#include <vector>
#include <igl/CommandBuffer.h>
#include <igl/ComputeCommandEncoder.h>
#include <igl/ComputePipelineState.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/ShaderCreator.h>
#include <igl/Texture.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanFunctionTable.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanSemaphore.h>
#include <igl/vulkan/VulkanTexture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

namespace {

constexpr uint32_t kSize = 4;
// RGBA_UNorm8 pixels read back as 0xAABBGGRR
constexpr uint32_t kBlue = 0xFFFF0000;
constexpr uint32_t kMagenta = 0xFFFF00FF;

using Pixels = std::array<uint32_t, kSize * kSize>;

constexpr Pixels fillPixels(uint32_t pixel) {
  Pixels pixels = {};
  pixels.fill(pixel);
  return pixels;
}

// The pixels of texture B after the passes
constexpr Pixels bluePixels() {
  return fillPixels(kBlue);
}

constexpr const char* kVertexShader = R"(#version 460
layout(location = 0) out vec2 uv;
void main() {
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

// renders the color of a uniform buffer
constexpr const char* kColorShader = R"(#version 460
layout(set = 1, binding = 0) uniform Color {
  vec4 color;
};
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = color;
})";

// samples a texture and swaps its red and blue channels
constexpr const char* kSwizzleShader = R"(#version 460
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = texture(tex, uv).bgra;
})";

// adds the colors of two textures
[[maybe_unused]] constexpr const char* kAddShader = R"(#version 460
layout(set = 0, binding = 0) uniform sampler2D tex0;
layout(set = 0, binding = 1) uniform sampler2D tex1;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(texture(tex0, uv).rgb + texture(tex1, uv).rgb, 1.0);
})";

// stores the color of a uniform buffer into every pixel of an image
constexpr const char* kStoreShader = R"(#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 1, binding = 0) uniform Color {
  vec4 color;
};
layout(set = 2, binding = 0, rgba8) uniform writeonly image2D image;
void main() {
  imageStore(image, ivec2(gl_GlobalInvocationID.xy), color);
})";

// copies an image into another
constexpr const char* kCopyImageShader = R"(#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 2, binding = 0, rgba8) uniform readonly image2D src;
layout(set = 2, binding = 1, rgba8) uniform writeonly image2D dst;
void main() {
  const ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(dst, xy, imageLoad(src, xy));
})";

// stores the sum of a texture and the color of a uniform buffer into an image
constexpr const char* kAddColorShader = R"(#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(set = 1, binding = 0) uniform Color {
  vec4 color;
};
layout(set = 2, binding = 0, rgba8) uniform writeonly image2D image;
void main() {
  const ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(image, xy, texelFetch(tex, xy, 0) + color);
})";

// Records the global memory barriers and the dispatches that command buffers record while it is in
// scope, in their order, and puts the original functions back when it goes out of scope. There is
// one MemoryBarrierRecorder at a time.
class MemoryBarrierRecorder {
 public:
  explicit MemoryBarrierRecorder(vulkan::VulkanContext& ctx) :
    table_(const_cast<VulkanFunctionTable&>(ctx.vf_)) {
    state() = {
        .pipelineBarrier = table_.vkCmdPipelineBarrier,
        .dispatch = table_.vkCmdDispatch,
    };
    table_.vkCmdPipelineBarrier = pipelineBarrier;
    table_.vkCmdDispatch = dispatch;
  }

  ~MemoryBarrierRecorder() {
    table_.vkCmdPipelineBarrier = state().pipelineBarrier;
    table_.vkCmdDispatch = state().dispatch;
  }

  MemoryBarrierRecorder(const MemoryBarrierRecorder&) = delete;
  MemoryBarrierRecorder& operator=(const MemoryBarrierRecorder&) = delete;

  // Returns true if a global memory barrier between the first two dispatches makes the writes of
  // the first one's compute shader visible to the reads of the second one's, whatever the image
  [[nodiscard]] bool coversComputeShaderWritesBetweenDispatches() const {
    bool isAfterFirstDispatch = false;
    for (uint32_t i = 0; i != state().numCommands; ++i) {
      const Command& command = state().commands[i];
      if (command.isDispatch) {
        if (isAfterFirstDispatch) {
          return false;
        }
        isAfterFirstDispatch = true;
      } else if (isAfterFirstDispatch &&
                 (command.srcStageMask & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0 &&
                 (command.dstStageMask & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0 &&
                 (command.srcAccessMask & VK_ACCESS_SHADER_WRITE_BIT) != 0 &&
                 (command.dstAccessMask & VK_ACCESS_SHADER_READ_BIT) != 0) {
        return true;
      }
    }
    return false;
  }

 private:
  // a dispatch, or a global memory barrier
  struct Command {
    bool isDispatch = false;
    VkPipelineStageFlags srcStageMask = 0;
    VkPipelineStageFlags dstStageMask = 0;
    VkAccessFlags srcAccessMask = 0;
    VkAccessFlags dstAccessMask = 0;
  };

  struct State {
    PFN_vkCmdPipelineBarrier pipelineBarrier = nullptr;
    PFN_vkCmdDispatch dispatch = nullptr;
    std::array<Command, 64> commands = {};
    uint32_t numCommands = 0;
  };

  static State& state() {
    static State state;
    return state;
  }

  static void record(const Command& command) {
    if (state().numCommands != state().commands.size()) {
      state().commands[state().numCommands++] = command;
    }
  }

  static VKAPI_ATTR void VKAPI_CALL
  pipelineBarrier(VkCommandBuffer cmdBuf,
                  VkPipelineStageFlags srcStageMask,
                  VkPipelineStageFlags dstStageMask,
                  VkDependencyFlags dependencyFlags,
                  uint32_t memoryBarrierCount,
                  const VkMemoryBarrier* memoryBarriers,
                  uint32_t bufferMemoryBarrierCount,
                  const VkBufferMemoryBarrier* bufferMemoryBarriers,
                  uint32_t imageMemoryBarrierCount,
                  const VkImageMemoryBarrier* imageMemoryBarriers) {
    for (uint32_t i = 0; i != memoryBarrierCount; ++i) {
      record({
          .srcStageMask = srcStageMask,
          .dstStageMask = dstStageMask,
          .srcAccessMask = memoryBarriers[i].srcAccessMask,
          .dstAccessMask = memoryBarriers[i].dstAccessMask,
      });
    }
    if (state().pipelineBarrier) {
      state().pipelineBarrier(cmdBuf,
                              srcStageMask,
                              dstStageMask,
                              dependencyFlags,
                              memoryBarrierCount,
                              memoryBarriers,
                              bufferMemoryBarrierCount,
                              bufferMemoryBarriers,
                              imageMemoryBarrierCount,
                              imageMemoryBarriers);
    }
  }

  static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer cmdBuf,
                                             uint32_t groupCountX,
                                             uint32_t groupCountY,
                                             uint32_t groupCountZ) {
    record({.isDispatch = true});
    if (state().dispatch) {
      state().dispatch(cmdBuf, groupCountX, groupCountY, groupCountZ);
    }
  }

  VulkanFunctionTable& table_;
};

// Makes the calls of one function fail for the command buffers that vkAllocateCommandBuffers()
// allocates while it is in scope, which are those of reusable command buffers: every call fails
// once `numCalls` more calls have succeeded. Puts the original functions back when it goes out of
// scope. There is one CommandBufferFailure at a time.
class CommandBufferFailure {
 public:
  enum class Call { Allocate, Begin, End };

  CommandBufferFailure(vulkan::VulkanContext& ctx, Call call, uint32_t numCalls) :
    // VulkanContext exposes its function table as a const reference to a table that it owns
    table_(const_cast<VulkanFunctionTable&>(ctx.vf_)) {
    state() = {
        .allocate = table_.vkAllocateCommandBuffers,
        .begin = table_.vkBeginCommandBuffer,
        .end = table_.vkEndCommandBuffer,
        .call = call,
        .numCallsLeft = numCalls,
        .isInScope = true,
    };
    table_.vkAllocateCommandBuffers = allocate;
    table_.vkBeginCommandBuffer = begin;
    table_.vkEndCommandBuffer = end;
  }

  ~CommandBufferFailure() {
    table_.vkAllocateCommandBuffers = state().allocate;
    table_.vkBeginCommandBuffer = state().begin;
    table_.vkEndCommandBuffer = state().end;
    state().isInScope = false;
  }

  CommandBufferFailure(const CommandBufferFailure&) = delete;
  CommandBufferFailure& operator=(const CommandBufferFailure&) = delete;

  [[nodiscard]] static bool isInScope() {
    return state().isInScope;
  }

 private:
  struct State {
    PFN_vkAllocateCommandBuffers allocate = nullptr;
    PFN_vkBeginCommandBuffer begin = nullptr;
    PFN_vkEndCommandBuffer end = nullptr;
    Call call = Call::Allocate;
    uint32_t numCallsLeft = 0;
    std::array<VkCommandBuffer, 64> allocated = {};
    uint32_t numAllocated = 0;
    bool isInScope = false;
  };

  static State& state() {
    static State state;
    return state;
  }

  // Returns true if this call of `call` fails
  static bool fails(Call call) {
    if (state().call != call) {
      return false;
    }
    if (state().numCallsLeft == 0) {
      return true;
    }
    --state().numCallsLeft;
    return false;
  }

  static bool isAllocated(VkCommandBuffer cmdBuf) {
    const auto allocated = state().allocated.begin();
    const auto end = allocated + state().numAllocated;
    return std::find(allocated, end, cmdBuf) != end;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice device,
                                                 const VkCommandBufferAllocateInfo* info,
                                                 VkCommandBuffer* cmdBufs) {
    if (fails(Call::Allocate)) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    const VkResult result = state().allocate(device, info, cmdBufs);
    for (uint32_t i = 0; result == VK_SUCCESS && i != info->commandBufferCount &&
                         state().numAllocated != state().allocated.size();
         ++i) {
      state().allocated[state().numAllocated++] = cmdBufs[i];
    }
    return result;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL begin(VkCommandBuffer cmdBuf,
                                              const VkCommandBufferBeginInfo* info) {
    if (isAllocated(cmdBuf) && fails(Call::Begin)) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return state().begin(cmdBuf, info);
  }

  static VKAPI_ATTR VkResult VKAPI_CALL end(VkCommandBuffer cmdBuf) {
    if (isAllocated(cmdBuf) && fails(Call::End)) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return state().end(cmdBuf);
  }

  VulkanFunctionTable& table_;
};

// The debug aborts that the current test expects in a function whose name contains `func`
struct ExpectedDebugAborts {
  const char* IGL_NULLABLE func = nullptr;
  uint32_t count = 0;
};

ExpectedDebugAborts& expectedDebugAborts() {
  static ExpectedDebugAborts expected;
  return expected;
}

// Fails the test in which IGL reports a debug abort, unless the test expects it, or the abort is a
// failed Vulkan call that a CommandBufferFailure injects
void failOnDebugAbort(const char* IGL_NONNULL /*category*/,
                      const char* IGL_NONNULL reason,
                      const char* IGL_NONNULL file,
                      const char* IGL_NONNULL func,
                      int line,
                      const char* IGL_NONNULL format,
                      va_list ap) {
  if (CommandBufferFailure::isInScope() &&
      std::string_view(format).starts_with("Vulkan API call failed")) {
    return;
  }
  ExpectedDebugAborts& expected = expectedDebugAborts();
  if (expected.count != 0 && expected.func != nullptr &&
      std::string_view(func).find(expected.func) != std::string_view::npos) {
    --expected.count;
    return;
  }
  std::array<char, 512> message = {};
  (void)std::vsnprintf(message.data(), message.size(), format, ap);
  ADD_FAILURE() << reason << " in " << func << " (" << file << ":" << line
                << "): " << message.data();
}

// Counts the command pools and the descriptor pools that are destroyed while it is in scope, and
// puts the original functions back when it goes out of scope. There is one PoolDestructions at a
// time.
class PoolDestructions {
 public:
  explicit PoolDestructions(vulkan::VulkanContext& ctx) :
    table_(const_cast<VulkanFunctionTable&>(ctx.vf_)) {
    state() = {
        .destroyCommandPool = table_.vkDestroyCommandPool,
        .destroyDescriptorPool = table_.vkDestroyDescriptorPool,
    };
    table_.vkDestroyCommandPool = destroyCommandPool;
    table_.vkDestroyDescriptorPool = destroyDescriptorPool;
  }

  ~PoolDestructions() {
    table_.vkDestroyCommandPool = state().destroyCommandPool;
    table_.vkDestroyDescriptorPool = state().destroyDescriptorPool;
  }

  PoolDestructions(const PoolDestructions&) = delete;
  PoolDestructions& operator=(const PoolDestructions&) = delete;

  [[nodiscard]] uint32_t numCommandPools() const {
    return state().numCommandPools;
  }

  [[nodiscard]] uint32_t numDescriptorPools() const {
    return state().numDescriptorPools;
  }

 private:
  struct State {
    PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
    PFN_vkDestroyDescriptorPool destroyDescriptorPool = nullptr;
    uint32_t numCommandPools = 0;
    uint32_t numDescriptorPools = 0;
  };

  static State& state() {
    static State state;
    return state;
  }

  static VKAPI_ATTR void VKAPI_CALL destroyCommandPool(VkDevice device,
                                                       VkCommandPool commandPool,
                                                       const VkAllocationCallbacks* allocator) {
    ++state().numCommandPools;
    if (state().destroyCommandPool) {
      state().destroyCommandPool(device, commandPool, allocator);
    }
  }

  static VKAPI_ATTR void VKAPI_CALL destroyDescriptorPool(VkDevice device,
                                                          VkDescriptorPool descriptorPool,
                                                          const VkAllocationCallbacks* allocator) {
    ++state().numDescriptorPools;
    if (state().destroyDescriptorPool) {
      state().destroyDescriptorPool(device, descriptorPool, allocator);
    }
  }

  VulkanFunctionTable& table_;
};

// Holds back the GPU work that is submitted after it until open() is called, without making the
// host wait: a submission that waits for a timeline semaphore, which open() signals, runs a command
// that every later IGL command buffer waits for when it begins with a full pipeline barrier (see
// ReusableCommandBufferPendingRunsVulkanTest). Opens itself, and waits until the device is idle,
// when it goes out of scope.
class HostGate {
 public:
  HostGate(IDevice& device, vulkan::VulkanContext& ctx) :
    ctx_(ctx), semaphore_(ctx.vf_, ctx.getVkDevice(), 0, false, "Host gate") {
    if (!ctx_.features().has_VK_KHR_timeline_semaphore || !ctx_.vf_.vkSignalSemaphoreKHR) {
      return;
    }
    Result ret;
    buffer_ = device.createBuffer(
        BufferDesc{.type = BufferDesc::BufferTypeBits::Storage, .length = sizeof(uint32_t)}, &ret);
    if (!ret.isOk() || !buffer_) {
      return;
    }
    const VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx_.deviceQueues_.graphicsQueueFamilyIndex,
    };
    if (ctx_.vf_.vkCreateCommandPool(ctx_.getVkDevice(), &poolInfo, nullptr, &commandPool_) !=
        VK_SUCCESS) {
      commandPool_ = VK_NULL_HANDLE;
      return;
    }
    const VkCommandBufferAllocateInfo allocateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (ctx_.vf_.vkAllocateCommandBuffers(ctx_.getVkDevice(), &allocateInfo, &cmdBuf_) !=
            VK_SUCCESS ||
        ctx_.vf_.vkBeginCommandBuffer(cmdBuf_, &beginInfo) != VK_SUCCESS) {
      return;
    }
    ctx_.vf_.vkCmdFillBuffer(
        cmdBuf_, static_cast<vulkan::Buffer&>(*buffer_).getVkBuffer(), 0, sizeof(uint32_t), 0);
    if (ctx_.vf_.vkEndCommandBuffer(cmdBuf_) != VK_SUCCESS) {
      return;
    }
    const uint64_t waitValue = 1;
    const VkTimelineSemaphoreSubmitInfo timelineInfo = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &waitValue,
    };
    const VkSemaphore semaphore = semaphore_.getVkSemaphore();
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timelineInfo,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &semaphore,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuf_,
    };
    isClosed_ = ctx_.vf_.vkQueueSubmit(
                    ctx_.deviceQueues_.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS;
  }

  ~HostGate() {
    open();
    ctx_.waitIdle();
    if (commandPool_ != VK_NULL_HANDLE) {
      ctx_.vf_.vkDestroyCommandPool(ctx_.getVkDevice(), commandPool_, nullptr);
    }
  }

  HostGate(const HostGate&) = delete;
  HostGate& operator=(const HostGate&) = delete;

  [[nodiscard]] bool isClosed() const {
    return isClosed_;
  }

  void open() {
    if (!isClosed_) {
      return;
    }
    const VkSemaphoreSignalInfo signalInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = semaphore_.getVkSemaphore(),
        .value = 1,
    };
    EXPECT_EQ(ctx_.vf_.vkSignalSemaphoreKHR(ctx_.getVkDevice(), &signalInfo), VK_SUCCESS);
    isClosed_ = false;
  }

 private:
  vulkan::VulkanContext& ctx_;
  const vulkan::VulkanSemaphore semaphore_;
  std::shared_ptr<IBuffer> buffer_;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmdBuf_ = VK_NULL_HANDLE;
  bool isClosed_ = false;
};

#if IGL_PLATFORM_LINUX
/// Counts the presentations, image acquisitions and submissions of a context, and makes its
/// submissions run out of memory while asked to.
struct SwapchainCalls {
  PFN_vkQueuePresentKHR realPresent = nullptr;
  PFN_vkAcquireNextImageKHR realAcquire = nullptr;
  PFN_vkQueueSubmit realSubmit = nullptr;
  PFN_vkQueueSubmit2KHR realSubmit2 = nullptr;
  uint32_t presents = 0;
  uint32_t acquires = 0;
  uint32_t submits = 0;
  std::vector<uint32_t> acquiredImages;
  std::vector<uint32_t> presentedImages;
  bool submitsRunOutOfMemory = false;
  /// While set, a presentation fails the test and does not reach the driver, where it would wait
  /// for a semaphore that nothing signals.
  bool presentsForbidden = false;
};
thread_local SwapchainCalls gSwapchainCalls;

VKAPI_ATTR VkResult VKAPI_CALL countingQueuePresent(VkQueue queue, const VkPresentInfoKHR* info) {
  ++gSwapchainCalls.presents;
  gSwapchainCalls.presentedImages.push_back(info->pImageIndices[0]);
  if (gSwapchainCalls.presentsForbidden) {
    ADD_FAILURE() << "A frame that was not submitted is presented";
    return VK_SUCCESS;
  }
  return gSwapchainCalls.realPresent(queue, info);
}

VKAPI_ATTR VkResult VKAPI_CALL countingAcquireNextImage(VkDevice device,
                                                        VkSwapchainKHR swapchain,
                                                        uint64_t timeout,
                                                        VkSemaphore semaphore,
                                                        VkFence fence,
                                                        uint32_t* imageIndex) {
  ++gSwapchainCalls.acquires;
  const VkResult result =
      gSwapchainCalls.realAcquire(device, swapchain, timeout, semaphore, fence, imageIndex);
  gSwapchainCalls.acquiredImages.push_back(*imageIndex);
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL faultyQueueSubmit(VkQueue queue,
                                                 uint32_t count,
                                                 const VkSubmitInfo* submits,
                                                 VkFence fence) {
  ++gSwapchainCalls.submits;
  return gSwapchainCalls.submitsRunOutOfMemory
             ? VK_ERROR_OUT_OF_HOST_MEMORY
             : gSwapchainCalls.realSubmit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL faultyQueueSubmit2(VkQueue queue,
                                                  uint32_t count,
                                                  const VkSubmitInfo2* submits,
                                                  VkFence fence) {
  ++gSwapchainCalls.submits;
  return gSwapchainCalls.submitsRunOutOfMemory
             ? VK_ERROR_OUT_OF_HOST_MEMORY
             : gSwapchainCalls.realSubmit2(queue, count, submits, fence);
}

/// Routes a context's presentations, image acquisitions and submissions through the functions
/// above until destroyed.
class ScopedSwapchainCalls {
 public:
  explicit ScopedSwapchainCalls(vulkan::VulkanContext& ctx) :
    table_(const_cast<VulkanFunctionTable&>(ctx.vf_)), saved_(table_) {
    gSwapchainCalls = {
        .realPresent = table_.vkQueuePresentKHR,
        .realAcquire = table_.vkAcquireNextImageKHR,
        .realSubmit = table_.vkQueueSubmit,
        .realSubmit2 = table_.vkQueueSubmit2KHR,
    };
    table_.vkQueuePresentKHR = countingQueuePresent;
    table_.vkAcquireNextImageKHR = countingAcquireNextImage;
    table_.vkQueueSubmit = faultyQueueSubmit;
    table_.vkQueueSubmit2KHR = faultyQueueSubmit2;
  }
  ScopedSwapchainCalls(const ScopedSwapchainCalls&) = delete;
  ScopedSwapchainCalls& operator=(const ScopedSwapchainCalls&) = delete;
  ~ScopedSwapchainCalls() {
    table_ = saved_;
  }

 private:
  VulkanFunctionTable& table_;
  const VulkanFunctionTable saved_;
};
#endif // IGL_PLATFORM_LINUX

class ScopedFenceTimeout {
 public:
  ScopedFenceTimeout(vulkan::VulkanContext& ctx, uint64_t timeout) :
    ctx_(ctx), timeout_(ctx.config_.fenceTimeoutNanoseconds) {
    ctx_.config_.fenceTimeoutNanoseconds = timeout;
  }

  ~ScopedFenceTimeout() {
    ctx_.config_.fenceTimeoutNanoseconds = timeout_;
  }

  ScopedFenceTimeout(const ScopedFenceTimeout&) = delete;
  ScopedFenceTimeout& operator=(const ScopedFenceTimeout&) = delete;

 private:
  vulkan::VulkanContext& ctx_;
  const uint64_t timeout_;
};

// Opens the host gate before forwarding a presentation, which may wait for GPU work.
class PresentGateRelease {
 public:
  PresentGateRelease(vulkan::VulkanContext& ctx, HostGate& gate) :
    table_(const_cast<VulkanFunctionTable&>(ctx.vf_)) {
    state() = {.gate = &gate, .present = table_.vkQueuePresentKHR};
    table_.vkQueuePresentKHR = present;
  }

  ~PresentGateRelease() {
    table_.vkQueuePresentKHR = state().present;
    state() = {};
  }

  PresentGateRelease(const PresentGateRelease&) = delete;
  PresentGateRelease& operator=(const PresentGateRelease&) = delete;

 private:
  struct State {
    HostGate* gate = nullptr;
    PFN_vkQueuePresentKHR present = nullptr;
  };

  static State& state() {
    static State state;
    return state;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL present(VkQueue queue, const VkPresentInfoKHR* info) {
    const State& current = state();
    if (current.gate == nullptr || current.present == nullptr) {
      ADD_FAILURE() << "The presentation gate hook is not installed";
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    current.gate->open();
    return current.present(queue, info);
  }

  VulkanFunctionTable& table_;
};

bool hasUpdateAfterBindImageFeatures(const vulkan::VulkanContext& ctx) {
  const auto* indexing = static_cast<const VkPhysicalDeviceDescriptorIndexingFeaturesEXT*>(
      ctx.features().feature(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT));
  return indexing != nullptr && indexing->descriptorBindingSampledImageUpdateAfterBind == VK_TRUE &&
         indexing->descriptorBindingStorageImageUpdateAfterBind == VK_TRUE;
}

} // namespace

class ReusableCommandBufferVulkanTest : public ::testing::Test {
 public:
  void SetUp() override {
    // debug aborts fail the test instead of stopping it
    igl::setDebugBreakEnabled(false);
    expectedDebugAborts() = {};
    iglSetDebugAbortListener(failOnDebugAbort);
    iglDev_ = createDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";

    Result ret;
    cmdQueue_ = iglDev_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);

    texA_ = iglDev_->createTexture(texDesc_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    texB_ = iglDev_->createTexture(texDesc_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    fbA_ = iglDev_->createFramebuffer({.colorAttachments = {{.texture = texA_}}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    fbB_ = iglDev_->createFramebuffer({.colorAttachments = {{.texture = texB_}}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
    colorBuffer_ = iglDev_->createBuffer(
        BufferDesc{
            .type = BufferDesc::BufferTypeBits::Uniform,
            .data = red.data(),
            .length = sizeof(red),
        },
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    sampler_ = iglDev_->createSamplerState({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    colorPipeline_ = createPipeline(kColorShader);
    ASSERT_NE(colorPipeline_, nullptr);
    swizzlePipeline_ = createPipeline(kSwizzleShader);
    ASSERT_NE(swizzlePipeline_, nullptr);
  }

  void TearDown() override {
    // destroys everything while debug aborts fail the test
    swizzlePipeline_.reset();
    colorPipeline_.reset();
    sampler_.reset();
    colorBuffer_.reset();
    fbB_.reset();
    fbA_.reset();
    texB_.reset();
    texA_.reset();
    cmdQueue_.reset();
    iglDev_.reset();
    iglSetDebugAbortListener(nullptr);
  }

 protected:
  [[nodiscard]] virtual std::shared_ptr<IDevice> createDevice() const {
    return util::createTestDevice();
  }

  [[nodiscard]] vulkan::VulkanContext& getVulkanContext() const {
    return static_cast<vulkan::Device&>(*iglDev_).getVulkanContext();
  }

  // the Vulkan backend's API of reusable command buffers
  [[nodiscard]] static vulkan::PlatformDevice& getPlatformDevice(IDevice& device) {
    return *device.getPlatformDevice<vulkan::PlatformDevice>();
  }

  [[nodiscard]] vulkan::PlatformDevice& getPlatformDevice() const {
    return getPlatformDevice(*iglDev_);
  }

  [[nodiscard]] bool isRecorded(const ICommandBuffer& reusable) const {
    return getPlatformDevice().isRecorded(reusable);
  }

  void executeCommands(ICommandBuffer& cmdBuf, ICommandBuffer& reusable, Result* outResult) const {
    getPlatformDevice().executeCommands(cmdBuf, reusable, outResult);
  }

  void replaceTexture(ICommandBuffer& reusable,
                      const ITexture& recorded,
                      const std::shared_ptr<ITexture>& texture,
                      Result* outResult) const {
    getPlatformDevice().replaceTexture(reusable, recorded, texture, outResult);
  }

  [[nodiscard]] uint32_t getNumSlots() const {
    return getVulkanContext().config_.maxResourceCount;
  }

  // the layout of `texture` that IGL tracks
  [[nodiscard]] static VkImageLayout getImageLayout(const ITexture& texture) {
    return static_cast<const vulkan::Texture&>(texture).getVulkanTexture().image.imageLayout_;
  }

  [[nodiscard]] std::shared_ptr<IRenderPipelineState> createPipeline(
      const char* fragmentShader) const {
    Result ret;
    std::shared_ptr<IShaderStages> stages = ShaderStagesCreator::fromModuleStringInput(
        *iglDev_, kVertexShader, "main", "", fragmentShader, "main", "", &ret);
    if (!ret.isOk()) {
      return nullptr;
    }
    const RenderPipelineDesc desc{
        .shaderStages = std::move(stages),
        .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
    };
    return iglDev_->createRenderPipeline(desc, &ret);
  }

  [[nodiscard]] std::shared_ptr<ICommandBuffer> createReusable(bool executeInCommandBuffer) const {
    Result ret;
    std::shared_ptr<ICommandBuffer> reusable = getPlatformDevice().createReusableCommandBuffer(
        {.debugName = "Two passes", .executeInCommandBuffer = executeInCommandBuffer}, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
    return reusable;
  }

  [[nodiscard]] std::shared_ptr<ICommandBuffer> createCommandBuffer() const {
    Result ret;
    std::shared_ptr<ICommandBuffer> cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
    return cmdBuf;
  }

  // A submission processes the deferred tasks whose submissions have completed: an offscreen
  // context, whose frame number stays 0, makes them wait for nothing else
  void processDeferredTasks() const {
    ASSERT_EQ(getVulkanContext().getFrameNumber(), 0u);
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    cmdQueue_->submit(*cmdBuf);
  }

  static void clear(ICommandBuffer& cmdBuf,
                    const std::shared_ptr<IFramebuffer>& fb,
                    const Color& color) {
    const RenderPassDesc renderPass{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                          .storeAction = StoreAction::Store,
                                                          .clearColor = color}}};
    Result ret;
    const auto encoder = cmdBuf.createRenderCommandEncoder(renderPass, fb, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encoder->endEncoding();
  }

  // Clears both textures to magenta, which no pass renders
  void clearTextures(ICommandBuffer& cmdBuf) const {
    for (const std::shared_ptr<IFramebuffer>& fb : {fbA_, fbB_}) {
      clear(cmdBuf, fb, {1.0f, 0.0f, 1.0f, 1.0f});
    }
  }

  // Renders red, or the color of `colors` if it is not null, into texture A, or only loads it if
  // `loadTextureA` is set, then samples it into texture B with red and blue swapped
  void encodePasses(ICommandBuffer& cmdBuf,
                    bool loadTextureA = false,
                    IBuffer* IGL_NULLABLE colors = nullptr) const {
    Result ret;
    auto encoder =
        cmdBuf.createRenderCommandEncoder(loadTextureA ? loadPass_ : renderPass_, fbA_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    if (!loadTextureA) {
      encoder->bindRenderPipelineState(colorPipeline_);
      encoder->bindBuffer(0, colors ? colors : colorBuffer_.get());
      encoder->draw(3);
    }
    encoder->endEncoding();

    encoder = cmdBuf.createRenderCommandEncoder(renderPass_, fbB_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encoder->bindRenderPipelineState(swizzlePipeline_);
    encoder->bindTexture(0, BindTarget::kFragment, texA_.get());
    encoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());
    encoder->draw(3);
    encoder->endEncoding();
  }

  // A uniform buffer of 4 floats
  [[nodiscard]] std::shared_ptr<IBuffer> createColorBuffer(BufferDesc::BufferAPIHint hint) const {
    Result ret;
    std::shared_ptr<IBuffer> buffer = iglDev_->createBuffer(
        BufferDesc{
            .type = BufferDesc::BufferTypeBits::Uniform,
            .length = 4 * sizeof(float),
            .hint = hint,
        },
        &ret);
    return ret.isOk() ? buffer : nullptr;
  }

  // Runs `reusable` with nothing submitted between its recording, which can read a Ring buffer, and
  // the run. Returns whether `reusable` ran.
  [[nodiscard]] bool run(ICommandBuffer& reusable, bool executeInCommandBuffer) const {
    if (!executeInCommandBuffer) {
      return cmdQueue_->submit(reusable) != 0;
    }
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    return cmdBuf && submitAndRun(*cmdBuf, reusable, true);
  }

  [[nodiscard]] Pixels readTextureB() const {
    Pixels pixels = {};
    fbB_->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), texB_->getFullRange(0));
    return pixels;
  }

  // Submits `cmdBuf` and runs `reusable`: from `cmdBuf` if `executeInCommandBuffer` is set, and
  // after it otherwise. Returns whether `reusable` ran.
  [[nodiscard]] bool submitAndRun(ICommandBuffer& cmdBuf,
                                  ICommandBuffer& reusable,
                                  bool executeInCommandBuffer) const {
    if (!executeInCommandBuffer) {
      cmdQueue_->submit(cmdBuf);
      return cmdQueue_->submit(reusable) != 0;
    }
    Result ret;
    executeCommands(cmdBuf, reusable, &ret);
    cmdQueue_->submit(cmdBuf);
    return ret.isOk();
  }

  // Clears both textures and runs `reusable`, after encoding the passes into it unless its current
  // slot holds a recording. Returns whether `reusable` ran.
  [[nodiscard]] bool renderFrame(ICommandBuffer& reusable, bool executeInCommandBuffer) const {
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    if (!cmdBuf) {
      return false;
    }
    clearTextures(*cmdBuf);
    if (!isRecorded(reusable)) {
      encodePasses(reusable);
    }
    return submitAndRun(*cmdBuf, reusable, executeInCommandBuffer);
  }

  // Renders the passes with a reusable command buffer for 3 frames per slot, after clearing both
  // textures in every frame, and checks every frame against a command buffer that encodes the
  // passes again
  void renderFrames(bool executeInCommandBuffer) const {
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    encodePasses(*cmdBuf);
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
    const Pixels expected = readTextureB();
    ASSERT_EQ(expected, bluePixels());

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);

    const uint32_t numSlots = getNumSlots();
    uint32_t numRecordings = 0;
    for (uint32_t frame = 0; frame != 3 * numSlots; ++frame) {
      ASSERT_EQ(isRecorded(*reusable), frame >= numSlots) << "frame " << frame;
      if (isRecorded(*reusable)) {
        Result ret;
        EXPECT_EQ(reusable->createRenderCommandEncoder(renderPass_, fbA_, &ret), nullptr);
        EXPECT_EQ(ret.code, Result::Code::InvalidOperation);
      } else {
        ++numRecordings;
      }
      EXPECT_TRUE(renderFrame(*reusable, executeInCommandBuffer)) << "frame " << frame;
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTextureB(), expected) << "frame " << frame;
    }
    EXPECT_EQ(numRecordings, numSlots);
    EXPECT_TRUE(getVulkanContext().recordingJournals_.empty());
  }

  // Renders 3 frames per slot without waiting between frames, and checks the output of every frame
  // against a command buffer that encodes the passes again in that frame. Every frame clears
  // texture A to a color of its own before the passes, which load it, and copies texture B into a
  // buffer of its own after them. The first frames of all the slots but one stay pending behind a
  // gate until all of them are submitted, and so do the replays of those slots in the second round
  // of frames, which begins once the first round completes.
  void replayOutstandingFrames(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const uint32_t numSlots = getNumSlots();
    if (numSlots < 2) {
      GTEST_SKIP() << "A single frame in flight";
    }
    const uint32_t numFrames = 3 * numSlots;
    const uint32_t numGatedFrames = numSlots - 1;
    const auto colorOf = [](uint32_t frame) {
      return Color(static_cast<float>(10 * frame + 5) / 255.0f,
                   static_cast<float>(250 - 10 * frame) / 255.0f,
                   static_cast<float>(frame % 2) * (128.0f / 255.0f),
                   1.0f);
    };
    // what the swizzle pass renders from a texture A cleared to colorOf(frame)
    const auto pixelOf = [](uint32_t frame) {
      return 0xFF000000u | ((10u * frame + 5u) << 16) | ((250u - 10u * frame) << 8) |
             ((frame % 2) * 128u);
    };
    constexpr size_t kFrameBytes = sizeof(Pixels);
    const BufferDesc bufferDesc{
        .type = BufferDesc::BufferTypeBits::Storage,
        .length = numFrames * kFrameBytes,
        .storage = ResourceStorage::Shared,
    };
    Result ret;
    const std::shared_ptr<IBuffer> expectedFrames = iglDev_->createBuffer(bufferDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<IBuffer> replayedFrames = iglDev_->createBuffer(bufferDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    std::shared_ptr<ICommandBuffer> cmdBuf;
    for (uint32_t frame = 0; frame != numFrames; ++frame) {
      cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      clear(*cmdBuf, fbA_, colorOf(frame));
      encodePasses(*cmdBuf, true);
      cmdBuf->copyTextureToBuffer(*texB_, *expectedFrames, frame * kFrameBytes);
      cmdQueue_->submit(*cmdBuf);
    }
    cmdBuf->waitUntilCompleted();

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    HostGate gate(*iglDev_, ctx);
    if (!gate.isClosed()) {
      GTEST_SKIP() << "Timeline semaphores are not supported";
    }
    std::optional<HostGate> replayGate;
    std::vector<SubmitHandle> runs;
    for (uint32_t frame = 0; frame != numFrames; ++frame) {
      if (frame == numSlots) {
        cmdBuf->waitUntilCompleted();
        replayGate.emplace(*iglDev_, ctx);
        ASSERT_TRUE(replayGate->isClosed());
      }
      cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      clear(*cmdBuf, fbA_, colorOf(frame));
      if (!isRecorded(*reusable)) {
        encodePasses(*reusable, true);
      }
      if (executeInCommandBuffer) {
        executeCommands(*cmdBuf, *reusable, &ret);
        ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
        cmdBuf->copyTextureToBuffer(*texB_, *replayedFrames, frame * kFrameBytes);
        runs.push_back(cmdQueue_->submit(*cmdBuf));
      } else {
        cmdQueue_->submit(*cmdBuf);
        runs.push_back(cmdQueue_->submit(*reusable));
        cmdBuf = createCommandBuffer();
        ASSERT_NE(cmdBuf, nullptr);
        cmdBuf->copyTextureToBuffer(*texB_, *replayedFrames, frame * kFrameBytes);
        cmdQueue_->submit(*cmdBuf);
      }
      ASSERT_NE(runs.back(), 0u) << "frame " << frame;
      if (runs.size() == numGatedFrames) {
        for (const SubmitHandle run : runs) {
          EXPECT_FALSE(ctx.immediate_->isReady(vulkan::VulkanImmediateCommands::SubmitHandle(run)));
        }
        gate.open();
      }
      if (runs.size() == numSlots + numGatedFrames) {
        for (size_t replay = numSlots; replay != runs.size(); ++replay) {
          EXPECT_FALSE(
              ctx.immediate_->isReady(vulkan::VulkanImmediateCommands::SubmitHandle(runs[replay])))
              << "frame " << replay;
        }
        replayGate->open();
      }
    }
    cmdBuf->waitUntilCompleted();

    const auto* expected = static_cast<const Pixels*>(
        expectedFrames->map(BufferRange(expectedFrames->getSizeInBytes()), &ret));
    ASSERT_NE(expected, nullptr) << ret.message.c_str();
    const auto* replayed = static_cast<const Pixels*>(
        replayedFrames->map(BufferRange(replayedFrames->getSizeInBytes()), &ret));
    ASSERT_NE(replayed, nullptr) << ret.message.c_str();
    for (uint32_t frame = 0; frame != numFrames; ++frame) {
      EXPECT_EQ(expected[frame], fillPixels(pixelOf(frame))) << "frame " << frame;
      EXPECT_EQ(replayed[frame], expected[frame]) << "frame " << frame;
    }
    replayedFrames->unmap();
    expectedFrames->unmap();
  }

  // Runs a slot in a submission that waits for a semaphore, comes back to the slot, and checks that
  // running it fails, and stays in the slot, until the host signals the semaphore
  void runPendingSlot(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    if (!ctx.features().has_VK_KHR_timeline_semaphore || !ctx.vf_.vkSignalSemaphoreKHR) {
      GTEST_SKIP() << "Timeline semaphores are not supported";
    }
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    const std::shared_ptr<ICommandBuffer> nextCmdBuf = createCommandBuffer();
    ASSERT_NE(nextCmdBuf, nullptr);

    // Until the host signals the semaphore, nothing may return, as the device could not become
    // idle, and nothing else may be submitted, as the submissions wait for each other with binary
    // semaphores, which cannot wait for a timeline semaphore that is signaled later
    const vulkan::VulkanSemaphore semaphore(ctx.vf_, ctx.getVkDevice(), 0, false, "Pending slot");
    const uint64_t fenceTimeout = ctx.config_.fenceTimeoutNanoseconds;
    ctx.config_.fenceTimeoutNanoseconds = 10'000'000;
    const uint32_t slot = ctx.currentSyncIndex();
    clearTextures(*cmdBuf);
    encodePasses(*reusable);
    Result ret;
    if (executeInCommandBuffer) {
      executeCommands(*cmdBuf, *reusable, &ret);
      EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
    } else {
      cmdQueue_->submit(*cmdBuf);
    }
    EXPECT_TRUE(ctx.immediate_->waitSemaphore(semaphore.getVkSemaphore(), 1));
    EXPECT_NE(cmdQueue_->submit(executeInCommandBuffer ? *cmdBuf : *reusable), 0u);
    // moves to the slot again as submits that read Ring buffers do
    while (ctx.currentSyncIndex() != slot) {
      ctx.markRingBufferRead();
      ctx.syncMarkSubmitted({});
    }
    if (executeInCommandBuffer) {
      executeCommands(*nextCmdBuf, *reusable, &ret);
      EXPECT_EQ(ret.code, Result::Code::RuntimeError) << ret.message.c_str();
    } else {
      EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
    }
    EXPECT_EQ(ctx.currentSyncIndex(), slot);
    EXPECT_TRUE(isRecorded(*reusable));

    ctx.config_.fenceTimeoutNanoseconds = fenceTimeout;
    const VkSemaphoreSignalInfo signalInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = semaphore.getVkSemaphore(),
        .value = 1,
    };
    EXPECT_EQ(ctx.vf_.vkSignalSemaphoreKHR(ctx.getVkDevice(), &signalInfo), VK_SUCCESS);
    cmdQueue_->submit(*nextCmdBuf);
    EXPECT_TRUE(renderFrame(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  // Renders red into texture A, which leaves it ready to be sampled, and records two passes that
  // load texture A while `call` fails, for the command buffers of the recording, once `numCalls`
  // more calls of it have succeeded. Checks that the encoders fail if `failsInEncoders` is set,
  // that running the slot fails, that texture A keeps its layout, that the slot stays the current
  // one, and that it records again a pass that loads the red of texture A.
  void discardFailedRecording(bool executeInCommandBuffer,
                              CommandBufferFailure::Call call,
                              uint32_t numCalls,
                              bool failsInEncoders = true) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    encodePasses(*cmdBuf);
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
    ASSERT_EQ(getImageLayout(*texA_), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    const uint32_t slot = ctx.currentSyncIndex();
    cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    {
      const CommandBufferFailure failure(ctx, call, numCalls);
      Result encoderRet;
      std::unique_ptr<IRenderCommandEncoder> encoder =
          reusable->createRenderCommandEncoder(loadPass_, fbA_, &encoderRet);
      if (encoder) {
        // a failure when the render pass ends shows in the next encoder
        encoder->endEncoding();
        encoder = reusable->createRenderCommandEncoder(loadPass_, fbA_, &encoderRet);
        if (encoder) {
          encoder->endEncoding();
        }
      }
      EXPECT_EQ(encoder == nullptr, failsInEncoders);
      if (failsInEncoders) {
        EXPECT_EQ(encoderRet.code, Result::Code::RuntimeError) << encoderRet.message.c_str();
      }
      // a recording can also fail to end when it runs first
      EXPECT_FALSE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    }
    EXPECT_EQ(getImageLayout(*texA_), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    EXPECT_EQ(ctx.currentSyncIndex(), slot);
    EXPECT_FALSE(isRecorded(*reusable));

    cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    encodePasses(*reusable, true);
    EXPECT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
    EXPECT_TRUE(ctx.recordingJournals_.empty());
  }

  // Records a render pass with a depth attachment that is not sampled, which the render pass leaves
  // ready to be rendered to, and checks that the run fails, and that the depth attachment goes back
  // to its layout, when the last command buffer of the recording fails to end
  void discardRecordingThatFailsToEnd(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    Result ret;
    const std::shared_ptr<ITexture> depthTex = iglDev_->createTexture(
        TextureDesc::new2D(
            TextureFormat::Z_UNorm16, kSize, kSize, TextureDesc::TextureUsageBits::Attachment),
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<IFramebuffer> fb = iglDev_->createFramebuffer(
        {.colorAttachments = {{.texture = texA_}}, .depthAttachment = {.texture = depthTex}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    {
      // the primary command buffer of the slot, or the secondary command buffer after the render
      // pass
      const CommandBufferFailure failure(
          ctx, CommandBufferFailure::Call::End, executeInCommandBuffer ? 2 : 0);
      const std::unique_ptr<IRenderCommandEncoder> encoder =
          reusable->createRenderCommandEncoder(depthPass_, fb, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      encoder->endEncoding();
      EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      EXPECT_FALSE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    }
    EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
    EXPECT_FALSE(isRecorded(*reusable));
    EXPECT_TRUE(ctx.recordingJournals_.empty());
  }

  // Records the passes, and moves the shared slot with the run of another reusable command buffer,
  // as another engine can, before the first run. Checks that encoding into the slot, if
  // `encodeAfterTheMove` is set, and running it fail with Result::Code::InvalidOperation, that
  // nothing runs, that the image layouts of the recording are put back, and that the current slot
  // records and runs the passes.
  void failRecordingWhenTheSlotMoves(bool executeInCommandBuffer, bool encodeAfterTheMove) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    Result ret;
    const std::shared_ptr<ITexture> depthTex = iglDev_->createTexture(
        TextureDesc::new2D(
            TextureFormat::Z_UNorm16, kSize, kSize, TextureDesc::TextureUsageBits::Attachment),
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<IFramebuffer> fb = iglDev_->createFramebuffer(
        {.colorAttachments = {{.texture = texA_}}, .depthAttachment = {.texture = depthTex}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> otherEngine = createReusable(false);
    ASSERT_NE(otherEngine, nullptr);
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    encodePasses(*reusable);
    {
      const std::unique_ptr<IRenderCommandEncoder> encoder =
          reusable->createRenderCommandEncoder(depthPass_, fb, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      encoder->endEncoding();
    }
    ASSERT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    const uint32_t slot = ctx.currentSyncIndex();
    ASSERT_NE(cmdQueue_->submit(*otherEngine), 0u);
    ASSERT_NE(ctx.currentSyncIndex(), slot);
    if (encodeAfterTheMove) {
      EXPECT_EQ(reusable->createRenderCommandEncoder(renderPass_, fbB_, &ret), nullptr);
      EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();
    }
    if (executeInCommandBuffer) {
      executeCommands(*cmdBuf, *reusable, &ret);
      EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();
      cmdQueue_->submit(*cmdBuf);
    } else {
      cmdQueue_->submit(*cmdBuf);
      EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
    }
    cmdBuf->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), fillPixels(kMagenta));
    EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
    EXPECT_FALSE(isRecorded(*reusable));
    EXPECT_TRUE(ctx.recordingJournals_.empty());

    EXPECT_TRUE(renderFrame(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  // Runs the reusable command buffer while one of its render passes is being encoded, and checks
  // that the run fails and leaves the recording intact, which runs once the render pass ends
  void runWhileARenderPassIsEncoded(bool executeInCommandBuffer) const {
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    const RenderPassDesc bluePass{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                        .storeAction = StoreAction::Store,
                                                        .clearColor = {0.0f, 0.0f, 1.0f, 1.0f}}}};
    Result ret;
    std::unique_ptr<IRenderCommandEncoder> encoder =
        reusable->createRenderCommandEncoder(bluePass, fbB_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    if (executeInCommandBuffer) {
      executeCommands(*cmdBuf, *reusable, &ret);
      EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();
    } else {
      cmdQueue_->submit(*cmdBuf);
      EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
      cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
    }
    EXPECT_FALSE(isRecorded(*reusable));

    encoder->endEncoding();
    EXPECT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  // Runs the passes for 2 frames per slot, counting their 2 draws while each slot records, and
  // checks that every frame adds them to the draw statistics of the queue, while the draws that
  // the device counts as encoded grow only while slots record
  void countReplayedDraws(bool executeInCommandBuffer) const {
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const size_t numEncodedDraws = iglDev_->getCurrentDrawCount();
    cmdQueue_->endFrame();

    const uint32_t numSlots = getNumSlots();
    for (uint32_t frame = 0; frame != 2 * numSlots; ++frame) {
      const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      clearTextures(*cmdBuf);
      if (!isRecorded(*reusable)) {
        encodePasses(*reusable);
        reusable->incrementCurrentDrawCount();
        reusable->incrementCurrentDrawCount();
      }
      EXPECT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer)) << "frame " << frame;
      cmdQueue_->endFrame();
      EXPECT_EQ(cmdQueue_->getLastFrameDrawCount(), 2u) << "frame " << frame;
    }
    EXPECT_EQ(iglDev_->getCurrentDrawCount() - numEncodedDraws, 2u * numSlots);
  }

  // Records a clear of an attachment that is also a storage image, uploads into it, which leaves it
  // in the general layout, records another clear of it, and fails the recording by moving the
  // shared slot: the layout goes back to the one that the upload left. With
  // `executeInCommandBuffer`, the recording starts in the attachment layout: the second clear's
  // transition goes into the secondary command buffer after the first render pass, which holds no
  // earlier layout of the image. A primary command buffer records the second clear after the first
  // one, so there the recording starts in the general layout and goes back to it before the
  // upload, which keeps the upload and the recording valid.
  void failRecordingAfterAnUpload(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    Result ret;
    const std::shared_ptr<ITexture> tex =
        iglDev_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                  kSize,
                                                  kSize,
                                                  TextureDesc::TextureUsageBits::Attachment |
                                                      TextureDesc::TextureUsageBits::Storage),
                               &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<IFramebuffer> fb =
        iglDev_->createFramebuffer({.colorAttachments = {{.texture = tex}}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clear(*cmdBuf, fb, {1.0f, 0.0f, 1.0f, 1.0f});
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
    const Pixels pixels = bluePixels();
    if (!executeInCommandBuffer) {
      ASSERT_TRUE(tex->upload(tex->getFullRange(0), pixels.data()).isOk());
    }
    const VkImageLayout initialLayout = getImageLayout(*tex);
    ASSERT_EQ(initialLayout,
              executeInCommandBuffer ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                     : VK_IMAGE_LAYOUT_GENERAL);

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> otherEngine = createReusable(false);
    ASSERT_NE(otherEngine, nullptr);
    clear(*reusable, fb, {0.0f, 0.0f, 1.0f, 1.0f});
    if (!executeInCommandBuffer) {
      vulkan::transitionToGeneral(
          static_cast<vulkan::CommandBuffer&>(*reusable).getVkCommandBuffer(), tex.get());
    }
    ASSERT_EQ(getImageLayout(*tex), initialLayout);
    ASSERT_TRUE(tex->upload(tex->getFullRange(0), pixels.data()).isOk());
    ASSERT_EQ(getImageLayout(*tex), VK_IMAGE_LAYOUT_GENERAL);
    clear(*reusable, fb, {1.0f, 0.0f, 0.0f, 1.0f});
    ASSERT_EQ(getImageLayout(*tex), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    ASSERT_NE(cmdQueue_->submit(*otherEngine), 0u);
    cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    EXPECT_FALSE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    EXPECT_EQ(getImageLayout(*tex), VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_TRUE(ctx.recordingJournals_.empty());
  }

  // Records the passes, lets the runs of another reusable command buffer move the shared slot
  // around all the slots, back to the slot of the recording, and checks that the recording runs
  void runAfterAFullCycleOfSlots(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> otherEngine = createReusable(false);
    ASSERT_NE(otherEngine, nullptr);
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    encodePasses(*reusable);
    const uint32_t slot = ctx.currentSyncIndex();
    for (uint32_t i = 0; i != getNumSlots(); ++i) {
      ASSERT_NE(cmdQueue_->submit(*otherEngine), 0u);
    }
    ASSERT_EQ(ctx.currentSyncIndex(), slot);
    EXPECT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  // Checks over 3 frames per slot that `reusable` replays the passes into textures that every frame
  // clears
  void replayPasses(ICommandBuffer& reusable, bool executeInCommandBuffer) const {
    for (uint32_t frame = 0; frame != 3 * getNumSlots(); ++frame) {
      EXPECT_TRUE(renderFrame(reusable, executeInCommandBuffer)) << "frame " << frame;
      reusable.waitUntilCompleted();
      EXPECT_EQ(readTextureB(), bluePixels()) << "frame " << frame;
    }
  }

  // Presents a swapchain image from command buffer A, runs `reusable` from A if
  // `executeInCommandBuffer` is set and before A's submission otherwise, acquires command buffer B,
  // and fills the injected semaphores, so that the presentation of A, which is not the current
  // command buffer, fails, in 3 frames. A command buffer that executes a slot gives back its Vulkan
  // command buffer and its slot; any other one goes through VulkanImmediateCommands::discard(),
  // which keeps a command buffer that is not the current one. Then the presentation of the current
  // command buffer fails: discard() gives it back and drops the injected semaphores. Then
  // `reusable` replays.
  void failPresentOfAnOlderCommandBuffer(bool executeInCommandBuffer,
                                         bool useTimelineSemaphore) const {
#if IGL_PLATFORM_LINUX
    vulkan::VulkanContext& ctx = getVulkanContext();
    ASSERT_TRUE(ctx.initSwapchain(kSize, kSize).isOk());
    if (!useTimelineSemaphore) {
      ctx.timelineSemaphore_.reset();
    } else if (!ctx.timelineSemaphore_) {
      GTEST_SKIP() << "Timeline semaphores are not supported";
    }
    // the command buffers that never run leave the layouts that they record to these
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
    Result ret;
    const std::shared_ptr<ITexture> drawable =
        getPlatformDevice().createTextureFromNativeDrawable(&ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(drawable, nullptr);
    // the next submission can wait for each of them, or signal it with `value`, at once
    std::vector<std::unique_ptr<vulkan::VulkanSemaphore>> semaphores;
    for (uint32_t i = 0; i != vulkan::VulkanImmediateCommands::kMaxInjectedSemaphores; ++i) {
      semaphores.push_back(std::make_unique<vulkan::VulkanSemaphore>(
          ctx.vf_, ctx.getVkDevice(), uint64_t{0}, false, "Injected semaphore"));
    }
    const auto injectSemaphores = [&ctx, &semaphores, useTimelineSemaphore](uint64_t value) {
      return std::all_of(
          semaphores.begin(),
          semaphores.end(),
          [&ctx, useTimelineSemaphore, value](
              const std::unique_ptr<vulkan::VulkanSemaphore>& semaphore) {
            return useTimelineSemaphore
                       ? ctx.immediate_->signalSemaphore(semaphore->getVkSemaphore(), value)
                       : ctx.immediate_->waitSemaphore(semaphore->getVkSemaphore(), 0);
          });
    };

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    for (uint32_t frame = 0; frame != 3; ++frame) {
      const std::shared_ptr<ICommandBuffer> older = createCommandBuffer();
      ASSERT_NE(older, nullptr);
      const vulkan::VulkanImmediateCommands::SubmitHandle handle =
          static_cast<vulkan::CommandBuffer&>(*older).getNextSubmitHandle();
      clearTextures(*older);
      if (!isRecorded(*reusable)) {
        encodePasses(*reusable);
      }
      if (executeInCommandBuffer) {
        executeCommands(*older, *reusable, &ret);
        ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      } else {
        ASSERT_NE(cmdQueue_->submit(*reusable), 0u);
      }
      older->present(drawable);
      const std::shared_ptr<ICommandBuffer> later = createCommandBuffer();
      ASSERT_NE(later, nullptr);
      ASSERT_TRUE(injectSemaphores(frame + 1));
#if IGL_DEBUG_ABORT_ENABLED
      expectedDebugAborts() = {.func = "discard", .count = executeInCommandBuffer ? 0u : 1u};
#endif
      EXPECT_EQ(cmdQueue_->submit(*older), 0u) << "frame " << frame;
#if IGL_DEBUG_ABORT_ENABLED
      EXPECT_EQ(expectedDebugAborts().count, 0u) << "frame " << frame;
#endif
      if (executeInCommandBuffer) {
        // a released handle is ready once the work submitted before it finished
        EXPECT_EQ(ctx.immediate_->wait(handle, 10'000'000'000ull), VK_SUCCESS) << "frame " << frame;
        EXPECT_TRUE(ctx.immediate_->isReady(handle)) << "frame " << frame;
      } else {
        EXPECT_FALSE(ctx.immediate_->isReady(handle)) << "frame " << frame;
      }
      cmdQueue_->submit(*later);
    }

    if (!isRecorded(*reusable)) {
      encodePasses(*reusable);
    }
    if (!executeInCommandBuffer) {
      ASSERT_NE(cmdQueue_->submit(*reusable), 0u);
    }
    const std::shared_ptr<ICommandBuffer> current = createCommandBuffer();
    ASSERT_NE(current, nullptr);
    const vulkan::VulkanImmediateCommands::SubmitHandle handle =
        static_cast<vulkan::CommandBuffer&>(*current).getNextSubmitHandle();
    if (executeInCommandBuffer) {
      executeCommands(*current, *reusable, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    }
    current->present(drawable);
    ASSERT_TRUE(injectSemaphores(4));
    EXPECT_EQ(cmdQueue_->submit(*current), 0u);
    // a discarded handle is ready once the work submitted before it finished
    EXPECT_EQ(ctx.immediate_->wait(handle, 10'000'000'000ull), VK_SUCCESS);
    EXPECT_TRUE(ctx.immediate_->isReady(handle));
    EXPECT_TRUE(injectSemaphores(5));
    cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    cmdQueue_->submit(*cmdBuf);
    replayPasses(*reusable, executeInCommandBuffer);
#else
    (void)executeInCommandBuffer;
    (void)useTimelineSemaphore;
    GTEST_SKIP() << "Headless swapchains are tested on Linux";
#endif // IGL_PLATFORM_LINUX
  }

  // A texture with 2 mip levels whose UNORM image has sRGB views, or null if the device cannot
  // create one
  [[nodiscard]] std::shared_ptr<ITexture> createSrgbViewedTexture() const {
    vulkan::VulkanImage image(getVulkanContext(),
                              VkExtent3D{.width = kSize, .height = kSize, .depth = 1},
                              VK_IMAGE_TYPE_2D,
                              VK_FORMAT_R8G8B8A8_UNORM,
                              2,
                              1,
                              VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                              VK_SAMPLE_COUNT_1_BIT,
                              "Image: sRGB views",
                              true);
    if (!image.valid() || !image.isSrgbMutableFormat_) {
      return nullptr;
    }
    vulkan::VulkanImageView imageView = image.createImageView(VK_IMAGE_VIEW_TYPE_2D,
                                                              VK_FORMAT_R8G8B8A8_SRGB,
                                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                                              0,
                                                              VK_REMAINING_MIP_LEVELS,
                                                              0,
                                                              1,
                                                              "Image View: sRGB");
    TextureDesc desc = TextureDesc::new2D(
        TextureFormat::RGBA_SRGB, kSize, kSize, TextureDesc::TextureUsageBits::Sampled);
    desc.numMipLevels = 2;
    return std::make_shared<vulkan::Texture>(
        static_cast<vulkan::Device&>(*iglDev_),
        std::make_shared<vulkan::VulkanTexture>(std::move(image), std::move(imageView)),
        desc);
  }

  // Records the passes, a render pass with a depth attachment, then the mipmaps of a texture whose
  // UNORM image has sRGB views, which go through a scratch image that lives only as long as the
  // call. Checks that the recording fails with Result::Code::Unsupported at once, which puts back
  // the image layouts, that the run fails without running anything, and that the slot records the
  // passes again, which replay.
  void failRecordingThatGeneratesSrgbMipmaps(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<ITexture> srgbTex = createSrgbViewedTexture();
    if (!srgbTex) {
      GTEST_SKIP() << "UNORM images with sRGB views are not supported";
    }
    Result ret;
    const std::shared_ptr<ITexture> depthTex = iglDev_->createTexture(
        TextureDesc::new2D(
            TextureFormat::Z_UNorm16, kSize, kSize, TextureDesc::TextureUsageBits::Attachment),
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<IFramebuffer> fb = iglDev_->createFramebuffer(
        {.colorAttachments = {{.texture = texA_}}, .depthAttachment = {.texture = depthTex}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    const uint32_t slot = ctx.currentSyncIndex();
    encodePasses(*reusable);
    {
      const std::unique_ptr<IRenderCommandEncoder> encoder =
          reusable->createRenderCommandEncoder(depthPass_, fb, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      encoder->endEncoding();
    }
    ASSERT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    srgbTex->generateMipmap(*reusable);
    EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
    EXPECT_EQ(getImageLayout(*srgbTex), VK_IMAGE_LAYOUT_UNDEFINED);
    EXPECT_TRUE(ctx.recordingJournals_.empty());
    EXPECT_EQ(reusable->createRenderCommandEncoder(renderPass_, fbB_, &ret), nullptr);
    EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();
    if (executeInCommandBuffer) {
      executeCommands(*cmdBuf, *reusable, &ret);
      EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();
      cmdQueue_->submit(*cmdBuf);
    } else {
      cmdQueue_->submit(*cmdBuf);
      EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
    }
    cmdBuf->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), fillPixels(kMagenta));
    EXPECT_EQ(ctx.currentSyncIndex(), slot);
    EXPECT_FALSE(isRecorded(*reusable));
    replayPasses(*reusable, executeInCommandBuffer);
  }

  // Generates the mipmaps of an image with sRGB views straight into the command buffer that records
  // a slot, as a caller of vulkan::VulkanImage can: the scratch image that they go through lives
  // only as long as the call, so nothing is recorded or journaled, and the recording goes on
  void generateSrgbMipmapsIntoASlot(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<ITexture> srgbTex = createSrgbViewedTexture();
    if (!srgbTex) {
      GTEST_SKIP() << "UNORM images with sRGB views are not supported";
    }
    const vulkan::VulkanImage& image =
        static_cast<vulkan::Texture&>(*srgbTex).getVulkanTexture().image;
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    vulkan::transitionToShaderReadOnly(
        static_cast<vulkan::CommandBuffer&>(*cmdBuf).getVkCommandBuffer(), srgbTex.get());
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
    ASSERT_EQ(image.imageLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    encodePasses(*reusable);
    const VkCommandBuffer recording =
        static_cast<vulkan::CommandBuffer&>(*reusable).getVkCommandBuffer();
    ASSERT_TRUE(ctx.isRecordingSlot(recording));
    ASSERT_EQ(ctx.recordingJournals_.size(), 1u);
    const size_t numJournaled = ctx.recordingJournals_.front()->layouts.size();
    const Result result = image.generateMipmap(recording, srgbTex->getFullMipRange());
    EXPECT_EQ(result.code, Result::Code::Unsupported) << result.message.c_str();
    EXPECT_EQ(image.imageLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ASSERT_EQ(ctx.recordingJournals_.front()->layouts.size(), numJournaled);
    EXPECT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  // Records the mipmap generation of a texture whose first level every frame clears to a color of
  // its own, and checks over 3 frames per slot that every run generates the second level from that
  // color. In the frames whose slot holds a recording, generating the mipmaps again records
  // nothing.
  void replayGeneratedMipmaps(bool executeInCommandBuffer) const {
    TextureDesc desc = texDesc_;
    desc.numMipLevels = 2;
    Result ret;
    const std::shared_ptr<ITexture> tex = iglDev_->createTexture(desc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<IFramebuffer> fb =
        iglDev_->createFramebuffer({.colorAttachments = {{.texture = tex}}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    // a color of its own for every frame, which RGBA_UNorm8 holds exactly
    const auto colorOf = [](uint32_t frame) {
      return Color(static_cast<float>(20 * frame + 15) / 255.0f, 0.0f, 1.0f, 1.0f);
    };
    const auto pixelOf = [](uint32_t frame) { return 0xFFFF0000u | (20u * frame + 15u); };

    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    for (uint32_t frame = 0; frame != 3 * getNumSlots(); ++frame) {
      const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      clear(*cmdBuf, fb, colorOf(frame));
      const bool isSlotRecorded = isRecorded(*reusable);
#if IGL_DEBUG_ABORT_ENABLED
      expectedDebugAborts() = {.func = "getReusableVkCommandBuffer",
                               .count = isSlotRecorded ? 1u : 0u};
#endif
      tex->generateMipmap(*reusable);
#if IGL_DEBUG_ABORT_ENABLED
      EXPECT_EQ(expectedDebugAborts().count, 0u) << "frame " << frame;
#endif
      EXPECT_EQ(isRecorded(*reusable), isSlotRecorded) << "frame " << frame;
      ASSERT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer)) << "frame " << frame;
      reusable->waitUntilCompleted();
      std::array<uint32_t, (kSize / 2) * (kSize / 2)> secondLevel = {};
      fb->copyBytesColorAttachment(*cmdQueue_, 0, secondLevel.data(), tex->getFullRange(1));
      for (const uint32_t pixel : secondLevel) {
        EXPECT_EQ(pixel, pixelOf(frame)) << "frame " << frame;
      }
    }
  }

  // Records the passes, which render the color of a Ring buffer that holds red, runs them in a
  // submission that waits for a semaphore, and comes back to their slot, whose wait for the run
  // times out. A command buffer that reads the Ring buffer and presents a swapchain image is then
  // submitted there, and the queue discards it, since no semaphore can be injected for the
  // presentation. Checks that the run stays the reader of the slot's copy: when the slot comes back
  // again, uploading green into the Ring buffer fails until the run completes, so that the run
  // renders the red of its frame, and the next run renders green.
  void uploadAfterADiscardedPresent() const {
#if IGL_PLATFORM_LINUX
    vulkan::VulkanContext& ctx = getVulkanContext();
    ASSERT_TRUE(ctx.initSwapchain(kSize, kSize).isOk());
    // the swapchain signals it when it presents
    if (!ctx.timelineSemaphore_ || !ctx.vf_.vkSignalSemaphoreKHR) {
      GTEST_SKIP() << "Timeline semaphores are not supported";
    }
    Result ret;
    const std::shared_ptr<ITexture> drawable =
        getPlatformDevice().createTextureFromNativeDrawable(&ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(drawable, nullptr);
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 4> green = {0.0f, 1.0f, 0.0f, 1.0f};
    ASSERT_TRUE(ringColors->upload(red.data(), {sizeof(red)}).isOk());
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
    ASSERT_NE(reusable, nullptr);
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    // before the recording, which reads the Ring buffer
    cmdQueue_->submit(*cmdBuf);

    ASSERT_FALSE(ctx.config_.chainSubmitsWithSemaphores);
    std::vector<std::unique_ptr<vulkan::VulkanSemaphore>> injected;
    for (uint32_t i = 0; i != vulkan::VulkanImmediateCommands::kMaxInjectedSemaphores; ++i) {
      injected.push_back(std::make_unique<vulkan::VulkanSemaphore>(
          ctx.vf_, ctx.getVkDevice(), uint64_t{0}, false, "Injected semaphore"));
    }
    {
      HostGate gate(*iglDev_, ctx);
      if (!gate.isClosed()) {
        GTEST_SKIP() << "Timeline semaphores are not supported";
      }
      const ScopedFenceTimeout timeout(ctx, 10'000'000);
      const PresentGateRelease releaseGate(ctx, gate);
      const uint32_t slot = ctx.currentSyncIndex();
      const auto comeBackToTheSlot = [&ctx, slot]() {
        while (ctx.currentSyncIndex() != slot) {
          ctx.markRingBufferRead();
          ctx.syncMarkSubmitted({});
        }
      };
      encodePasses(*reusable, false, ringColors.get());
      ASSERT_NE(cmdQueue_->submit(*reusable), 0u);
      comeBackToTheSlot();

      cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      encodePasses(*cmdBuf, false, ringColors.get());
      cmdBuf->present(drawable);
      for (const auto& semaphore : injected) {
        ASSERT_TRUE(ctx.immediate_->signalSemaphore(semaphore->getVkSemaphore(), 1));
      }
      const SubmitHandle submitted = cmdQueue_->submit(*cmdBuf);
      EXPECT_EQ(submitted, 0u);
      if (submitted != 0) {
        gate.open();
        return;
      }
      comeBackToTheSlot();
      EXPECT_EQ(ringColors->upload(green.data(), {sizeof(green)}).code, Result::Code::RuntimeError);
    }
    reusable->waitUntilCompleted();
    // red, with red and blue swapped
    EXPECT_EQ(readTextureB(), bluePixels());
    EXPECT_TRUE(ringColors->upload(green.data(), {sizeof(green)}).isOk());
    EXPECT_TRUE(renderFrame(*reusable, false));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), fillPixels(0xFF00FF00));
#else
    GTEST_SKIP() << "Headless swapchains are tested on Linux";
#endif // IGL_PLATFORM_LINUX
  }

  // Records the passes, which render the color of a Ring buffer that holds red, runs the slot in a
  // submission that waits for a semaphore, and comes back to the slot as runPendingSlot() does.
  // Checks that uploading green into the Ring buffer fails until the run completes, so that the run
  // renders the red of its frame, and that the next run renders green. With
  // `submitAfterExecuteCommands`, another command buffer is submitted between executeCommands() and
  // the submission of the command buffer that runs the slot, which moves the slot before the run.
  void uploadWhileTheSlotIsPending(bool executeInCommandBuffer,
                                   bool submitAfterExecuteCommands = false) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    if (!ctx.features().has_VK_KHR_timeline_semaphore || !ctx.vf_.vkSignalSemaphoreKHR) {
      GTEST_SKIP() << "Timeline semaphores are not supported";
    }
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 4> green = {0.0f, 1.0f, 0.0f, 1.0f};
    ASSERT_TRUE(ringColors->upload(red.data(), {sizeof(red)}).isOk());
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);

    // Nothing may return, or be submitted, until the host signals the semaphore (see
    // runPendingSlot())
    const vulkan::VulkanSemaphore semaphore(ctx.vf_, ctx.getVkDevice(), 0, false, "Pending slot");
    const uint64_t fenceTimeout = ctx.config_.fenceTimeoutNanoseconds;
    ctx.config_.fenceTimeoutNanoseconds = 10'000'000;
    const uint32_t slot = ctx.currentSyncIndex();
    clearTextures(*cmdBuf);
    if (executeInCommandBuffer) {
      encodePasses(*reusable, false, ringColors.get());
      Result ret;
      executeCommands(*cmdBuf, *reusable, &ret);
      EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
      if (submitAfterExecuteCommands) {
        const std::shared_ptr<ICommandBuffer> otherCmdBuf = createCommandBuffer();
        ASSERT_NE(otherCmdBuf, nullptr);
        EXPECT_NE(cmdQueue_->submit(*otherCmdBuf), 0u);
        ASSERT_NE(ctx.currentSyncIndex(), slot);
      }
    } else {
      // before the recording, which reads the Ring buffer
      cmdQueue_->submit(*cmdBuf);
      encodePasses(*reusable, false, ringColors.get());
    }
    EXPECT_TRUE(ctx.immediate_->waitSemaphore(semaphore.getVkSemaphore(), 1));
    EXPECT_NE(cmdQueue_->submit(executeInCommandBuffer ? *cmdBuf : *reusable), 0u);
    // the wait that comes back to the slot times out
    while (ctx.currentSyncIndex() != slot) {
      ctx.markRingBufferRead();
      ctx.syncMarkSubmitted({});
    }
    EXPECT_EQ(ringColors->upload(green.data(), {sizeof(green)}).code, Result::Code::RuntimeError);

    ctx.config_.fenceTimeoutNanoseconds = fenceTimeout;
    const VkSemaphoreSignalInfo signalInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = semaphore.getVkSemaphore(),
        .value = 1,
    };
    EXPECT_EQ(ctx.vf_.vkSignalSemaphoreKHR(ctx.getVkDevice(), &signalInfo), VK_SUCCESS);
    reusable->waitUntilCompleted();
    // red, with red and blue swapped
    EXPECT_EQ(readTextureB(), bluePixels());

    EXPECT_TRUE(ringColors->upload(green.data(), {sizeof(green)}).isOk());
    EXPECT_TRUE(renderFrame(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), fillPixels(0xFF00FF00));
  }

  // Records the passes, which render the color of a Ring buffer, and submits a command buffer that
  // reads no Ring buffer before their first run. Checks that the submission moves the shared slot,
  // as the first submission after a recording reads a Ring buffer does, so that the run fails with
  // Result::Code::InvalidOperation and puts back the image layouts of the recording, and that the
  // next slot records and runs the passes when nothing is submitted before the run.
  void submitBetweenARecordingAndItsRun(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
    ASSERT_TRUE(ringColors->upload(red.data(), {sizeof(red)}).isOk());
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
    const VkImageLayout layout = getImageLayout(*texA_);

    const uint32_t slot = ctx.currentSyncIndex();
    encodePasses(*reusable, false, ringColors.get());
    cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    cmdQueue_->submit(*cmdBuf);
    ASSERT_NE(ctx.currentSyncIndex(), slot);
    if (executeInCommandBuffer) {
      cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      Result ret;
      executeCommands(*cmdBuf, *reusable, &ret);
      EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();
      cmdQueue_->submit(*cmdBuf);
    } else {
      EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
    }
    EXPECT_EQ(getImageLayout(*texA_), layout);
    EXPECT_FALSE(isRecorded(*reusable));
    EXPECT_TRUE(ctx.recordingJournals_.empty());

    // the copy of the Ring buffer for the current slot
    ASSERT_TRUE(ringColors->upload(red.data(), {sizeof(red)}).isOk());
    encodePasses(*reusable, false, ringColors.get());
    EXPECT_TRUE(run(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  // Executes the passes, which render the color of a Ring buffer that holds red, from a command
  // buffer, and submits another command buffer before it, which moves the shared slot on to the
  // next frame. Checks that uploading green into the Ring buffer fails until the command buffer
  // that executes the slot is submitted, so that the run renders the red of its frame, and that the
  // next run renders green.
  void uploadWhileAnEarlierFrameIsNotSubmitted() const {
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 4> green = {0.0f, 1.0f, 0.0f, 1.0f};
    ASSERT_TRUE(ringColors->upload(red.data(), {sizeof(red)}).isOk());
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
    ASSERT_NE(reusable, nullptr);
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    encodePasses(*reusable, false, ringColors.get());
    Result ret;
    executeCommands(*cmdBuf, *reusable, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<ICommandBuffer> otherCmdBuf = createCommandBuffer();
    ASSERT_NE(otherCmdBuf, nullptr);
    EXPECT_NE(cmdQueue_->submit(*otherCmdBuf), 0u);
    EXPECT_EQ(ringColors->upload(green.data(), {sizeof(green)}).code,
              Result::Code::InvalidOperation);

    EXPECT_NE(cmdQueue_->submit(*cmdBuf), 0u);
    cmdBuf->waitUntilCompleted();
    // red, with red and blue swapped
    EXPECT_EQ(readTextureB(), bluePixels());
    EXPECT_TRUE(ringColors->upload(green.data(), {sizeof(green)}).isOk());
    EXPECT_TRUE(renderFrame(*reusable, true));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), fillPixels(0xFF00FF00));
  }

  const TextureDesc texDesc_ = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                  kSize,
                                                  kSize,
                                                  TextureDesc::TextureUsageBits::Attachment |
                                                      TextureDesc::TextureUsageBits::Sampled);
  const RenderPassDesc renderPass_{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}}};
  const RenderPassDesc loadPass_{
      .colorAttachments = {{.loadAction = LoadAction::Load, .storeAction = StoreAction::Store}}};
  const RenderPassDesc depthPass_{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}},
      .depthAttachment = {.loadAction = LoadAction::Clear,
                          .storeAction = StoreAction::DontCare,
                          .clearDepth = 1.0f}};
  void discardRecordingWithoutReplacementJournals(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    ASSERT_FALSE(ctx.config_.enableUpdateAfterBindTextures);
    const VkImageLayout layoutA = getImageLayout(*texA_);
    const VkImageLayout layoutB = getImageLayout(*texB_);
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    {
      const CommandBufferFailure failure(
          ctx, CommandBufferFailure::Call::End, executeInCommandBuffer ? 4 : 0);
      encodePasses(*reusable);
      ASSERT_EQ(ctx.recordingJournals_.size(), 1u);
      const vulkan::RecordingJournal& journal = *ctx.recordingJournals_.front();
      EXPECT_FALSE(journal.layouts.empty());
      EXPECT_TRUE(journal.imageDescriptors.empty());
      EXPECT_TRUE(journal.imageUses.empty());
      EXPECT_FALSE(run(*reusable, executeInCommandBuffer));
    }
    EXPECT_TRUE(ctx.recordingJournals_.empty());
    EXPECT_EQ(getImageLayout(*texA_), layoutA);
    EXPECT_EQ(getImageLayout(*texB_), layoutB);
    EXPECT_FALSE(isRecorded(*reusable));
    ASSERT_TRUE(renderFrame(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ITexture> texA_;
  std::shared_ptr<ITexture> texB_;
  std::shared_ptr<IFramebuffer> fbA_;
  std::shared_ptr<IFramebuffer> fbB_;
  std::shared_ptr<IBuffer> colorBuffer_;
  std::shared_ptr<ISamplerState> sampler_;
  std::shared_ptr<IRenderPipelineState> colorPipeline_;
  std::shared_ptr<IRenderPipelineState> swizzlePipeline_;
};

// Submissions that do not wait for each other with binary semaphores, so that the host keeps
// submitting while earlier submissions are pending: Lavapipe blocks the vkQueueSubmit() of a
// submission that waits for a binary semaphore until the submission that signals it has run
class ReusableCommandBufferPendingRunsVulkanTest : public ReusableCommandBufferVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.chainSubmitsWithSemaphores = false;
    return util::device::vulkan::createTestDevice(config);
  }
};

// A single frame in flight, so the shared slot comes back to every recording at the next move
class ReusableCommandBufferOneSlotVulkanTest : public ReusableCommandBufferVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.maxResourceCount = 1;
    return util::device::vulkan::createTestDevice(config);
  }
};

// A headless surface on Linux, for a swapchain whose images command buffers present
class ReusableCommandBufferSwapchainVulkanTest : public ReusableCommandBufferVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
#if IGL_PLATFORM_LINUX
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.headless = true;
    return util::device::vulkan::createTestDevice(config);
#else
    return ReusableCommandBufferVulkanTest::createDevice();
#endif // IGL_PLATFORM_LINUX
  }
};

// A single frame in flight, with a headless surface on Linux for a swapchain
class ReusableCommandBufferOneSlotSwapchainVulkanTest : public ReusableCommandBufferVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.maxResourceCount = 1;
#if IGL_PLATFORM_LINUX
    config.headless = true;
#endif // IGL_PLATFORM_LINUX
    return util::device::vulkan::createTestDevice(config);
  }
};

class ReusableCommandBufferPendingPresentVulkanTest : public ReusableCommandBufferVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.chainSubmitsWithSemaphores = false;
#if IGL_PLATFORM_LINUX
    config.headless = true;
#endif // IGL_PLATFORM_LINUX
    return util::device::vulkan::createTestDevice(config);
  }
};

class ReusableCommandBufferOneSlotPendingPresentVulkanTest
  : public ReusableCommandBufferPendingPresentVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.chainSubmitsWithSemaphores = false;
    config.maxResourceCount = 1;
#if IGL_PLATFORM_LINUX
    config.headless = true;
#endif // IGL_PLATFORM_LINUX
    return util::device::vulkan::createTestDevice(config);
  }
};

// The same, without the semaphore chain
class ReusableCommandBufferUnchainedSwapchainVulkanTest : public ReusableCommandBufferVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
#if IGL_PLATFORM_LINUX
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.headless = true;
    config.chainSubmitsWithSemaphores = false;
    return util::device::vulkan::createTestDevice(config);
#else
    return ReusableCommandBufferVulkanTest::createDevice();
#endif // IGL_PLATFORM_LINUX
  }
};

TEST_F(ReusableCommandBufferVulkanTest, SubmitReplaysRecordedPasses) {
  renderFrames(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsReplaysRecordedPasses) {
  renderFrames(true);
}

TEST_F(ReusableCommandBufferPendingRunsVulkanTest, SubmitReplaysOutstandingFrames) {
  replayOutstandingFrames(false);
}

TEST_F(ReusableCommandBufferPendingRunsVulkanTest, ExecuteCommandsReplaysOutstandingFrames) {
  replayOutstandingFrames(true);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitFailsUntilTheLastRunOfTheSlotCompletes) {
  runPendingSlot(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsFailsUntilTheLastRunOfTheSlotCompletes) {
  runPendingSlot(true);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitDiscardsAFailedRecording) {
  // the primary command buffer of the slot
  discardFailedRecording(false, CommandBufferFailure::Call::Allocate, 0);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsDiscardsAFailedRecording) {
  // the secondary command buffer of the render pass, then the one after the render pass
  discardFailedRecording(true, CommandBufferFailure::Call::Allocate, 1);
  discardFailedRecording(true, CommandBufferFailure::Call::Allocate, 2);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitDiscardsARecordingThatFailsToBegin) {
  discardFailedRecording(false, CommandBufferFailure::Call::Begin, 0);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsDiscardsARecordingThatFailsToBegin) {
  // the secondary command buffer before the render pass, the one of the render pass, then the one
  // after the render pass
  discardFailedRecording(true, CommandBufferFailure::Call::Begin, 0);
  discardFailedRecording(true, CommandBufferFailure::Call::Begin, 1);
  discardFailedRecording(true, CommandBufferFailure::Call::Begin, 2);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitDiscardsARecordingThatFailsToEnd) {
  // the primary command buffer of the slot, when it runs first
  discardFailedRecording(false, CommandBufferFailure::Call::End, 0, false);
  discardRecordingThatFailsToEnd(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsDiscardsARecordingThatFailsToEnd) {
  // the secondary command buffer before the first render pass, the one of the first render pass,
  // then, when the slot runs first, the one after the second render pass
  discardFailedRecording(true, CommandBufferFailure::Call::End, 0);
  discardFailedRecording(true, CommandBufferFailure::Call::End, 1);
  discardFailedRecording(true, CommandBufferFailure::Call::End, 4, false);
  discardRecordingThatFailsToEnd(true);
}

TEST_F(ReusableCommandBufferVulkanTest, DestructionPutsBackTheImageLayoutsOfARecording) {
  // the render pass leaves a depth attachment that is not sampled ready to be rendered to
  Result ret;
  const std::shared_ptr<ITexture> depthTex = iglDev_->createTexture(
      TextureDesc::new2D(
          TextureFormat::Z_UNorm16, kSize, kSize, TextureDesc::TextureUsageBits::Attachment),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<IFramebuffer> fb = iglDev_->createFramebuffer(
      {.colorAttachments = {{.texture = texA_}}, .depthAttachment = {.texture = depthTex}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);

  std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  {
    const std::unique_ptr<IRenderCommandEncoder> encoder =
        reusable->createRenderCommandEncoder(depthPass_, fb, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encoder->endEncoding();
  }
  EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  reusable.reset();
  EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
  EXPECT_TRUE(getVulkanContext().recordingJournals_.empty());
}

TEST_F(ReusableCommandBufferVulkanTest, ExecutingCommandBufferKeepsTheRecordingsAlive) {
  // The sequence of the review: execute the reusable command buffer from A, acquire B, release the
  // reusable command buffer, complete B and the deferred tasks that its completion lets run, then
  // submit A
  const PoolDestructions destructions(getVulkanContext());
  std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  const std::shared_ptr<ICommandBuffer> cmdBufA = createCommandBuffer();
  ASSERT_NE(cmdBufA, nullptr);
  clearTextures(*cmdBufA);
  encodePasses(*reusable);
  Result ret;
  executeCommands(*cmdBufA, *reusable, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const std::shared_ptr<ICommandBuffer> cmdBufB = createCommandBuffer();
  ASSERT_NE(cmdBufB, nullptr);
  reusable.reset();
  cmdQueue_->submit(*cmdBufB);
  cmdBufB->waitUntilCompleted();
  processDeferredTasks();
  // A references the secondary command buffers and descriptor sets of the recording
  ASSERT_EQ(destructions.numCommandPools(), 0u);
  ASSERT_EQ(destructions.numDescriptorPools(), 0u);

  cmdQueue_->submit(*cmdBufA);
  cmdBufA->waitUntilCompleted();
  EXPECT_EQ(readTextureB(), bluePixels());
  processDeferredTasks();
  EXPECT_EQ(destructions.numCommandPools(), 1u);
  EXPECT_GT(destructions.numDescriptorPools(), 0u);
}

TEST_F(ReusableCommandBufferPendingRunsVulkanTest, DestructionWaitsForTheOutstandingRuns) {
  // the runs of all the slots but one stay pending behind a gate while the reusable command buffer
  // is destroyed
  vulkan::VulkanContext& ctx = getVulkanContext();
  if (getNumSlots() < 2) {
    GTEST_SKIP() << "A single frame in flight";
  }
  HostGate gate(*iglDev_, ctx);
  if (!gate.isClosed()) {
    GTEST_SKIP() << "Timeline semaphores are not supported";
  }
  const PoolDestructions destructions(ctx);
  std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  std::shared_ptr<ICommandBuffer> cmdBuf;
  std::vector<SubmitHandle> runs;
  for (uint32_t frame = 0; frame + 1 != getNumSlots(); ++frame) {
    cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clearTextures(*cmdBuf);
    encodePasses(*reusable);
    Result ret;
    executeCommands(*cmdBuf, *reusable, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    runs.push_back(cmdQueue_->submit(*cmdBuf));
  }
  reusable.reset();
  processDeferredTasks();
  for (const SubmitHandle run : runs) {
    EXPECT_FALSE(ctx.immediate_->isReady(vulkan::VulkanImmediateCommands::SubmitHandle(run)));
  }
  EXPECT_EQ(destructions.numCommandPools(), 0u);
  EXPECT_EQ(destructions.numDescriptorPools(), 0u);

  gate.open();
  cmdBuf->waitUntilCompleted();
  processDeferredTasks();
  EXPECT_EQ(destructions.numCommandPools(), 1u);
  EXPECT_GT(destructions.numDescriptorPools(), 0u);
  EXPECT_EQ(readTextureB(), bluePixels());
}

TEST_F(ReusableCommandBufferVulkanTest, SlotOfADiscardedExecutingCommandBufferRunsAgain) {
  const PoolDestructions destructions(getVulkanContext());
  std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  std::shared_ptr<ICommandBuffer> discarded = createCommandBuffer();
  ASSERT_NE(discarded, nullptr);
  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  clearTextures(*cmdBuf);
  encodePasses(*reusable);
  Result ret;
  executeCommands(*discarded, *reusable, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // the slot runs from one command buffer at a time
  executeCommands(*cmdBuf, *reusable, &ret);
  EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();
  executeCommands(*discarded, *reusable, &ret);
  EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();

  discarded.reset();
  EXPECT_TRUE(isRecorded(*reusable));
  executeCommands(*cmdBuf, *reusable, &ret);
  EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
  EXPECT_EQ(readTextureB(), bluePixels());

  reusable.reset();
  processDeferredTasks();
  EXPECT_EQ(destructions.numCommandPools(), 1u);
}

TEST_F(ReusableCommandBufferVulkanTest, DiscardedExecutingCommandBufferReleasesTheRecordings) {
  // no submission runs the recordings, so they go when the last command buffer that holds them goes
  vulkan::VulkanContext& ctx = getVulkanContext();
  const PoolDestructions destructions(ctx);
  std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  std::shared_ptr<ICommandBuffer> discarded = createCommandBuffer();
  ASSERT_NE(discarded, nullptr);
  const vulkan::VulkanImmediateCommands::SubmitHandle handle =
      static_cast<vulkan::CommandBuffer&>(*discarded).getNextSubmitHandle();
  encodePasses(*reusable);
  Result ret;
  executeCommands(*discarded, *reusable, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  reusable.reset();
  EXPECT_EQ(destructions.numCommandPools(), 0u);
  discarded.reset();
  EXPECT_EQ(destructions.numCommandPools(), 1u);
  EXPECT_GT(destructions.numDescriptorPools(), 0u);
  // the discarded command buffer goes back to the queue, so deferred tasks stop waiting for it
  EXPECT_TRUE(ctx.immediate_->isReady(handle));
}

TEST_F(ReusableCommandBufferVulkanTest, QueueStillCountsAbandonedCommandBuffersThatExecuteNothing) {
  // An executing command buffer that is destroyed without being submitted leaves the queue's count
  // of command buffers left to submit and gives its Vulkan command buffer back; any other one stays
  // in the count and keeps its Vulkan command buffer, as without reusable command buffers
  vulkan::VulkanContext& ctx = getVulkanContext();
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  vulkan::VulkanImmediateCommands::SubmitHandle executingHandle;
  vulkan::VulkanImmediateCommands::SubmitHandle ordinaryHandle;
  {
    const std::shared_ptr<ICommandBuffer> executing = createCommandBuffer();
    ASSERT_NE(executing, nullptr);
    executingHandle = static_cast<vulkan::CommandBuffer&>(*executing).getNextSubmitHandle();
    encodePasses(*reusable);
    Result ret;
    executeCommands(*executing, *reusable, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<ICommandBuffer> ordinary = createCommandBuffer();
    ASSERT_NE(ordinary, nullptr);
    ordinaryHandle = static_cast<vulkan::CommandBuffer&>(*ordinary).getNextSubmitHandle();
  }
  EXPECT_TRUE(ctx.immediate_->isReady(executingHandle));
  EXPECT_FALSE(ctx.immediate_->isReady(ordinaryHandle));
#if IGL_DEBUG_ABORT_ENABLED
  expectedDebugAborts() = {.func = "~CommandQueue", .count = 1};
  cmdQueue_.reset();
  EXPECT_EQ(expectedDebugAborts().count, 0u);
#endif
  // the context reports the Vulkan command buffer that the other one kept when the device goes
  iglSetDebugAbortListener(nullptr);
}

TEST_F(ReusableCommandBufferVulkanTest, DestructionWaitsOnlyForTheSubmissionsThatRanTheRecordings) {
  vulkan::VulkanContext& ctx = getVulkanContext();
  std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  encodePasses(*reusable);
  Result ret;
  executeCommands(*cmdBuf, *reusable, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  // a command buffer acquired later, which is the next submission of the context, ran nothing of
  // the reusable command buffer
  const std::shared_ptr<ICommandBuffer> next = createCommandBuffer();
  ASSERT_NE(next, nullptr);
  const PoolDestructions destructions(ctx);
  reusable.reset();
  EXPECT_EQ(destructions.numCommandPools(), 1u);
  EXPECT_GT(destructions.numDescriptorPools(), 0u);
  cmdQueue_->submit(*next);
}

TEST_F(ReusableCommandBufferVulkanTest,
       ExecuteCommandsRejectsAReusableCommandBufferOfAnotherDevice) {
  const std::shared_ptr<IDevice> otherDevice = util::createTestDevice();
  ASSERT_NE(otherDevice, nullptr);
  Result ret;
  const std::shared_ptr<ICommandQueue> otherQueue = otherDevice->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<ICommandBuffer> otherReusable =
      getPlatformDevice(*otherDevice)
          .createReusableCommandBuffer({.executeInCommandBuffer = true}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<ITexture> otherTex = otherDevice->createTexture(texDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<IFramebuffer> otherFb =
      otherDevice->createFramebuffer({.colorAttachments = {{.texture = otherTex}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  clear(*otherReusable, otherFb, {0.0f, 0.0f, 1.0f, 1.0f});

  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  executeCommands(*cmdBuf, *otherReusable, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid) << ret.message.c_str();
  // the first run of the slot ends its recording, so the call ended nothing
  EXPECT_FALSE(getPlatformDevice(*otherDevice).isRecorded(*otherReusable));
  cmdQueue_->submit(*cmdBuf);

  Result otherRet;
  const std::shared_ptr<ICommandBuffer> otherCmdBuf =
      otherQueue->createCommandBuffer({}, &otherRet);
  ASSERT_TRUE(otherRet.isOk()) << otherRet.message.c_str();
  getPlatformDevice(*otherDevice).executeCommands(*otherCmdBuf, *otherReusable, &ret);
  EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
  EXPECT_TRUE(getPlatformDevice(*otherDevice).isRecorded(*otherReusable));
  otherQueue->submit(*otherCmdBuf);
  otherCmdBuf->waitUntilCompleted();
}

TEST_F(ReusableCommandBufferVulkanTest, SupportsReusableCommandBuffersExactlyWhenItCreatesThem) {
  Result ret;
  EXPECT_EQ(getPlatformDevice().supportsReusableCommandBuffers(),
            getPlatformDevice().createReusableCommandBuffer({}, &ret) != nullptr);

  // the context rewrites the descriptors of descriptor indexing while recordings could bind them
  vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
  config.enableDescriptorIndexing = true;
  const std::shared_ptr<IDevice> device = util::device::vulkan::createTestDevice(config);
  if (!device) {
    GTEST_SKIP() << "Descriptor indexing is not supported";
  }
  EXPECT_FALSE(getPlatformDevice(*device).supportsReusableCommandBuffers());
  EXPECT_EQ(getPlatformDevice(*device).createReusableCommandBuffer({}, &ret), nullptr);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitFailsWhenTheSlotMovesBeforeItsFirstRun) {
  failRecordingWhenTheSlotMoves(false, false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsFailsWhenTheSlotMovesBeforeItsFirstRun) {
  failRecordingWhenTheSlotMoves(true, false);
}

TEST_F(ReusableCommandBufferVulkanTest, EncodingFailsWhenTheSlotMovesWhileItRecords) {
  failRecordingWhenTheSlotMoves(false, true);
  failRecordingWhenTheSlotMoves(true, true);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitCountsReplayedDraws) {
  countReplayedDraws(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsCountsReplayedDraws) {
  countReplayedDraws(true);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitFailsWhileARenderPassIsEncoded) {
  runWhileARenderPassIsEncoded(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsFailsWhileARenderPassIsEncoded) {
  runWhileARenderPassIsEncoded(true);
}

TEST_F(ReusableCommandBufferVulkanTest, FailedRecordingKeepsTheLayoutsThatOthersChanged) {
  // an attachment that is not sampled stays ready to be rendered to, and uploads leave a storage
  // image in the general layout
  vulkan::VulkanContext& ctx = getVulkanContext();
  Result ret;
  const std::shared_ptr<ITexture> tex =
      iglDev_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                kSize,
                                                kSize,
                                                TextureDesc::TextureUsageBits::Attachment |
                                                    TextureDesc::TextureUsageBits::Storage),
                             &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<IFramebuffer> fb =
      iglDev_->createFramebuffer({.colorAttachments = {{.texture = tex}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  clear(*cmdBuf, fb, {1.0f, 0.0f, 1.0f, 1.0f});
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
  ASSERT_EQ(getImageLayout(*tex), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  const std::shared_ptr<ICommandBuffer> otherEngine = createReusable(false);
  ASSERT_NE(otherEngine, nullptr);
  clear(*reusable, fb, {0.0f, 0.0f, 1.0f, 1.0f});
  const Pixels pixels = bluePixels();
  ASSERT_TRUE(tex->upload(tex->getFullRange(0), pixels.data()).isOk());
  ASSERT_EQ(getImageLayout(*tex), VK_IMAGE_LAYOUT_GENERAL);
  ASSERT_NE(cmdQueue_->submit(*otherEngine), 0u);
  EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
  // the upload is the last change of the layout that ran
  EXPECT_EQ(getImageLayout(*tex), VK_IMAGE_LAYOUT_GENERAL);
  EXPECT_TRUE(ctx.recordingJournals_.empty());
}

TEST_F(ReusableCommandBufferVulkanTest, FailedRecordingKeepsAnOverriddenLayout) {
  // overrideImageLayout() reports a layout that commands outside of IGL changed
  vulkan::VulkanContext& ctx = getVulkanContext();
  Result ret;
  const std::shared_ptr<ITexture> tex = iglDev_->createTexture(
      TextureDesc::new2D(
          TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Attachment),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<IFramebuffer> fb =
      iglDev_->createFramebuffer({.colorAttachments = {{.texture = tex}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(getImageLayout(*tex), VK_IMAGE_LAYOUT_GENERAL);

  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  const std::shared_ptr<ICommandBuffer> otherEngine = createReusable(false);
  ASSERT_NE(otherEngine, nullptr);
  clear(*reusable, fb, {0.0f, 0.0f, 1.0f, 1.0f});
  vulkan::overrideImageLayout(tex.get(), VK_IMAGE_LAYOUT_GENERAL);
  ASSERT_NE(cmdQueue_->submit(*otherEngine), 0u);
  EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
  EXPECT_EQ(getImageLayout(*tex), VK_IMAGE_LAYOUT_GENERAL);
  EXPECT_TRUE(ctx.recordingJournals_.empty());
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitPutsBackTheLayoutThatAnUploadLeft) {
  failRecordingAfterAnUpload(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsPutsBackTheLayoutThatAnUploadLeft) {
  failRecordingAfterAnUpload(true);
}

TEST_F(ReusableCommandBufferVulkanTest, DiscardedRecordingsDestroyTheirDescriptorPools) {
  // every recording that fails binds more descriptor sets than a descriptor pool holds, in a slot
  // that it leaves for the next attempt
  vulkan::VulkanContext& ctx = getVulkanContext();
  const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
  Result ret;
  const std::shared_ptr<IBuffer> otherColorBuffer = iglDev_->createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Uniform,
          .data = red.data(),
          .length = sizeof(red),
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  const PoolDestructions destructions(ctx);
  for (uint32_t attempt = 0; attempt != 3; ++attempt) {
    const uint32_t numDestroyed = destructions.numDescriptorPools();
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    {
      // the primary command buffer of the slot, when it runs first
      const CommandBufferFailure failure(ctx, CommandBufferFailure::Call::End, 0);
      const std::unique_ptr<IRenderCommandEncoder> encoder =
          reusable->createRenderCommandEncoder(renderPass_, fbA_, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      encoder->bindRenderPipelineState(colorPipeline_);
      // each draw binds another buffer than the draw before it, which takes a new descriptor set
      for (uint32_t draw = 0; draw != 70; ++draw) {
        encoder->bindBuffer(0, (draw % 2 == 0 ? colorBuffer_ : otherColorBuffer).get());
        encoder->draw(3);
      }
      encoder->endEncoding();
      EXPECT_FALSE(submitAndRun(*cmdBuf, *reusable, false));
    }
    EXPECT_FALSE(isRecorded(*reusable));
    EXPECT_GE(destructions.numDescriptorPools() - numDestroyed, 2u) << "attempt " << attempt;
  }
}

TEST_F(ReusableCommandBufferVulkanTest, RecordingRunsInItsSlotAfterAFullCycleOfSlots) {
  runAfterAFullCycleOfSlots(false);
  runAfterAFullCycleOfSlots(true);
}

TEST_F(ReusableCommandBufferOneSlotVulkanTest, ReplaysRecordedPasses) {
  renderFrames(false);
  renderFrames(true);
}

TEST_F(ReusableCommandBufferOneSlotVulkanTest, RecordingRunsInItsSlotAfterAFullCycleOfSlots) {
  runAfterAFullCycleOfSlots(false);
  runAfterAFullCycleOfSlots(true);
}

TEST_F(ReusableCommandBufferOneSlotVulkanTest, RingUploadFailsUntilAnEarlierFrameIsSubmitted) {
  uploadWhileAnEarlierFrameIsNotSubmitted();
}

TEST_F(ReusableCommandBufferPendingRunsVulkanTest, DestructionDoesNotWaitForLaterCommandBuffers) {
  // The recordings go once the submission that ran them completes, while a command buffer acquired
  // after that submission, which deferred tasks wait for by default, stays unsubmitted
  vulkan::VulkanContext& ctx = getVulkanContext();
  HostGate gate(*iglDev_, ctx);
  if (!gate.isClosed()) {
    GTEST_SKIP() << "Timeline semaphores are not supported";
  }
  const PoolDestructions destructions(ctx);
  std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  clearTextures(*cmdBuf);
  encodePasses(*reusable);
  Result ret;
  executeCommands(*cmdBuf, *reusable, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const SubmitHandle run = cmdQueue_->submit(*cmdBuf);
  const std::shared_ptr<ICommandBuffer> later = createCommandBuffer();
  ASSERT_NE(later, nullptr);
  reusable.reset();
  EXPECT_FALSE(ctx.immediate_->isReady(vulkan::VulkanImmediateCommands::SubmitHandle(run)));
  EXPECT_EQ(destructions.numCommandPools(), 0u);

  gate.open();
  cmdBuf->waitUntilCompleted();
  processDeferredTasks();
  EXPECT_EQ(destructions.numCommandPools(), 1u);
  EXPECT_GT(destructions.numDescriptorPools(), 0u);
  EXPECT_EQ(readTextureB(), bluePixels());
  cmdQueue_->submit(*later);
}

TEST_F(ReusableCommandBufferPendingRunsVulkanTest, DiscardingASlotKeepsTheDescriptorPoolsOfOthers) {
  // the run of a slot stays pending behind a gate while the recording of the next slot fails
  vulkan::VulkanContext& ctx = getVulkanContext();
  if (getNumSlots() < 2) {
    GTEST_SKIP() << "A single frame in flight";
  }
  HostGate gate(*iglDev_, ctx);
  if (!gate.isClosed()) {
    GTEST_SKIP() << "Timeline semaphores are not supported";
  }
  const PoolDestructions destructions(ctx);
  std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  clearTextures(*cmdBuf);
  encodePasses(*reusable);
  Result ret;
  executeCommands(*cmdBuf, *reusable, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const SubmitHandle run = cmdQueue_->submit(*cmdBuf);

  const std::shared_ptr<ICommandBuffer> failedCmdBuf = createCommandBuffer();
  ASSERT_NE(failedCmdBuf, nullptr);
  {
    // the secondary command buffer after the second render pass, when the slot runs
    const CommandBufferFailure failure(ctx, CommandBufferFailure::Call::End, 4);
    encodePasses(*reusable);
    executeCommands(*failedCmdBuf, *reusable, &ret);
    EXPECT_EQ(ret.code, Result::Code::RuntimeError) << ret.message.c_str();
  }
  cmdQueue_->submit(*failedCmdBuf);
  EXPECT_FALSE(ctx.immediate_->isReady(vulkan::VulkanImmediateCommands::SubmitHandle(run)));
  const uint32_t numDiscarded = destructions.numDescriptorPools();
  EXPECT_GT(numDiscarded, 0u);

  gate.open();
  failedCmdBuf->waitUntilCompleted();
  EXPECT_EQ(readTextureB(), bluePixels());
  // the descriptor pools of the slot that ran go with the reusable command buffer
  reusable.reset();
  processDeferredTasks();
  EXPECT_EQ(destructions.numCommandPools(), 1u);
  EXPECT_GT(destructions.numDescriptorPools(), numDiscarded);
}

TEST_F(ReusableCommandBufferVulkanTest, RejectsTimestampQueries) {
  Result ret;
  const std::shared_ptr<ITimestampQueries> queries = iglDev_->createTimestampQueries(2, &ret);
  if (!queries) {
    GTEST_SKIP() << "Timestamp queries are not supported";
  }
  const std::shared_ptr<ICommandBuffer> reusable =
      getPlatformDevice().createReusableCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(reusable, nullptr);

  RenderPassDesc renderPass = renderPass_;
  renderPass.timestampQuery.queries = queries;
  EXPECT_EQ(reusable->createRenderCommandEncoder(renderPass, fbA_, &ret), nullptr);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);
}

TEST_F(ReusableCommandBufferSwapchainVulkanTest,
       SubmitLeavesAnOlderCommandBufferThatFailsToPresentToDiscard) {
  failPresentOfAnOlderCommandBuffer(false, true);
  // the context reports the Vulkan command buffers that discard() kept when the device goes
  iglSetDebugAbortListener(nullptr);
}

TEST_F(ReusableCommandBufferSwapchainVulkanTest,
       ExecuteCommandsGivesBackAnOlderCommandBufferThatFailsToPresent) {
  failPresentOfAnOlderCommandBuffer(true, true);
}

TEST_F(ReusableCommandBufferSwapchainVulkanTest,
       ExecuteCommandsGivesBackAnOlderCommandBufferThatFailsToPresentWithBinarySemaphores) {
  failPresentOfAnOlderCommandBuffer(true, false);
}

TEST_F(ReusableCommandBufferUnchainedSwapchainVulkanTest, AFrameThatIsNotSubmittedIsNotPresented) {
#if IGL_PLATFORM_LINUX
  vulkan::VulkanContext& ctx = getVulkanContext();
  ASSERT_TRUE(ctx.initSwapchain(kSize, kSize).isOk());
  if (!ctx.timelineSemaphore_) {
    GTEST_SKIP() << "Timeline semaphores are not supported";
  }
  const ScopedSwapchainCalls calls(ctx);
  Result ret;
  const std::shared_ptr<ITexture> drawable =
      getPlatformDevice().createTextureFromNativeDrawable(&ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_EQ(gSwapchainCalls.acquires, 1u);
  // The next submission waits for the acquired image, so the frame carries no injected wait.
  const std::shared_ptr<ICommandBuffer> first = createCommandBuffer();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(cmdQueue_->submit(*first), 0u);

  // Both tries to submit the frame run out of memory: it is not submitted, and it is not
  // presented, as nothing signals its presentation semaphore. The image stays acquired.
  const std::shared_ptr<ICommandBuffer> dropped = createCommandBuffer();
  ASSERT_NE(dropped, nullptr);
  dropped->present(drawable);
  gSwapchainCalls.submitsRunOutOfMemory = true;
  gSwapchainCalls.presentsForbidden = true;
#if IGL_DEBUG_ABORT_ENABLED
  // Each failed submission asserts, as it always did.
  expectedDebugAborts() = {.func = "VulkanImmediateCommands::submit", .count = 2u};
#endif
  EXPECT_NE(cmdQueue_->submit(*dropped), 0u);
#if IGL_DEBUG_ABORT_ENABLED
  EXPECT_EQ(expectedDebugAborts().count, 0u);
#endif
  gSwapchainCalls.submitsRunOutOfMemory = false;
  gSwapchainCalls.presentsForbidden = false;
  // Acquiring another image would wait for the dropped frame's timeline signal.
  ASSERT_EQ(gSwapchainCalls.presents, 0u);

  // Without the chain or an injected wait, nothing waits on the dropped frame: the next frame is
  // submitted and presents the same image, which is not acquired again.
  const std::shared_ptr<ITexture> sameDrawable =
      getPlatformDevice().createTextureFromNativeDrawable(&ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  EXPECT_EQ(gSwapchainCalls.acquires, 1u);
  const std::shared_ptr<ICommandBuffer> next = createCommandBuffer();
  ASSERT_NE(next, nullptr);
  next->present(sameDrawable);
  const uint32_t submits = gSwapchainCalls.submits;
  EXPECT_NE(cmdQueue_->submit(*next), 0u);
  EXPECT_EQ(gSwapchainCalls.submits, submits + 1);
  EXPECT_EQ(gSwapchainCalls.presentedImages, gSwapchainCalls.acquiredImages);
  next->waitUntilCompleted();
#else
  GTEST_SKIP() << "Headless swapchains are tested on Linux";
#endif // IGL_PLATFORM_LINUX
}

TEST_F(ReusableCommandBufferSwapchainVulkanTest,
       AFrameThatIsNotSubmittedStopsTheQueueAndNothingIsPresented) {
#if IGL_PLATFORM_LINUX
  vulkan::VulkanContext& ctx = getVulkanContext();
  ASSERT_TRUE(ctx.initSwapchain(kSize, kSize).isOk());
  const ScopedSwapchainCalls calls(ctx);
  Result ret;
  const std::shared_ptr<ITexture> drawable =
      getPlatformDevice().createTextureFromNativeDrawable(&ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_EQ(gSwapchainCalls.acquires, 1u);

  // The frame waits for the acquired image, and the next submission would wait for the frame: when
  // both tries to submit it run out of memory, nothing is presented, and the queue stops.
  const std::shared_ptr<ICommandBuffer> dropped = createCommandBuffer();
  ASSERT_NE(dropped, nullptr);
  dropped->present(drawable);
  gSwapchainCalls.submitsRunOutOfMemory = true;
  gSwapchainCalls.presentsForbidden = true;
#if IGL_DEBUG_ABORT_ENABLED
  // Each failed submission asserts, as it always did.
  expectedDebugAborts() = {.func = "VulkanImmediateCommands::submit", .count = 2u};
#endif
  EXPECT_NE(cmdQueue_->submit(*dropped), 0u);
#if IGL_DEBUG_ABORT_ENABLED
  EXPECT_EQ(expectedDebugAborts().count, 0u);
#endif
  gSwapchainCalls.submitsRunOutOfMemory = false;
  // Acquiring another image would wait for the dropped frame's timeline signal.
  ASSERT_EQ(gSwapchainCalls.presents, 0u);
  const uint32_t submits = gSwapchainCalls.submits;
  for (uint32_t frame = 0; frame != 3; ++frame) {
    const std::shared_ptr<ITexture> sameDrawable =
        getPlatformDevice().createTextureFromNativeDrawable(&ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<ICommandBuffer> later = createCommandBuffer();
    ASSERT_NE(later, nullptr);
    later->present(sameDrawable);
    EXPECT_NE(cmdQueue_->submit(*later), 0u) << frame;
    ASSERT_EQ(gSwapchainCalls.presents, 0u) << frame;
  }
  gSwapchainCalls.presentsForbidden = false;
  EXPECT_EQ(gSwapchainCalls.submits, submits);
  EXPECT_EQ(gSwapchainCalls.acquires, 1u);
#else
  GTEST_SKIP() << "Headless swapchains are tested on Linux";
#endif // IGL_PLATFORM_LINUX
}

TEST_F(ReusableCommandBufferPendingPresentVulkanTest,
       SubmitRingUploadFailsAfterADiscardedPresentWhileTheSlotIsPending) {
  uploadAfterADiscardedPresent();
}

TEST_F(ReusableCommandBufferOneSlotPendingPresentVulkanTest,
       SubmitRingUploadFailsAfterADiscardedPresentWhileTheSlotIsPending) {
  uploadAfterADiscardedPresent();
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitFailsARecordingThatGeneratesSrgbMipmaps) {
  failRecordingThatGeneratesSrgbMipmaps(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsFailsARecordingThatGeneratesSrgbMipmaps) {
  failRecordingThatGeneratesSrgbMipmaps(true);
}

TEST_F(ReusableCommandBufferVulkanTest, SrgbMipmapsRecordNothingIntoASlot) {
  generateSrgbMipmapsIntoASlot(false);
  generateSrgbMipmapsIntoASlot(true);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitReplaysGeneratedMipmaps) {
  replayGeneratedMipmaps(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsReplaysGeneratedMipmaps) {
  replayGeneratedMipmaps(true);
}

TEST_F(ReusableCommandBufferVulkanTest, HelpersRejectTheCommandBuffersOfAnotherDevice) {
  // the context of the other device rewrites the descriptors of descriptor indexing, so it has no
  // reusable command buffers
  vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
  config.enableDescriptorIndexing = true;
  const std::shared_ptr<IDevice> otherDevice = util::device::vulkan::createTestDevice(config);
  if (!otherDevice) {
    GTEST_SKIP() << "Descriptor indexing is not supported";
  }
  vulkan::PlatformDevice& otherPlatformDevice = getPlatformDevice(*otherDevice);
  ASSERT_FALSE(otherPlatformDevice.supportsReusableCommandBuffers());
  Result ret;
  const std::shared_ptr<ICommandQueue> otherQueue = otherDevice->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<ICommandBuffer> otherCmdBuf = otherQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const std::shared_ptr<ICommandBuffer> reusable = createReusable(true);
  ASSERT_NE(reusable, nullptr);
  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  clearTextures(*cmdBuf);
  encodePasses(*reusable);
  const auto& vkReusable = static_cast<const vulkan::CommandBuffer&>(*reusable);
  EXPECT_TRUE(vkReusable.belongsTo(getVulkanContext()));
  EXPECT_FALSE(vkReusable.belongsTo(static_cast<vulkan::Device&>(*otherDevice).getVulkanContext()));

  // either command buffer of another device fails the call before it records anything
  otherPlatformDevice.executeCommands(*cmdBuf, *reusable, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid) << ret.message.c_str();
  otherPlatformDevice.executeCommands(*otherCmdBuf, *reusable, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid) << ret.message.c_str();
  getPlatformDevice().executeCommands(*otherCmdBuf, *reusable, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid) << ret.message.c_str();
  // the first run of the slot ends its recording, so the calls ended nothing
  EXPECT_FALSE(isRecorded(*reusable));

  executeCommands(*cmdBuf, *reusable, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  EXPECT_TRUE(isRecorded(*reusable));
  EXPECT_FALSE(otherPlatformDevice.isRecorded(*reusable));
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
  EXPECT_EQ(readTextureB(), bluePixels());
  otherQueue->submit(*otherCmdBuf);
}

TEST_F(ReusableCommandBufferVulkanTest, RingUploadIgnoresAFailedWaitWithoutReusableCommandBuffers) {
  // A device that never creates a reusable command buffer keeps the behavior that IGL has without
  // replay: after a wait for the submission that read the copy of the current slot fails, an upload
  // still writes that copy
  vulkan::VulkanContext& ctx = getVulkanContext();
  if (!ctx.features().has_VK_KHR_timeline_semaphore || !ctx.vf_.vkSignalSemaphoreKHR) {
    GTEST_SKIP() << "Timeline semaphores are not supported";
  }
  const std::shared_ptr<IBuffer> ringColors =
      createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
  ASSERT_NE(ringColors, nullptr);
  const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
  const std::array<float, 4> green = {0.0f, 1.0f, 0.0f, 1.0f};
  ASSERT_TRUE(ringColors->upload(red.data(), {sizeof(red)}).isOk());
  const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
  ASSERT_NE(cmdBuf, nullptr);
  encodePasses(*cmdBuf, false, ringColors.get());

  const vulkan::VulkanSemaphore semaphore(ctx.vf_, ctx.getVkDevice(), 0, false, "Pending copy");
  const uint64_t fenceTimeout = ctx.config_.fenceTimeoutNanoseconds;
  ctx.config_.fenceTimeoutNanoseconds = 10'000'000;
  const uint32_t slot = ctx.currentSyncIndex();
  EXPECT_TRUE(ctx.immediate_->waitSemaphore(semaphore.getVkSemaphore(), 1));
  EXPECT_NE(cmdQueue_->submit(*cmdBuf), 0u);
  // the wait that comes back to the copy times out
  while (ctx.currentSyncIndex() != slot) {
    ctx.markRingBufferRead();
    ctx.syncMarkSubmitted({});
  }
  EXPECT_TRUE(ringColors->upload(green.data(), {sizeof(green)}).isOk());
  EXPECT_FALSE(ctx.hasReusableCommandBuffers_);

  ctx.config_.fenceTimeoutNanoseconds = fenceTimeout;
  const VkSemaphoreSignalInfo signalInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = semaphore.getVkSemaphore(),
      .value = 1,
  };
  EXPECT_EQ(ctx.vf_.vkSignalSemaphoreKHR(ctx.getVkDevice(), &signalInfo), VK_SUCCESS);
  cmdBuf->waitUntilCompleted();
  // the gated submission read the copy that the upload wrote: green, with red and blue swapped
  EXPECT_EQ(readTextureB(), fillPixels(0xFF00FF00));
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitRingUploadFailsWhileTheSlotIsPending) {
  uploadWhileTheSlotIsPending(false);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsRingUploadFailsWhileTheSlotIsPending) {
  uploadWhileTheSlotIsPending(true);
}

TEST_F(ReusableCommandBufferVulkanTest, ExecuteCommandsRingUploadWaitsForARunAfterTheSlotMoves) {
  uploadWhileTheSlotIsPending(true, true);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitFailsWhenASubmissionFollowsARecordingThatReadsARing) {
  submitBetweenARecordingAndItsRun(false);
}

TEST_F(ReusableCommandBufferVulkanTest,
       ExecuteCommandsFailsWhenASubmissionFollowsARecordingThatReadsARing) {
  submitBetweenARecordingAndItsRun(true);
}

TEST_F(ReusableCommandBufferVulkanTest, ReplaceTextureIsUnsupportedWithoutUpdateAfterBindTextures) {
  EXPECT_FALSE(getPlatformDevice().supportsTextureReplacement());
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  Result ret;
  replaceTexture(*reusable, *texA_, texB_, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);
}

TEST_F(ReusableCommandBufferVulkanTest, SupportsTextureReplacementExactlyWhenItReplacesTextures) {
  for (const bool enableUpdateAfterBindTextures : {false, true}) {
    for (const bool enableDescriptorIndexing : {false, true}) {
      vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
      config.enableUpdateAfterBindTextures = enableUpdateAfterBindTextures;
      config.enableDescriptorIndexing = enableDescriptorIndexing;
      const std::shared_ptr<IDevice> device = util::device::vulkan::createTestDevice(config);
      ASSERT_NE(device, nullptr);
      const vulkan::VulkanContext& ctx = static_cast<vulkan::Device&>(*device).getVulkanContext();
      const bool canRecord = !enableDescriptorIndexing &&
                             !ctx.features().has_VK_EXT_descriptor_buffer;
      const bool expected = !IGL_PLATFORM_ANDROID && enableUpdateAfterBindTextures && canRecord &&
                            hasUpdateAfterBindImageFeatures(ctx);
      const vulkan::PlatformDevice& platformDevice = getPlatformDevice(*device);
      EXPECT_EQ(platformDevice.supportsTextureReplacement(), expected);
      EXPECT_EQ(ctx.supportsUpdateAfterBindTextures(), expected);
      Result ret;
      const std::shared_ptr<ICommandBuffer> reusable =
          platformDevice.createReusableCommandBuffer({}, &ret);
      ASSERT_EQ(reusable != nullptr, canRecord) << ret.message.c_str();
      if (!reusable) {
        continue;
      }
      const std::shared_ptr<ITexture> texture = device->createTexture(texDesc_, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      ASSERT_NE(texture, nullptr);
      const std::shared_ptr<ITexture> otherTexture = device->createTexture(texDesc_, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      ASSERT_NE(otherTexture, nullptr);
      platformDevice.replaceTexture(*reusable, *texture, otherTexture, &ret);
      EXPECT_EQ(ret.code, expected ? Result::Code::Ok : Result::Code::Unsupported)
          << ret.message.c_str();
    }
  }
}

// Replays whose recordings read the textures of each frame, on a device whose image descriptors can
// be updated after they are bound
class ReplaceTextureVulkanTest : public ReusableCommandBufferVulkanTest {
 public:
  void SetUp() override {
    ReusableCommandBufferVulkanTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
#if IGL_PLATFORM_ANDROID
    GTEST_SKIP() << "Texture replacement is unsupported on Android";
#else
    if (!hasUpdateAfterBindImageFeatures(getVulkanContext())) {
      GTEST_SKIP() << "Image descriptors cannot be updated after they are bound";
    }
    ASSERT_FALSE(getVulkanContext().config_.enableDescriptorIndexing);
    ASSERT_FALSE(getVulkanContext().features().has_VK_EXT_descriptor_buffer);
    ASSERT_TRUE(getPlatformDevice().supportsTextureReplacement());
    addPipeline_ = createPipeline(kAddShader);
    ASSERT_NE(addPipeline_, nullptr);
#endif // IGL_PLATFORM_ANDROID
  }

  void TearDown() override {
    // before the device goes
    addPipeline_.reset();
    ReusableCommandBufferVulkanTest::TearDown();
  }

 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.enableUpdateAfterBindTextures = true;
    config.enableDescriptorIndexing = false;
    return util::device::vulkan::createTestDevice(config);
  }

  // A texture whose pixels are all `pixel`
  [[nodiscard]] std::shared_ptr<ITexture> createFilledTexture(uint32_t pixel) const {
    Result ret;
    std::shared_ptr<ITexture> texture = iglDev_->createTexture(
        TextureDesc::new2D(
            TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Sampled),
        &ret);
    if (!ret.isOk()) {
      return nullptr;
    }
    Pixels pixels = {};
    pixels.fill(pixel);
    return texture->upload(texture->getFullRange(0), pixels.data()).isOk() ? texture : nullptr;
  }

  // Renders the color of `colors` into texture A, then adds texture A and `frameTexture` into
  // texture B
  void encodeAddPasses(ICommandBuffer& cmdBuf, IBuffer& colors, ITexture& frameTexture) const {
    Result ret;
    auto encoder = cmdBuf.createRenderCommandEncoder(renderPass_, fbA_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encoder->bindRenderPipelineState(colorPipeline_);
    encoder->bindBuffer(0, &colors);
    encoder->draw(3);
    encoder->endEncoding();

    encoder = cmdBuf.createRenderCommandEncoder(renderPass_, fbB_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encoder->bindRenderPipelineState(addPipeline_);
    encoder->bindTexture(0, BindTarget::kFragment, texA_.get());
    encoder->bindTexture(1, BindTarget::kFragment, &frameTexture);
    encoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());
    encoder->bindSamplerState(1, BindTarget::kFragment, sampler_.get());
    encoder->draw(3);
    encoder->endEncoding();
  }

  // Renders the passes over 3 frames per slot with a reusable command buffer whose recordings read
  // the color from a Ring buffer and the frame texture through a texture that no frame uses, and
  // checks every frame against a command buffer that encodes the passes with the frame's color and
  // texture again. The frame texture changes every frame. Every frame uploads its own color but the
  // last frame of each slot, which reads the color that the copy of its slot holds.
  void renderFrames(bool executeInCommandBuffer) const {
    const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
    ASSERT_NE(colors, nullptr);
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    // green, which no frame texture has
    const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
    ASSERT_NE(recorded, nullptr);
    // shades of blue
    std::array<std::shared_ptr<ITexture>, 4> frameTextures;
    for (uint32_t i = 0; i != frameTextures.size(); ++i) {
      frameTextures[i] = createFilledTexture(0xFF000000 | ((51 * (i + 1)) << 16));
      ASSERT_NE(frameTextures[i], nullptr);
    }
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);

    const uint32_t numSlots = getVulkanContext().config_.maxResourceCount;
    uint32_t numRecordings = 0;
    for (uint32_t frame = 0; frame != 3 * numSlots; ++frame) {
      const bool uploadsColor = frame < 2 * numSlots;
      // shades of red, one per frame that uploads its color
      const uint32_t colorIndex = uploadsColor ? frame : frame - numSlots;
      const std::array<float, 4> color = {
          static_cast<float>(8 * (colorIndex + 1)) / 255.0f, 0.0f, 0.0f, 1.0f};
      const std::shared_ptr<ITexture>& frameTexture = frameTextures[frame % frameTextures.size()];

      ASSERT_TRUE(colors->upload(color.data(), {sizeof(color)}).isOk());
      Result ret;
      std::shared_ptr<ICommandBuffer> cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      clearTextures(*cmdBuf);
      encodeAddPasses(*cmdBuf, *colors, *frameTexture);
      cmdQueue_->submit(*cmdBuf);
      cmdBuf->waitUntilCompleted();
      const Pixels expected = readTextureB();

      cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      clearTextures(*cmdBuf);
      cmdQueue_->submit(*cmdBuf);
      if (uploadsColor) {
        ASSERT_TRUE(ringColors->upload(color.data(), {sizeof(color)}).isOk());
      }
      replaceTexture(*reusable, *recorded, frameTexture, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      if (!isRecorded(*reusable)) {
        ++numRecordings;
        encodeAddPasses(*reusable, *ringColors, *recorded);
      }
      EXPECT_TRUE(run(*reusable, executeInCommandBuffer)) << "frame " << frame;
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTextureB(), expected) << "frame " << frame;
    }
    EXPECT_EQ(numRecordings, numSlots);
  }

  [[nodiscard]] std::shared_ptr<IComputePipelineState> createComputePipeline(
      const char* shader) const {
    Result ret;
    std::shared_ptr<IShaderStages> stages =
        ShaderStagesCreator::fromModuleStringInput(*iglDev_, shader, "main", "", &ret);
    if (!ret.isOk()) {
      return nullptr;
    }
    return iglDev_->createComputePipeline({.shaderStages = std::move(stages)}, &ret);
  }

  [[nodiscard]] std::shared_ptr<ITexture> createTexture(const TextureDesc& desc) const {
    Result ret;
    std::shared_ptr<ITexture> texture = iglDev_->createTexture(desc, &ret);
    return ret.isOk() ? texture : nullptr;
  }

  // A texture that compute shaders write as a storage image, and that is read back as an attachment
  [[nodiscard]] std::shared_ptr<ITexture> createStorageTexture(
      TextureDesc::TextureUsage usage = TextureDesc::TextureUsageBits::Storage |
                                        TextureDesc::TextureUsageBits::Attachment) const {
    return createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8, kSize, kSize, usage));
  }

  // Gets `textures` ready to be read and written by compute shaders as storage images, as the
  // recordings that write them leave them, unless they can be sampled
  void makeStorageImages(const std::vector<std::shared_ptr<ITexture>>& textures) const {
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    {
      const std::unique_ptr<IComputeCommandEncoder> encoder = cmdBuf->createComputeCommandEncoder();
      ASSERT_NE(encoder, nullptr);
      for (const std::shared_ptr<ITexture>& texture : textures) {
        encoder->bindImageTexture(0, texture.get(), TextureFormat::RGBA_UNorm8);
      }
      encoder->endEncoding();
    }
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
  }

  // The pixels of `texture`, an attachment
  [[nodiscard]] Pixels readTexture(const std::shared_ptr<ITexture>& texture) const {
    Pixels pixels = {};
    Result ret;
    const std::shared_ptr<IFramebuffer> fb =
        iglDev_->createFramebuffer({.colorAttachments = {{.texture = texture}}}, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
    if (fb) {
      fb->copyBytesColorAttachment(*cmdQueue_, 0, pixels.data(), texture->getFullRange(0));
    }
    return pixels;
  }

  // Stores the color of `colors` into every pixel of `image`, with `pipeline` of kStoreShader
  static void encodeStore(ICommandBuffer& cmdBuf,
                          const std::shared_ptr<IComputePipelineState>& pipeline,
                          IBuffer& colors,
                          ITexture& image) {
    const std::unique_ptr<IComputeCommandEncoder> encoder = cmdBuf.createComputeCommandEncoder();
    ASSERT_NE(encoder, nullptr);
    encoder->bindComputePipelineState(pipeline);
    encoder->bindBuffer(0, &colors);
    encoder->bindImageTexture(0, &image, TextureFormat::RGBA_UNorm8);
    encoder->dispatchThreadGroups({kSize, kSize, 1}, {1, 1, 1});
    encoder->endEncoding();
  }

  // Stores the sum of `texture` and the color of `colors` into every pixel of `image`, with
  // `pipeline` of kAddColorShader
  static void encodeAddColor(ICommandBuffer& cmdBuf,
                             const std::shared_ptr<IComputePipelineState>& pipeline,
                             ITexture& texture,
                             IBuffer& colors,
                             ITexture& image) {
    const std::unique_ptr<IComputeCommandEncoder> encoder = cmdBuf.createComputeCommandEncoder();
    ASSERT_NE(encoder, nullptr);
    encoder->bindComputePipelineState(pipeline);
    encoder->bindTexture(0, &texture);
    encoder->bindBuffer(0, &colors);
    encoder->bindImageTexture(0, &image, TextureFormat::RGBA_UNorm8);
    encoder->dispatchThreadGroups({kSize, kSize, 1}, {1, 1, 1});
    encoder->endEncoding();
  }

  // Returns the result of replacing `recorded` with `texture` in a reusable command buffer that has
  // recorded nothing
  [[nodiscard]] Result replace(const ITexture& recorded,
                               const std::shared_ptr<ITexture>& texture) const {
    Result ret(Result::Code::RuntimeError, "No reusable command buffer");
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
    if (reusable) {
      replaceTexture(*reusable, recorded, texture, &ret);
    }
    return ret;
  }

  // Stores the color of a Ring buffer into a frame texture over 3 frames per slot, each with its
  // own color and another frame texture than the frame before, with a reusable command buffer whose
  // recordings write the frame texture through a texture that no frame uses, and checks the frame
  // texture after every frame
  void storeFrames(bool executeInCommandBuffer) const {
    const std::shared_ptr<IComputePipelineState> pipeline = createComputePipeline(kStoreShader);
    ASSERT_NE(pipeline, nullptr);
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    const std::shared_ptr<ITexture> recorded = createStorageTexture();
    ASSERT_NE(recorded, nullptr);
    std::vector<std::shared_ptr<ITexture>> frameTextures(4);
    for (std::shared_ptr<ITexture>& texture : frameTextures) {
      texture = createStorageTexture();
      ASSERT_NE(texture, nullptr);
    }
    std::vector<std::shared_ptr<ITexture>> storageImages = frameTextures;
    storageImages.push_back(recorded);
    makeStorageImages(storageImages);
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);

    for (uint32_t frame = 0; frame != 3 * getNumSlots(); ++frame) {
      const uint32_t red = 8 * (frame + 1);
      const std::array<float, 4> color = {static_cast<float>(red) / 255.0f, 0.0f, 0.0f, 1.0f};
      const std::shared_ptr<ITexture>& frameTexture = frameTextures[frame % frameTextures.size()];

      ASSERT_TRUE(ringColors->upload(color.data(), {sizeof(color)}).isOk());
      Result ret;
      replaceTexture(*reusable, *recorded, frameTexture, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      if (!isRecorded(*reusable)) {
        encodeStore(*reusable, pipeline, *ringColors, *recorded);
      }
      EXPECT_TRUE(run(*reusable, executeInCommandBuffer)) << "frame " << frame;
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTexture(frameTexture), fillPixels(0xFF000000 | red)) << "frame " << frame;
    }
  }

  // Records a compute pass that stores the color of a Ring buffer into a storage image, and one
  // that copies that image into an output image, and runs them over 3 frames per slot with another
  // texture in place of the first image in every frame. Checks that the recording has a memory
  // dependency between the passes that covers any image, so that the second pass of a run reads
  // what the first one wrote into the texture that replaces the image, and checks the output of
  // every frame.
  void replayDependentPasses(bool executeInCommandBuffer) const {
    const std::shared_ptr<IComputePipelineState> store = createComputePipeline(kStoreShader);
    ASSERT_NE(store, nullptr);
    const std::shared_ptr<IComputePipelineState> copy = createComputePipeline(kCopyImageShader);
    ASSERT_NE(copy, nullptr);
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    // the images that replace the recorded one are storage images only, while the recorded one is
    // read back at the end
    const std::vector<std::shared_ptr<ITexture>> images = {
        createStorageTexture(),
        createStorageTexture(TextureDesc::TextureUsageBits::Storage),
        createStorageTexture(TextureDesc::TextureUsageBits::Storage),
    };
    for (const std::shared_ptr<ITexture>& image : images) {
      ASSERT_NE(image, nullptr);
    }
    const std::shared_ptr<ITexture> output = createStorageTexture();
    ASSERT_NE(output, nullptr);
    std::vector<std::shared_ptr<ITexture>> storageImages = images;
    storageImages.push_back(output);
    makeStorageImages(storageImages);
    const std::shared_ptr<ITexture>& recorded = images[0];
    // green, which no frame stores
    const std::shared_ptr<IBuffer> greenColor = createColorBuffer(0);
    ASSERT_NE(greenColor, nullptr);
    const std::array<float, 4> green = {0.0f, 1.0f, 0.0f, 1.0f};
    ASSERT_TRUE(greenColor->upload(green.data(), {sizeof(green)}).isOk());
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    encodeStore(*cmdBuf, store, *greenColor, *recorded);
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);

    for (uint32_t frame = 0; frame != 3 * getNumSlots(); ++frame) {
      const uint32_t red = 8 * (frame + 1);
      const std::array<float, 4> color = {static_cast<float>(red) / 255.0f, 0.0f, 0.0f, 1.0f};
      ASSERT_TRUE(ringColors->upload(color.data(), {sizeof(color)}).isOk());
      Result ret;
      replaceTexture(*reusable, *recorded, images[1 + frame % 2], &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      if (!isRecorded(*reusable)) {
        const MemoryBarrierRecorder barriers(getVulkanContext());
        encodeStore(*reusable, store, *ringColors, *recorded);
        {
          const std::unique_ptr<IComputeCommandEncoder> encoder =
              reusable->createComputeCommandEncoder();
          ASSERT_NE(encoder, nullptr);
          encoder->bindComputePipelineState(copy);
          encoder->bindImageTexture(0, recorded.get(), TextureFormat::RGBA_UNorm8);
          encoder->bindImageTexture(1, output.get(), TextureFormat::RGBA_UNorm8);
          encoder->dispatchThreadGroups({kSize, kSize, 1}, {1, 1, 1});
          encoder->endEncoding();
        }
        EXPECT_TRUE(barriers.coversComputeShaderWritesBetweenDispatches()) << "frame " << frame;
      }
      EXPECT_TRUE(run(*reusable, executeInCommandBuffer)) << "frame " << frame;
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTexture(output), fillPixels(0xFF000000 | red)) << "frame " << frame;
    }
    // the runs wrote the textures that replaced the recorded image, never the recorded image
    EXPECT_EQ(readTexture(recorded), fillPixels(0xFF00FF00));
  }

  // Replaces the frame texture of the passes with a texture that is not ready to be sampled, and
  // checks that the first run fails with Result::Code::InvalidOperation and fails the recording,
  // which puts back at once the layout of a depth attachment that the recording renders to. Once
  // the texture is uploaded, which leaves it ready to be sampled, the slot records the passes
  // again, which run. A first reusable command buffer is destroyed after the failed run instead,
  // which changes no layout.
  void runWithAReplacementThatIsNotReady(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
    ASSERT_NE(colors, nullptr);
    const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
    ASSERT_TRUE(colors->upload(red.data(), {sizeof(red)}).isOk());
    const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
    ASSERT_NE(recorded, nullptr);
    const std::shared_ptr<ITexture> texture = createTexture(TextureDesc::new2D(
        TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Sampled));
    ASSERT_NE(texture, nullptr);
    const std::shared_ptr<ITexture> depthTex = createTexture(TextureDesc::new2D(
        TextureFormat::Z_UNorm16, kSize, kSize, TextureDesc::TextureUsageBits::Attachment));
    ASSERT_NE(depthTex, nullptr);
    Result ret;
    const std::shared_ptr<IFramebuffer> depthFb = iglDev_->createFramebuffer(
        {.colorAttachments = {{.texture = texA_}}, .depthAttachment = {.texture = depthTex}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
    for (const bool destroy : {true, false}) {
      std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
      ASSERT_NE(reusable, nullptr);
      replaceTexture(*reusable, *recorded, texture, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      const auto record = [&]() {
        {
          const std::unique_ptr<IRenderCommandEncoder> encoder =
              reusable->createRenderCommandEncoder(depthPass_, depthFb, &ret);
          ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
          encoder->endEncoding();
        }
        encodeAddPasses(*reusable, *colors, *recorded);
      };
      record();
      ASSERT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      if (executeInCommandBuffer) {
        const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
        ASSERT_NE(cmdBuf, nullptr);
        executeCommands(*cmdBuf, *reusable, &ret);
        EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();
        cmdQueue_->submit(*cmdBuf);
      } else {
        EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
      }
      EXPECT_FALSE(isRecorded(*reusable));
      EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
      EXPECT_TRUE(ctx.recordingJournals_.empty());
      if (destroy) {
        reusable.reset();
        EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
        continue;
      }

      const Pixels blue = fillPixels(kBlue);
      ASSERT_TRUE(texture->upload(texture->getFullRange(0), blue.data()).isOk());
      record();
      EXPECT_TRUE(run(*reusable, executeInCommandBuffer));
      reusable->waitUntilCompleted();
      // the red of texture A and the blue of the texture
      EXPECT_EQ(readTextureB(), fillPixels(kMagenta));
    }
  }

  // Records passes that render to texture A and sample it, and checks that texture A cannot be
  // replaced, either after the recording or before it, where the first run fails the recording,
  // and the slot records the passes again once the replacement is undone
  void replaceATextureThatTheRecordingsRenderTo(bool executeInCommandBuffer) const {
    const std::shared_ptr<ITexture> texture = createTexture(texDesc_);
    ASSERT_NE(texture, nullptr);
    std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    encodePasses(*reusable);
    ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    Result ret;
    replaceTexture(*reusable, *texA_, texture, &ret);
    EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();

    reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    replaceTexture(*reusable, *texA_, texture, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encodePasses(*reusable);
    EXPECT_FALSE(run(*reusable, executeInCommandBuffer));
    EXPECT_FALSE(isRecorded(*reusable));
    replaceTexture(*reusable, *texA_, texA_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    EXPECT_TRUE(renderFrame(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  // Records the passes, whose second one samples the recorded texture and renders to texture B,
  // and checks that texture B cannot replace the recorded texture, either after the recording or
  // before it, where the first run fails the recording, and the slot records the passes again once
  // the replacement is undone
  void replaceWithATextureThatTheRecordingsRenderTo(bool executeInCommandBuffer) const {
    const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
    ASSERT_NE(colors, nullptr);
    const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
    ASSERT_TRUE(colors->upload(red.data(), {sizeof(red)}).isOk());
    const std::shared_ptr<ITexture> recorded = createFilledTexture(kBlue);
    ASSERT_NE(recorded, nullptr);
    std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    encodeAddPasses(*reusable, *colors, *recorded);
    ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    Result ret;
    replaceTexture(*reusable, *recorded, texB_, &ret);
    EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();

    reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    replaceTexture(*reusable, *recorded, texB_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    encodeAddPasses(*reusable, *colors, *recorded);
    EXPECT_FALSE(run(*reusable, executeInCommandBuffer));
    EXPECT_FALSE(isRecorded(*reusable));
    replaceTexture(*reusable, *recorded, recorded, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
    ASSERT_NE(cmdBuf, nullptr);
    clear(*cmdBuf, fbB_, {0.0f, 0.0f, 0.0f, 1.0f});
    encodeAddPasses(*reusable, *colors, *recorded);
    EXPECT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    // the red of texture A and the blue of the recorded texture
    EXPECT_EQ(readTextureB(), fillPixels(kMagenta));
  }

  // Records a pass that samples a texture through a BindGroup, and checks that the texture cannot
  // be replaced
  void replaceATextureOfABindGroup(bool executeInCommandBuffer) const {
    const std::shared_ptr<ITexture> recorded = createFilledTexture(kBlue);
    ASSERT_NE(recorded, nullptr);
    const std::shared_ptr<ITexture> texture = createFilledTexture(kBlue);
    ASSERT_NE(texture, nullptr);
    Result ret;
    const Holder<BindGroupTextureHandle> bindGroup = iglDev_->createBindGroup(
        BindGroupTextureDesc{
            .textures = {recorded}, .samplers = {sampler_}, .debugName = "Recorded texture"},
        swizzlePipeline_.get(),
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    {
      const std::unique_ptr<IRenderCommandEncoder> encoder =
          reusable->createRenderCommandEncoder(renderPass_, fbB_, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      encoder->bindRenderPipelineState(swizzlePipeline_);
      encoder->bindBindGroup(bindGroup);
      encoder->draw(3);
      encoder->endEncoding();
    }
    ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    replaceTexture(*reusable, *recorded, texture, &ret);
    EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();
  }

  // How the recording of a slot fails before it runs
  enum class RecordingFailure {
    // generating the mipmaps of a texture whose UNORM image has sRGB views fails the recording at
    // once with Result::Code::Unsupported
    Abort,
    // the first run fails the recording with Result::Code::Unsupported, since texture B, which the
    // passes render to, replaces the texture that they sample
    RejectFirstRun,
  };

  // Records a render pass with a depth attachment, then the passes, whose second one samples a
  // texture that a frame texture replaces, and fails the recording in each way of `failures` in
  // turn. Checks that each failure closes the journal of the recording, puts back the layout of the
  // depth attachment and runs nothing, so that the slot holds no recording. With `replay`, the slot
  // then records the passes again, which replay over 3 frames per slot with another frame texture
  // in every frame.
  void failRecordings(bool executeInCommandBuffer,
                      const std::vector<RecordingFailure>& failures,
                      bool replay) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const bool aborts =
        std::find(failures.begin(), failures.end(), RecordingFailure::Abort) != failures.end();
    const std::shared_ptr<ITexture> srgbTex = aborts ? createSrgbViewedTexture() : nullptr;
    if (aborts && !srgbTex) {
      GTEST_SKIP() << "UNORM images with sRGB views are not supported";
    }
    if (srgbTex) {
      // ready to be sampled, as the scratch image path of its mipmaps expects
      const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      vulkan::transitionToShaderReadOnly(
          static_cast<vulkan::CommandBuffer&>(*cmdBuf).getVkCommandBuffer(), srgbTex.get());
      cmdQueue_->submit(*cmdBuf);
      cmdBuf->waitUntilCompleted();
    }
    const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
    ASSERT_NE(colors, nullptr);
    const std::array<float, 4> black = {0.0f, 0.0f, 0.0f, 1.0f};
    ASSERT_TRUE(colors->upload(black.data(), {sizeof(black)}).isOk());
    // green, which no frame texture has
    const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
    ASSERT_NE(recorded, nullptr);
    // shades of blue
    std::array<uint32_t, 4> framePixels = {};
    std::array<std::shared_ptr<ITexture>, 4> frameTextures;
    for (uint32_t i = 0; i != frameTextures.size(); ++i) {
      framePixels[i] = 0xFF000000 | ((51 * (i + 1)) << 16);
      frameTextures[i] = createFilledTexture(framePixels[i]);
      ASSERT_NE(frameTextures[i], nullptr);
    }
    const std::shared_ptr<ITexture> depthTex = createTexture(TextureDesc::new2D(
        TextureFormat::Z_UNorm16, kSize, kSize, TextureDesc::TextureUsageBits::Attachment));
    ASSERT_NE(depthTex, nullptr);
    Result ret;
    const std::shared_ptr<IFramebuffer> depthFb = iglDev_->createFramebuffer(
        {.colorAttachments = {{.texture = texA_}}, .depthAttachment = {.texture = depthTex}}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const auto record = [&]() {
      {
        const std::unique_ptr<IRenderCommandEncoder> encoder =
            reusable->createRenderCommandEncoder(depthPass_, depthFb, &ret);
        ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
        encoder->endEncoding();
      }
      encodeAddPasses(*reusable, *colors, *recorded);
    };

    for (const RecordingFailure failure : failures) {
      const bool isAbort = failure == RecordingFailure::Abort;
      replaceTexture(*reusable, *recorded, isAbort ? frameTextures[0] : texB_, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      record();
      ASSERT_EQ(ctx.recordingJournals_.size(), 1u);
      ASSERT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      if (isAbort) {
        // the scratch image that the mipmaps go through reaches no part of the journal
        const vulkan::RecordingJournal& journal = *ctx.recordingJournals_.front();
        const auto numJournaled = [&journal]() {
          return std::array<size_t, 3>{
              journal.layouts.size(), journal.imageDescriptors.size(), journal.imageUses.size()};
        };
        const std::array<size_t, 3> numJournaledBefore = numJournaled();
        const vulkan::VulkanImage& image =
            static_cast<vulkan::Texture&>(*srgbTex).getVulkanTexture().image;
        EXPECT_EQ(image.generateMipmap(journal.cmdBuf, srgbTex->getFullMipRange()).code,
                  Result::Code::Unsupported);
        EXPECT_EQ(numJournaled(), numJournaledBefore);
        srgbTex->generateMipmap(*reusable);
        EXPECT_TRUE(ctx.recordingJournals_.empty());
        EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
      }
      const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      clearTextures(*cmdBuf);
      if (executeInCommandBuffer) {
        executeCommands(*cmdBuf, *reusable, &ret);
        EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();
        cmdQueue_->submit(*cmdBuf);
      } else {
        cmdQueue_->submit(*cmdBuf);
        EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
      }
      cmdBuf->waitUntilCompleted();
      EXPECT_TRUE(ctx.recordingJournals_.empty());
      EXPECT_EQ(getImageLayout(*depthTex), VK_IMAGE_LAYOUT_UNDEFINED);
      EXPECT_FALSE(isRecorded(*reusable));
      EXPECT_EQ(readTextureB(), fillPixels(kMagenta));
    }
    if (!replay) {
      return;
    }

    uint32_t numRecordings = 0;
    for (uint32_t frame = 0; frame != 3 * getNumSlots(); ++frame) {
      const size_t index = frame % frameTextures.size();
      replaceTexture(*reusable, *recorded, frameTextures[index], &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      clearTextures(*cmdBuf);
      if (!isRecorded(*reusable)) {
        ++numRecordings;
        record();
      }
      EXPECT_TRUE(submitAndRun(*cmdBuf, *reusable, executeInCommandBuffer)) << "frame " << frame;
      reusable->waitUntilCompleted();
      EXPECT_TRUE(ctx.recordingJournals_.empty()) << "frame " << frame;
      EXPECT_EQ(readTextureB(), fillPixels(framePixels[index])) << "frame " << frame;
    }
    EXPECT_EQ(numRecordings, getNumSlots());
  }

  void restoreTextureMapping(bool executeInCommandBuffer, bool rejectIncompatible) const {
    const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
    ASSERT_NE(colors, nullptr);
    const std::array<float, 4> black = {0.0f, 0.0f, 0.0f, 1.0f};
    ASSERT_TRUE(colors->upload(black.data(), {sizeof(black)}).isOk());
    const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
    ASSERT_NE(recorded, nullptr);
    std::shared_ptr<ITexture> texture = createFilledTexture(kBlue);
    ASSERT_NE(texture, nullptr);
    const std::weak_ptr<ITexture> replacement = texture;
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    const uint32_t numSlots = getNumSlots();
    for (uint32_t frame = 0; frame != numSlots; ++frame) {
      ASSERT_FALSE(isRecorded(*reusable));
      encodeAddPasses(*reusable, *colors, *recorded);
      ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTextureB(), fillPixels(0xFF00FF00));
    }
    Result ret;
    replaceTexture(*reusable, *recorded, texture, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    for (uint32_t frame = 0; frame != numSlots; ++frame) {
      ASSERT_TRUE(isRecorded(*reusable));
      ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTextureB(), bluePixels());
    }
    if (rejectIncompatible) {
      const std::shared_ptr<ITexture> incompatible = createStorageTexture();
      ASSERT_NE(incompatible, nullptr);
      replaceTexture(*reusable, *recorded, incompatible, &ret);
      ASSERT_EQ(ret.code, Result::Code::ArgumentInvalid) << ret.message.c_str();
      for (uint32_t frame = 0; frame != numSlots; ++frame) {
        ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
        reusable->waitUntilCompleted();
        EXPECT_EQ(readTextureB(), bluePixels());
      }
      return;
    }
    replaceTexture(*reusable, *recorded, recorded, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    texture.reset();
    EXPECT_FALSE(replacement.expired());
    for (uint32_t frame = 0; frame != numSlots; ++frame) {
      ASSERT_TRUE(isRecorded(*reusable));
      ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTextureB(), fillPixels(0xFF00FF00));
      EXPECT_EQ(replacement.expired(), frame + 1 == numSlots);
    }
  }

  void replayWithAnUnreadyUnchangedReplacement(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
    ASSERT_NE(colors, nullptr);
    const std::array<float, 4> black = {0.0f, 0.0f, 0.0f, 1.0f};
    ASSERT_TRUE(colors->upload(black.data(), {sizeof(black)}).isOk());
    const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
    ASSERT_NE(recorded, nullptr);
    const std::shared_ptr<ITexture> texture = createFilledTexture(kBlue);
    ASSERT_NE(texture, nullptr);
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    Result ret;
    replaceTexture(*reusable, *recorded, texture, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    for (uint32_t frame = 0; frame != getNumSlots(); ++frame) {
      encodeAddPasses(*reusable, *colors, *recorded);
      ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
      reusable->waitUntilCompleted();
      EXPECT_EQ(readTextureB(), bluePixels());
    }
    const uint32_t slot = ctx.currentSyncIndex();
    const vulkan::VulkanImage& image =
        static_cast<vulkan::Texture&>(*texture).getVulkanTexture().image;
    const auto transition =
        [&](VkImageLayout layout, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
          const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
          ASSERT_NE(cmdBuf, nullptr);
          image.transitionLayout(
              static_cast<vulkan::CommandBuffer&>(*cmdBuf).getVkCommandBuffer(),
              layout,
              srcStage,
              dstStage,
              {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1});
          ASSERT_NE(cmdQueue_->submit(*cmdBuf), 0u);
          cmdBuf->waitUntilCompleted();
        };
    transition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
    ASSERT_EQ(ctx.currentSyncIndex(), slot);
    ASSERT_TRUE(isRecorded(*reusable));
    if (executeInCommandBuffer) {
      const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
      ASSERT_NE(cmdBuf, nullptr);
      executeCommands(*cmdBuf, *reusable, &ret);
      EXPECT_EQ(ret.code, Result::Code::InvalidOperation) << ret.message.c_str();
      cmdQueue_->submit(*cmdBuf);
    } else {
      EXPECT_EQ(cmdQueue_->submit(*reusable), 0u);
    }
    EXPECT_TRUE(isRecorded(*reusable));
    EXPECT_EQ(ctx.currentSyncIndex(), slot);
    transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    ASSERT_TRUE(run(*reusable, executeInCommandBuffer));
    reusable->waitUntilCompleted();
    EXPECT_EQ(readTextureB(), bluePixels());
  }

  std::shared_ptr<IRenderPipelineState> addPipeline_;
};

// Replays whose runs stay pending behind a gate while every frame replaces their textures (see
// ReusableCommandBufferPendingRunsVulkanTest)
class ReplaceTexturePendingRunsVulkanTest : public ReplaceTextureVulkanTest {
 protected:
  [[nodiscard]] std::shared_ptr<IDevice> createDevice() const override {
    vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig();
    config.enableUpdateAfterBindTextures = true;
    config.enableDescriptorIndexing = false;
    config.chainSubmitsWithSemaphores = false;
    config.maxResourceCount = 4;
    return util::device::vulkan::createTestDevice(config);
  }

  void destroyWithPendingReplacementTextures(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const std::shared_ptr<IComputePipelineState> pipeline = createComputePipeline(kAddColorShader);
    ASSERT_NE(pipeline, nullptr);
    const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
    ASSERT_NE(colors, nullptr);
    const std::array<float, 4> black = {0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(colors->upload(black.data(), {sizeof(black)}).isOk());
    const std::shared_ptr<ITexture> recordedInput = createFilledTexture(0xFF00FF00);
    ASSERT_NE(recordedInput, nullptr);
    const std::shared_ptr<ITexture> recordedOutput = createStorageTexture();
    ASSERT_NE(recordedOutput, nullptr);
    std::shared_ptr<ITexture> input = createFilledTexture(kBlue);
    ASSERT_NE(input, nullptr);
    std::shared_ptr<ITexture> output = createStorageTexture();
    ASSERT_NE(output, nullptr);
    makeStorageImages({recordedOutput, output});
    const std::weak_ptr<ITexture> weakInput = input;
    const std::weak_ptr<ITexture> weakOutput = output;
    std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);
    Result ret;
    replaceTexture(*reusable, *recordedInput, input, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    replaceTexture(*reusable, *recordedOutput, output, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const std::shared_ptr<IBuffer> pixels = iglDev_->createBuffer(
        {.type = BufferDesc::BufferTypeBits::Storage, .length = sizeof(Pixels)}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(pixels, nullptr);
    {
      HostGate gate(*iglDev_, ctx);
      if (!gate.isClosed()) {
        GTEST_SKIP() << "Timeline semaphores are not supported";
      }
      encodeAddColor(*reusable, pipeline, *recordedInput, *colors, *recordedOutput);
      SubmitHandle runHandle = 0;
      if (executeInCommandBuffer) {
        const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
        ASSERT_NE(cmdBuf, nullptr);
        executeCommands(*cmdBuf, *reusable, &ret);
        runHandle = cmdQueue_->submit(*cmdBuf);
        ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      } else {
        runHandle = cmdQueue_->submit(*reusable);
      }
      ASSERT_NE(runHandle, 0u);
      ASSERT_FALSE(
          ctx.immediate_->isReady(vulkan::VulkanImmediateCommands::SubmitHandle(runHandle)));
      reusable.reset();
      input.reset();
      output.reset();
      processDeferredTasks();
      EXPECT_FALSE(weakInput.expired());
      EXPECT_FALSE(weakOutput.expired());
      gate.open();
      ctx.waitIdle();
      {
        const std::shared_ptr<ITexture> finishedOutput = weakOutput.lock();
        ASSERT_NE(finishedOutput, nullptr);
        const std::shared_ptr<ICommandBuffer> copy = createCommandBuffer();
        ASSERT_NE(copy, nullptr);
        copy->copyTextureToBuffer(*finishedOutput, *pixels, 0);
        ASSERT_NE(cmdQueue_->submit(*copy), 0u);
        copy->waitUntilCompleted();
      }
      ctx.waitDeferredTasks();
      EXPECT_TRUE(weakInput.expired());
      EXPECT_TRUE(weakOutput.expired());
    }
    const auto* actual = static_cast<const Pixels*>(pixels->map({sizeof(Pixels)}, &ret));
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(*actual, bluePixels());
    pixels->unmap();
  }

  // Replays a compute pass that stores the sum of an input texture and the color of a Ring buffer
  // into an output texture, over 3 frames per slot without waiting between frames. Every frame has
  // an input texture, an output texture and a color of its own, which it sets before its run, and
  // then lets its input texture go, as a camera frame would. The runs of all the slots but one stay
  // pending behind a gate, while their slots keep their input textures alive. Checks the output of
  // every frame, and that the slots let those input textures go once they run again.
  void replayOutstandingFrames(bool executeInCommandBuffer) const {
    vulkan::VulkanContext& ctx = getVulkanContext();
    const uint32_t numSlots = getNumSlots();
    const uint32_t numFrames = 3 * numSlots;
    const uint32_t numGatedFrames = numSlots - 1;
    const auto inputPixelOf = [](uint32_t frame) {
      return 0xFF000000u | ((16u * (frame + 1)) << 8);
    };
    const auto redOf = [](uint32_t frame) { return 8u * (frame + 1); };
    const std::shared_ptr<IComputePipelineState> pipeline = createComputePipeline(kAddColorShader);
    ASSERT_NE(pipeline, nullptr);
    const std::shared_ptr<IBuffer> ringColors =
        createColorBuffer(BufferDesc::BufferAPIHintBits::Ring);
    ASSERT_NE(ringColors, nullptr);
    const std::shared_ptr<ITexture> recordedInput = createFilledTexture(0xFF000000);
    ASSERT_NE(recordedInput, nullptr);
    const std::shared_ptr<ITexture> recordedOutput = createStorageTexture();
    ASSERT_NE(recordedOutput, nullptr);
    std::vector<std::shared_ptr<ITexture>> inputs(numFrames);
    std::vector<std::shared_ptr<ITexture>> outputs(numFrames);
    for (uint32_t frame = 0; frame != numFrames; ++frame) {
      inputs[frame] = createFilledTexture(inputPixelOf(frame));
      ASSERT_NE(inputs[frame], nullptr);
      outputs[frame] = createStorageTexture();
      ASSERT_NE(outputs[frame], nullptr);
    }
    std::vector<std::shared_ptr<ITexture>> storageImages = outputs;
    storageImages.push_back(recordedOutput);
    makeStorageImages(storageImages);
    const std::shared_ptr<ICommandBuffer> reusable = createReusable(executeInCommandBuffer);
    ASSERT_NE(reusable, nullptr);

    HostGate gate(*iglDev_, ctx);
    if (!gate.isClosed()) {
      GTEST_SKIP() << "Timeline semaphores are not supported";
    }
    std::vector<SubmitHandle> runs;
    std::vector<std::weak_ptr<ITexture>> gatedInputs;
    for (uint32_t frame = 0; frame != numFrames; ++frame) {
      const std::array<float, 4> color = {
          static_cast<float>(redOf(frame)) / 255.0f, 0.0f, 0.0f, 0.0f};
      ASSERT_TRUE(ringColors->upload(color.data(), {sizeof(color)}).isOk()) << "frame " << frame;
      Result ret;
      replaceTexture(*reusable, *recordedInput, inputs[frame], &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      replaceTexture(*reusable, *recordedOutput, outputs[frame], &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      if (!isRecorded(*reusable)) {
        encodeAddColor(*reusable, pipeline, *recordedInput, *ringColors, *recordedOutput);
      }
      if (executeInCommandBuffer) {
        const std::shared_ptr<ICommandBuffer> cmdBuf = createCommandBuffer();
        ASSERT_NE(cmdBuf, nullptr);
        executeCommands(*cmdBuf, *reusable, &ret);
        ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
        runs.push_back(cmdQueue_->submit(*cmdBuf));
      } else {
        runs.push_back(cmdQueue_->submit(*reusable));
      }
      ASSERT_NE(runs.back(), 0u) << "frame " << frame;
      if (frame < numGatedFrames) {
        gatedInputs.push_back(inputs[frame]);
      }
      inputs[frame].reset();
      if (runs.size() == numGatedFrames) {
        for (const SubmitHandle run : runs) {
          EXPECT_FALSE(ctx.immediate_->isReady(vulkan::VulkanImmediateCommands::SubmitHandle(run)));
        }
        for (const std::weak_ptr<ITexture>& input : gatedInputs) {
          EXPECT_FALSE(input.expired());
        }
        gate.open();
      }
    }
    reusable->waitUntilCompleted();

    for (uint32_t frame = 0; frame != numFrames; ++frame) {
      EXPECT_EQ(readTexture(outputs[frame]), fillPixels(inputPixelOf(frame) | redOf(frame)))
          << "frame " << frame;
    }
    for (const std::weak_ptr<ITexture>& input : gatedInputs) {
      EXPECT_TRUE(input.expired());
    }
  }
};

TEST_F(ReplaceTextureVulkanTest, SubmitReadsTheParamsAndTexturesOfEachFrame) {
  renderFrames(false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsReadsTheParamsAndTexturesOfEachFrame) {
  renderFrames(true);
}

TEST_F(ReplaceTextureVulkanTest, SubmitWritesTheTexturesOfEachFrame) {
  storeFrames(false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsWritesTheTexturesOfEachFrame) {
  storeFrames(true);
}

TEST_F(ReplaceTextureVulkanTest, RejectsInvalidReplacements) {
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  Result ret;
  replaceTexture(*reusable, *texA_, nullptr, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentNull);
  // not Sampled, as texture A is
  const std::shared_ptr<ITexture> storageTexture = iglDev_->createTexture(
      TextureDesc::new2D(
          TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Storage),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  replaceTexture(*reusable, *texA_, storageTexture, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
  replaceTexture(*reusable, *texA_, texB_, &ret);
  EXPECT_TRUE(ret.isOk()) << ret.message.c_str();

  const std::shared_ptr<ICommandBuffer> cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  replaceTexture(*cmdBuf, *texA_, texB_, &ret);
  EXPECT_EQ(ret.code, Result::Code::InvalidOperation);
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(ReplaceTextureVulkanTest, SubmitKeepsTheDependenciesBetweenPassesOnReplacedTextures) {
  replayDependentPasses(false);
}

TEST_F(ReplaceTextureVulkanTest,
       ExecuteCommandsKeepsTheDependenciesBetweenPassesOnReplacedTextures) {
  replayDependentPasses(true);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureThatTheRecordingsWriteAndSample) {
  // a pass writes the texture as a storage image, and the next one samples it
  const std::shared_ptr<IComputePipelineState> store = createComputePipeline(kStoreShader);
  ASSERT_NE(store, nullptr);
  const std::shared_ptr<IComputePipelineState> add = createComputePipeline(kAddColorShader);
  ASSERT_NE(add, nullptr);
  const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
  ASSERT_NE(colors, nullptr);
  const TextureDesc desc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                              kSize,
                                              kSize,
                                              TextureDesc::TextureUsageBits::Sampled |
                                                  TextureDesc::TextureUsageBits::Storage);
  const std::shared_ptr<ITexture> recorded = createTexture(desc);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createTexture(desc);
  ASSERT_NE(texture, nullptr);
  const std::shared_ptr<ITexture> output = createStorageTexture();
  ASSERT_NE(output, nullptr);
  makeStorageImages({recorded, texture, output});
  const auto record = [&](ICommandBuffer& reusable) {
    encodeStore(reusable, store, *colors, *recorded);
    encodeAddColor(reusable, add, *recorded, *colors, *output);
  };

  std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  record(*reusable);
  ASSERT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  Result ret;
  replaceTexture(*reusable, *recorded, texture, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();

  // replaced before the slot records: the first run fails the recording, and the slot records
  // again once the replacement is undone
  reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  replaceTexture(*reusable, *recorded, texture, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  record(*reusable);
  EXPECT_FALSE(run(*reusable, false));
  EXPECT_FALSE(isRecorded(*reusable));
  replaceTexture(*reusable, *recorded, recorded, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  record(*reusable);
  EXPECT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureOfAnotherDevice) {
  const std::shared_ptr<IDevice> otherDevice = util::createTestDevice();
  ASSERT_NE(otherDevice, nullptr);
  Result ret;
  const std::shared_ptr<ITexture> otherTexture = otherDevice->createTexture(texDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  EXPECT_EQ(replace(*texA_, otherTexture).code, Result::Code::ArgumentInvalid);
  EXPECT_EQ(replace(*otherTexture, texB_).code, Result::Code::ArgumentInvalid);
  EXPECT_TRUE(replace(*texA_, texB_).isOk());
}

TEST_F(ReplaceTextureVulkanTest, RejectsAReusableCommandBufferOfAnotherDevice) {
  // a device that cannot replace textures
  const std::shared_ptr<IDevice> otherDevice = util::createTestDevice();
  ASSERT_NE(otherDevice, nullptr);
  const vulkan::PlatformDevice& otherPlatformDevice = getPlatformDevice(*otherDevice);
  ASSERT_FALSE(otherPlatformDevice.supportsTextureReplacement());
  const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
  ASSERT_NE(colors, nullptr);
  const std::array<float, 4> black = {0.0f, 0.0f, 0.0f, 1.0f};
  ASSERT_TRUE(colors->upload(black.data(), {sizeof(black)}).isOk());
  const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createFilledTexture(kBlue);
  ASSERT_NE(texture, nullptr);
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  encodeAddPasses(*reusable, *colors, *recorded);

  // the other device's helper leaves the reusable command buffer of this device as it is
  Result ret;
  otherPlatformDevice.replaceTexture(*reusable, *recorded, texture, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid) << ret.message.c_str();
  ASSERT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  EXPECT_EQ(readTextureB(), fillPixels(0xFF00FF00));

  // and the other way around
  const std::shared_ptr<ICommandBuffer> otherReusable =
      otherPlatformDevice.createReusableCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(otherReusable, nullptr);
  const std::shared_ptr<ITexture> otherRecorded = otherDevice->createTexture(texDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<ITexture> otherTexture = otherDevice->createTexture(texDesc_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  replaceTexture(*otherReusable, *otherRecorded, otherTexture, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid) << ret.message.c_str();
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureOfAnotherType) {
  const std::shared_ptr<ITexture> texture = createTexture(
      TextureDesc::new2DArray(TextureFormat::RGBA_UNorm8, kSize, kSize, 1, texDesc_.usage));
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(replace(*texA_, texture).code, Result::Code::ArgumentInvalid);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureWithAnotherAspect) {
  // the depth of a depth-stencil texture, and a view of the stencil of another one
  const TextureDesc desc = TextureDesc::new2D(TextureFormat::S8_UInt_Z24_UNorm,
                                              kSize,
                                              kSize,
                                              TextureDesc::TextureUsageBits::Sampled |
                                                  TextureDesc::TextureUsageBits::Attachment);
  const std::shared_ptr<ITexture> recorded = createTexture(desc);
  const std::shared_ptr<ITexture> stencilTexture = createTexture(desc);
  if (!recorded || !stencilTexture) {
    GTEST_SKIP() << "Sampled depth-stencil textures are not supported";
  }
  Result ret;
  const std::shared_ptr<ITexture> texture = iglDev_->createTextureView(
      stencilTexture,
      TextureViewDesc{.type = TextureType::TwoD, .aspect = ImageAspectBits_Stencil},
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(replace(*recorded, texture).code, Result::Code::ArgumentInvalid);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureWithAnotherNumberOfSamples) {
  TextureDesc desc = texDesc_;
  desc.numSamples = 4;
  const std::shared_ptr<ITexture> texture = createTexture(desc);
  if (!texture) {
    GTEST_SKIP() << "Sampled textures with 4 samples are not supported";
  }
  EXPECT_EQ(replace(*texA_, texture).code, Result::Code::ArgumentInvalid);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureWithAnotherNumberOfMipLevels) {
  TextureDesc desc = texDesc_;
  desc.numMipLevels = 2;
  const std::shared_ptr<ITexture> texture = createTexture(desc);
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(replace(*texA_, texture).code, Result::Code::ArgumentInvalid);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureWithAnotherNumberOfLayers) {
  const std::shared_ptr<ITexture> recorded = createTexture(
      TextureDesc::new2DArray(TextureFormat::RGBA_UNorm8, kSize, kSize, 2, texDesc_.usage));
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createTexture(
      TextureDesc::new2DArray(TextureFormat::RGBA_UNorm8, kSize, kSize, 3, texDesc_.usage));
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(replace(*recorded, texture).code, Result::Code::ArgumentInvalid);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureWithAnotherYcbcrConversion) {
  const std::shared_ptr<ITexture> recorded = createTexture(TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Sampled));
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createTexture(TextureDesc::new2D(
      TextureFormat::YUV_NV12, kSize, kSize, TextureDesc::TextureUsageBits::Sampled));
  if (!texture) {
    GTEST_SKIP() << "YUV textures are not supported";
  }
  const Result result = replace(*recorded, texture);
  EXPECT_EQ(result.code, Result::Code::ArgumentInvalid);
  // the format differs too
  EXPECT_NE(result.message.find("YCbCr"), std::string::npos) << result.message;
}

TEST_F(ReplaceTextureVulkanTest, RejectsAViewOfTheSameFormatWithAnotherYcbcrConversion) {
  vulkan::VulkanContext& ctx = getVulkanContext();
  if (!ctx.features().featuresSamplerYcbcrConversion.samplerYcbcrConversion) {
    GTEST_SKIP() << "YCbCr conversions are not supported";
  }
  const TextureDesc desc = TextureDesc::new2D(
      TextureFormat::YUV_NV12, kSize, kSize, TextureDesc::TextureUsageBits::Sampled);
  const VkFormat format = vulkan::textureFormatToVkFormat(desc.format);
  VkFormatProperties props = {};
  ctx.vf_.vkGetPhysicalDeviceFormatProperties(ctx.getVkPhysicalDevice(), format, &props);
  const bool midpoint =
      (props.optimalTilingFeatures & VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT) != 0;
  if (!midpoint &&
      (props.optimalTilingFeatures & VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT) == 0) {
    GTEST_SKIP() << "YUV textures are not supported";
  }
  // two conversions of the format that differ in their color model, which outlive the textures
  std::array<VkSamplerYcbcrConversion, 2> conversions = {};
  IGL_SCOPE_EXIT {
    for (const VkSamplerYcbcrConversion conversion : conversions) {
      ctx.vf_.vkDestroySamplerYcbcrConversion(ctx.getVkDevice(), conversion, nullptr);
    }
  };
  const VkChromaLocation chromaLocation = midpoint ? VK_CHROMA_LOCATION_MIDPOINT
                                                   : VK_CHROMA_LOCATION_COSITED_EVEN;
  for (size_t i = 0; i != conversions.size(); ++i) {
    const VkSamplerYcbcrConversionCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .format = format,
        .ycbcrModel = i == 0 ? VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709
                             : VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
        .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
        .components = {},
        .xChromaOffset = chromaLocation,
        .yChromaOffset = chromaLocation,
        .chromaFilter = VK_FILTER_NEAREST,
        .forceExplicitReconstruction = VK_FALSE,
    };
    ASSERT_EQ(
        ctx.vf_.vkCreateSamplerYcbcrConversion(ctx.getVkDevice(), &ci, nullptr, &conversions[i]),
        VK_SUCCESS);
  }
  // a texture whose image view the caller creates with `conversion`
  const auto createTexture = [&](VkSamplerYcbcrConversion conversion) -> std::shared_ptr<ITexture> {
    Result ret;
    vulkan::VulkanImage image =
        ctx.createImage(VK_IMAGE_TYPE_2D,
                        VkExtent3D{.width = kSize, .height = kSize, .depth = 1},
                        format,
                        1,
                        1,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        VK_IMAGE_CREATE_DISJOINT_BIT | VK_IMAGE_CREATE_ALIAS_BIT |
                            VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                        VK_SAMPLE_COUNT_1_BIT,
                        &ret,
                        "Image: YUV");
    if (!ret.isOk()) {
      return nullptr;
    }
    const VkSamplerYcbcrConversionInfo conversionInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
        .conversion = conversion,
    };
    vulkan::VulkanImageView imageView(
        ctx,
        VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &conversionInfo,
            .image = image.vkImage_,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {},
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
        },
        "Image View: YUV with a conversion of the caller");
    return std::make_shared<vulkan::Texture>(
        static_cast<vulkan::Device&>(*iglDev_),
        ctx.createTexture(std::move(image), std::move(imageView), "YUV"),
        desc);
  };
  const std::shared_ptr<ITexture> recorded = createTexture(conversions[0]);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> sameConversion = createTexture(conversions[0]);
  ASSERT_NE(sameConversion, nullptr);
  const std::shared_ptr<ITexture> otherConversion = createTexture(conversions[1]);
  ASSERT_NE(otherConversion, nullptr);
  EXPECT_TRUE(replace(*recorded, sameConversion).isOk());
  const Result result = replace(*recorded, otherConversion);
  EXPECT_EQ(result.code, Result::Code::ArgumentInvalid);
  EXPECT_NE(result.message.find("YCbCr"), std::string::npos) << result.message;
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureOfAnotherFormat) {
  const std::shared_ptr<ITexture> texture =
      createTexture(TextureDesc::new2D(TextureFormat::RGBA_SRGB, kSize, kSize, texDesc_.usage));
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(replace(*texA_, texture).code, Result::Code::ArgumentInvalid);
}

TEST_F(ReplaceTextureVulkanTest, SubmitRejectsATextureThatTheRecordingsRenderTo) {
  replaceATextureThatTheRecordingsRenderTo(false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsRejectsATextureThatTheRecordingsRenderTo) {
  replaceATextureThatTheRecordingsRenderTo(true);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureThatTheRecordingsCopy) {
  // the passes sample the texture, and the recording copies it into a buffer
  const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
  ASSERT_NE(colors, nullptr);
  const std::shared_ptr<ITexture> recorded = createFilledTexture(kBlue);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createFilledTexture(kBlue);
  ASSERT_NE(texture, nullptr);
  Result ret;
  const std::shared_ptr<IBuffer> buffer = iglDev_->createBuffer(
      BufferDesc{.type = BufferDesc::BufferTypeBits::Storage, .length = sizeof(Pixels)}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  encodeAddPasses(*reusable, *colors, *recorded);
  reusable->copyTextureToBuffer(*recorded, *buffer, 0);
  ASSERT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  replaceTexture(*reusable, *recorded, texture, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureThatTheRecordingsBlit) {
  // the recording generates the mip levels of the texture
  TextureDesc desc = texDesc_;
  desc.numMipLevels = 2;
  const std::shared_ptr<ITexture> recorded = createTexture(desc);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createTexture(desc);
  ASSERT_NE(texture, nullptr);
  const Pixels pixels = bluePixels();
  ASSERT_TRUE(recorded->upload(recorded->getFullRange(0), pixels.data()).isOk());
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  recorded->generateMipmap(*reusable);
  ASSERT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  Result ret;
  replaceTexture(*reusable, *recorded, texture, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();
}

TEST_F(ReplaceTextureVulkanTest, SubmitRejectsATextureOfABindGroup) {
  replaceATextureOfABindGroup(false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsRejectsATextureOfABindGroup) {
  replaceATextureOfABindGroup(true);
}

TEST_F(ReplaceTextureVulkanTest, SubmitRejectsATextureThatTheRecordingsRenderToInPlaceOfAnother) {
  replaceWithATextureThatTheRecordingsRenderTo(false);
}

TEST_F(ReplaceTextureVulkanTest,
       ExecuteCommandsRejectsATextureThatTheRecordingsRenderToInPlaceOfAnother) {
  replaceWithATextureThatTheRecordingsRenderTo(true);
}

TEST_F(ReplaceTextureVulkanTest, RejectsATextureThatTheRecordingsTransitionInPlaceOfAnother) {
  // the pass samples the recorded texture and writes the texture as a storage image, which takes
  // layout transitions of the texture, since it can be sampled
  const std::shared_ptr<IComputePipelineState> add = createComputePipeline(kAddColorShader);
  ASSERT_NE(add, nullptr);
  const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
  ASSERT_NE(colors, nullptr);
  const std::shared_ptr<ITexture> recorded = createFilledTexture(kBlue);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createStorageTexture(
      TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Storage |
      TextureDesc::TextureUsageBits::Attachment);
  ASSERT_NE(texture, nullptr);
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  encodeAddColor(*reusable, add, *recorded, *colors, *texture);
  ASSERT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  Result ret;
  replaceTexture(*reusable, *recorded, texture, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();
}

TEST_F(ReplaceTextureVulkanTest, ReplacesATextureWithATextureOfAnotherSize) {
  // the passes sample a frame texture of a single color, which looks the same at any size
  const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
  ASSERT_NE(colors, nullptr);
  const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
  ASSERT_TRUE(colors->upload(red.data(), {sizeof(red)}).isOk());
  const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createTexture(TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 2 * kSize, 2 * kSize, TextureDesc::TextureUsageBits::Sampled));
  ASSERT_NE(texture, nullptr);
  const std::vector<uint32_t> pixels(4 * kSize * kSize, kBlue);
  ASSERT_TRUE(texture->upload(texture->getFullRange(0), pixels.data()).isOk());
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  Result ret;
  replaceTexture(*reusable, *recorded, texture, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  encodeAddPasses(*reusable, *colors, *recorded);
  EXPECT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  // the red of texture A and the blue of the texture
  EXPECT_EQ(readTextureB(), fillPixels(kMagenta));
}

TEST_F(ReplaceTextureVulkanTest, RejectsASampledTextureInPlaceOfAStorageImage) {
  // passes leave a texture that can be sampled ready to be sampled, not written
  const std::shared_ptr<IComputePipelineState> store = createComputePipeline(kStoreShader);
  ASSERT_NE(store, nullptr);
  const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
  ASSERT_NE(colors, nullptr);
  const std::shared_ptr<ITexture> recorded = createStorageTexture();
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createStorageTexture(
      TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Storage |
      TextureDesc::TextureUsageBits::Attachment);
  ASSERT_NE(texture, nullptr);
  makeStorageImages({recorded, texture});
  std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  encodeStore(*reusable, store, *colors, *recorded);
  ASSERT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  Result ret;
  replaceTexture(*reusable, *recorded, texture, &ret);
  EXPECT_EQ(ret.code, Result::Code::Unsupported) << ret.message.c_str();

  // replaced before the slot records: the first run fails the recording, and the slot records
  // again once the replacement is undone
  reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  replaceTexture(*reusable, *recorded, texture, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  encodeStore(*reusable, store, *colors, *recorded);
  EXPECT_FALSE(run(*reusable, false));
  EXPECT_FALSE(isRecorded(*reusable));
  replaceTexture(*reusable, *recorded, recorded, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  encodeStore(*reusable, store, *colors, *recorded);
  EXPECT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
}

TEST_F(ReplaceTextureVulkanTest, RejectsAnArrayTextureInPlaceOfA2DViewOfALayer) {
  // a texture view has the type of its view, while getType() returns the type of its texture
  const std::shared_ptr<ITexture> array = createTexture(
      TextureDesc::new2DArray(TextureFormat::RGBA_UNorm8, kSize, kSize, 2, texDesc_.usage));
  ASSERT_NE(array, nullptr);
  Result ret;
  const std::shared_ptr<ITexture> recorded = iglDev_->createTextureView(
      array, TextureViewDesc{.type = TextureType::TwoD, .layer = 1, .numLayers = 1}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> texture = createTexture(
      TextureDesc::new2DArray(TextureFormat::RGBA_UNorm8, kSize, kSize, 1, texDesc_.usage));
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(replace(*recorded, texture).code, Result::Code::ArgumentInvalid);
}

TEST_F(ReplaceTextureVulkanTest, ReplacesATextureWithA2DViewOfALayer) {
  // the frame textures are the layers of an array texture
  const std::shared_ptr<IBuffer> colors = createColorBuffer(0);
  ASSERT_NE(colors, nullptr);
  const std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};
  ASSERT_TRUE(colors->upload(red.data(), {sizeof(red)}).isOk());
  const std::shared_ptr<ITexture> recorded = createFilledTexture(0xFF00FF00);
  ASSERT_NE(recorded, nullptr);
  const std::shared_ptr<ITexture> array = createTexture(TextureDesc::new2DArray(
      TextureFormat::RGBA_UNorm8, kSize, kSize, 2, TextureDesc::TextureUsageBits::Sampled));
  ASSERT_NE(array, nullptr);
  // green, then blue
  std::vector<uint32_t> pixels(2 * kSize * kSize, 0xFF00FF00);
  std::fill(pixels.begin() + kSize * kSize, pixels.end(), kBlue);
  ASSERT_TRUE(array->upload(array->getFullRange(0), pixels.data()).isOk());
  Result ret;
  const std::shared_ptr<ITexture> layer = iglDev_->createTextureView(
      array, TextureViewDesc{.type = TextureType::TwoD, .layer = 1, .numLayers = 1}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(layer, nullptr);
  const std::shared_ptr<ICommandBuffer> reusable = createReusable(false);
  ASSERT_NE(reusable, nullptr);
  replaceTexture(*reusable, *recorded, layer, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  encodeAddPasses(*reusable, *colors, *recorded);
  EXPECT_TRUE(run(*reusable, false));
  reusable->waitUntilCompleted();
  // the red of texture A and the blue of the layer
  EXPECT_EQ(readTextureB(), fillPixels(kMagenta));
}

TEST_F(ReplaceTextureVulkanTest, SubmitFailsUntilTheReplacementIsReady) {
  runWithAReplacementThatIsNotReady(false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsFailsUntilTheReplacementIsReady) {
  runWithAReplacementThatIsNotReady(true);
}

TEST_F(ReplaceTextureVulkanTest, SubmitAbortedRecordingClosesItsJournal) {
  failRecordings(false, {RecordingFailure::Abort}, false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsAbortedRecordingClosesItsJournal) {
  failRecordings(true, {RecordingFailure::Abort}, false);
}

TEST_F(ReplaceTextureVulkanTest, SubmitRejectedFirstRunClosesItsJournal) {
  failRecordings(false, {RecordingFailure::RejectFirstRun}, false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsRejectedFirstRunClosesItsJournal) {
  failRecordings(true, {RecordingFailure::RejectFirstRun}, false);
}

TEST_F(ReplaceTextureVulkanTest, SubmitReplaysRepeatedlyAfterFailedRecordings) {
  failRecordings(false, {RecordingFailure::Abort, RecordingFailure::RejectFirstRun}, true);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsReplaysRepeatedlyAfterFailedRecordings) {
  failRecordings(true, {RecordingFailure::Abort, RecordingFailure::RejectFirstRun}, true);
}

TEST_F(ReplaceTexturePendingRunsVulkanTest, SubmitReplaysOutstandingFramesWithTheirTextures) {
  replayOutstandingFrames(false);
}

TEST_F(ReplaceTexturePendingRunsVulkanTest,
       ExecuteCommandsReplaysOutstandingFramesWithTheirTextures) {
  replayOutstandingFrames(true);
}

TEST_F(ReusableCommandBufferVulkanTest, SubmitKeepsLayoutRollbackWithoutReplacementJournals) {
  discardRecordingWithoutReplacementJournals(false);
}

TEST_F(ReusableCommandBufferVulkanTest,
       ExecuteCommandsKeepsLayoutRollbackWithoutReplacementJournals) {
  discardRecordingWithoutReplacementJournals(true);
}

TEST_F(ReplaceTextureVulkanTest, SubmitRestoresRecordedTexturesInEverySlot) {
  restoreTextureMapping(false, false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsRestoresRecordedTexturesInEverySlot) {
  restoreTextureMapping(true, false);
}

TEST_F(ReplaceTextureVulkanTest, SubmitKeepsAMappingAfterRejectingAnIncompatibleReplacement) {
  restoreTextureMapping(false, true);
}

TEST_F(ReplaceTextureVulkanTest,
       ExecuteCommandsKeepsAMappingAfterRejectingAnIncompatibleReplacement) {
  restoreTextureMapping(true, true);
}

TEST_F(ReplaceTextureVulkanTest, SubmitChecksReadinessOfAnUnchangedReplacement) {
  replayWithAnUnreadyUnchangedReplacement(false);
}

TEST_F(ReplaceTextureVulkanTest, ExecuteCommandsChecksReadinessOfAnUnchangedReplacement) {
  replayWithAnUnreadyUnchangedReplacement(true);
}

TEST_F(ReplaceTexturePendingRunsVulkanTest, SubmitDestructionKeepsPendingReplacementTextures) {
  destroyWithPendingReplacementTextures(false);
}

TEST_F(ReplaceTexturePendingRunsVulkanTest,
       ExecuteCommandsDestructionKeepsPendingReplacementTextures) {
  destroyWithPendingReplacementTextures(true);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
