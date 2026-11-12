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

  // Renders red into texture A, or only loads it if `loadTextureA` is set, then samples it into
  // texture B with red and blue swapped
  void encodePasses(ICommandBuffer& cmdBuf, bool loadTextureA = false) const {
    Result ret;
    auto encoder =
        cmdBuf.createRenderCommandEncoder(loadTextureA ? loadPass_ : renderPass_, fbA_, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    if (!loadTextureA) {
      encoder->bindRenderPipelineState(colorPipeline_);
      encoder->bindBuffer(0, colorBuffer_.get());
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
    EXPECT_TRUE(getVulkanContext().imageLayoutJournals_.empty());
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
    EXPECT_TRUE(ctx.imageLayoutJournals_.empty());
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
    EXPECT_TRUE(ctx.imageLayoutJournals_.empty());
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
    EXPECT_TRUE(ctx.imageLayoutJournals_.empty());

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
    EXPECT_TRUE(ctx.imageLayoutJournals_.empty());
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
      EXPECT_EQ(ctx.immediate_->isReady(handle), executeInCommandBuffer) << "frame " << frame;
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
    EXPECT_TRUE(ctx.imageLayoutJournals_.empty());
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
    ASSERT_EQ(ctx.imageLayoutJournals_.size(), 1u);
    const size_t numJournaled = ctx.imageLayoutJournals_.front()->layouts.size();
    const Result result = image.generateMipmap(recording, srgbTex->getFullMipRange());
    EXPECT_EQ(result.code, Result::Code::Unsupported) << result.message.c_str();
    EXPECT_EQ(image.imageLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ASSERT_EQ(ctx.imageLayoutJournals_.front()->layouts.size(), numJournaled);
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
  EXPECT_TRUE(getVulkanContext().imageLayoutJournals_.empty());
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
  EXPECT_TRUE(ctx.imageLayoutJournals_.empty());
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
  EXPECT_TRUE(ctx.imageLayoutJournals_.empty());
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

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
