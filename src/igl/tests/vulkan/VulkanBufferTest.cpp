/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanBuffer.h>

#include "../util/TestDevice.h"

#include <array>
#include <cstring>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/CommandQueue.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImmediateCommands.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

namespace {
thread_local uint32_t invalidBufferAborts = 0;

// Counts the debug aborts that name an invalid VulkanBuffer, for the lifetime of the observer.
class InvalidBufferAbortObserver {
 public:
  InvalidBufferAbortObserver() : previous_(iglGetDebugAbortListener()) {
    invalidBufferAborts = 0;
    iglSetDebugAbortListener([](const char* /*category*/,
                                const char* /*reason*/,
                                const char* /*file*/,
                                const char* /*func*/,
                                int /*line*/,
                                const char* format,
                                va_list /*ap*/) {
      if (format != nullptr && std::strstr(format, "invalid VulkanBuffer") != nullptr) {
        ++invalidBufferAborts;
      }
    });
  }
  ~InvalidBufferAbortObserver() {
    iglSetDebugAbortListener(previous_);
  }
  InvalidBufferAbortObserver(const InvalidBufferAbortObserver&) = delete;
  InvalidBufferAbortObserver& operator=(const InvalidBufferAbortObserver&) = delete;

 private:
  IGLErrorHandlerFunc previous_;
};
} // namespace

class VulkanBufferTest : public ::testing::Test {
 public:
  VulkanBufferTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;

  igl::vulkan::VulkanContext& getVulkanContext() {
    auto& device = static_cast<igl::vulkan::Device&>(*iglDev_);
    return device.getVulkanContext();
  }

  void expectSharedStorageBufferIsTransferSource(BufferDesc::BufferAPIHint hint) {
    Result ret;
    auto buffer = iglDev_->createBuffer(
        BufferDesc{
            .type = BufferDesc::BufferTypeBits::Storage,
            .length = 64,
            .storage = ResourceStorage::Shared,
            .hint = hint,
        },
        &ret);

    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(buffer, nullptr);

    const VkBufferUsageFlags usageFlags =
        static_cast<igl::vulkan::Buffer&>(*buffer).getBufferUsageFlags();
    EXPECT_NE(usageFlags & VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0u);
  }
};

TEST_F(VulkanBufferTest, CreateDeviceLocal) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer =
      ctx.createBuffer(256,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &ret,
                       "testDeviceLocalBuffer");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  EXPECT_NE(buffer->getVkBuffer(), VK_NULL_HANDLE);
  EXPECT_EQ(buffer->getSize(), 256u);
  EXPECT_TRUE(buffer->getMemoryPropertyFlags() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

TEST_F(VulkanBufferTest, CreateHostVisible) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer =
      ctx.createBuffer(512,
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                       &ret,
                       "testHostVisibleBuffer");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  EXPECT_NE(buffer->getVkBuffer(), VK_NULL_HANDLE);
  EXPECT_EQ(buffer->getSize(), 512u);
  EXPECT_TRUE(buffer->isMapped());
  EXPECT_NE(buffer->getMappedPtr(), nullptr);
}

TEST_F(VulkanBufferTest, BufferDeviceAddress) {
  auto& ctx = getVulkanContext();

  if (!ctx.features().has_VK_KHR_buffer_device_address) {
    GTEST_SKIP() << "VK_KHR_buffer_device_address not supported.";
  }

  Result ret;

  auto buffer = iglDev_->createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Uniform,
          .length = 256,
          .storage = ResourceStorage::Shared,
      },
      &ret);

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  EXPECT_NE(buffer->gpuAddress(), 0u);
}

TEST_F(VulkanBufferTest, GetVkBuffer) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer = ctx.createBuffer(128,
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                 &ret,
                                 "testGetVkBuffer");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  VkBuffer vkBuf = buffer->getVkBuffer();
  EXPECT_NE(vkBuf, VK_NULL_HANDLE);
}

TEST_F(VulkanBufferTest, HostVisibleBufferSubData) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer =
      ctx.createBuffer(256,
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &ret,
                       "testSubData");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->isMapped());

  std::vector<uint32_t> srcData(64, 0xDEADBEEF);
  buffer->bufferSubData(0, srcData.size() * sizeof(uint32_t), srcData.data());

  std::vector<uint32_t> dstData(64, 0);
  buffer->getBufferSubData(0, dstData.size() * sizeof(uint32_t), dstData.data());

  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(dstData[i], 0xDEADBEEF);
  }
}

TEST_F(VulkanBufferTest, GetBufferUsageFlags) {
  auto& ctx = getVulkanContext();

  Result ret;
  const VkBufferUsageFlags expectedFlags =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  auto buffer = ctx.createBuffer(
      64, expectedFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ret, "testUsageFlags");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  EXPECT_TRUE(buffer->getBufferUsageFlags() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  EXPECT_TRUE(buffer->getBufferUsageFlags() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
}

// A Shared storage buffer can be the source of ICommandBuffer::copyBuffer(), ring or not.
TEST_F(VulkanBufferTest, SharedStorageBufferIsTransferSource) {
  expectSharedStorageBufferIsTransferSource(0);
}

TEST_F(VulkanBufferTest, SharedStorageRingBufferIsTransferSource) {
  expectSharedStorageBufferIsTransferSource(BufferDesc::BufferAPIHintBits::Ring);
}

TEST_F(VulkanBufferTest, DeviceLocalBufferIsNotMapped) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer = ctx.createBuffer(128,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 &ret,
                                 "testDeviceLocalNotMapped");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  EXPECT_FALSE(buffer->isMapped());
  EXPECT_EQ(buffer->getMappedPtr(), nullptr);
}

TEST_F(VulkanBufferTest, GetSizeMatchesRequested) {
  auto& ctx = getVulkanContext();

  const VkDeviceSize sizes[] = {1, 64, 1024, 65536};

  for (const VkDeviceSize size : sizes) {
    Result ret;
    auto buffer = ctx.createBuffer(size,
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                   &ret,
                                   "testGetSize");

    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->getSize(), size);
  }
}

TEST_F(VulkanBufferTest, BufferSubDataWithOffset) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer =
      ctx.createBuffer(256,
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &ret,
                       "testSubDataOffset");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->isMapped());

  const std::array<uint32_t, 4> srcData = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD};
  const size_t offset = 64;
  buffer->bufferSubData(offset, srcData.size() * sizeof(uint32_t), srcData.data());

  std::array<uint32_t, 4> dstData = {};
  buffer->getBufferSubData(offset, dstData.size() * sizeof(uint32_t), dstData.data());

  EXPECT_EQ(dstData[0], 0xAAAAAAAA);
  EXPECT_EQ(dstData[1], 0xBBBBBBBB);
  EXPECT_EQ(dstData[2], 0xCCCCCCCC);
  EXPECT_EQ(dstData[3], 0xDDDDDDDD);
}

TEST_F(VulkanBufferTest, CoherentMemoryFlag) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto coherentBuffer =
      ctx.createBuffer(128,
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &ret,
                       "testCoherentBuffer");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(coherentBuffer, nullptr);

  if (!coherentBuffer->isCoherentMemory()) {
    GTEST_SKIP() << "Allocated memory type does not have VK_MEMORY_PROPERTY_HOST_COHERENT_BIT.";
  }

  auto deviceLocalBuffer = ctx.createBuffer(128,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                            &ret,
                                            "testDeviceLocalBuffer");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(deviceLocalBuffer, nullptr);
  EXPECT_FALSE(deviceLocalBuffer->isCoherentMemory());
}

TEST_F(VulkanBufferTest, BufferSubDataWithNullDataZerosMemory) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer =
      ctx.createBuffer(256,
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &ret,
                       "testNullDataZeros");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->isMapped());

  const std::array<uint32_t, 4> srcData = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD};
  buffer->bufferSubData(0, srcData.size() * sizeof(uint32_t), srcData.data());

  buffer->bufferSubData(0, srcData.size() * sizeof(uint32_t), nullptr);

  std::array<uint32_t, 4> dstData = {};
  dstData.fill(0xFF);
  buffer->getBufferSubData(0, dstData.size() * sizeof(uint32_t), dstData.data());

  for (const uint32_t val : dstData) {
    EXPECT_EQ(val, 0u);
  }
}

TEST_F(VulkanBufferTest, FlushAndInvalidateMappedMemory) {
  auto& ctx = getVulkanContext();

  Result ret;
  auto buffer =
      ctx.createBuffer(256,
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &ret,
                       "testFlushInvalidate");

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->isMapped());

  const std::array<uint32_t, 4> srcData = {1, 2, 3, 4};
  buffer->bufferSubData(0, srcData.size() * sizeof(uint32_t), srcData.data());

  buffer->flushMappedMemory(0, srcData.size() * sizeof(uint32_t));
  buffer->invalidateMappedMemory(0, srcData.size() * sizeof(uint32_t));

  std::array<uint32_t, 4> dstData = {};
  buffer->getBufferSubData(0, dstData.size() * sizeof(uint32_t), dstData.data());

  EXPECT_EQ(dstData[0], 1u);
  EXPECT_EQ(dstData[1], 2u);
  EXPECT_EQ(dstData[2], 3u);
  EXPECT_EQ(dstData[3], 4u);
}

TEST_F(VulkanBufferTest, CreateBufferWithInvalidStorageConvertsToPrivate) {
  Result ret;
  auto buffer = iglDev_->createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Uniform,
          .length = 256,
          .storage = ResourceStorage::Invalid,
      },
      &ret);

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  auto& ctx = getVulkanContext();
  if (ctx.useStagingForBuffers_) {
    EXPECT_EQ(buffer->storage(), ResourceStorage::Private);
  } else {
    EXPECT_EQ(buffer->storage(), ResourceStorage::Shared);
  }
}

// ---------------------------------------------------------------------------------------------
// Checked allocation. These run on a device whose function table lets the test fail chosen calls
// (util::device::vulkan::FaultingDevice); VulkanBuffer, VulkanContext and VMA are the production
// objects.
// ---------------------------------------------------------------------------------------------

class VulkanBufferFaultTest : public ::testing::Test {
 protected:
  using Call = util::device::vulkan::FaultingDevice::Call;
  using Counted = util::device::vulkan::FaultingDevice::Counted;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    if (!IGL_VULKAN_USE_VMA) {
      GTEST_SKIP() << "The allocation tests run on the VMA path";
    }
  }

  static std::unique_ptr<util::device::vulkan::FaultingDevice> makeDevice() {
    return std::make_unique<util::device::vulkan::FaultingDevice>(
        util::device::vulkan::getContextConfig());
  }

  static std::unique_ptr<igl::vulkan::VulkanBuffer> createHostBuffer(
      igl::vulkan::VulkanContext& ctx,
      Result* result) {
    return ctx.createBuffer(256,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                            result,
                            "Buffer: fault test");
  }

  static void expectNoDependentCall(const util::device::vulkan::FaultingDevice& fd,
                                    uint32_t namesBefore,
                                    uint32_t destroysBefore,
                                    uint32_t freesBefore,
                                    uint32_t mapsBefore) {
    if (fd.namingObservable()) {
      EXPECT_EQ(fd.counted(Counted::SetDebugUtilsObjectName), namesBefore);
    }
    EXPECT_EQ(fd.counted(Counted::DestroyBuffer), destroysBefore);
    EXPECT_EQ(fd.counted(Counted::FreeMemory), freesBefore);
    EXPECT_EQ(fd.reached(Call::MapMemory), mapsBefore);
  }
};

TEST_F(VulkanBufferFaultTest, FaultingDeviceStubsAreReachedThroughVma) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();
  // The context's own buffers (staging, dummy) went through VMA and the table's stubs.
  EXPECT_GT(fd->reached(Call::CreateBuffer), 0u);
  EXPECT_GT(fd->reached(Call::AllocateMemory), 0u);

  // One injected failure: the first attempt fails inside VMA, the drain succeeds, the retry works.
  fd->resetCounters();
  fd->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  Result result;
  const auto buffer = createHostBuffer(ctx, &result);
  EXPECT_EQ(fd->reached(Call::CreateBuffer), 2u);
  EXPECT_EQ(fd->failed(Call::CreateBuffer), 1u);
  ASSERT_NE(buffer, nullptr) << result.message.c_str();
  EXPECT_TRUE(result.isOk());
  EXPECT_TRUE(buffer->valid());
  EXPECT_EQ(buffer->initResult(), VK_SUCCESS);
  EXPECT_NE(buffer->getVkBuffer(), VK_NULL_HANDLE);
  EXPECT_TRUE(buffer->isMapped());
  if (fd->namingObservable()) {
    // Positive control for the "names nothing" assertions below: a valid buffer is named.
    EXPECT_GT(fd->counted(Counted::SetDebugUtilsObjectName), 0u);
  }
}

TEST_F(VulkanBufferFaultTest, AnOpenMainRecordingDoesNotStopTheRetry) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();

  // A recording of the context's own queue is open while the allocation fails (the normal
  // mid-frame state). The allocation is retried after the drain regardless; as on master, the
  // drain before the retry can run deferred tasks that are not certified retired (fixed by the
  // retirement redesign, not by this change).
  Result queueResult;
  auto cmdQueue = fd->device()->createCommandQueue({}, &queueResult);
  ASSERT_TRUE(queueResult.isOk()) << queueResult.message.c_str();
  auto openBuffer = cmdQueue->createCommandBuffer({}, &queueResult);
  ASSERT_TRUE(queueResult.isOk()) << queueResult.message.c_str();
  ctx.deferredTask(std::packaged_task<void()>([]() {}));

  fd->resetCounters();
  fd->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  Result result;
  const auto buffer = createHostBuffer(ctx, &result);

  ASSERT_NE(buffer, nullptr) << result.message.c_str();
  EXPECT_TRUE(result.isOk());
  EXPECT_TRUE(buffer->valid());
  EXPECT_EQ(fd->reached(Call::CreateBuffer), 2u);
  EXPECT_EQ(fd->failed(Call::CreateBuffer), 1u);

  cmdQueue->submit(*openBuffer);
  openBuffer->waitUntilCompleted();
}

TEST_F(VulkanBufferFaultTest, CreateBufferSecondAllocationFailure) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();

  // Both attempts fail, with different results: the caller sees the retry's.
  fd->resetCounters();
  fd->armSequence(Call::CreateBuffer, {VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_HOST_MEMORY});
  Result result;
  const auto buffer = createHostBuffer(ctx, &result);

  EXPECT_EQ(buffer, nullptr);
  EXPECT_FALSE(result.isOk());
  EXPECT_EQ(result.message,
            igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_HOST_MEMORY).message);
  EXPECT_NE(result.message,
            igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_DEVICE_MEMORY).message);
  EXPECT_EQ(fd->reached(Call::CreateBuffer), 2u);
  EXPECT_EQ(fd->failed(Call::CreateBuffer), 2u);
  expectNoDependentCall(*fd, 0u, 0u, 0u, 0u);
}

TEST_F(VulkanBufferFaultTest, BufferCreatePropagatesTheFactoryResult) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();
  const BufferDesc desc{
      .type = BufferDesc::BufferTypeBits::Uniform,
      .length = 256,
      .storage = ResourceStorage::Shared,
      .hint = BufferDesc::BufferAPIHintBits::Ring,
  };
  ASSERT_GT(ctx.config_.maxResourceCount, 1u) << "a ring buffer needs sub-buffers";

  // The second sub-buffer fails (and its retry): the creation fails as a whole, and the first
  // sub-buffer, which was created, is destroyed exactly once. (The failing `Buffer::create()` is
  // private to `Device`: the observable outcome of the public factory is the null buffer.)
  fd->resetCounters();
  fd->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/1, /*count=*/2);
  Result result;
  const auto buffer = fd->device()->createBuffer(desc, &result);

  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(fd->reached(Call::CreateBuffer), 3u); // sub-buffer 0, then sub-buffer 1 and its retry
  EXPECT_EQ(fd->failed(Call::CreateBuffer), 2u);
  // The sub-buffer was never used: it is destroyed by the failing creation itself, before any
  // drain, and exactly once.
  EXPECT_EQ(fd->counted(Counted::DestroyBuffer), 1u);
  ctx.waitDeferredTasks();
  EXPECT_EQ(fd->counted(Counted::DestroyBuffer), 1u);
}

TEST_F(VulkanBufferFaultTest, DeviceCreateBufferReportsTheFailureThroughOutResult) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  const BufferDesc desc{
      .type = BufferDesc::BufferTypeBits::Uniform,
      .length = 256,
      .storage = ResourceStorage::Shared,
      .hint = BufferDesc::BufferAPIHintBits::Ring,
  };
  ASSERT_GT(fd->context().config_.maxResourceCount, 1u) << "a ring buffer needs sub-buffers";

  // The second sub-buffer fails (and its retry). The caller's `Result` holds a stale non-OK value
  // before the call and the exact failure after it.
  fd->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/1, /*count=*/2);
  Result result(Result::Code::Unsupported, "stale");
  const auto buffer = fd->device()->createBuffer(desc, &result);

  const Result expected = igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_DEVICE_MEMORY);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_NE(result.code, Result::Code::Unsupported); // the stale value is gone
  EXPECT_EQ(result.code, expected.code);
  EXPECT_EQ(result.message, expected.message);
}

TEST_F(VulkanBufferFaultTest, DeviceCreateBufferWithoutDataClearsAStaleResult) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  const BufferDesc desc{
      .type = BufferDesc::BufferTypeBits::Uniform,
      .length = 256,
      .storage = ResourceStorage::Shared,
  };

  Result result(Result::Code::Unsupported, "stale");
  const auto buffer = fd->device()->createBuffer(desc, &result);

  ASSERT_NE(buffer, nullptr);
  EXPECT_TRUE(result.isOk()) << result.message.c_str();
}

TEST_F(VulkanBufferFaultTest, DummyBufferFailureFailsContextInit) {
  // The first buffer (the dummy uniform buffer) is allowed; every later one fails, so the dummy
  // storage buffer cannot be created. The fault is armed before create().
  const util::device::vulkan::FaultingDevice fd(
      util::device::vulkan::getContextConfig(), [](util::device::vulkan::FaultingDevice& self) {
        self.arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/1, UINT32_MAX);
      });

  EXPECT_EQ(fd.device(), nullptr);
  EXPECT_FALSE(fd.creationResult().isOk());
  EXPECT_NE(fd.creationResult().message.find("dummy uniform/storage buffers"), std::string::npos)
      << fd.creationResult().message.c_str();
  EXPECT_GE(fd.failed(Call::CreateBuffer), 2u);
}

TEST_F(VulkanBufferFaultTest, DummyUniformBufferFailureDestroysTheDummyStorageBufferAtOnce) {
  // The first dummy buffer (and its retry) fail; the second one is created and then discarded by
  // the failing context init, at once and not through a deferred task.
  const util::device::vulkan::FaultingDevice fd(
      util::device::vulkan::getContextConfig(), [](util::device::vulkan::FaultingDevice& self) {
        self.arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/0, /*count=*/2);
      });

  EXPECT_EQ(fd.device(), nullptr);
  EXPECT_FALSE(fd.creationResult().isOk());
  EXPECT_EQ(fd.failed(Call::CreateBuffer), 2u);
  // The storage buffer was destroyed by the failing init itself: it is counted before the context
  // is torn down (the teardown is what would run a deferred destruction).
  ASSERT_TRUE(fd.teardownObserved());
  EXPECT_EQ(fd.countedBeforeTeardown(Counted::DestroyBuffer), 1u);
}

TEST_F(VulkanBufferFaultTest, InvalidBufferOperationsAreGuarded) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();
  ctx.waitDeferredTasks();

  fd->resetCounters();
  fd->arm(Call::CreateBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, /*skip=*/0, /*count=*/2);
  const size_t deferredBefore = ctx.deferredTasks.size();
  const InvalidBufferAbortObserver aborts;
  {
    igl::vulkan::VulkanBuffer buffer(ctx,
                                     ctx.getVkDevice(),
                                     256,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     "Buffer: invalid");
    ASSERT_FALSE(buffer.valid());
    EXPECT_EQ(buffer.initResult(), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    EXPECT_EQ(buffer.getVkBuffer(), VK_NULL_HANDLE);
    EXPECT_FALSE(buffer.isMapped());
    EXPECT_EQ(buffer.getMappedPtr(), nullptr);

    uint32_t value = 0x12345678u;
    buffer.bufferSubData(0, sizeof(value), &value);
    uint32_t readBack = 0xAAAAAAAAu;
    buffer.getBufferSubData(0, sizeof(readBack), &readBack);
    EXPECT_EQ(readBack, 0xAAAAAAAAu); // untouched
    buffer.flushMappedMemory(0, sizeof(value));
    buffer.invalidateMappedMemory(0, sizeof(value));
    EXPECT_EQ(buffer.getVkDeviceAddress(), 0u);
    buffer.discardNeverSubmitted(); // a no-op on an invalid buffer
    EXPECT_FALSE(buffer.valid());
  }

  // Nothing was named, mapped or destroyed, and the destructor queued no deferred task.
  expectNoDependentCall(*fd, 0u, 0u, 0u, 0u);
  EXPECT_EQ(ctx.deferredTasks.size(), deferredBefore);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
