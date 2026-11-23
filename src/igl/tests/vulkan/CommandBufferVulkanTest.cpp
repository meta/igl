/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <vector>
#include <igl/ComputeCommandEncoder.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/Texture.h>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/CommandBuffer.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class CommandBufferVulkanTest : public ::testing::Test {
 public:
  CommandBufferVulkanTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";

    Result ret;
    cmdQueue_ = iglDev_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

TEST_F(CommandBufferVulkanTest, GetVkCommandBuffer) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdBuf, nullptr);

  auto* vulkanCmdBuf = static_cast<vulkan::CommandBuffer*>(cmdBuf.get());
  EXPECT_TRUE(vulkanCmdBuf->getVkCommandBuffer() != nullptr);

  cmdQueue_->submit(*cmdBuf);
}

TEST_F(CommandBufferVulkanTest, DebugGroupLabels) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(cmdBuf, nullptr);

  cmdBuf->pushDebugGroupLabel("TestGroup", Color(1.0f, 0.0f, 0.0f, 1.0f));
  cmdBuf->popDebugGroupLabel();

  cmdQueue_->submit(*cmdBuf);
}

TEST_F(CommandBufferVulkanTest, CopyBuffer) {
  Result ret;

  const BufferDesc srcDesc{.type = BufferDesc::BufferTypeBits::Storage,
                           .length = 128,
                           .storage = ResourceStorage::Shared};
  auto srcBuffer = iglDev_->createBuffer(srcDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const BufferDesc dstDesc{.type = BufferDesc::BufferTypeBits::Storage,
                           .length = 128,
                           .storage = ResourceStorage::Shared};
  auto dstBuffer = iglDev_->createBuffer(dstDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  std::vector<uint32_t> srcData(32, 0xCAFEBABE);
  ret = srcBuffer->upload(srcData.data(), BufferRange(128, 0));
  ASSERT_TRUE(ret.isOk());

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  cmdBuf->copyBuffer(*srcBuffer, *dstBuffer, 0, 0, 128);
  cmdQueue_->submit(*cmdBuf);

  auto& device = static_cast<vulkan::Device&>(*iglDev_);
  device.getVulkanContext().waitIdle();

  const auto* downloadedData = static_cast<uint32_t*>(dstBuffer->map(BufferRange(128, 0), &ret));
  ASSERT_TRUE(ret.isOk());
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(downloadedData[i], 0xCAFEBABE);
  }
  dstBuffer->unmap();
}

TEST_F(CommandBufferVulkanTest, FillBuffer) {
  Result ret;
  const uint32_t initialData[] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
  const BufferDesc desc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .data = initialData,
      .length = sizeof(initialData),
      .storage = ResourceStorage::Shared,
  };
  auto buffer = iglDev_->createBuffer(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  cmdBuf->fillBuffer(*buffer, BufferRange(2 * sizeof(uint32_t), sizeof(uint32_t)), 0x7Fu);
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  auto* mapped = static_cast<uint32_t*>(buffer->map(BufferRange(sizeof(initialData), 0), &ret));
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(mapped, nullptr);
  EXPECT_EQ(mapped[0], 0xFFFFFFFFu);
  EXPECT_EQ(mapped[1], 0x7F7F7F7Fu);
  EXPECT_EQ(mapped[2], 0x7F7F7F7Fu);
  EXPECT_EQ(mapped[3], 0xFFFFFFFFu);
  buffer->unmap();
}

namespace {
std::shared_ptr<IBuffer> createStorageBuffer(IDevice& device, size_t length, uint32_t fill) {
  const std::vector<uint32_t> initialData(length / sizeof(uint32_t), fill);
  const BufferDesc desc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .data = initialData.data(),
      .length = length,
      .storage = ResourceStorage::Shared,
  };
  Result ret;
  auto buffer = device.createBuffer(desc, &ret);
  EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
  return buffer;
}

std::vector<uint32_t> readBuffer(IBuffer& buffer) {
  Result ret;
  const size_t length = buffer.getSizeInBytes();
  const auto* mapped = static_cast<const uint32_t*>(buffer.map(BufferRange(length, 0), &ret));
  EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
  if (mapped == nullptr) {
    return {};
  }
  std::vector<uint32_t> contents(mapped, mapped + length / sizeof(uint32_t));
  buffer.unmap();
  return contents;
}
} // namespace

TEST_F(CommandBufferVulkanTest, UpdateBuffer) {
  ASSERT_TRUE(iglDev_->hasFeature(DeviceFeatures::UpdateBuffer));
  auto buffer = createStorageBuffer(*iglDev_, 4 * sizeof(uint32_t), 0xFFFFFFFFu);
  ASSERT_NE(buffer, nullptr);

  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const uint32_t data[] = {0x11111111u, 0x22222222u};
  cmdBuf->updateBuffer(*buffer, BufferRange(sizeof(data), sizeof(uint32_t)), data);
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  EXPECT_EQ(readBuffer(*buffer),
            (std::vector<uint32_t>{0xFFFFFFFFu, 0x11111111u, 0x22222222u, 0xFFFFFFFFu}));
}

// vkCmdUpdateBuffer() takes at most 64 KB per call, so larger ranges are split.
TEST_F(CommandBufferVulkanTest, UpdateBufferLargerThan64KB) {
  constexpr size_t kLength = 2 * 65536 + 16;
  auto buffer = createStorageBuffer(*iglDev_, kLength, 0u);
  ASSERT_NE(buffer, nullptr);

  std::vector<uint32_t> data(kLength / sizeof(uint32_t));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint32_t>(i) * 0x01000193u;
  }

  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  cmdBuf->updateBuffer(*buffer, BufferRange(kLength, 0), data.data());
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  EXPECT_EQ(readBuffer(*buffer), data);
}

// Commands recorded before an update read the old contents; commands recorded after read the new.
TEST_F(CommandBufferVulkanTest, UpdateBufferIsOrderedWithOtherCommands) {
  constexpr size_t kLength = 4 * sizeof(uint32_t);
  auto buffer = createStorageBuffer(*iglDev_, kLength, 0u);
  auto copyBefore = createStorageBuffer(*iglDev_, kLength, 0u);
  auto copyAfter = createStorageBuffer(*iglDev_, kLength, 0u);
  ASSERT_NE(buffer, nullptr);
  ASSERT_NE(copyBefore, nullptr);
  ASSERT_NE(copyAfter, nullptr);

  const std::vector<uint32_t> first(4, 0xAAAAAAAAu);
  const std::vector<uint32_t> second(4, 0x55555555u);

  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  cmdBuf->updateBuffer(*buffer, BufferRange(kLength, 0), first.data());
  cmdBuf->copyBuffer(*buffer, *copyBefore, 0, 0, kLength);
  cmdBuf->updateBuffer(*buffer, BufferRange(kLength, 0), second.data());
  cmdBuf->copyBuffer(*buffer, *copyAfter, 0, 0, kLength);
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  EXPECT_EQ(readBuffer(*copyBefore), first);
  EXPECT_EQ(readBuffer(*copyAfter), second);
}

// A command buffer submitted while an earlier one is still pending writes the same buffer without
// changing what the earlier one reads.
TEST_F(CommandBufferVulkanTest, UpdateBufferAcrossCommandBuffersInFlight) {
  constexpr size_t kLength = 4 * sizeof(uint32_t);
  auto buffer = createStorageBuffer(*iglDev_, kLength, 0u);
  auto copy1 = createStorageBuffer(*iglDev_, kLength, 0u);
  auto copy2 = createStorageBuffer(*iglDev_, kLength, 0u);
  ASSERT_NE(buffer, nullptr);
  ASSERT_NE(copy1, nullptr);
  ASSERT_NE(copy2, nullptr);

  const std::vector<uint32_t> first(4, 0x01020304u);
  const std::vector<uint32_t> second(4, 0x0A0B0C0Du);

  Result ret;
  auto cmdBuf1 = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  cmdBuf1->updateBuffer(*buffer, BufferRange(kLength, 0), first.data());
  cmdBuf1->copyBuffer(*buffer, *copy1, 0, 0, kLength);
  cmdQueue_->submit(*cmdBuf1);

  auto cmdBuf2 = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  cmdBuf2->updateBuffer(*buffer, BufferRange(kLength, 0), second.data());
  cmdBuf2->copyBuffer(*buffer, *copy2, 0, 0, kLength);
  cmdQueue_->submit(*cmdBuf2);

  cmdBuf1->waitUntilCompleted();
  cmdBuf2->waitUntilCompleted();

  EXPECT_EQ(readBuffer(*copy1), first);
  EXPECT_EQ(readBuffer(*copy2), second);
}

// Uniform buffers get TRANSFER_DST so updateBuffer() can write them whatever their storage.
TEST_F(CommandBufferVulkanTest, UniformBuffersAcceptUpdateBuffer) {
  for (const ResourceStorage storage : {ResourceStorage::Shared, ResourceStorage::Private}) {
    const BufferDesc desc{
        .type = BufferDesc::BufferTypeBits::Uniform,
        .length = 64,
        .storage = storage,
    };
    Result ret;
    auto buffer = iglDev_->createBuffer(desc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    const auto& vulkanBuffer = static_cast<vulkan::Buffer&>(*buffer);
    EXPECT_NE(vulkanBuffer.getBufferUsageFlags() & VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0u);
  }
}

TEST_F(CommandBufferVulkanTest, WaitUntilCompleted) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(cmdBuf, nullptr);

  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

TEST_F(CommandBufferVulkanTest, CreateComputeCommandEncoder) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(cmdBuf, nullptr);

  auto encoder = cmdBuf->createComputeCommandEncoder();
  EXPECT_NE(encoder, nullptr);
  encoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);
}

TEST_F(CommandBufferVulkanTest, GetNextSubmitHandle) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto* vulkanCmdBuf = static_cast<vulkan::CommandBuffer*>(cmdBuf.get());
  auto handle = vulkanCmdBuf->getNextSubmitHandle();
  EXPECT_FALSE(handle.empty());

  cmdQueue_->submit(*cmdBuf);
}

TEST_F(CommandBufferVulkanTest, CreateRenderCommandEncoder) {
  Result ret;

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Attachment |
                                                     TextureDesc::TextureUsageBits::Sampled);
  const auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(colorTex, nullptr);

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  const auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(fb, nullptr);

  const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(cmdBuf, nullptr);

  const RenderPassDesc rpDesc{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}}};

  const auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, fb, {}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);
  encoder->endEncoding();

  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

TEST_F(CommandBufferVulkanTest, MultipleSequentialSubmits) {
  for (int i = 0; i < 3; ++i) {
    Result ret;
    const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk()) << "Failed on iteration " << i;
    ASSERT_NE(cmdBuf, nullptr);

    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
  }
}

TEST_F(CommandBufferVulkanTest, NestedDebugGroupLabels) {
  Result ret;
  const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(cmdBuf, nullptr);

  cmdBuf->pushDebugGroupLabel("Outer", Color(1.0f, 0.0f, 0.0f, 1.0f));
  cmdBuf->pushDebugGroupLabel("Inner", Color(0.0f, 1.0f, 0.0f, 1.0f));
  cmdBuf->popDebugGroupLabel();
  cmdBuf->popDebugGroupLabel();

  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
