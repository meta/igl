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
#include <vector>
#include <igl/CommandBuffer.h>
#include <igl/Device.h>
#include <igl/Framebuffer.h>
#include <igl/Texture.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class FramebufferVulkanTest : public ::testing::Test {
 public:
  FramebufferVulkanTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
};

TEST_F(FramebufferVulkanTest, CreateWithColorAttachment) {
  Result ret;

  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           8,
                                           8,
                                           TextureDesc::TextureUsageBits::Attachment |
                                               TextureDesc::TextureUsageBits::Sampled);
  auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};

  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(fb, nullptr);

  auto indices = fb->getColorAttachmentIndices();
  EXPECT_FALSE(indices.empty());
  EXPECT_NE(fb->getColorAttachment(0), nullptr);
}

TEST_F(FramebufferVulkanTest, CreateWithColorAndDepth) {
  Result ret;

  TextureDesc colorDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                             8,
                                             8,
                                             TextureDesc::TextureUsageBits::Attachment |
                                                 TextureDesc::TextureUsageBits::Sampled);
  auto colorTex = iglDev_->createTexture(colorDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  TextureDesc depthDesc = TextureDesc::new2D(TextureFormat::Z_UNorm24,
                                             8,
                                             8,
                                             TextureDesc::TextureUsageBits::Attachment |
                                                 TextureDesc::TextureUsageBits::Sampled);
  auto depthTex = iglDev_->createTexture(depthDesc, &ret);
  if (!ret.isOk()) {
    depthDesc.format = TextureFormat::Z_UNorm16;
    depthTex = iglDev_->createTexture(depthDesc, &ret);
  }
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}},
                               .depthAttachment = {.texture = depthTex}};

  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(fb, nullptr);
  EXPECT_NE(fb->getDepthAttachment(), nullptr);
}

TEST_F(FramebufferVulkanTest, CopyBytesColorAttachment) {
  Result ret;

  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           2,
                                           2,
                                           TextureDesc::TextureUsageBits::Attachment |
                                               TextureDesc::TextureUsageBits::Sampled);
  auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  constexpr uint32_t kColor = 0xFF00FF00;
  const std::array<uint32_t, 4> pixels = {kColor, kColor, kColor, kColor};
  ret = colorTex->upload(colorTex->getFullRange(0), pixels.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  cmdQueue->submit(*cmdBuf);

  std::array<uint32_t, 4> result = {};
  fb->copyBytesColorAttachment(*cmdQueue, 0, result.data(), colorTex->getFullRange(0));

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(result[i], kColor);
  }
}

TEST_F(FramebufferVulkanTest, UpdateDrawable) {
  Result ret;

  TextureDesc texDesc1 = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                            4,
                                            4,
                                            TextureDesc::TextureUsageBits::Attachment |
                                                TextureDesc::TextureUsageBits::Sampled);
  auto colorTex1 = iglDev_->createTexture(texDesc1, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex1}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(fb, nullptr);

  TextureDesc texDesc2 = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                            8,
                                            8,
                                            TextureDesc::TextureUsageBits::Attachment |
                                                TextureDesc::TextureUsageBits::Sampled);
  auto colorTex2 = iglDev_->createTexture(texDesc2, &ret);
  ASSERT_TRUE(ret.isOk());

  fb->updateDrawable(colorTex2);
  EXPECT_EQ(fb->getColorAttachment(0), colorTex2);
}

TEST_F(FramebufferVulkanTest, MultipleColorAttachments) {
  Result ret;

  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           4,
                                           4,
                                           TextureDesc::TextureUsageBits::Attachment |
                                               TextureDesc::TextureUsageBits::Sampled);

  auto colorTex0 = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  auto colorTex1 = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{
      .colorAttachments = {{.texture = colorTex0}, {.texture = colorTex1}}};

  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(fb, nullptr);

  auto indices = fb->getColorAttachmentIndices();
  EXPECT_EQ(indices.size(), 2u);
  EXPECT_NE(fb->getColorAttachment(0), nullptr);
  EXPECT_NE(fb->getColorAttachment(1), nullptr);
}

TEST_F(FramebufferVulkanTest, DefaultModeIsMono) {
  Result ret;

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Attachment |
                                                     TextureDesc::TextureUsageBits::Sampled);
  const auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  const auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  EXPECT_EQ(fb->getMode(), FramebufferMode::Mono);
  EXPECT_FALSE(fb->isSwapchainBound());
}

TEST_F(FramebufferVulkanTest, UpdateDrawableWithNull) {
  Result ret;

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Attachment |
                                                     TextureDesc::TextureUsageBits::Sampled);
  const auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  const auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(fb->getColorAttachment(0), nullptr);

  fb->updateDrawable(nullptr);
  EXPECT_EQ(fb->getColorAttachment(0), nullptr);
  EXPECT_TRUE(fb->getColorAttachmentIndices().empty());
}

TEST_F(FramebufferVulkanTest, UpdateDrawableWithSurfaceTextures) {
  Result ret;

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Attachment |
                                                     TextureDesc::TextureUsageBits::Sampled);
  const auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  TextureDesc depthDesc =
      TextureDesc::new2D(TextureFormat::Z_UNorm24, 4, 4, TextureDesc::TextureUsageBits::Attachment);
  depthDesc.storage = ResourceStorage::Private;
  auto depthTex = iglDev_->createTexture(depthDesc, &ret);
  if (!ret.isOk()) {
    depthDesc.format = TextureFormat::Z_UNorm16;
    depthTex = iglDev_->createTexture(depthDesc, &ret);
  }
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  const auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  fb->updateDrawable(SurfaceTextures{.color = colorTex, .depth = depthTex});
  EXPECT_NE(fb->getColorAttachment(0), nullptr);
  EXPECT_NE(fb->getDepthAttachment(), nullptr);

  fb->updateDrawable(SurfaceTextures{.color = colorTex, .depth = nullptr});
  EXPECT_NE(fb->getColorAttachment(0), nullptr);
  EXPECT_EQ(fb->getDepthAttachment(), nullptr);
}

TEST_F(FramebufferVulkanTest, UpdateDrawableClearsStaleResolve) {
  Result ret;

  TextureDesc msaaDesc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Attachment);
  msaaDesc.numSamples = 4;
  const auto msaaColor = iglDev_->createTexture(msaaDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const TextureDesc resolveDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                     4,
                                                     4,
                                                     TextureDesc::TextureUsageBits::Attachment |
                                                         TextureDesc::TextureUsageBits::Sampled);
  const auto resolveTex = iglDev_->createTexture(resolveDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const FramebufferDesc fbDesc{
      .colorAttachments = {{.texture = msaaColor, .resolveTexture = resolveTex}}};
  const auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(fb, nullptr);
  ASSERT_EQ(fb->getResolveColorAttachment(0), resolveTex);

  // The single-texture overload swaps only the color attachment and carries no
  // resolve target. The stale resolve must be cleared so it does not linger
  // against the new color attachment.
  const auto newColor = iglDev_->createTexture(resolveDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  fb->updateDrawable(newColor);
  EXPECT_EQ(fb->getColorAttachment(0), newColor);
  EXPECT_EQ(fb->getResolveColorAttachment(0), nullptr);
}

TEST_F(FramebufferVulkanTest, StorageTextureAttachment) {
  Result ret;

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Attachment |
                                                     TextureDesc::TextureUsageBits::Storage);
  const auto storageTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = storageTex}}};
  const auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(fb, nullptr);
  EXPECT_NE(fb->getColorAttachment(0), nullptr);
}

TEST_F(FramebufferVulkanTest, FramebufferReadbackPropagatesStagingFailure) {
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "The allocation tests run on the VMA path";
  }
  util::device::vulkan::FaultingDevice fd(util::device::vulkan::getContextConfig());
  ASSERT_NE(fd.device(), nullptr) << fd.creationResult().message.c_str();
  Result ret;

  // 4 MiB of pixels: more than the 1 MiB minimum staging buffer, so the readback needs a new
  // staging buffer however much free staging space is left over.
  constexpr uint32_t kSize = 1024;
  const auto colorTex =
      fd.device()->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                    kSize,
                                                    kSize,
                                                    TextureDesc::TextureUsageBits::Attachment |
                                                        TextureDesc::TextureUsageBits::Sampled),
                                 &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::vector<uint32_t> pixels(static_cast<size_t>(kSize) * kSize, 0xFF00FF00u);
  ret = colorTex->upload(colorTex->getFullRange(0), pixels.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const auto fb = fd.device()->createFramebuffer(
      FramebufferDesc{.colorAttachments = {{.texture = colorTex}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const auto cmdQueue = fd.device()->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  // The readback cannot stage: the caller's memory is left as it was.
  fd.drainStaging();
  fd.resetCounters();
  fd.arm(util::device::vulkan::FaultingDevice::Call::CreateBuffer,
         VK_ERROR_OUT_OF_DEVICE_MEMORY,
         /*skip=*/0,
         /*count=*/UINT32_MAX);
  constexpr uint32_t kSentinel = 0xDEADBEEFu;
  std::vector<uint32_t> result(pixels.size(), kSentinel);
  fb->copyBytesColorAttachment(*cmdQueue, 0, result.data(), colorTex->getFullRange(0));
  EXPECT_GE(fd.failed(util::device::vulkan::FaultingDevice::Call::CreateBuffer), 1u);
  EXPECT_EQ(std::count(result.begin(), result.end(), kSentinel),
            static_cast<std::ptrdiff_t>(result.size()));
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
