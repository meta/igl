/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/ResourcesBinder.h>

#include "../util/TestDevice.h"

#include <cstdarg>
#include <iterator>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/Device.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/SamplerState.h>
#include <igl/Texture.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class ResourcesBinderTest : public ::testing::Test {
 public:
  ResourcesBinderTest() = default;

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

  void waitForGpu() {
    Result ret;
    auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
    if (cmdBuf) {
      cmdQueue_->submit(*cmdBuf);
      cmdBuf->waitUntilCompleted();
    }
  }
};

TEST_F(ResourcesBinderTest, BindBufferAndDraw) {
  Result ret;

  const BufferDesc bufDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                           .length = 256,
                           .storage = ResourceStorage::Shared};
  auto buffer = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  std::vector<float> data(64, 1.0f);
  ret = buffer->upload(data.data(), BufferRange(256, 0));
  ASSERT_TRUE(ret.isOk());

  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           4,
                                           4,
                                           TextureDesc::TextureUsageBits::Attachment |
                                               TextureDesc::TextureUsageBits::Sampled);
  auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());
  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const RenderPassDesc rpDesc{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}}};

  auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, fb, {}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  encoder->bindBuffer(0, buffer.get(), 0, 256);

  encoder->endEncoding();
  cmdQueue->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

TEST_F(ResourcesBinderTest, BindTextureAndSamplerAndDraw) {
  Result ret;

  TextureDesc sampledTexDesc =
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto sampledTex = iglDev_->createTexture(sampledTexDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const std::vector<uint32_t> pixels(16, 0xFF0000FF);
  ret = sampledTex->upload(sampledTex->getFullRange(0), pixels.data());
  ASSERT_TRUE(ret.isOk());

  const SamplerStateDesc samplerDesc{.minFilter = SamplerMinMagFilter::Linear,
                                     .magFilter = SamplerMinMagFilter::Linear};
  auto sampler = iglDev_->createSamplerState(samplerDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(sampler, nullptr);

  TextureDesc colorTexDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                4,
                                                4,
                                                TextureDesc::TextureUsageBits::Attachment |
                                                    TextureDesc::TextureUsageBits::Sampled);
  auto colorTex = iglDev_->createTexture(colorTexDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());
  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const RenderPassDesc rpDesc{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}}};

  auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, fb, {}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  encoder->bindTexture(0, sampledTex.get());
  encoder->bindSamplerState(0, 0, sampler.get());

  encoder->endEncoding();
  cmdQueue->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

TEST_F(ResourcesBinderTest, BindMultipleBuffers) {
  Result ret;

  const BufferDesc bufDesc{
      .type = BufferDesc::BufferTypeBits::Uniform,
      .length = 128,
      .storage = ResourceStorage::Shared,
  };

  auto buffer0 = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer0, nullptr);

  auto buffer1 = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer1, nullptr);

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Attachment |
                                                     TextureDesc::TextureUsageBits::Sampled);
  auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const RenderPassDesc rpDesc{
      .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}}};

  auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, fb, {}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  encoder->bindBuffer(0, buffer0.get(), 0, 128);
  encoder->bindBuffer(1, buffer1.get(), 0, 128);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

TEST_F(ResourcesBinderTest, CreateEncoderWithClearColor) {
  Result ret;

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 8,
                                                 8,
                                                 TextureDesc::TextureUsageBits::Attachment |
                                                     TextureDesc::TextureUsageBits::Sampled);
  auto colorTex = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk());

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const RenderPassDesc rpDesc{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                    .storeAction = StoreAction::Store,
                                                    .clearColor = {1.0f, 0.0f, 0.0f, 1.0f}}}};

  auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, fb, {}, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(encoder, nullptr);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

namespace {
bool sDebugAbortFired = false;
} // namespace

class ResourcesBinderStorageImageTest : public ResourcesBinderTest {
 public:
  void SetUp() override {
    ResourcesBinderTest::SetUp();
    sDebugAbortFired = false;
    iglSetDebugAbortListener([](const char* /*category*/,
                                const char* /*reason*/,
                                const char* /*file*/,
                                const char* /*func*/,
                                int /*line*/,
                                const char* /*format*/,
                                va_list /*ap*/) { sDebugAbortFired = true; });
  }

  void TearDown() override {
    iglSetDebugAbortListener(nullptr);
    ResourcesBinderTest::TearDown();
  }

 protected:
  vulkan::VulkanContext& getVulkanContext() {
    return static_cast<vulkan::Device&>(*iglDev_).getVulkanContext();
  }
};

TEST_F(ResourcesBinderStorageImageTest, BindingsAreSizedByStorageImagesMax) {
  EXPECT_EQ(std::size(vulkan::BindingsStorageImages{}.images), IGL_STORAGE_IMAGES_MAX);
}

TEST_F(ResourcesBinderStorageImageTest, StorageImageIndexIsBoundedByStorageImagesMax) {
#if !IGL_DEBUG_ABORT_ENABLED
  GTEST_SKIP() << "An out-of-range index is observable only through the debug abort listener";
#else
  vulkan::ResourcesBinder binder(nullptr, getVulkanContext(), VK_PIPELINE_BIND_POINT_COMPUTE);
  ASSERT_FALSE(sDebugAbortFired) << "Constructing the binder must not abort";

  // A null texture keeps both cases on the index bound alone: the usage and layout checks in
  // bindStorageImage() run only for a non-null texture.
  binder.bindStorageImage(IGL_STORAGE_IMAGES_MAX - 1, nullptr);
  EXPECT_FALSE(sDebugAbortFired) << "The highest storage image index must be accepted";

  binder.bindStorageImage(IGL_STORAGE_IMAGES_MAX, nullptr);
  EXPECT_TRUE(sDebugAbortFired) << "One past the highest storage image index must be rejected";
#endif // !IGL_DEBUG_ABORT_ENABLED
}

TEST_F(ResourcesBinderStorageImageTest, InputAttachmentIndexIsBoundedByColorAttachmentsMax) {
#if !IGL_DEBUG_ABORT_ENABLED
  GTEST_SKIP() << "An out-of-range index is observable only through the debug abort listener";
#else
  vulkan::ResourcesBinder binder(nullptr, getVulkanContext(), VK_PIPELINE_BIND_POINT_GRAPHICS);
  ASSERT_FALSE(sDebugAbortFired) << "Constructing the binder must not abort";

  // A null texture clears an input attachment slot
  binder.bindInputAttachment(0, nullptr, 0, 0);
  binder.bindInputAttachment(IGL_COLOR_ATTACHMENTS_MAX - 1, nullptr, 0, 0);
  EXPECT_FALSE(sDebugAbortFired)
      << "A null texture must be accepted at every input attachment index";

  binder.bindInputAttachment(IGL_COLOR_ATTACHMENTS_MAX, nullptr, 0, 0);
  EXPECT_TRUE(sDebugAbortFired) << "One past the highest input attachment index must be rejected";
#endif // !IGL_DEBUG_ABORT_ENABLED
}

TEST_F(ResourcesBinderStorageImageTest, InputAttachmentsNeedAGraphicsBinder) {
#if !IGL_DEBUG_ABORT_ENABLED
  GTEST_SKIP() << "The rejection is observable only through the debug abort listener";
#else
  vulkan::ResourcesBinder binder(nullptr, getVulkanContext(), VK_PIPELINE_BIND_POINT_COMPUTE);
  ASSERT_FALSE(sDebugAbortFired) << "Constructing the binder must not abort";

  binder.bindInputAttachment(0, nullptr, 0, 0);
  EXPECT_TRUE(sDebugAbortFired) << "A compute binder must reject input attachments";
#endif // !IGL_DEBUG_ABORT_ENABLED
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
