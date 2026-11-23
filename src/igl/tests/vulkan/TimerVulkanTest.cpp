/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <igl/CommandBuffer.h>
#include <igl/Device.h>
#include <igl/DeviceFeatures.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/Texture.h>
#include <igl/Timer.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class TimerVulkanTest : public ::testing::Test {
 public:
  TimerVulkanTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
#if IGL_PLATFORM_LINUX
    ASSERT_TRUE(iglDev_->hasFeature(DeviceFeatures::Timers));
#else
    if (!iglDev_->hasFeature(DeviceFeatures::Timers)) {
      GTEST_SKIP() << "Timers are not supported";
    }
#endif

    Result ret;
    cmdQueue_ = iglDev_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);

    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                   2048,
                                                   2048,
                                                   TextureDesc::TextureUsageBits::Attachment |
                                                       TextureDesc::TextureUsageBits::Sampled);
    colorTex_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(colorTex_, nullptr);

    const FramebufferDesc fbDesc{.colorAttachments = {{.texture = colorTex_}}};
    fb_ = iglDev_->createFramebuffer(fbDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(fb_, nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ITexture> colorTex_;
  std::shared_ptr<IFramebuffer> fb_;

  void clearPasses(ICommandBuffer& cmdBuf, uint32_t numPasses) {
    for (uint32_t i = 0; i < numPasses; ++i) {
      const float clearValue = static_cast<float>(i) / static_cast<float>(numPasses);
      const Color clearColor{clearValue, clearValue, clearValue, 1.0f};

      Result ret;
      const RenderPassDesc rpDesc{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                        .storeAction = StoreAction::Store,
                                                        .clearColor = clearColor}}};

      auto encoder = cmdBuf.createRenderCommandEncoder(rpDesc, fb_, {}, &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      ASSERT_NE(encoder, nullptr);

      encoder->endEncoding();
    }
  }
};

TEST_F(TimerVulkanTest, CreateTimer) {
  Result ret;
  const auto timer = iglDev_->createTimer(&ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(timer, nullptr);
  EXPECT_FALSE(timer->resultsAvailable());
  EXPECT_EQ(timer->getElapsedTimeNanos(), 0);
}

TEST_F(TimerVulkanTest, MeasuresACommandBuffer) {
  Result ret;
  const auto timer = iglDev_->createTimer(&ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(timer, nullptr);

  const CommandBufferDesc cmdDesc{.timer = timer};
  const auto cmdBuf = cmdQueue_->createCommandBuffer(cmdDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdBuf, nullptr);

  clearPasses(*cmdBuf, 1);
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  EXPECT_TRUE(timer->resultsAvailable());
  EXPECT_GT(timer->getElapsedTimeNanos(), 0);
}

TEST_F(TimerVulkanTest, ReusedAcrossCommandBuffersInFlight) {
  Result ret;
  const auto timer = iglDev_->createTimer(&ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(timer, nullptr);

  std::vector<std::shared_ptr<ICommandBuffer>> cmdBufs;
  constexpr uint32_t kNumBuffers = 6;

  for (uint32_t i = 0; i < kNumBuffers; ++i) {
    const CommandBufferDesc cmdDesc{.timer = timer};
    const auto cmdBuf = cmdQueue_->createCommandBuffer(cmdDesc, &ret);
    ASSERT_TRUE(ret.isOk());
    ASSERT_NE(cmdBuf, nullptr);

    clearPasses(*cmdBuf, 1);
    cmdQueue_->submit(*cmdBuf);
    cmdBufs.push_back(cmdBuf);
  }

  cmdBufs.back()->waitUntilCompleted();

  EXPECT_TRUE(timer->resultsAvailable());
  EXPECT_GT(timer->getElapsedTimeNanos(), 0);
}

TEST_F(TimerVulkanTest, LongerWorkTakesLonger) {
  constexpr uint32_t kNumRuns = 3;
  uint64_t minTime1Pass = UINT64_MAX;
  uint64_t minTime32Pass = UINT64_MAX;

  for (uint32_t run = 0; run < kNumRuns; ++run) {
    {
      Result ret;
      const auto timer = iglDev_->createTimer(&ret);
      ASSERT_TRUE(ret.isOk());

      const CommandBufferDesc cmdDesc{.timer = timer};
      const auto cmdBuf = cmdQueue_->createCommandBuffer(cmdDesc, &ret);
      ASSERT_TRUE(ret.isOk());

      clearPasses(*cmdBuf, 1);
      cmdQueue_->submit(*cmdBuf);
      cmdBuf->waitUntilCompleted();

      ASSERT_TRUE(timer->resultsAvailable());
      const uint64_t elapsed = timer->getElapsedTimeNanos();
      ASSERT_GT(elapsed, 0);
      minTime1Pass = std::min(minTime1Pass, elapsed);
    }

    {
      Result ret;
      const auto timer = iglDev_->createTimer(&ret);
      ASSERT_TRUE(ret.isOk());

      const CommandBufferDesc cmdDesc{.timer = timer};
      const auto cmdBuf = cmdQueue_->createCommandBuffer(cmdDesc, &ret);
      ASSERT_TRUE(ret.isOk());

      clearPasses(*cmdBuf, 32);
      cmdQueue_->submit(*cmdBuf);
      cmdBuf->waitUntilCompleted();

      ASSERT_TRUE(timer->resultsAvailable());
      const uint64_t elapsed = timer->getElapsedTimeNanos();
      ASSERT_GT(elapsed, 0);
      minTime32Pass = std::min(minTime32Pass, elapsed);
    }
  }

  EXPECT_GT(minTime32Pass, minTime1Pass);
}

TEST_F(TimerVulkanTest, TimerOutlivesItsCommandBuffer) {
  Result ret;
  const auto timer = iglDev_->createTimer(&ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(timer, nullptr);

  {
    const CommandBufferDesc cmdDesc{.timer = timer};
    const auto cmdBuf = cmdQueue_->createCommandBuffer(cmdDesc, &ret);
    ASSERT_TRUE(ret.isOk());
    ASSERT_NE(cmdBuf, nullptr);

    clearPasses(*cmdBuf, 1);
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
  }

  EXPECT_TRUE(timer->resultsAvailable());
  EXPECT_GT(timer->getElapsedTimeNanos(), 0);
}

} // namespace igl::tests

#endif
