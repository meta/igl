/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "util/Common.h"

#include <memory>
#include <string>
#include <vector>
#include <igl/CommandQueue.h>
#include <igl/Common.h>
#include <igl/Device.h>
#include <igl/FramebufferWrapper.h>
#include <igl/Texture.h>
#include <igl/base/IFramebufferInterop.h>

namespace igl::tests {

class FramebufferInteropTest : public ::testing::Test {
 public:
  FramebufferInteropTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(device_, cmdQueue_);
    ASSERT_TRUE(device_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> device_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  const std::string backend_ = IGL_BACKEND_TYPE;
};

TEST_F(FramebufferInteropTest, GetColorAttachment) {
  base::AttachmentInteropDesc colorDesc{
      .width = 512,
      .height = 512,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  auto* colorAttachment = fbInterop->getColorAttachment(0);
  ASSERT_NE(colorAttachment, nullptr);

  // Verify attachment properties via getDesc()
  const auto& desc = colorAttachment->getDesc();
  EXPECT_EQ(desc.width, 512u);
  EXPECT_EQ(desc.height, 512u);
  EXPECT_EQ(desc.format, base::TextureFormat::RGBA_SRGB);
}

TEST_F(FramebufferInteropTest, GetDepthAttachment) {
  base::AttachmentInteropDesc colorDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  // Use S8_UInt_Z32_UNorm for cross-backend compatibility (Metal remaps Z_UNorm24 to Z_UNorm32)
  auto depthFormat = base::TextureFormat::S8_UInt_Z32_UNorm;
  if (backend_ == util::kBackendVul) {
    depthFormat = base::TextureFormat::S8_UInt_Z24_UNorm;
  }

  base::AttachmentInteropDesc depthDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = depthFormat,
      .isSampled = false,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;
  fbDesc.depthAttachment = &depthDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  auto* depthAttachment = fbInterop->getDepthAttachment();
  ASSERT_NE(depthAttachment, nullptr);

  const auto& desc = depthAttachment->getDesc();
  EXPECT_EQ(desc.width, 256u);
  EXPECT_EQ(desc.height, 256u);
  EXPECT_EQ(desc.format, depthFormat);
}

TEST_F(FramebufferInteropTest, GetNativeFramebuffer) {
  base::AttachmentInteropDesc colorDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  void* nativeFb = fbInterop->getNativeFramebuffer();
  (void)nativeFb;
}

TEST_F(FramebufferInteropTest, AttachmentGetNativeImage) {
  base::AttachmentInteropDesc colorDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  auto* colorAttachment = fbInterop->getColorAttachment(0);
  ASSERT_NE(colorAttachment, nullptr);

  // For some backends it may be not possible to get native image
  if (device_->getBackendType() != BackendType::OpenGL) {
    void* nativeImage = colorAttachment->getNativeImage();
    EXPECT_NE(nativeImage, nullptr);
  }
}

TEST_F(FramebufferInteropTest, AttachmentDescProperties) {
  base::AttachmentInteropDesc colorDesc{
      .width = 320,
      .height = 240,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_UNorm8,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  auto* colorAttachment = fbInterop->getColorAttachment(0);
  ASSERT_NE(colorAttachment, nullptr);

  const auto& desc = colorAttachment->getDesc();
  EXPECT_EQ(desc.width, 320u);
  EXPECT_EQ(desc.height, 240u);
  EXPECT_EQ(desc.depth, 1u);
  EXPECT_EQ(desc.numLayers, 1u);
  EXPECT_EQ(desc.numSamples, 1u);
  EXPECT_EQ(desc.numMipLevels, 1u);
  EXPECT_EQ(desc.type, base::TextureType::TwoD);
  EXPECT_EQ(desc.format, base::TextureFormat::RGBA_UNorm8);
}

TEST_F(FramebufferInteropTest, MultipleColorAttachments) {
  base::AttachmentInteropDesc colorDesc0{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::AttachmentInteropDesc colorDesc1{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_UNorm8,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc0;
  fbDesc.colorAttachments[1] = &colorDesc1;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  auto* attachment0 = fbInterop->getColorAttachment(0);
  auto* attachment1 = fbInterop->getColorAttachment(1);

  ASSERT_NE(attachment0, nullptr);
  ASSERT_NE(attachment1, nullptr);

  // Verify different formats
  EXPECT_EQ(attachment0->getDesc().format, base::TextureFormat::RGBA_SRGB);
  EXPECT_EQ(attachment1->getDesc().format, base::TextureFormat::RGBA_UNorm8);
}

TEST_F(FramebufferInteropTest, StereoFramebuffer) {
  if (device_->getBackendType() == BackendType::OpenGL &&
      device_->getBackendVersion().majorVersion < 3) {
    GTEST_SKIP() << "Stereo rendering is not supported in OpenGL ES 2.0";
  }

  base::AttachmentInteropDesc colorDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 2,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoDArray,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  auto* colorAttachment = fbInterop->getColorAttachment(0);
  ASSERT_NE(colorAttachment, nullptr);

  const auto& desc = colorAttachment->getDesc();
  EXPECT_EQ(desc.numLayers, 2u);
  EXPECT_EQ(desc.type, base::TextureType::TwoDArray);
}

TEST_F(FramebufferInteropTest, MultipleFramebuffers) {
  constexpr size_t kNumFramebuffers = 5;
  std::vector<std::unique_ptr<base::IFramebufferInterop>> framebuffers;
  framebuffers.reserve(kNumFramebuffers);

  base::AttachmentInteropDesc colorDesc{
      .width = 128,
      .height = 128,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  for (size_t i = 0; i < kNumFramebuffers; ++i) {
    std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
    ASSERT_NE(fbInterop, nullptr) << "Failed to create framebuffer " << i;
    framebuffers.push_back(std::move(fbInterop));
  }

  // All framebuffers should have valid color attachments
  for (size_t i = 0; i < kNumFramebuffers; ++i) {
    EXPECT_NE(framebuffers[i]->getColorAttachment(0), nullptr)
        << "Framebuffer " << i << " has null color attachment";
  }
}

TEST_F(FramebufferInteropTest, GetColorAttachmentTexture) {
  base::AttachmentInteropDesc colorDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  std::shared_ptr<ITexture> colorTexture = fbInterop->getColorAttachmentTexture(0);
  ASSERT_NE(colorTexture, nullptr);
  EXPECT_EQ(colorTexture.get(), fbInterop->getColorAttachment(0));
}

TEST_F(FramebufferInteropTest, GetDepthAttachmentTexture) {
  base::AttachmentInteropDesc colorDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  auto depthFormat = base::TextureFormat::S8_UInt_Z32_UNorm;
  if (backend_ == util::kBackendVul) {
    depthFormat = base::TextureFormat::S8_UInt_Z24_UNorm;
  }

  base::AttachmentInteropDesc depthDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = depthFormat,
      .isSampled = false,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;
  fbDesc.depthAttachment = &depthDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  std::shared_ptr<ITexture> depthTexture = fbInterop->getDepthAttachmentTexture();
  ASSERT_NE(depthTexture, nullptr);
  EXPECT_EQ(depthTexture.get(), fbInterop->getDepthAttachment());
}

TEST_F(FramebufferInteropTest, GetDepthAttachmentTextureWithoutDepth) {
  base::AttachmentInteropDesc colorDesc{
      .width = 256,
      .height = 256,
      .depth = 1,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .type = base::TextureType::TwoD,
      .format = base::TextureFormat::RGBA_SRGB,
      .isSampled = true,
  };

  base::FramebufferInteropDesc fbDesc{};
  fbDesc.colorAttachments[0] = &colorDesc;

  std::unique_ptr<base::IFramebufferInterop> fbInterop(device_->createFramebufferInterop(fbDesc));
  ASSERT_NE(fbInterop, nullptr);

  EXPECT_EQ(fbInterop->getDepthAttachmentTexture(), nullptr);
}

TEST_F(FramebufferInteropTest, FramebufferWrapperNullFramebuffer) {
  FramebufferWrapper wrapper(nullptr);
  EXPECT_EQ(wrapper.getColorAttachment(0), nullptr);
  EXPECT_EQ(wrapper.getDepthAttachment(), nullptr);
  EXPECT_EQ(wrapper.getNativeFramebuffer(), nullptr);
  EXPECT_EQ(wrapper.getColorAttachmentTexture(0), nullptr);
  EXPECT_EQ(wrapper.getDepthAttachmentTexture(), nullptr);
}

} // namespace igl::tests
