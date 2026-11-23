/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/PlatformDevice.h>

#include "../util/Common.h"
#if IGL_PLATFORM_IOS
#include <igl/opengl/ios/PlatformDevice.h>
#elif IGL_PLATFORM_MACOSX
#include <igl/opengl/macos/PlatformDevice.h>
#elif IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX_USE_EGL
#include <igl/opengl/egl/PlatformDevice.h>
#elif IGL_PLATFORM_LINUX
#include <igl/opengl/glx/PlatformDevice.h>
#elif IGL_PLATFORM_WINDOWS
#if defined(FORCE_USE_ANGLE)
#include <igl/opengl/egl/PlatformDevice.h>
#else
#include <igl/opengl/wgl/PlatformDevice.h>
#endif // FORCE_USE_ANGLE
#else
#error "Unsupported testing platform"
#endif

#include <igl/opengl/TextureBufferExternal.h>

#if IGL_PLATFORM_IOS
#define PLATFORM_DEVICE opengl::ios::PlatformDevice
#elif IGL_PLATFORM_MACOSX
#define PLATFORM_DEVICE opengl::macos::PlatformDevice
#elif IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX_USE_EGL
#define PLATFORM_DEVICE opengl::egl::PlatformDevice
#elif IGL_PLATFORM_LINUX
#define PLATFORM_DEVICE opengl::glx::PlatformDevice
#elif IGL_PLATFORM_WINDOWS
#if defined(FORCE_USE_ANGLE)
#define PLATFORM_DEVICE opengl::egl::PlatformDevice
#else
#define PLATFORM_DEVICE opengl::wgl::PlatformDevice
#endif // FORCE_USE_ANGLE
#endif

// Use a 1x1 Framebuffer for this test
#define OFFSCREEN_RT_WIDTH 1
#define OFFSCREEN_RT_HEIGHT 1

namespace igl::tests {

class PlatformDeviceTest : public ::testing::Test {
 public:
  PlatformDeviceTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);

    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

// Test Cases
TEST_F(PlatformDeviceTest, GetPlatformDeviceParentCls) {
  auto* pd = iglDev_->getPlatformDevice<opengl::PlatformDevice>();
  ASSERT_NE(pd, nullptr);
}

TEST_F(PlatformDeviceTest, GetPlatformDeviceChildCls) {
  auto* pd = iglDev_->getPlatformDevice<PLATFORM_DEVICE>();
  ASSERT_NE(pd, nullptr);
}

// This test will exercise createTextureBufferExternal() API.
// Since this API creates an empty container with the settings provided, we can
// simply check that the settings provided are what are actually set.
TEST_F(PlatformDeviceTest, CreateTextureBufferExternal) {
  auto* pd = iglDev_->getPlatformDevice<opengl::PlatformDevice>();
  ASSERT_NE(pd, nullptr);

  std::unique_ptr<opengl::TextureBufferExternal> textureBuffer =
      pd->createTextureBufferExternal(1, // Randomly pick 1 as the Texture ID
                                      GL_TEXTURE_2D, // Randomly picking GL_TEXTURE_2D
                                      TextureDesc::TextureUsageBits::Sampled,
                                      OFFSCREEN_RT_WIDTH,
                                      OFFSCREEN_RT_HEIGHT,
                                      TextureFormat::RGBA_UNorm8);

  ASSERT_NE(textureBuffer, nullptr);

  ASSERT_EQ(textureBuffer->getTarget(), GL_TEXTURE_2D);
  ASSERT_EQ(textureBuffer->getId(), 1);
  ASSERT_EQ(textureBuffer->getUsage(), TextureDesc::TextureUsageBits::Sampled);
}

// This test will exercise CreateCurrentFrameBuffer() API.
// It simply checks whether CreateCurrentFrameBuffer() does not return null pointer.
TEST_F(PlatformDeviceTest, CreateCurrentFrameBuffer) {
  auto* pd = iglDev_->getPlatformDevice<opengl::PlatformDevice>();
  ASSERT_NE(pd, nullptr);

  auto frameBuffer = pd->createCurrentFramebuffer();
  ASSERT_NE(frameBuffer, nullptr);
}

//
// Test ITexture::getEstimatedSizeInBytes with external textures
//
TEST_F(PlatformDeviceTest, GetEstimatedSizeInBytesExternal) {
  auto* pd = iglDev_->getPlatformDevice<opengl::PlatformDevice>();

  auto calcSize = [&](size_t width, size_t height, TextureFormat format) -> size_t {
    auto texture = pd->createTextureBufferExternal(1, // Not actually using it
                                                   GL_TEXTURE_2D,
                                                   TextureDesc::TextureUsageBits::Sampled,
                                                   width,
                                                   height,
                                                   format);
    if (texture == nullptr) {
      return 0;
    }
    return texture->getEstimatedSizeInBytes();
  };

  ASSERT_EQ(calcSize(64, 32, TextureFormat::Invalid), 2048); // 64 * 32 -- no bpp information
  ASSERT_EQ(calcSize(12, 34, TextureFormat::RGBA_UNorm8), 1632); // 12 * 34 * 4
  ASSERT_EQ(calcSize(16, 1, TextureFormat::R_UNorm8), 16);
  ASSERT_EQ(calcSize(128, 333, TextureFormat::RG_UNorm8), 85248); // 128 * 333 * 2
}

#if IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX_USE_EGL
//
// Test for createTextureFromNativeDrawable() available on some platforms
//
TEST_F(PlatformDeviceTest, CreateTextureFromNativeDrawable) {
  auto* pd = iglDev_->getPlatformDevice<PLATFORM_DEVICE>();
  ASSERT_NE(pd, nullptr);

  Result result;
  auto tex = pd->createTextureFromNativeDrawable(TextureFormat::RGBA_UNorm8, &result);
  EXPECT_TRUE(result.isOk());
  EXPECT_NE(tex, nullptr);

  auto tex2 = pd->createTextureFromNativeDrawable(1, 1, TextureFormat::RGBA_UNorm8, &result);
  EXPECT_TRUE(result.isOk());
  EXPECT_NE(tex2, nullptr);
}

//
// Test for createTextureFromNativeDepth() available on some platforms
//
TEST_F(PlatformDeviceTest, CreateTextureFromNativeDepth) {
  auto* pd = iglDev_->getPlatformDevice<PLATFORM_DEVICE>();
  ASSERT_NE(pd, nullptr);

  Result result;
  auto tex = pd->createTextureFromNativeDepth(TextureFormat::Z_UNorm24, &result);
  EXPECT_TRUE(result.isOk());
  EXPECT_NE(tex, nullptr);
}
#endif

} // namespace igl::tests
