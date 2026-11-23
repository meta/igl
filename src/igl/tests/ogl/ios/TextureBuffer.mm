/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../../util/TextureFormatTestBase.h"

#include <CoreVideo/CVPixelBuffer.h>
#include <CoreVideo/CVPixelFormatDescription.h>
#include <igl/Device.h>
#include <igl/opengl/ios/Context.h>
#include <igl/opengl/ios/PlatformDevice.h>

namespace igl::tests {

class TextureBufferIosTest : public util::TextureFormatTestBase {
 public:
  TextureBufferIosTest() = default;

  std::shared_ptr<ITexture> createCVPixelBufferTextureWithSize(OSType pixelFormat,
                                                               size_t width,
                                                               size_t height,
                                                               TextureDesc::TextureUsage usage,
                                                               Result& outResult);
};

std::shared_ptr<ITexture> TextureBufferIosTest::createCVPixelBufferTextureWithSize(
    OSType pixelFormat,
    const size_t width,
    const size_t height,
    TextureDesc::TextureUsage usage,
    Result& outResult) {
  const BackendType backend = iglDev_->getBackendType();
  CVPixelBufferRef pixelBuffer = nullptr;
  NSDictionary* bufferAttributes = @{
    (NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (NSString*)kCVPixelFormatOpenGLESCompatibility : @(backend == BackendType::OpenGL),
  };

  const CVReturn result = CVPixelBufferCreate(kCFAllocatorDefault,
                                              width,
                                              height,
                                              pixelFormat,
                                              (__bridge CFDictionaryRef)(bufferAttributes),
                                              &pixelBuffer);
  if (result != kCVReturnSuccess) {
    Result::setResult(&outResult,
                      Result::Code::RuntimeError,
                      "CVPixelBufferCreate failed to create pixel buffer");
    return nullptr;
  }

  auto* platformDevice = iglDev_->getPlatformDevice<opengl::ios::PlatformDevice>();
  auto& context = static_cast<opengl::ios::Context&>(platformDevice->getContext());
  auto* textureCache = context.getTextureCache();
  std::shared_ptr<ITexture> texture = platformDevice->createTextureFromNativePixelBuffer(
      pixelBuffer, textureCache, 0, usage, &outResult);
  if (!outResult.isOk()) {
    return nullptr;
  }
  if (texture == nullptr) {
    Result::setResult(
        &outResult, Result::Code::RuntimeError, "failed to create igl texture from CVPixelBuffer");
    return nullptr;
  }
  CVPixelBufferRelease(pixelBuffer);
  Result::setOk(&outResult);
  return texture;
}

#define PIXEL_FORMAT(pf) {pf, #pf}

TEST_F(TextureBufferIosTest, createTextureFromNativePixelBuffer) {
  const std::vector<std::pair<OSType, const char*>> pixelFormats = {
      PIXEL_FORMAT(kCVPixelFormatType_32BGRA),
      // TODO: These currently returns kCVReturnPixelBufferNotOpenGLCompatible
      // PIXEL_FORMAT(kCVPixelFormatType_64RGBAHalf),
      // PIXEL_FORMAT(kCVPixelFormatType_OneComponent8),
      // TODO: Figure out how to test YUV textures
      // PIXEL_FORMAT(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
      // PIXEL_FORMAT(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
      // PIXEL_FORMAT(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange),
  };
  for (auto [pixelFormat, name] : pixelFormats) {
    Result result;
    auto usage = TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Attachment;
    auto texture = createCVPixelBufferTextureWithSize(pixelFormat, 100, 100, usage, result);
    ASSERT_EQ(result.isOk(), true) << name << ": " << result.message.c_str();
    testUsage(texture, usage, "SampledAttachment");
  }
}

} // namespace igl::tests
