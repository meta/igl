/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/metal/Texture.h>

#import <CoreVideo/CVPixelBuffer.h>
#include <algorithm>
#include <utility>
#include <igl/metal/Buffer.h>
#include <igl/metal/CommandBuffer.h>
#include <igl/metal/PlatformDevice.h>
#include <igl/metal/Texture.h>
#include <igl/tests/util/Common.h>
#include <igl/tests/util/TextureFormatTestBase.h>
#include <igl/tests/util/TextureValidationHelpers.h>

namespace igl::tests {

#define OFFSCREEN_TEX_WIDTH 2
#define OFFSCREEN_TEX_HEIGHT 2

// Picking this to check mipmap validity, max mipmap level = log(16) - 1 = 3
#define MIPMAP_TEX_WIDTH 16
#define MIPMAP_TEX_HEIGHT 16

class TextureMTLTest : public ::testing::Test {
 public:
  TextureMTLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(device_, cmdQueue_);

    texDesc_ = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                  OFFSCREEN_TEX_WIDTH,
                                  OFFSCREEN_TEX_HEIGHT,
                                  TextureDesc::TextureUsageBits::Sampled);
    Result res;
    texture_ = device_->createTexture(texDesc_, &res);
    ASSERT_TRUE(res.isOk());
  }

  void TearDown() override {}

 protected:
  TextureDesc texDesc_;
  std::shared_ptr<ITexture> texture_;
  std::shared_ptr<IDevice> device_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

class TextureFormatMTLTest : public util::TextureFormatTestBase {
 public:
  TextureFormatMTLTest() = default;
};

// Test basic getter methods for successful construction
TEST_F(TextureMTLTest, ConstructionFromMTLTexture) {
  auto* mtlTexture = static_cast<metal::Texture*>(texture_.get());
  ASSERT_NE(mtlTexture->get(), nullptr);

  auto dimensions = texture_->getDimensions();
  ASSERT_EQ(dimensions.width, texDesc_.width);
  ASSERT_EQ(dimensions.height, texDesc_.height);
  ASSERT_EQ(dimensions.depth, texDesc_.depth);

  auto samples = texture_->getSamples();
  ASSERT_EQ(samples, texDesc_.numSamples);

  auto format = texture_->getFormat();
  ASSERT_EQ(format, texDesc_.format);
}

// Test upload
TEST_F(TextureMTLTest, Upload) {
  const auto texRangeDesc = TextureRangeDesc();
  const Result res = texture_->upload(texRangeDesc, nullptr);
  ASSERT_TRUE(res.isOk());
}

// Test getDrawable
TEST_F(TextureMTLTest, GetDrawable) {
  auto mtlTexture = std::dynamic_pointer_cast<metal::Texture>(texture_);
  ASSERT_EQ(mtlTexture->getDrawable(), nullptr);
}

// Test mipmap generation and methods
TEST_F(TextureMTLTest, GetMipmapsCount) {
  Result res;
  TextureDesc miptexDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                              MIPMAP_TEX_WIDTH,
                                              MIPMAP_TEX_HEIGHT,
                                              TextureDesc::TextureUsageBits::Sampled);

  int targetlevel = 0;
  size_t ind = std::min<size_t>(miptexDesc.width, miptexDesc.height);
  while (ind >>= 1) {
    ++targetlevel;
  }

  miptexDesc.numMipLevels = targetlevel; // log(16) - 1
  const std::shared_ptr<ITexture> mipTexture = device_->createTexture(miptexDesc, &res);
  ASSERT_TRUE(res.isOk());

  auto mtlTexture = std::static_pointer_cast<metal::Texture>(mipTexture);
  mtlTexture->generateMipmap(*cmdQueue_);

  // Wait for completion
  const CommandBufferDesc desc = {};
  const std::shared_ptr<ICommandBuffer> cmdBuf = cmdQueue_->createCommandBuffer(desc, &res);
  id<MTLCommandBuffer> mtlCmdBuf = static_cast<igl::metal::CommandBuffer&>(*cmdBuf).get();
  [mtlCmdBuf commit];
  cmdBuf->waitUntilCompleted();

  ASSERT_EQ(mtlTexture->getNumMipLevels(), targetlevel); // Should get log(16) - 1 = 3
}

// Test conversion from MTLTextureType to IGL TextureType
TEST_F(TextureMTLTest, ToTextureType) {
  std::vector<std::pair<MTLTextureType, TextureType>> inputAndExpectedList = {
      std::make_pair(MTLTextureType2D, TextureType::TwoD),
      std::make_pair(MTLTextureType2DMultisample, TextureType::TwoD),
      std::make_pair(MTLTextureType2DArray, TextureType::TwoDArray),
      std::make_pair(MTLTextureType3D, TextureType::ThreeD),
      std::make_pair(MTLTextureTypeCube, TextureType::Cube),
      std::make_pair(MTLTextureTypeCubeArray, TextureType::Invalid),
  };
  if (@available(macOS 10.14, iOS 14.0, *)) {
    inputAndExpectedList.emplace_back(MTLTextureType2DMultisampleArray, TextureType::TwoDArray);
  }

  for (auto inputAndExpected : inputAndExpectedList) {
    auto input = inputAndExpected.first;
    auto expected = inputAndExpected.second;
    auto result = igl::metal::Texture::convertType(input);
    ASSERT_EQ(expected, result);
  }
}

// Test mipmap generation flag initialization
TEST_F(TextureMTLTest, MipmapGenerationFlagInitialization) {
  const auto mtlTexture = std::dynamic_pointer_cast<metal::Texture>(texture_);
  ASSERT_NE(mtlTexture, nullptr);

  // Test that the mipmapGeneration flag is initialized to Manual by default
  ASSERT_EQ(mtlTexture->getMipmapGeneration(), TextureDesc::TextureMipmapGeneration::Manual);
}

// Test auto-generation of mipmaps on upload
TEST_F(TextureMTLTest, AutoGenerateMipmapOnUpload) {
  Result ret;

  // Create a texture with AutoGenerateOnUpload flag - similar to existing mipmap tests
  constexpr uint32_t kNumMipLevels = 2;
  constexpr uint32_t kTexWidth = 2u;
  constexpr uint32_t kTexHeight = 2u;

  constexpr uint32_t kColor = 0xdeadbeef;
  constexpr std::array<uint32_t, 4> kBaseMipData = {kColor, kColor, kColor, kColor};
  constexpr std::array<uint32_t, 1> kExpectedMip1Data = {kColor}; // Should be same color after
                                                                  // generation

  // Create texture with AutoGenerateOnUpload flag
  TextureDesc textureDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                               kTexWidth,
                                               kTexHeight,
                                               TextureDesc::TextureUsageBits::Sampled |
                                                   TextureDesc::TextureUsageBits::Attachment);
  textureDesc.numMipLevels = kNumMipLevels;
  textureDesc.mipmapGeneration = TextureDesc::TextureMipmapGeneration::AutoGenerateOnUpload;

  auto texture = device_->createTexture(textureDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  // Cast to Metal texture to access Metal-specific methods
  auto* mtlTexture = static_cast<metal::Texture*>(texture.get());
  ASSERT_NE(mtlTexture, nullptr);

  // Verify the texture was created with the correct mipmap generation flag
  ASSERT_EQ(mtlTexture->getMipmapGeneration(),
            TextureDesc::TextureMipmapGeneration::AutoGenerateOnUpload);

  // Verify the texture has the expected number of mip levels
  ASSERT_EQ(mtlTexture->getNumMipLevels(), kNumMipLevels);

  // Upload data to mip level 0 - this should trigger automatic mipmap generation
  ret = texture->upload(texture->getFullRange(0), kBaseMipData.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  //  Validate that mip level 0 contains the uploaded data
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue_,
                                     texture,
                                     texture->getFullRange(0),
                                     kBaseMipData.data(),
                                     "AutoGen: Base level (0)");

  // Validate that mip level 1 was auto-generated with expected content
  // The auto-generated mip should contain the same solid color (averaged from base level)
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue_,
                                     texture,
                                     texture->getFullRange(1),
                                     kExpectedMip1Data.data(),
                                     "AutoGen: Generated level (1)");

  // Verify that the Metal texture resources remain valid
  ASSERT_NE(mtlTexture->get(), nullptr);
}

// Test manual mipmap generation for comparison
TEST_F(TextureMTLTest, ManualMipmapGeneration) {
  Result ret;

  // Create a texture with Manual mipmap generation - follow same pattern as existing tests
  constexpr uint32_t kNumMipLevels = 2;
  constexpr uint32_t kTexWidth = 2u;
  constexpr uint32_t kTexHeight = 2u;

  constexpr uint32_t kColor = 0x8badf00d; // Different color from auto test
  constexpr std::array<uint32_t, 4> kBaseMipData = {kColor, kColor, kColor, kColor};
  constexpr std::array<uint32_t, 1> kInitialMip1Data = {0}; // Initialize mip 1 with zero
  constexpr std::array<uint32_t, 1> kExpectedMip1Data = {kColor}; // Should match base after
                                                                  // generation

  // Create texture with Manual mipmap generation
  TextureDesc textureDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                               kTexWidth,
                                               kTexHeight,
                                               TextureDesc::TextureUsageBits::Sampled |
                                                   TextureDesc::TextureUsageBits::Attachment);
  textureDesc.numMipLevels = kNumMipLevels;
  textureDesc.mipmapGeneration = TextureDesc::TextureMipmapGeneration::Manual;

  auto texture = device_->createTexture(textureDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  auto* mtlTexture = static_cast<metal::Texture*>(texture.get());
  ASSERT_NE(mtlTexture, nullptr);

  // Verify the texture was created with Manual mipmap generation
  ASSERT_EQ(mtlTexture->getMipmapGeneration(), TextureDesc::TextureMipmapGeneration::Manual);

  // Verify the texture has the expected number of mip levels
  ASSERT_EQ(mtlTexture->getNumMipLevels(), kNumMipLevels);

  // Upload data to base mip level (level 0)
  ret = texture->upload(texture->getFullRange(0), kBaseMipData.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Upload initial data to mip level 1 (will be overwritten by generateMipmap)
  ret = texture->upload(texture->getFullRange(1), kInitialMip1Data.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Validate initial state - base level should have uploaded data
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue_,
                                     texture,
                                     texture->getFullRange(0),
                                     kBaseMipData.data(),
                                     "Manual: Initial base level (0)");

  // Validate initial state - mip level 1 should have initial zero data
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue_,
                                     texture,
                                     texture->getFullRange(1),
                                     kInitialMip1Data.data(),
                                     "Manual: Initial mip level (1)");

  // Now manually generate mipmaps - this should overwrite mip level 1
  mtlTexture->generateMipmap(*cmdQueue_, nullptr);

  // Validate final state - base level should still have original data
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue_,
                                     texture,
                                     texture->getFullRange(0),
                                     kBaseMipData.data(),
                                     "Manual: Final base level (0)");

  // Validate final state - mip level 1 should now have generated data
  util::validateUploadedTextureRange(*device_,
                                     *cmdQueue_,
                                     texture,
                                     texture->getFullRange(1),
                                     kExpectedMip1Data.data(),
                                     "Manual: Generated mip level (1)");
}

// Test conversion from IGL TextureType to MTLTextureType
// Current IGL makes the following assumptions, which may not be valid in the future
//   * Falls back to MTLTextureType1D for TextureType::Invalid
TEST_F(TextureMTLTest, ToMTLTextureType) {
  std::vector<std::tuple<TextureType, size_t, MTLTextureType>> inputAndExpectedList = {
      std::make_tuple(TextureType::Invalid, 1, MTLTextureType1D),
      std::make_tuple(TextureType::TwoD, 1, MTLTextureType2D),
      std::make_tuple(TextureType::TwoD, 2, MTLTextureType2DMultisample),
      std::make_tuple(TextureType::TwoDArray, 1, MTLTextureType2DArray),
      std::make_tuple(TextureType::ThreeD, 1, MTLTextureType3D),
      std::make_tuple(TextureType::Cube, 1, MTLTextureTypeCube)};
  if (@available(macOS 10.14, iOS 14.0, *)) {
    inputAndExpectedList.emplace_back(TextureType::TwoDArray, 2, MTLTextureType2DMultisampleArray);
  }

  for (auto inputAndExpected : inputAndExpectedList) {
    auto input = std::get<0>(inputAndExpected);
    auto numSamples = std::get<1>(inputAndExpected);
    auto expected = std::get<2>(inputAndExpected);
    auto result = igl::metal::Texture::convertType(input, numSamples);
    ASSERT_EQ(expected, result);
  }
}

namespace {
std::shared_ptr<ITexture> createCVPixelBufferTextureWithSize(TextureFormat format,
                                                             const size_t width,
                                                             const size_t height,
                                                             const std::shared_ptr<IDevice>& device,
                                                             Result& outResult) {
  const BackendType backend = device->getBackendType();
  CVPixelBufferRef pixelBuffer = nullptr;
  NSDictionary* bufferAttributes = @{
    (NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (NSString*)kCVPixelBufferMetalCompatibilityKey : @(backend == igl::BackendType::Metal),
  };

  const CVReturn result = CVPixelBufferCreate(kCFAllocatorDefault,
                                              width,
                                              height,
                                              kCVPixelFormatType_32BGRA,
                                              (__bridge CFDictionaryRef)(bufferAttributes),
                                              &pixelBuffer);
  if (result != kCVReturnSuccess) {
    Result::setResult(&outResult,
                      Result::Code::RuntimeError,
                      "CVPixelBufferCreate failed to create pixel buffer");
    return nullptr;
  }

  auto* platformDevice = device->getPlatformDevice<igl::metal::PlatformDevice>();
  std::shared_ptr<ITexture> texture =
      platformDevice->createTextureFromNativePixelBuffer(pixelBuffer, format, 0, &outResult);
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
} // namespace

TEST_F(TextureFormatMTLTest, createTextureFromNativePixelBufferSuccess) {
  Result result;
  auto texture =
      createCVPixelBufferTextureWithSize(TextureFormat::BGRA_UNorm8, 100, 100, iglDev_, result);
  ASSERT_EQ(result.isOk(), true) << result.message.c_str();
  testUsage(texture,
            TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Attachment,
            "SampledAttachment");
}

TEST_F(TextureMTLTest, createTextureFromNativePixelBufferWithInvalidFormat) {
  Result result;
  auto texture =
      createCVPixelBufferTextureWithSize(TextureFormat::Invalid, 100, 100, device_, result);
  ASSERT_EQ(result.isOk(), false) << result.message.c_str();
}

TEST_F(TextureMTLTest, resetNativeTextureCacheAllowsBufferReuse) {
  auto* platformDevice = device_->getPlatformDevice<igl::metal::PlatformDevice>();
  platformDevice->resetNativeTextureCache();
  Result result;
  @autoreleasepool {
    auto texture =
        createCVPixelBufferTextureWithSize(TextureFormat::BGRA_UNorm8, 64, 64, device_, result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(texture, nullptr);
  }
  platformDevice->resetNativeTextureCache();
  platformDevice->resetNativeTextureCache();
  @autoreleasepool {
    auto texture =
        createCVPixelBufferTextureWithSize(TextureFormat::BGRA_UNorm8, 128, 64, device_, result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(texture, nullptr);
    EXPECT_EQ(texture->getDimensions().width, 128u);
    EXPECT_EQ(texture->getDimensions().height, 64u);
  }
  platformDevice->resetNativeTextureCache();
}

TEST_F(TextureMTLTest, createTextureFromNativePixelBufferWithInvalidPlaneSetsError) {
  CVPixelBufferRef pixelBuffer = nullptr;
  NSDictionary* bufferAttributes = @{
    (NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (NSString*)kCVPixelBufferMetalCompatibilityKey : @YES,
  };
  const CVReturn pixelBufferResult =
      CVPixelBufferCreate(kCFAllocatorDefault,
                          100,
                          100,
                          kCVPixelFormatType_32BGRA,
                          (__bridge CFDictionaryRef)(bufferAttributes),
                          &pixelBuffer);
  ASSERT_EQ(pixelBufferResult, kCVReturnSuccess);

  auto* platformDevice = device_->getPlatformDevice<igl::metal::PlatformDevice>();
  Result result;
  auto texture = platformDevice->createTextureFromNativePixelBuffer(
      pixelBuffer, TextureFormat::BGRA_UNorm8, 1, &result);
  CVPixelBufferRelease(pixelBuffer);

  EXPECT_EQ(texture, nullptr);
  EXPECT_FALSE(result.isOk());
}

TEST_F(TextureMTLTest, B10G11R11UFloatRoundTrip) {
  constexpr auto kIglFormat = TextureFormat::B10G11R11_UFloat;
  constexpr auto kMtlFormat = MTLPixelFormatRG11B10Float;

  EXPECT_EQ(metal::Texture::textureFormatToMTLPixelFormat(kIglFormat), kMtlFormat);
  EXPECT_EQ(metal::Texture::mtlPixelFormatToTextureFormat(kMtlFormat), kIglFormat);
  EXPECT_EQ(metal::Texture::mtlPixelFormatToTextureFormat(
                metal::Texture::textureFormatToMTLPixelFormat(kIglFormat)),
            kIglFormat);
}

TEST_F(TextureMTLTest, ConvertTextureFormats) {
  const std::vector<TextureFormat> inputFormats = {
      TextureFormat::A_UNorm8,
      TextureFormat::R_UNorm8,
      TextureFormat::R_F16,
      TextureFormat::R_UInt16,
#if !IGL_PLATFORM_MACOSX
      TextureFormat::B5G5R5A1_UNorm,
#endif
#if !(IGL_PLATFORM_MACOSX || IGL_PLATFORM_IOS_SIMULATOR)
      // https://developer.apple.com/documentation/metal/developing_metal_apps_that_run_in_simulator
      TextureFormat::B5G6R5_UNorm,
      TextureFormat::ABGR_UNorm4,
#endif
      TextureFormat::RG_UNorm8,
      TextureFormat::R4G2B2_UNorm_Apple,
      TextureFormat::R4G2B2_UNorm_Rev_Apple,
      TextureFormat::RGBA_UNorm8,
      TextureFormat::BGRA_UNorm8,
      TextureFormat::RG_F16,
      TextureFormat::RG_UInt16,
      TextureFormat::RGB10_A2_UNorm_Rev,
      TextureFormat::RGB10_A2_Uint_Rev,
      TextureFormat::BGR10_A2_Unorm,
      TextureFormat::R_F32,
      TextureFormat::R_UInt32,
      TextureFormat::RG_F32,
      TextureFormat::RGBA_F16,
      TextureFormat::RGBA_UInt32,
      TextureFormat::RGBA_F32,
#if !IGL_PLATFORM_MACOSX
      TextureFormat::RGBA_ASTC_4x4,
      TextureFormat::SRGB8_A8_ASTC_4x4,
      TextureFormat::RGBA_ASTC_5x4,
      TextureFormat::SRGB8_A8_ASTC_5x4,
      TextureFormat::RGBA_ASTC_5x5,
      TextureFormat::SRGB8_A8_ASTC_5x5,
      TextureFormat::RGBA_ASTC_6x5,
      TextureFormat::SRGB8_A8_ASTC_6x5,
      TextureFormat::RGBA_ASTC_6x6,
      TextureFormat::SRGB8_A8_ASTC_6x6,
      TextureFormat::RGBA_ASTC_8x5,
      TextureFormat::SRGB8_A8_ASTC_8x5,
      TextureFormat::RGBA_ASTC_8x6,
      TextureFormat::SRGB8_A8_ASTC_8x6,
      TextureFormat::RGBA_ASTC_8x8,
      TextureFormat::SRGB8_A8_ASTC_8x8,
      TextureFormat::RGBA_ASTC_10x5,
      TextureFormat::SRGB8_A8_ASTC_10x5,
      TextureFormat::RGBA_ASTC_10x6,
      TextureFormat::SRGB8_A8_ASTC_10x6,
      TextureFormat::RGBA_ASTC_10x8,
      TextureFormat::SRGB8_A8_ASTC_10x8,
      TextureFormat::RGBA_ASTC_10x10,
      TextureFormat::SRGB8_A8_ASTC_10x10,
      TextureFormat::RGBA_ASTC_12x10,
      TextureFormat::SRGB8_A8_ASTC_12x10,
      TextureFormat::RGBA_ASTC_12x12,
      TextureFormat::SRGB8_A8_ASTC_12x12,
      TextureFormat::RGBA_PVRTC_2BPPV1,
      TextureFormat::RGB_PVRTC_2BPPV1,
      TextureFormat::RGBA_PVRTC_4BPPV1,
      TextureFormat::RGB_PVRTC_4BPPV1,
      TextureFormat::RGB8_ETC1,
      TextureFormat::RGB8_ETC2,
      TextureFormat::SRGB8_ETC2,
      TextureFormat::RGBA8_EAC_ETC2,
      TextureFormat::SRGB8_A8_EAC_ETC2,
      TextureFormat::RG_EAC_UNorm,
      TextureFormat::RG_EAC_SNorm,
      TextureFormat::R_EAC_UNorm,
      TextureFormat::R_EAC_SNorm,
#endif
      TextureFormat::Z_UNorm16,
      TextureFormat::Z_UNorm24,
      TextureFormat::Z_UNorm32,
      TextureFormat::S8_UInt_Z24_UNorm,
      TextureFormat::S8_UInt_Z32_UNorm,
      TextureFormat::S_UInt8};

  const std::vector<TextureFormat> invalidTextureFormats = {
      TextureFormat::Invalid,
      TextureFormat::L_UNorm8,
#if IGL_PLATFORM_MACOSX
      TextureFormat::B5G5R5A1_UNorm,
#endif
#if IGL_PLATFORM_MACOSX || IGL_PLATFORM_IOS_SIMULATOR
      TextureFormat::B5G6R5_UNorm,
      TextureFormat::ABGR_UNorm4,
#endif
      TextureFormat::LA_UNorm8,
      TextureFormat::R5G5B5A1_UNorm,
      TextureFormat::RGBX_UNorm8,
      TextureFormat::BGRA_UNorm8_Rev,
      TextureFormat::RGB_F16,
      TextureFormat::RGB_F32,
#if IGL_PLATFORM_MACOSX
      TextureFormat::RGBA_ASTC_4x4,
      TextureFormat::SRGB8_A8_ASTC_4x4,
      TextureFormat::RGBA_ASTC_5x4,
      TextureFormat::SRGB8_A8_ASTC_5x4,
      TextureFormat::RGBA_ASTC_5x5,
      TextureFormat::SRGB8_A8_ASTC_5x5,
      TextureFormat::RGBA_ASTC_6x5,
      TextureFormat::SRGB8_A8_ASTC_6x5,
      TextureFormat::RGBA_ASTC_6x6,
      TextureFormat::SRGB8_A8_ASTC_6x6,
      TextureFormat::RGBA_ASTC_8x5,
      TextureFormat::SRGB8_A8_ASTC_8x5,
      TextureFormat::RGBA_ASTC_8x6,
      TextureFormat::SRGB8_A8_ASTC_8x6,
      TextureFormat::RGBA_ASTC_8x8,
      TextureFormat::SRGB8_A8_ASTC_8x8,
      TextureFormat::RGBA_ASTC_10x5,
      TextureFormat::SRGB8_A8_ASTC_10x5,
      TextureFormat::RGBA_ASTC_10x6,
      TextureFormat::SRGB8_A8_ASTC_10x6,
      TextureFormat::RGBA_ASTC_10x8,
      TextureFormat::SRGB8_A8_ASTC_10x8,
      TextureFormat::RGBA_ASTC_10x10,
      TextureFormat::SRGB8_A8_ASTC_10x10,
      TextureFormat::RGBA_ASTC_12x10,
      TextureFormat::SRGB8_A8_ASTC_12x10,
      TextureFormat::RGBA_ASTC_12x12,
      TextureFormat::SRGB8_A8_ASTC_12x12,
      TextureFormat::RGBA_PVRTC_2BPPV1,
      TextureFormat::RGB_PVRTC_2BPPV1,
      TextureFormat::RGBA_PVRTC_4BPPV1,
      TextureFormat::RGB_PVRTC_4BPPV1,
      TextureFormat::RGB8_ETC1,
      TextureFormat::RGB8_ETC2,
      TextureFormat::SRGB8_ETC2,
#endif
      TextureFormat::RGB8_Punchthrough_A1_ETC2,
      TextureFormat::SRGB8_Punchthrough_A1_ETC2,
#if IGL_PLATFORM_MACOSX
      TextureFormat::RGBA8_EAC_ETC2,
      TextureFormat::SRGB8_A8_EAC_ETC2,
      TextureFormat::RG_EAC_UNorm,
      TextureFormat::RG_EAC_SNorm,
      TextureFormat::R_EAC_UNorm,
      TextureFormat::R_EAC_SNorm,
#endif
  };

  for (auto format : invalidTextureFormats) {
    ASSERT_EQ(true,
              igl::metal::Texture::textureFormatToMTLPixelFormat(format) == MTLPixelFormatInvalid);
  }
}

// Test conversion from IGL TextureUsage bits to MTLTextureUsage
TEST_F(TextureMTLTest, ToMTLTextureUsage) {
  EXPECT_EQ(igl::metal::Texture::toMTLTextureUsage(0), MTLTextureUsageUnknown);
  EXPECT_EQ(igl::metal::Texture::toMTLTextureUsage(TextureDesc::TextureUsageBits::Sampled),
            MTLTextureUsageShaderRead);
  EXPECT_EQ(igl::metal::Texture::toMTLTextureUsage(TextureDesc::TextureUsageBits::Storage),
            MTLTextureUsageShaderWrite);
  EXPECT_EQ(igl::metal::Texture::toMTLTextureUsage(TextureDesc::TextureUsageBits::Attachment),
            MTLTextureUsageRenderTarget);
  EXPECT_EQ(igl::metal::Texture::toMTLTextureUsage(TextureDesc::TextureUsageBits::Sampled |
                                                   TextureDesc::TextureUsageBits::Storage |
                                                   TextureDesc::TextureUsageBits::Attachment),
            MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget);
}

// Test conversion from MTLTextureUsage to IGL TextureUsage bits
TEST_F(TextureMTLTest, ToTextureUsage) {
  EXPECT_EQ(igl::metal::Texture::toTextureUsage(MTLTextureUsageUnknown), 0);
  EXPECT_EQ(igl::metal::Texture::toTextureUsage(MTLTextureUsageShaderRead),
            TextureDesc::TextureUsageBits::Sampled);
  EXPECT_EQ(igl::metal::Texture::toTextureUsage(MTLTextureUsageShaderWrite),
            TextureDesc::TextureUsageBits::Storage);
  EXPECT_EQ(igl::metal::Texture::toTextureUsage(MTLTextureUsageRenderTarget),
            TextureDesc::TextureUsageBits::Attachment);
  EXPECT_EQ(
      igl::metal::Texture::toTextureUsage(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                                          MTLTextureUsageRenderTarget),
      TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Storage |
          TextureDesc::TextureUsageBits::Attachment);
}

// Test memoryless attachment texture storage mode
TEST_F(TextureMTLTest, MemorylessTextureStorageMode) {
  Result ret;

  // Create a memoryless attachment texture
  TextureDesc memorylessDesc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Attachment);
  memorylessDesc.storage = ResourceStorage::Memoryless;

  auto memorylessTexture = device_->createTexture(memorylessDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(memorylessTexture, nullptr);

  auto* mtlMemorylessTexture = static_cast<metal::Texture*>(memorylessTexture.get());
  ASSERT_NE(mtlMemorylessTexture, nullptr);

  // Check the storage mode
  if (device_->hasFeature(DeviceFeatures::MemorylessAttachments)) {
    EXPECT_EQ(mtlMemorylessTexture->get().storageMode, MTLStorageModeMemoryless);
  } else {
    EXPECT_EQ(mtlMemorylessTexture->get().storageMode, MTLStorageModePrivate);
  }
}

// Test memoryless buffer storage mode
TEST_F(TextureMTLTest, MemorylessBufferStorageMode) {
  Result ret;

  const auto memorylessBuffer =
      device_->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                                       .length = 256,
                                       .storage = ResourceStorage::Memoryless},
                            &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(memorylessBuffer, nullptr);

  auto* mtlBuffer = static_cast<metal::Buffer*>(memorylessBuffer.get());

  // Check the storage mode
#if IGL_PLATFORM_MACOSX
  EXPECT_EQ(mtlBuffer->get().storageMode, MTLStorageModeManaged);
#else
  EXPECT_EQ(mtlBuffer->get().storageMode, MTLStorageModeShared);
#endif
}

} // namespace igl::tests
