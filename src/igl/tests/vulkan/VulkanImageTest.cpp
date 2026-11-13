/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanImage.h>

#include <array>
#include <cstdint>
#include <glm/common.hpp>
#include <glm/gtc/color_space.hpp>
#include <memory>
#include <igl/CommandBuffer.h>
#include <igl/CommandQueue.h>
#include <igl/Common.h>
#include <igl/Texture.h>
#include <igl/tests/util/device/TestDevice.h>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanImmediateCommands.h>
#include <igl/vulkan/VulkanStagingDevice.h>
#include <igl/vulkan/VulkanTexture.h>
#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
#include <android/hardware_buffer.h>
#endif

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX

namespace igl::tests {

namespace {
constexpr uint32_t kWidth = 1024;
constexpr uint32_t kHeight = 1024;
constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

constexpr uint32_t kBytesPerTexel = 4;
constexpr uint8_t kOpaque = 255;
constexpr int kTolerance = 4; // 8-bit quantization + transfer-function approximation

constexpr uint32_t kCheckerSize = 8; // 8x8 -> 4x4 -> 2x2 -> 1x1
constexpr uint32_t kCheckerMipLevels = 4;

// The mean linear light of the checkerboard source: half its texels sit at 0.0, half at 1.0.
// Downsampling is an averaging filter, so every generated level must carry this same mean.
constexpr float kSourceMeanLight = 0.5f;

// A saturated orange. Its channels sit at three very different points on the sRGB curve (1.0, mid,
// 0.0), where the gap between filtering in linear and in encoded space is both large and *unequal*
// per channel. A grey probe moves all three channels identically, so it cannot tell a gamma
// mistake apart from a swizzle or from a conversion applied to only some channels.
constexpr uint8_t kOrangeR = 255;
constexpr uint8_t kOrangeG = 128;
constexpr uint8_t kOrangeB = 0;

uint32_t packRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(b) << 16) |
         (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(r);
}

int channel(uint32_t texel, uint32_t index) {
  return static_cast<int>((texel >> (index * 8)) & 0xFFu);
}

glm::vec3 unitSrgb(uint8_t r, uint8_t g, uint8_t b) {
  return glm::vec3(r, g, b) / 255.0f;
}

glm::ivec3 toBytes(const glm::vec3& color) {
  return {glm::round(color * 255.0f)};
}

// Encoded byte -> the light it actually represents.
float toLinear(int encoded) {
  return glm::convertSRGBToLinear(glm::vec3(static_cast<float>(encoded) / 255.0f)).r;
}

// Light -> the byte that encodes it. Expectations are derived through the transfer function rather
// than hardcoded so they cannot drift from the conversion the implementation performs.
int toEncoded(float linear) {
  return static_cast<int>(glm::round(glm::convertLinearToSRGB(glm::vec3(linear)).r * 255.0f));
}

// Re-encodes the *linear* mean of `color` and black in equal parts: what a gamma-correct 2x
// downsample of a half-black, half-`color` image must produce. For orange this is {188, 92, 0},
// against the {127, 64, 0} an encoded-space blit lands on.
glm::ivec3 gammaCorrectHalfBlack(const glm::vec3& color) {
  return toBytes(glm::convertLinearToSRGB(0.5f * glm::convertSRGBToLinear(color)));
}

void expectRGB(uint32_t texel, const glm::ivec3& expected, int tolerance) {
  EXPECT_NEAR(channel(texel, 0), expected.r, tolerance) << "red";
  EXPECT_NEAR(channel(texel, 1), expected.g, tolerance) << "green";
  EXPECT_NEAR(channel(texel, 2), expected.b, tolerance) << "blue";
  // The sRGB transfer function applies to RGB only, so alpha must survive untouched.
  EXPECT_NEAR(channel(texel, 3), kOpaque, 1) << "alpha must not be gamma-converted";
}

void expectGreyAndOpaque(uint32_t texel) {
  EXPECT_NEAR(channel(texel, 1), channel(texel, 0), 1) << "green must match red";
  EXPECT_NEAR(channel(texel, 2), channel(texel, 0), 1) << "blue must match red";
  // The sRGB transfer function applies to RGB only, so alpha must survive untouched.
  EXPECT_NEAR(channel(texel, 3), kOpaque, 1) << "alpha must stay opaque";
}

// Alternating black and white texels. Every 2x2 block holds two of each, so mip 1 and every level
// below it is a uniform grey -- which makes the top-left texel representative of the whole level.
std::array<uint32_t, kCheckerSize * kCheckerSize> checkerboardMip0() {
  std::array<uint32_t, kCheckerSize * kCheckerSize> mip0{};
  for (uint32_t y = 0; y < kCheckerSize; ++y) {
    for (uint32_t x = 0; x < kCheckerSize; ++x) {
      const uint8_t value = ((x + y) % 2 == 0) ? 0 : 255;
      mip0[(y * kCheckerSize) + x] = packRGBA(value, value, value, kOpaque);
    }
  }
  return mip0;
}

// Puts the size of the error in the test log, so "how far off is it" is a measured number rather
// than something the reader has to infer from the expectation.
void logRetainedLight(uint32_t level, int encoded) {
  const float light = toLinear(encoded);
  GTEST_LOG_(INFO) << "mip " << level << ": encoded " << encoded << ", linear light " << light
                   << " = " << (100.0f * light / kSourceMeanLight) << "% of the source's "
                   << kSourceMeanLight;
}
} // namespace

//
// VulkanImageTest
//
// Unit tests for igl::vulkan::VulkanImage.
//
class VulkanImageTest : public ::testing::Test {
 public:
  // Set up common resources.
  void SetUp() override {
    // Turn off debug break so unit tests can run
    igl::setDebugBreakEnabled(false);

    device_ = igl::tests::util::device::createTestDevice(igl::BackendType::Vulkan);
    ASSERT_TRUE(device_ != nullptr);
    auto& device = static_cast<igl::vulkan::Device&>(*device_);
    context_ = &device.getVulkanContext();
    ASSERT_TRUE(context_ != nullptr);
    ASSERT_TRUE(context_->stagingDevice_ != nullptr);
    ASSERT_TRUE(context_->immediate_ != nullptr);
  }

 protected:
  [[nodiscard]] vulkan::VulkanImage makeImage(
      VkFormat format,
      VkImageCreateFlags createFlags,
      const char* debugName,
      bool isSrgbMutableFormat = false,
      uint32_t size = 2,
      uint32_t mipLevels = 1,
      uint32_t arrayLayers = 1,
      VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT) const {
    return vulkan::VulkanImage(*context_,
                               VkExtent3D{.width = size, .height = size, .depth = 1},
                               VK_IMAGE_TYPE_2D,
                               format,
                               mipLevels,
                               arrayLayers,
                               VK_IMAGE_TILING_OPTIMAL,
                               usageFlags,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               createFlags,
                               VK_SAMPLE_COUNT_1_BIT,
                               debugName,
                               isSrgbMutableFormat);
  }

  [[nodiscard]] static TextureFormatProperties properties() {
    return TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
  }

  [[nodiscard]] bool supportsLinearBlitDownscale(VkFormat format = VK_FORMAT_R8G8B8A8_UNORM) const {
    VkFormatProperties formatProperties{};
    context_->vf_.vkGetPhysicalDeviceFormatProperties(
        context_->getVkPhysicalDevice(), format, &formatProperties);
    constexpr VkFormatFeatureFlags kRequiredFeatures =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (formatProperties.optimalTilingFeatures & kRequiredFeatures) == kRequiredFeatures;
  }

  // generateMipmapSrgb() downsamples on a native-sRGB scratch, so that format has to support the
  // blit as well -- the UNORM base image only ever takes part in copies on this path.
  [[nodiscard]] bool supportsSrgbScratchMipmap() const {
    return supportsLinearBlitDownscale(VK_FORMAT_R8G8B8A8_UNORM) &&
           supportsLinearBlitDownscale(VK_FORMAT_R8G8B8A8_SRGB);
  }

  void uploadMip0(vulkan::VulkanImage& image,
                  TextureType type,
                  const TextureRangeDesc& range,
                  uint32_t size,
                  const void* data) const {
    EXPECT_TRUE(context_->stagingDevice_->imageData(
        image, type, range, properties(), size * kBytesPerTexel, VK_IMAGE_ASPECT_COLOR_BIT, data));
  }

  Result generateMips(vulkan::VulkanImage& image, const TextureRangeDesc& range) const {
    auto& ctx = *context_;
    const auto& wrapper = ctx.immediate_->acquire();
    Result result = image.generateMipmap(wrapper.cmdBuf, range);
    ctx.immediate_->wait(ctx.immediate_->submit(wrapper), ctx.config_.fenceTimeoutNanoseconds);
    return result;
  }

  // Reads the top-left texel of `level` / `layer`.
  [[nodiscard]] uint32_t readTexel(const vulkan::VulkanImage& image,
                                   uint32_t level,
                                   uint32_t layer) const {
    uint32_t texel = 0;
    EXPECT_TRUE(
        context_->stagingDevice_->getImageData2D(image.getVkImage(),
                                                 level,
                                                 layer,
                                                 VkRect2D{.offset = {0, 0}, .extent = {1, 1}},
                                                 properties(),
                                                 VK_FORMAT_R8G8B8A8_UNORM,
                                                 image.imageLayout_,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 &texel,
                                                 kBytesPerTexel,
                                                 false /* flipImageVertical */));
    return texel;
  }

  // Uploads `mip0` into a 2x2 R8G8B8A8_UNORM image, generates its mipmap and returns the single
  // texel of mip 1. `createFlags` selects the path under test: VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
  // sets isSrgbMutableFormat_ (sRGB scratch), 0 leaves the plain in-place blit.
  void downsampleToMip1(VkImageCreateFlags createFlags,
                        const std::array<uint32_t, 4>& mip0,
                        uint32_t& outTexel) {
    constexpr uint32_t kSize = 2; // 2x2 -> 1x1
    constexpr uint32_t kNumMipLevels = 2;

    vulkan::VulkanImage image =
        makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                  createFlags,
                  "Image: sRGB mipmap",
                  (createFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0,
                  kSize,
                  kNumMipLevels,
                  1 /* arrayLayers */,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ASSERT_TRUE(image.valid());
    ASSERT_EQ(image.isSrgbMutableFormat_, (createFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0);

    uploadMip0(
        image, TextureType::TwoD, TextureRangeDesc::new2D(0, 0, kSize, kSize), kSize, mip0.data());
    generateMips(image, TextureRangeDesc::new2D(0, 0, kSize, kSize, 0, kNumMipLevels));

    outTexel = readTexel(image, 1 /* level */, 0 /* layer */);
  }

  // Half black, half `color`: gamma-correct downsampling re-encodes the linear mean, while
  // averaging the encoded bytes lands on half the encoded value.
  static std::array<uint32_t, 4> halfBlackMip0(uint32_t color) {
    return {packRGBA(0, 0, 0, kOpaque), packRGBA(0, 0, 0, kOpaque), color, color};
  }

  static uint32_t orange() {
    return packRGBA(kOrangeR, kOrangeG, kOrangeB, kOpaque);
  }

  static glm::ivec3 orangeBytes() {
    return {kOrangeR, kOrangeG, kOrangeB};
  }

  static glm::ivec3 gammaCorrectHalfOrange() {
    return gammaCorrectHalfBlack(unitSrgb(kOrangeR, kOrangeG, kOrangeB));
  }

  // Uploads a black-and-white checkerboard, generates the full mip chain and returns the grey each
  // generated level settled on. `createFlags` selects the path under test.
  void checkerboardMipChainGreys(VkImageCreateFlags createFlags,
                                 std::array<int, kCheckerMipLevels>& outGreys) {
    const std::array<uint32_t, kCheckerSize * kCheckerSize> mip0 = checkerboardMip0();

    vulkan::VulkanImage image =
        makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                  createFlags,
                  "Image: sRGB checkerboard mipmap",
                  (createFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0,
                  kCheckerSize,
                  kCheckerMipLevels,
                  1 /* arrayLayers */,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ASSERT_TRUE(image.valid());
    ASSERT_EQ(image.isSrgbMutableFormat_, (createFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0);

    uploadMip0(image,
               TextureType::TwoD,
               TextureRangeDesc::new2D(0, 0, kCheckerSize, kCheckerSize),
               kCheckerSize,
               mip0.data());
    generateMips(image,
                 TextureRangeDesc::new2D(0, 0, kCheckerSize, kCheckerSize, 0, kCheckerMipLevels));

    for (uint32_t level = 1; level < kCheckerMipLevels; ++level) {
      const uint32_t texel = readTexel(image, level, 0 /* layer */);
      SCOPED_TRACE(testing::Message() << "mip level " << level);
      // The generated levels are uniform grey, so one channel characterizes the level.
      expectGreyAndOpaque(texel);
      outGreys[level] = channel(texel, 0);
      logRetainedLight(level, outGreys[level]);
    }
  }

  std::shared_ptr<IDevice> device_;
  vulkan::VulkanContext* context_ = nullptr;
};

TEST_F(VulkanImageTest, DefaultConstructedIsInvalid) {
  const vulkan::VulkanImage image;

  EXPECT_FALSE(image.valid());
  EXPECT_EQ(image.getVkImage(), VK_NULL_HANDLE);
  EXPECT_EQ(image.getVkImageUsageFlags(), 0u);
  EXPECT_FALSE(image.isSampledImage());
  EXPECT_FALSE(image.isStorageImage());
}

TEST_F(VulkanImageTest, CreateBasicImage) {
  vulkan::VulkanImage image(*context_,
                            VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                            VK_IMAGE_TYPE_2D,
                            kFormat,
                            1, /* mipLevels */
                            1, /* arrayLayers */
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /* memFlags */
                            0, /* createFlags */
                            VK_SAMPLE_COUNT_1_BIT,
                            "Test Basic Image");

  EXPECT_TRUE(image.valid());
  EXPECT_NE(image.getVkImage(), VK_NULL_HANDLE);
  EXPECT_EQ(image.extent_.width, kWidth);
  EXPECT_EQ(image.extent_.height, kHeight);
  EXPECT_EQ(image.extent_.depth, 1u);
  EXPECT_EQ(image.imageFormat_, kFormat);
  EXPECT_EQ(image.mipLevels_, 1u);
  EXPECT_EQ(image.arrayLayers_, 1u);
  EXPECT_EQ(image.samples_, VK_SAMPLE_COUNT_1_BIT);
  EXPECT_FALSE(image.isExternallyManaged_);
  EXPECT_FALSE(image.isImported_);
  EXPECT_FALSE(image.isExported_);
}

TEST_F(VulkanImageTest, UsageFlagQueries) {
  const vulkan::VulkanImage sampledImage(*context_,
                                         VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                                         VK_IMAGE_TYPE_2D,
                                         kFormat,
                                         1, /* mipLevels */
                                         1, /* arrayLayers */
                                         VK_IMAGE_TILING_OPTIMAL,
                                         VK_IMAGE_USAGE_SAMPLED_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /* memFlags */
                                         0, /* createFlags */
                                         VK_SAMPLE_COUNT_1_BIT,
                                         "Sampled Image");

  EXPECT_TRUE(sampledImage.isSampledImage());
  EXPECT_FALSE(sampledImage.isStorageImage());
  EXPECT_EQ(sampledImage.getVkImageUsageFlags() & VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_USAGE_SAMPLED_BIT);

  const vulkan::VulkanImage storageImage(*context_,
                                         VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                                         VK_IMAGE_TYPE_2D,
                                         kFormat,
                                         1, /* mipLevels */
                                         1, /* arrayLayers */
                                         VK_IMAGE_TILING_OPTIMAL,
                                         VK_IMAGE_USAGE_STORAGE_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /* memFlags */
                                         0, /* createFlags */
                                         VK_SAMPLE_COUNT_1_BIT,
                                         "Storage Image");

  EXPECT_FALSE(storageImage.isSampledImage());
  EXPECT_TRUE(storageImage.isStorageImage());
  EXPECT_EQ(storageImage.getVkImageUsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_USAGE_STORAGE_BIT);
}

TEST_F(VulkanImageTest, CreateImageWithExportedMemory) {
  auto vulkanImage = igl::vulkan::VulkanImage::createWithExportMemory(
      *context_,
      VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
      VK_IMAGE_TYPE_2D,
      kFormat,
      1, /* mipLevels */
      1, /* arrayLayers */
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      0, /* createFlags */
      VK_SAMPLE_COUNT_1_BIT,
      "Image: vulkan export memory");
  ASSERT_NE(vulkanImage.valid(), false);
  EXPECT_TRUE(vulkanImage.isExported_);
  // The exported memory's size must be populated for consumers (e.g.
  // RemoteVulkanIglSwapchain) that forward it to importers via IPC.
  EXPECT_GT(vulkanImage.allocatedSize, 0u);
#if IGL_PLATFORM_WINDOWS
  EXPECT_NE(vulkanImage.exportedMemoryHandle_, nullptr);
  EXPECT_NE(vulkanImage.getVkImage(), static_cast<VkImage_T*>(VK_NULL_HANDLE));
#elif IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX
  EXPECT_NE(vulkanImage.exportedFd_, -1);
  EXPECT_NE(vulkanImage.getVkImage(), VK_NULL_HANDLE);
#endif
}

#if IGL_PLATFORM_WINDOWS
TEST_F(VulkanImageTest, CreateImageWithImportedMemoryWin32) {
  auto exportedImage = igl::vulkan::VulkanImage::createWithExportMemory(
      *context_,
      VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
      VK_IMAGE_TYPE_2D,
      kFormat,
      1, /* mipLevels */
      1, /* arrayLayers */
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      0, /* createFlags */
      VK_SAMPLE_COUNT_1_BIT,
      "Image: vulkan export memory");
  ASSERT_NE(exportedImage.valid(), false);
  EXPECT_NE(exportedImage.exportedMemoryHandle_, nullptr);

  auto importedImage =
      igl::vulkan::VulkanImage(*context_,
                               exportedImage.exportedMemoryHandle_,
                               VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                               VK_IMAGE_TYPE_2D,
                               kFormat,
                               1, /* mipLevels */
                               1, /* arrayLayers */
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                               0, /* createFlags */
                               VK_SAMPLE_COUNT_1_BIT,
                               "Image: vulkan import memory");
  EXPECT_TRUE(importedImage.isImported_);
  EXPECT_NE(importedImage.getVkImage(), static_cast<VkImage_T*>(VK_NULL_HANDLE));
}
#endif // IGL_PLATFORM_WINDOWS

TEST_F(VulkanImageTest, CreateInfoDefaultValues) {
  const vulkan::VulkanImageCreateInfo ci;

  EXPECT_EQ(ci.usageFlags, 0u);
  EXPECT_TRUE(ci.isExternallyManaged);
  EXPECT_EQ(ci.extent.width, 0u);
  EXPECT_EQ(ci.extent.height, 0u);
  EXPECT_EQ(ci.extent.depth, 0u);
  EXPECT_EQ(ci.type, VK_IMAGE_TYPE_MAX_ENUM);
  EXPECT_EQ(ci.imageFormat, VK_FORMAT_UNDEFINED);
  EXPECT_EQ(ci.mipLevels, 1u);
  EXPECT_EQ(ci.arrayLayers, 1u);
  EXPECT_EQ(ci.samples, VK_SAMPLE_COUNT_1_BIT);
  EXPECT_FALSE(ci.isImported);
  EXPECT_FALSE(ci.isSrgbMutableFormat);
}

TEST_F(VulkanImageTest, GetImageAspectFlagsColor) {
  vulkan::VulkanImage image(*context_,
                            VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                            VK_IMAGE_TYPE_2D,
                            kFormat,
                            1,
                            1,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            0,
                            VK_SAMPLE_COUNT_1_BIT,
                            "Color Aspect Image");
  ASSERT_TRUE(image.valid());

  EXPECT_EQ(image.getImageAspectFlags(), VK_IMAGE_ASPECT_COLOR_BIT);
  EXPECT_FALSE(image.isDepthFormat_);
  EXPECT_FALSE(image.isStencilFormat_);
  EXPECT_FALSE(image.isDepthOrStencilFormat_);
}

TEST_F(VulkanImageTest, MoveConstruction) {
  vulkan::VulkanImage source(*context_,
                             VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                             VK_IMAGE_TYPE_2D,
                             kFormat,
                             1,
                             1,
                             VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             0,
                             VK_SAMPLE_COUNT_1_BIT,
                             "Move Source Image");
  ASSERT_TRUE(source.valid());
  const VkImage sourceHandle = source.getVkImage();

  vulkan::VulkanImage dest(std::move(source));

  EXPECT_TRUE(dest.valid());
  EXPECT_EQ(dest.getVkImage(), sourceHandle);
  EXPECT_EQ(dest.extent_.width, kWidth);
  EXPECT_EQ(dest.extent_.height, kHeight);
  EXPECT_EQ(dest.imageFormat_, kFormat);

  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(source.valid());
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(source.getVkImage(), VK_NULL_HANDLE);
}

TEST_F(VulkanImageTest, MoveAssignment) {
  vulkan::VulkanImage source(*context_,
                             VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                             VK_IMAGE_TYPE_2D,
                             kFormat,
                             1,
                             1,
                             VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             0,
                             VK_SAMPLE_COUNT_1_BIT,
                             "Move Assign Source");
  ASSERT_TRUE(source.valid());
  const VkImage sourceHandle = source.getVkImage();

  vulkan::VulkanImage dest;
  ASSERT_FALSE(dest.valid());

  dest = std::move(source);

  EXPECT_TRUE(dest.valid());
  EXPECT_EQ(dest.getVkImage(), sourceHandle);
  EXPECT_EQ(dest.extent_.width, kWidth);
  EXPECT_EQ(dest.extent_.height, kHeight);
  EXPECT_EQ(dest.imageFormat_, kFormat);

  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(source.valid());
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(source.getVkImage(), VK_NULL_HANDLE);
}

TEST_F(VulkanImageTest, CreateImageViewBasic) {
  vulkan::VulkanImage image(*context_,
                            VkExtent3D{.width = kWidth, .height = kHeight, .depth = 1},
                            VK_IMAGE_TYPE_2D,
                            kFormat,
                            1,
                            1,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            0,
                            VK_SAMPLE_COUNT_1_BIT,
                            "ImageView Source");
  ASSERT_TRUE(image.valid());

  auto imageView = image.createImageView(
      VK_IMAGE_VIEW_TYPE_2D, kFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, "Test ImageView");

  EXPECT_TRUE(imageView.valid());
  EXPECT_NE(imageView.getVkImageView(), VK_NULL_HANDLE);
  EXPECT_EQ(imageView.getVkImageAspectFlags(), VK_IMAGE_ASPECT_COLOR_BIT);
}

TEST_F(VulkanImageTest, IsStorageImage) {
  auto& ctx = *context_;

  const VkImageUsageFlags storageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  igl::vulkan::VulkanImage image(ctx,
                                 VkExtent3D{.width = 4, .height = 4, .depth = 1},
                                 VK_IMAGE_TYPE_2D,
                                 kFormat,
                                 1,
                                 1,
                                 VK_IMAGE_TILING_OPTIMAL,
                                 storageUsage,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 0,
                                 VK_SAMPLE_COUNT_1_BIT,
                                 "Storage Image");
  ASSERT_TRUE(image.valid());

  EXPECT_TRUE(image.isStorageImage());
  EXPECT_TRUE(image.isSampledImage());
}

TEST_F(VulkanImageTest, CreateImageMultipleMipLevels) {
  auto& ctx = *context_;

  const uint32_t kMipLevels = 4;

  igl::vulkan::VulkanImage image(ctx,
                                 VkExtent3D{.width = 16, .height = 16, .depth = 1},
                                 VK_IMAGE_TYPE_2D,
                                 kFormat,
                                 kMipLevels,
                                 1,
                                 VK_IMAGE_TILING_OPTIMAL,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 0,
                                 VK_SAMPLE_COUNT_1_BIT,
                                 "Multi Mip Image");
  ASSERT_TRUE(image.valid());
  EXPECT_NE(image.getVkImage(), VK_NULL_HANDLE);
  EXPECT_EQ(image.mipLevels_, kMipLevels);
}

TEST_F(VulkanImageTest, CreateImageViewWithMipLevelSubset) {
  auto& ctx = *context_;

  const uint32_t kMipLevels = 4;

  igl::vulkan::VulkanImage image(ctx,
                                 VkExtent3D{.width = 16, .height = 16, .depth = 1},
                                 VK_IMAGE_TYPE_2D,
                                 kFormat,
                                 kMipLevels,
                                 1,
                                 VK_IMAGE_TILING_OPTIMAL,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 0,
                                 VK_SAMPLE_COUNT_1_BIT,
                                 "Mip Subset Source");
  ASSERT_TRUE(image.valid());
  EXPECT_EQ(image.mipLevels_, kMipLevels);

  auto viewAllMips = image.createImageView(VK_IMAGE_VIEW_TYPE_2D,
                                           kFormat,
                                           VK_IMAGE_ASPECT_COLOR_BIT,
                                           0,
                                           kMipLevels,
                                           0,
                                           1,
                                           "View All Mips");
  EXPECT_TRUE(viewAllMips.valid());
  EXPECT_NE(viewAllMips.getVkImageView(), VK_NULL_HANDLE);
  EXPECT_EQ(viewAllMips.getVkImageAspectFlags(), VK_IMAGE_ASPECT_COLOR_BIT);

  auto viewSingleMip = image.createImageView(
      VK_IMAGE_VIEW_TYPE_2D, kFormat, VK_IMAGE_ASPECT_COLOR_BIT, 2, 1, 0, 1, "View Mip 2");
  EXPECT_TRUE(viewSingleMip.valid());
  EXPECT_NE(viewSingleMip.getVkImageView(), VK_NULL_HANDLE);
  EXPECT_EQ(viewSingleMip.getVkImageAspectFlags(), VK_IMAGE_ASPECT_COLOR_BIT);
}

//
// SrgbMutableRequiresMutableBitAndSrgbCounterpart
//
// isSrgbMutableFormat_ is set when a caller explicitly requests sRGB mutable handling for a UNORM
// VkImage with VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT and an sRGB counterpart, in which case creation
// chains a VkImageFormatListCreateInfo constraining views to exactly {UNORM, sRGB}. The flag must
// be set only when all conditions hold; every image the texture layer creates today misses at least
// one, which is what keeps this change inert.
//
TEST_F(VulkanImageTest, SrgbMutableRequiresMutableBitAndSrgbCounterpart) {
  const vulkan::VulkanImage plain =
      makeImage(VK_FORMAT_R8G8B8A8_UNORM, 0, "Image: sRGB mutable flag test (plain)", true);
  ASSERT_TRUE(plain.valid());
  EXPECT_FALSE(plain.isSrgbMutableFormat_);

  const vulkan::VulkanImage mutablePaired = makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                                                      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                                                      "Image: sRGB mutable flag test (paired)",
                                                      true);
  ASSERT_TRUE(mutablePaired.valid());
  EXPECT_TRUE(mutablePaired.isSrgbMutableFormat_);

  // No sRGB counterpart exists for a float format, so there is no second view format to constrain.
  const vulkan::VulkanImage mutableUnpaired = makeImage(VK_FORMAT_R16G16B16A16_SFLOAT,
                                                        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                                                        "Image: sRGB mutable flag test (unpaired)",
                                                        true);
  ASSERT_TRUE(mutableUnpaired.valid());
  EXPECT_FALSE(mutableUnpaired.isSrgbMutableFormat_);
}

//
// SrgbMutableSurvivesMove
//
// VulkanImage's move assignment is hand-written member-by-member (and the move constructor
// delegates to it), so a newly added field is silently dropped unless it is listed there.
// Texture::create() reaches every VkImage through exactly that path -- `VulkanImage image;` then
// `image = ctx.createImage(...)` -- so a lost flag here means every consumer of it silently sees
// a plain UNORM image.
//
TEST_F(VulkanImageTest, SrgbMutableSurvivesMove) {
  const auto makeMutableImage = [this]() {
    return makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                     VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                     "Image: sRGB mutable move test",
                     true);
  };

  // Move assignment onto a default-constructed image: what Texture::create() does.
  vulkan::VulkanImage assigned;
  assigned = makeMutableImage();
  ASSERT_TRUE(assigned.valid());
  EXPECT_TRUE(assigned.isSrgbMutableFormat_);

  // Named source, so this is a real move construction and not an elided prvalue init.
  vulkan::VulkanImage source = makeMutableImage();
  const vulkan::VulkanImage constructed(std::move(source));
  ASSERT_TRUE(constructed.valid());
  EXPECT_TRUE(constructed.isSrgbMutableFormat_);
}

//
// CheckerboardMipChainPreservesLightOnUnormBackedSrgb
//
// The counterpart to CheckerboardMipChainLosesLightOnUnormBackedSrgb, which pinned
// this same chain losing light.
//
// Backing an sRGB texture with a UNORM VkImage -- what storage-capable sRGB textures must do,
// because Vulkan forbids STORAGE usage on sRGB formats -- gives up the correctness the native-sRGB
// backing had for free. vkCmdBlitImage() filters in the image's creation-format numeric space, so
// it averaged the encoded bytes of black and white to 128, a byte carrying 0.216 of the light
// against the source's 0.5. generateMipmapSrgb() downsamples on a native-sRGB scratch, where the
// blit decodes and re-encodes around the filter, so the source's mean light reaches every level
// again -- mid grey, which sRGB encodes as 188.
//
TEST_F(VulkanImageTest, CheckerboardMipChainPreservesLightOnUnormBackedSrgb) {
  if (!supportsSrgbScratchMipmap()) {
    GTEST_SKIP() << "RGBA8 linear blit downscaling is unavailable on this Vulkan device";
  }

  std::array<int, kCheckerMipLevels> greys{};
  ASSERT_NO_FATAL_FAILURE(checkerboardMipChainGreys(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, greys));

  // Consistency: downsampling an already-uniform level is the identity, so whatever grey mip 1
  // settles on must hold all the way down.
  for (uint32_t level = 2; level < kCheckerMipLevels; ++level) {
    EXPECT_NEAR(greys[level], greys[1], 1) << "mip " << level << " drifted from mip 1";
  }

  // Mid grey: the source's mean light, re-encoded.
  EXPECT_NEAR(greys[1], toEncoded(kSourceMeanLight), kTolerance);
  EXPECT_NEAR(toLinear(greys[1]), kSourceMeanLight, 0.01f);

  // ... and measurably clear of the 127.5 encoded-byte average this used to produce.
  EXPECT_GT(greys[1], 127.5f + kTolerance) << "mip 1 still looks like an encoded-byte average";
}

//
// UniformColorSurvivesEveryMipLevel
//
// The point of gamma-correct mip generation: a flat color must still be that color at every
// smaller mip. Downsampling a uniform 8x8 orange runs the full sRGB round-trip -- copy into the
// scratch, blit chain across three levels there, copy each generated level back -- and every level
// must read back the exact source orange.
//
// This does not discriminate the two paths: averaging identical texels is the identity in any
// numeric space, so a plain encoded-space blit preserves a flat color too. What it pins is that
// the round-trip itself is lossless -- a stray sRGB<->linear conversion would shift 255,128,0 to a
// different orange, and a mis-sized or skipped region in the per-level copy-back would leave a
// level unwritten. Neither is reachable by a two-level test.
//
TEST_F(VulkanImageTest, UniformColorSurvivesEveryMipLevel) {
  if (!supportsSrgbScratchMipmap()) {
    GTEST_SKIP() << "RGBA8 linear blit downscaling is unavailable on this Vulkan device";
  }

  constexpr uint32_t kSize = 8; // 8x8 -> 4x4 -> 2x2 -> 1x1
  constexpr uint32_t kNumMipLevels = 4;

  std::array<uint32_t, kSize * kSize> mip0{};
  mip0.fill(orange());

  vulkan::VulkanImage image =
      makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                "Image: sRGB uniform mipmap",
                true /* isSrgbMutableFormat */,
                kSize,
                kNumMipLevels,
                1 /* arrayLayers */,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  ASSERT_TRUE(image.valid());
  ASSERT_TRUE(image.isSrgbMutableFormat_);

  uploadMip0(
      image, TextureType::TwoD, TextureRangeDesc::new2D(0, 0, kSize, kSize), kSize, mip0.data());
  generateMips(image, TextureRangeDesc::new2D(0, 0, kSize, kSize, 0, kNumMipLevels));

  for (uint32_t level = 1; level < kNumMipLevels; ++level) {
    SCOPED_TRACE(testing::Message() << "mip level " << level);
    expectRGB(readTexel(image, level, 0 /* layer */), orangeBytes(), 1);
  }
}

//
// GammaCorrectDownsampleOnMutableFormatImage
//
// vkCmdBlitImage() filters in the VkImage's creation-format numeric space, so blitting a UNORM
// image holding sRGB-encoded texels averages them as if they were linear. generateMipmapSrgb()
// round-trips through a native-sRGB scratch instead.
//
// Half black, half orange. Gamma-correct downsampling re-encodes the linear mean and lands on
// {188, 92, 0}; averaging the encoded bytes lands on {127, 64, 0}. The three channels are pulled
// apart by different amounts, so this also fails on a swizzle or on a conversion applied to only
// part of the texel.
//
TEST_F(VulkanImageTest, GammaCorrectDownsampleOnMutableFormatImage) {
  if (!supportsSrgbScratchMipmap()) {
    GTEST_SKIP() << "RGBA8 linear blit downscaling is unavailable on this Vulkan device";
  }

  const std::array<uint32_t, 4> mip0 = halfBlackMip0(orange());

  uint32_t texel = 0;
  ASSERT_NO_FATAL_FAILURE(downsampleToMip1(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, mip0, texel));

  expectRGB(texel, gammaCorrectHalfOrange(), kTolerance);
}

//
// GammaCorrectDownsampleOnEveryArrayLayer
//
// generateMipmapSrgb() copies generated mips back from the sRGB scratch image across the full
// array-layer range. Use distinct source layers so a missing layer in the copy-in, scratch blit or
// copy-back path is observable.
//
TEST_F(VulkanImageTest, GammaCorrectDownsampleOnEveryArrayLayer) {
  if (!supportsSrgbScratchMipmap()) {
    GTEST_SKIP() << "RGBA8 linear blit downscaling is unavailable on this Vulkan device";
  }

  constexpr uint32_t kSize = 2; // 2x2 -> 1x1
  constexpr uint32_t kNumMipLevels = 2;
  constexpr uint32_t kArrayLayers = 2;

  const std::array<uint32_t, 4> layer0Mip0 = halfBlackMip0(orange());
  std::array<uint32_t, 4> layer1Mip0{};
  layer1Mip0.fill(packRGBA(255, 255, 255, kOpaque));

  vulkan::VulkanImage image =
      makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                "Image: sRGB array mipmap",
                true /* isSrgbMutableFormat */,
                kSize,
                kNumMipLevels,
                kArrayLayers,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  ASSERT_TRUE(image.valid());
  ASSERT_TRUE(image.isSrgbMutableFormat_);

  uploadMip0(image,
             TextureType::TwoDArray,
             TextureRangeDesc::new2DArray(0, 0, kSize, kSize, 0, 1),
             kSize,
             layer0Mip0.data());
  uploadMip0(image,
             TextureType::TwoDArray,
             TextureRangeDesc::new2DArray(0, 0, kSize, kSize, 1, 1),
             kSize,
             layer1Mip0.data());
  generateMips(image,
               TextureRangeDesc::new2DArray(0, 0, kSize, kSize, 0, kArrayLayers, 0, kNumMipLevels));

  expectRGB(readTexel(image, 1 /* level */, 0 /* layer */), gammaCorrectHalfOrange(), kTolerance);
  expectRGB(readTexel(image, 1 /* level */, 1 /* layer */), glm::ivec3(255), 1);
}

//
// PlainImageStillBlitsInEncodedSpace
//
// The counterpart to GammaCorrectDownsampleOnMutableFormatImage, on the same input: without
// MUTABLE_FORMAT_BIT the image takes the unchanged in-place blit path and averages the encoded
// bytes, landing on {127, 64, 0}. Pinning the two paths against each other proves the gamma-correct
// result above comes from generateMipmapSrgb() and not from the driver, and that the existing blit
// path is untouched.
//
TEST_F(VulkanImageTest, PlainImageStillBlitsInEncodedSpace) {
  if (!supportsLinearBlitDownscale()) {
    GTEST_SKIP() << "RGBA8 linear blit downscaling is unavailable on this Vulkan device";
  }

  const std::array<uint32_t, 4> mip0 = halfBlackMip0(orange());

  uint32_t texel = 0;
  ASSERT_NO_FATAL_FAILURE(downsampleToMip1(0 /* createFlags */, mip0, texel));

  expectRGB(texel, orangeBytes() / 2, kTolerance);
  EXPECT_LT(channel(texel, 0), gammaCorrectHalfOrange().r - kTolerance)
      << "blit path unexpectedly produced a gamma-correct result";
}

//
// GenerateMipmapReturnsOkOnBothPaths
//
// The happy path of both generateMipmap() branches reports success.
//
TEST_F(VulkanImageTest, GenerateMipmapReturnsOkOnBothPaths) {
  if (!supportsSrgbScratchMipmap()) {
    GTEST_SKIP() << "RGBA8 linear blit downscaling is unavailable on this Vulkan device";
  }

  constexpr uint32_t kSize = 2; // 2x2 -> 1x1
  constexpr uint32_t kNumMipLevels = 2;
  constexpr VkImageUsageFlags kUsage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  const std::array<uint32_t, 4> mip0 = halfBlackMip0(orange());
  const TextureRangeDesc range = TextureRangeDesc::new2D(0, 0, kSize, kSize, 0, kNumMipLevels);

  for (const VkImageCreateFlags createFlags :
       {VkImageCreateFlags{0}, VkImageCreateFlags{VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT}}) {
    SCOPED_TRACE(testing::Message() << "createFlags " << createFlags);

    vulkan::VulkanImage image = makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                                          createFlags,
                                          "Image: mipmap result",
                                          (createFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0,
                                          kSize,
                                          kNumMipLevels,
                                          1 /* arrayLayers */,
                                          kUsage);
    ASSERT_TRUE(image.valid());

    uploadMip0(
        image, TextureType::TwoD, TextureRangeDesc::new2D(0, 0, kSize, kSize), kSize, mip0.data());

    const Result result = generateMips(image, range);
    EXPECT_TRUE(result.isOk()) << result.message;
  }
}

//
// GenerateMipmapReportsUnsupportedPartialLayerRange
//
// The sRGB scratch path only handles the full layer/face range; a partial range must surface as a
// Result rather than silently leaving the requested mips untouched.
//
TEST_F(VulkanImageTest, GenerateMipmapReportsUnsupportedPartialLayerRange) {
  constexpr uint32_t kSize = 2; // 2x2 -> 1x1
  constexpr uint32_t kNumMipLevels = 2;
  constexpr uint32_t kArrayLayers = 2;

  vulkan::VulkanImage image =
      makeImage(VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                "Image: sRGB partial-range mipmap",
                true /* isSrgbMutableFormat */,
                kSize,
                kNumMipLevels,
                kArrayLayers,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  ASSERT_TRUE(image.valid());
  ASSERT_TRUE(image.isSrgbMutableFormat_);

  const std::array<uint32_t, 4> mip0 = halfBlackMip0(orange());
  uploadMip0(image,
             TextureType::TwoDArray,
             TextureRangeDesc::new2DArray(0, 0, kSize, kSize, 0, 1),
             kSize,
             mip0.data());

  // One layer out of two.
  const Result result =
      generateMips(image, TextureRangeDesc::new2DArray(0, 0, kSize, kSize, 0, 1, 0, kNumMipLevels));

  EXPECT_EQ(result.code, Result::Code::Unsupported);
  EXPECT_FALSE(result.message.empty());
}

//
// Checked image setup (items 3, 23 and 30): the allocation, AHB-import and view steps report their
// failures and clean up what they created. The calls are failed through the FaultingDevice fixture,
// so the production VulkanImage, VulkanImageView, VulkanContext and VMA code run as compiled.
//

class VulkanImageFaultTest : public ::testing::Test {
 protected:
  using Call = util::device::vulkan::FaultingDevice::Call;
  using Counted = util::device::vulkan::FaultingDevice::Counted;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
  }

  static std::unique_ptr<util::device::vulkan::FaultingDevice> makeDevice() {
    return std::make_unique<util::device::vulkan::FaultingDevice>(
        util::device::vulkan::getContextConfig());
  }

  static igl::vulkan::VulkanImage createSmallImage(igl::vulkan::VulkanContext& ctx,
                                                   Result* result) {
    return ctx.createImage(VK_IMAGE_TYPE_2D,
                           VkExtent3D{.width = 64, .height = 64, .depth = 1},
                           kFormat,
                           1,
                           1,
                           VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           0,
                           VK_SAMPLE_COUNT_1_BIT,
                           result,
                           "Image: fault test");
  }

  static void expectNoDependentCall(const util::device::vulkan::FaultingDevice& fd,
                                    uint32_t namesBefore,
                                    uint32_t destroysBefore,
                                    uint32_t freesBefore) {
    if (fd.namingObservable()) {
      EXPECT_EQ(fd.counted(Counted::SetDebugUtilsObjectName), namesBefore);
    }
    EXPECT_EQ(fd.counted(Counted::DestroyImage), destroysBefore);
    EXPECT_EQ(fd.counted(Counted::FreeMemory), freesBefore);
  }
};

TEST_F(VulkanImageFaultTest, FaultingDeviceReachesImageCreationThroughVma) {
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "The allocation tests run on the VMA path";
  }
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();
  // The context's own dummy texture went through VMA and the table's stubs.
  EXPECT_GT(fd->reached(Call::CreateImage), 0u);

  // One injected failure: the first attempt fails inside VMA, the drain succeeds, the retry works.
  fd->resetCounters();
  fd->arm(Call::CreateImage, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  Result result;
  const auto image = createSmallImage(ctx, &result);
  EXPECT_EQ(fd->reached(Call::CreateImage), 2u);
  EXPECT_EQ(fd->failed(Call::CreateImage), 1u);
  ASSERT_TRUE(image.valid()) << result.message.c_str();
  EXPECT_TRUE(result.isOk());
  EXPECT_EQ(image.initResult(), VK_SUCCESS);
  EXPECT_NE(image.getVkImage(), VK_NULL_HANDLE);
}

TEST_F(VulkanImageFaultTest, AnOpenMainRecordingDoesNotStopTheRetry) {
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "The allocation tests run on the VMA path";
  }
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
  fd->arm(Call::CreateImage, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  Result result;
  const auto image = createSmallImage(ctx, &result);

  EXPECT_TRUE(image.valid()) << result.message.c_str();
  EXPECT_TRUE(result.isOk());
  EXPECT_EQ(fd->reached(Call::CreateImage), 2u);
  EXPECT_EQ(fd->failed(Call::CreateImage), 1u);

  cmdQueue->submit(*openBuffer);
  openBuffer->waitUntilCompleted();
}

TEST_F(VulkanImageFaultTest, CreateImageSecondAllocationFailure) {
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "The allocation tests run on the VMA path";
  }
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();

  // Both attempts fail, with different results: the caller sees the retry's.
  fd->resetCounters();
  fd->armSequence(Call::CreateImage, {VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_HOST_MEMORY});
  Result result;
  const auto image = createSmallImage(ctx, &result);

  EXPECT_FALSE(image.valid());
  EXPECT_FALSE(result.isOk());
  EXPECT_EQ(result.message,
            igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_HOST_MEMORY).message);
  EXPECT_NE(result.message,
            igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_DEVICE_MEMORY).message);
  EXPECT_EQ(fd->reached(Call::CreateImage), 2u);
  EXPECT_EQ(fd->failed(Call::CreateImage), 2u);
  expectNoDependentCall(*fd, 0u, 0u, 0u);
}

TEST_F(VulkanImageFaultTest, HostVisibleImageMapFailureDestroysTheImage) {
  if (!IGL_VULKAN_USE_VMA) {
    GTEST_SKIP() << "The allocation tests run on the VMA path";
  }
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();
  const auto createLinear = [&](Result* result) {
    // Large enough (over half of VMA's default block) to get a dedicated memory object of its own:
    // a block shared with an earlier allocation is mapped already and would not reach
    // vkMapMemory().
    return ctx.createImage(VK_IMAGE_TYPE_2D,
                           VkExtent3D{.width = 6144, .height = 6144, .depth = 1},
                           kFormat,
                           1,
                           1,
                           VK_IMAGE_TILING_LINEAR,
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                           0,
                           VK_SAMPLE_COUNT_1_BIT,
                           result,
                           "Image: host-visible fault test");
  };

  // A failed map: the image that was created is destroyed once, and nothing is published.
  Result result;
  fd->resetCounters();
  fd->arm(Call::MapMemory, VK_ERROR_MEMORY_MAP_FAILED);
  const auto image = createLinear(&result);
  if (fd->reached(Call::MapMemory) == 0u) {
    GTEST_SKIP() << "The device has no host-visible linear image: " << result.message.c_str();
  }
  EXPECT_FALSE(image.valid());
  EXPECT_FALSE(result.isOk());
  EXPECT_EQ(result.message, igl::vulkan::getResultFromVkResult(VK_ERROR_MEMORY_MAP_FAILED).message);
  EXPECT_EQ(fd->failed(Call::MapMemory), 1u);
  EXPECT_EQ(fd->counted(Counted::DestroyImage), 1u);
  if (fd->namingObservable()) {
    EXPECT_EQ(fd->counted(Counted::SetDebugUtilsObjectName), 0u);
  }

  // Unfaulted, the same request creates and maps the image, and destroys nothing. It names the
  // image, which is the positive control of the naming assertion above.
  fd->disarmAll();
  fd->resetCounters();
  const auto mapped = createLinear(&result);
  ASSERT_TRUE(mapped.valid()) << result.message.c_str();
  EXPECT_TRUE(result.isOk());
  EXPECT_TRUE(mapped.isMappedPtrAccessible());
  EXPECT_EQ(fd->counted(Counted::DestroyImage), 0u);
  if (fd->namingObservable()) {
    EXPECT_GT(fd->counted(Counted::SetDebugUtilsObjectName), 0u);
  }
}

TEST_F(VulkanImageFaultTest, DisjointImageStepFailuresCleanUpOnlySuccessfulParts) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();

  constexpr VkFormat kPlanarFormat = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
  VkFormatProperties properties{};
  ctx.vf_.vkGetPhysicalDeviceFormatProperties(
      ctx.getVkPhysicalDevice(), kPlanarFormat, &properties);
  if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DISJOINT_BIT) == 0) {
    GTEST_SKIP() << "The device has no disjoint 3-plane 4:2:0 images";
  }
  const auto createPlanar = [&](Result* result) {
    return ctx.createImage(VK_IMAGE_TYPE_2D,
                           VkExtent3D{.width = 64, .height = 64, .depth = 1},
                           kPlanarFormat,
                           1,
                           1,
                           VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           VK_IMAGE_CREATE_DISJOINT_BIT,
                           VK_SAMPLE_COUNT_1_BIT,
                           result,
                           "Image: disjoint fault test");
  };

  // The unfaulted creation works and takes the raw path (one memory object per plane).
  {
    Result result;
    fd->resetCounters();
    const auto image = createPlanar(&result);
    ASSERT_TRUE(image.valid()) << result.message.c_str();
    EXPECT_EQ(fd->reached(Call::AllocateMemory), 3u);
    EXPECT_EQ(fd->counted(Counted::FreeMemory), 0u);
  }

  // A failed vkCreateImage: nothing was created, nothing is destroyed.
  {
    Result result;
    fd->resetCounters();
    fd->arm(Call::CreateImage, VK_ERROR_OUT_OF_HOST_MEMORY);
    const auto image = createPlanar(&result);
    EXPECT_FALSE(image.valid());
    EXPECT_EQ(result.message,
              igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_HOST_MEMORY).message);
    EXPECT_EQ(fd->counted(Counted::DestroyImage), 0u);
    EXPECT_EQ(fd->counted(Counted::FreeMemory), 0u);
    fd->disarmAll();
  }

  // A failed allocation of plane k: the image and the k planes allocated before it are destroyed,
  // each exactly once, and no later plane is attempted.
  for (uint32_t plane = 0; plane < 3; ++plane) {
    Result result;
    fd->resetCounters();
    fd->arm(Call::AllocateMemory, VK_ERROR_OUT_OF_DEVICE_MEMORY, plane);
    const auto image = createPlanar(&result);
    EXPECT_FALSE(image.valid()) << "plane " << plane;
    EXPECT_EQ(result.message,
              igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_DEVICE_MEMORY).message);
    EXPECT_EQ(fd->reached(Call::AllocateMemory), plane + 1) << "plane " << plane;
    EXPECT_EQ(fd->counted(Counted::DestroyImage), 1u) << "plane " << plane;
    EXPECT_EQ(fd->counted(Counted::FreeMemory), plane) << "plane " << plane;
    fd->disarmAll();
  }

  // A failed bind: the image and all three planes are destroyed.
  {
    Result result;
    fd->resetCounters();
    fd->arm(Call::BindImageMemory2, VK_ERROR_UNKNOWN);
    const auto image = createPlanar(&result);
    EXPECT_FALSE(image.valid());
    EXPECT_EQ(result.message, igl::vulkan::getResultFromVkResult(VK_ERROR_UNKNOWN).message);
    EXPECT_EQ(fd->counted(Counted::DestroyImage), 1u);
    EXPECT_EQ(fd->counted(Counted::FreeMemory), 3u);
    fd->disarmAll();
  }
}

TEST_F(VulkanImageFaultTest, CreateImageViewOfAnInvalidImageIsRefused) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();

  // An image whose construction failed (built directly, as `createImage()` returns an empty one).
  fd->resetCounters();
  fd->arm(Call::CreateImage, VK_ERROR_OUT_OF_HOST_MEMORY, 0, 2);
  const igl::vulkan::VulkanImage image(ctx,
                                       VkExtent3D{.width = 64, .height = 64, .depth = 1},
                                       VK_IMAGE_TYPE_2D,
                                       kFormat,
                                       1,
                                       1,
                                       VK_IMAGE_TILING_OPTIMAL,
                                       VK_IMAGE_USAGE_SAMPLED_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                       0,
                                       VK_SAMPLE_COUNT_1_BIT,
                                       "Image: invalid");
  ASSERT_FALSE(image.valid());
  fd->resetCounters();

  // The view creations are failed too, so a call that reached the driver would be counted: none
  // does, and each returns an invalid view.
  fd->arm(Call::CreateImageView, VK_ERROR_OUT_OF_HOST_MEMORY, 0, 3);
  const igl::vulkan::VulkanImageView plain = image.createImageView(
      VK_IMAGE_VIEW_TYPE_2D, kFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, "View: refused");
  igl::vulkan::VulkanImageViewCreateInfo info;
  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.format = kFormat;
  const igl::vulkan::VulkanImageView fromInfo = image.createImageView(info, "View: refused");
  EXPECT_FALSE(plain.valid());
  EXPECT_FALSE(fromInfo.valid());
  EXPECT_EQ(fd->reached(Call::CreateImageView), 0u);
}

TEST_F(VulkanImageFaultTest, DummyTextureViewFailureDestroysTheDummyImageAtOnce) {
  // The first image view of the context (the dummy texture's) fails: the dummy image, which
  // nothing has used, is destroyed by the failing init itself, before the context is torn down.
  const util::device::vulkan::FaultingDevice fd(
      util::device::vulkan::getContextConfig(), [](util::device::vulkan::FaultingDevice& self) {
        self.arm(Call::CreateImageView, VK_ERROR_OUT_OF_HOST_MEMORY, /*skip=*/0, UINT32_MAX);
      });

  EXPECT_EQ(fd.device(), nullptr);
  EXPECT_FALSE(fd.creationResult().isOk());
  EXPECT_GE(fd.failed(Call::CreateImageView), 1u);
  ASSERT_TRUE(fd.teardownObserved());
  EXPECT_EQ(fd.countedBeforeTeardown(Counted::DestroyImage), 1u);
}

TEST_F(VulkanImageFaultTest, CreateTextureRefusesAnInvalidView) {
  const auto fd = makeDevice();
  ASSERT_NE(fd->device(), nullptr) << fd->creationResult().message.c_str();
  auto& ctx = fd->context();
  const auto fixtureCounts = [&]() { return fd->counted(Counted::DestroyImage); };

  Result result;
  auto image = createSmallImage(ctx, &result);
  ASSERT_TRUE(image.valid()) << result.message.c_str();

  // A view whose creation failed is invalid: no texture is made, and the caller's image stays
  // whole until its own destruction.
  fd->arm(Call::CreateImageView, VK_ERROR_OUT_OF_HOST_MEMORY);
  igl::vulkan::VulkanImageView failedView = image.createImageView(
      VK_IMAGE_VIEW_TYPE_2D, kFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, "View: refused");
  fd->disarmAll();
  ASSERT_FALSE(failedView.valid());
  EXPECT_EQ(failedView.initResult(), VK_ERROR_OUT_OF_HOST_MEMORY);
  fd->resetCounters();
  EXPECT_EQ(ctx.createTexture(std::move(image), std::move(failedView), "refused"), nullptr);
  // The refused image was created for this texture only: it is destroyed at once, not through a
  // deferred task.
  EXPECT_EQ(fixtureCounts(), 1u);

  // An invalid image with a valid view is refused as well.
  auto validImage = createSmallImage(ctx, &result);
  ASSERT_TRUE(validImage.valid());
  igl::vulkan::VulkanImageView view = validImage.createImageView(
      VK_IMAGE_VIEW_TYPE_2D, kFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, "View: valid");
  ASSERT_TRUE(view.valid());
  EXPECT_EQ(ctx.createTexture(igl::vulkan::VulkanImage(), std::move(view), "refused"), nullptr);

  // A valid pair still makes a texture.
  auto goodImage = createSmallImage(ctx, &result);
  ASSERT_TRUE(goodImage.valid());
  igl::vulkan::VulkanImageView goodView = goodImage.createImageView(
      VK_IMAGE_VIEW_TYPE_2D, kFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, "View: good");
  EXPECT_NE(ctx.createTexture(std::move(goodImage), std::move(goodView), "accepted"), nullptr);
}

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

// The AHardwareBuffer import steps (Android lane, run on a device).
class VulkanImageAhbFaultTest : public VulkanImageFaultTest {
 protected:
  void SetUp() override {
    VulkanImageFaultTest::SetUp();
    fd_ = makeDevice();
    ASSERT_NE(fd_->device(), nullptr) << fd_->creationResult().message.c_str();
    const AHardwareBuffer_Desc description{
        .width = 64,
        .height = 64,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
        .usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT,
    };
    ASSERT_EQ(AHardwareBuffer_allocate(&description, &buffer_), 0);
  }

  void TearDown() override {
    fd_.reset();
    if (buffer_ != nullptr) {
      AHardwareBuffer_release(buffer_);
    }
  }

  std::unique_ptr<igl::vulkan::VulkanImage> importBuffer(Result* result) {
    return fd_->context().createImageFromAndroidHardwareBuffer(
        buffer_,
        0,
        VK_IMAGE_TYPE_2D,
        VkExtent3D{.width = 64, .height = 64, .depth = 1},
        kFormat,
        1,
        1,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT,
        0,
        VK_SAMPLE_COUNT_1_BIT,
        result,
        "Image: AHB fault test");
  }

  std::unique_ptr<util::device::vulkan::FaultingDevice> fd_;
  AHardwareBuffer* buffer_ = nullptr;
};

TEST_F(VulkanImageAhbFaultTest, AhbImportFailsAtImageCreate) {
  fd_->resetCounters();
  fd_->arm(Call::CreateImage, VK_ERROR_OUT_OF_HOST_MEMORY);
  Result result;
  EXPECT_EQ(importBuffer(&result), nullptr);
  EXPECT_EQ(result.message,
            igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_HOST_MEMORY).message);
  EXPECT_EQ(fd_->reached(Call::GetAndroidHardwareBufferProperties), 0u);
  EXPECT_EQ(fd_->reached(Call::AllocateMemory), 0u);
  EXPECT_EQ(fd_->counted(Counted::DestroyImage), 0u);
  EXPECT_EQ(fd_->counted(Counted::FreeMemory), 0u);
}

TEST_F(VulkanImageAhbFaultTest, AhbImportFailsAtPropertyQuery) {
  fd_->resetCounters();
  fd_->arm(Call::GetAndroidHardwareBufferProperties, VK_ERROR_INVALID_EXTERNAL_HANDLE);
  Result result;
  EXPECT_EQ(importBuffer(&result), nullptr);
  EXPECT_EQ(result.message,
            igl::vulkan::getResultFromVkResult(VK_ERROR_INVALID_EXTERNAL_HANDLE).message);
  EXPECT_EQ(fd_->reached(Call::AllocateMemory), 0u);
  EXPECT_EQ(fd_->counted(Counted::DestroyImage), 1u); // the image created before it
  EXPECT_EQ(fd_->counted(Counted::FreeMemory), 0u);
}

TEST_F(VulkanImageAhbFaultTest, AhbImportFailsAtAllocate) {
  fd_->resetCounters();
  fd_->arm(Call::AllocateMemory, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  Result result;
  EXPECT_EQ(importBuffer(&result), nullptr);
  EXPECT_EQ(result.message,
            igl::vulkan::getResultFromVkResult(VK_ERROR_OUT_OF_DEVICE_MEMORY).message);
  EXPECT_EQ(fd_->reached(Call::BindImageMemory), 0u);
  EXPECT_EQ(fd_->counted(Counted::DestroyImage), 1u);
  EXPECT_EQ(fd_->counted(Counted::FreeMemory), 0u);
}

TEST_F(VulkanImageAhbFaultTest, AhbImportFailsAtBind) {
  fd_->resetCounters();
  fd_->arm(Call::BindImageMemory, VK_ERROR_UNKNOWN);
  Result result;
  EXPECT_EQ(importBuffer(&result), nullptr);
  EXPECT_EQ(result.message, igl::vulkan::getResultFromVkResult(VK_ERROR_UNKNOWN).message);
  EXPECT_EQ(fd_->counted(Counted::DestroyImage), 1u);
  EXPECT_EQ(fd_->counted(Counted::FreeMemory), 1u); // the imported memory
}

TEST_F(VulkanImageAhbFaultTest, AhbImportFailsAtViewCreate) {
  Result result;
  const auto image = importBuffer(&result);
  ASSERT_NE(image, nullptr) << result.message.c_str();
  ASSERT_TRUE(image->valid());

  fd_->resetCounters();
  fd_->arm(Call::CreateImageView, VK_ERROR_OUT_OF_HOST_MEMORY);
  igl::vulkan::VulkanImageViewCreateInfo info;
  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.format = kFormat;
  const igl::vulkan::VulkanImageView view = image->createImageView(info, "View: AHB fault test");
  EXPECT_FALSE(view.valid());
  EXPECT_EQ(view.initResult(), VK_ERROR_OUT_OF_HOST_MEMORY);
  // The view was not named, and the image is untouched.
  if (fd_->namingObservable()) {
    EXPECT_EQ(fd_->counted(Counted::SetDebugUtilsObjectName), 0u);
  }
  EXPECT_TRUE(image->valid());
}

TEST_F(VulkanImageAhbFaultTest, AhbImportRetriesCleanlyAfterAPartialFailure) {
  fd_->arm(Call::BindImageMemory, VK_ERROR_UNKNOWN);
  Result result;
  EXPECT_EQ(importBuffer(&result), nullptr);
  fd_->disarmAll();

  fd_->resetCounters();
  const auto image = importBuffer(&result);
  ASSERT_NE(image, nullptr) << result.message.c_str();
  EXPECT_TRUE(result.isOk());
  EXPECT_TRUE(image->valid());
  EXPECT_EQ(image->initResult(), VK_SUCCESS);
  EXPECT_NE(image->getVkImage(), VK_NULL_HANDLE);
  // The retry created exactly what a first success creates, and destroyed nothing.
  EXPECT_EQ(fd_->counted(Counted::DestroyImage), 0u);
  EXPECT_EQ(fd_->counted(Counted::FreeMemory), 0u);
}

#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX
