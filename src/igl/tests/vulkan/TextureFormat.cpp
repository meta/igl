/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/util/TextureFormat.h>

#include <igl/vulkan/Common.h>
#include <igl/vulkan/YcbcrFormat.h>

namespace igl::vulkan::tests {
TEST(TextureFormatUtilTest, invertRedAndBlue) {
  ASSERT_EQ(VK_FORMAT_B8G8R8A8_UNORM, invertRedAndBlue(VK_FORMAT_R8G8B8A8_UNORM));
  ASSERT_EQ(VK_FORMAT_R8G8B8A8_UNORM, invertRedAndBlue(VK_FORMAT_B8G8R8A8_UNORM));
  ASSERT_EQ(VK_FORMAT_R8G8B8A8_SRGB, invertRedAndBlue(VK_FORMAT_B8G8R8A8_SRGB));
  ASSERT_EQ(VK_FORMAT_B8G8R8A8_SRGB, invertRedAndBlue(VK_FORMAT_R8G8B8A8_SRGB));
  ASSERT_EQ(VK_FORMAT_A2R10G10B10_UNORM_PACK32,
            invertRedAndBlue(VK_FORMAT_A2B10G10R10_UNORM_PACK32));
  ASSERT_EQ(VK_FORMAT_A2B10G10R10_UNORM_PACK32,
            invertRedAndBlue(VK_FORMAT_A2R10G10B10_UNORM_PACK32));
}

TEST(TextureFormatUtilTest, isTextureFormatRGB) {
  ASSERT_TRUE(isTextureFormatRGB(VK_FORMAT_R8G8B8A8_UNORM));
  ASSERT_TRUE(isTextureFormatRGB(VK_FORMAT_R8G8B8A8_SRGB));
  ASSERT_TRUE(isTextureFormatRGB(VK_FORMAT_A2B10G10R10_UNORM_PACK32));
  ASSERT_FALSE(isTextureFormatRGB(VK_FORMAT_B8G8R8A8_UNORM));
  ASSERT_FALSE(isTextureFormatRGB(VK_FORMAT_B8G8R8A8_SRGB));
  ASSERT_FALSE(isTextureFormatRGB(VK_FORMAT_A2R10G10B10_UNORM_PACK32));
}

TEST(TextureFormatUtilTest, isTextureFormatBGR) {
  ASSERT_FALSE(isTextureFormatBGR(VK_FORMAT_R8G8B8A8_UNORM));
  ASSERT_FALSE(isTextureFormatBGR(VK_FORMAT_R8G8B8A8_SRGB));
  ASSERT_FALSE(isTextureFormatBGR(VK_FORMAT_A2B10G10R10_UNORM_PACK32));
  ASSERT_TRUE(isTextureFormatBGR(VK_FORMAT_B8G8R8A8_UNORM));
  ASSERT_TRUE(isTextureFormatBGR(VK_FORMAT_B8G8R8A8_SRGB));
  ASSERT_TRUE(isTextureFormatBGR(VK_FORMAT_A2R10G10B10_UNORM_PACK32));
}

TEST(TextureFormatUtilTest, B10G11R11UFloatRoundTrip) {
  constexpr TextureFormat kIglFormat = igl::TextureFormat::B10G11R11_UFloat;
  constexpr VkFormat kVkFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;

  EXPECT_EQ(textureFormatToVkFormat(kIglFormat), kVkFormat);
  EXPECT_EQ(vkFormatToTextureFormat(kVkFormat), kIglFormat);
  EXPECT_EQ(vkFormatToTextureFormat(textureFormatToVkFormat(kIglFormat)), kIglFormat);
}

TEST(TextureFormatUtilTest, textureFormatToVkFormat) {
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::Invalid), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::A_UNorm8), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::L_UNorm8), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_UNorm8), VK_FORMAT_R8_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_UNorm16), VK_FORMAT_R16_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_F16), VK_FORMAT_R16_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_UInt16), VK_FORMAT_R16_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::B5G5R5A1_UNorm),
            VK_FORMAT_B5G5R5A1_UNORM_PACK16);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::B5G6R5_UNorm),
            VK_FORMAT_B5G6R5_UNORM_PACK16);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::ABGR_UNorm4),
            VK_FORMAT_B4G4R4A4_UNORM_PACK16);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::LA_UNorm8), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RG_UNorm8), VK_FORMAT_R8G8_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RG_UNorm16), VK_FORMAT_R16G16_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R4G2B2_UNorm_Apple), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R4G2B2_UNorm_Rev_Apple),
            VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R5G5B5A1_UNorm),
            VK_FORMAT_R5G5B5A1_UNORM_PACK16);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::BGRA_UNorm8), VK_FORMAT_B8G8R8A8_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::BGRA_UNorm8_Rev), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_UNorm8), VK_FORMAT_R8G8B8A8_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBX_UNorm8), VK_FORMAT_R8G8B8A8_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_SRGB), VK_FORMAT_R8G8B8A8_SRGB);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::BGRA_SRGB), VK_FORMAT_B8G8R8A8_SRGB);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RG_F16), VK_FORMAT_R16G16_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RG_UInt16), VK_FORMAT_R16G16_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB10_A2_UNorm_Rev),
            VK_FORMAT_A2B10G10R10_UNORM_PACK32);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB10_A2_Uint_Rev),
            VK_FORMAT_A2B10G10R10_UINT_PACK32);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::BGR10_A2_Unorm),
            VK_FORMAT_A2R10G10B10_UNORM_PACK32);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_F32), VK_FORMAT_R32_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_UInt32), VK_FORMAT_R32_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RG_F32), VK_FORMAT_R32G32_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB_F16), VK_FORMAT_R16G16B16_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_F16), VK_FORMAT_R16G16B16A16_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB_F32), VK_FORMAT_R32G32B32_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_UInt32), VK_FORMAT_R32G32B32A32_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_F32), VK_FORMAT_R32G32B32A32_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_4x4),
            VK_FORMAT_ASTC_4x4_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_4x4),
            VK_FORMAT_ASTC_4x4_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_5x4),
            VK_FORMAT_ASTC_5x4_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_5x4),
            VK_FORMAT_ASTC_5x4_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_5x5),
            VK_FORMAT_ASTC_5x5_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_5x5),
            VK_FORMAT_ASTC_5x5_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_6x5),
            VK_FORMAT_ASTC_6x5_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_6x5),
            VK_FORMAT_ASTC_6x5_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_6x6),
            VK_FORMAT_ASTC_6x6_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_6x6),
            VK_FORMAT_ASTC_6x6_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_8x5),
            VK_FORMAT_ASTC_8x5_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_8x5),
            VK_FORMAT_ASTC_8x5_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_8x6),
            VK_FORMAT_ASTC_8x6_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_8x6),
            VK_FORMAT_ASTC_8x6_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_8x8),
            VK_FORMAT_ASTC_8x8_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_8x8),
            VK_FORMAT_ASTC_8x8_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_10x5),
            VK_FORMAT_ASTC_10x5_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_10x5),
            VK_FORMAT_ASTC_10x5_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_10x6),
            VK_FORMAT_ASTC_10x6_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_10x6),
            VK_FORMAT_ASTC_10x6_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_10x8),
            VK_FORMAT_ASTC_10x8_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_10x8),
            VK_FORMAT_ASTC_10x8_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_10x10),
            VK_FORMAT_ASTC_10x10_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_10x10),
            VK_FORMAT_ASTC_10x10_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_12x10),
            VK_FORMAT_ASTC_12x10_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_12x10),
            VK_FORMAT_ASTC_12x10_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_ASTC_12x12),
            VK_FORMAT_ASTC_12x12_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_ASTC_12x12),
            VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_PVRTC_2BPPV1),
            VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB_PVRTC_2BPPV1),
            VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_PVRTC_4BPPV1),
            VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB_PVRTC_4BPPV1),
            VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB8_ETC1), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB8_ETC2),
            VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_ETC2),
            VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGB8_Punchthrough_A1_ETC2),
            VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_Punchthrough_A1_ETC2),
            VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA8_EAC_ETC2), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::SRGB8_A8_EAC_ETC2), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RG_EAC_UNorm),
            VK_FORMAT_EAC_R11G11_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RG_EAC_SNorm),
            VK_FORMAT_EAC_R11G11_SNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_EAC_UNorm),
            VK_FORMAT_EAC_R11_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::R_EAC_SNorm),
            VK_FORMAT_EAC_R11_SNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::Invalid), VK_FORMAT_UNDEFINED);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_BC7_UNORM_4x4),
            VK_FORMAT_BC7_UNORM_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::RGBA_BC7_SRGB_4x4),
            VK_FORMAT_BC7_SRGB_BLOCK);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::Z_UNorm16), VK_FORMAT_D16_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::Z_UNorm24), VK_FORMAT_D24_UNORM_S8_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::Z_UNorm32), VK_FORMAT_D32_SFLOAT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::S8_UInt_Z24_UNorm),
            VK_FORMAT_D24_UNORM_S8_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::S8_UInt_Z32_UNorm),
            VK_FORMAT_D32_SFLOAT_S8_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::S_UInt8), VK_FORMAT_S8_UINT);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::YUV_NV12),
            VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);
  ASSERT_EQ(textureFormatToVkFormat(igl::TextureFormat::YUV_420p),
            VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM);
}

TEST(TextureFormatUtilTest, vkTextureFormatToTextureFormat) {
  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_UNDEFINED), igl::TextureFormat::Invalid);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R8_UNORM), igl::TextureFormat::R_UNorm8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16_UNORM),
            igl::TextureFormat::R_UNorm16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16_SFLOAT), igl::TextureFormat::R_F16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16_UINT), igl::TextureFormat::R_UInt16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_B5G5R5A1_UNORM_PACK16),
            igl::TextureFormat::B5G5R5A1_UNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_B5G6R5_UNORM_PACK16),
            igl::TextureFormat::B5G6R5_UNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_B4G4R4A4_UNORM_PACK16),
            igl::TextureFormat::ABGR_UNorm4);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R8G8_UNORM),
            igl::TextureFormat::RG_UNorm8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R5G5B5A1_UNORM_PACK16),
            igl::TextureFormat::R5G5B5A1_UNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_B8G8R8A8_UNORM),
            igl::TextureFormat::BGRA_UNorm8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R8G8B8A8_UNORM),
            igl::TextureFormat::RGBA_UNorm8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R8G8B8A8_SRGB),
            igl::TextureFormat::RGBA_SRGB);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_B8G8R8A8_SRGB),
            igl::TextureFormat::BGRA_SRGB);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16G16_UNORM),
            igl::TextureFormat::RG_UNorm16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16G16_SFLOAT),
            igl::TextureFormat::RG_F16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16G16_UINT),
            igl::TextureFormat::RG_UInt16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_A2B10G10R10_UNORM_PACK32),
            igl::TextureFormat::RGB10_A2_UNorm_Rev);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_A2B10G10R10_UINT_PACK32),
            igl::TextureFormat::RGB10_A2_Uint_Rev);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_A2R10G10B10_UNORM_PACK32),
            igl::TextureFormat::BGR10_A2_Unorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R32_SFLOAT), igl::TextureFormat::R_F32);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R32_UINT), igl::TextureFormat::R_UInt32);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R32G32_SFLOAT),
            igl::TextureFormat::RG_F32);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16G16B16_SFLOAT),
            igl::TextureFormat::RGB_F16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R16G16B16A16_SFLOAT),
            igl::TextureFormat::RGBA_F16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R32G32B32_SFLOAT),
            igl::TextureFormat::RGB_F32);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R32G32B32A32_UINT),
            igl::TextureFormat::RGBA_UInt32);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R32G32B32A32_SFLOAT),
            igl::TextureFormat::RGBA_F32);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_4x4_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_4x4);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_4x4_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_4x4);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_5x4_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_5x4);
  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_5x4_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_5x4);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_5x5_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_5x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_5x5_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_5x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_6x5_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_6x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_6x5_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_6x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_6x6_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_6x6);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_6x6_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_6x6);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_8x5_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_8x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_8x5_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_8x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_8x6_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_8x6);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_8x6_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_8x6);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_8x8_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_8x8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_8x8_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_8x8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x5_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_10x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x5_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_10x5);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x6_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_10x6);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x6_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_10x6);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x8_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_10x8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x8_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_10x8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x10_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_10x10);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_10x10_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_10x10);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_12x10_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_12x10);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_12x10_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_12x10);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_12x12_UNORM_BLOCK),
            igl::TextureFormat::RGBA_ASTC_12x12);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ASTC_12x12_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_A8_ASTC_12x12);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG),
            igl::TextureFormat::RGBA_PVRTC_2BPPV1);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG),
            igl::TextureFormat::RGBA_PVRTC_4BPPV1);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK),
            igl::TextureFormat::RGB8_ETC2);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_ETC2);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK),
            igl::TextureFormat::RGB8_Punchthrough_A1_ETC2);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK),
            igl::TextureFormat::SRGB8_Punchthrough_A1_ETC2);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_EAC_R11G11_UNORM_BLOCK),
            igl::TextureFormat::RG_EAC_UNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_EAC_R11G11_SNORM_BLOCK),
            igl::TextureFormat::RG_EAC_SNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_EAC_R11_UNORM_BLOCK),
            igl::TextureFormat::R_EAC_UNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_EAC_R11_SNORM_BLOCK),
            igl::TextureFormat::R_EAC_SNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_D16_UNORM),
            igl::TextureFormat::Z_UNorm16);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_BC7_UNORM_BLOCK),
            igl::TextureFormat::RGBA_BC7_UNORM_4x4);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_BC7_SRGB_BLOCK),
            igl::TextureFormat::RGBA_BC7_SRGB_4x4);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_X8_D24_UNORM_PACK32),
            igl::TextureFormat::Z_UNorm24);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_D24_UNORM_S8_UINT),
            igl::TextureFormat::S8_UInt_Z24_UNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_S8_UINT), igl::TextureFormat::S_UInt8);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_D32_SFLOAT_S8_UINT),
            igl::TextureFormat::S8_UInt_Z32_UNorm);

  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_D32_SFLOAT),
            igl::TextureFormat::Z_UNorm32);
  ASSERT_EQ(util::vkTextureFormatToTextureFormat(VK_FORMAT_R8G8B8A8_UNORM),
            igl::TextureFormat::RGBA_UNorm8);
}

TEST(TextureFormatUtilTest, tryVkTextureFormatToTextureFormatReportsUnmappedFormats) {
  TextureFormat format = igl::TextureFormat::RGBA_UNorm8;
  ASSERT_TRUE(util::tryVkTextureFormatToTextureFormat(VK_FORMAT_R8_UNORM, format));
  ASSERT_EQ(format, igl::TextureFormat::R_UNorm8);
  ASSERT_TRUE(util::tryVkTextureFormatToTextureFormat(VK_FORMAT_UNDEFINED, format));
  ASSERT_EQ(format, igl::TextureFormat::Invalid);

  // A format with no TextureFormat is reported, not aborted on, and the output stays as it was.
  format = igl::TextureFormat::RGBA_UNorm8;
  ASSERT_FALSE(util::tryVkTextureFormatToTextureFormat(
      VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16, format));
  ASSERT_EQ(format, igl::TextureFormat::RGBA_UNorm8);
  ASSERT_FALSE(util::tryVkTextureFormatToTextureFormat(0x7fffffff, format));
  ASSERT_EQ(format, igl::TextureFormat::RGBA_UNorm8);
}

TEST(TextureFormatUtilTest, vkFormatRequiresYcbcrConversion) {
  // Every packed 4:2:2 and multi-planar format of Vulkan 1.1.
  constexpr VkFormat kNeedsConversion[] = {
      VK_FORMAT_G8B8G8R8_422_UNORM,
      VK_FORMAT_B8G8R8G8_422_UNORM,
      VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
      VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
      VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
      VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
      VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
      VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
      VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
      VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
      VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
      VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
      VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
      VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
      VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
      VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
      VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
      VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
      VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
      VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
      VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
      VK_FORMAT_G16B16G16R16_422_UNORM,
      VK_FORMAT_B16G16R16G16_422_UNORM,
      VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
      VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
      VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
      VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
      VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
  };
  for (const VkFormat format : kNeedsConversion) {
    ASSERT_TRUE(vkFormatRequiresYcbcrConversion(format)) << static_cast<int>(format);
  }

  // The R10X6 and R12X4 formats that sit between them in the enum have no chroma and need none.
  constexpr VkFormat kNoConversion[] = {
      VK_FORMAT_R10X6_UNORM_PACK16,
      VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
      VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
      VK_FORMAT_R12X4_UNORM_PACK16,
      VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
      VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
      VK_FORMAT_UNDEFINED,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_R16G16B16A16_SFLOAT,
      VK_FORMAT_D32_SFLOAT,
  };
  for (const VkFormat format : kNoConversion) {
    ASSERT_FALSE(vkFormatRequiresYcbcrConversion(format)) << static_cast<int>(format);
  }
}

TEST(TextureFormatUtilTest, formatFeaturesAllowLinearChromaFilter) {
  constexpr VkFormatFeatureFlags kLinear =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT;
  ASSERT_TRUE(formatFeaturesAllowLinearChromaFilter(kLinear));
  ASSERT_TRUE(formatFeaturesAllowLinearChromaFilter(kLinear | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
  ASSERT_FALSE(formatFeaturesAllowLinearChromaFilter(0));

  // The features a PowerVR driver lists for the camera's external format: sampled image with a
  // linear filter, the transfer bits, the midpoint and cosited chroma samples, separate, explicit
  // and forceable-explicit chroma reconstruction and disjoint planes, but not the linear chroma
  // filter.
  constexpr VkFormatFeatureFlags kPowerVrCameraFormat =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
      VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT | VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT |
      VK_FORMAT_FEATURE_DISJOINT_BIT;
  ASSERT_FALSE(formatFeaturesAllowLinearChromaFilter(kPowerVrCameraFormat));
  ASSERT_TRUE(formatFeaturesAllowLinearChromaFilter(kPowerVrCameraFormat | kLinear));
}

} // namespace igl::vulkan::tests
