/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/util/TextureFormat.h>

namespace igl::vulkan::util {

// @fb-only
// NOLINTBEGIN(readability-identifier-naming)
// Vulkan Defines
#define VK_FORMAT_UNDEFINED 0
#define VK_FORMAT_R5G6B5_UNORM_PACK16 4
#define VK_FORMAT_R8_UNORM 9
#define VK_FORMAT_R16_UNORM 70
#define VK_FORMAT_R16_SFLOAT 76
#define VK_FORMAT_R16_UINT 74
#define VK_FORMAT_B5G5R5A1_UNORM_PACK16 7
#define VK_FORMAT_B5G6R5_UNORM_PACK16 5
#define VK_FORMAT_B4G4R4A4_UNORM_PACK16 3
#define VK_FORMAT_R8G8_UNORM 16
#define VK_FORMAT_R5G5B5A1_UNORM_PACK16 6
#define VK_FORMAT_B8G8R8A8_UNORM 44
#define VK_FORMAT_R8G8B8A8_UNORM 37
#define VK_FORMAT_R8G8B8A8_SRGB 43
#define VK_FORMAT_B8G8R8A8_SRGB 50
#define VK_FORMAT_R16G16_UNORM 77
#define VK_FORMAT_R16G16_SFLOAT 83
#define VK_FORMAT_R16G16_UINT 81
#define VK_FORMAT_R16G16B16A16_UNORM 91
#define VK_FORMAT_A2R10G10B10_UNORM_PACK32 58
#define VK_FORMAT_A2R10G10B10_UINT_PACK32 62
#define VK_FORMAT_A2B10G10R10_UNORM_PACK32 64
#define VK_FORMAT_A2B10G10R10_UINT_PACK32 68
#define VK_FORMAT_B10G11R11_UFLOAT_PACK32 122
#define VK_FORMAT_R32_UINT 98
#define VK_FORMAT_R32_SFLOAT 100
#define VK_FORMAT_R32G32_SFLOAT 103
#define VK_FORMAT_R16G16B16_SFLOAT 90
#define VK_FORMAT_R16G16B16A16_SFLOAT 97
#define VK_FORMAT_R32G32B32_SFLOAT 106
#define VK_FORMAT_R32G32B32A32_UINT 107
#define VK_FORMAT_R32G32B32A32_SFLOAT 109
#define VK_FORMAT_ASTC_4x4_UNORM_BLOCK 157
#define VK_FORMAT_ASTC_4x4_SRGB_BLOCK 158
#define VK_FORMAT_ASTC_5x4_UNORM_BLOCK 159
#define VK_FORMAT_ASTC_5x4_SRGB_BLOCK 160
#define VK_FORMAT_ASTC_5x5_UNORM_BLOCK 161
#define VK_FORMAT_ASTC_5x5_SRGB_BLOCK 162
#define VK_FORMAT_ASTC_6x5_UNORM_BLOCK 163
#define VK_FORMAT_ASTC_6x5_SRGB_BLOCK 164
#define VK_FORMAT_ASTC_6x6_UNORM_BLOCK 165
#define VK_FORMAT_ASTC_6x6_SRGB_BLOCK 166
#define VK_FORMAT_ASTC_8x5_UNORM_BLOCK 167
#define VK_FORMAT_ASTC_8x5_SRGB_BLOCK 168
#define VK_FORMAT_ASTC_8x6_UNORM_BLOCK 169
#define VK_FORMAT_ASTC_8x6_SRGB_BLOCK 170
#define VK_FORMAT_ASTC_8x8_UNORM_BLOCK 171
#define VK_FORMAT_ASTC_8x8_SRGB_BLOCK 172
#define VK_FORMAT_ASTC_10x5_UNORM_BLOCK 173
#define VK_FORMAT_ASTC_10x5_SRGB_BLOCK 174
#define VK_FORMAT_ASTC_10x6_UNORM_BLOCK 175
#define VK_FORMAT_ASTC_10x6_SRGB_BLOCK 176
#define VK_FORMAT_ASTC_10x8_UNORM_BLOCK 177
#define VK_FORMAT_ASTC_10x8_SRGB_BLOCK 178
#define VK_FORMAT_ASTC_10x10_UNORM_BLOCK 179
#define VK_FORMAT_ASTC_10x10_SRGB_BLOCK 180
#define VK_FORMAT_ASTC_12x10_UNORM_BLOCK 181
#define VK_FORMAT_ASTC_12x10_SRGB_BLOCK 182
#define VK_FORMAT_ASTC_12x12_UNORM_BLOCK 183
#define VK_FORMAT_ASTC_12x12_SRGB_BLOCK 184
#define VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG 1000054000
#define VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG 1000054001
#define VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG 1000054002
#define VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG 1000054003
#define VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG 1000054004
#define VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG 1000054005
#define VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG 1000054006
#define VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG 1000054007
#define VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK 147
#define VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK 148
#define VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK 149
#define VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK 150
#define VK_FORMAT_EAC_R11G11_UNORM_BLOCK 155
#define VK_FORMAT_EAC_R11G11_SNORM_BLOCK 156
#define VK_FORMAT_EAC_R11_UNORM_BLOCK 153
#define VK_FORMAT_EAC_R11_SNORM_BLOCK 154
#define VK_FORMAT_D16_UNORM 124
#define VK_FORMAT_BC7_UNORM_BLOCK 145
#define VK_FORMAT_BC7_SRGB_BLOCK 146
#define VK_FORMAT_X8_D24_UNORM_PACK32 125
#define VK_FORMAT_D24_UNORM_S8_UINT 129
#define VK_FORMAT_S8_UINT 127
#define VK_FORMAT_D32_SFLOAT_S8_UINT 130
#define VK_FORMAT_D32_SFLOAT 126
// NOLINTEND(readability-identifier-naming)

namespace {
// `outMapped` is false when `vkFormat` has no TextureFormat, in which case the result is Invalid.
TextureFormat mapVkTextureFormat(int32_t vkFormat, bool& outMapped) {
  outMapped = true;
  switch (vkFormat) {
  case VK_FORMAT_UNDEFINED:
    return TextureFormat::Invalid;
  case VK_FORMAT_R8_UNORM:
    return TextureFormat::R_UNorm8;
  case VK_FORMAT_R16_UNORM:
    return TextureFormat::R_UNorm16;
  case VK_FORMAT_R16_SFLOAT:
    return TextureFormat::R_F16;
  case VK_FORMAT_R16_UINT:
    return TextureFormat::R_UInt16;
  case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
    return TextureFormat::B5G5R5A1_UNorm;
  case VK_FORMAT_B5G6R5_UNORM_PACK16:
    return TextureFormat::B5G6R5_UNorm;
  case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
    return TextureFormat::ABGR_UNorm4;
  case VK_FORMAT_R8G8_UNORM:
    return TextureFormat::RG_UNorm8;
  case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
    return TextureFormat::R5G5B5A1_UNorm;
  case VK_FORMAT_R5G6B5_UNORM_PACK16:
    return TextureFormat::R5G6B5_UNorm;
  case VK_FORMAT_B8G8R8A8_UNORM:
    return TextureFormat::BGRA_UNorm8;
  case VK_FORMAT_R8G8B8A8_UNORM:
    return TextureFormat::RGBA_UNorm8;
  case VK_FORMAT_R8G8B8A8_SRGB:
    return TextureFormat::RGBA_SRGB;
  case VK_FORMAT_B8G8R8A8_SRGB:
    return TextureFormat::BGRA_SRGB;
  case VK_FORMAT_R16G16_UNORM:
    return TextureFormat::RG_UNorm16;
  case VK_FORMAT_R16G16_SFLOAT:
    return TextureFormat::RG_F16;
  case VK_FORMAT_R16G16_UINT:
    return TextureFormat::RG_UInt16;
  case VK_FORMAT_R16G16B16A16_UNORM:
    return TextureFormat::RGBA_UNorm16;
  case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    return TextureFormat::RGB10_A2_UNorm_Rev;
  case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    return TextureFormat::RGB10_A2_Uint_Rev;
  case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    return TextureFormat::BGR10_A2_Unorm;
  case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    // IGL has no unsigned-integer format with B in bits 0-9.
    return TextureFormat::Invalid;
  case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    return TextureFormat::B10G11R11_UFloat;
  case VK_FORMAT_R32_SFLOAT:
    return TextureFormat::R_F32;
  case VK_FORMAT_R32_UINT:
    return TextureFormat::R_UInt32;
  case VK_FORMAT_R32G32_SFLOAT:
    return TextureFormat::RG_F32;
  case VK_FORMAT_R16G16B16_SFLOAT:
    return TextureFormat::RGB_F16;
  case VK_FORMAT_R16G16B16A16_SFLOAT:
    return TextureFormat::RGBA_F16;
  case VK_FORMAT_R32G32B32_SFLOAT:
    return TextureFormat::RGB_F32;
  case VK_FORMAT_R32G32B32A32_UINT:
    return TextureFormat::RGBA_UInt32;
  case VK_FORMAT_R32G32B32A32_SFLOAT:
    return TextureFormat::RGBA_F32;
  case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_4x4;
  case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_4x4;
  case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_5x4;
  case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_5x4;
  case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_5x5;
  case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_5x5;
  case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_6x5;
  case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_6x5;
  case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_6x6;
  case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_6x6;
  case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_8x5;
  case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_8x5;
  case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_8x6;
  case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_8x6;
  case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_8x8;
  case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_8x8;
  case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_10x5;
  case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_10x5;
  case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_10x6;
  case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_10x6;
  case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_10x8;
  case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_10x8;
  case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_10x10;
  case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_10x10;
  case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_12x10;
  case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_12x10;
  case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
    return TextureFormat::RGBA_ASTC_12x12;
  case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
    return TextureFormat::SRGB8_A8_ASTC_12x12;
  case VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG:
    return TextureFormat::RGBA_PVRTC_2BPPV1;
  case VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG:
    return TextureFormat::RGBA_PVRTC_4BPPV1;
  case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
    return TextureFormat::RGB8_ETC2;
  case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
    return TextureFormat::SRGB8_ETC2;
  case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
    return TextureFormat::RGB8_Punchthrough_A1_ETC2;
  case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
    return TextureFormat::SRGB8_Punchthrough_A1_ETC2;
  case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
    return TextureFormat::RG_EAC_UNorm;
  case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
    return TextureFormat::RG_EAC_SNorm;
  case VK_FORMAT_EAC_R11_UNORM_BLOCK:
    return TextureFormat::R_EAC_UNorm;
  case VK_FORMAT_EAC_R11_SNORM_BLOCK:
    return TextureFormat::R_EAC_SNorm;
  case VK_FORMAT_D16_UNORM:
    return TextureFormat::Z_UNorm16;
  case VK_FORMAT_BC7_UNORM_BLOCK:
    return TextureFormat::RGBA_BC7_UNORM_4x4;
  case VK_FORMAT_BC7_SRGB_BLOCK:
    return TextureFormat::RGBA_BC7_SRGB_4x4;
  case VK_FORMAT_X8_D24_UNORM_PACK32:
    return TextureFormat::Z_UNorm24;
  case VK_FORMAT_D24_UNORM_S8_UINT:
    return TextureFormat::S8_UInt_Z24_UNorm;
  case VK_FORMAT_S8_UINT:
    return TextureFormat::S_UInt8;
  case VK_FORMAT_D32_SFLOAT_S8_UINT:
    return TextureFormat::S8_UInt_Z32_UNorm;
  case VK_FORMAT_D32_SFLOAT:
    return TextureFormat::Z_UNorm32;
  default:
    outMapped = false;
    return TextureFormat::Invalid;
  }
}
} // namespace

TextureFormat vkTextureFormatToTextureFormat(int32_t vkFormat) {
  bool mapped = false;
  const TextureFormat format = mapVkTextureFormat(vkFormat, mapped);
  if (!mapped) {
    IGL_DEBUG_ABORT("VkFormat value not handled: %d", static_cast<int>(vkFormat));
  }
  return format;
}

bool tryVkTextureFormatToTextureFormat(int32_t vkFormat, TextureFormat& outFormat) {
  bool mapped = false;
  const TextureFormat format = mapVkTextureFormat(vkFormat, mapped);
  if (mapped) {
    outFormat = format;
  }
  return mapped;
}

} // namespace igl::vulkan::util
