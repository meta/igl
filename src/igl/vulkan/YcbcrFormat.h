/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES
#endif // !defined(VK_NO_PROTOTYPES)
#include <vulkan/vulkan_core.h>

namespace igl::vulkan {

/// Whether an image view of `format` needs a VkSamplerYcbcrConversion: the packed 4:2:2 formats and
/// every multi-planar format of Vulkan 1.1. The single-plane R10X6 and R12X4 formats and the Vulkan
/// 1.3 two-plane 4:4:4 formats are not listed.
inline bool vkFormatRequiresYcbcrConversion(VkFormat format) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case VK_FORMAT_G8B8G8R8_422_UNORM:
  case VK_FORMAT_B8G8R8G8_422_UNORM:
  case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
  case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
  case VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM:
  case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM:
  case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM:
  case VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16:
  case VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16:
  case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16:
  case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
  case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16:
  case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16:
  case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16:
  case VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16:
  case VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16:
  case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16:
  case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16:
  case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16:
  case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16:
  case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16:
  case VK_FORMAT_G16B16G16R16_422_UNORM:
  case VK_FORMAT_B16G16R16G16_422_UNORM:
  case VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM:
  case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:
  case VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM:
  case VK_FORMAT_G16_B16R16_2PLANE_422_UNORM:
  case VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM:
    return true;
  default:
    return false;
  }
}

/// Whether the features a driver lists for an external format allow a linear chroma filter in the
/// sampler YCbCr conversion for it. Without
/// VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT the conversion's chroma
/// filter must be nearest (VUID-VkSamplerYcbcrConversionCreateInfo-chromaFilter-01657).
inline bool formatFeaturesAllowLinearChromaFilter(VkFormatFeatureFlags formatFeatures) {
  return (formatFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT) != 0;
}

} // namespace igl::vulkan
