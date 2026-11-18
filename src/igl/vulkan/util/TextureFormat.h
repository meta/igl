/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <igl/TextureFormat.h>

namespace igl::vulkan::util {

/// Converts Vulkan texture format to an IGL TextureFormat.
/// @return The corresponding IGL format if known; otherwise returns TextureFormat::Invalid.
/// Aborts in builds with debug aborts enabled for a VkFormat that has no TextureFormat; use
/// tryVkTextureFormatToTextureFormat() for formats a driver chooses.
TextureFormat vkTextureFormatToTextureFormat(int32_t vkFormat);

/// Like vkTextureFormatToTextureFormat(), but returns false and leaves `outFormat` unchanged for a
/// VkFormat that has no TextureFormat, instead of aborting.
bool tryVkTextureFormatToTextureFormat(int32_t vkFormat, TextureFormat& outFormat);

} // namespace igl::vulkan::util
