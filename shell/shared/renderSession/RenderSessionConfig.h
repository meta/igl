/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <igl/DeviceFeatures.h>

namespace igl::shell {

struct RenderSessionConfig {
  std::string displayName;
  BackendVersion backendVersion;
  TextureFormat swapchainColorTextureFormat = TextureFormat::BGRA_UNorm8;
  TextureFormat depthTextureFormat = TextureFormat::Z_UNorm16;
  ColorSpace swapchainColorSpace = ColorSpace::SRGBNonlinear;
};

} // namespace igl::shell
