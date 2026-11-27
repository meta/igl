/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/opengl/TextureBufferBase.h>

namespace igl::opengl {

// TextureBufferExternal encapsulates OpenGL textures without the guarantee of the lifecycle
// Specifically, this class does not delete the GL texture it encapsulates on destruction
class TextureBufferExternal : public TextureBufferBase {
  using Super = TextureBufferBase;
  friend class PlatformDevice; // So that PlatformDevice can do setTextureBufferProperties

 public:
  explicit TextureBufferExternal(IContext& context,
                                 TextureFormat format,
                                 TextureDesc::TextureUsage usage);
  ~TextureBufferExternal() override = default;

  [[nodiscard]] bool supportsUpload() const final;
};

[[nodiscard]] inline bool TextureBufferExternal::supportsUpload() const {
  return false;
}

} // namespace igl::opengl
