/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/SamplerState.h>

#include <igl/Common.h>
#include <igl/Macros.h>
#include <igl/Texture.h>
#include <igl/opengl/DepthStencilState.h>
#include <igl/opengl/Texture.h>

namespace igl::opengl {
namespace {
bool isPowerOfTwo(size_t number) {
  return (number & (number - 1)) == 0;
}
} // namespace

SamplerState::SamplerState(IContext& context, const SamplerStateDesc& desc) :
  WithContext(context),
  minMipFilter_(convertMinMipFilter(desc.minFilter, desc.mipFilter)),
  magFilter_(convertMagFilter(desc.magFilter)),
  mipLodMin_(desc.mipLodMin),
  mipLodMax_(desc.mipLodMax),
  maxAnisotropy_(desc.maxAnisotropic),
  addressU_(convertAddressMode(desc.addressModeU)),
  addressV_(convertAddressMode(desc.addressModeV)),
  addressW_(convertAddressMode(desc.addressModeW)),
  depthCompareFunction_(DepthStencilState::convertCompareFunction(desc.depthCompareFunction)),
  depthCompareEnabled_(desc.depthCompareEnabled),
  isYUV_(desc.yuvFormat != TextureFormat::Invalid) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const std::hash<SamplerStateDesc> h;
  hash_ = h(desc);
}

void SamplerState::bind(ITexture* t) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY_NOT(t == nullptr)) {
    return;
  }

  auto* texture = static_cast<Texture*>(t);
  if (texture->getSamplerHash() == hash_) {
    return;
  }
  texture->setSamplerHash(hash_);

  auto type = texture->getType();
  auto target = texture->toGLTarget(type);
  if (target == 0) {
    return;
  }

  const auto& deviceFeatures = getContext().deviceFeatures();
  const bool isDepthOrDepthStencil = texture->getProperties().isDepthOrDepthStencil();
  // From OpenGL ES 3.1 spec
  // The effective internal format specified for the texture arrays is a sized internal depth or
  // depth and stencil format (see table 8.14), the value of TEXTURE_COMPARE_MODE is NONE , and
  // either the magnification filter is not NEAREST or the minification filter is neither
  // NEAREST nor NEAREST_MIPMAP_NEAREST.
  applyMinFilter(target, isDepthOrDepthStencil);
  applyMagFilter(target, isDepthOrDepthStencil);

  // See https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glTexParameter.xml
  // for OpenGL version information.
  // Ensure we have mipmaps before setting this state. This should also catch special
  // texture types that may not support mipmaps, like ExternalOES textures on Android
  if (texture->getNumMipLevels() > 1 &&
      deviceFeatures.hasFeature(DeviceFeatures::SamplerMinMaxLod)) {
    getContext().texParameteri(target, GL_TEXTURE_MIN_LOD, mipLodMin_);
    getContext().texParameteri(target, GL_TEXTURE_MAX_LOD, mipLodMax_);
  }
  if (deviceFeatures.hasFeature(DeviceFeatures::TextureFilterAnisotropic)) {
    // @fb-only
    // Disable the anisotropic filter for now, it's causing a crash on some devices
    (void)maxAnisotropy_;
  }

  if (isDepthOrDepthStencil &&
      deviceFeatures.hasInternalFeature(InternalFeatures::TextureCompare)) {
    getContext().texParameteri(target,
                               GL_TEXTURE_COMPARE_MODE,
                               depthCompareEnabled_ ? GL_COMPARE_REF_TO_TEXTURE : GL_NONE);
    getContext().texParameteri(target, GL_TEXTURE_COMPARE_FUNC, depthCompareFunction_);
  }

  // Fall back to GL_CLAMP_TO_EDGE when GL_CLAMP_TO_BORDER is not supported
  const GLint wrapS = resolveWrapMode(addressU_);
  const GLint wrapT = resolveWrapMode(addressV_);
  const GLint wrapR = resolveWrapMode(addressW_);

  if (!deviceFeatures.hasFeature(DeviceFeatures::TextureNotPot)) {
    const auto dimensions = texture->getDimensions();
    if (!isPowerOfTwo(dimensions.width) || !isPowerOfTwo(dimensions.height)) {
      getContext().texParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      getContext().texParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
      getContext().texParameteri(target, GL_TEXTURE_WRAP_S, wrapS);
      getContext().texParameteri(target, GL_TEXTURE_WRAP_T, wrapT);
    }
  } else {
    getContext().texParameteri(target, GL_TEXTURE_WRAP_S, wrapS);
    getContext().texParameteri(target, GL_TEXTURE_WRAP_T, wrapT);
  }

  if (type == TextureType::TwoDArray || type == TextureType::ThreeD) {
    getContext().texParameteri(target, GL_TEXTURE_WRAP_R, wrapR);
  }
}

void SamplerState::applyMinFilter(GLint target, bool isDepthOrDepthStencil) const {
  IGL_PROFILER_FUNCTION();
  if (!depthCompareEnabled_ && isDepthOrDepthStencil && minMipFilter_ != GL_NEAREST &&
      minMipFilter_ != GL_NEAREST_MIPMAP_NEAREST) {
    IGL_LOG_INFO_ONCE(
        "OpenGL requires a GL_NEAREST or NEAREST_MIPMAP_NEAREST min filter for depth/stencil "
        "samplers when DepthCompareEnabled is false, falling back to supported mode instead of "
        "requested format.");
    const auto supportedMode =
        (minMipFilter_ == GL_LINEAR_MIPMAP_NEAREST || minMipFilter_ == GL_NEAREST_MIPMAP_LINEAR ||
         minMipFilter_ == GL_LINEAR_MIPMAP_LINEAR)
            ? GL_NEAREST_MIPMAP_NEAREST
            : GL_NEAREST;
    getContext().texParameteri(target, GL_TEXTURE_MIN_FILTER, supportedMode);
  } else {
    getContext().texParameteri(target, GL_TEXTURE_MIN_FILTER, minMipFilter_);
  }
}

void SamplerState::applyMagFilter(GLint target, bool isDepthOrDepthStencil) const {
  IGL_PROFILER_FUNCTION();
  if (!depthCompareEnabled_ && isDepthOrDepthStencil && magFilter_ != GL_NEAREST) {
    IGL_LOG_INFO_ONCE(
        "OpenGL requires a GL_NEAREST mag filter for depth/stencil samplers when "
        "DepthCompareEnabled is false, falling back to GL_NEAREST instead of requested format.");
    getContext().texParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  } else {
    getContext().texParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter_);
  }
}

GLint SamplerState::resolveWrapMode(GLint wrap) const {
  IGL_PROFILER_FUNCTION();
  if (!getContext().deviceFeatures().hasInternalFeature(InternalFeatures::TextureClampToBorder) &&
      wrap == GL_CLAMP_TO_BORDER) {
    return GL_CLAMP_TO_EDGE;
  }
  return wrap;
}

// utility functions for converting from IGL sampler state enums to GL enums
GLint SamplerState::convertMinMipFilter(SamplerMinMagFilter minFilter, SamplerMipFilter mipFilter) {
  switch (mipFilter) {
  case SamplerMipFilter::Disabled:
    return (minFilter == SamplerMinMagFilter::Nearest) ? GL_NEAREST : GL_LINEAR;

  case SamplerMipFilter::Nearest:
    return (minFilter == SamplerMinMagFilter::Nearest) ? GL_NEAREST_MIPMAP_NEAREST
                                                       : GL_LINEAR_MIPMAP_NEAREST;

  case SamplerMipFilter::Linear:
    return (minFilter == SamplerMinMagFilter::Nearest) ? GL_NEAREST_MIPMAP_LINEAR
                                                       : GL_LINEAR_MIPMAP_LINEAR;

  default:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return 0;
  }
}

GLint SamplerState::convertMagFilter(SamplerMinMagFilter magFilter) {
  return (magFilter == SamplerMinMagFilter::Nearest) ? GL_NEAREST : GL_LINEAR;
}

SamplerMinMagFilter SamplerState::convertGLMagFilter(GLint glMagFilter) {
  return (glMagFilter == GL_NEAREST) ? SamplerMinMagFilter::Nearest : SamplerMinMagFilter::Linear;
}

SamplerMinMagFilter SamplerState::convertGLMinFilter(GLint glMinFilter) {
  switch (glMinFilter) {
  case GL_NEAREST:
  case GL_NEAREST_MIPMAP_NEAREST:
  case GL_NEAREST_MIPMAP_LINEAR:
    return SamplerMinMagFilter::Nearest;

  case GL_LINEAR:
  case GL_LINEAR_MIPMAP_NEAREST:
  case GL_LINEAR_MIPMAP_LINEAR:
    return SamplerMinMagFilter::Linear;

  default:
#ifndef GTEST
    IGL_DEBUG_ASSERT_NOT_REACHED();
#endif
  }

  return SamplerMinMagFilter::Nearest;
}

SamplerMipFilter SamplerState::convertGLMipFilter(GLint glMinFilter) {
  switch (glMinFilter) {
  case GL_NEAREST:
  case GL_LINEAR:
    return SamplerMipFilter::Disabled;

  case GL_NEAREST_MIPMAP_NEAREST:
  case GL_LINEAR_MIPMAP_NEAREST:
    return SamplerMipFilter::Nearest;

  case GL_NEAREST_MIPMAP_LINEAR:
  case GL_LINEAR_MIPMAP_LINEAR:
    return SamplerMipFilter::Linear;

  default:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return SamplerMipFilter::Disabled;
  }
  IGL_UNREACHABLE_RETURN(SamplerMipFilter::Disabled)
}

GLint SamplerState::convertAddressMode(SamplerAddressMode addressMode) {
  switch (addressMode) {
  case SamplerAddressMode::Repeat:
    return GL_REPEAT;

  case SamplerAddressMode::Clamp:
    return GL_CLAMP_TO_EDGE;

  case SamplerAddressMode::MirrorRepeat:
    return GL_MIRRORED_REPEAT;

  case SamplerAddressMode::ClampToBorder:
    return GL_CLAMP_TO_BORDER;

  default:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return 0;
  }
}

SamplerAddressMode SamplerState::convertGLAddressMode(GLint glAddressMode) {
  switch (glAddressMode) {
  case GL_REPEAT:
    return SamplerAddressMode::Repeat;

  case GL_CLAMP_TO_EDGE:
    return SamplerAddressMode::Clamp;

  case GL_MIRRORED_REPEAT:
    return SamplerAddressMode::MirrorRepeat;

  case GL_CLAMP_TO_BORDER:
    return SamplerAddressMode::ClampToBorder;

  default:
    return SamplerAddressMode::Repeat;
  }
  IGL_UNREACHABLE_RETURN(SamplerAddressMode::Repeat)
}

bool SamplerState::isYUV() const noexcept {
  return isYUV_;
}

} // namespace igl::opengl
