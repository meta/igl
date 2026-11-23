/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/TextureBuffer.h>

#include <array>
#include <utility>
#include <igl/Macros.h>

namespace igl::opengl {

namespace {
// maps TextureCube::CubeFace to GL target type for cube map faces
// required for glTexImageXXX APIs
constexpr std::array<GLenum, 6> kCubeFaceTargets = {GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                                                    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                                                    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
                                                    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                                                    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
                                                    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z};
void swapTextureChannelsForFormat(IContext& context, GLuint target, TextureFormat iglFormat) {
  if (iglFormat == TextureFormat::A_UNorm8 && context.deviceFeatures().hasInternalRequirement(
                                                  InternalRequirement::SwizzleAlphaTexturesReq)) {
    if (iglFormat == TextureFormat::A_UNorm8) {
      // In GL3, GL_RED is used since GL_ALPHA is removed. To keep parity, red value must be set
      // to the alpha channel.
      context.texParameteri(target, GL_TEXTURE_SWIZZLE_R, GL_ZERO);
      context.texParameteri(target, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
      context.texParameteri(target, GL_TEXTURE_SWIZZLE_B, GL_ZERO);
      context.texParameteri(target, GL_TEXTURE_SWIZZLE_A, GL_RED);
    }
  }
}
} // namespace

TextureBuffer::~TextureBuffer() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  const GLuint textureId = getId();
  if (textureId != 0) {
    if (textureHandle_ != 0) {
      getContext().makeTextureHandleNonResident(textureHandle_);
    }
    getContext().deleteTextures(1, &textureId);
  }
}

uint64_t TextureBuffer::getTextureId() const {
  IGL_PROFILER_FUNCTION();
  if (textureHandle_ == 0) {
    textureHandle_ = getContext().getTextureHandle(getId());
    IGL_DEBUG_ASSERT(textureHandle_);
    getContext().makeTextureHandleResident(textureHandle_);
  }
  return textureHandle_;
}

// create a 2D texture given the specified dimensions and format
Result TextureBuffer::create(const TextureDesc& desc, bool hasStorageAlready) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  Result result = Super::create(desc, hasStorageAlready);
  if (result.isOk()) {
    const auto isSampledOrStorage = (desc.usage & (TextureDesc::TextureUsageBits::Sampled |
                                                   TextureDesc::TextureUsageBits::Storage)) != 0;
    if (isSampledOrStorage || desc.type != TextureType::TwoD || desc.numMipLevels > 1) {
      result = createTexture(desc);
    } else {
      result = Result(Result::Code::Unsupported, "invalid usage!");
    }
  }
  return result;
}

void TextureBuffer::bindImage(size_t unit) {
  IGL_PROFILER_FUNCTION();
  // The entire codebase used only combined kShaderRead|kShaderWrite access (except tests)
  // @fb-only
  // Here we used to have this condition:
  //    getUsage() & TextureUsage::kShaderWrite ? GL_WRITE_ONLY : GL_READ_ONLY,
  // So it is safe to replace it with GL_READ_WRITE
  IGL_DEBUG_ASSERT(getUsage() & TextureDesc::TextureUsageBits::Storage,
                   "Should be a storage image");
  getContext().bindImageTexture(static_cast<GLuint>(unit),
                                getId(),
                                0,
                                getTarget() == GL_TEXTURE_2D ? GL_TRUE : GL_FALSE,
                                0,
                                GL_READ_WRITE,
                                glInternalFormat_);
}

// create a texture for shader read/write usages
Result TextureBuffer::createTexture(const TextureDesc& desc) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const auto target = toGLTarget(desc.type);
  if (target == 0) {
    return Result(Result::Code::Unsupported, "Unsupported texture target");
  }

  // If usage doesn't include Storage, ensure usage includes sampled for correct format selection
  const auto usageForFormat = (desc.usage & TextureDesc::TextureUsageBits::Storage) == 0
                                  ? desc.usage | TextureDesc::TextureUsageBits::Sampled
                                  : desc.usage;
  if (!toFormatDescGL(desc.format, usageForFormat, formatDescGL_)) {
    // can't create a texture with the given format
    return Result(Result::Code::ArgumentInvalid, "Invalid texture format");
  }

  if (!getProperties().isCompressed() && formatDescGL_.type == GL_NONE) {
    return Result(Result::Code::ArgumentInvalid, "Invalid texture type");
  }

  if (desc.usage & TextureDesc::TextureUsageBits::Storage) {
    if (!getContext().deviceFeatures().hasInternalFeature(InternalFeatures::TexStorage)) {
      return Result(Result::Code::Unsupported, "Texture Storage not supported");
    }
  }

  glInternalFormat_ = formatDescGL_.internalFormat;

  // create the GL texture ID
  GLuint textureID = 0;
  getContext().genTextures(1, &textureID);

  if (textureID == 0) {
    return Result(Result::Code::RuntimeError, "Failed to create texture ID");
  }

  setTextureBufferProperties(textureID, target);
  setUsage(desc.usage);

  if (desc.type == TextureType::ExternalImage) {
    // No further initialization needed for external image textures
    return Result{};
  } else {
    return initialize(desc.debugName);
  }
}

Result TextureBuffer::initialize(const std::string& debugName) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const auto target = getTarget();
  if (target == 0) {
    return Result{Result::Code::InvalidOperation, "Unknown texture type"};
  }
  getContext().bindTexture(target, getId());
  setMaxMipLevel();
  if (getNumMipLevels() == 1) { // Change default min filter to ensure mipmapping is disabled
    getContext().texParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  }
  if (!getProperties().isCompressed()) {
    swapTextureChannelsForFormat(getContext(), target, getFormat());
  }
  if (!debugName.empty() &&
      getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DebugLabel)) {
    getContext().objectLabel(
        GL_TEXTURE, getId(), static_cast<GLsizei>(debugName.size()), debugName.c_str());
  }

  Result result;
  if (canInitialize()) {
    if (!supportsTexStorage()) {
      result = initializeWithUpload();
    } else {
      result = initializeWithTexStorage();
    }
  }

  getContext().bindTexture(getTarget(), 0);
  return result;
}

Result TextureBuffer::initializeWithUpload() const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const auto target = getTarget();
  const auto range = getFullMipRange();
  return uploadInternal(target, range, nullptr);
}

Result TextureBuffer::initializeWithTexStorage() const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const auto range = getFullMipRange();
  const auto target = getTarget();
  switch (getType()) {
  case TextureType::TwoD:
    getContext().texStorage2D(target,
                              range.numMipLevels,
                              glInternalFormat_,
                              static_cast<GLsizei>(range.width),
                              static_cast<GLsizei>(range.height));
    break;
  case TextureType::TwoDArray:
    getContext().texStorage3D(target,
                              range.numMipLevels,
                              glInternalFormat_,
                              static_cast<GLsizei>(range.width),
                              static_cast<GLsizei>(range.height),
                              static_cast<GLsizei>(range.numLayers));
    break;
  case TextureType::ThreeD:
    getContext().texStorage3D(target,
                              range.numMipLevels,
                              glInternalFormat_,
                              static_cast<GLsizei>(range.width),
                              static_cast<GLsizei>(range.height),
                              static_cast<GLsizei>(range.depth));
    break;
  case TextureType::Cube:
    getContext().texStorage2D(target,
                              range.numMipLevels,
                              glInternalFormat_,
                              static_cast<GLsizei>(range.width),
                              static_cast<GLsizei>(range.height));
    break;
  case TextureType::ExternalImage:
  case TextureType::Invalid:
  default:
    IGL_DEBUG_ABORT("Unknown texture type");
    return Result{Result::Code::InvalidOperation, "Unknown texture type"};
  }
  return getContext().getLastError();
}

Result TextureBuffer::upload2D(GLenum target,
                               const TextureRangeDesc& range,
                               bool texImage,
                               const void* IGL_NULLABLE data) const {
  IGL_PROFILER_FUNCTION();
  if (data == nullptr || !getProperties().isCompressed()) {
    if (texImage) {
      getContext().texImage2D(target,
                              static_cast<GLsizei>(range.mipLevel),
                              formatDescGL_.internalFormat,
                              static_cast<GLsizei>(range.width),
                              static_cast<GLsizei>(range.height),
                              0, // border
                              formatDescGL_.format,
                              formatDescGL_.type,
                              data);
    } else {
      getContext().texSubImage2D(target,
                                 static_cast<GLsizei>(range.mipLevel),
                                 static_cast<GLsizei>(range.x),
                                 static_cast<GLsizei>(range.y),
                                 static_cast<GLsizei>(range.width),
                                 static_cast<GLsizei>(range.height),
                                 formatDescGL_.format,
                                 formatDescGL_.type,
                                 data);
    }
  } else {
    const auto numCompressedBytes = getProperties().getBytesPerRange(range);
    IGL_DEBUG_ASSERT(numCompressedBytes > 0);
    if (texImage) {
      getContext().compressedTexImage2D(target,
                                        static_cast<GLint>(range.mipLevel),
                                        formatDescGL_.internalFormat,
                                        static_cast<GLsizei>(range.width),
                                        static_cast<GLsizei>(range.height),
                                        0, // border
                                        static_cast<GLsizei>(numCompressedBytes), // TODO: does not
                                                                                  // work for
                                                                                  // compressed
                                                                                  // mipmaps
                                        data);
    } else {
      getContext().compressedTexSubImage2D(getTarget(),
                                           static_cast<GLint>(range.mipLevel),
                                           static_cast<GLint>(range.x),
                                           static_cast<GLint>(range.y),
                                           static_cast<GLsizei>(range.width),
                                           static_cast<GLsizei>(range.height),
                                           formatDescGL_.internalFormat,
                                           static_cast<GLsizei>(numCompressedBytes), // TODO: does
                                                                                     // not work for
                                                                                     // compressed
                                                                                     // mipmaps
                                           data);
    }
  }
  return getContext().getLastError();
}
Result TextureBuffer::upload2DArray(GLenum target,
                                    const TextureRangeDesc& range,
                                    bool texImage,
                                    const void* IGL_NULLABLE data) const {
  IGL_PROFILER_FUNCTION();
  if (data == nullptr || !getProperties().isCompressed()) {
    if (texImage) {
      getContext().texImage3D(target,
                              static_cast<GLint>(range.mipLevel),
                              formatDescGL_.internalFormat,
                              static_cast<GLsizei>(range.width),
                              static_cast<GLsizei>(range.height),
                              static_cast<GLsizei>(range.numLayers),
                              0, // border
                              formatDescGL_.format,
                              formatDescGL_.type,
                              data);
    } else {
      getContext().texSubImage3D(target,
                                 static_cast<GLsizei>(range.mipLevel),
                                 static_cast<GLsizei>(range.x),
                                 static_cast<GLsizei>(range.y),
                                 static_cast<GLsizei>(range.layer),
                                 static_cast<GLsizei>(range.width),
                                 static_cast<GLsizei>(range.height),
                                 static_cast<GLsizei>(range.numLayers),
                                 formatDescGL_.format,
                                 formatDescGL_.type,
                                 data);
    }
  } else {
    const auto numCompressedBytes = getProperties().getBytesPerRange(range);
    IGL_DEBUG_ASSERT(numCompressedBytes > 0);
    if (texImage) {
      getContext().compressedTexImage3D(target,
                                        static_cast<GLint>(range.mipLevel),
                                        formatDescGL_.internalFormat,
                                        static_cast<GLsizei>(range.width),
                                        static_cast<GLsizei>(range.height),
                                        static_cast<GLsizei>(range.numLayers),
                                        0, // border
                                        static_cast<GLsizei>(numCompressedBytes), // TODO: does not
                                                                                  // work for
                                                                                  // compressed
                                                                                  // mipmaps
                                        data);
    } else {
      getContext().compressedTexSubImage3D(getTarget(),
                                           static_cast<GLint>(range.mipLevel),
                                           static_cast<GLint>(range.x),
                                           static_cast<GLint>(range.y),
                                           static_cast<GLint>(range.layer),
                                           static_cast<GLsizei>(range.width),
                                           static_cast<GLsizei>(range.height),
                                           static_cast<GLsizei>(range.numLayers),
                                           formatDescGL_.internalFormat,
                                           static_cast<GLsizei>(numCompressedBytes),
                                           data);
    }
  }
  return getContext().getLastError();
}

Result TextureBuffer::upload3D(GLenum target,
                               const TextureRangeDesc& range,
                               bool texImage,
                               const void* IGL_NULLABLE data) const {
  IGL_PROFILER_FUNCTION();
  if (data == nullptr || !getProperties().isCompressed()) {
    if (texImage) {
      getContext().texImage3D(target,
                              static_cast<GLint>(range.mipLevel),
                              formatDescGL_.internalFormat,
                              static_cast<GLsizei>(range.width),
                              static_cast<GLsizei>(range.height),
                              static_cast<GLsizei>(range.depth),
                              0, // border
                              formatDescGL_.format,
                              formatDescGL_.type,
                              data);
    } else {
      getContext().texSubImage3D(target,
                                 static_cast<GLsizei>(range.mipLevel),
                                 static_cast<GLsizei>(range.x),
                                 static_cast<GLsizei>(range.y),
                                 static_cast<GLsizei>(range.z),
                                 static_cast<GLsizei>(range.width),
                                 static_cast<GLsizei>(range.height),
                                 static_cast<GLsizei>(range.depth),
                                 formatDescGL_.format,
                                 formatDescGL_.type,
                                 data);
    }
  } else {
    const auto numCompressedBytes = getProperties().getBytesPerRange(range);
    IGL_DEBUG_ASSERT(numCompressedBytes > 0);
    if (texImage) {
      getContext().compressedTexImage3D(target,
                                        static_cast<GLint>(range.mipLevel),
                                        formatDescGL_.internalFormat,
                                        static_cast<GLsizei>(range.width),
                                        static_cast<GLsizei>(range.height),
                                        static_cast<GLsizei>(range.depth),
                                        0, // border
                                        static_cast<GLsizei>(numCompressedBytes), // TODO: does not
                                                                                  // work for
                                                                                  // compressed
                                                                                  // mipmaps
                                        data);
    } else {
      getContext().compressedTexSubImage3D(getTarget(),
                                           static_cast<GLint>(range.mipLevel),
                                           static_cast<GLint>(range.x),
                                           static_cast<GLint>(range.y),
                                           static_cast<GLint>(range.z),
                                           static_cast<GLsizei>(range.width),
                                           static_cast<GLsizei>(range.height),
                                           static_cast<GLsizei>(range.depth),
                                           formatDescGL_.internalFormat,
                                           static_cast<GLsizei>(numCompressedBytes),
                                           data);
    }
  }
  return getContext().getLastError();
}

bool TextureBuffer::needsRepacking(const TextureRangeDesc& range, size_t bytesPerRow) const {
  IGL_PROFILER_FUNCTION();
  if (bytesPerRow == 0) {
    return false;
  }

  const auto rangeBytesPerRow = getProperties().getBytesPerRow(range);
  if (rangeBytesPerRow == bytesPerRow) {
    return false;
  }

  // GL_UNPACK_ALIGNMENT supports padding up to, but not including, the alignment value.
  // If bytesPerRow is equal to the packed row length rounded up to the nearest alignment multiple,
  // the data does not need to be repacked.
  const auto delta = bytesPerRow - rangeBytesPerRow;
  if (delta < 8 && bytesPerRow % 8 == 0) {
    return false;
  } else if (delta < 4 && bytesPerRow % 4 == 0) {
    return false;
  } else if (delta < 2 && bytesPerRow % 2 == 0) {
    return false;
  }

  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::UnpackRowLength)) {
    // GL_UNPACK_ROW_LENGTH supports cases where bytesPerRow is a multiple of the texel size or, for
    // compressed textures, the texel block size.
    return bytesPerRow % getProperties().bytesPerBlock != 0;
  }

  return true;
}

// upload data into the given mip level
// a sub-rect of the texture may be specified to only upload the sub-rect
Result TextureBuffer::uploadInternal(TextureType /*type*/,
                                     const TextureRangeDesc& range,
                                     const void* IGL_NULLABLE data,
                                     size_t bytesPerRow,
                                     const uint32_t* IGL_NULLABLE /*mipLevelBytes*/) const {
  IGL_PROFILER_FUNCTION();
  if (data == nullptr) {
    return Result{};
  }
  const auto target = getTarget();
  if (target == 0) {
    return Result{Result::Code::InvalidOperation, "Unknown texture type"};
  }
  getContext().bindTexture(target, getId());

  auto result = uploadInternal(target, range, data, bytesPerRow);

  getContext().bindTexture(getTarget(), 0);
  return result;
}

Result TextureBuffer::uploadInternal(GLenum target,
                                     const TextureRangeDesc& range,
                                     const void* IGL_NULLABLE data,
                                     size_t bytesPerRow) const {
  IGL_PROFILER_FUNCTION();
  // Use TexImage when range covers full texture AND texture was not initialized with TexStorage
  const auto texImage = isValidForTexImage(range) && !supportsTexStorage();

  const bool unpackRowLengthSupported =
      getContext().deviceFeatures().hasInternalFeature(InternalFeatures::UnpackRowLength);
  const int unpackRowLength = unpackRowLengthSupported &&
                                      bytesPerRow % getProperties().bytesPerBlock == 0
                                  ? static_cast<int>(bytesPerRow / getProperties().bytesPerBlock)
                                  : 0;

  if (unpackRowLength > 0) {
    getContext().pixelStorei(GL_UNPACK_ROW_LENGTH, unpackRowLength);
    getContext().pixelStorei(GL_UNPACK_ALIGNMENT, 1);
  } else {
    if (unpackRowLengthSupported) {
      getContext().pixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    }
    getContext().pixelStorei(
        GL_UNPACK_ALIGNMENT,
        this->getAlignment(static_cast<uint32_t>(bytesPerRow), range.mipLevel, range.width));
  }

  Result result;
  for (auto mipLevel = range.mipLevel; mipLevel < range.mipLevel + range.numMipLevels; ++mipLevel) {
    const auto mipRange = range.atMipLevel(mipLevel);
    for (auto face = range.face; face < range.face + range.numFaces; ++face) {
      const auto faceRange = mipRange.atFace(face);
      const auto* faceData =
          data == nullptr ? nullptr : getSubRangeStart(data, range, faceRange, bytesPerRow);
      switch (type_) {
      case TextureType::TwoD:
        result = upload2D(target, faceRange, texImage, faceData);
        break;
      case TextureType::TwoDArray:
        result = upload2DArray(target, faceRange, texImage, faceData);
        break;
      case TextureType::ThreeD:
        result = upload3D(target, faceRange, texImage, faceData);
        break;
      case TextureType::Cube:
        result = upload2D(kCubeFaceTargets[faceRange.face], faceRange, texImage, faceData);
        break;
      case TextureType::ExternalImage:
      case TextureType::Invalid:
      default:
        return Result{Result::Code::InvalidOperation, "Unknown texture type"};
      }
      if (!result.isOk()) {
        return result;
      }
    }
    if (!result.isOk()) {
      break;
    }
  }

  if (mipmapGeneration_ == TextureDesc::TextureMipmapGeneration::AutoGenerateOnUpload) {
    if (range.mipLevel != 0) {
      return Result{Result::Code::InvalidOperation,
                    "AutoGenerateOnUpload requires mipLevel to be uploaded to be 0"};
    }
    generateMipmap();
    mipmapsAreAvailableAndUploaded_ = true;
  }

  return result;
}

bool TextureBuffer::canInitialize() const {
  return !getProperties().isCompressed() ||
         (supportsTexStorage() && getContext().deviceFeatures().hasTextureFeature(
                                      TextureFeatures::TextureCompressionTexStorage)) ||
         getContext().deviceFeatures().hasTextureFeature(
             TextureFeatures::TextureCompressionTexImage);
}

bool TextureBuffer::supportsTexStorage() const {
  return (getUsage() & TextureDesc::TextureUsageBits::Storage) != 0 &&
         contains(getContext().deviceFeatures().getTextureFormatCapabilities(getFormat()),
                  ICapabilities::TextureFormatCapabilityBits::Storage);
}

} // namespace igl::opengl
