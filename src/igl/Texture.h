/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <igl/CommandQueue.h>
#include <igl/Common.h>
#include <igl/ITrackedResource.h>
#include <igl/TextureFormat.h>
#include <igl/base/IAttachmentInterop.h>

namespace igl {

/**
 * @brief TextureCubeFace denotes side of the face in a cubemap setting.
   Based on https://www.khronos.org/opengl/wiki/Cubemap_Texture

  PosX            - The U coordinate is going behind the viewer, with the V coordinate going down.
  NegX            - The U coordinate is going forward, with the V coordinate going down.
  PosY            - The U coordinate goes to the right, with the V coordinate going forward.
  NegY            - The U coordinate goes to the right, with the V coordinate going backward.
  PosZ            - The U coordinate goes to the right, with the V coordinate going down.
  NegZ            - The U coordinate goes to the left (relative to us facing forwards), with the V
 coordinate going down.
 */
enum class TextureCubeFace : uint8_t { PosX = 0, NegX, PosY, NegY, PosZ, NegZ };

/**
 * @brief Descriptor for texture dimensions
 *
 *  x            - offset position in width
 *  y            - offset position in height
 *  z            - offset position in depth
 *  width        - width of the range
 *  height       - height of the range
 *  depth        - depth of the range
 *  layer        - layer offset for 1D/2D array textures. Not used for cube textures faces.
 *  numLayers    - number of layers in the range
 *  mipLevel     - mipmap level offset of the range
 *  numMipLevels - number of mipmap levels in the range
 *  face         - face offset for cube textures
 *  numFaces     - number of cube texture faces in the range
 */
struct TextureRangeDesc {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
  uint32_t layer = 0;
  uint32_t numLayers = 1;
  uint32_t mipLevel = 0;
  uint32_t numMipLevels = 1;
  uint32_t face = 0;
  uint32_t numFaces = 1;

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc new1D(uint32_t x,
                                uint32_t width,
                                uint32_t mipLevel = 0,
                                uint32_t numMipLevels = 1);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc new1DArray(uint32_t x,
                                     uint32_t width,
                                     uint32_t layer,
                                     uint32_t numLayers,
                                     uint32_t mipLevel = 0,
                                     uint32_t numMipLevels = 1);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc new2D(uint32_t x,
                                uint32_t y,
                                uint32_t width,
                                uint32_t height,
                                uint32_t mipLevel = 0,
                                uint32_t numMipLevels = 1);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc new2DArray(uint32_t x,
                                     uint32_t y,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t layer,
                                     uint32_t numLayers,
                                     uint32_t mipLevel = 0,
                                     uint32_t numMipLevels = 1);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc new3D(uint32_t x,
                                uint32_t y,
                                uint32_t z,
                                uint32_t width,
                                uint32_t height,
                                uint32_t depth,
                                uint32_t mipLevel = 0,
                                uint32_t numMipLevels = 1);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc newCube(uint32_t x,
                                  uint32_t y,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t mipLevel = 0,
                                  uint32_t numMipLevels = 1);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc newCubeFace(uint32_t x,
                                      uint32_t y,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t face,
                                      uint32_t mipLevel = 0,
                                      uint32_t numMipLevels = 1);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static TextureRangeDesc newCubeFace(uint32_t x,
                                      uint32_t y,
                                      uint32_t width,
                                      uint32_t height,
                                      TextureCubeFace face,
                                      uint32_t mipLevel = 0,
                                      uint32_t numMipLevels = 1);

  /**
   * @brief Returns a new TextureRangeDesc based on this one but reduced to the specified mipLevel.
   *
   * @param newMipLevel The mip level of the returned range.
   * @remark The returned range only has 1 mip level.
   */
  [[nodiscard]] TextureRangeDesc atMipLevel(uint32_t newMipLevel) const noexcept;
  /**
   * @brief Returns a new TextureRangeDesc based on this one but with the specified number of
   * mip levels.
   *
   * @param newNumMipLevels The number of mip levels in the returned range.
   */
  [[nodiscard]] TextureRangeDesc withNumMipLevels(uint32_t newNumMipLevels) const noexcept;
  /**
   * @brief Returns a new TextureRangeDesc based on this one but reduced to the specified layer.
   *
   * @param newLayer The layer of the returned range.
   * @remark The returned range only has 1 layer.
   */
  [[nodiscard]] TextureRangeDesc atLayer(uint32_t newLayer) const noexcept;
  /**
   * @brief Returns a new TextureRangeDesc based on this one but with the specified number of
   * layers.
   *
   * @param newNumLayers The number of layers in the returned range.
   */
  [[nodiscard]] TextureRangeDesc withNumLayers(uint32_t newNumLayers) const noexcept;
  /**
   * @brief Returns a new TextureRangeDesc based on this one but reduced to the specified face.
   *
   * @param newFace The face of the returned range.
   * @remark The returned range only has 1 face.
   */
  [[nodiscard]] TextureRangeDesc atFace(uint32_t newFace) const noexcept;
  /**
   * @brief Returns a new TextureRangeDesc based on this one but reduced to the specified face.
   *
   * @param newFace The face of the returned range.
   * @remark The returned range only has 1 face.
   */
  [[nodiscard]] TextureRangeDesc atFace(TextureCubeFace newFace) const noexcept;
  /**
   * @brief Returns a new TextureRangeDesc based on this one but with the specified number of faces.
   *
   * @param newNumFaces The number of faces in the returned range.
   */
  [[nodiscard]] TextureRangeDesc withNumFaces(uint32_t newNumFaces) const noexcept;

  /**
   * Validates the range.
   *
   * A range is valid if:
   *  1) width, height, depth, numFaces, numLayers and numMipLevels are all at least 1.
   *  2) numMipLevels is less than or equal to the max mip levels for the width, height and depth.
   *  3) mipLevel, x + width, y + height, z + depth and layer + numLayers are all <=
   *     std::numeric_limits<uint32_t>::max().
   *  4) (x + width) * (y + height) * (z + depth) * (layer + numLayers) * numFaces <=
   *     std::numeric_limits<uint32_t>::max().
   *  5) face < 6 and numFaces <= 6
   */
  [[nodiscard]] Result validate() const noexcept;

  bool operator==(const TextureRangeDesc& rhs) const noexcept = default;
  bool operator!=(const TextureRangeDesc& rhs) const noexcept = default;
};

/**
 * @brief Encapsulates properties of a texture format
 *
 *  name               - Stringified IGL enum for the format
 *  format             - IGL enum for the format
 *  componentsPerPixel - Number of components for each pixel (e.g., RGB has 3)
 *  bytesPerBlock      - Bytes per pixel block (compressed) or per pixel (uncompressed)
 *  blockWidth         - Block width for compressed textures (always 1 for uncompressed)
 *  blockHeight        - Block height for compressed textures (always 1 for uncompressed)
 *  blockDepth         - Block depth for compressed textures (always 1 for uncompressed)
 *  minBlocksX         - Minimum number of blocks in the X direction for compressed textures
 *  minBlocksY         - Minimum number of blocks in the Y direction for compressed textures
 *  minBlocksZ         - Minimum number of blocks in the Z direction for compressed textures
 *  flags              - Additional boolean flags for the format:
 *                        - Depth:          Depth texture format
 *                        - Stencil:        Stencil texture format
 *                        - Compressed:     Compressed texture format
 *                        - sRGB:           sRGB texture format
 *                        - Integer:        Integer formats do not support sampling via
 *                                          float samplers `texture2D` (require `utexture2D`)
 *                        - HDR:            More than 8 bits per component
 *                        - VariableLength: Variable length blocks
 */
struct TextureFormatProperties {
  static TextureFormatProperties fromTextureFormat(TextureFormat format);

  enum Flags : uint8_t {
    Depth = 1 << 0,
    Stencil = 1 << 1,
    Compressed = 1 << 2,
    sRGB = 1 << 3,
    Integer = 1 << 4,
    HDR = 1 << 5, // Color format with more than 8 bits per component
    VariableLength = 1 << 6, // Variable length blocks
  };

  const char* IGL_NONNULL name = "Invalid";
  const TextureFormat format = TextureFormat::Invalid;
  const uint8_t componentsPerPixel = 1;
  const uint8_t bytesPerBlock = 1;
  const uint8_t blockWidth = 1;
  const uint8_t blockHeight = 1;
  const uint8_t blockDepth = 1;
  const uint8_t minBlocksX = 1;
  const uint8_t minBlocksY = 1;
  const uint8_t minBlocksZ = 1;
  const uint8_t numPlanes = 1;
  const uint8_t flags = 0;

  /**
   * @brief true for TextureFormat::Invalid
   */
  [[nodiscard]] bool isValid() const noexcept {
    return format != TextureFormat::Invalid;
  }
  /**
   * @brief true for compressed texture formats.
   */
  [[nodiscard]] bool isCompressed() const noexcept {
    return (flags & Flags::Compressed) != 0;
  }
  /**
   * @brief true for integer texture formats.
   */
  [[nodiscard]] bool isInteger() const noexcept {
    return (flags & Flags::Integer) != 0;
  }
  /**
   * @brief true for sRGB texture formats.
   */
  [[nodiscard]] bool isSRGB() const noexcept {
    return (flags & Flags::sRGB) != 0;
  }
  /**
   * @brief true for high precision color texture formats.
   */
  [[nodiscard]] bool isHDR() const noexcept {
    return (flags & Flags::HDR) != 0;
  }

  [[nodiscard]] bool hasDepth() const noexcept {
    return (flags & Flags::Depth) != 0;
  }
  [[nodiscard]] bool hasStencil() const noexcept {
    return (flags & Flags::Stencil) != 0;
  }
  [[nodiscard]] bool hasColor() const noexcept {
    return !hasDepth() && !hasStencil();
  }

  /**
   * @brief true for depth-only texture formats (e.g., TextureFormat::Z_UNorm24).
   */
  [[nodiscard]] bool isDepthOnly() const noexcept {
    return hasDepth() && !hasStencil();
  }
  /**
   * @brief true for depth-only and depth-stencil texture formats.
   */
  [[nodiscard]] bool isDepthOrDepthStencil() const noexcept {
    return hasDepth();
  }
  /**
   * @brief true for stencil-only texture formats (e.g., TextureFormat::S_UInt8).
   */
  [[nodiscard]] bool isStencilOnly() const noexcept {
    return !hasDepth() && hasStencil();
  }
  /**
   * @brief true for depth-only, stencil-only and depth-stencil texture formats.
   */
  [[nodiscard]] bool isDepthOrStencil() const noexcept {
    return (flags & Flags::Depth) != 0 || (flags & Flags::Stencil) != 0;
  }

  [[nodiscard]] bool isVariableLength() const noexcept {
    return (flags & Flags::VariableLength) != 0;
  }

  /**
   * @brief Utility function to calculate the number of rows in the range for the texture format.
   * For uncompressed textures, this will be range.height. For compressed textures, range.height
   * rounded up to the nearest multiple of blockHeight.
   *
   * @param range  range.width, range.height, and range.depth should be the actual dimensions of the
   * range to calculate for. For subranges and mip levels other than 0, these should be the
   * dimensions of the subrange and/or mip level, which may be be less than the full texture
   * dimensions.
   * @return Calculated number of rows of texture data for the texture format.
   */
  [[nodiscard]] uint32_t getRows(TextureRangeDesc range) const noexcept;

  /**
   * @brief Utility function to calculate the size in bytes per row for a texture format.
   *
   * @param texWidth  The width, in pixels, of the texture data. This should be the row width to
   * calculate for. For subranges and mip levels other than 0, this should be the width of the
   * subrange and/or mip level, which may be be less than the full texture width.
   * @return Calculated total size in bytes of a row of texture data for the texture format.
   */
  [[nodiscard]] uint32_t getBytesPerRow(uint32_t texWidth) const noexcept;

  /**
   * @brief Utility function to calculate the size in bytes per row for a texture format.
   *
   * @param range  The range, in pixels, of the texture data row. range.width be the row width to
   * calculate for. For subranges and mip levels other than 0, this should be the width of the
   * subrange and/or mip level, which may be be less than the full texture width.
   * @return Calculated total size in bytes of a row of texture data for the texture format.
   */
  [[nodiscard]] uint32_t getBytesPerRow(TextureRangeDesc range) const noexcept;

  /**
   * @brief Utility function to calculate the size in bytes per texture layer for a texture format.
   *
   * @param texWidth  The width of the texture layer.
   * @param texHeight  The height of the texture layer.
   * @param texDepth  The depth of the texture layer.
   * @param bytesPerRow The size in bytes of each texture row. 0 for the format's default.
   * @remark texWidth, texHeight and texDepth should be the actual dimensions of the range to
   * calculate for. For subranges and mip levels other than 0, this should be the dimensions of the
   * subrange and/or mip level, which may be be less than the full texture dimensions.
   * @return Calculated total size in bytes of a layer of texture data for the texture format.
   */
  [[nodiscard]] size_t getBytesPerLayer(uint32_t texWidth,
                                        uint32_t texHeight,
                                        uint32_t texDepth,
                                        uint32_t bytesPerRow = 0) const noexcept;

  /**
   * @brief Utility function to calculate the size in bytes per texture layer for a texture format.
   *
   * @param range  The range of the texture to calculate for. The range should be the full size of
   * the first mip level to calculate for. range.x, range.y, range.z, range.mipLevel and range.layer
   * are not used.
   * @param bytesPerRow The size in bytes of each texture row. 0 for the format's default.
   * @remark range.width, range.height, and range.depth should be the actual dimensions of the range
   * to calculate for. For subranges and mip levels other than 0, these should be the dimensions of
   * the subrange and/or mip level, which may be be less than the full texture dimensions.
   * @return Calculated total size in bytes of a layer of texture data for the texture format.
   */
  [[nodiscard]] size_t getBytesPerLayer(TextureRangeDesc range,
                                        uint32_t bytesPerRow = 0) const noexcept;

  /**
   * @brief Utility function to calculate the size in bytes per texture range for a texture format.
   *
   * @param range  The range of the texture to calculate for. The range should be the full size of
   * the first mip level to calculate for. range.x, range.y and range.z are not used.
   * @param bytesPerRow The size in bytes of each texture row. 0 for the format's default. This
   * mustt be 0 if numMipLevels is more than 1.
   * @remark range can include more than one layer. range can also include more than one mip level.
   * When range includes more than one mip level, dimensions  are divided by two for each subsequent
   * mip level.
   * @return Calculated total size in bytes of a the range of texture data for the texture format.
   */
  [[nodiscard]] size_t getBytesPerRange(TextureRangeDesc range,
                                        uint32_t bytesPerRow = 0) const noexcept;

  /**
   * @brief Utility function to calculate the number of mip levels given a total size in bytes of
   * texture data.
   *
   * @param texWidth  The width of the first mip level of the texture.
   * @param texHeight  The height of the first mip level of the texture.
   * @return Calculated number of mip levels.
   */
  [[nodiscard]] uint32_t getNumMipLevels(uint32_t texWidth,
                                         uint32_t texHeight,
                                         size_t totalBytes) const noexcept;

  /**
   * @brief Utility function to calculate the byte offset of the start of a subrange within a block
   * of data.
   *
   * This method assumes the following data hierarchy:
   *   mip level
   *     array layer
   *       cube face
   *         z slice
   *           row
   *
   * This method only handles the case where the subrange is a proper subset of the full block of
   * data. It also only handles subranges in terms of mip levels, layers or faces. It does not
   * handle subsets along the x, y or z dimensions.
   *
   * @param range The range of the full block of data.
   * @param subRange The subrange within the full block of data for which to get the byte offset.
   * @param bytesPerRow The number of bytes in each row of range (the full block of data). 0 means
   * the default for the texture format. Must be 0 if subrange starts at a different mip level than
   * the full range or covers more than one mip level.
   * @return The byte offset within the full block of data for the start of the subrange. */
  [[nodiscard]] size_t getSubRangeByteOffset(const TextureRangeDesc& range,
                                             const TextureRangeDesc& subRange,
                                             uint32_t bytesPerRow = 0) const noexcept;
};

enum ImageAspectBits : uint32_t {
  ImageAspectBits_Invalid = 0xffffffff,
  ImageAspectBits_None = 0,
  ImageAspectBits_Color = 0x1,
  ImageAspectBits_Depth = 0x2,
  ImageAspectBits_Stencil = 0x4,
  ImageAspectBits_Plane_0 = 0x10,
  ImageAspectBits_Plane_1 = 0x20,
  ImageAspectBits_Plane_2 = 0x40,
};

using ImageAspectFlags = uint32_t;

enum Swizzle : uint8_t {
  Swizzle_Default = 0,
  Swizzle_0,
  Swizzle_1,
  Swizzle_R,
  Swizzle_G,
  Swizzle_B,
  Swizzle_A,
};

struct ComponentMapping {
  Swizzle r = Swizzle_Default;
  Swizzle g = Swizzle_Default;
  Swizzle b = Swizzle_Default;
  Swizzle a = Swizzle_Default;
  [[nodiscard]] bool identity() const {
    return r == Swizzle_Default && g == Swizzle_Default && b == Swizzle_Default &&
           a == Swizzle_Default;
  }
};

/**
 * @brief Descriptor for texture view creation methods
 */
struct TextureViewDesc {
  TextureType type = TextureType::TwoD;
  // mutate the original texture format, subject to the underlying graphics API limitations
  TextureFormat format = TextureFormat::Invalid; // use Invalid to keep the original format
  ImageAspectFlags aspect = ImageAspectBits_Invalid; // use Invalid to keep the original aspect
  uint32_t layer = 0;
  uint32_t numLayers = 1;
  uint32_t mipLevel = 0;
  uint32_t numMipLevels = 1;
  // swizzle RGBA components in a custom way
  ComponentMapping swizzle = {};

  std::string debugName;
};

/**
 * @brief Descriptor for internal texture creation methods used in IGL
 *
 *  width              - width of the texture
 *  height             - height of the texture
 *  depth              - depth of the texture
 *  numLayers          - Number of layers for array texture
 *  numSamples         - Number of samples for multisampling
 *  usage              - Bitwise flag for containing a mask of TextureUsageBits
 *  numMipLevels       - Number of mipmaps to generate
 *  type               - The type of texture (2D, 3D, Cube, etc.)
 *  format             - Internal texture format type
 *  storage            - Internal resource storage type. ResourceStorage::Memoryless textures must
 *                       be single-mip 2D or 2D array attachments (optionally also InputAttachment)
 *                       that are not exportable and use optimal tiling. Their contents exist only
 *                       during a render pass: load actions must be Clear or DontCare and store
 *                       actions DontCare or MsaaResolve. Uploading or reading back the contents is
 *                       invalid. Backends without tile memory fall back to private memory.
 *  tiling             - Image layout for texture storage (Optimal or Linear)
 *  exportability      - Whether the texture can be exported (NoExport or Exportable)
 *  mipmapGeneration   - Specifies when and how mipmaps should be generated (Manual,
 *                       AutoGenerateOnUpload)
 *  debugName          - Optional debug name for the texture
 */
struct TextureDesc {
  /**
   * @brief Bitwise flags for texture usage
   *
   *  Sampled - Can be used as read-only texture in vertex/fragment shaders
   *  Storage - Can be used as read/write storage texture in vertex/fragment/compute shaders
   *  Attachment - Can be bound for render target
   *  InputAttachment - Can be read as an input attachment by a later subpass of the same render
   *                    pass (requires Attachment)
   */
  enum TextureUsageBits : uint8_t {
    Sampled = 1 << 0,
    Storage = 1 << 1,
    Attachment = 1 << 2,
    InputAttachment = 1 << 3, // NOLINT(readability-identifier-naming)
  };

  using TextureUsage = uint8_t;

  /**
   * @brief Flag for texture image storage (Vulkan Only)
   *
   *  Optimal - Image layout for best texture read performance
   *            Corresponds to VK_IMAGE_TILING_OPTIMAL
   *  Linear - Image layout for linearly storing texture data
   *           Corresponds to VK_IMAGE_TILING_LINEAR
   */
  enum class TextureTiling : uint8_t { Optimal, Linear };

  enum class TextureExportability : uint8_t { NoExport, Exportable };

  /**
   * @brief Flag to specify whether IGL needs to generate mipmaps and when
   *
   *  Manual - Mip levels are expected to be generated after the texture
   *           is created and uploaded by the caller by calling one of IGL's
   *           mipmap generation functions OR provided during upload.
   *
   *  AutoGenerateOnUpload - IGL will generate mipmaps when the texture
   *                         is uploaded to the device
   */
  enum class TextureMipmapGeneration : uint8_t { Manual, AutoGenerateOnUpload };

  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
  uint32_t numLayers = 1;
  uint32_t numSamples = 1;
  TextureUsage usage = 0;
  uint32_t numMipLevels = 1;
  TextureType type = TextureType::Invalid;
  TextureFormat format = TextureFormat::Invalid;
  ResourceStorage storage = ResourceStorage::Invalid;
  TextureTiling tiling = TextureTiling::Optimal;
  TextureExportability exportability = TextureExportability::NoExport;
  TextureMipmapGeneration mipmapGeneration = TextureMipmapGeneration::Manual;

  std::string debugName;

  bool operator==(const TextureDesc& rhs) const;
  bool operator!=(const TextureDesc& rhs) const;

  /**
   * @brief Utility to create a new 2D texture
   *
   * @param format The format of the texture
   * @param width  The width of the texture
   * @param height The height of the texture
   * @param usage A combination of TextureUsage flags
   * @param debugName An optional debug name
   * @return TextureDesc
   */
  static TextureDesc new2D(TextureFormat format,
                           uint32_t width,
                           uint32_t height,
                           TextureUsage usage,
                           const char* IGL_NULLABLE debugName = nullptr) {
    return TextureDesc{
        .width = width,
        .height = height,
        .usage = usage,
        .type = TextureType::TwoD,
        .format = format,
        .debugName = debugName != nullptr ? std::string(static_cast<const char*>(debugName))
                                          : std::string(),
    };
  }

  /**
   * @brief Utility to create a new 2D texture array
   *
   * @param format The format of the texture
   * @param width  The width of the texture
   * @param height The height of the texture
   * @param numLayers  The number layers of the texture array
   * @param usage A combination of TextureUsage flags
   * @param debugName An optional debug name
   * @return TextureDesc
   */
  static TextureDesc new2DArray(TextureFormat format,
                                uint32_t width,
                                uint32_t height,
                                uint32_t numLayers,
                                TextureUsage usage,
                                const char* IGL_NULLABLE debugName = nullptr) {
    return TextureDesc{
        .width = width,
        .height = height,
        .numLayers = numLayers,
        .usage = usage,
        .type = TextureType::TwoDArray,
        .format = format,
        .debugName = debugName != nullptr ? std::string(static_cast<const char*>(debugName))
                                          : std::string(),
    };
  }

  /**
   * @brief Utility to create a new cube texture
   *
   * @param format The format of the texture
   * @param width  The width of the texture
   * @param height The height of the texture
   * @param usage A combination of TextureUsage flags
   * @param debugName An optional debug name
   * @return TextureDesc
   */
  static TextureDesc newCube(TextureFormat format,
                             uint32_t width,
                             uint32_t height,
                             TextureUsage usage,
                             const char* IGL_NULLABLE debugName = nullptr) {
    return TextureDesc{
        .width = width,
        .height = height,
        .usage = usage,
        .type = TextureType::Cube,
        .format = format,
        .debugName = debugName != nullptr ? std::string(static_cast<const char*>(debugName))
                                          : std::string(),
    };
  }

  /**
   * @brief Utility to create a new 3D texture
   *
   * @param format The format of the texture
   * @param width  The width of the texture
   * @param height The height of the texture
   * @param depth  The depth of the texture
   * @param usage A combination of TextureUsage flags
   * @param debugName An optional debug name
   * @return TextureDesc
   */
  static TextureDesc new3D(TextureFormat format,
                           uint32_t width,
                           uint32_t height,
                           uint32_t depth,
                           TextureUsage usage,
                           const char* IGL_NULLABLE debugName = nullptr) {
    return TextureDesc{
        .width = width,
        .height = height,
        .depth = depth,
        .usage = usage,
        .type = TextureType::ThreeD,
        .format = format,
        .debugName = debugName != nullptr ? std::string(static_cast<const char*>(debugName))
                                          : std::string(),
    };
  }

  /**
   * @brief Utility to create a new external image texture
   *
   * @param format The format of the texture
   * @param width  The width of the texture
   * @param height The height of the texture
   * @param debugName An optional debug name
   * @return TextureDesc
   */
  static TextureDesc newExternalImage(TextureFormat format,
                                      uint32_t width,
                                      uint32_t height,
                                      TextureUsage usage,
                                      const char* IGL_NULLABLE debugName = nullptr) {
    return TextureDesc{
        .width = width,
        .height = height,
        .usage = usage,
        .type = TextureType::ExternalImage,
        .format = format,
        .debugName = debugName != nullptr ? std::string(static_cast<const char*>(debugName))
                                          : std::string(),
    };
  }

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  /**
   * @brief Utility to create a new image texture with linked hardware buffer
   *
   * @param format The format of the texture
   * @param usage A combination of TextureUsage flags
   * @param width  The width of the texture
   * @param height The height of the texture
   * @param debugName An optional debug name
   * @return TextureDesc
   */
  static TextureDesc newNativeHWBufferImage(TextureFormat format,
                                            TextureUsage usage,
                                            uint32_t width,
                                            uint32_t height,
                                            const char* IGL_NULLABLE debugName = nullptr) {
    return TextureDesc{
        .width = width,
        .height = height,
        .usage = usage,
        .type = TextureType::TwoD,
        .format = format,
        .storage = ResourceStorage::Shared,
        .debugName = debugName != nullptr ? std::string(static_cast<const char*>(debugName))
                                          : std::string(),
    };
  }
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

  /**
   * @brief Creates a TextureRangeDesc equivalent to descriptor.
   *
   * The range includes the full width, height, depth, number of layers, number of cube faces, and
   * number of mip levels in the texture descriptor. */
  [[nodiscard]] TextureRangeDesc asRange() const noexcept;

  /**
   * @brief Utility to calculate maximum mipmap level support
   *
   * @param width  The width of the texture
   * @param height The height of the texture
   * @param depth The depth of the texture
   * @return uint32_t
   */
  static constexpr uint32_t calcNumMipLevels(uint32_t width, uint32_t height, uint32_t depth = 1) {
    if (!width || !height || !depth) {
      return 0;
    }

    uint32_t levels = 1;

    const size_t combinedValue = width | height | depth;
    while (combinedValue >> levels) {
      levels++;
    }

    return levels;
  }
};

/**
 * @brief Interface class for all textures.
 * This should only be used for the purpose of getting information about the texture using the
 * gettor methods defined below.
 */
class ITexture : public ITrackedResource<ITexture>, public base::IAttachmentInterop {
 public:
  explicit ITexture(TextureFormat format) :
    properties_(TextureFormatProperties::fromTextureFormat(format)) {}
  ~ITexture() override = default;

  /**
   * @brief Indicates if this type of texture supports upload.
   */
  [[nodiscard]] virtual bool supportsUpload() const {
    return ((getUsage() & (TextureDesc::TextureUsageBits::Sampled |
                           TextureDesc::TextureUsageBits::Storage)) != 0) &&
           !properties_.isDepthOrStencil();
  }

  /**
   * @brief Uploads the given data into texture memory.
   *
   * Upload supports arbitrary ranges. That is, data may point to data for multiple mip levels, cube
   * faces, array layers and Z slices.
   *
   * This method assumes the following data hierarchy:
   *   mip level
   *     array layer
   *       cube face
   *         z slice
   *           row
   *
   * @param range        The texture range descriptor containing the offset & dimensions of the
   * upload process.
   * @param data         The pointer to the data. May be a nullptr to force initialization without
   * providing data.
   * @param bytesPerRow Number of bytes per row. If 0, it will be autocalculated assuming no
   * padding.
   * @param mipLevelBytes Number of bytes per mip level. If nullptr, it will be autocalculated
   * @return Result      A flag for the result of operation
   */
  Result upload(const TextureRangeDesc& range,
                const void* IGL_NULLABLE data,
                size_t bytesPerRow = 0,
                const uint32_t* IGL_NULLABLE mipLevelBytes = nullptr) const;

  // Texture Accessor Methods
  /**
   * @brief Returns the aspect ratio (width / height) of the texture.
   *
   * @return float
   */
  [[nodiscard]] float getAspectRatio() const;
  /**
   * @brief Returns size (width x height) dimension of the texture.
   * For 1D textures, return (width,1)
   * For 2D textures, 3D textures and/or cube, return (width,height)
   *
   * @return Size
   */
  [[nodiscard]] Size getSize() const;
  /**
   * @brief Returns depth dimension of the texture
   * For 1D, 2D textures, return 1
   * For 3D textures and/or cube, returns depth
   *
   * @return size_t
   */
  [[nodiscard]] uint32_t getDepth() const;
  /**
   * @brief Returns dimensions (width, height and depth) of the texture.
   *
   * @return size_t
   */
  [[nodiscard]] virtual Dimensions getDimensions() const = 0;
  /**
   * @brief Returns the number of layers of the texture
   * For non array textures, return 1
   *
   * @return size_t
   */
  [[nodiscard]] virtual uint32_t getNumLayers() const = 0;
  /**
   * @brief Returns the number of faces the texture has
   * For non-cube textures, return 1
   *
   * @return size_t
   */
  [[nodiscard]] uint32_t getNumFaces() const;
  /**
   * @brief Returns texture format properties of the texture
   *
   * @return TextureFormatProperties
   */
  [[nodiscard]] const TextureFormatProperties& getProperties() const {
    return properties_;
  }
  /**
   * @brief Returns texture type of the texture
   *
   * @return TextureType
   */
  [[nodiscard]] virtual TextureType getType() const = 0;
  /**
   * @brief Returns bitwise flag containing the usage of the texture
   *
   * @return unsigned bitwise flag
   */
  [[nodiscard]] virtual TextureDesc::TextureUsage getUsage() const = 0;
  /**
   * @brief Returns the storage requested in TextureDesc::storage, not necessarily the storage the
   * backend uses. Invalid for textures not created from a TextureDesc.
   *
   * @return ResourceStorage
   */
  [[nodiscard]] virtual ResourceStorage getStorage() const {
    return ResourceStorage::Invalid;
  }
  /**
   * @brief Returns number of samples
   *
   * @return The count of samples in the underlying texture
   */
  [[nodiscard]] virtual uint32_t getSamples() const = 0;
  /**
   * @brief Generates mipmap command using the command queue
   *
   * @param cmdQueue The command queue that is generated from the graphics device.
   * @param range The texture range descriptor containing the extents of the image to generate
   * mipmaps. If nullptr is provided, the function generates mips for all levels. Note: not all
   * parameters in the TextureRangeDesc structure are supported for mipmap generation. For example,
   * the x, y, z offsets and the dimensions are not used by this function. Also, not all
   * implementations support generating a subset of miplevels for an image (OpenGL, for example)
   */
  virtual void generateMipmap(ICommandQueue& cmdQueue,
                              const TextureRangeDesc* IGL_NULLABLE range = nullptr) const = 0;
  /**
   * @brief Generates mipmap command using an existing command buffer
   * @param range The texture range descriptor containing the extents of the image to generate
   * mipmaps. If nullptr is provided, the function generates mips for all levels. Note: not all
   * parameters in the TextureRangeDesc structure are supported for mipmap generation. For example,
   * the x, y, z offsets and the dimensions are not used by this function. Also, not all
   * implementations support generating a subset of miplevels for an image (OpenGL, for example)
   * @param cmdBuffer A command buffer that is generated from an ICommandQueue.
   */
  virtual void generateMipmap(ICommandBuffer& cmdBuffer,
                              const TextureRangeDesc* IGL_NULLABLE range = nullptr) const = 0;

  /**
   * @brief Returns the number of mipmap levels
   */
  [[nodiscard]] virtual uint32_t getNumMipLevels() const = 0;
  /**
   * @brief Returns a flag to indicate mipmap for the texture has been generated.
   *
   * @return True  Mipmap has been generated
   *         False Otherwise
   */
  [[nodiscard]] virtual bool isRequiredGenerateMipmap() const = 0;
  /**
   * @brief Attempts to calculate how much memory this texture uses. There are many factors that
   * make this calculation difficult and we can't be confident about driver implementations, so this
   * number can't be fully trusted.
   *
   * @return The estimated size of this texture, in bytes.
   */
  [[nodiscard]] size_t getEstimatedSizeInBytes() const;
  /**
   * @brief Returns a texture id suitable for bindless rendering (descriptor indexing on Vulkan and
   * gpuResourceID on Metal)
   *
   * @return uint64_t
   */
  [[nodiscard]] virtual uint64_t getTextureId() const = 0;

  /**
   * @brief Validates the range against texture dimensions at the range's mip level.
   *
   * @return The returned Result indicates whether the range is valid or not.
   */
  [[nodiscard]] Result validateRange(const TextureRangeDesc& range) const noexcept;
  /**
   * @brief Returns a TextureRangeDesc for the texture's full range at the specified mip level.
   *
   * For cube map textures, this range includes all faces.
   * @return TextureRangeDesc.
   */
  [[nodiscard]] TextureRangeDesc getFullRange(size_t mipLevel = 0,
                                              size_t numMipLevels = 1) const noexcept;
  /**
   * @brief Returns a TextureRangeDesc for the texture's full range, including all mip levels.
   *
   * @return TextureRangeDesc.
   */
  [[nodiscard]] TextureRangeDesc getFullMipRange() const noexcept;
  /**
   * @brief Returns a TextureRangeDesc for the texture's full range for a single cube face at the
   * specified mip level.
   *
   * @return TextureRangeDesc.
   */
  [[nodiscard]] TextureRangeDesc getCubeFaceRange(TextureCubeFace face,
                                                  size_t mipLevel = 0,
                                                  size_t numMipLevels = 1) const noexcept;
  /**
   * @brief Returns a TextureRangeDesc for the texture's full range for a single cube face at the
   * specified mip level.
   *
   * @return TextureRangeDesc.
   */
  [[nodiscard]] TextureRangeDesc getCubeFaceRange(size_t face,
                                                  size_t mipLevel = 0,
                                                  size_t numMipLevels = 1) const noexcept;

  /**
   * @brief Returns a TextureRangeDesc for the texture's full range for a single array layer at the
   * specified mip level.
   *
   * @return TextureRangeDesc.
   */
  [[nodiscard]] TextureRangeDesc getLayerRange(size_t layer,
                                               size_t mipLevel = 0,
                                               size_t numMipLevels = 1) const noexcept;

  /**
   * @brief A helper function to quickly access TextureFormat.
   *
   * @return TextureFormat.
   */
  [[nodiscard]] TextureFormat getFormat() const {
    return properties_.format;
  }

  /**
   * @brief Returns a flag checking if this texture belongs to swapchain.
   *
   * @return Boolean.
   */
  [[nodiscard]] virtual bool isSwapchainTexture() const {
    return false;
  }

  /**
   * @brief Helper method to repack texture data to achieve a desired alignment.
   *
   * Copies data from originalData to repackedData, one row of data at a time. Each row of data will
   * be repackedBytesPerRow bytes long. If repackedBytesPerRow is less than originalDataBytesPerRow,
   * data will NOT be 0 padded.
   *
   * Repacking only works correctly for 1 mip level.
   *
   * This method assumes the following data hierarchy:
   *   mip level
   *     array layer
   *       cube face
   *         z slice
   *           row
   *
   * @param properties The texture format properties for the data being repacked.
   * @param range The texture range of texture data that is being repacked.
   * @param originalData Pointer to the texture data to repack.
   * @param originalDataBytesPerRow Bytes per row of original data. 0 means no padding per row
   * (i.e., the data is packed).
   * @param repackedData Pointer to the destination to write repacked data to.
   * @param repackedBytesPerRow Bytes per row of repacked data. 0 means no padding per row
   * (i.e., the data should be packed).
   * @param flipVertical If true, the repacked data will be flipped vertically for each texture
   * layer, cube face, and Z slice.
   */
  static void repackData(const TextureFormatProperties& properties,
                         const TextureRangeDesc& range,
                         const uint8_t* IGL_NONNULL originalData,
                         size_t originalDataBytesPerRow,
                         uint8_t* IGL_NONNULL repackedData,
                         size_t repackedBytesPerRow,
                         bool flipVertical = false);

 protected:
  [[nodiscard]] const void* IGL_NONNULL getSubRangeStart(const void* IGL_NONNULL data,
                                                         const TextureRangeDesc& range,
                                                         const TextureRangeDesc& subRange,
                                                         size_t bytesPerRow) const noexcept;

  [[nodiscard]] virtual bool needsRepacking([[maybe_unused]] const TextureRangeDesc& range,
                                            [[maybe_unused]] size_t bytesPerRow) const {
    return false;
  }

  [[nodiscard]] virtual Result uploadInternal(
      [[maybe_unused]] TextureType type,
      [[maybe_unused]] const TextureRangeDesc& range,
      [[maybe_unused]] const void* IGL_NULLABLE data,
      [[maybe_unused]] size_t bytesPerRow = 0,
      [[maybe_unused]] const uint32_t* IGL_NULLABLE mipLevelBytes = nullptr) const {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    return Result{Result::Code::Unimplemented, "Upload not implemented."};
  }

  const TextureFormatProperties properties_;
};

/**
 * @brief Holds textures associated with an externally owned surface (e.g., a window).
 */
struct SurfaceTextures {
  /** @brief The surface's color texture. */
  std::shared_ptr<ITexture> color;
  /** @brief The surface's depth texture. */
  std::shared_ptr<ITexture> depth;
  /** @brief Optional single-sample resolve target for an MSAA color texture. When set, the
   * render pass resolves the multisampled color into this texture. Null when MSAA is disabled. */
  std::shared_ptr<ITexture> colorResolve;
};

} // namespace igl

namespace std {

// Add a hash function for older compilers
template<>
struct hash<igl::TextureFormat> {
  // Declare member
  size_t operator()(const igl::TextureFormat& /*key*/) const;
};

} // namespace std
