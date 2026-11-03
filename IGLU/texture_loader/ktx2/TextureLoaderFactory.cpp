/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IGLU/texture_loader/ktx2/TextureLoaderFactory.h>

#include <IGLU/texture_loader/ktx2/Header.h>
#include <algorithm>
#include <cstring>
#include <ktx.h>
#include <numeric>
#include <igl/Macros.h>
#include <igl/vulkan/util/TextureFormat.h>

#if IGL_BACKEND_D3D12 && !IGL_BACKEND_VULKAN
// Provide a fallback for D3D12-only builds where Vulkan support is not compiled in.
namespace igl::vulkan::util {
inline igl::TextureFormat vkTextureFormatToTextureFormat(int32_t /*vkFormat*/) {
  return igl::TextureFormat::Invalid;
}
} // namespace igl::vulkan::util
#endif

// @fb-only
// @fb-only
// @fb-only

namespace iglu::textureloader::ktx2 {
namespace {
template<typename T>
T align(T offset, T alignment) {
  return (offset + (alignment - 1)) & ~(alignment - 1);
}
} // namespace

uint32_t TextureLoaderFactory::minHeaderLength() const noexcept {
  return kHeaderLength;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
bool TextureLoaderFactory::canCreateInternal(DataReader headerReader,
                                             igl::Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION();
  if (headerReader.data() == nullptr) {
    igl::Result::setResult(
        outResult, igl::Result::Code::ArgumentInvalid, "Reader's data is nullptr.");
    return false;
  }
  if (headerReader.size() < kHeaderLength) {
    igl::Result::setResult(
        outResult, igl::Result::Code::ArgumentOutOfRange, "Not enough data for header.");
    return false;
  }

  const Header* header = headerReader.as<Header>();
  if (!header->tagIsValid()) {
    igl::Result::setResult(outResult, igl::Result::Code::InvalidOperation, "Incorrect identifier.");
    return false;
  }

  // vkFormat = 0 means basis universal or some non-Vulkan format.
  // In either case, we need to process the DFD to understand whether we can really handle the
  // format or not.
  if (header->vkFormat != 0 &&
      igl::vulkan::util::vkTextureFormatToTextureFormat(static_cast<int32_t>(header->vkFormat)) ==
          igl::TextureFormat::Invalid) {
    igl::Result::setResult(
        outResult, igl::Result::Code::InvalidOperation, "Unrecognized texture format.");
    return false;
  }

  return true;
}

igl::TextureRangeDesc TextureLoaderFactory::textureRange(DataReader reader) const noexcept {
  IGL_PROFILER_FUNCTION();
  const Header* header = reader.as<Header>();

  const igl::TextureRangeDesc range{
      .width = std::max(header->pixelWidth, 1u),
      .height = std::max(header->pixelHeight, 1u),
      .depth = std::max(header->pixelDepth, 1u),
      .numLayers = std::max(header->layerCount, 1u),
      .numMipLevels = std::max(header->levelCount, 1u),
      .numFaces = header->faceCount,
  };

  return range;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
bool TextureLoaderFactory::validate(DataReader reader,
                                    const igl::TextureRangeDesc& range,
                                    igl::Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION();
  const Header* header = reader.as<Header>();
  const uint32_t length = reader.size();

  if (header->sgdByteLength > std::numeric_limits<uint32_t>::max()) {
    igl::Result::setResult(outResult,
                           igl::Result::Code::InvalidOperation,
                           "Super compression global data is too large to fit in uint32_t.");
    return false;
  }
  const uint32_t sgdByteLength = static_cast<uint32_t>(header->sgdByteLength);

  if (static_cast<uint64_t>(header->dfdByteLength) + static_cast<uint64_t>(header->kvdByteLength) +
          header->sgdByteLength >
      static_cast<uint64_t>(length)) {
    igl::Result::setResult(outResult, igl::Result::Code::InvalidOperation, "Length is too short.");
    return false;
  }

  // Bound the level index against the actual input length for ALL formats. The
  // vkFormat != 0 block below does stricter, format-aware checks, but a vkFormat == 0
  // (Basis/non-Vulkan) header skips that block entirely, leaving the level index
  // unvalidated; ktxTexture_CreateFromMemory(LOAD_IMAGE_DATA) then eagerly malloc()s
  // the header-declared data size in ktxTexture2_LoadImageData, so a header declaring a
  // byteLength larger than the file drives an out-of-memory. A valid file always stores its level
  // data within the input, so this only rejects malformed headers.
  const uint64_t levelIndexEnd =
      static_cast<uint64_t>(kHeaderLength) + static_cast<uint64_t>(range.numMipLevels) * 24u;
  if (levelIndexEnd > static_cast<uint64_t>(length)) {
    igl::Result::setResult(
        outResult, igl::Result::Code::InvalidOperation, "Length is too short for the level index.");
    return false;
  }

  // Upper bound on a single level's uncompressed size.
  // Reject when a level's uncompressedByteLength exceeds what its dimensions can hold.
  // The product is accumulated with an explicit overflow check: with attacker-controlled
  // dimensions each near 2^32-1, a naive chained uint64_t multiply would wrap and could
  // yield a bound large enough to let an over-large uncompressedByteLength through. Treat
  // any overflow as absurd dimensions and reject.
  constexpr uint64_t kMaxBytesPerTexel = 16u; // RGBA32F; >= any uncompressed format
  // Maximum plausible expansion ratio for a supercompressed (ZSTD/ZLIB) level. Real texture
  // data never approaches this; a decompression bomb exceeds it by orders of magnitude.
  constexpr uint64_t kMaxSupercompressionRatio = 1024u;
  uint64_t maxLevelBytes = kMaxBytesPerTexel;
  for (const uint32_t dimension :
       {range.width, range.height, range.depth, range.numLayers, std::max(range.numFaces, 1u)}) {
    const uint64_t factor = static_cast<uint64_t>(dimension);
    if (factor != 0u && maxLevelBytes > std::numeric_limits<uint64_t>::max() / factor) {
      igl::Result::setResult(
          outResult, igl::Result::Code::InvalidOperation, "Texture dimensions are too large.");
      return false;
    }
    maxLevelBytes *= factor;
  }

  for (uint32_t mipLevel = 0; mipLevel < range.numMipLevels; ++mipLevel) {
    const uint32_t levelOffset = kHeaderLength + mipLevel * 24u;
    const uint64_t levelByteOffset = reader.readAt<uint64_t>(levelOffset);
    const uint64_t levelByteLength = reader.readAt<uint64_t>(levelOffset + 8u);
    const uint64_t levelUncompressedByteLength = reader.readAt<uint64_t>(levelOffset + 16u);
    if (levelByteOffset > static_cast<uint64_t>(length) ||
        levelByteLength > static_cast<uint64_t>(length) - levelByteOffset) {
      igl::Result::setResult(
          outResult, igl::Result::Code::InvalidOperation, "Level data exceeds the input length.");
      return false;
    }
    if (levelUncompressedByteLength > maxLevelBytes) {
      igl::Result::setResult(outResult,
                             igl::Result::Code::InvalidOperation,
                             "Uncompressed level data exceeds the texture dimensions.");
      return false;
    }
    // For supercompressed levels, uncompressedByteLength sizes the inflate destination
    // (ktxTexture2_LoadImageData allocates inflatedDataCapacity, then ktxTexture2_inflate*Int
    // spins filling it). The dimension bound above is attacker-influenced: a header declaring
    // absurd dimensions (e.g. depth in the millions) inflates maxLevelBytes enough to admit a
    // multi-gigabyte uncompressedByteLength, turning a ~1KB compressed payload into a
    // multi-second decompression bomb. Independently cap the expansion ratio against the
    // compressed size (levelByteLength is already bounded by the input length above).
    if (header->supercompressionScheme != 0u &&
        levelUncompressedByteLength > levelByteLength * kMaxSupercompressionRatio) {
      igl::Result::setResult(outResult,
                             igl::Result::Code::InvalidOperation,
                             "Supercompressed level expansion ratio is implausible.");
      return false;
    }

    // Each ZLIB-supercompressed level is a zlib stream (RFC 1950); miniz tolerates non-conformant
    // headers (e.g. CINFO > 7) on which tinfl_decompress can spin, so reject them before decoding.
    if (header->supercompressionScheme == static_cast<uint32_t>(KTX_SS_ZLIB)) {
      if (levelByteLength < 2u) {
        igl::Result::setResult(outResult,
                               igl::Result::Code::InvalidOperation,
                               "ZLIB supercompressed level is too short for a zlib header.");
        return false;
      }
      const uint8_t cmf = reader.readAt<uint8_t>(static_cast<uint32_t>(levelByteOffset));
      const uint8_t flg = reader.readAt<uint8_t>(static_cast<uint32_t>(levelByteOffset) + 1u);
      const uint32_t compressionMethod = cmf & 0x0Fu;
      const uint32_t cinfo = cmf >> 4u;
      const bool hasPresetDictionary = (flg & 0x20u) != 0u;
      const bool checkBitsValid = (((static_cast<uint32_t>(cmf) << 8u) | flg) % 31u) == 0u;
      if (compressionMethod != 8u || cinfo > 7u || hasPresetDictionary || !checkBitsValid) {
        igl::Result::setResult(outResult,
                               igl::Result::Code::InvalidOperation,
                               "ZLIB supercompressed level has a non-conformant zlib header.");
        return false;
      }
    }
  }

  if (header->vkFormat != 0u) {
    const auto format =
        igl::vulkan::util::vkTextureFormatToTextureFormat(static_cast<int32_t>(header->vkFormat));
    const auto properties = igl::TextureFormatProperties::fromTextureFormat(format);

    const uint32_t mipLevelAlignment =
        std::lcm(static_cast<uint32_t>(properties.bytesPerBlock), 4u);

    size_t rangeBytesAsSizeT = 0;
    for (uint32_t mipLevel = 0; mipLevel < range.numMipLevels; ++mipLevel) {
      rangeBytesAsSizeT += align(properties.getBytesPerRange(range.atMipLevel(mipLevel)),
                                 static_cast<size_t>(mipLevelAlignment));
    }

    if (rangeBytesAsSizeT > length) {
      igl::Result::setResult(
          outResult, igl::Result::Code::InvalidOperation, "Length is too short.");
      return false;
    }
    const uint32_t rangeBytes = static_cast<uint32_t>(rangeBytesAsSizeT);

    // Mipmap metadata is:
    //   UInt64 byteOffset
    //   UInt64 byteLength
    //   UInt64 uncompressedByteLength
    const uint32_t mipmapMetadataLength = range.numMipLevels * 24u;

    const uint32_t preSupercompressionMetadataLength =
        kHeaderLength + mipmapMetadataLength + header->dfdByteLength + header->kvdByteLength;

    const uint32_t metadataLength =
        sgdByteLength > 0 ? align(preSupercompressionMetadataLength, 8u) + sgdByteLength
                          : preSupercompressionMetadataLength;

    uint32_t expectedDataOffset = align(metadataLength, mipLevelAlignment);

    const uint32_t expectedLength = expectedDataOffset + rangeBytes;
    if (length < expectedLength) {
      igl::Result::setResult(
          outResult, igl::Result::Code::InvalidOperation, "Length shorter than expected length.");
      return false;
    }

    for (uint32_t i = 0; i < range.numMipLevels; ++i) {
      // ktx2 stores actual mip data in 'reverse' order (smallest images to largest) but the
      // metadata in 'normal' order (largest to smallest). We process the list in the same order the
      // data is stored to simplify the bookkeeping validation.
      const uint32_t mipLevel = range.numMipLevels - i - 1;

      const uint32_t offset = kHeaderLength + static_cast<uint32_t>(mipLevel) * 24u;
      const uint64_t byteOffset = reader.readAt<uint64_t>(offset);
      const uint64_t byteLength = reader.readAt<uint64_t>(offset + 8u);
      const uint64_t uncompressedByteLength = reader.readAt<uint64_t>(offset + 16u);

      if (byteLength != uncompressedByteLength && header->supercompressionScheme == 0) {
        igl::Result::setResult(
            outResult,
            igl::Result::Code::InvalidOperation,
            "Unexpected difference between byteLength and uncompressedByteLength.");
        return false;
      }

      if (byteOffset != static_cast<uint64_t>(expectedDataOffset)) {
        igl::Result::setResult(
            outResult, igl::Result::Code::InvalidOperation, "Unexpected byteOffset.");
        return false;
      }

      if (static_cast<size_t>(uncompressedByteLength) !=
          properties.getBytesPerRange(range.atMipLevel(mipLevel))) {
        igl::Result::setResult(
            outResult, igl::Result::Code::InvalidOperation, "Unexpected byteLength.");
        return false;
      }

      expectedDataOffset =
          align(expectedDataOffset + static_cast<uint32_t>(byteLength), mipLevelAlignment);
    }
  }

  return true;
}

namespace {
// @fb-only
// @fb-only
// @fb-only
  // @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
  // @fb-only
// @fb-only
// @fb-only
} // namespace

igl::TextureFormat TextureLoaderFactory::textureFormat(const ktxTexture* texture) const noexcept {
  if (texture->classId == ktxTexture2_c) {
// @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
    // @fb-only
// @fb-only
      const auto* texture2 = reinterpret_cast<const ktxTexture2*>(texture);
      return igl::vulkan::util::vkTextureFormatToTextureFormat(
          static_cast<int32_t>(texture2->vkFormat));
// @fb-only
    // @fb-only

    // @fb-only
    // @fb-only

    // @fb-only
      // @fb-only
    // @fb-only
// @fb-only
  }

  return igl::TextureFormat::Invalid;
}

} // namespace iglu::textureloader::ktx2
