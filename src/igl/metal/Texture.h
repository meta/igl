/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

@protocol MTLCommandBuffer;
@protocol MTLTexture;
#import <Metal/MTLPixelFormat.h>
#import <Metal/MTLTexture.h>
#import <QuartzCore/CAMetalLayer.h>
#include <igl/DeviceFeatures.h>
#include <igl/Macros.h>
#include <igl/Texture.h>

#if IGL_PLATFORM_APPLE
NS_ASSUME_NONNULL_BEGIN
#endif

namespace igl::metal {

class Texture final : public ITexture {
  friend class Device;
  friend class PlatformDevice;

 public:
  /// `usage` and `storage` are the requested TextureDesc values. Wrapped native textures have a 0
  /// `usage` (getUsage() derives it from the MTLTextureUsage) and an Invalid `storage`.
  Texture(id<MTLTexture> texture,
          const ICapabilities& capabilities,
          TextureDesc::TextureMipmapGeneration mipmapGeneration =
              TextureDesc::TextureMipmapGeneration::Manual,
          TextureDesc::TextureUsage usage = 0,
          ResourceStorage storage = ResourceStorage::Invalid);
  Texture(id<CAMetalDrawable> drawable,
          const ICapabilities& capabilities,
          TextureDesc::TextureMipmapGeneration mipmapGeneration =
              TextureDesc::TextureMipmapGeneration::Manual);
  ~Texture() override;
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;
  Texture(Texture&&) = delete;
  Texture& operator=(Texture&&) = delete;

  Result getBytes(const TextureRangeDesc& range, void* outData, size_t bytesPerRow = 0) const;

  // Accessors
  [[nodiscard]] Dimensions getDimensions() const override;
  [[nodiscard]] uint32_t getNumLayers() const override;
  [[nodiscard]] TextureType getType() const override;
  [[nodiscard]] TextureDesc::TextureUsage getUsage() const override;
  [[nodiscard]] ResourceStorage getStorage() const override;
  [[nodiscard]] uint32_t getSamples() const override;
  [[nodiscard]] uint32_t getNumMipLevels() const override;
  void generateMipmap(ICommandQueue& cmdQueue,
                      const TextureRangeDesc* IGL_NULLABLE range = nullptr) const override;
  void generateMipmap(ICommandBuffer& cmdBuffer,
                      const TextureRangeDesc* IGL_NULLABLE range = nullptr) const override;
  [[nodiscard]] bool isRequiredGenerateMipmap() const override;
  [[nodiscard]] uint64_t getTextureId() const override;
  [[nodiscard]] TextureDesc::TextureMipmapGeneration getMipmapGeneration() const;

  IGL_INLINE id<MTLTexture> _Nullable get() const {
    return (drawable_) ? drawable_.texture : value_;
  }
  IGL_INLINE id<CAMetalDrawable> _Nullable getDrawable() const {
    return drawable_;
  }

  // IAttachmentInterop interface
  [[nodiscard]] void* IGL_NULLABLE getNativeImage() const override;
  [[nodiscard]] void* IGL_NULLABLE getNativeImageView() const override;
  [[nodiscard]] const base::AttachmentInteropDesc& getDesc() const override;

  static TextureDesc::TextureUsage toTextureUsage(MTLTextureUsage usage);
  static MTLTextureUsage toMTLTextureUsage(TextureDesc::TextureUsage usage);

  static MTLPixelFormat textureFormatToMTLPixelFormat(TextureFormat value);
  static TextureFormat mtlPixelFormatToTextureFormat(MTLPixelFormat value);
  static MTLTextureType convertType(TextureType value, size_t numSamples);
  static TextureType convertType(MTLTextureType value);
  static NSUInteger getMetalSlice(TextureType type, uint32_t face, uint32_t layer);
  static TextureRangeDesc atMetalSlice(TextureType type,
                                       const TextureRangeDesc& range,
                                       NSUInteger metalSlice);

  TextureDesc textureDesc_;

 private:
  [[nodiscard]] bool needsRepacking(const TextureRangeDesc& range, size_t bytesPerRow) const final;
  Result uploadInternal(TextureType type,
                        const TextureRangeDesc& range,
                        const void* IGL_NULLABLE data,
                        size_t bytesPerRow,
                        const uint32_t* IGL_NULLABLE mipLevelBytes) const final;

  [[nodiscard]] Result uploadViaStagingBuffer(TextureType type,
                                              const TextureRangeDesc& range,
                                              const void* IGL_NULLABLE data,
                                              size_t bytesPerRow) const;
  
  void generateMipmap(id<MTLCommandBuffer> cmdBuffer) const;

  // Given bytes per row of an input texture, return bytesPerRow value
  // accepted by Texture::upload and MTL replaceRegion.
  [[nodiscard]] size_t toMetalBytesPerRow(size_t bytesPerRow) const;

  id<MTLTexture> _Nullable value_;
  id<CAMetalDrawable> _Nullable drawable_;
  const ICapabilities& capabilities_;
  TextureDesc::TextureMipmapGeneration mipmapGeneration_ =
      TextureDesc::TextureMipmapGeneration::Manual;
  TextureDesc::TextureUsage usage_ = 0;
  ResourceStorage storage_ = ResourceStorage::Invalid;

  /// @brief To record whether mipmaps are available and uploaded to the GPU. This is used by the
  /// `isRequiredGenerateMipmap()` function
  mutable bool mipmapsAreAvailableAndUploaded_ = false;

  mutable base::AttachmentInteropDesc attachmentDesc_; // Cached for IAttachmentInterop::getDesc()
  mutable int uploadTimes_ = 0;
};

} // namespace igl::metal

#if IGL_PLATFORM_APPLE
NS_ASSUME_NONNULL_END
#endif
