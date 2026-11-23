/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <igl/DepthStencilState.h>
#include <igl/ITrackedResource.h>
#include <igl/TextureFormat.h>

namespace igl {

/**
 * @brief Filtering option to use when sampling textures within the same mipmap level.
 *
 * Nearest : The sampled value is the value from the texel closest to the sampling point.
 * Linear  : The sampled value is linearly interpolated from the texel values nearest to the
 *           sampling point.
 */
enum class SamplerMinMagFilter : uint8_t { Nearest = 0, Linear };

/**
 * @brief Filtering option to use when sampling textures between mipmap levels.
 *
 * Disabled: The sampled value is the selected from mipmap level 0.
 * Nearest : The sampled value is the selected from the nearest mipmap level to the filter.
 * Linear  : The sampled value is linearly interpolated from the sampled values from the nearest
 *           mipmap levels.
 */
enum class SamplerMipFilter : uint8_t { Disabled = 0, Nearest, Linear };

/**
 * @brief Filtering option to use when sampling outside the boundary of a texture
 *
 * Repeat       : The texture repeats outside the range [0, 1].
 * Clamp        : Sampling locations < 0 are clamped to 0. Sampling locations > 1 are clamped to 1.
 * MirrorRepeat : The texture repeats outside the range [0, 1]. Every other repetition is a mirror
 *                image.
 * ClampToBorder : Sampling locations outside [0, 1] return the border color (default: 0, 0, 0, 0).
 */
enum class SamplerAddressMode : uint8_t { Repeat = 0, Clamp, MirrorRepeat, ClampToBorder };

/**
 * @brief Describes the texture sampling configuration for a texture.
 *
 * This describes what sampling filters to use, what mipmap levels to sample from
 * how to sample between mipmap levels, how to handle sampling locations outside
 * the texture, and what comparison operation to use when sampling depth textures.
 *
 * The type and valid ranges for mip lod min/max were decided based on:
 * - Ability to hash a SamplerStateDesc: the current implementation
 *   relies on perfect hashing and we don't have many free bits left.
 * - Realistic use cases: negative values aren't useful (Metal says so)
 *   and big values aren't either (GL says so). In practice, a texture
 *   will only have log2(max_size) lods so a value of 16 is more than
 *   any of our platforms support.
 */
struct SamplerStateDesc {
  /**
   * @brief The filtering option to use when a texel is smaller than a fragment.
   */
  SamplerMinMagFilter minFilter = SamplerMinMagFilter::Nearest;
  /**
   * @brief The filtering option to use when a texel is larger than a fragment.
   */
  SamplerMinMagFilter magFilter = SamplerMinMagFilter::Nearest;
  /**
   * @brief The filtering option to use between mipmap levels.
   */
  SamplerMipFilter mipFilter = SamplerMipFilter::Disabled;
  /**
   * @brief The sampling address mode to use for the U texture coordinate.
   */
  SamplerAddressMode addressModeU = SamplerAddressMode::Repeat;
  /**
   * @brief The sampling address mode to use for the V texture coordinate.
   */
  SamplerAddressMode addressModeV = SamplerAddressMode::Repeat;
  /**
   * @brief The sampling address mode to use for the W texture coordinate.
   */
  SamplerAddressMode addressModeW = SamplerAddressMode::Repeat;
  /**
   * @brief The depth comparison to use when comparing samples from a depth texture.
   */
  CompareFunction depthCompareFunction = CompareFunction::LessEqual;
  /**
   * @brief The minimum mipmap level to use when sampling a texture.
   *
   * The valid range is [0, 15]
   */
  uint8_t mipLodMin = 0;
  /**
   * @brief The maximum mipmap level to use when sampling a texture.
   *
   * The valid range is [mipLodMin, 15]
   */
  uint8_t mipLodMax = 15;
  /**
   * @brief The maximum number of samples that can be taken from a texture to improve the quality of
   * the sampled result.
   *
   * The valid range is [1, 16]
   */
  uint8_t maxAnisotropic = 1;
  /**
   * @brief Whether or not depth comparison is enabled.
   *
   * If true, the depthCompareFunction will be configured in the sampler state.
   * If false, the depthCompareFunction will be disabled.
   */
  bool depthCompareEnabled = false;

  /**
   * @brief A user-readable debug name associated with this sampler.
   *
   */
  std::string debugName;

  TextureFormat yuvFormat = TextureFormat::Invalid;

  /**
   * @brief Creates a new SamplerStateDesc instance set up for linearly interpolating within mipmap
   * level 0.
   *
   * minFilter and magFilter set to SamplerMinMagFilter::Linear.
   * mipFilter is set to SamplerMipFilter::Disabled.
   * All other parameters have their default value.
   */
  static SamplerStateDesc newLinear() {
    return SamplerStateDesc{
        .minFilter = SamplerMinMagFilter::Linear,
        .magFilter = SamplerMinMagFilter::Linear,
        .mipFilter = SamplerMipFilter::Disabled,
        .debugName = "newLinear()",
    };
  }

  /**
   * @brief Creates a new SamplerStateDesc instance set up for linearly interpolating within and
   * between mipmap levels.
   *
   * minFilter and magFilter set to SamplerMinMagFilter::Linear.
   * mipFilter is set to SamplerMipFilter::Linear.
   * All other parameters have their default value.
   */
  static SamplerStateDesc newLinearMipmapped() {
    return SamplerStateDesc{
        .minFilter = SamplerMinMagFilter::Linear,
        .magFilter = SamplerMinMagFilter::Linear,
        .mipFilter = SamplerMipFilter::Linear,
        .debugName = "newLinearMipmapped()",
    };
  }

  /**
   * @brief Creates a new SamplerStateDesc instance set up for YUV conversion
   */
  static SamplerStateDesc newYUV(TextureFormat yuvFormat, const char* debugName) {
    return SamplerStateDesc{
        .minFilter = SamplerMinMagFilter::Linear,
        .magFilter = SamplerMinMagFilter::Linear,
        .mipFilter = SamplerMipFilter::Disabled,
        .addressModeU = SamplerAddressMode::Clamp,
        .addressModeV = SamplerAddressMode::Clamp,
        .addressModeW = SamplerAddressMode::Clamp,
        .debugName = debugName,
        .yuvFormat = yuvFormat,
    };
  }

  /**
   * @brief Compares two SamplerStateDesc objects for equality.
   *
   * Returns true if all descriptor properties are identical.
   */
  bool operator==(const SamplerStateDesc& rhs) const;
  /**
   * @brief Compares two SamplerStateDesc objects for inequality.
   *
   * Returns true if any descriptor properties differ.
   */
  bool operator!=(const SamplerStateDesc& rhs) const;
};

/**
 * @brief A texture sampling configuration.
 *
 * This interface is the backend agnostic representation for a texture sampling configuration.
 * To create an instance, populate a SamplerStateDesc and call IDevice.createSamplerState.
 * To use an instance in a render pass for a texture, call IRenderCommandEncoder.bindSamplerState.
 */
class ISamplerState : public ITrackedResource<ISamplerState> {
 protected:
  ISamplerState() = default;

 public:
  ~ISamplerState() override = default;

  /**
   * @brief Returns whether this sampler state is configured for YUV color space conversion.
   * Required for Vulkan to take proper care of immutable samplers
   *
   * @return true if the sampler is configured for YUV format, false otherwise.
   */
  [[nodiscard]] virtual bool isYUV() const noexcept = 0;
};

} // namespace igl

/**
 * @brief Provides a hash function for igl::SamplerStateDesc.
 */
namespace std {
template<>
struct hash<igl::SamplerStateDesc> {
  /**
   * @brief Computes a hash value for igl::SamplerStateDesc.
   *
   * The hash value is based on all properties in the igl::SamplerStateDesc;
   */
  size_t operator()(const igl::SamplerStateDesc& /*key*/) const;
};
} // namespace std
