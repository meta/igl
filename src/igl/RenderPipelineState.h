/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <igl/Common.h>
#include <igl/NameHandle.h>
#include <igl/RenderPipelineReflection.h>
#include <igl/Shader.h>
#include <igl/Texture.h>

namespace igl {

class IVertexInputState;

/**
 * @brief BlendOp determines how to combine and weight the source and destination fragment values.
 * Some blend operations multiply the source values by
 * a source blend factor (SBF), multiply the destination values by a destination blend factor (DBF),
 * and then combine the results using addition or subtraction. Other blend operations use either a
 * minimum or maximum function to determine the result.
 *
 * Add : Add portions of both source and destination pixel values.
 * Subtract : Subtract a portion of the destination pixel values from a portion of the source.
 * ReverseSubtract : Subtract a portion of the source values from a portion of the destination pixel
 * values.
 * Min : Minimum of the source and destination pixel values.
 * Max : Maximum of the source and
 * destination pixel values.
 */
enum class BlendOp : uint8_t { Add = 0, Subtract, ReverseSubtract, Min, Max };

/**
 * @brief The source and destination blend factors are often needed to complete specification of a
 * blend operation. In most cases, the blend factor for both RGB values (F(rgb)) and alpha values
 * (F(a)) are similar to one another, but in some cases, such as SrcAlphaSaturated, the blend factor
 * is slightly different. Four blend factors (BlendColor, OneMinusBlendColor, BlendAlpha, and
 * OneMinusBlendAlpha) refer to a constant blend color value that is set by the
 * setBlendColor(red:green:blue:alpha:) method of RenderCommandEncoder.
 *
 * Zero : Blend factor of zero.
 * One : Blend factor of one.
 * SrcColor : Blend factor of source values.
 * OneMinusSrcColor : Blend factor of one minus source values.
 * SrcAlpha : Blend factor of source alpha
 * OneMinusSrcAlpha : Blend factor of one minus source alpha.
 * DstColor : Blend factor of destination values.
 * OneMinusDstColor : Blend factor of one minus destination values.
 * DstAlpha : Blend factor of destination alpha.
 * OneMinusDstAlpha : Blend factor of one minus destination alpha.
 * SrcAlphaSaturated : Blend factor of the minimum of
 *                    either source alpha or one minus destination alpha.
 * BlendColor : Blend factor of RGB values.
 * OneMinusBlendColor : Blend factor of one minus RGB values.
 * BlendAlpha : Blend factor of alpha value.
 * OneMinusBlendAlpha : Blend factor of one minus alpha value.
 * Src1Color : Blend factor of source values. This option supports dual-source blending
 *             and reads from the second color output of the fragment function.
 * OneMinusSrc1Color : Blend factor of one minus source values. This option supports dual-source
 *                     blending and reads from the second color output of the fragment function.
 * Src1Alpha : Blend factor of source alpha. This option supports dual-source blending
 *             and reads from the second color output of the fragment function.
 * OneMinusSrc1Alpha : Blend factor of one minus source alpha. This option supports dual-source
 *                     blending and reads from the second color output of the fragment function.
 */
enum class BlendFactor : uint8_t {
  Zero = 0,
  One,
  SrcColor,
  OneMinusSrcColor,
  SrcAlpha,
  OneMinusSrcAlpha,
  DstColor,
  OneMinusDstColor,
  DstAlpha,
  OneMinusDstAlpha,
  SrcAlphaSaturated,
  BlendColor,
  OneMinusBlendColor,
  BlendAlpha,
  OneMinusBlendAlpha,
  Src1Color,
  OneMinusSrc1Color,
  Src1Alpha,
  OneMinusSrc1Alpha
};

/*
 * @brief Values used to specify a mask to permit or restrict writing to color channels of a color
 * value. The values red, green, blue, and alpha select one color channel each, and they can be
 * bitwise combined.
 */
using ColorWriteMask = uint8_t;
enum ColorWriteBits : uint8_t {
  kColorWriteBitsDisabled = 0,
  kColorWriteBitsRed = 1 << 0,
  kColorWriteBitsGreen = 1 << 1,
  kColorWriteBitsBlue = 1 << 2,
  kColorWriteBitsAlpha = 1 << 3,
  kColorWriteBitsAll =
      kColorWriteBitsRed | kColorWriteBitsGreen | kColorWriteBitsBlue | kColorWriteBitsAlpha,
};

/*
 * @brief Control polygon rasterization modes
 * Fill : Polygons are rendered using standard polygon rasterization rules
 * Line : Polygon edges are drawn as line segments
 */
enum class PolygonFillMode : uint8_t {
  Fill = 0,
  Line = 1,
};

/*
 * @brief An argument of options you pass to a device to get a render pipeline state object.
 */
struct RenderPipelineDesc {
  struct TargetDesc {
    /*
     * @brief Description of render pipeline's color render target
     */
    struct ColorAttachment {
      TextureFormat textureFormat = TextureFormat::Invalid;
      /*
       * @brief Identify which color channels are blended.
       */
      ColorWriteMask colorWriteMask = kColorWriteBitsAll;
      bool blendEnabled = false;
      /*
       * @brief Assign the blend operations for RGB and alpha pixel data.
       */
      BlendOp rgbBlendOp = BlendOp::Add;
      BlendOp alphaBlendOp = BlendOp::Add;
      /*
       * @brief Assign the source and destination blend factors.
       */
      BlendFactor srcRGBBlendFactor = BlendFactor::One;
      BlendFactor srcAlphaBlendFactor = BlendFactor::One;
      BlendFactor dstRGBBlendFactor = BlendFactor::Zero;
      BlendFactor dstAlphaBlendFactor = BlendFactor::Zero;
      bool operator==(const ColorAttachment& other) const = default;
      bool operator!=(const ColorAttachment& other) const = default;
    };

    /*
     * @brief Array of attachments that store color data
     */
    std::vector<ColorAttachment> colorAttachments;
    /*
     * @brief Pixel format of the attachment that stores depth data
     */
    TextureFormat depthAttachmentFormat = TextureFormat::Invalid;
    /*
     * @brief Pixel format of the attachment that stores stencil data
     */
    TextureFormat stencilAttachmentFormat = TextureFormat::Invalid;
    bool operator==(const TargetDesc& other) const = default;
    bool operator!=(const TargetDesc& other) const = default;
  };

  /*
   * @brief Describes the primitive topology for this graphics pipeline
   */
  PrimitiveType topology = PrimitiveType::Triangle;

  /*
   * @brief Describes the organization of per-vertex input data passed to a vertex shader function
   */
  std::shared_ptr<IVertexInputState> vertexInputState;

  /*
   * @brief Describes the vertex and fragment functions
   */
  std::shared_ptr<IShaderStages> shaderStages;

  TargetDesc targetDesc;
  CullMode cullMode = CullMode::Disabled;
  WindingMode frontFaceWinding = WindingMode::CounterClockwise;
  PolygonFillMode polygonFillMode = PolygonFillMode::Fill;

  /*
   * GL Only: Mapping of Texture Unit <-> Sampler Name
   * Texture unit should be < IGL_TEXTURE_SAMPLERS_MAX
   */
  std::unordered_map<size_t, NameHandle> vertexUnitSamplerMap;
  std::unordered_map<size_t, NameHandle> fragmentUnitSamplerMap;

  /*
   * GL Only:
   * Maps block binding points to vectors of (BlockName, blockInstanceName) pairs of all uniform
   * blocks that should use this binding point.
   * Uniform Block Binding Point should be < IGL_UNIFORM_BLOCKS_BINDING_MAX.
   * This should only be populated if explicit binding is not supported or used.
   */
  std::unordered_map<size_t, std::vector<std::pair<NameHandle, NameHandle>>> uniformBlockBindingMap;

  uint32_t sampleCount = 1u; // MSAA

  // Vulkan only: specify if buffer binding locations correspond to Vulkan dynamic buffers
  uint32_t isDynamicBufferMask = 0; // one bit per each buffer binding slot

  // Vulkan only: immutable samplers per each binding slot (for example, Ycbcr conversion etc)
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  std::shared_ptr<ISamplerState> immutableSamplers[IGL_TEXTURE_SAMPLERS_MAX] = {};

  /*
   * @brief When enabled, the GPU computes a fragment's coverage from its alpha value,
   * AND-ed with the geometric sample mask. This is a fixed-function alternative to
   * shader-side `discard` for alpha-tested geometry. It does not break Early-Z because
   * the decision happens in hardware after the fragment shader, not inside it.
   *
   * Requires `sampleCount > 1` (MSAA must be enabled) to take effect; otherwise the
   * driver silently ignores it.
   */
  bool alphaToCoverageEnabled = false;

  NameHandle debugName;

  bool operator==(const RenderPipelineDesc& other) const;
  bool operator!=(const RenderPipelineDesc& other) const;
};

class IRenderPipelineState {
 public:
  explicit IRenderPipelineState(RenderPipelineDesc desc) : desc_(std::move(desc)) {}
  virtual ~IRenderPipelineState() = default;

  [[nodiscard]] virtual std::shared_ptr<IRenderPipelineReflection> renderPipelineReflection() = 0;
  virtual void setRenderPipelineReflection(
      const IRenderPipelineReflection& renderPipelineReflection) = 0;

  [[nodiscard]] virtual int getIndexByName(const NameHandle& /* name */,
                                           ShaderStage /* stage */) const {
    return -1;
  }

  [[nodiscard]] virtual int getIndexByName(const std::string& /* name */,
                                           ShaderStage /* stage */) const {
    return -1;
  }

  [[nodiscard]] const RenderPipelineDesc& getRenderPipelineDesc() const {
    return desc_;
  }

 protected:
  const RenderPipelineDesc desc_{};
};

} // namespace igl

/// Hashing function declarations
///
namespace std {

template<>
struct hash<igl::RenderPipelineDesc> {
  size_t operator()(const igl::RenderPipelineDesc& /*key*/) const;
};

template<>
struct hash<igl::RenderPipelineDesc::TargetDesc> {
  size_t operator()(const igl::RenderPipelineDesc::TargetDesc& /*key*/) const;
};

template<>
struct hash<igl::RenderPipelineDesc::TargetDesc::ColorAttachment> {
  size_t operator()(const igl::RenderPipelineDesc::TargetDesc::ColorAttachment& /*key*/) const;
};

} // namespace std
