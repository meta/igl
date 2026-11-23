/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/openxr/XrCompositionQuad.h>

#include <igl/RenderPipelineState.h>

namespace igl::shell::openxr {
namespace {
#ifdef XR_FB_composition_layer_alpha_blend
inline XrBlendFactorFB iglToOpenXR(BlendFactor factor) noexcept {
  switch (factor) {
  case BlendFactor::Zero:
    return XR_BLEND_FACTOR_ZERO_FB;
  case BlendFactor::One:
    return XR_BLEND_FACTOR_ONE_FB;
  case BlendFactor::SrcAlpha:
    return XR_BLEND_FACTOR_SRC_ALPHA_FB;
  case BlendFactor::OneMinusSrcAlpha:
    return XR_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA_FB;
  case BlendFactor::DstAlpha:
    return XR_BLEND_FACTOR_DST_ALPHA_FB;
  case BlendFactor::OneMinusDstAlpha:
    return XR_BLEND_FACTOR_ONE_MINUS_DST_ALPHA_FB;
  default:
    IGL_DEBUG_ABORT("Not supported blend factor (%d)", static_cast<int>(factor));
    break;
  }
  return XR_BLEND_FACTOR_ZERO_FB;
}
#endif // XR_FB_composition_layer_alpha_blend
} // namespace

XrCompositionQuad::XrCompositionQuad(impl::XrAppImpl& appImpl,
                                     std::shared_ptr<Platform> platform,
                                     XrSession session,
                                     bool useSinglePassStereo,
                                     bool isAlphaBlendCompositionSupported,
                                     const QuadLayerInfo& info) noexcept :
  XrComposition(appImpl, std::move(platform), session, useSinglePassStereo),
  isAlphaBlendCompositionSupported_(isAlphaBlendCompositionSupported) {
  updateQuadLayerInfo(info);
}

void XrCompositionQuad::updateQuadLayerInfo(const QuadLayerInfo& info) noexcept {
  info_ = info;

#ifdef XR_FB_composition_layer_alpha_blend
  customBlending_ = {.type = XR_TYPE_COMPOSITION_LAYER_ALPHA_BLEND_FB,
                     .next = nullptr,
                     .srcFactorColor = iglToOpenXR(info.customSrcRGBBlendFactor),
                     .dstFactorColor = iglToOpenXR(info.customDstRGBBlendFactor),
                     .srcFactorAlpha = iglToOpenXR(info.customSrcAlphaBlendFactor),
                     .dstFactorAlpha = iglToOpenXR(info.customDstAlphaBlendFactor)};
#endif
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void XrCompositionQuad::doComposition(
    const DepthParams& /* depthParams */,
    const std::array<XrView, kNumViews>& /* views */,
    const std::array<XrPosef, kNumViews>& /* viewStagePoses */,
    XrSpace currentSpace,
    XrCompositionLayerFlags compositionFlags,
    std::vector<const XrCompositionLayerBaseHeader*>& layers) noexcept {
  compositionFlags |= (info_.blendMode == LayerBlendMode::AlphaBlend)
                          ? XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT
                          : 0;

  for (uint8_t view = 0; view < kNumViews; ++view) {
    const auto subImageIndex = useSinglePassStereo_ ? static_cast<uint32_t>(view) : 0u;
    const auto swapchainProviderIndex = useSinglePassStereo_ ? 0u : static_cast<uint32_t>(view);

    const XrRect2Di imageRect = {
        .offset = {.x = 0, .y = 0},
        .extent =
            {
                .width = static_cast<int32_t>(swapchainImageInfo_[view].imageWidth),
                .height = static_cast<int32_t>(swapchainImageInfo_[view].imageHeight),
            },
    };

    quadLayers_[view] = {
        .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
#ifdef XR_FB_composition_layer_alpha_blend
        .next = (isAlphaBlendCompositionSupported_ && info_.blendMode == LayerBlendMode::Custom)
                    ? &customBlending_
                    : nullptr,
#else
        .next = nullptr,
#endif // XR_FB_composition_layer_alpha_blend
        .layerFlags = compositionFlags,
        .space = currentSpace,
        .eyeVisibility = (view == 0) ? XR_EYE_VISIBILITY_LEFT : XR_EYE_VISIBILITY_RIGHT,
        .subImage =
            {
                .swapchain = swapchainProviders_[swapchainProviderIndex]->colorSwapchain(),
                .imageRect = imageRect,
                .imageArrayIndex = subImageIndex,
            },
        .pose =
            {
                .orientation = {.x = 0.f, .y = 0.f, .z = 0.f, .w = 1.f},
                .position = {.x = info_.position.x, .y = info_.position.y, .z = info_.position.z},
            },
        .size = {.width = info_.size.x, .height = info_.size.y},
    };

    layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quadLayers_[view]));
  }
}

} // namespace igl::shell::openxr
