/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @MARK:COVERAGE_EXCLUDE_FILE

#pragma once

#include "VertexData.h"

#include <IGLU/simdtypes/SimdTypes.h>
#include <memory>

namespace iglu::vertexdata {

// [Convention] UV origin is bottom left and +Y points up.

struct VertexPosUv {
  simdtypes::float3 position;
  simdtypes::float2 uv;
};

/// Simple XY-aligned quad.
namespace Quad {

/// Descriptor matching the VertexPosUv type used in 'create'.
igl::VertexInputStateDesc inputStateDesc();

std::shared_ptr<VertexData> create(igl::IDevice& device,
                                   simdtypes::float2 posMin,
                                   simdtypes::float2 posMax,
                                   simdtypes::float2 uvMin,
                                   simdtypes::float2 uvMax);

} // namespace Quad

/// Simple XY-aligned quad.
///
/// Use RenderToTextureQuad instead of Quad if you're rendering to a texture and your results
/// are flipped on some graphics backends. Use it in one of two ways:
/// 1. When rendering into a texture in a **full screen** pass
/// 2. In draw calls where **all** the inputs to your shader program are color attachments
///
/// An in-depth explanation of the problem, solution and limitations can be found in the
/// implementation file.
namespace RenderToTextureQuad {

/// Descriptor matching the VertexPosUv type used in 'create'.
igl::VertexInputStateDesc inputStateDesc();

std::shared_ptr<VertexData> create(igl::IDevice& device,
                                   simdtypes::float2 posMin,
                                   simdtypes::float2 posMax,
                                   simdtypes::float2 uvMin,
                                   simdtypes::float2 uvMax);

} // namespace RenderToTextureQuad

} // namespace iglu::vertexdata
