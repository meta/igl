/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @MARK:COVERAGE_EXCLUDE_FILE

#include "ParametricVertexData.h"

#include <array>
#include <type_traits>
#include <igl/Macros.h>

static_assert(std::is_trivially_copyable_v<iglu::vertexdata::VertexPosUv>);

namespace iglu::vertexdata {

namespace {
// Assumption: <name, location> for OpenGL and Metal, respectively
const std::pair<const char*, int> kSAttrPosition("a_position", 0);
const std::pair<const char*, int> kSAttrUv("a_uv", 1);
} // namespace

namespace Quad {

igl::VertexInputStateDesc inputStateDesc() {
  IGL_PROFILER_FUNCTION();
  const igl::VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes = {{.bufferIndex = 0,
                      .format = igl::VertexAttributeFormat::Float3,
                      .offset = offsetof(VertexPosUv, position),
                      .name = kSAttrPosition.first,
                      .location = kSAttrPosition.second},
                     {.bufferIndex = 0,
                      .format = igl::VertexAttributeFormat::Float2,
                      .offset = offsetof(VertexPosUv, uv),
                      .name = kSAttrUv.first,
                      .location = kSAttrUv.second}},
      .numInputBindings = 1,
      .inputBindings = {{.stride = sizeof(VertexPosUv)}},
  };
  return inputDesc;
}

std::shared_ptr<VertexData> create(igl::IDevice& device,
                                   simdtypes::float2 posMin,
                                   simdtypes::float2 posMax,
                                   simdtypes::float2 uvMin,
                                   simdtypes::float2 uvMax) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // - UV origin: bottom left
  // - Vertex layout:
  // 0 -- 2
  // |    |
  // |    |
  // 1 -- 3
  const std::array vertexData{
      VertexPosUv{.position = {posMin[0], posMax[1], 0.0}, .uv = {uvMin[0], uvMax[1]}},
      VertexPosUv{.position = {posMin[0], posMin[1], 0.0}, .uv = {uvMin[0], uvMin[1]}},
      VertexPosUv{.position = {posMax[0], posMax[1], 0.0}, .uv = {uvMax[0], uvMax[1]}},
      VertexPosUv{.position = {posMax[0], posMin[1], 0.0}, .uv = {uvMax[0], uvMin[1]}},
  };
  const std::array indexData{uint16_t{0}, uint16_t{1}, uint16_t{2}, uint16_t{3}};

  const igl::BufferDesc vbDesc{.type = igl::BufferDesc::BufferTypeBits::Vertex,
                               .data = vertexData.data(),
                               .length = sizeof(VertexPosUv) * vertexData.size()};
  const igl::BufferDesc ibDesc{.type = igl::BufferDesc::BufferTypeBits::Index,
                               .data = indexData.data(),
                               .length = sizeof(uint16_t) * indexData.size()};

  const igl::VertexInputStateDesc inputDesc = inputStateDesc();
  // Per IGL Error Handling rule #24, every resource creation call must pass a
  // Result* and check it; passing nullptr silently swallows errors.
  igl::Result result;
  const std::shared_ptr<igl::IVertexInputState> vertexInput =
      device.createVertexInputState(inputDesc, &result);
  IGL_DEBUG_ASSERT(result.isOk(), "createVertexInputState() failed: %s", result.message.c_str());
  IGL_DEBUG_ASSERT(vertexInput != nullptr);

  const PrimitiveDesc primitiveDesc{.numEntries = sizeof(indexData) / sizeof(indexData[0])};

  auto vb = device.createBuffer(vbDesc, &result);
  IGL_DEBUG_ASSERT(result.isOk(), "createBuffer(vertex) failed: %s", result.message.c_str());
  IGL_DEBUG_ASSERT(vb != nullptr);
  auto ib = device.createBuffer(ibDesc, &result);
  IGL_DEBUG_ASSERT(result.isOk(), "createBuffer(index) failed: %s", result.message.c_str());
  IGL_DEBUG_ASSERT(ib != nullptr);

  std::shared_ptr<VertexData> vertData =
      std::make_shared<VertexData>(vertexInput,
                                   std::move(vb),
                                   std::move(ib),
                                   igl::IndexFormat::UInt16,
                                   primitiveDesc,
                                   igl::PrimitiveType::TriangleStrip);
  return vertData;
}

} // namespace Quad

namespace RenderToTextureQuad {

igl::VertexInputStateDesc inputStateDesc() {
  return Quad::inputStateDesc();
}

std::shared_ptr<VertexData> create(igl::IDevice& device,
                                   simdtypes::float2 posMin,
                                   simdtypes::float2 posMax,
                                   simdtypes::float2 uvMin,
                                   simdtypes::float2 uvMax) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  simdtypes::float2 uvMinAdjusted = uvMin;
  simdtypes::float2 uvMaxAdjusted = uvMax;

  // Here's how to think about the conventions that led to this workaround.
  //
  // Summary of conventions:
  // - In OpenGL, all origins (texture, framebuffer, clip) are the bottom left corner.
  // - In Metal, texture and framebuffer origins are the top left corner.
  // - The conventions in this library follow OpenGL. For example:
  //   - This file creates VertexData with UV origin at the bottom left.
  //   - The first pixel in a texture is expected to be the bottom left.
  //   - ForwardRenderPass assumes the viewport (framebuffer space) origin is the bottom left.
  // - We are forced to have our own conventions because the graphics APIs have their own
  //   conventions that aren't compatible with each other. Furthermore, we must correct for them.
  //
  // Correcting for discrepancies across graphics APIs:
  // - Although we can modify texture content, texture coordinates, viewport and clip space at
  //   will to handle discrepancies across graphics APIs, none provide a way to alter the origin
  //   of a framebuffer.
  // - When a framebuffer color attachment is used a shader program input, things break. Our
  //   convention is to use bottom left origin for texture data, which matches the origin of OpenGL
  //   framebuffer; all good. However, in Metal, the origin of framebuffer attachments is top left,
  //   so our color attachment texture is flipped in relation to textures loaded from images.
  // - If we followed Metal's coordinate conventions instead, we'd have the same problem but in
  //   OpenGL. This problem can't be avoided because the APIs don't allow us to compensate for
  //   our conventions, no matter what they are.
  //
  // This workaround doesn't cover use cases like:
  // - Imported 3D meshes
  // - Shaders that use a combination of color attachments and image-based textures as input
  // A general solution to this problem would involve being able to tell shader code about the
  // orientation of every input texture and a strict set of conventions in shader code to ensure
  // texture sampling accounts for that information.
  //
  // Some external resources I found useful for understanding this issue:
  // - https://veldrid.dev/articles/backend-differences.html
  // - http://hacksoflife.blogspot.com/2019/04/keeping-blue-side-up-coordinate.html
  //
  if (device.getBackendType() == igl::BackendType::Metal) {
    uvMinAdjusted[1] = 1.0f - uvMinAdjusted[1];
    uvMaxAdjusted[1] = 1.0f - uvMaxAdjusted[1];
  }

  return Quad::create(device, posMin, posMax, uvMinAdjusted, uvMaxAdjusted);
}

} // namespace RenderToTextureQuad

} // namespace iglu::vertexdata
