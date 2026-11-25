/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/SubpassDeferredSession.h>

#include <unordered_map>
#include <vector>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>
#include <igl/opengl/Version.h> // IWYU pragma: keep

namespace igl::shell {

namespace {

// Four rectangles of two triangles each
constexpr size_t kSceneVertexCount = 24;
constexpr size_t kFullscreenTriangleVertexCount = 3;

constexpr TextureFormat kGBufferFormat = TextureFormat::RGBA_UNorm8;
constexpr TextureDesc::TextureUsage kSubpassGBufferUsage =
    TextureDesc::TextureUsageBits::Attachment | TextureDesc::TextureUsageBits::InputAttachment;
constexpr TextureDesc::TextureUsage kSampledGBufferUsage =
    TextureDesc::TextureUsageBits::Attachment | TextureDesc::TextureUsageBits::Sampled;
constexpr std::array<const char*, 3> kGBufferNames = {"SubpassDeferredSession: albedo",
                                                      "SubpassDeferredSession: normal",
                                                      "SubpassDeferredSession: material"};

// ---------------------------------------------------------------------------
// The scene and the lighting model
//
// Every shader draws the same four rectangles from the vertex index alone, six vertices per
// rectangle, with no vertex buffer. Every edge lies on a multiple of 0.5 in normalized device
// coordinates, which falls on a pixel boundary if the width and the height are multiples of 4, as
// they are in the tests and the screenshots. At other sizes, the rows and columns along the edges
// may differ between backends. The layout is symmetric about y = 0, so the image does not depend
// on the Y direction of the backend.
//
//   rectangle  x range       y range       albedo   normal  material
//   A          [-1.0, -0.5]  [-0.5, 0.5]   red      +Z      0
//   B          [-0.5,  0.0]  [-1.0, 1.0]   green    +X      0
//   C          [ 0.0,  0.5]  [-0.5, 0.5]   blue     +Z      0
//   D          [ 0.5,  1.0]  [-1.0, 1.0]   white    +Z      1
//
// The G-buffer holds the albedo (alpha 1 marks geometry), the normal and the material (red is 1 for
// ambient light only). Every value is 0 or 1, so the RGBA8 G-buffer stores it exactly. The lighting
// pass leaves the background alone and lights the rest with a light along +Z: a fragment gets full
// light, or 64/255 of it when its material is ambient only or its normal faces away from the light.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GLSL shared by Vulkan and OpenGL
// ---------------------------------------------------------------------------

constexpr const char* kGlslScene = R"(
const vec4 kRects[4] = vec4[4](vec4(-1.0, -0.5, -0.5, 0.5),
                               vec4(-0.5, -1.0, 0.0, 1.0),
                               vec4(0.0, -0.5, 0.5, 0.5),
                               vec4(0.5, -1.0, 1.0, 1.0));
const vec3 kAlbedos[4] = vec3[4](vec3(1.0, 0.0, 0.0),
                                 vec3(0.0, 1.0, 0.0),
                                 vec3(0.0, 0.0, 1.0),
                                 vec3(1.0, 1.0, 1.0));
const vec3 kNormals[4] = vec3[4](vec3(0.0, 0.0, 1.0),
                                 vec3(1.0, 0.0, 0.0),
                                 vec3(0.0, 0.0, 1.0),
                                 vec3(0.0, 0.0, 1.0));
const float kMaterials[4] = float[4](0.0, 0.0, 0.0, 1.0);
const vec2 kCorners[6] = vec2[6](vec2(0.0, 0.0),
                                 vec2(1.0, 0.0),
                                 vec2(0.0, 1.0),
                                 vec2(0.0, 1.0),
                                 vec2(1.0, 0.0),
                                 vec2(1.0, 1.0));
)";

// The outputs of a G-buffer fragment shader: color attachments 1 to 3 in a subpass, where
// attachment 0 is the image, and color attachments 0 to 2 otherwise
constexpr const char* kGlslGBufferSubpassOutputs = R"(
layout(location = 1) out vec4 outAlbedo;
layout(location = 2) out vec4 outNormal;
layout(location = 3) out vec4 outMaterial;
)";

constexpr const char* kGlslGBufferOutputs = R"(
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;
)";

constexpr const char* kGlslGBufferMain = R"(
void main() {
  outAlbedo = vec4(vAlbedo, 1.0);
  outNormal = vec4(vNormal, 1.0);
  outMaterial = vec4(vMaterial, 0.0, 0.0, 1.0);
}
)";

// The lighting fragment shader reads the G-buffer through loadAlbedo(), loadNormal() and
// loadMaterial(), which differ between input attachments and sampled textures
constexpr const char* kGlslLightingMain = R"(
layout(location = 0) out vec4 outColor;

void main() {
  vec4 albedo = loadAlbedo();
  if (albedo.a < 0.5) {
    discard;
  }
  const vec3 lightDirection = vec3(0.0, 0.0, 1.0);
  float factor = (loadMaterial().r > 0.5 || dot(loadNormal().xyz, lightDirection) < 0.5)
                     ? 64.0 / 255.0
                     : 1.0;
  outColor = vec4(albedo.rgb * factor, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Vulkan
// ---------------------------------------------------------------------------

std::string getVulkanGBufferVertexShader() {
  return std::string("#version 460\n") + kGlslScene + R"(
layout(location = 0) flat out vec3 vAlbedo;
layout(location = 1) flat out vec3 vNormal;
layout(location = 2) flat out float vMaterial;

void main() {
  int rect = gl_VertexIndex / 6;
  vec4 bounds = kRects[rect];
  gl_Position = vec4(mix(bounds.xy, bounds.zw, kCorners[gl_VertexIndex % 6]), 0.0, 1.0);
  vAlbedo = kAlbedos[rect];
  vNormal = kNormals[rect];
  vMaterial = kMaterials[rect];
}
)";
}

std::string getVulkanGBufferFragmentShader(bool subpass) {
  return std::string(R"(#version 460
layout(location = 0) flat in vec3 vAlbedo;
layout(location = 1) flat in vec3 vNormal;
layout(location = 2) flat in float vMaterial;
)") + (subpass ? kGlslGBufferSubpassOutputs : kGlslGBufferOutputs) +
         kGlslGBufferMain;
}

constexpr const char* kVulkanLightingVertexShader = R"(#version 460
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)";

// Input attachment i is color attachment i, at binding 16 + i of set 2
constexpr const char* kVulkanSubpassInputs = R"(#version 460
layout(input_attachment_index = 1, set = 2, binding = 17) uniform subpassInput inAlbedo;
layout(input_attachment_index = 2, set = 2, binding = 18) uniform subpassInput inNormal;
layout(input_attachment_index = 3, set = 2, binding = 19) uniform subpassInput inMaterial;

vec4 loadAlbedo() {
  return subpassLoad(inAlbedo);
}
vec4 loadNormal() {
  return subpassLoad(inNormal);
}
vec4 loadMaterial() {
  return subpassLoad(inMaterial);
}
)";

// Texture unit i is binding i of set 0. The G-buffer is read at the pixel of the fragment.
constexpr const char* kVulkanSampledInputs = R"(#version 460
layout(set = 0, binding = 0) uniform sampler2D uAlbedo;
layout(set = 0, binding = 1) uniform sampler2D uNormal;
layout(set = 0, binding = 2) uniform sampler2D uMaterial;

vec4 loadAlbedo() {
  return texelFetch(uAlbedo, ivec2(gl_FragCoord.xy), 0);
}
vec4 loadNormal() {
  return texelFetch(uNormal, ivec2(gl_FragCoord.xy), 0);
}
vec4 loadMaterial() {
  return texelFetch(uMaterial, ivec2(gl_FragCoord.xy), 0);
}
)";

std::string getVulkanLightingFragmentShader(bool subpass) {
  return std::string(subpass ? kVulkanSubpassInputs : kVulkanSampledInputs) + kGlslLightingMain;
}

// ---------------------------------------------------------------------------
// OpenGL
// ---------------------------------------------------------------------------

// The subpass shaders are GLSL ES 3.00 only. The others use the newest version of the device.
std::string getOpenGLVersionLine([[maybe_unused]] const IDevice& device, bool subpass) {
  if (subpass) {
    return "#version 300 es\n";
  }
#if IGL_BACKEND_OPENGL
  return igl::opengl::getStringFromShaderVersion(device.getShaderVersion()) + "\n";
#else
  return "";
#endif // IGL_BACKEND_OPENGL
}

std::string getOpenGLGBufferVertexShader(const IDevice& device, bool subpass) {
  return getOpenGLVersionLine(device, subpass) + kGlslScene + R"(
flat out vec3 vAlbedo;
flat out vec3 vNormal;
flat out float vMaterial;

void main() {
  int rect = gl_VertexID / 6;
  vec4 bounds = kRects[rect];
  gl_Position = vec4(mix(bounds.xy, bounds.zw, kCorners[gl_VertexID % 6]), 0.0, 1.0);
  vAlbedo = kAlbedos[rect];
  vNormal = kNormals[rect];
  vMaterial = kMaterials[rect];
}
)";
}

std::string getOpenGLGBufferFragmentShader(const IDevice& device, bool subpass) {
  return getOpenGLVersionLine(device, subpass) + R"(
precision highp float;
flat in vec3 vAlbedo;
flat in vec3 vNormal;
flat in float vMaterial;
)" + (subpass ? kGlslGBufferSubpassOutputs : kGlslGBufferOutputs) +
         kGlslGBufferMain;
}

std::string getOpenGLLightingVertexShader(const IDevice& device, bool subpass) {
  return getOpenGLVersionLine(device, subpass) + R"(
void main() {
  vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)";
}

// Reads the G-buffer through framebuffer fetch: color attachments 1 to 3 are inout fragment outputs
constexpr const char* kOpenGLSubpassInputs = R"(#version 300 es
#extension GL_EXT_shader_framebuffer_fetch : require
precision highp float;
layout(location = 1) inout vec4 inAlbedo;
layout(location = 2) inout vec4 inNormal;
layout(location = 3) inout vec4 inMaterial;

vec4 loadAlbedo() {
  return inAlbedo;
}
vec4 loadNormal() {
  return inNormal;
}
vec4 loadMaterial() {
  return inMaterial;
}
)";

// The texture units are mapped to these sampler names by the fragmentUnitSamplerMap of the pipeline
constexpr const char* kOpenGLSampledInputs = R"(
precision highp float;
precision highp sampler2D;
uniform sampler2D uAlbedo;
uniform sampler2D uNormal;
uniform sampler2D uMaterial;

vec4 loadAlbedo() {
  return texelFetch(uAlbedo, ivec2(gl_FragCoord.xy), 0);
}
vec4 loadNormal() {
  return texelFetch(uNormal, ivec2(gl_FragCoord.xy), 0);
}
vec4 loadMaterial() {
  return texelFetch(uMaterial, ivec2(gl_FragCoord.xy), 0);
}
)";

std::string getOpenGLLightingFragmentShader(const IDevice& device, bool subpass) {
  return (subpass ? std::string(kOpenGLSubpassInputs)
                  : getOpenGLVersionLine(device, false) + kOpenGLSampledInputs) +
         kGlslLightingMain;
}

// ---------------------------------------------------------------------------
// Metal
// ---------------------------------------------------------------------------

constexpr const char* kMetalGBufferVertexShader = R"(
#include <metal_stdlib>
using namespace metal;

constant float4 kRects[4] = {float4(-1.0, -0.5, -0.5, 0.5),
                             float4(-0.5, -1.0, 0.0, 1.0),
                             float4(0.0, -0.5, 0.5, 0.5),
                             float4(0.5, -1.0, 1.0, 1.0)};
constant float3 kAlbedos[4] = {float3(1.0, 0.0, 0.0),
                               float3(0.0, 1.0, 0.0),
                               float3(0.0, 0.0, 1.0),
                               float3(1.0, 1.0, 1.0)};
constant float3 kNormals[4] = {float3(0.0, 0.0, 1.0),
                               float3(1.0, 0.0, 0.0),
                               float3(0.0, 0.0, 1.0),
                               float3(0.0, 0.0, 1.0)};
constant float kMaterials[4] = {0.0, 0.0, 0.0, 1.0};
constant float2 kCorners[6] = {float2(0.0, 0.0),
                               float2(1.0, 0.0),
                               float2(0.0, 1.0),
                               float2(0.0, 1.0),
                               float2(1.0, 0.0),
                               float2(1.0, 1.0)};

struct GBufferVertexOut {
  float4 position [[position]];
  float3 albedo [[flat]];
  float3 normal [[flat]];
  float material [[flat]];
};

vertex GBufferVertexOut gBufferVertexShader(uint vertexId [[vertex_id]]) {
  const uint rect = vertexId / 6;
  const float4 bounds = kRects[rect];
  GBufferVertexOut out;
  out.position = float4(mix(bounds.xy, bounds.zw, kCorners[vertexId % 6]), 0.0, 1.0);
  out.albedo = kAlbedos[rect];
  out.normal = kNormals[rect];
  out.material = kMaterials[rect];
  return out;
}
)";

// Metal does not mask writes, so a subpass function declares only the attachments it writes
constexpr const char* kMetalGBufferSubpassFragmentShader = R"(
struct GBufferOut {
  float4 albedo [[color(1)]];
  float4 normal [[color(2)]];
  float4 material [[color(3)]];
};

fragment GBufferOut gBufferFragmentShader(GBufferVertexOut in [[stage_in]]) {
  GBufferOut out;
  out.albedo = float4(in.albedo, 1.0);
  out.normal = float4(in.normal, 1.0);
  out.material = float4(in.material, 0.0, 0.0, 1.0);
  return out;
}
)";

constexpr const char* kMetalGBufferFragmentShader = R"(
struct GBufferOut {
  float4 albedo [[color(0)]];
  float4 normal [[color(1)]];
  float4 material [[color(2)]];
};

fragment GBufferOut gBufferFragmentShader(GBufferVertexOut in [[stage_in]]) {
  GBufferOut out;
  out.albedo = float4(in.albedo, 1.0);
  out.normal = float4(in.normal, 1.0);
  out.material = float4(in.material, 0.0, 0.0, 1.0);
  return out;
}
)";

constexpr const char* kMetalLightingVertexShader = R"(
#include <metal_stdlib>
using namespace metal;

struct LightingVertexOut {
  float4 position [[position]];
};

vertex LightingVertexOut lightingVertexShader(uint vertexId [[vertex_id]]) {
  const float2 uv = float2(float((vertexId << 1) & 2), float(vertexId & 2));
  LightingVertexOut out;
  out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
  return out;
}

float4 light(float4 albedo, float4 normal, float4 material) {
  const float3 lightDirection = float3(0.0, 0.0, 1.0);
  const float factor =
      (material.r > 0.5 || dot(normal.xyz, lightDirection) < 0.5) ? 64.0 / 255.0 : 1.0;
  return float4(albedo.rgb * factor, 1.0);
}
)";

// Color attachments 1 to 3 are fragment inputs
constexpr const char* kMetalLightingSubpassFragmentShader = R"(
fragment float4 lightingFragmentShader(float4 albedo [[color(1)]],
                                       float4 normal [[color(2)]],
                                       float4 material [[color(3)]]) {
  if (albedo.a < 0.5) {
    discard_fragment();
  }
  return light(albedo, normal, material);
}
)";

constexpr const char* kMetalLightingFragmentShader = R"(
fragment float4 lightingFragmentShader(LightingVertexOut in [[stage_in]],
                                       texture2d<float> albedoTexture [[texture(0)]],
                                       texture2d<float> normalTexture [[texture(1)]],
                                       texture2d<float> materialTexture [[texture(2)]]) {
  const uint2 pixel = uint2(in.position.xy);
  const float4 albedo = albedoTexture.read(pixel);
  if (albedo.a < 0.5) {
    discard_fragment();
  }
  return light(albedo, normalTexture.read(pixel), materialTexture.read(pixel));
}
)";

// ---------------------------------------------------------------------------
// D3D12: the fallback only
// ---------------------------------------------------------------------------

constexpr const char* kHlslGBufferVertexShader = R"(
static const float4 kRects[4] = {float4(-1.0, -0.5, -0.5, 0.5),
                                 float4(-0.5, -1.0, 0.0, 1.0),
                                 float4(0.0, -0.5, 0.5, 0.5),
                                 float4(0.5, -1.0, 1.0, 1.0)};
static const float3 kAlbedos[4] = {float3(1.0, 0.0, 0.0),
                                   float3(0.0, 1.0, 0.0),
                                   float3(0.0, 0.0, 1.0),
                                   float3(1.0, 1.0, 1.0)};
static const float3 kNormals[4] = {float3(0.0, 0.0, 1.0),
                                   float3(1.0, 0.0, 0.0),
                                   float3(0.0, 0.0, 1.0),
                                   float3(0.0, 0.0, 1.0)};
static const float kMaterials[4] = {0.0, 0.0, 0.0, 1.0};
static const float2 kCorners[6] = {float2(0.0, 0.0),
                                   float2(1.0, 0.0),
                                   float2(0.0, 1.0),
                                   float2(0.0, 1.0),
                                   float2(1.0, 0.0),
                                   float2(1.0, 1.0)};

struct VSOut {
  float4 position : SV_POSITION;
  nointerpolation float3 albedo : TEXCOORD0;
  nointerpolation float3 normal : TEXCOORD1;
  nointerpolation float material : TEXCOORD2;
};

VSOut main(uint vertexId : SV_VertexID) {
  const uint rect = vertexId / 6;
  const float4 bounds = kRects[rect];
  VSOut o;
  o.position = float4(lerp(bounds.xy, bounds.zw, kCorners[vertexId % 6]), 0.0, 1.0);
  o.albedo = kAlbedos[rect];
  o.normal = kNormals[rect];
  o.material = kMaterials[rect];
  return o;
}
)";

constexpr const char* kHlslGBufferPixelShader = R"(
struct PSIn {
  float4 position : SV_POSITION;
  nointerpolation float3 albedo : TEXCOORD0;
  nointerpolation float3 normal : TEXCOORD1;
  nointerpolation float material : TEXCOORD2;
};

struct PSOut {
  float4 albedo : SV_Target0;
  float4 normal : SV_Target1;
  float4 material : SV_Target2;
};

PSOut main(PSIn i) {
  PSOut o;
  o.albedo = float4(i.albedo, 1.0);
  o.normal = float4(i.normal, 1.0);
  o.material = float4(i.material, 0.0, 0.0, 1.0);
  return o;
}
)";

constexpr const char* kHlslLightingVertexShader = R"(
float4 main(uint vertexId : SV_VertexID) : SV_POSITION {
  const float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
  return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)";

constexpr const char* kHlslLightingPixelShader = R"(
Texture2D albedoTexture : register(t0);
Texture2D normalTexture : register(t1);
Texture2D materialTexture : register(t2);

float4 main(float4 position : SV_POSITION) : SV_Target {
  const int3 pixel = int3(int2(position.xy), 0);
  const float4 albedo = albedoTexture.Load(pixel);
  if (albedo.a < 0.5) {
    discard;
  }
  const float4 normal = normalTexture.Load(pixel);
  const float4 material = materialTexture.Load(pixel);
  const float3 lightDirection = float3(0.0, 0.0, 1.0);
  const float factor =
      (material.r > 0.5 || dot(normal.xyz, lightDirection) < 0.5) ? 64.0 / 255.0 : 1.0;
  return float4(albedo.rgb * factor, 1.0);
}
)";

// ---------------------------------------------------------------------------
// WebGPU: the fallback only
// ---------------------------------------------------------------------------

constexpr const char* kWgslGBufferShader = R"(
struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) @interpolate(flat) albedo : vec3f,
  @location(1) @interpolate(flat) normal : vec3f,
  @location(2) @interpolate(flat) material : f32,
};

@vertex
fn gBufferVertexShader(@builtin(vertex_index) vertexId : u32) -> VertexOut {
  var rects = array<vec4f, 4>(vec4f(-1.0, -0.5, -0.5, 0.5),
                              vec4f(-0.5, -1.0, 0.0, 1.0),
                              vec4f(0.0, -0.5, 0.5, 0.5),
                              vec4f(0.5, -1.0, 1.0, 1.0));
  var albedos = array<vec3f, 4>(vec3f(1.0, 0.0, 0.0),
                                vec3f(0.0, 1.0, 0.0),
                                vec3f(0.0, 0.0, 1.0),
                                vec3f(1.0, 1.0, 1.0));
  var normals = array<vec3f, 4>(vec3f(0.0, 0.0, 1.0),
                                vec3f(1.0, 0.0, 0.0),
                                vec3f(0.0, 0.0, 1.0),
                                vec3f(0.0, 0.0, 1.0));
  var materials = array<f32, 4>(0.0, 0.0, 0.0, 1.0);
  var corners = array<vec2f, 6>(vec2f(0.0, 0.0),
                                vec2f(1.0, 0.0),
                                vec2f(0.0, 1.0),
                                vec2f(0.0, 1.0),
                                vec2f(1.0, 0.0),
                                vec2f(1.0, 1.0));
  let rect = vertexId / 6u;
  let bounds = rects[rect];
  let position = mix(bounds.xy, bounds.zw, corners[vertexId % 6u]);
  return VertexOut(vec4f(position, 0.0, 1.0), albedos[rect], normals[rect], materials[rect]);
}

struct GBufferOut {
  @location(0) albedo : vec4f,
  @location(1) normal : vec4f,
  @location(2) material : vec4f,
};

@fragment
fn gBufferFragmentShader(v : VertexOut) -> GBufferOut {
  return GBufferOut(vec4f(v.albedo, 1.0), vec4f(v.normal, 1.0), vec4f(v.material, 0.0, 0.0, 1.0));
}
)";

// Bind convention: texture unit i at @group(0) @binding(2i). The G-buffer is read at the pixel of
// the fragment, which needs no sampler.
constexpr const char* kWgslLightingShader = R"(
@group(0) @binding(0) var albedoTexture : texture_2d<f32>;
@group(0) @binding(2) var normalTexture : texture_2d<f32>;
@group(0) @binding(4) var materialTexture : texture_2d<f32>;

@vertex
fn lightingVertexShader(@builtin(vertex_index) vertexId : u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vertexId << 1u) & 2u), f32(vertexId & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn lightingFragmentShader(@builtin(position) position : vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  let albedo = textureLoad(albedoTexture, pixel, 0);
  if (albedo.a < 0.5) {
    discard;
  }
  let normal = textureLoad(normalTexture, pixel, 0);
  let material = textureLoad(materialTexture, pixel, 0);
  let lightDirection = vec3f(0.0, 0.0, 1.0);
  let factor = select(1.0, 64.0 / 255.0, material.r > 0.5 || dot(normal.xyz, lightDirection) < 0.5);
  return vec4f(albedo.rgb * factor, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Shader stages
// ---------------------------------------------------------------------------

// `subpass` selects the shaders of the subpass path, which writes color attachments 1 to 3 and is
// not available on D3D12 and WebGPU
std::unique_ptr<IShaderStages> getGBufferShaderStages(const IDevice& device, bool subpass) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    IGL_DEBUG_ASSERT(!subpass);
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, kWgslGBufferShader, "gBufferVertexShader", "gBufferFragmentShader", "", nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getVulkanGBufferVertexShader().c_str(),
        "main",
        "",
        getVulkanGBufferFragmentShader(subpass).c_str(),
        "main",
        "",
        nullptr);
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device,
        (std::string(kMetalGBufferVertexShader) +
         (subpass ? kMetalGBufferSubpassFragmentShader : kMetalGBufferFragmentShader))
            .c_str(),
        "gBufferVertexShader",
        "gBufferFragmentShader",
        "",
        nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getOpenGLGBufferVertexShader(device, subpass).c_str(),
        "main",
        "",
        getOpenGLGBufferFragmentShader(device, subpass).c_str(),
        "main",
        "",
        nullptr);
  case BackendType::D3D12:
    IGL_DEBUG_ASSERT(!subpass);
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kHlslGBufferVertexShader, "main", "", kHlslGBufferPixelShader, "main", "", nullptr);
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

std::unique_ptr<IShaderStages> getLightingShaderStages(const IDevice& device, bool subpass) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    IGL_DEBUG_ASSERT(!subpass);
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, kWgslLightingShader, "lightingVertexShader", "lightingFragmentShader", "", nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        kVulkanLightingVertexShader,
        "main",
        "",
        getVulkanLightingFragmentShader(subpass).c_str(),
        "main",
        "",
        nullptr);
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device,
        (std::string(kMetalLightingVertexShader) +
         (subpass ? kMetalLightingSubpassFragmentShader : kMetalLightingFragmentShader))
            .c_str(),
        "lightingVertexShader",
        "lightingFragmentShader",
        "",
        nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getOpenGLLightingVertexShader(device, subpass).c_str(),
        "main",
        "",
        getOpenGLLightingFragmentShader(device, subpass).c_str(),
        "main",
        "",
        nullptr);
  case BackendType::D3D12:
    IGL_DEBUG_ASSERT(!subpass);
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           kHlslLightingVertexShader,
                                                           "main",
                                                           "",
                                                           kHlslLightingPixelShader,
                                                           "main",
                                                           "",
                                                           nullptr);
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

// Returns why the device cannot run the G-buffer and lighting passes, or nullptr if it can
const char* IGL_NULLABLE getDeferredUnsupportedReason(const IDevice& device) {
  if (!device.hasFeature(DeviceFeatures::MultipleRenderTargets)) {
    return "multiple render targets are not supported";
  }
  // The OpenGL shaders use flat interpolation, gl_VertexID, texelFetch() and explicit fragment
  // output locations, which need GLSL ES 3.00 or GLSL 3.30. The major version of GLSL ES 1.00 and
  // of GLSL 1.10 to 1.50 is 1.
  const bool isOpenGL = device.getBackendType() == BackendType::OpenGL;
  if (isOpenGL && device.getShaderVersion().majorVersion < 3) {
    return "the OpenGL shaders need GLSL ES 3.00 or GLSL 3.30";
  }
  return nullptr;
}

bool canUseSubpasses(const IDevice& device, const ITexture& surface) {
  if (!device.hasFeature(DeviceFeatures::Subpasses) || surface.getNumLayers() != 1) {
    return false;
  }
  // The OpenGL shaders of the subpass path are GLSL ES 3.00
  if (device.getBackendType() == BackendType::OpenGL) {
    const BackendVersion version = device.getBackendVersion();
    return version.flavor == BackendFlavor::OpenGL_ES && version.majorVersion >= 3;
  }
  return true;
}

// Returns three null pointers if a texture cannot be created. The subpass path keeps the G-buffer
// in tile memory and reads it as input attachments, and the other path samples it in a later pass.
std::array<std::shared_ptr<ITexture>, 3> createGBuffer(IDevice& device,
                                                       const Dimensions& size,
                                                       bool subpass) {
  const TextureDesc::TextureUsage usage = subpass ? kSubpassGBufferUsage : kSampledGBufferUsage;
  const ResourceStorage storage = subpass ? ResourceStorage::Memoryless : ResourceStorage::Private;

  std::array<std::shared_ptr<ITexture>, 3> gBuffer;
  for (size_t i = 0; i != gBuffer.size(); ++i) {
    const TextureDesc desc{
        .width = size.width,
        .height = size.height,
        .usage = usage,
        .type = TextureType::TwoD,
        .format = kGBufferFormat,
        .storage = storage,
        .debugName = kGBufferNames[i],
    };
    Result result;
    gBuffer[i] = device.createTexture(desc, &result);
    if (!result.isOk() || gBuffer[i] == nullptr) {
      IGL_LOG_ERROR("SubpassDeferredSession: cannot create the G-buffer: %s\n",
                    result.message.c_str());
      return {};
    }
  }
  return gBuffer;
}

std::shared_ptr<IRenderPipelineState> createPipeline(
    IDevice& device,
    std::unique_ptr<IShaderStages> shaderStages,
    const std::vector<TextureFormat>& colorFormats,
    std::unordered_map<size_t, NameHandle> fragmentUnitSamplerMap,
    const char* debugName) {
  if (shaderStages == nullptr) {
    return nullptr;
  }
  RenderPipelineDesc::TargetDesc targetDesc;
  for (const TextureFormat format : colorFormats) {
    targetDesc.colorAttachments.push_back({.textureFormat = format});
  }
  Result result;
  auto pipeline = device.createRenderPipeline(
      RenderPipelineDesc{
          .shaderStages = std::move(shaderStages),
          .targetDesc = std::move(targetDesc),
          .fragmentUnitSamplerMap = std::move(fragmentUnitSamplerMap),
          .debugName = genNameHandle(debugName),
      },
      &result);
  if (!result.isOk() || pipeline == nullptr) {
    IGL_LOG_ERROR("SubpassDeferredSession: cannot create a pipeline: %s\n", result.message.c_str());
    return nullptr;
  }
  return pipeline;
}

} // namespace

void SubpassDeferredSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  const char* const unsupportedReason = getDeferredUnsupportedReason(device);
  deferred_ = unsupportedReason == nullptr;
  if (!deferred_) {
    IGL_LOG_INFO("SubpassDeferredSession: %s; clearing\n", unsupportedReason);
  }

  // The lighting pass fetches texels without filtering, but Vulkan and OpenGL need a sampler for
  // each sampled texture
  sampler_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Nearest,
          .magFilter = SamplerMinMagFilter::Nearest,
          .debugName = "Sampler: nearest",
      },
      nullptr);
  IGL_DEBUG_ASSERT(sampler_ != nullptr);

  // Clearing the G-buffer to zero leaves the alpha of the albedo at zero, which marks the
  // background
  const RenderPassDesc::AttachmentDesc image{
      .loadAction = LoadAction::Clear,
      .storeAction = StoreAction::Store,
      .clearColor = getPreferredClearColor(),
  };
  const RenderPassDesc::AttachmentDesc storedGBuffer{.loadAction = LoadAction::Clear,
                                                     .storeAction = StoreAction::Store};
  const RenderPassDesc::AttachmentDesc tileGBuffer{.loadAction = LoadAction::Clear,
                                                   .storeAction = StoreAction::DontCare};

  // The first subpass writes the G-buffer in color attachments 1 to 3. The second reads it as
  // input attachments and writes the image in color attachment 0.
  subpassRenderPass_ = {
      .colorAttachments = {image, tileGBuffer, tileGBuffer, tileGBuffer},
      .subpasses = {{.colorAttachmentMask = 0b1110},
                    {.colorAttachmentMask = 0b0001, .inputAttachmentMask = 0b1110}},
  };
  gBufferRenderPass_ = {.colorAttachments = {storedGBuffer, storedGBuffer, storedGBuffer}};
  lightingRenderPass_ = {.colorAttachments = {image}};
}

void SubpassDeferredSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  // The G-buffer is as large as the surface, and not every platform calls releaseFramebuffer()
  // when the surface is resized
  if (gBuffer_[0] != nullptr && gBuffer_[0]->getSize() != surfaceTextures.color->getSize()) {
    releaseFramebuffer();
  }
  auto& device = getPlatform().getDevice();

  const bool useSubpasses = deferred_ && !subpassesUnsupported_ &&
                            canUseSubpasses(device, *surfaceTextures.color);
  if (useSubpasses != usesSubpasses_) {
    // The framebuffer and the G-buffer differ between the two paths
    releaseFramebuffer();
  }

  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);

  usesSubpasses_ = useSubpasses && renderWithSubpasses(*buffer, surfaceTextures);
  if (!usesSubpasses_) {
    renderWithPasses(*buffer, surfaceTextures);
  }

  if (!pathLogged_ && deferred_) {
    pathLogged_ = true;
    IGL_LOG_INFO("SubpassDeferredSession: the G-buffer and lighting passes run as %s\n",
                 usesSubpasses_ ? "two subpasses of one render pass" : "two render passes");
  }

  if (shellParams().shouldPresent) {
    buffer->present(surfaceTextures.color);
  }
  commandQueue_->submit(*buffer);
}

void SubpassDeferredSession::releaseFramebuffer() {
  RenderSession::releaseFramebuffer();
  gBufferFramebuffer_ = nullptr;
  gBuffer_ = {};
}

// Returns false, and leaves the encoding of the frame to renderWithPasses(), if the device cannot
// run the subpasses with these framebuffer attachments
bool SubpassDeferredSession::renderWithSubpasses(ICommandBuffer& buffer,
                                                 const SurfaceTextures& surfaceTextures) {
  auto& device = getPlatform().getDevice();

  if (framebuffer_ == nullptr) {
    gBuffer_ = createGBuffer(device, surfaceTextures.color->getDimensions(), true);
    if (gBuffer_[0] == nullptr) {
      return disableSubpasses("cannot create the G-buffer");
    }
    Result result;
    framebuffer_ = device.createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color},
                                 {.texture = gBuffer_[0]},
                                 {.texture = gBuffer_[1]},
                                 {.texture = gBuffer_[2]}},
            .debugName = "SubpassDeferredSession: subpass framebuffer",
        },
        &result);
    if (!result.isOk() || framebuffer_ == nullptr) {
      return disableSubpasses(result.message);
    }
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  // Both pipelines declare the four color formats of the framebuffer
  const std::vector<TextureFormat> colorFormats = {
      surfaceTextures.color->getFormat(), kGBufferFormat, kGBufferFormat, kGBufferFormat};
  if (gBufferSubpassPipeline_ == nullptr) {
    gBufferSubpassPipeline_ = createPipeline(device,
                                             getGBufferShaderStages(device, true),
                                             colorFormats,
                                             {},
                                             "SubpassDeferredSession: G-buffer subpass");
  }
  if (lightingSubpassPipeline_ == nullptr) {
    lightingSubpassPipeline_ = createPipeline(device,
                                              getLightingShaderStages(device, true),
                                              colorFormats,
                                              {},
                                              "SubpassDeferredSession: lighting subpass");
  }
  if (gBufferSubpassPipeline_ == nullptr || lightingSubpassPipeline_ == nullptr) {
    return disableSubpasses("cannot create the pipelines");
  }

  // Fails with Result::Code::Unsupported if the framebuffer cannot run subpasses, like the
  // framebuffer of the window system on OpenGL
  Result result;
  const auto encoder = buffer.createRenderCommandEncoder(subpassRenderPass_, framebuffer_, &result);
  if (encoder == nullptr) {
    return disableSubpasses(result.message);
  }
  encoder->bindRenderPipelineState(gBufferSubpassPipeline_);
  encoder->draw(kSceneVertexCount);
  encoder->nextSubpass();
  encoder->bindRenderPipelineState(lightingSubpassPipeline_);
  encoder->draw(kFullscreenTriangleVertexCount);
  encoder->endEncoding();
  return true;
}

// Only clears the image if the device cannot run the G-buffer and lighting passes
void SubpassDeferredSession::renderWithPasses(ICommandBuffer& buffer,
                                              const SurfaceTextures& surfaceTextures) {
  auto& device = getPlatform().getDevice();

  if (framebuffer_ == nullptr) {
    Result result;
    framebuffer_ = device.createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color}},
            .debugName = "SubpassDeferredSession: lighting framebuffer",
        },
        &result);
    if (!result.isOk() || framebuffer_ == nullptr) {
      IGL_LOG_ERROR("SubpassDeferredSession: cannot create the lighting framebuffer: %s\n",
                    result.message.c_str());
      framebuffer_ = nullptr;
    }
    if (deferred_) {
      gBuffer_ = createGBuffer(device, surfaceTextures.color->getDimensions(), false);
      if (gBuffer_[0] != nullptr) {
        gBufferFramebuffer_ = device.createFramebuffer(
            FramebufferDesc{
                .colorAttachments = {{.texture = gBuffer_[0]},
                                     {.texture = gBuffer_[1]},
                                     {.texture = gBuffer_[2]}},
                .debugName = "SubpassDeferredSession: G-buffer framebuffer",
            },
            &result);
        if (!result.isOk() || gBufferFramebuffer_ == nullptr) {
          IGL_LOG_ERROR("SubpassDeferredSession: cannot create the G-buffer framebuffer: %s\n",
                        result.message.c_str());
          gBufferFramebuffer_ = nullptr;
        }
      }
    }
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  if (deferred_) {
    if (gBufferPipeline_ == nullptr) {
      gBufferPipeline_ = createPipeline(device,
                                        getGBufferShaderStages(device, false),
                                        {kGBufferFormat, kGBufferFormat, kGBufferFormat},
                                        {},
                                        "SubpassDeferredSession: G-buffer");
    }
    if (lightingPipeline_ == nullptr) {
      lightingPipeline_ = createPipeline(device,
                                         getLightingShaderStages(device, false),
                                         {surfaceTextures.color->getFormat()},
                                         {{0, IGL_NAMEHANDLE("uAlbedo")},
                                          {1, IGL_NAMEHANDLE("uNormal")},
                                          {2, IGL_NAMEHANDLE("uMaterial")}},
                                         "SubpassDeferredSession: lighting");
    }
  }

  if (deferred_ && (gBufferFramebuffer_ == nullptr || gBufferPipeline_ == nullptr ||
                    lightingPipeline_ == nullptr)) {
    disableDeferred();
  }
  if (framebuffer_ == nullptr) {
    IGL_DEBUG_ABORT("SubpassDeferredSession: cannot create the lighting framebuffer");
    return;
  }

  if (deferred_) {
    const auto gBufferEncoder =
        buffer.createRenderCommandEncoder(gBufferRenderPass_, gBufferFramebuffer_);
    IGL_DEBUG_ASSERT(gBufferEncoder != nullptr);
    if (gBufferEncoder) {
      gBufferEncoder->bindRenderPipelineState(gBufferPipeline_);
      gBufferEncoder->draw(kSceneVertexCount);
      gBufferEncoder->endEncoding();
    }
  }

  // The lighting pass samples the textures that the G-buffer pass rendered to
  const Dependencies dependencies{
      .textures = {gBuffer_[0].get(), gBuffer_[1].get(), gBuffer_[2].get()}};
  const auto lightingEncoder =
      buffer.createRenderCommandEncoder(lightingRenderPass_, framebuffer_, dependencies);
  IGL_DEBUG_ASSERT(lightingEncoder != nullptr);
  if (lightingEncoder) {
    if (deferred_) {
      lightingEncoder->bindRenderPipelineState(lightingPipeline_);
      for (size_t i = 0; i != gBuffer_.size(); ++i) {
        lightingEncoder->bindTexture(i, BindTarget::kFragment, gBuffer_[i].get());
        lightingEncoder->bindSamplerState(i, BindTarget::kFragment, sampler_.get());
      }
      lightingEncoder->draw(kFullscreenTriangleVertexCount);
    }
    lightingEncoder->endEncoding();
  }
}

bool SubpassDeferredSession::disableSubpasses(const std::string& reason) {
  IGL_LOG_INFO("SubpassDeferredSession: no subpasses (%s); using two render passes\n",
               reason.c_str());
  subpassesUnsupported_ = true;
  gBufferSubpassPipeline_ = nullptr;
  lightingSubpassPipeline_ = nullptr;
  releaseFramebuffer();
  return false;
}

void SubpassDeferredSession::disableDeferred() {
  IGL_DEBUG_ABORT("SubpassDeferredSession: cannot create the render pass resources; clearing");
  deferred_ = false;
  gBufferPipeline_ = nullptr;
  lightingPipeline_ = nullptr;
  gBufferFramebuffer_ = nullptr;
  gBuffer_ = {};
}

} // namespace igl::shell
