/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @MARK:COVERAGE_EXCLUDE_FILE

// Ported from the LightweightVK sample app `008_MeshShaderFireworks.cpp`
// (https://github.com/corporateshark/lightweightvk). Uses vertex shader
// billboarding with 4 vertices per particle instead of mesh shaders.

#include <shell/renderSessions/FireworksSession.h>

#include <algorithm>
#include <cmath>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <shell/shared/renderSession/AppParams.h>
#include <shell/shared/renderSession/Fov.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>
#if IGL_BACKEND_OPENGL
#include <igl/opengl/Device.h>
#endif
// @fb-only
// @fb-only
// @fb-only
// @fb-only
// @fb-only

namespace igl::shell {

namespace {

// Uniforms: projection * view matrix (2 entries for single-pass stereo)
struct Uniforms {
  glm::mat4 mvp[2];
};

[[maybe_unused]] glm::mat4 perspectiveAsymmetricFovRH(const Fov& fov, float nearZ, float farZ) {
  const float tanLeft = std::tan(fov.angleLeft);
  const float tanRight = std::tan(fov.angleRight);
  const float tanDown = std::tan(fov.angleDown);
  const float tanUp = std::tan(fov.angleUp);

  const float tanWidth = tanRight - tanLeft;
  const float tanHeight = tanUp - tanDown;

  glm::mat4 mat;
  mat[0][0] = 2.0f / tanWidth;
  mat[1][0] = 0.0f;
  mat[2][0] = (tanRight + tanLeft) / tanWidth;
  mat[3][0] = 0.0f;
  mat[0][1] = 0.0f;
  mat[1][1] = 2.0f / tanHeight;
  mat[2][1] = (tanUp + tanDown) / tanHeight;
  mat[3][1] = 0.0f;
  mat[0][2] = 0.0f;
  mat[1][2] = 0.0f;
  mat[2][2] = -(farZ + nearZ) / (farZ - nearZ);
  mat[3][2] = -2.0f * farZ * nearZ / (farZ - nearZ);
  mat[0][3] = 0.0f;
  mat[1][3] = 0.0f;
  mat[2][3] = -1.0f;
  mat[3][3] = 0.0f;
  return mat;
}

[[maybe_unused]] void stringReplaceAll(std::string& s,
                                       const std::string& searchString,
                                       const std::string& replaceString) {
  size_t pos = 0;
  while ((pos = s.find(searchString, pos)) != std::string::npos) {
    s.replace(pos, searchString.length(), replaceString);
  }
}

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
//   normal (builtin, location 2) = per-vertex particle color (RGB), stored in VertexPTN::nor //
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
// Each particle is a billboarded quad: 4 vertices, 6 indices.
// The vertex shader expands particle center + corner offset into a screen-aligned quad.

[[nodiscard]] std::string getVulkanVertexShaderSource(bool stereoRendering) {
  const std::string prolog = stereoRendering ? R"(#version 460
#extension GL_OVR_multiview2 : require
layout(num_views = 2) in;
#define VIEW_ID int(gl_ViewID_OVR)
)"
                                             : R"(#version 460
#define VIEW_ID 0
)";
  return prolog + R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec3 color;
layout (location=2) in float flare;
layout (location=3) in vec2 corner;

layout (set = 1, binding = 0, std140) uniform UniformBlock {
  mat4 mvp[2];
} ub;

layout (location=0) out vec3 vColor;
layout (location=1) out vec2 vUV;

void main() {
  vec4 center = ub.mvp[VIEW_ID] * vec4(pos, 1.0);

  vec2 size = flare > 0.5 ? vec2(0.05, 0.25) : vec2(0.15, 0.15);
  vec3 col = flare > 0.5 ? 0.5 * color : color;

  // Billboard offset in clip space
  vec2 offset = corner * size;
  gl_Position = center + vec4(offset, 0.0, 0.0);

  vColor = col;
  vUV = corner * 0.5 + 0.5;
}
)";
}

const char* getVulkanFragmentShaderSource() {
  return R"(#version 460
precision mediump float;
layout (location=0) in vec3 vColor;
layout (location=1) in vec2 vUV;
layout (location=0) out vec4 out_FragColor;

layout(set = 0, binding = 0) uniform sampler2D particleTex;

void main() {
  float alpha = texture(particleTex, vUV).r;
  out_FragColor = vec4(vColor * alpha, alpha);
}
)";
}

const char* getMetalShaderSource() {
  return R"(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

struct Uniforms {
  float4x4 mvp[2];
};

struct VertexIn {
  packed_float3 pos;
  packed_float3 color;
  float  flare;
  packed_float2 corner;
};

struct VertexOut {
  float4 position [[position]];
  float3 color;
  float2 uv;
};

vertex VertexOut vertexShader(
    uint vid [[vertex_id]],
    constant VertexIn* vertices [[buffer(1)]],
    constant Uniforms& ub [[buffer(0)]]) {
  VertexIn v = vertices[vid];
  VertexOut out;

  float3 pos = float3(v.pos);
  float3 col = float3(v.color);
  float2 crn = float2(v.corner);

  float4 center = ub.mvp[0] * float4(pos, 1.0);

  float2 size = v.flare > 0.5 ? float2(0.05, 0.25) : float2(0.15, 0.15);
  float3 color = v.flare > 0.5 ? 0.5 * col : col;

  float2 offset = crn * size;
  out.position = center + float4(offset, 0.0, 0.0);
  out.color = color;
  out.uv = crn * 0.5 + 0.5;
  return out;
}

fragment float4 fragmentShader(
    VertexOut in [[stage_in]],
    texture2d<float> particleTex [[texture(0)]],
    sampler linearSampler [[sampler(0)]]) {
  float alpha = particleTex.sample(linearSampler, in.uv).r;
  return float4(in.color * alpha, alpha);
}
)";
}

const char* getD3D12VertexShaderSource() {
  return R"(
cbuffer UniformBlock : register(b0) {
  float4x4 mvp[2];
};

struct VSInput {
  float3 pos    : POSITION;
  float3 color  : COLOR0;
  float  flare  : TEXCOORD0;
  float2 corner : TEXCOORD1;
};

struct VSOutput {
  float4 position : SV_POSITION;
  float3 color    : COLOR0;
  float2 uv       : TEXCOORD0;
};

VSOutput main(VSInput input) {
  VSOutput output;
  float4 center = mul(mvp[0], float4(input.pos, 1.0));

  float2 size = input.flare > 0.5 ? float2(0.05, 0.25) : float2(0.15, 0.15);
  float3 color = input.flare > 0.5 ? 0.5 * input.color : input.color;

  float2 offset = input.corner * size;
  output.position = center + float4(offset.x, -offset.y, 0.0, 0.0);
  output.color = color;
  output.uv = input.corner * 0.5 + 0.5;
  return output;
}
)";
}

const char* getD3D12FragmentShaderSource() {
  return R"(
Texture2D particleTex : register(t0);
SamplerState linearSampler : register(s0);

struct PSInput {
  float4 position : SV_POSITION;
  float3 color    : COLOR0;
  float2 uv       : TEXCOORD0;
};

float4 main(PSInput input) : SV_Target {
  float alpha = particleTex.Sample(linearSampler, input.uv).r;
  return float4(input.color * alpha, alpha);
}
)";
}

const char* getWgslShaderSource() {
  return R"(
struct UniformBlock {
  mvp : array<mat4x4f, 2>,
};

@group(1) @binding(0) var<uniform> ub : UniformBlock;
@group(0) @binding(0) var particleTex : texture_2d<f32>;
@group(0) @binding(1) var linearSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec3f,
  @location(1) uv : vec2f,
};

@vertex
fn vertexShader(@location(0) pos : vec3f,
                @location(1) color : vec3f,
                @location(2) flare : f32,
                @location(3) corner : vec2f) -> VertexOut {
  let center = ub.mvp[0] * vec4f(pos, 1.0);
  let size = select(vec2f(0.15, 0.15), vec2f(0.05, 0.25), flare > 0.5);
  let col = select(color, 0.5 * color, flare > 0.5);
  let offset = corner * size;
  return VertexOut(center + vec4f(offset, 0.0, 0.0), col, corner * 0.5 + 0.5);
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  let alpha = textureSample(particleTex, linearSampler, v.uv).r;
  return vec4f(v.color * alpha, alpha);
}
)";
}

// Interleaved vertex: particle data + corner offset
struct InterleavedVertex {
  glm::vec3 pos;
  glm::vec3 color;
  float flare = 0.0f;
  glm::vec2 corner;
};

const glm::vec2 kCornerOffsets[4] = {
    {-1.0f, -1.0f},
    {+1.0f, -1.0f},
    {-1.0f, +1.0f},
    {+1.0f, +1.0f},
};

} // namespace

// --- Particle ---

FireworksSession::ParticleStateMessage FireworksSession::Particle::step(const glm::vec3& gravity) {
  pos += velocity;
  velocity += gravity;
  ttl--;

  if (fadingOut) {
    const float t = static_cast<float>(ttl) / static_cast<float>(initialLifetime);
    currentColor = baseColor * std::max(t, 0.0f);
  }

  if (ttl < 0) {
    return ParticleStateMessage::Kill;
  }
  return emission ? ParticleStateMessage::Emission : ParticleStateMessage::None;
}

// --- ParticleSystem ---

void FireworksSession::ParticleSystem::nextFrame(const glm::vec3& gravity,
                                                 const glm::vec3& viewerPos,
                                                 std::mt19937& rng) {
  std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
  int32_t processedParticles = 0;

  for (int32_t i = 0; i < kMaxParticles; i++) {
    if (particles[i].alive) {
      processedParticles++;

      switch (particles[i].step(gravity)) {
      case ParticleStateMessage::None:
        break;
      case ParticleStateMessage::Kill:
        if (particles[i].spawnExplosion) {
          addExplosion(particles[i].pos, viewerPos, rng);
        }
        particles[i].alive = false;
        totalParticles--;
        break;
      case ParticleStateMessage::Emission: {
        Particle trail;
        trail.pos = particles[i].pos;
        trail.velocity = particles[i].velocity * (dist01(rng) * 0.8f + 0.1f);
        trail.baseColor = particles[i].currentColor * 0.9f;
        trail.currentColor = trail.baseColor;
        trail.ttl = particles[i].ttl >> 2;
        trail.initialLifetime = std::max(trail.ttl, 1);
        trail.alive = true;
        trail.fadingOut = true;
        addParticle(trail);
        break;
      }
      }
    } else if (queuedParticles > 0) {
      particles[i] = particlesStack[--queuedParticles];
      totalParticles++;
    } else if (processedParticles >= totalParticles) {
      return;
    }
  }
}

void FireworksSession::ParticleSystem::addParticle(const Particle& particle) {
  if (queuedParticles < kMaxParticles) {
    particlesStack[queuedParticles++] = particle;
  }
}

void FireworksSession::ParticleSystem::addExplosion(const glm::vec3& pos,
                                                    const glm::vec3& viewerPos,
                                                    std::mt19937& rng) {
  const glm::vec3 palette[3] = {
      {0.15f, 0.2f, 1.0f},
      {1.0f, 0.15f, 0.2f},
      {0.1f, 1.0f, 0.15f},
  };

  // Build an orthonormal basis for the explosion plane perpendicular to the view direction
  const glm::vec3 toViewer = viewerPos - pos;
  const float dist = glm::length(toViewer);
  // Fallback axes if viewer is at the explosion center
  glm::vec3 right(1.0f, 0.0f, 0.0f);
  glm::vec3 up(0.0f, 1.0f, 0.0f);
  if (dist > 0.001f) {
    const glm::vec3 viewDir = toViewer / dist;
    // Choose a reference vector that isn't parallel to viewDir
    const glm::vec3 ref = std::fabs(viewDir.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f)
                                                       : glm::vec3(1.0f, 0.0f, 0.0f);
    right = glm::normalize(glm::cross(ref, viewDir));
    up = glm::cross(viewDir, right);
  }

  std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
  std::uniform_int_distribution<int32_t> palDist(0, 2);
  const int32_t paletteIndex = palDist(rng);

  for (int32_t i = 0; i < 300; i++) {
    const float radius = dist01(rng) / 10.0f;
    const float angle = dist01(rng) * 2.0f * static_cast<float>(M_PI);
    const float depthSpread = (dist01(rng) * 100.0f - 50.0f) / 5000.0f;
    const glm::vec3 vel =
        radius * std::cos(angle) * right + radius * std::sin(angle) * up +
        depthSpread *
            glm::normalize(glm::vec3::length() > 0.001f ? toViewer : glm::vec3(0.0f, 0.0f, 1.0f));
    const glm::vec3 color = palette[paletteIndex] +
                            glm::vec3(dist01(rng) / 5.0f, dist01(rng) / 5.0f, dist01(rng) / 5.0f);

    const int32_t lifetime = 90 + static_cast<int32_t>(dist01(rng) * 20.0f);
    addParticle({
        .pos = pos,
        .velocity = vel,
        .baseColor = color,
        .currentColor = color,
        .ttl = lifetime,
        .initialLifetime = lifetime,
        .alive = true,
        .fadingOut = true,
        .emission = true,
    });
  }
}

// --- FireworksSession ---

void FireworksSession::generateParticleTexture(std::vector<uint8_t>& image) {
  const auto numPixels = static_cast<size_t>(kParticleTextureSize) * kParticleTextureSize;
// @fb-only
  // @fb-only
  // @fb-only
      // @fb-only
  // @fb-only
// @fb-only
  image.resize(numPixels);
// @fb-only
  const float center = 0.5f * (kParticleTextureSize - 1);

  for (int32_t y = 0; y < kParticleTextureSize; y++) {
    for (int32_t x = 0; x < kParticleTextureSize; x++) {
      const float dx = static_cast<float>(x) - center;
      const float dy = static_cast<float>(y) - center;
      const float dist = std::sqrt(dx * dx + dy * dy);
      const float normalizedDist = dist < center ? dist / center : 1.0f;
      const float falloff = 1.0f - normalizedDist;
      const auto value = static_cast<uint8_t>(
          std::fmin(255.0f, std::fmax(0.0f, falloff * falloff * falloff * 255.0f)));
      const size_t pixel = static_cast<size_t>(y) * kParticleTextureSize + x;
// @fb-only
      // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
      // @fb-only
// @fb-only
      {
        // image is resized to numPixels above and pixel < numPixels by loop bounds
        // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
        image[pixel] = value;
      }
    }
  }
}

std::unique_ptr<IShaderStages> FireworksSession::getShaderStagesForBackend(IDevice& device,
                                                                           bool stereoRendering) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::Vulkan: {
    const std::string vsSource = getVulkanVertexShaderSource(stereoRendering);
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, vsSource.c_str(), "main", "", getVulkanFragmentShaderSource(), "main", "", nullptr);
  }
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
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getMetalShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::D3D12:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getD3D12VertexShaderSource(),
                                                           "main",
                                                           "",
                                                           getD3D12FragmentShaderSource(),
                                                           "main",
                                                           "",
                                                           nullptr);
  case BackendType::OpenGL: {
#if IGL_BACKEND_OPENGL
    auto glVersion =
        static_cast<opengl::Device&>(device).getContext().deviceFeatures().getGLVersion();

    if (glVersion <= opengl::GLVersion::v2_1) {
      // GLSL 120: attribute/varying, plain uniforms, texture2D
      const char* vs120 = R"(#version 120
attribute vec3 pos;
attribute vec3 color;
attribute float flare;
attribute vec2 corner;

uniform mat4 mvp[2];

varying vec3 vColor;
varying vec2 vUV;

void main() {
  vec4 center = mvp[0] * vec4(pos, 1.0);

  vec2 size = flare > 0.5 ? vec2(0.05, 0.25) : vec2(0.15, 0.15);
  vec3 col = flare > 0.5 ? 0.5 * color : color;

  vec2 offset = corner * size;
  gl_Position = center + vec4(offset, 0.0, 0.0);

  vColor = col;
  vUV = corner * 0.5 + 0.5;
}
)";
      const char* fs120 = R"(#version 120
varying vec3 vColor;
varying vec2 vUV;

uniform sampler2D particleTex;

void main() {
  float alpha = texture2D(particleTex, vUV).r;
  gl_FragColor = vec4(vColor * alpha, alpha);
}
)";
      if (opengl::DeviceFeatureSet::usesOpenGLES()) {
        // OpenGL ES 2.0 takes the same shaders as GLSL ES 1.00.
        std::string codeVS(vs120);
        std::string codeFS(fs120);
        stringReplaceAll(codeVS, "#version 120", "#version 100");
        stringReplaceAll(codeFS, "#version 120", "#version 100\nprecision mediump float;");
        return igl::ShaderStagesCreator::fromModuleStringInput(
            device, codeVS.c_str(), "main", "", codeFS.c_str(), "main", "", nullptr);
      }
      return igl::ShaderStagesCreator::fromModuleStringInput(
          device, vs120, "main", "", fs120, "main", "", nullptr);
    }

    auto usesOpenGLES = opengl::DeviceFeatureSet::usesOpenGLES();

    std::string codeVS(getVulkanVertexShaderSource(false));
    stringReplaceAll(codeVS, "gl_VertexIndex", "gl_VertexID");
    stringReplaceAll(codeVS, "#version 460", usesOpenGLES ? "#version 300 es" : "#version 410");
    stringReplaceAll(codeVS,
                     "layout (set = 1, binding = 0, std140) uniform UniformBlock",
                     "layout (std140) uniform UniformBlock");

    std::string codeFS(getVulkanFragmentShaderSource());
    stringReplaceAll(codeFS, "#version 460", usesOpenGLES ? "#version 300 es" : "#version 410");
    if (usesOpenGLES) {
      // GLSL ES 3.00 has no locations on vertex outputs and fragment inputs.
      stringReplaceAll(codeVS, "layout (location=0) out", "out");
      stringReplaceAll(codeVS, "layout (location=1) out", "out");
      stringReplaceAll(codeFS, "layout (location=0) in", "in");
      stringReplaceAll(codeFS, "layout (location=1) in", "in");
    } else {
      stringReplaceAll(codeFS, "precision mediump float;\n", "");
    }
    stringReplaceAll(codeFS, "layout(set = 0, binding = 0) uniform", "uniform");

    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, codeVS.c_str(), "main", "", codeFS.c_str(), "main", "", nullptr);
#else
    return nullptr;
#endif // IGL_BACKEND_OPENGL
  }
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void FireworksSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Enable passthrough so the background is transparent, dimmed to 0.2
  appParamsRef().passthroughGetter = []() { return true; };
  appParamsRef().passthroughOpacity = 0.2f;

  // Allocate particle system on the heap
  particleSystem_ = std::make_unique<ParticleSystem>();

  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  renderPass_.colorAttachments = {{
      .loadAction = LoadAction::Clear,
      .storeAction = StoreAction::Store,
      .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
  }};
  renderPass_.depthAttachment = {.loadAction = LoadAction::DontCare};

  // Particle texture
  std::vector<uint8_t> texData;
  generateParticleTexture(texData);

// @fb-only
  // @fb-only
                             // @fb-only
                             // @fb-only
// @fb-only
  const auto texFormat = TextureFormat::R_UNorm8;
// @fb-only
  const TextureDesc texDesc = TextureDesc::new2D(texFormat,
                                                 kParticleTextureSize,
                                                 kParticleTextureSize,
                                                 TextureDesc::TextureUsageBits::Sampled,
                                                 "Particle Texture");
  particleTexture_ = device.createTexture(texDesc, nullptr);
  IGL_DEBUG_ASSERT(particleTexture_ != nullptr);
  particleTexture_->upload(
      TextureRangeDesc::new2D(0, 0, kParticleTextureSize, kParticleTextureSize), texData.data());

  sampler_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
          .debugName = "Sampler: linear",
      },
      nullptr);
  IGL_DEBUG_ASSERT(sampler_ != nullptr);

  // Uniform buffer; without uniform blocks the matrices go through bindUniform().
  const Uniforms uniforms{.mvp = {glm::mat4(1.0f), glm::mat4(1.0f)}};
  uniformBuffer_ = !device.hasFeature(DeviceFeatures::UniformBlocks)
                       ? nullptr
                       : device.createBuffer(
                             {
                                 .type = BufferDesc::BufferTypeBits::Uniform,
                                 .data = &uniforms,
                                 .length = sizeof(Uniforms),
                                 .storage = ResourceStorage::Shared,
                                 .hint = BufferDesc::BufferAPIHintBits::UniformBlock,
                                 .debugName = "fireworks_uniforms",
                             },
                             nullptr);
  IGL_DEBUG_ASSERT(uniformBuffer_ != nullptr || !device.hasFeature(DeviceFeatures::UniformBlocks));

  // Pre-allocate vertex buffer for max particles * 4 vertices
  const size_t maxVerts = static_cast<size_t>(kMaxParticles) * 4;
// @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
      // @fb-only
  // @fb-only
// @fb-only
  const auto vbStorage = ResourceStorage::Shared;
  const size_t vertexStride = sizeof(InterleavedVertex);
  const auto ibStorage = ResourceStorage::Shared;
// @fb-only
  vertexBuffer_ = device.createBuffer(
      {
          .type = BufferDesc::BufferTypeBits::Vertex,
          .length = maxVerts * vertexStride,
          .storage = vbStorage,
          .debugName = "fireworks_vertices",
      },
      nullptr);
  IGL_DEBUG_ASSERT(vertexBuffer_ != nullptr);

// @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
// @fb-only

  // Pre-allocate index buffer (uint16: max vertices = 16384*4 = 65536 <= 65536)
  const size_t maxIndices = static_cast<size_t>(kMaxParticles) * 6;
  std::vector<uint16_t> indices(maxIndices);
  for (int32_t i = 0; i < kMaxParticles; i++) {
    const uint16_t base = static_cast<uint16_t>(i) * 4;
    const size_t idx = static_cast<size_t>(i) * 6;
    indices[idx + 0] = base + 0;
    indices[idx + 1] = base + 1;
    indices[idx + 2] = base + 2;
    indices[idx + 3] = base + 1;
    indices[idx + 4] = base + 3;
    indices[idx + 5] = base + 2;
  }
  indexBuffer_ = device.createBuffer(
      {
          .type = BufferDesc::BufferTypeBits::Index,
          .data = indices.data(),
          .length = maxIndices * sizeof(uint16_t),
          .storage = ibStorage,
          .debugName = "fireworks_indices",
      },
      nullptr);
  IGL_DEBUG_ASSERT(indexBuffer_ != nullptr);

  gpuVertices_.reserve(static_cast<size_t>(kMaxParticles));
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void FireworksSession::update(SurfaceTextures surfaceTextures) noexcept {
  if (!surfaceTextures.color) {
    return;
  }
  auto& device = getPlatform().getDevice();
  const auto dimensions = surfaceTextures.color->getDimensions();

  // Detect single-pass stereo: viewParams has 2 entries and surface texture is a 2-layer array
  const bool useStereo = shellParams().shellControlsViewParams &&
                         shellParams().viewParams.size() > 1 &&
                         surfaceTextures.color->getNumLayers() > 1;

  // Create/update framebuffer
  if (framebuffer_ == nullptr) {
    const auto mode = useStereo ? FramebufferMode::Stereo : FramebufferMode::Mono;
    framebuffer_ = device.createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color}},
            .mode = mode,
        },
        nullptr);
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  // Create pipeline on first use
  if (!renderPipelineState_) {
    auto shaderStages = getShaderStagesForBackend(device, useStereo);
    if (!shaderStages) {
      return;
    }

// @fb-only
    // @fb-only
        // @fb-only
// @fb-only
    const uint32_t vbIndex = 1u;
// @fb-only
    VertexInputStateDesc inputDesc = {
        .numAttributes = 4,
        .attributes =
            {
                {
                    .bufferIndex = vbIndex,
                    .format = VertexAttributeFormat::Float3,
                    .offset = offsetof(InterleavedVertex, pos),
                    .name = "pos",
                    .location = 0,
                },
                {
                    .bufferIndex = vbIndex,
                    .format = VertexAttributeFormat::Float3,
                    .offset = offsetof(InterleavedVertex, color),
                    .name = "color",
                    .location = 1,
                },
                {
                    .bufferIndex = vbIndex,
                    .format = VertexAttributeFormat::Float1,
                    .offset = offsetof(InterleavedVertex, flare),
                    .name = "flare",
                    .location = 2,
                },
                {
                    .bufferIndex = vbIndex,
                    .format = VertexAttributeFormat::Float2,
                    .offset = offsetof(InterleavedVertex, corner),
                    .name = "corner",
                    .location = 3,
                },
            },
        .numInputBindings = 1,
    };
// @fb-only
    // @fb-only
      // @fb-only
    // @fb-only
// @fb-only
    {
      inputDesc.inputBindings[vbIndex].stride = sizeof(InterleavedVertex);
    }
    vertexInput_ = device.createVertexInputState(inputDesc, nullptr);
    IGL_DEBUG_ASSERT(vertexInput_ != nullptr);

    const RenderPipelineDesc pipelineDesc = {
        .vertexInputState = vertexInput_,
        .shaderStages = std::move(shaderStages),
        .targetDesc =
            {
                .colorAttachments = {{
                    .textureFormat =
                        framebuffer_->getColorAttachment(0)
                            ? framebuffer_->getColorAttachment(0)->getProperties().format
                            : TextureFormat::Invalid,
                    .blendEnabled = true,
                    .rgbBlendOp = BlendOp::Add,
                    .alphaBlendOp = BlendOp::Add,
                    .srcRGBBlendFactor = BlendFactor::SrcAlpha,
                    .srcAlphaBlendFactor = BlendFactor::SrcAlpha,
                    .dstRGBBlendFactor = BlendFactor::One,
                    .dstAlphaBlendFactor = BlendFactor::One,
                }},
            },
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE("particleTex")}},
    };
    renderPipelineState_ = device.createRenderPipeline(pipelineDesc, nullptr);
    IGL_DEBUG_ASSERT(renderPipelineState_ != nullptr);
  }

  // Update uniforms: build per-eye MVP matrices
  const float aspectRatio =
      static_cast<float>(dimensions.width) / static_cast<float>(dimensions.height);
  // Scene offset: place particle origin 8m in front of the viewer
  const glm::mat4 sceneOffset = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -8.0f));
  const glm::mat4 monoProj = glm::perspective(glm::radians(90.0f), aspectRatio, 0.1f, 100.0f);

  Uniforms uniforms{};
  if (useStereo) {
    // Anchor the scene in world space on the first stereo frame:
    // compute a model matrix that places the fireworks 8m in front of the initial head pose.
    if (!sceneAnchored_) {
      const glm::mat4 headPose = glm::inverse(shellParams().viewParams[0].viewMatrix);
      sceneModelMatrix_ = headPose * sceneOffset;
      sceneAnchored_ = true;
    }
    // Single-pass stereo: fill both MVP matrices from shell-provided view params
    for (size_t i = 0; i < std::min(shellParams().viewParams.size(), static_cast<size_t>(2)); ++i) {
      const auto viewIdx = shellParams().viewParams[i].viewIndex;
      const glm::mat4 proj =
          perspectiveAsymmetricFovRH(shellParams().viewParams[i].fov, 0.1f, 100.0f);
      uniforms.mvp[viewIdx] = proj * shellParams().viewParams[i].viewMatrix * sceneModelMatrix_;
    }
  } else {
    uniforms.mvp[0] = monoProj * sceneOffset;
    uniforms.mvp[1] = uniforms.mvp[0];
  }

  // Compute head position and forward direction in scene-local space so
  // new rockets launch in front of wherever the user is currently looking,
  // and explosions orient toward the viewer.
  // In scene-local space, the initial head is at (0, 0, 8) looking toward -Z.
  // Fireworks originally launch 8 units ahead at (spreadX, -5, 0).
  glm::vec3 viewerLocalPos(0.0f, 0.0f, 8.0f); // default: initial head position
  glm::vec2 launchCenter(0.0f, 0.0f); // XZ center of launch area
  glm::vec2 launchPerp(1.0f, 0.0f); // perpendicular spread direction
  if (useStereo && sceneAnchored_) {
    const glm::mat4 invScene = glm::inverse(sceneModelMatrix_);
    const glm::mat4 headPose = glm::inverse(shellParams().viewParams[0].viewMatrix);
    // Head position in scene-local space
    viewerLocalPos = glm::vec3(invScene * headPose[3]);
    // Head forward direction in scene-local XZ plane
    const glm::vec3 fwd3 = glm::vec3(invScene * headPose * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));
    const glm::vec2 fwd(fwd3.x, fwd3.z);
    const float fwdLen = glm::length(fwd);
    if (fwdLen > 0.001f) {
      const glm::vec2 fwdNorm = fwd / fwdLen;
      // Launch center: 8 units ahead of head in XZ
      launchCenter = glm::vec2(viewerLocalPos.x, viewerLocalPos.z) + fwdNorm * 8.0f;
      // Spread direction: perpendicular to forward in XZ
      launchPerp = glm::vec2(-fwdNorm.y, fwdNorm.x);
    }
  }

  // Simulate particles (single-pass: update() called once per frame)
  {
    const float deltaSeconds = getDeltaSeconds();
    const glm::vec3 gravity(0.0f, -0.001f, 0.0f);

    accTime_ += deltaSeconds;

    while (accTime_ >= kTimeQuantum) {
      accTime_ -= kTimeQuantum;
      particleSystem_->nextFrame(gravity, viewerLocalPos, rng_);

      // Randomly shoot new fireworks in front of the user
      std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
      if (dist01(rng_) * 150.0f <= 1.0f) {
        const glm::vec3 baseColor(0.8f, 0.9f, 1.0f);
        const float spread = (dist01(rng_) * 100.0f - 50.0f) / 10.0f;
        const glm::vec2 launchXZ = launchCenter + launchPerp * spread;
        particleSystem_->addParticle({
            .pos = glm::vec3(launchXZ.x, -5.0f, launchXZ.y),
            .velocity = glm::vec3((dist01(rng_) * 100.0f - 50.0f) / 500.0f,
                                  0.25f + dist01(rng_) * 0.4f,
                                  (dist01(rng_) * 100.0f - 50.0f) / 500.0f),
            .baseColor = baseColor,
            .currentColor = baseColor,
            .ttl = 20,
            .initialLifetime = 20,
            .alive = true,
            .flare = true,
            .spawnExplosion = true,
        });
      }
    }
  }

  // Build vertex data (4 vertices per particle).
  size_t numParticles = 0;

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
    // clang-format off
    // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
    // @fb-only
    // clang-format on
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
  {
    // Collect alive particles into GPU vertex data
    gpuVertices_.clear();
    for (int32_t i = 0; i < kMaxParticles; i++) {
      if (particleSystem_->particles[i].alive) {
        const auto& p = particleSystem_->particles[i];
        gpuVertices_.push_back({
            .pos = p.pos,
            .color = p.currentColor,
            .flare = p.flare ? 1.0f : 0.0f,
        });
      }
    }
    numParticles = gpuVertices_.size();

    std::vector<InterleavedVertex> interleavedVerts;
    interleavedVerts.reserve(numParticles * 4);
    for (size_t i = 0; i < numParticles; i++) {
      const auto& gv = gpuVertices_[i];
      for (auto kCornerOffset : kCornerOffsets) {
        interleavedVerts.push_back({
            .pos = gv.pos,
            .color = gv.color,
            .flare = gv.flare,
            .corner = kCornerOffset,
        });
      }
    }

    if (!interleavedVerts.empty()) {
      vertexBuffer_->upload(interleavedVerts.data(),
                            BufferRange(interleavedVerts.size() * sizeof(InterleavedVertex), 0));
    }
  }

// @fb-only
  // @fb-only
  // @fb-only
// @fb-only
  {
    if (uniformBuffer_) {
      uniformBuffer_->upload(&uniforms, BufferRange(sizeof(Uniforms), 0));
    }
  }

  // Use shell-provided clear color (transparent for passthrough, black otherwise)
  if (shellParams().clearColorValue.has_value()) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto& c = shellParams().clearColorValue.value();
    renderPass_.colorAttachments[0].clearColor = {c.r, c.g, c.b, c.a};
  }

  // Render
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);

  const Viewport viewport = {.x = 0.0f,
                             .y = 0.0f,
                             .width = static_cast<float>(dimensions.width),
                             .height = static_cast<float>(dimensions.height),
                             .minDepth = 0.0f,
                             .maxDepth = +1.0f};
  const ScissorRect scissor = {.x = 0,
                               .y = 0,
                               .width = static_cast<uint32_t>(dimensions.width),
                               .height = static_cast<uint32_t>(dimensions.height)};

  const auto commands = buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    commands->bindRenderPipelineState(renderPipelineState_);
    commands->bindViewport(viewport);
    commands->bindScissorRect(scissor);
    commands->pushDebugGroupLabel("Fireworks", Color(1, 0.5f, 0));

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
    {
      commands->bindVertexBuffer(1, *vertexBuffer_);
      if (device.hasFeature(DeviceFeatures::UniformBlocks)) {
        commands->bindBuffer(0, uniformBuffer_.get());
      } else if (device.hasFeature(DeviceFeatures::BindUniform)) {
        const UniformDesc mvpDesc = {
            .location = renderPipelineState_->getIndexByName("mvp", ShaderStage::Vertex),
            .type = UniformType::Mat4x4,
            .numElements = 2,
            .offset = offsetof(Uniforms, mvp),
            .elementStride = sizeof(glm::mat4),
        };
        commands->bindUniform(mvpDesc, &uniforms);
      }
    }

    commands->bindTexture(0, BindTarget::kFragment, particleTexture_.get());
    commands->bindSamplerState(0, BindTarget::kFragment, sampler_.get());

    commands->bindIndexBuffer(*indexBuffer_, IndexFormat::UInt16);
    if (numParticles > 0) {
      commands->drawIndexed(numParticles * 6);
    }

    commands->popDebugGroupLabel();
    commands->endEncoding();
  }

  if (shellParams().shouldPresent) {
    buffer->present(surfaceTextures.color);
  }

  commandQueue_->submit(*buffer);
  RenderSession::update(surfaceTextures);
}

} // namespace igl::shell
