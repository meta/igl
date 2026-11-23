/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/ColorSession.h>

#include <cstring>
#include <glm/gtc/color_space.hpp>
#include <shell/shared/imageLoader/ImageLoader.h>
#include <shell/shared/platform/DisplayContext.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>
// @fb-only
// @fb-only
// @fb-only

namespace igl::shell {

namespace {

struct VertexPosUv {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float2 uv;
};
const VertexPosUv kVertexData[] = {
    {.position = {-1.f, 1.f, 0.0}, .uv = {0.0, 0.0}},
    {.position = {1.f, 1.f, 0.0}, .uv = {1.0, 0.0}},
    {.position = {-1.f, -1.f, 0.0}, .uv = {0.0, 1.0}},
    {.position = {1.f, -1.f, 0.0}, .uv = {1.0, 1.0}},
};
const uint16_t kIndexData[] = {0, 1, 2, 1, 3, 2};

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

BufferDesc getVertexBufferDesc(const IDevice& device) {
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
  return {.type = BufferDesc::BufferTypeBits::Vertex,
          .data = kVertexData,
          .length = sizeof(kVertexData),
          .debugName = "vertex"};
}

uint32_t getVertexBufferIndex(const IDevice& device) {
// @fb-only
  // @fb-only
    return 0;
  // @fb-only
// @fb-only
  return 1;
}

ResourceStorage getIndexBufferResourceStorage(const IDevice& device) {
// @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
// @fb-only
  return BufferDesc{}.storage;
}

std::string getVersion() {
  return "#version 100";
}

std::string getMetalShaderSource() {
  return R"(
              using namespace metal;

              typedef struct {
                 float3 color;
                 float4x4 mvp;
               } UniformBlock;

              typedef struct {
                float3 position [[attribute(0)]];
                float2 uv [[attribute(1)]];
              } VertexIn;

              typedef struct {
                float4 position [[position]];
                float2 uv;
              } VertexOut;

              vertex VertexOut vertexShader(
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]],
                  constant UniformBlock * ub [[buffer(0)]]) {
                VertexOut out;
                out.position = ub->mvp * float4(vertices[vid].position, 1.0);
                out.uv = vertices[vid].uv;
                return out;
              }

              fragment float4 fragmentShader(
                  VertexOut IN [[stage_in]],
                  texture2d<float> diffuseTex [[texture(0)]],
                  sampler linearSampler [[sampler(0)]],
                  constant UniformBlock * ub [[buffer(0)]]) {
                float4 tex = diffuseTex.sample(linearSampler, IN.uv);
                return float4(ub->color.r, ub->color.g, ub->color.b, 1.0) *
                      tex;
              }
    )";
}

std::string getMetalShaderSourceGradient() {
  return R"(
              using namespace metal;

              typedef struct {
                 float3 color;
                 float4x4 mvp;
               } UniformBlock;

              typedef struct {
                float3 position [[attribute(0)]];
                float2 uv [[attribute(1)]];
              } VertexIn;

              typedef struct {
                float4 position [[position]];
                float2 uv;
              } VertexOut;

              vertex VertexOut vertexShader(
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]]) {
                VertexOut out;
                out.position = float4(vertices[vid].position, 1.0);
                out.uv = vertices[vid].uv;
                return out;
              }

              fragment float4 fragmentShader(
                  VertexOut IN [[stage_in]],
                  texture2d<float> diffuseTex [[texture(0)]],
                  sampler linearSampler [[sampler(0)]],
                  constant UniformBlock * color [[buffer(0)]]) {

                  float numSteps = 20.0;
                  float uvX;
                  if (IN.uv.y<0.25) {
                   uvX = IN.uv.x;
                  } else if (IN.uv.y<0.5) {
                    uvX = floor(IN.uv.x*numSteps+0.5)/numSteps;
                  } else if (IN.uv.y<0.75) {
                    uvX = 1.0-IN.uv.x;
                  } else {
                    uvX = floor((1.0-IN.uv.x)*numSteps+0.5)/numSteps;
                  }
                  return float4(uvX, uvX, uvX, 1.0);
              }
    )";
}

std::string getOpenGLVertexShaderSource() {
  return getVersion() + R"(
                precision highp float;
                attribute vec3 position;
                attribute vec2 uv_in;

                uniform vec3 color;
                uniform mat4 mvp;
                uniform sampler2D inputImage;

                varying vec3 vColor;
                varying vec2 uv;

                void main() {
                  gl_Position = mvp * vec4(position, 1.0);
                  uv = uv_in; // position.xy * 0.5 + 0.5;
                  vColor = color;
                })";
}

std::string getOpenGLFragmentShaderSource() {
  return getVersion() + std::string(R"(
                precision highp float;
                uniform vec3 color;
                uniform mat4 mvp;
                uniform sampler2D inputImage;
                varying vec3 vColor;
                varying vec2 uv;

                void main() {
                  gl_FragColor =
                      vec4(vColor, 1.0) * texture2D(inputImage, uv);
                })");
}

std::string getOpenGLFragmentShaderSourceGradient() {
  return getVersion() + R"(
                precision highp float;
                uniform vec3 color;
                uniform mat4 mvp;
                uniform sampler2D inputImage;
                varying vec3 vColor;
                varying vec2 uv;

                void main() {
                  float numSteps = 20.0;
                  float uvX;
                  if (uv.y<0.25) {
                    uvX = uv.x;
                  } else if (uv.y<0.5) {
                   uvX = floor(uv.x*numSteps+0.5)/numSteps;
                  } else if (uv.y<0.75) {
                   uvX = 1.0-uv.x;
                  } else {
                   uvX = floor((1.0-uv.x)*numSteps+0.5)/numSteps;
                  }
                  gl_FragColor = vec4(vec3(uvX), 1.0);
                }
                )";
}

std::string getVulkanVertexShaderSource() {
  return R"(precision highp float;
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 uv_in;
            layout(location = 0) out vec2 uv;
            layout(location = 1) out vec3 color;

            layout (set = 1, binding = 0, std140) uniform UniformsPerObject {
              vec3 color;
              mat4 mvp;
            } perObject;

            void main() {
              gl_Position = perObject.mvp * vec4(position, 1.0);
              uv = uv_in;
              color = perObject.color;
            }
            )";
}

std::string getVulkanFragmentShaderSource() {
  return R"(
                layout(location = 0) in vec2 uv;
                layout(location = 1) in vec3 color;
                layout(location = 0) out vec4 out_FragColor;

                layout(set = 0, binding = 0) uniform sampler2D in_texture;

                void main() {
                  out_FragColor = vec4(color, 1.0) * texture(in_texture, uv);
                }
                )";
}

std::string getVulkanFragmentShaderSourceGradient() {
  return R"(
                layout(location = 0) in vec2 uv;
                layout(location = 1) in vec3 color;
                layout(location = 0) out vec4 out_FragColor;

                void main() {
                  float numSteps = 20.0;
                  float uvX;
                  if (uv.y<0.25) {
                    uvX = uv.x;
                  } else if (uv.y<0.5) {
                   uvX = floor(uv.x*numSteps+0.5)/numSteps;
                  } else if (uv.y<0.75) {
                   uvX = 1.0-uv.x;
                  } else {
                   uvX = floor((1.0-uv.x)*numSteps+0.5)/numSteps;
                  }
                  out_FragColor = vec4(vec3(uvX), 1.0);
                }
                )";
}

// Bind convention: texture unit 0 at @group(0) @binding(0), its sampler at @binding(1), buffer 0
// at @group(1) @binding(0).
const char* getWgslShaderSource() {
  return R"(
struct UniformBlock {
  color : vec3f,
  mvp : mat4x4f,
};

@group(1) @binding(0) var<uniform> ub : UniformBlock;
@group(0) @binding(0) var diffuseTex : texture_2d<f32>;
@group(0) @binding(1) var linearSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vertexShader(@location(0) position : vec3f, @location(1) uv : vec2f) -> VertexOut {
  return VertexOut(ub.mvp * vec4f(position, 1.0), uv);
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return vec4f(ub.color, 1.0) * textureSample(diffuseTex, linearSampler, v.uv);
}

@vertex
fn vertexShaderGradient(@location(0) position : vec3f, @location(1) uv : vec2f) -> VertexOut {
  return VertexOut(vec4f(position, 1.0), uv);
}

@fragment
fn fragmentShaderGradient(v : VertexOut) -> @location(0) vec4f {
  let numSteps = 20.0;
  var uvX : f32;
  if (v.uv.y < 0.25) {
    uvX = v.uv.x;
  } else if (v.uv.y < 0.5) {
    uvX = floor(v.uv.x * numSteps + 0.5) / numSteps;
  } else if (v.uv.y < 0.75) {
    uvX = 1.0 - v.uv.x;
  } else {
    uvX = floor((1.0 - v.uv.x) * numSteps + 0.5) / numSteps;
  }
  return vec4f(vec3f(uvX), 1.0);
}
)";
}

// @fb-only

} // namespace

std::unique_ptr<IShaderStages> ColorSession::getShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU: {
    const bool gradient = colorTestModes_ == ColorTestModes::Gradient;
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device,
        getWgslShaderSource(),
        gradient ? "vertexShaderGradient" : "vertexShader",
        gradient ? "fragmentShaderGradient" : "fragmentShader",
        "",
        nullptr);
  }
  case BackendType::Vulkan: {
    auto vertexSource = getVulkanVertexShaderSource();
    if (device.hasFeature(DeviceFeatures::Multiview)) {
      vertexSource = R"(#version 450
                        #extension GL_OVR_multiview2 : require
                        layout(num_views = 2) in;)" +
                     vertexSource;
    }
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        vertexSource.c_str(),
        "main",
        "",
        colorTestModes_ == ColorTestModes::Gradient
            ? getVulkanFragmentShaderSourceGradient().c_str()
            : getVulkanFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
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
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device,
        colorTestModes_ == ColorTestModes::Gradient ? getMetalShaderSourceGradient().c_str()
                                                    : getMetalShaderSource().c_str(),
        "vertexShader",
        "fragmentShader",
        "",
        nullptr);
  case BackendType::D3D12: {
    if (colorTestModes_ == ColorTestModes::Gradient) {
      // Gradient mode - no MVP matrix, just positional gradient
      static const char* kVS = R"(
        cbuffer UniformBlock : register(b0) {
          float3 color;
          float4x4 mvp;
        };

        struct VSInput {
          float3 position : POSITION;
          float2 uv : TEXCOORD0;
        };

        struct VSOutput {
          float4 position : SV_POSITION;
          float2 uv : TEXCOORD0;
        };

        VSOutput main(VSInput input) {
          VSOutput output;
          output.position = float4(input.position, 1.0);
          output.uv = input.uv;
          return output;
        }
      )";

      static const char* kPS = R"(
        cbuffer UniformBlock : register(b0) {
          float3 color;
          float4x4 mvp;
        };

        Texture2D diffuseTex : register(t0);
        SamplerState linearSampler : register(s0);

        struct PSInput {
          float4 position : SV_POSITION;
          float2 uv : TEXCOORD0;
        };

        float4 main(PSInput input) : SV_Target {
          float numSteps = 20.0;
          float uvX;
          if (input.uv.y < 0.25) {
            uvX = input.uv.x;
          } else if (input.uv.y < 0.5) {
            uvX = floor(input.uv.x * numSteps) / numSteps;
          } else if (input.uv.y < 0.75) {
            uvX = (floor(input.uv.x * numSteps) + 0.5) / numSteps;
          } else {
            uvX = 1.0 - input.uv.x;
          }
          return float4(uvX, input.uv.y, 0.5, 1.0);
        }
      )";

      return igl::ShaderStagesCreator::fromModuleStringInput(
          device, kVS, "main", "", kPS, "main", "", nullptr);
    } else {
      // Regular textured mode
      static const char* kVS = R"(
        cbuffer UniformBlock : register(b0) {
          float3 color;
          float4x4 mvp;
        };

        struct VSInput {
          float3 position : POSITION;
          float2 uv : TEXCOORD0;
        };

        struct VSOutput {
          float4 position : SV_POSITION;
          float2 uv : TEXCOORD0;
        };

        VSOutput main(VSInput input) {
          VSOutput output;
          output.position = mul(mvp, float4(input.position, 1.0));
          output.uv = input.uv;
          return output;
        }
      )";

      static const char* kPS = R"(
        cbuffer UniformBlock : register(b0) {
          float3 color;
          float4x4 mvp;
        };

        Texture2D diffuseTex : register(t0);
        SamplerState linearSampler : register(s0);

        struct PSInput {
          float4 position : SV_POSITION;
          float2 uv : TEXCOORD0;
        };

        float4 main(PSInput input) : SV_Target {
          float4 tex = diffuseTex.Sample(linearSampler, input.uv);
          return float4(color.r, color.g, color.b, 1.0) * tex;
        }
      )";

      return igl::ShaderStagesCreator::fromModuleStringInput(
          device, kVS, "main", "", kPS, "main", "", nullptr);
    }
  }
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getOpenGLVertexShaderSource().c_str(),
        "main",
        "",
        colorTestModes_ == ColorTestModes::Gradient
            ? getOpenGLFragmentShaderSourceGradient().c_str()
            : getOpenGLFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void ColorSession::initializeImpl() noexcept {
  IDevice& device = getPlatform().getDevice();
  const glm::vec3 fLinearOrangeColor =
      (swapchainColorTextureformat_ == TextureFormat::RGBA_SRGB &&
       device.hasFeature(DeviceFeatures::SRGB))
          ? glm::vec3(glm::convertSRGBToLinear(glm::dvec3{1.0, 0.5, 0.0}))
          : glm::vec3{1.0f, 0.5f, 0.0f};
  const iglu::simdtypes::float3 gpuLinearOrangeColor = {
      fLinearOrangeColor.x, fLinearOrangeColor.y, fLinearOrangeColor.z};

  // Vertex & Index buffer
  vb0_ = device.createBuffer(getVertexBufferDesc(device), nullptr);
  IGL_DEBUG_ASSERT(vb0_ != nullptr);
  ib0_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                        .data = kIndexData,
                                        .length = sizeof(kIndexData),
                                        .storage = getIndexBufferResourceStorage(device),
                                        .debugName = "index"},
                             nullptr);
  IGL_DEBUG_ASSERT(ib0_ != nullptr);

  const uint32_t vertexBufferIndex = getVertexBufferIndex(device);
  VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes =
          {
              {.bufferIndex = vertexBufferIndex,
               .format = VertexAttributeFormat::Float3,
               .offset = offsetof(VertexPosUv, position),
               .name = "position",
               .location = 0},
              {.bufferIndex = vertexBufferIndex,
               .format = VertexAttributeFormat::Float2,
               .offset = offsetof(VertexPosUv, uv),
               .name = "uv_in",
               .location = 1},
          },
      .numInputBindings = 1,
  };
  inputDesc.inputBindings[vertexBufferIndex].stride = sizeof(VertexPosUv);
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);
  IGL_DEBUG_ASSERT(vertexInput0_ != nullptr);

  // Sampler & Texture
  samp0_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
          .debugName = "Sampler: linear",
      },
      nullptr);
  IGL_DEBUG_ASSERT(samp0_ != nullptr);

  // OpenGL ES 2.0 without GL_OES_texture_npot cannot mipmap the non-power-of-two images; the
  // sampler does not use mipmaps anyway.
  const bool withMipmaps = device.hasFeature(DeviceFeatures::TextureNotPot);
  if (colorTestModes_ == ColorTestModes::MacbethTexture) {
    tex0_ = getPlatform().loadTexture("macbeth.png", withMipmaps, swapchainColorTextureformat_);
  } else if (colorTestModes_ == ColorTestModes::MacbethTexture720) {
    tex0_ = getPlatform().loadTexture("macbeth720.png", withMipmaps, swapchainColorTextureformat_);
  } else if (colorTestModes_ == ColorTestModes::MacbethTextureKtx) {
    tex0_ = getPlatform().loadTexture("macbeth.ktx", withMipmaps, swapchainColorTextureformat_);
  } else if (colorTestModes_ == ColorTestModes::MacbethTextureKtx2) {
    tex0_ = getPlatform().loadTexture("macbeth.ktx2", withMipmaps, swapchainColorTextureformat_);
  // @fb-only
    // @fb-only
        // @fb-only
  // @fb-only
    // @fb-only
        // @fb-only
  } else if (colorTestModes_ == ColorTestModes::EyeChartTexture720) {
    tex0_ = getPlatform().loadTexture("eyechart720.png", withMipmaps, swapchainColorTextureformat_);
  } else if (colorTestModes_ == ColorTestModes::OrangeTexture) {
    tex0_ = getPlatform().loadTexture("orange.png", withMipmaps, swapchainColorTextureformat_);
  } else if (colorTestModes_ == ColorTestModes::OrangeClear) {
    tex0_ = getPlatform().loadTexture(ImageLoader::white());
    setPreferredClearColor(
        Color{fLinearOrangeColor.x, fLinearOrangeColor.y, fLinearOrangeColor.z, 1.0f});
  } else if (colorTestModes_ == ColorTestModes::Gradient) {
    tex0_ = getPlatform().loadTexture(ImageLoader::white());
  }
  shaderStages_ = getShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = getPreferredClearColor(),
      }},
      .depthAttachment = {.loadAction = LoadAction::Clear, .clearDepth = 1.0},
  };

  // init uniforms
  const glm::mat4x4 mvp(1.0f);
  std::memcpy(&fragmentParameters_.mvp, &mvp, sizeof(mvp));
  fragmentParameters_.color = (colorTestModes_ == ColorTestModes::OrangeClear)
                                  ? gpuLinearOrangeColor
                                  : iglu::simdtypes::float3{1.0f, 1.0f, 1.0f};

  fragmentParamBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                                                        .data = &fragmentParameters_,
                                                        .length = sizeof(fragmentParameters_),
                                                        .storage = ResourceStorage::Shared,
                                                        .debugName = "uniforms"},
                                             nullptr);
  IGL_DEBUG_ASSERT(fragmentParamBuffer_ != nullptr);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void ColorSession::updateImpl(const SurfaceTextures& surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  if (framebuffer_ == nullptr) {
    Result ret;
    framebuffer_ = getPlatform().getDevice().createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color}},
            .depthAttachment = {.texture = surfaceTextures.depth},
            .mode = surfaceTextures.color->getNumLayers() > 1 ? FramebufferMode::Stereo
                                                              : FramebufferMode::Mono,
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  const size_t textureUnit = 0;

  // Graphics pipeline
  if (pipelineState_ == nullptr) {
    pipelineState_ = getPlatform().getDevice().createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInput0_,
            .shaderStages = shaderStages_,
            .targetDesc =
                {
                    .colorAttachments = {{
                        .textureFormat =
                            framebuffer_->getColorAttachment(0)->getProperties().format,
                        .blendEnabled = true,
                        .rgbBlendOp = BlendOp::Add,
                        .alphaBlendOp = BlendOp::Add,
                        .srcRGBBlendFactor = BlendFactor::SrcAlpha,
                        .srcAlphaBlendFactor = BlendFactor::SrcAlpha,
                        .dstRGBBlendFactor = BlendFactor::OneMinusSrcAlpha,
                        .dstAlphaBlendFactor = BlendFactor::OneMinusSrcAlpha,
                    }},
                    .depthAttachmentFormat =
                        framebuffer_->getDepthAttachment()->getProperties().format,
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
            .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputImage")}},
        },
        nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  // Command Buffers
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  const auto drawableSurface = framebuffer_->getColorAttachment(0);

  framebuffer_->updateDrawable(drawableSurface);

  fragmentUniformDescriptors_.clear();
  // Uniform: "color"
  fragmentUniformDescriptors_.emplace_back();
  // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
  // @fb-only
    if (getPlatform().getDevice().hasFeature(DeviceFeatures::BindUniform)) {
      fragmentUniformDescriptors_.back().location =
          pipelineState_->getIndexByName("color", ShaderStage::Fragment);
    }
  fragmentUniformDescriptors_.back().type = UniformType::Float3;
  fragmentUniformDescriptors_.back().offset = offsetof(FragmentFormat, color);

  // Uniform: "mvp"
  fragmentUniformDescriptors_.emplace_back();
  // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
  // @fb-only
    if (getPlatform().getDevice().hasFeature(DeviceFeatures::BindUniform)) {
      fragmentUniformDescriptors_.back().location =
          pipelineState_->getIndexByName("mvp", ShaderStage::Fragment);
    }
  fragmentUniformDescriptors_.back().type = UniformType::Mat4x4;
  fragmentUniformDescriptors_.back().offset = offsetof(FragmentFormat, mvp);

  const auto& mvp = getPlatform().getDisplayContext().preRotationMatrix;
  std::memcpy(&fragmentParameters_.mvp, &mvp, sizeof(mvp));
  fragmentParamBuffer_->upload(&fragmentParameters_, {sizeof(fragmentParameters_)});

  // Submit commands
  const auto commands = buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    commands->bindVertexBuffer(getVertexBufferIndex(getPlatform().getDevice()), *vb0_);
    commands->bindRenderPipelineState(pipelineState_);
    if (getPlatform().getDevice().hasFeature(DeviceFeatures::BindUniform)) {
      // Bind non block uniforms
      for (const auto& uniformDesc : fragmentUniformDescriptors_) {
        // Uniforms the shader compiler optimized out have no location.
        if (uniformDesc.location >= 0) {
          commands->bindUniform(uniformDesc, &fragmentParameters_);
        }
      }
    } else if (getPlatform().getDevice().hasFeature(DeviceFeatures::UniformBlocks)) {
      // @fb-only
        // @fb-only
                            // @fb-only
                            // @fb-only
                            // @fb-only
      // @fb-only
        commands->bindBuffer(0, fragmentParamBuffer_.get());
      // @fb-only
    } else {
      IGL_DEBUG_ASSERT_NOT_REACHED();
    }
    // if (colorTestModes_ != ColorTestModes::eGradient) {
    commands->bindTexture(textureUnit, BindTarget::kFragment, tex0_.get());
    commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());
    //}
    commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
    commands->drawIndexed(6);

    commands->endEncoding();
  }

  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  // Use async submission in headless mode to avoid vsync-locked waitUntilCompleted
  const bool waitForCompletion = !shellParams().isHeadless;
  commandQueue_->submit(*buffer, waitForCompletion);
  RenderSession::update(surfaceTextures);
}

} // namespace igl::shell
