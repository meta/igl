/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/TQSession.h>

#include <IGLU/simdtypes/SimdTypes.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>
// @fb-only
// @fb-only
// @fb-only
// @fb-only

namespace igl::shell {
namespace {
struct VertexPosUv {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float2 uv;
};

std::string getVersion() {
  return {"#version 100"};
}

std::string getMetalShaderSource() {
  return R"(
              using namespace metal;

              typedef struct { float3 color; } UniformBlock;

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
                float4 tex = diffuseTex.sample(linearSampler, IN.uv);
                return float4(color->color.r, color->color.g, color->color.b, 1.0) *
                      tex;
              }
              )";
}

std::string getD3D12VertexShaderSource() {
  return R"(
                struct UniformsPerObject {
                  float3 color;
                };

                cbuffer PerObject : register(b0) {
                  UniformsPerObject perObject;
                };

                struct VSInput {
                  float3 position : POSITION;
                  float2 uv_in : TEXCOORD0;
                };

                struct VSOutput {
                  float4 position : SV_POSITION;
                  float2 uv : TEXCOORD0;
                  float3 color : COLOR0;
                };

                VSOutput main(VSInput input) {
                  VSOutput output;
                  output.position = float4(input.position, 1.0);
                  output.uv = input.uv_in;
                  output.color = perObject.color;
                  return output;
                }
                )";
}

std::string getD3D12FragmentShaderSource() {
  return R"(
                Texture2D in_texture : register(t0);
                SamplerState in_sampler : register(s0);

                struct PSInput {
                  float4 position : SV_POSITION;
                  float2 uv : TEXCOORD0;
                  float3 color : COLOR0;
                };

                float4 main(PSInput input) : SV_Target {
                  return float4(input.color, 1.0) * in_texture.Sample(in_sampler, input.uv);
                }
                )";
}

std::string getOpenGLVertexShaderSource() {
  return getVersion() + R"(
                precision highp float;
                attribute vec3 position;
                attribute vec2 uv_in;

                varying vec2 uv;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  uv = uv_in; // position.xy * 0.5 + 0.5;
                })";
}

std::string getOpenGLFragmentShaderSource() {
  return getVersion() + std::string(R"(
                precision highp float;
                uniform vec3 color;
                uniform sampler2D inputImage;

                varying vec2 uv;

                void main() {
                  gl_FragColor =
                      vec4(color, 1.0) * texture2D(inputImage, uv);
                })");
}

std::string getVulkanVertexShaderSource() {
  return R"(
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec2 uv_in;
                layout(location = 0) out vec2 uv;
                layout(location = 1) out vec3 color;

                struct UniformsPerObject {
                  vec3 color;
                };

                layout (set = 1, binding = 0, std140) uniform PerObject {
                  UniformsPerObject perObject;
                } object;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  uv = uv_in;
                  color = object.perObject.color;
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

// Bind convention: texture unit 0 at @group(0) @binding(0), its sampler at @binding(1), buffer 0
// at @group(1) @binding(0).
const char* getWgslShaderSource() {
  return R"(
struct UniformBlock {
  color : vec3f,
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
  return VertexOut(vec4f(position, 1.0), uv);
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return vec4f(ub.color, 1.0) * textureSample(diffuseTex, linearSampler, v.uv);
}
)";
}

std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getVulkanVertexShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           getVulkanFragmentShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           nullptr);
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
        device, getMetalShaderSource().c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getOpenGLVertexShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           getOpenGLFragmentShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           nullptr);
  case BackendType::D3D12:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getD3D12VertexShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           getD3D12FragmentShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           nullptr);
  }
  IGL_UNREACHABLE_RETURN(nullptr)
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

BufferDesc getVertexBufferDesc(const IDevice& device, const VertexPosUv* vertexData) {
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
          .data = vertexData,
          .length = sizeof(VertexPosUv) * 4};
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
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
void TQSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Vertex & Index buffer
  const VertexPosUv vertexData[] = {
      {.position = {-0.8f, 0.8f, 0.0}, .uv = {0.0, 0.0}},
      {.position = {0.8f, 0.8f, 0.0}, .uv = {uvScale_, 0.0}},
      {.position = {-0.8f, -0.8f, 0.0}, .uv = {0.0, uvScale_}},
      {.position = {0.8f, -0.8f, 0.0}, .uv = {uvScale_, uvScale_}},
  };
  const BufferDesc vbDesc = getVertexBufferDesc(device, &vertexData[0]);
  vb0_ = device.createBuffer(vbDesc, nullptr);
  IGL_DEBUG_ASSERT(vb0_ != nullptr);
  const uint16_t indexData[] = {0, 1, 2, 1, 3, 2};
  const BufferDesc ibDesc{.type = BufferDesc::BufferTypeBits::Index,
                          .data = indexData,
                          .length = sizeof(indexData),
                          .storage = getIndexBufferResourceStorage(device)};
  ib0_ = device.createBuffer(ibDesc, nullptr);
  IGL_DEBUG_ASSERT(ib0_ != nullptr);

  const auto vertexBufferIndex = getVertexBufferIndex(getPlatform().getDevice());
  VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes =
          {
              {
                  .bufferIndex = vertexBufferIndex,
                  .format = VertexAttributeFormat::Float3,
                  .offset = offsetof(VertexPosUv, position),
                  .name = "position",
                  .location = 0,
              },
              {
                  .bufferIndex = vertexBufferIndex,
                  .format = VertexAttributeFormat::Float2,
                  .offset = offsetof(VertexPosUv, uv),
                  .name = "uv_in",
                  .location = 1,
              },
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
  tex0_ = getPlatform().loadTexture("igl.png");

  shaderStages_ = getShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  // Generate mipmaps for texture for D3D12
  if (device.getBackendType() == BackendType::D3D12) {
    tex0_->generateMipmap(*commandQueue_);
  }

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = getPreferredClearColor(),
      }},
      .depthAttachment = {.loadAction = LoadAction::Clear, .clearDepth = 1.0},
  };

  // init uniforms
  fragmentParameters_ = FragmentFormat{.color = {1.0f, 1.0f, 1.0f}};

  const BufferDesc fpDesc{
      .type = BufferDesc::BufferTypeBits::Uniform,
      .data = &fragmentParameters_,
      .length = sizeof(fragmentParameters_),
      .storage = ResourceStorage::Shared,
  };

  fragmentParamBuffer_ = device.createBuffer(fpDesc, nullptr);
  IGL_DEBUG_ASSERT(fragmentParamBuffer_ != nullptr);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void TQSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  Result ret;
  if (framebuffer_ == nullptr) {
    FramebufferDesc framebufferDesc{
        .colorAttachments = {{.texture = surfaceTextures.color}},
        .depthAttachment = {.texture = surfaceTextures.depth},
    };
    if (surfaceTextures.depth && surfaceTextures.depth->getProperties().hasStencil()) {
      framebufferDesc.stencilAttachment.texture = surfaceTextures.depth;
    }
    IGL_DEBUG_ASSERT(ret.isOk());
    framebuffer_ = getPlatform().getDevice().createFramebuffer(framebufferDesc, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures);
  }

  const size_t textureUnit = 0;

  // Graphics pipeline
  if (pipelineState_ == nullptr) {
    const RenderPipelineDesc graphicsDesc = {
        .vertexInputState = vertexInput0_,
        .shaderStages = shaderStages_,
        .targetDesc =
            {
                .colorAttachments = {{.textureFormat =
                                          framebuffer_->getColorAttachment(0)->getFormat()}},
                .depthAttachmentFormat = framebuffer_->getDepthAttachment()->getFormat(),
                .stencilAttachmentFormat = framebuffer_->getStencilAttachment()
                                               ? framebuffer_->getStencilAttachment()->getFormat()
                                               : TextureFormat::Invalid,
            },
        .cullMode = CullMode::Back,
        .frontFaceWinding = WindingMode::Clockwise,
        .fragmentUnitSamplerMap =
            {
                std::pair<size_t, NameHandle>(textureUnit, IGL_NAMEHANDLE("inputImage")),
            },
    };

    pipelineState_ = getPlatform().getDevice().createRenderPipeline(graphicsDesc, nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);

    // Set up uniformdescriptors
    fragmentUniformDescriptors_.emplace_back();
  }

  // Command Buffers
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  const auto drawableSurface = framebuffer_->getColorAttachment(0);

  // Uniform: "color"
  if (!fragmentUniformDescriptors_.empty()) {
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
  }

  // Submit commands
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    commands->bindVertexBuffer(getVertexBufferIndex(getPlatform().getDevice()), *vb0_);
    commands->bindRenderPipelineState(pipelineState_);
    if (getPlatform().getDevice().hasFeature(DeviceFeatures::BindUniform)) {
      // Bind non block uniforms
      for (const auto& uniformDesc : fragmentUniformDescriptors_) {
        commands->bindUniform(uniformDesc, &fragmentParameters_);
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

    commands->bindTexture(textureUnit, BindTarget::kFragment, tex0_.get());
    commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());
    commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
    commands->drawIndexed(6);

    commands->endEncoding();
  }

  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer, true);
  RenderSession::update(surfaceTextures);
}

} // namespace igl::shell
