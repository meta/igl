/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/TQMultiRenderPassSession.h>

#include <IGLU/simdtypes/SimdTypes.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {
namespace {
struct VertexPosUv {
  iglu::simdtypes::float3 position; // SIMD 128b aligned
  iglu::simdtypes::float2 uv; // SIMD 128b aligned
};
// clang-format off
const VertexPosUv kVertexData0[] = {
    {.position = {-1.0f,  1.0f, 0.0f}, .uv = {0.0f, 1.0f}},
    {.position = { 1.0f,  1.0f, 0.0f}, .uv = {1.0f, 1.0f}},
    {.position = {-1.0f, -1.0f, 0.0f}, .uv = {0.0f, 0.0f}},
    {.position = { 1.0f, -1.0f, 0.0f}, .uv = {1.0f, 0.0f}},
};
const VertexPosUv kVertexData1[] = {
    {.position = {-0.8f,  0.8f, 0.0f}, .uv = {0.0f, 1.0f}},
    {.position = { 0.8f,  0.8f, 0.0f}, .uv = {1.0f, 1.0f}},
    {.position = {-0.8f, -0.8f, 0.0f}, .uv = {0.0f, 0.0f}},
    {.position = { 0.8f, -0.8f, 0.0f}, .uv = {1.0f, 0.0f}},
};
constexpr uint16_t kIndexData[] = {0, 1, 2, 1, 3, 2};
// clang-format on

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
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(0)]]) {
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

std::string getOpenGLVertexShaderSource() {
  return R"(#version 100
                attribute vec3 position;
                attribute vec2 uv_in;

                varying vec2 uv;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  uv = uv_in; // position.xy * 0.5 + 0.5;
                })";
}

std::string getOpenGLFragmentShaderSource() {
  return R"(#version 100
                precision highp float;

                uniform vec3 color;
                uniform sampler2D inputImage;

                varying vec2 uv;

                void main() {
                  gl_FragColor =
                      vec4(color, 1.0) * texture2D(inputImage, uv);
                })";
}

std::string getD3D12VertexShaderSource() {
  return R"(
struct VertexIn {
  float3 position : POSITION;
  float2 uv : TEXCOORD0;
};

struct VertexOut {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

VertexOut main(VertexIn IN) {
  VertexOut OUT;
  OUT.position = float4(IN.position, 1.0);
  OUT.uv = IN.uv;
  return OUT;
}
)";
}

std::string getD3D12FragmentShaderSource() {
  return R"(
cbuffer UniformBlock : register(b0) {
  float3 color;
};

Texture2D inputImage : register(t0);
SamplerState linearSampler : register(s0);

struct VertexOut {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

float4 main(VertexOut IN) : SV_Target {
  float4 tex = inputImage.Sample(linearSampler, IN.uv);
  return float4(color.r, color.g, color.b, 1.0) * tex;
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
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::Vulkan:
    IGL_LOG_INFO("TQMultiRenderPassSession: no Vulkan shaders; skipping\n");
    return nullptr;
  case BackendType::Custom:
    IGL_DEBUG_ABORT("IGLSamples not set up for Custom");
    return nullptr;
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
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

void render(std::shared_ptr<ICommandBuffer>& buffer,
            std::shared_ptr<IBuffer>& vertexBuffer,
            std::shared_ptr<ITexture>& inputTexture,
            std::shared_ptr<IRenderPipelineState>& pipelineState,
            std::shared_ptr<IFramebuffer>& framebuffer,
            RenderPassDesc& renderPass,
            std::shared_ptr<ISamplerState>& samplerState,
            std::shared_ptr<IBuffer>& ib,
            size_t textureUnit,
            BackendType& backend,
            std::shared_ptr<IBuffer>& fragmentParamBuffer,
            std::vector<UniformDesc>& fragmentUniformDescriptors,
            FragmentFormat fragmentParameters) {
  // Submit commands
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass, framebuffer);

  commands->bindRenderPipelineState(pipelineState);

  if (backend != BackendType::OpenGL) {
    if (fragmentParamBuffer) {
      commands->bindBuffer(0, fragmentParamBuffer.get());
    }
  } else {
    // Bind non block uniforms
    for (const auto& uniformDesc : fragmentUniformDescriptors) {
      commands->bindUniform(uniformDesc, &fragmentParameters);
    }
  }
  commands->bindTexture(textureUnit, BindTarget::kFragment, inputTexture.get());
  commands->bindSamplerState(textureUnit, BindTarget::kFragment, samplerState.get());
  commands->bindVertexBuffer(0, *vertexBuffer);
  commands->bindIndexBuffer(*ib, IndexFormat::UInt16);
  commands->drawIndexed(6);
  commands->endEncoding();
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
void TQMultiRenderPassSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Vertex buffer, Index buffer and Vertex Input
  vb0_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                        .data = kVertexData0,
                                        .length = sizeof(kVertexData0)},
                             nullptr);
  vb1_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                        .data = kVertexData1,
                                        .length = sizeof(kVertexData1)},
                             nullptr);
  ib0_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                        .data = kIndexData,
                                        .length = sizeof(kIndexData)},
                             nullptr);

  const VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes =
          {
              {
                  .bufferIndex = 0,
                  .format = VertexAttributeFormat::Float3,
                  .offset = offsetof(VertexPosUv, position),
                  .name = "position",
                  .location = 0,
              },
              {
                  .bufferIndex = 0,
                  .format = VertexAttributeFormat::Float2,
                  .offset = offsetof(VertexPosUv, uv),
                  .name = "uv_in",
                  .location = 1,
              },
          },
      .numInputBindings = 1,
      .inputBindings =
          {
              {
                  .stride = sizeof(VertexPosUv),
              },
          },
  };
  vertexInputState_ = device.createVertexInputState(inputDesc, nullptr);

  // Sampler & Texture
  samplerState_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
          .debugName = "Sampler: linear",
      },
      nullptr);
  tex0_ = getPlatform().loadTexture("igl.png");

  shaderStages_ = getShaderStagesForBackend(device);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);

  renderPass0_ = {
      .colorAttachments =
          {
              {
                  .loadAction = LoadAction::Clear,
                  .storeAction = StoreAction::Store,
                  .clearColor = getPreferredClearColor(),
              },
          },
      .depthAttachment = {.loadAction = LoadAction::Clear, .clearDepth = 1.0},
  };

  renderPass1_ = {
      .colorAttachments =
          {
              {
                  .loadAction = LoadAction::Clear,
                  .storeAction = StoreAction::Store,
                  .clearColor = getPreferredClearColor(),
              },
          },
      .depthAttachment = {.loadAction = LoadAction::Clear, .clearDepth = 1.0},
  };

  // init uniforms
  fragmentParameters_ = FragmentFormat{.color = {1.0f, 1.0f, 1.0f}};

  fragmentParamBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                                                        .data = &fragmentParameters_,
                                                        .length = sizeof(fragmentParameters_),
                                                        .storage = ResourceStorage::Shared},
                                             nullptr);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void TQMultiRenderPassSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color || !shaderStages_) {
    return;
  }
  Result ret;
  if (framebuffer0_ == nullptr) {
    const auto dimensions = surfaceTextures.color->getDimensions();
    const TextureDesc desc1 =
        igl::TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                dimensions.width,
                                dimensions.height,
                                igl::TextureDesc::TextureUsageBits::Sampled |
                                    igl::TextureDesc::TextureUsageBits::Attachment);
    tex1_ = getPlatform().getDevice().createTexture(desc1, nullptr);

    framebuffer0_ = getPlatform().getDevice().createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = tex1_}},
            .depthAttachment =
                {
                    .texture = getPlatform().getDevice().createTexture(
                        TextureDesc{
                            .width = dimensions.width,
                            .height = dimensions.height,
                            .usage = igl::TextureDesc::TextureUsageBits::Attachment,
                            .type = TextureType::TwoD,
                            .format = TextureFormat::Z_UNorm24,
                            .storage = ResourceStorage::Private,
                        },
                        &ret),
                },
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer0_ != nullptr);
  }

  if (framebuffer1_ == nullptr) {
    framebuffer1_ = getPlatform().getDevice().createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color}},
            .depthAttachment = {.texture = surfaceTextures.depth},
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer1_ != nullptr);
  } else {
    framebuffer1_->updateDrawable(surfaceTextures);
  }
  const size_t textureUnit = 0;

  // Graphics pipeline
  if (pipelineState0_ == nullptr) {
    pipelineState0_ = getPlatform().getDevice().createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInputState_,
            .shaderStages = shaderStages_,
            .targetDesc =
                {
                    .colorAttachments =
                        {
                            {
                                .textureFormat = tex1_->getProperties().format,
                            },
                        },
                    .depthAttachmentFormat =
                        framebuffer0_->getDepthAttachment()->getProperties().format,
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
            .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputImage")}},
        },
        nullptr);

    pipelineState1_ = getPlatform().getDevice().createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInputState_,
            .shaderStages = shaderStages_,
            .targetDesc =
                {
                    .colorAttachments =
                        {
                            {
                                .textureFormat =
                                    framebuffer1_->getColorAttachment(0)->getProperties().format,
                            },
                        },
                    .depthAttachmentFormat =
                        framebuffer1_->getDepthAttachment()->getProperties().format,
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
            .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputImage")}},
        },
        nullptr);

    // Set up uniform descriptors
    fragmentUniformDescriptors_.emplace_back();
  }

  // Command buffer
  auto buffer = commandQueue_->createCommandBuffer({}, nullptr);

  // Draw render pass 0
  auto drawableSurface = framebuffer1_->getColorAttachment(0);
  framebuffer1_->updateDrawable(drawableSurface);

  if (!fragmentUniformDescriptors_.empty()) {
    fragmentUniformDescriptors_.back().location =
        pipelineState0_->getIndexByName("color", ShaderStage::Fragment);
    fragmentUniformDescriptors_.back().type = UniformType::Float3;
    fragmentUniformDescriptors_.back().offset = offsetof(FragmentFormat, color);
  }

  BackendType backendType = getPlatform().getDevice().getBackendType();

  // Draw render pass 0
  render(buffer,
         vb0_,
         tex0_,
         pipelineState0_,
         framebuffer0_,
         renderPass0_,
         samplerState_,
         ib0_,
         textureUnit,
         backendType,
         fragmentParamBuffer_,
         fragmentUniformDescriptors_,
         fragmentParameters_);

  // Draw render pass 1
  render(buffer,
         vb1_,
         tex1_,
         pipelineState1_,
         framebuffer1_,
         renderPass1_,
         samplerState_,
         ib0_,
         textureUnit,
         backendType,
         fragmentParamBuffer_,
         fragmentUniformDescriptors_,
         fragmentParameters_);

  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  commandQueue_->submit(*buffer);
}

} // namespace igl::shell
