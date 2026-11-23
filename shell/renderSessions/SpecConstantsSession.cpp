/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @MARK:COVERAGE_EXCLUDE_FILE

#include <shell/renderSessions/SpecConstantsSession.h>

#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/Shader.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {

namespace {
struct VertexPosColor {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float4 color;
};
const VertexPosColor kVertexData[] = {
    {.position = {-0.6f, -0.4f, 0.0}, .color = {1.0, 0.0, 0.0, 1.0}},
    {.position = {0.6f, -0.4f, 0.0}, .color = {0.0, 1.0, 0.0, 1.0}},
    {.position = {0.0f, 0.6f, 0.0}, .color = {0.0, 0.0, 1.0, 1.0}},
};
const uint16_t kIndexData[] = {2, 1, 0};

// Demonstration of the specialization-constants API on Metal and WebGPU: nine Float1 constants
// (Metal function constant indices 0..8, WGSL override ids 0..8) carry the RGB triplets for the
// three triangle vertices, baked into the vertex shader at pipeline-build time. The other
// backends source the same colors from the per-vertex `color_in` attribute instead.
// NOLINTNEXTLINE(*-avoid-c-arrays)
float kVertexColors[3][3] = {
    {1.0f, 0.0f, 0.0f}, // vertex 0: red
    {0.0f, 1.0f, 0.0f}, // vertex 1: green
    {0.0f, 0.0f, 1.0f}, // vertex 2: blue
};

FunctionConstantValues getVertexSpecConstants() {
  FunctionConstantValues fcv;
  for (uint8_t v = 0; v < 3; ++v) {
    for (uint8_t c = 0; c < 3; ++c) {
      fcv.setConstantValue(v * 3 + c, ConstantValueType::Float1, &kVertexColors[v][c]);
    }
  }
  return fcv;
}

std::string getVersion() {
  return {"#version 100"};
}

std::string getMetalShaderSource() {
  return R"(
              using namespace metal;

              typedef struct {
                float3 position [[attribute(0)]];
                float4 color [[attribute(1)]];
              } VertexIn;

              typedef struct {
                float4 position [[position]];
                float4 color;
              } VertexOut;

              // Specialization constants — values supplied at pipeline build time
              // through MTLFunctionConstantValues (see getVertexSpecConstants()).
              constant float kV0R [[function_constant(0)]];
              constant float kV0G [[function_constant(1)]];
              constant float kV0B [[function_constant(2)]];
              constant float kV1R [[function_constant(3)]];
              constant float kV1G [[function_constant(4)]];
              constant float kV1B [[function_constant(5)]];
              constant float kV2R [[function_constant(6)]];
              constant float kV2G [[function_constant(7)]];
              constant float kV2B [[function_constant(8)]];

              vertex VertexOut vertexShader(
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]]) {
                const float3 colors[3] = {
                    float3(kV0R, kV0G, kV0B),
                    float3(kV1R, kV1G, kV1B),
                    float3(kV2R, kV2G, kV2B),
                };
                VertexOut out;
                out.position = float4(vertices[vid].position, 1.0);
                out.color = float4(colors[vid], 1.0);
                return out;
              }

              fragment float4 fragmentShader(
                  VertexOut IN [[stage_in]]) {
                  return IN.color;
              }
    )";
}

std::string getOpenGLVertexShaderSource() {
  return getVersion() + R"(
                precision highp float;
                attribute vec3 position;
                attribute vec4 color_in;

                varying vec4 vColor;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  vColor = color_in;
                })";
}

std::string getOpenGLFragmentShaderSource() {
  return getVersion() + std::string(R"(
                precision highp float;

                varying vec4 vColor;

                void main() {
                  gl_FragColor = vColor;
                })");
}

std::string getVulkanVertexShaderSource() {
  return R"(
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec4 color_in;
                layout(location = 0) out vec4 color;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  color = color_in;
                }
                )";
}

std::string getVulkanFragmentShaderSource() {
  return R"(
                layout(location = 0) in vec4 color;
                layout(location = 0) out vec4 out_FragColor;

                void main() {
                  out_FragColor = color;
                }
                )";
}

std::string getWgslShaderSource() {
  return R"(
@id(0) override kV0R : f32;
@id(1) override kV0G : f32;
@id(2) override kV0B : f32;
@id(3) override kV1R : f32;
@id(4) override kV1G : f32;
@id(5) override kV1B : f32;
@id(6) override kV2R : f32;
@id(7) override kV2G : f32;
@id(8) override kV2B : f32;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
};

@vertex
fn vertexShader(@builtin(vertex_index) vid : u32, @location(0) position : vec3f) -> VertexOut {
  var colors = array<vec3f, 3>(
      vec3f(kV0R, kV0G, kV0B), vec3f(kV1R, kV1G, kV1B), vec3f(kV2R, kV2G, kV2B));
  return VertexOut(vec4f(position, 1.0), vec4f(colors[vid], 1.0));
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return v.color;
}
)";
}

std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  // Build the Metal/WebGPU vertex/fragment modules with FunctionConstantValues up-front so
  // their case statements remain simple single-statement returns.
  // ShaderStagesCreator::fromLibraryStringInput() does not expose FunctionConstantValues,
  // so we go through ShaderLibraryCreator and feed the resulting modules into
  // fromRenderModules() below.
  const BackendType backend = device.getBackendType();
  std::shared_ptr<IShaderModule> vertexModule;
  std::shared_ptr<IShaderModule> fragmentModule;
  if (backend == BackendType::Metal || backend == BackendType::WebGPU) {
    auto library = igl::ShaderLibraryCreator::fromStringInput(
        device,
        backend == BackendType::Metal ? getMetalShaderSource().c_str()
                                      : getWgslShaderSource().c_str(),
        {{.stage = ShaderStage::Vertex,
          .entryPoint = "vertexShader",
          .functionConstantValues = getVertexSpecConstants()},
         {.stage = ShaderStage::Fragment, .entryPoint = "fragmentShader"}},
        "",
        nullptr);
    if (library) {
      vertexModule = library->getShaderModule("vertexShader");
      fragmentModule = library->getShaderModule("fragmentShader");
    }
  }

  switch (backend) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
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
  case BackendType::Metal:
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromRenderModules(
        device, std::move(vertexModule), std::move(fragmentModule), nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getOpenGLVertexShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           getOpenGLFragmentShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           nullptr);
  case BackendType::D3D12: {
    static const char* kVS = R"(
      struct VSIn { float3 position : POSITION; float4 color : COLOR; };
      struct VSOut { float4 position : SV_POSITION; float4 color : COLOR; };
      VSOut main(VSIn v) {
        VSOut o; o.position = float4(v.position, 1.0); o.color = v.color; return o; }
    )";
    static const char* kPS = R"(
      struct PSIn { float4 position : SV_POSITION; float4 color : COLOR; };
      float4 main(PSIn i) : SV_TARGET { return i.color; }
    )";
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kVS, "main", "", kPS, "main", "", nullptr);
  }
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}
} // namespace

void SpecConstantsSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Vertex & Index buffer
  vb0_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                        .data = kVertexData,
                                        .length = sizeof(kVertexData)},
                             nullptr);
  IGL_DEBUG_ASSERT(vb0_ != nullptr);
  ib0_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                        .data = kIndexData,
                                        .length = sizeof(kIndexData)},
                             nullptr);
  IGL_DEBUG_ASSERT(ib0_ != nullptr);

  vertexInput0_ = device.createVertexInputState(
      VertexInputStateDesc{
          .numAttributes = 2,
          .attributes =
              {
                  VertexAttribute{.bufferIndex = 1,
                                  .format = VertexAttributeFormat::Float3,
                                  .offset = offsetof(VertexPosColor, position),
                                  .name = "position",
                                  .location = 0},
                  VertexAttribute{.bufferIndex = 1,
                                  .format = VertexAttributeFormat::Float4,
                                  .offset = offsetof(VertexPosColor, color),
                                  .name = "color_in",
                                  .location = 1},
              },
          .numInputBindings = 1,
          .inputBindings = {{}, {.stride = sizeof(VertexPosColor)}},
      },
      nullptr);
  IGL_DEBUG_ASSERT(vertexInput0_ != nullptr);

  shaderStages_ = getShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  renderPass_ = {
      .colorAttachments =
          {
              {
                  .loadAction = LoadAction::Clear,
                  .storeAction = StoreAction::Store,
                  .clearColor = getPreferredClearColor(),
              },
          },
      .depthAttachment =
          {
              .loadAction = LoadAction::Clear,
              .clearDepth = 1.0,
          },
  };
}

void SpecConstantsSession::update(SurfaceTextures textures) noexcept {
  // Per IGL guidelines, textures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!textures.color) {
    return;
  }
  Result ret;
  if (framebuffer_ == nullptr) {
    framebuffer_ = getPlatform().getDevice().createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = textures.color}},
            .depthAttachment = {.texture = textures.depth},
            .stencilAttachment = textures.depth && textures.depth->getProperties().hasStencil()
                                     ? FramebufferDesc::AttachmentDesc{.texture = textures.depth}
                                     : FramebufferDesc::AttachmentDesc{},
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(textures);
  }

  // Graphics pipeline
  if (pipelineState_ == nullptr) {
    pipelineState_ = getPlatform().getDevice().createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInput0_,
            .shaderStages = shaderStages_,
            .targetDesc =
                {
                    .colorAttachments = {{.textureFormat =
                                              framebuffer_->getColorAttachment(0)->getFormat()}},
                    .depthAttachmentFormat = framebuffer_->getDepthAttachment()->getFormat(),
                    .stencilAttachmentFormat =
                        framebuffer_->getStencilAttachment()
                            ? framebuffer_->getStencilAttachment()->getFormat()
                            : TextureFormat::Invalid,
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
        },
        nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  // Command Buffers
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  const auto drawableSurface = framebuffer_->getColorAttachment(0);

  // Submit commands
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    commands->bindVertexBuffer(1, *vb0_);
    commands->bindRenderPipelineState(pipelineState_);
    commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
    commands->drawIndexed(3);

    commands->endEncoding();
  }

  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer);
  RenderSession::update(textures);
}

} // namespace igl::shell
