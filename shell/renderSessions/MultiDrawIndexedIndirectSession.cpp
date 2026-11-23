/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/MultiDrawIndexedIndirectSession.h>

#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {

namespace {

struct DrawElementsIndirectCommand {
  uint32_t count = 0;
  uint32_t instanceCount = 0;
  uint32_t firstIndex = 0;
  int32_t baseVertex = 0;
  uint32_t reservedMustBeZero = 0;
};

struct VertexPosColor {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float4 color;
};

// Triangle (red) at left, Square (green) at center, Pentagon (blue) at right.
// All shapes share one vertex buffer; index buffer selects which vertices to draw.

// clang-format off
VertexPosColor vertexData[] = {
    // --- Triangle (3 vertices, red) ---
    // Vertices 0-2
    {.position = {-0.9f, 0.0f, 0.0f}, .color = {1.0f, 0.0f, 0.0f, 1.0f}},
    {.position = {-0.5f, 0.0f, 0.0f}, .color = {1.0f, 0.0f, 0.0f, 1.0f}},
    {.position = {-0.7f, 0.4f, 0.0f}, .color = {1.0f, 0.0f, 0.0f, 1.0f}},

    // --- Square (4 vertices, green) ---
    // Vertices 3-6
    {.position = {-0.2f, -0.2f, 0.0f}, .color = {0.0f, 1.0f, 0.0f, 1.0f}},
    {.position = {0.2f, -0.2f, 0.0f}, .color = {0.0f, 1.0f, 0.0f, 1.0f}},
    {.position = {0.2f, 0.2f, 0.0f}, .color = {0.0f, 1.0f, 0.0f, 1.0f}},
    {.position = {-0.2f, 0.2f, 0.0f}, .color = {0.0f, 1.0f, 0.0f, 1.0f}},

    // --- Pentagon (5 vertices, blue) ---
    // Vertices 7-11
    {.position = {0.7f, 0.35f, 0.0f}, .color = {0.0f, 0.0f, 1.0f, 1.0f}},
    {.position = {0.52f, 0.05f, 0.0f}, .color = {0.0f, 0.0f, 1.0f, 1.0f}},
    {.position = {0.58f, -0.3f, 0.0f}, .color = {0.0f, 0.0f, 1.0f, 1.0f}},
    {.position = {0.82f, -0.3f, 0.0f}, .color = {0.0f, 0.0f, 1.0f, 1.0f}},
    {.position = {0.88f, 0.05f, 0.0f}, .color = {0.0f, 0.0f, 1.0f, 1.0f}},
};

// Index data: triangle (3 indices), square as 2 triangles (6 indices),
// pentagon as 3 triangles (9 indices).
uint16_t indexData[] = {
    // Triangle (indices 0-2, referencing vertices 0-2)
    0, 1, 2,

    // Square (indices 3-8, referencing vertices 3-6, two triangles)
    3, 4, 5,
    3, 5, 6,

    // Pentagon (indices 9-17, referencing vertices 7-11, three triangles)
    7, 8, 9,
    7, 9, 10,
    7, 10, 11,
};
// clang-format on

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

              vertex VertexOut vertexShader(
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]]) {
                VertexOut out;
                out.position = float4(vertices[vid].position, 1.0);
                out.color = vertices[vid].color;
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
struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
};

@vertex
fn vertexShader(@location(0) position : vec3f, @location(1) color : vec4f) -> VertexOut {
  return VertexOut(vec4f(position, 1.0), color);
}

@fragment
fn fragmentShader(in : VertexOut) -> @location(0) vec4f {
  return in.color;
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
        device, getWgslShaderSource().c_str(), "vertexShader", "fragmentShader", "", nullptr);
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

void MultiDrawIndexedIndirectSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();
  if (!device.hasFeature(DeviceFeatures::DrawIndexedIndirect) ||
      !device.hasFeature(DeviceFeatures::StorageBuffers)) {
    IGL_LOG_INFO("MultiDrawIndexedIndirectSession: indirect draws are not supported; skipping\n");
    return;
  }

  // Vertex buffer (all shapes share one buffer)
  vertexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                                 .data = vertexData,
                                                 .length = sizeof(vertexData)},
                                      nullptr);
  IGL_DEBUG_ASSERT(vertexBuffer_ != nullptr);

  // Index buffer (indices for all shapes consecutively)
  indexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                                .data = indexData,
                                                .length = sizeof(indexData)},
                                     nullptr);
  IGL_DEBUG_ASSERT(indexBuffer_ != nullptr);

  // Indirect buffer with 3 DrawElementsIndirectCommand entries
  {
    const DrawElementsIndirectCommand indirectCommands[3] = {
        // Command 0: Triangle (3 indices starting at firstIndex=0, baseVertex=0)
        {.count = 3, .instanceCount = 1, .firstIndex = 0, .baseVertex = 0, .reservedMustBeZero = 0},
        // Command 1: Square (6 indices starting at firstIndex=3, baseVertex=0)
        {.count = 6, .instanceCount = 1, .firstIndex = 3, .baseVertex = 0, .reservedMustBeZero = 0},
        // Command 2: Pentagon (9 indices starting at firstIndex=9, baseVertex=0)
        {.count = 9, .instanceCount = 1, .firstIndex = 9, .baseVertex = 0, .reservedMustBeZero = 0},
    };

    const BufferDesc indirectBufDesc{
        .type = BufferDesc::BufferTypeBits::Storage | BufferDesc::BufferTypeBits::Indirect,
        .data = indirectCommands,
        .length = sizeof(indirectCommands),
    };
    indirectBuffer_ = device.createBuffer(indirectBufDesc, nullptr);
    IGL_DEBUG_ASSERT(indirectBuffer_ != nullptr);
  }

  // Vertex input state
  vertexInputState_ = device.createVertexInputState(
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
  IGL_DEBUG_ASSERT(vertexInputState_ != nullptr);

  // Shaders
  shaderStages_ = getShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  // Render pass
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

void MultiDrawIndexedIndirectSession::update(SurfaceTextures textures) noexcept {
  // Per IGL guidelines, textures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!textures.color || !indirectBuffer_) {
    return;
  }
  Result ret;

  // Create/update framebuffer
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

  // Graphics pipeline (cached)
  if (pipelineState_ == nullptr) {
    pipelineState_ = getPlatform().getDevice().createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInputState_,
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
            .frontFaceWinding = WindingMode::CounterClockwise,
        },
        nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  // Command buffer
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  auto drawableSurface = framebuffer_->getColorAttachment(0);

  // Render commands
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    commands->bindVertexBuffer(1, *vertexBuffer_);
    commands->bindRenderPipelineState(pipelineState_);
    commands->bindIndexBuffer(*indexBuffer_, IndexFormat::UInt16);

    // Issue all 3 draw calls (triangle, square, pentagon) from the indirect buffer
    commands->multiDrawIndexedIndirect(*indirectBuffer_, 0, 3, sizeof(DrawElementsIndirectCommand));

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
