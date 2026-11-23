/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/StencilOutlineSession.h>

#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {

namespace {
struct VertexPosColor {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float4 color;
};

// A hexagon shape for a more interesting outline demo
VertexPosColor vertexData[] = {
    // Center
    {.position = {0.0f, 0.0f, 0.0f}, .color = {0.2f, 0.6f, 1.0f, 1.0f}},
    // Hexagon vertices (6 points)
    {.position = {0.0f, 0.5f, 0.0f}, .color = {0.4f, 0.8f, 1.0f, 1.0f}},
    {.position = {0.433f, 0.25f, 0.0f}, .color = {0.3f, 0.7f, 1.0f, 1.0f}},
    {.position = {0.433f, -0.25f, 0.0f}, .color = {0.2f, 0.6f, 0.9f, 1.0f}},
    {.position = {0.0f, -0.5f, 0.0f}, .color = {0.1f, 0.5f, 0.8f, 1.0f}},
    {.position = {-0.433f, -0.25f, 0.0f}, .color = {0.2f, 0.6f, 0.9f, 1.0f}},
    {.position = {-0.433f, 0.25f, 0.0f}, .color = {0.3f, 0.7f, 1.0f, 1.0f}},
};

// Triangles forming the hexagon (6 triangles, all sharing center vertex 0)
uint16_t indexData[] = {
    0,
    1,
    2,
    0,
    2,
    3,
    0,
    3,
    4,
    0,
    4,
    5,
    0,
    5,
    6,
    0,
    6,
    1,
};

// ---------------------------------------------------------------------------
// Object shaders: standard position + color passthrough
// ---------------------------------------------------------------------------

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
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return v.color;
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

// ---------------------------------------------------------------------------
// Outline shaders: scale geometry up by 1.1x and output solid outline color
// ---------------------------------------------------------------------------

std::string getOutlineMetalShaderSource() {
  return R"(
              using namespace metal;

              typedef struct {
                float3 position [[attribute(0)]];
                float4 color [[attribute(1)]];
              } VertexIn;

              typedef struct {
                float4 position [[position]];
              } VertexOut;

              vertex VertexOut outlineVertexShader(
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]]) {
                VertexOut out;
                out.position = float4(vertices[vid].position * 1.1, 1.0);
                return out;
              }

              fragment float4 outlineFragmentShader(
                  VertexOut IN [[stage_in]]) {
                  return float4(1.0, 0.5, 0.0, 1.0);
              }
    )";
}

std::string getOutlineOpenGLVertexShaderSource() {
  return getVersion() + R"(
                precision highp float;
                attribute vec3 position;
                attribute vec4 color_in;

                void main() {
                  gl_Position = vec4(position * 1.1, 1.0);
                })";
}

std::string getOutlineOpenGLFragmentShaderSource() {
  return getVersion() + std::string(R"(
                precision highp float;

                void main() {
                  gl_FragColor = vec4(1.0, 0.5, 0.0, 1.0);
                })");
}

std::string getOutlineVulkanVertexShaderSource() {
  return R"(
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec4 color_in;

                void main() {
                  gl_Position = vec4(position * 1.1, 1.0);
                }
                )";
}

std::string getOutlineVulkanFragmentShaderSource() {
  return R"(
                layout(location = 0) out vec4 out_FragColor;

                void main() {
                  out_FragColor = vec4(1.0, 0.5, 0.0, 1.0);
                }
                )";
}

std::string getOutlineWgslShaderSource() {
  return R"(
@vertex
fn outlineVertexShader(@location(0) position : vec3f) -> @builtin(position) vec4f {
  return vec4f(position * 1.1, 1.0);
}

@fragment
fn outlineFragmentShader() -> @location(0) vec4f {
  return vec4f(1.0, 0.5, 0.0, 1.0);
}
)";
}

std::unique_ptr<IShaderStages> getOutlineShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(device,
                                                            getOutlineWgslShaderSource().c_str(),
                                                            "outlineVertexShader",
                                                            "outlineFragmentShader",
                                                            "",
                                                            nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getOutlineVulkanVertexShaderSource().c_str(),
        "main",
        "",
        getOutlineVulkanFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(device,
                                                            getOutlineMetalShaderSource().c_str(),
                                                            "outlineVertexShader",
                                                            "outlineFragmentShader",
                                                            "",
                                                            nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getOutlineOpenGLVertexShaderSource().c_str(),
        "main",
        "",
        getOutlineOpenGLFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  case BackendType::D3D12: {
    static const char* kVS = R"(
      struct VSIn { float3 position : POSITION; float4 color : COLOR; };
      struct VSOut { float4 position : SV_POSITION; };
      VSOut main(VSIn v) {
        VSOut o; o.position = float4(v.position * 1.1, 1.0); return o; }
    )";
    static const char* kPS = R"(
      struct PSIn { float4 position : SV_POSITION; };
      float4 main(PSIn i) : SV_TARGET { return float4(1.0, 0.5, 0.0, 1.0); }
    )";
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kVS, "main", "", kPS, "main", "", nullptr);
  }
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}
} // namespace

void StencilOutlineSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Vertex & Index buffer
  vertexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                                 .data = vertexData,
                                                 .length = sizeof(vertexData)},
                                      nullptr);
  IGL_DEBUG_ASSERT(vertexBuffer_ != nullptr);
  indexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                                .data = indexData,
                                                .length = sizeof(indexData)},
                                     nullptr);
  IGL_DEBUG_ASSERT(indexBuffer_ != nullptr);

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

  // Create shader stages for both object and outline
  shaderStages_ = getShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);

  outlineShaderStages_ = getOutlineShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(outlineShaderStages_ != nullptr);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  // Depth/stencil state for the first pass: always pass stencil test, write reference value
  {
    const StencilStateDesc stencilWrite{
        .stencilFailureOperation = StencilOperation::Keep,
        .depthFailureOperation = StencilOperation::Keep,
        .depthStencilPassOperation = StencilOperation::Replace,
        .stencilCompareFunction = CompareFunction::AlwaysPass,
        .readMask = 0xFF,
        .writeMask = 0xFF,
    };
    const DepthStencilStateDesc desc{
        .compareFunction = CompareFunction::AlwaysPass,
        .isDepthWriteEnabled = true,
        .backFaceStencil = stencilWrite,
        .frontFaceStencil = stencilWrite,
    };
    depthStencilStateWrite_ = device.createDepthStencilState(desc, nullptr);
    IGL_DEBUG_ASSERT(depthStencilStateWrite_ != nullptr);
  }

  // Depth/stencil state for the outline pass: draw only where stencil != reference value
  {
    const StencilStateDesc stencilOutline{
        .stencilFailureOperation = StencilOperation::Keep,
        .depthFailureOperation = StencilOperation::Keep,
        .depthStencilPassOperation = StencilOperation::Keep,
        .stencilCompareFunction = CompareFunction::NotEqual,
        .readMask = 0xFF,
        .writeMask = 0x00,
    };
    const DepthStencilStateDesc desc{
        .compareFunction = CompareFunction::AlwaysPass,
        .isDepthWriteEnabled = false,
        .backFaceStencil = stencilOutline,
        .frontFaceStencil = stencilOutline,
    };
    depthStencilStateOutline_ = device.createDepthStencilState(desc, nullptr);
    IGL_DEBUG_ASSERT(depthStencilStateOutline_ != nullptr);
  }

  // Render pass descriptor with color, depth, and stencil clear
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
      .stencilAttachment =
          {
              .loadAction = LoadAction::Clear,
              .clearStencil = 0,
          },
  };
}

void StencilOutlineSession::update(SurfaceTextures textures) noexcept {
  // Per IGL guidelines, textures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!textures.color) {
    return;
  }
  Result ret;

  // Create or update framebuffer (with stencil attachment if depth texture has stencil)
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

  // Create object pipeline state (lazy, cached)
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
            .frontFaceWinding = WindingMode::Clockwise,
        },
        nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  // Create outline pipeline state (lazy, cached)
  if (outlinePipelineState_ == nullptr) {
    outlinePipelineState_ = getPlatform().getDevice().createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInputState_,
            .shaderStages = outlineShaderStages_,
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
    IGL_DEBUG_ASSERT(outlinePipelineState_ != nullptr);
  }

  // Command Buffers
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  auto drawableSurface = framebuffer_->getColorAttachment(0);

  // Submit commands
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    commands->bindVertexBuffer(1, *vertexBuffer_);
    commands->bindIndexBuffer(*indexBuffer_, IndexFormat::UInt16);

    // --- Pass 1: Draw the object and write stencil reference value ---
    commands->bindDepthStencilState(depthStencilStateWrite_);
    commands->setStencilReferenceValue(1);
    commands->bindRenderPipelineState(pipelineState_);
    commands->drawIndexed(18); // 6 triangles * 3 indices

    // --- Pass 2: Draw the scaled-up outline where stencil != reference ---
    commands->bindDepthStencilState(depthStencilStateOutline_);
    commands->setStencilReferenceValue(1);
    commands->bindRenderPipelineState(outlinePipelineState_);
    commands->drawIndexed(18); // 6 triangles * 3 indices

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
