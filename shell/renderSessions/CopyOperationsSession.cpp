/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/CopyOperationsSession.h>

#include <cstdint>
#include <vector>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {

namespace {
struct VertexPosColor {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float4 color;
};
VertexPosColor vertexData[] = {
    {.position = {-0.6f, -0.4f, 0.0}, .color = {1.0, 0.0, 0.0, 1.0}},
    {.position = {0.6f, -0.4f, 0.0}, .color = {0.0, 1.0, 0.0, 1.0}},
    {.position = {0.0f, 0.6f, 0.0}, .color = {0.0, 0.0, 1.0, 1.0}},
};
uint16_t indexData[] = {
    2,
    1,
    0,
};

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
} // namespace

void CopyOperationsSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Create source vertex buffer with vertex data
  srcVertexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                                    .data = vertexData,
                                                    .length = sizeof(vertexData)},
                                         nullptr);
  IGL_DEBUG_ASSERT(srcVertexBuffer_ != nullptr);

  // Create destination vertex buffer (same size, no initial data)
  dstVertexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                                    .data = nullptr,
                                                    .length = sizeof(vertexData)},
                                         nullptr);
  IGL_DEBUG_ASSERT(dstVertexBuffer_ != nullptr);

  // Index buffer
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

// NOLINTNEXTLINE(bugprone-exception-escape)
void CopyOperationsSession::update(SurfaceTextures textures) noexcept {
  // Per IGL guidelines, textures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!textures.color) {
    return;
  }
  Result ret;

  // Create or update framebuffer
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

  // Create graphics pipeline (cached)
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

  // Create command buffer
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);

  // Step 1: Buffer-to-buffer copy (once)
  // Copy vertex data from srcVertexBuffer_ to dstVertexBuffer_ using ICommandBuffer::copyBuffer()
  if (!hasCopied_) {
    buffer->copyBuffer(*srcVertexBuffer_, *dstVertexBuffer_, 0, 0, sizeof(vertexData));
    hasCopied_ = true;
    IGL_LOG_INFO("[CopyOperationsSession] Copied %zu bytes from src to dst vertex buffer\n",
                 sizeof(vertexData));
  }

  auto drawableSurface = framebuffer_->getColorAttachment(0);

  // Step 2: Render the triangle using the DESTINATION buffer (the copied one)
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    commands->bindVertexBuffer(1, *dstVertexBuffer_);
    commands->bindRenderPipelineState(pipelineState_);
    commands->bindIndexBuffer(*indexBuffer_, IndexFormat::UInt16);
    commands->drawIndexed(3);

    commands->endEncoding();
  }

  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer);

  // Step 3: Framebuffer readback -- read rendered pixels and log first pixel color
  // Uses IFramebuffer::copyBytesColorAttachment() to demonstrate GPU-to-CPU readback
  if (!hasReadBack_) {
    const auto colorAttachment = framebuffer_->getColorAttachment(0);
    if (colorAttachment) {
      const uint32_t width = colorAttachment->getDimensions().width;
      const uint32_t height = colorAttachment->getDimensions().height;
      if (width > 0 && height > 0) {
        std::vector<uint32_t> pixels(static_cast<size_t>(width) * height);
        framebuffer_->copyBytesColorAttachment(
            *commandQueue_, 0, pixels.data(), TextureRangeDesc::new2D(0, 0, width, height));

        // Log first pixel RGBA (packed as 0xAABBGGRR in most backends)
        const uint32_t firstPixel = pixels[0];
        [[maybe_unused]] const uint8_t r = static_cast<uint8_t>(firstPixel & 0xFF);
        [[maybe_unused]] const uint8_t g = static_cast<uint8_t>((firstPixel >> 8) & 0xFF);
        [[maybe_unused]] const uint8_t b = static_cast<uint8_t>((firstPixel >> 16) & 0xFF);
        [[maybe_unused]] const uint8_t a = static_cast<uint8_t>((firstPixel >> 24) & 0xFF);
        IGL_LOG_INFO(
            "[CopyOperationsSession] Framebuffer readback: first pixel RGBA = (%u, %u, %u, %u)\n",
            r,
            g,
            b,
            a);
        hasReadBack_ = true;
      }
    }
  }

  RenderSession::update(textures);
}

} // namespace igl::shell
