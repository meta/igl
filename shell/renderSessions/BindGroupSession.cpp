/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/BindGroupSession.h>

#include <IGLU/managedUniformBuffer/ManagedUniformBuffer.h>
#include <cstddef>
#include <shell/shared/imageLoader/ImageLoader.h>
#include <shell/shared/platform/DisplayContext.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/CommandBuffer.h>
#include <igl/NameHandle.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/ShaderCreator.h>
#if IGL_BACKEND_OPENGL
#include <igl/opengl/Version.h>
#endif

namespace {
struct VertexPosUvw {
  glm::vec3 position;
  glm::vec3 color;
  glm::vec2 uv;
};

const float kHalf = 1.0f;

// UV-mapped cube with indices: 24 vertices, 36 indices
const VertexPosUvw kVertexData0[] = {
    // top
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.5, 0.5, 1.0}, .uv = {0, 0}}, // 0
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.0, 1.0}, .uv = {1, 0}}, // 1
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 2
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.5, 1.0, 1.0}, .uv = {0, 1}}, // 3
    // bottom
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 4
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.5, 1.0, 0.5}, .uv = {0, 1}}, // 5
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.5}, .uv = {1, 1}}, // 6
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.5, 0.5}, .uv = {1, 0}}, // 7
    // left
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.5}, .uv = {1, 0}}, // 8
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.5, 1.0, 0.5}, .uv = {0, 0}}, // 9
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.5, 1.0, 1.0}, .uv = {0, 1}}, // 10
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 11
    // right
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 12
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.5, 0.5}, .uv = {1, 0}}, // 13
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.5, 1.0}, .uv = {1, 1}}, // 14
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.5, 0.5, 1.0}, .uv = {0, 1}}, // 15
    // front
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.5, 0.5}, .uv = {0, 0}}, // 16
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.5}, .uv = {1, 0}}, // 17
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 18
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.5, 1.0}, .uv = {0, 1}}, // 19
    // back
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.5, 1.0, 0.5}, .uv = {1, 0}}, // 20
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 21
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.5, 0.5, 1.0}, .uv = {0, 1}}, // 22
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.5, 1.0, 1.0}, .uv = {1, 1}}, // 23
};

const uint16_t kIndexData[] = {0,  1,  2,  2,  3,  0,  4,  5,  6,  6,  7,  4,
                               8,  9,  10, 10, 11, 8,  12, 13, 14, 14, 15, 12,
                               16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};

std::string getProlog(igl::IDevice& device) {
#if IGL_BACKEND_OPENGL
  const auto shaderVersion = device.getShaderVersion();
  if (shaderVersion.majorVersion >= 3 || shaderVersion.minorVersion >= 30) {
    std::string prependVersionString = igl::opengl::getStringFromShaderVersion(shaderVersion);
    prependVersionString += "\nprecision highp float;\n";
    return prependVersionString;
  }
#else
  (void)device; // Suppress unused parameter warning
#endif // IGL_BACKEND_OPENGL
  return "";
}

std::string getMetalShaderSource() {
  return R"(
          #include <metal_stdlib>
          #include <simd/simd.h>
          using namespace metal;

          struct VertexUniformBlock {
            float4x4 mvpMatrix;
          };

          struct VertexIn {
            float3 position [[attribute(0)]];
            float2 uv [[attribute(1)]];
            float3 color [[attribute(2)]];
          };

          struct VertexOut {
            float4 position [[position]];
            float2 uv;
            float4 color;
          };

          vertex VertexOut vertexShader(VertexIn in [[stage_in]],
                 constant VertexUniformBlock &vUniform[[buffer(1)]]) {
            VertexOut out;
            out.position = vUniform.mvpMatrix * float4(in.position, 1.0);
            out.uv = in.uv;
            out.color = float4(in.color, 1.0);
            return out;
           }

           fragment float4 fragmentShader(
                 VertexOut in[[stage_in]],
                 texture2d<float> tex0 [[texture(0)]],
                 texture2d<float> tex1 [[texture(1)]],
                 sampler linearSampler [[sampler(0)]]) {
             constexpr sampler s(s_address::clamp_to_edge,
                                 t_address::clamp_to_edge,
                                 min_filter::linear,
                                 mag_filter::linear);
             return tex0.sample(s, in.uv) * tex1.sample(s, in.uv) * in.color;
           }
        )";
}

std::string getOpenGLFragmentShaderSource(igl::IDevice& device) {
  return getProlog(device) + std::string(R"(
                      precision highp float; precision highp sampler2D;
                      in vec2 uv;
                      in vec4 color;
                      uniform sampler2D input2D;
                      uniform sampler2D inputXOR;
                      out vec4 fragmentColor;
                      void main() {
                        fragmentColor = texture(input2D, uv) * texture(inputXOR, uv) * color;
                      })");
}

std::string getOpenGLVertexShaderSource(igl::IDevice& device) {
  return getProlog(device) + R"(
                      precision highp float;
                      uniform mat4 mvpMatrix;
                      in vec3 position;
                      in vec2 uv_in;
                      in vec3 color_in;
                      out vec2 uv;
                      out vec4 color;

                      void main() {
                        gl_Position =  mvpMatrix * vec4(position, 1.0);
                        uv = uv_in;
                        color = vec4(color_in, 1.0);
                      })";
}

const char* getVulkanFragmentShaderSource() {
  return R"(
                      precision highp float;
                      layout(location = 0) in vec2 uv;
                      layout(location = 1) in vec4 color;
                      layout(location = 0) out vec4 out_FragColor;

                      layout(set = 0, binding = 0) uniform sampler2D in_texture0;
                      layout(set = 0, binding = 1) uniform sampler2D in_texture1;

                      void main() {
                        out_FragColor = texture(in_texture0, uv) * texture(in_texture1, uv) * color;
                      })";
}

const char* getVulkanVertexShaderSource() {
  return R"(
                      precision highp float;

                      layout (set = 1, binding = 1, std140) uniform PerFrame {
                        mat4 mvpMatrix;
                      } perFrame;

                      layout(location = 0) in vec3 position;
                      layout(location = 1) in vec2 uvw_in;
                      layout(location = 2) in vec3 color_in;
                      layout(location = 0) out vec2 uvw;
                      layout(location = 1) out vec4 color;

                      void main() {
                        gl_Position =  perFrame.mvpMatrix * vec4(position, 1.0);
                        uvw = uvw_in;
                        color = vec4(color_in, 1.0);
                      })";
}

const char* getWgslShaderSource() {
  return R"(
struct PerFrame {
  mvpMatrix : mat4x4f,
};

@group(1) @binding(1) var<uniform> perFrame : PerFrame;
@group(0) @binding(0) var texture0 : texture_2d<f32>;
@group(0) @binding(1) var sampler0 : sampler;
@group(0) @binding(2) var texture1 : texture_2d<f32>;
@group(0) @binding(3) var sampler1 : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
  @location(1) color : vec4f,
};

@vertex
fn vertexShader(@location(0) position : vec3f,
                @location(1) uv : vec2f,
                @location(2) color : vec3f) -> VertexOut {
  return VertexOut(perFrame.mvpMatrix * vec4f(position, 1.0), uv, vec4f(color, 1.0));
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return textureSample(texture0, sampler0, v.uv) * textureSample(texture1, sampler1, v.uv) *
         v.color;
}
)";
}

std::unique_ptr<igl::IShaderStages> getShaderStagesForBackend(igl::IDevice& device) {
  switch (device.getBackendType()) {
  case igl::BackendType::Invalid:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case igl::BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getVulkanVertexShaderSource(),
                                                           "main",
                                                           "",
                                                           getVulkanFragmentShaderSource(),
                                                           "main",
                                                           "",
                                                           nullptr);
  case igl::BackendType::D3D12: {
    // D3D12 HLSL shaders using register bindings
    static const char* kVS = R"(
      cbuffer PerFrame : register(b1) {
        float4x4 mvpMatrix;
      };

      struct VSInput {
        float3 position : POSITION;
        float2 uvw : TEXCOORD0;
        float3 color_in : COLOR0;
      };

      struct VSOutput {
        float4 position : SV_POSITION;
        float2 uv : TEXCOORD0;
        float4 color : COLOR0;
      };

      VSOutput main(VSInput input) {
        VSOutput output;
        output.position = mul(mvpMatrix, float4(input.position, 1.0));
        output.uv = input.uvw;
        output.color = float4(input.color_in, 1.0);
        return output;
      }
    )";

    static const char* kPS = R"(
      Texture2D in_texture0 : register(t0);
      Texture2D in_texture1 : register(t1);
      SamplerState sampler0 : register(s0);
      SamplerState sampler1 : register(s1);

      struct PSInput {
        float4 position : SV_POSITION;
        float2 uv : TEXCOORD0;
        float4 color : COLOR0;
      };

      float4 main(PSInput input) : SV_Target {
        float4 tex0 = in_texture0.Sample(sampler0, input.uv);
        float4 tex1 = in_texture1.Sample(sampler1, input.uv);
        return tex0 * tex1 * input.color;
      }
    )";

    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kVS, "main", "", kPS, "main", "", nullptr);
  }
  case igl::BackendType::Custom:
    IGL_DEBUG_ABORT("IGLSamples not set up for Custom");
    return nullptr;
  // @fb-only
    // @fb-only
    // @fb-only
  case igl::BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
  case igl::BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getMetalShaderSource().c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case igl::BackendType::OpenGL:
    if (getProlog(device).empty()) {
      IGL_LOG_INFO("BindGroupSession: needs GLSL ES 3.00 / GLSL 1.30; skipping\n");
      return nullptr;
    }
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getOpenGLVertexShaderSource(device).c_str(),
        "main",
        "",
        getOpenGLFragmentShaderSource(device).c_str(),
        "main",
        "",
        nullptr);
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

} // namespace

namespace igl::shell {

void BindGroupSession::createSamplerAndTextures(const IDevice& device) {
  // Sampler & Texture
  const auto sampler = device.createSamplerState(SamplerStateDesc::newLinearMipmapped(), nullptr);

  std::shared_ptr<ITexture> tex0, tex1;

  {
    const auto imageData = getPlatform().getImageLoader().loadImageData("igl.png");
    TextureDesc desc = igl::TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                               imageData.desc.width,
                                               imageData.desc.height,
                                               igl::TextureDesc::TextureUsageBits::Sampled |
                                                   igl::TextureDesc::TextureUsageBits::Attachment,
                                               "igl.png");
    desc.numMipLevels = TextureDesc::calcNumMipLevels(imageData.desc.width, imageData.desc.height);
    tex0 = device.createTexture(desc, nullptr);
    tex0->upload(tex0->getFullRange(), imageData.data->data());
    tex0->generateMipmap(*commandQueue_);
  }

  {
    const uint32_t texWidth = 256;
    const uint32_t texHeight = 256;
    TextureDesc desc = TextureDesc::new2D(TextureFormat::BGRA_UNorm8,
                                          texWidth,
                                          texHeight,
                                          TextureDesc::TextureUsageBits::Sampled |
                                              TextureDesc::TextureUsageBits::Attachment,
                                          "XOR pattern");
    desc.numMipLevels = TextureDesc::calcNumMipLevels(texWidth, texHeight);
    tex1 = getPlatform().getDevice().createTexture(desc, nullptr);
    std::vector<uint32_t> pixels(static_cast<size_t>(texWidth * texHeight));
    for (uint32_t y = 0; y != texHeight; y++) {
      for (uint32_t x = 0; x != texWidth; x++) {
        // create a XOR pattern
        pixels[y * texWidth + x] = 0xFF000000 + ((x ^ y) << 16) + ((x ^ y) << 8) + (x ^ y);
      }
    }
    tex1->upload(TextureRangeDesc::new2D(0, 0, texWidth, texHeight), pixels.data());
    tex1->generateMipmap(*commandQueue_);
  }

  bindGroupTextures_ = getPlatform().getDevice().createBindGroup(BindGroupTextureDesc{
      .textures = {tex0, tex1},
      .samplers = {sampler, sampler},
      .debugName = "bindGroupTextures_",
  });
}

BindGroupSession::BindGroupSession(std::shared_ptr<Platform> platform) :
  RenderSession(std::move(platform)) {
  imguiSession_ = std::make_unique<iglu::imgui::Session>(getPlatform().getDevice(),
                                                         getPlatform().getInputDispatcher());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void BindGroupSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Vertex buffer, Index buffer and Vertex Input
  vb0_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                        .data = kVertexData0,
                                        .length = sizeof(kVertexData0)},
                             nullptr);
  ib0_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                        .data = kIndexData,
                                        .length = sizeof(kIndexData)},
                             nullptr);

  const VertexInputStateDesc inputDesc = {
      .numAttributes = 3,
      .attributes = {{.bufferIndex = 0,
                      .format = VertexAttributeFormat::Float3,
                      .offset = offsetof(VertexPosUvw, position),
                      .name = "position",
                      .location = 0},
                     {.bufferIndex = 0,
                      .format = VertexAttributeFormat::Float2,
                      .offset = offsetof(VertexPosUvw, uv),
                      .name = "uv_in",
                      .location = 1},
                     {.bufferIndex = 0,
                      .format = VertexAttributeFormat::Float3,
                      .offset = offsetof(VertexPosUvw, color),
                      .name = "color_in",
                      .location = 2}},
      .numInputBindings = 1,
      .inputBindings = {{.stride = sizeof(VertexPosUvw)}},
  };
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);

  // Command queue: backed by different types of GPU HW queues. The mipmaps need it.
  commandQueue_ = device.createCommandQueue({}, nullptr);

  createSamplerAndTextures(device);
  shaderStages_ = getShaderStagesForBackend(device);

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = getPreferredClearColor(),
      }},
      .depthAttachment =
          {
              .loadAction = LoadAction::Clear,
              .storeAction = StoreAction::DontCare,
              .clearDepth = 1.0,
          },
  };
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void BindGroupSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color || !shaderStages_) {
    return;
  }
  auto& device = getPlatform().getDevice();

  const float deltaSeconds = getDeltaSeconds();

  fps_.updateFPS(deltaSeconds);

  // cube animation
  const glm::mat4 projectionMat = glm::perspectiveLH(
      glm::radians(45.0f), surfaceTextures.color->getAspectRatio(), 0.1f, 100.0f);
  angle_ += 180.0f * deltaSeconds;
  vertexParameters_.mvpMatrix =
      projectionMat * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.f, 8.0f)) *
      glm::rotate(glm::mat4(1.0f), -0.2f, glm::vec3(1.0f, 0.0f, 0.0f)) *
      glm::rotate(glm::mat4(1.0f), glm::radians(angle_), glm::vec3(0.0f, 1.0f, 0.0f));

  Result ret;
  if (framebuffer_ == nullptr) {
    framebufferDesc_.colorAttachments[0].texture = surfaceTextures.color;
    framebufferDesc_.depthAttachment.texture = surfaceTextures.depth;
    framebuffer_ = device.createFramebuffer(framebufferDesc_, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  if (pipelineState_ == nullptr) {
    pipelineState_ = device.createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInput0_,
            .shaderStages = shaderStages_,
            .targetDesc =
                {
                    .colorAttachments = {{
                        .textureFormat =
                            framebuffer_->getColorAttachment(0)->getProperties().format,
                    }},
                    .depthAttachmentFormat =
                        framebuffer_->getDepthAttachment()->getProperties().format,
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
            .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE("input2D")},
                                       {1, IGL_NAMEHANDLE("inputXOR")}},
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
  }

  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);

  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);

  const iglu::ManagedUniformBufferInfo info = {
      .index = 1,
      .length = sizeof(VertexFormat),
      .uniforms = {{.name = "mvpMatrix",
                    .type = UniformType::Mat4x4,
                    .offset = offsetof(VertexFormat, mvpMatrix)}},
  };

  const std::shared_ptr<iglu::ManagedUniformBuffer> vertUniformBuffer =
      std::make_shared<iglu::ManagedUniformBuffer>(device, info);
  IGL_DEBUG_ASSERT(vertUniformBuffer->result.isOk());
  *static_cast<VertexFormat*>(vertUniformBuffer->getData()) = vertexParameters_;
  vertUniformBuffer->bind(device, *pipelineState_, *commands);
  commands->bindBindGroup(bindGroupTextures_);
  commands->bindRenderPipelineState(pipelineState_);
  commands->bindVertexBuffer(0, *vb0_);
  commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
  commands->drawIndexed(static_cast<size_t>(3u * 6u * 2u));

  imguiSession_->beginFrame(framebufferDesc_, getPlatform().getDisplayContext().pixelsPerPoint);
  imguiSession_->drawFPS(fps_.getAverageFPS());
  imguiSession_->endFrame(device, *commands);

  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(framebuffer_->getColorAttachment(0));
  }

  commandQueue_->submit(*buffer);
}

} // namespace igl::shell
