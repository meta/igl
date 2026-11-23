/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @MARK:COVERAGE_EXCLUDE_FILE

#include <shell/renderSessions/TextureViewSession.h>

#include <cstddef>
#include <shell/shared/platform/DisplayContext.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/CommandBuffer.h>
#include <igl/Device.h>
#include <igl/NameHandle.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/ShaderCreator.h>
#include <igl/VertexInputState.h>

namespace {

struct VertexPosUvw {
  glm::vec3 position;
  glm::vec2 uv;
};

constexpr float kHalf = 1.2f;

// UV-mapped cube with indices: 24 vertices, 36 indices
constexpr VertexPosUvw kVertexData[] = {
    // top
    {.position = {-kHalf, -kHalf, +kHalf}, .uv = {0, 0}}, // 0
    {.position = {+kHalf, -kHalf, +kHalf}, .uv = {1, 0}}, // 1
    {.position = {+kHalf, +kHalf, +kHalf}, .uv = {1, 1}}, // 2
    {.position = {-kHalf, +kHalf, +kHalf}, .uv = {0, 1}}, // 3
    // bottom
    {.position = {-kHalf, -kHalf, -kHalf}, .uv = {0, 0}}, // 4
    {.position = {-kHalf, +kHalf, -kHalf}, .uv = {0, 1}}, // 5
    {.position = {+kHalf, +kHalf, -kHalf}, .uv = {1, 1}}, // 6
    {.position = {+kHalf, -kHalf, -kHalf}, .uv = {1, 0}}, // 7
    // left
    {.position = {+kHalf, +kHalf, -kHalf}, .uv = {1, 0}}, // 8
    {.position = {-kHalf, +kHalf, -kHalf}, .uv = {0, 0}}, // 9
    {.position = {-kHalf, +kHalf, +kHalf}, .uv = {0, 1}}, // 10
    {.position = {+kHalf, +kHalf, +kHalf}, .uv = {1, 1}}, // 11
    // right
    {.position = {-kHalf, -kHalf, -kHalf}, .uv = {0, 0}}, // 12
    {.position = {+kHalf, -kHalf, -kHalf}, .uv = {1, 0}}, // 13
    {.position = {+kHalf, -kHalf, +kHalf}, .uv = {1, 1}}, // 14
    {.position = {-kHalf, -kHalf, +kHalf}, .uv = {0, 1}}, // 15
    // front
    {.position = {+kHalf, -kHalf, -kHalf}, .uv = {0, 0}}, // 16
    {.position = {+kHalf, +kHalf, -kHalf}, .uv = {1, 0}}, // 17
    {.position = {+kHalf, +kHalf, +kHalf}, .uv = {1, 1}}, // 18
    {.position = {+kHalf, -kHalf, +kHalf}, .uv = {0, 1}}, // 19
    // back
    {.position = {-kHalf, +kHalf, -kHalf}, .uv = {1, 0}}, // 20
    {.position = {-kHalf, -kHalf, -kHalf}, .uv = {0, 0}}, // 21
    {.position = {-kHalf, -kHalf, +kHalf}, .uv = {0, 1}}, // 22
    {.position = {-kHalf, +kHalf, +kHalf}, .uv = {1, 1}}, // 23
};

constexpr uint16_t kIndexData[] = {0,  1,  2,  2,  3,  0,  4,  5,  6,  6,  7,  4,
                                   8,  9,  10, 10, 11, 8,  12, 13, 14, 14, 15, 12,
                                   16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};

std::string getMetalShaderSource() {
  return R"(
    #include <metal_stdlib>
    #include <simd/simd.h>
    using namespace metal;

    struct VertexIn {
      float3 position [[attribute(0)]];
      float2 uv [[attribute(1)]];
    };

    struct VertexOut {
      float4 position [[position]];
      float2 uv;
    };

    vertex VertexOut vertexShader(VertexIn in [[stage_in]],
        constant float4x4 mvpMatrix [[function_constant(0)]]) {
      VertexOut out;
      out.position = mvpMatrix * float4(in.position, 1.0);
      out.uv = in.uv;
      return out;
    }

    fragment float4 fragmentShader(
        VertexOut in[[stage_in]],
        texture2d<float> input2D [[texture(0)]],
        sampler linearSampler [[sampler(0)]]) {
      return input2D.sample(linearSampler, in.uv);
    }
  )";
}

const char* getVulkanFragmentShaderSource() {
  return R"(
    precision highp float;
    layout(location = 0) in vec2 uv;
    layout(location = 0) out vec4 out_FragColor;

    layout(set = 0, binding = 0) uniform sampler2D input2D;

    void main() {
      out_FragColor = texture(input2D, uv);
    }
  )";
}

const char* getVulkanVertexShaderSource() {
  return R"(
    precision highp float;

    layout (push_constant) uniform PerFrame {
      mat4 mvpMatrix;
    } perFrame;

    layout(location = 0) in vec3 position;
    layout(location = 1) in vec2 uvw_in;
    layout(location = 0) out vec2 uvw;

    void main() {
      gl_Position =  perFrame.mvpMatrix * vec4(position, 1.0);
      uvw = uvw_in;
    }
  )";
}

const char* getWgslShaderSource() {
  return R"(
struct PerFrame {
  mvpMatrix : mat4x4f,
};

@group(3) @binding(0) var<uniform> perFrame : PerFrame;
@group(0) @binding(0) var input2D : texture_2d<f32>;
@group(0) @binding(1) var linearSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vertexShader(@location(0) position : vec3f, @location(1) uv : vec2f) -> VertexOut {
  return VertexOut(perFrame.mvpMatrix * vec4f(position, 1.0), uv);
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return textureSample(input2D, linearSampler, v.uv);
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
    static const char* kVS = R"(
      cbuffer PushConstants : register(b2) { float4x4 mvpMatrix; };
      struct VSIn { float3 position : POSITION; float2 uv : TEXCOORD0; };
      struct VSOut { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
      VSOut main(VSIn v) {
        VSOut o;
        o.position = mul(mvpMatrix, float4(v.position, 1.0));
        o.uv = v.uv;
        return o;
      }
    )";
    static const char* kPS = R"(
      Texture2D<float4> input2D : register(t0);
      SamplerState linearSampler : register(s0);
      struct PSIn { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
      float4 main(PSIn i) : SV_TARGET { return input2D.Sample(linearSampler, i.uv); }
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
    IGL_LOG_INFO("TextureViewSession: OpenGL is not supported; skipping\n");
    return nullptr;
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

} // namespace

namespace igl::shell {

TextureViewSession::TextureViewSession(std::shared_ptr<Platform> platform) :
  RenderSession(std::move(platform)) {
  imguiSession_ = std::make_unique<iglu::imgui::Session>(getPlatform().getDevice(),
                                                         getPlatform().getInputDispatcher());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void TextureViewSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  const bool hasTextureViews = device.hasFeature(DeviceFeatures::TextureViews);
  if (!hasTextureViews) {
    IGL_LOG_INFO("TextureViewSession: texture views are not supported; drawing without them\n");
  }

  vb_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                       .data = kVertexData,
                                       .length = sizeof(kVertexData)},
                            nullptr);
  ib_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
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
                  .offset = offsetof(VertexPosUvw, position),
                  .name = "position",
                  .location = 0,
              },
              {
                  .bufferIndex = 0,
                  .format = VertexAttributeFormat::Float2,
                  .offset = offsetof(VertexPosUvw, uv),
                  .name = "uv_in",
                  .location = 1,
              },
          },
      .numInputBindings = 1,
      .inputBindings = {{.stride = sizeof(VertexPosUvw)}},
  };
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);

  shaderStages_ = getShaderStagesForBackend(device);
  if (!shaderStages_) {
    return;
  }

  commandQueue_ = device.createCommandQueue({}, nullptr);

  sampler_ = device.createSamplerState(SamplerStateDesc::newLinearMipmapped(), nullptr);

  const uint32_t texWidth = 256;
  const uint32_t texHeight = 256;
  TextureDesc desc = TextureDesc::new2D(TextureFormat::BGRA_UNorm8,
                                        texWidth,
                                        texHeight,
                                        TextureDesc::TextureUsageBits::Attachment |
                                            TextureDesc::TextureUsageBits::Sampled,
                                        "Colored mipmaps");
  desc.numMipLevels = TextureDesc::calcNumMipLevels(texWidth, texHeight);
  texture_ = device.createTexture(desc, nullptr);

  if (hasTextureViews) {
    textureViews_.reserve(desc.numMipLevels);
    for (uint32_t mip = 0; mip != desc.numMipLevels; mip++) {
      textureViews_.push_back(device.createTextureView(texture_, {.mipLevel = mip}, nullptr));
    }
  }

  // render into the texture to generate custom colored mipmap pyramid
  auto fb = device.createFramebuffer(
      FramebufferDesc{
          .colorAttachments = {{.texture = texture_}},
      },
      nullptr);
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  const std::array<Color, 10> kColors = {
      Color{1, 0, 0},
      Color{0, 1, 0},
      Color{0, 0, 1},

      Color{1, 1, 0},
      Color{0, 1, 1},
      Color{1, 0, 1},

      Color{1, 0, 0},
      Color{0, 1, 0},
      Color{0, 0, 1},

      Color{0, 0, 0},
  };
  for (uint32_t i = 0; i != desc.numMipLevels; i++) {
    const RenderPassDesc pass = {
        .colorAttachments = {{
            .loadAction = LoadAction::Clear,
            .storeAction = StoreAction::Store,
            .mipLevel = static_cast<uint8_t>(i),
            .clearColor = kColors[i % kColors.size()],
        }},
    };
    const auto commands = buffer->createRenderCommandEncoder(pass, fb);
    commands->endEncoding();
  }
  commandQueue_->submit(*buffer);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void TextureViewSession::update(SurfaceTextures surfaceTextures) noexcept {
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
  angle_ += 90.0f * deltaSeconds;
  while (angle_ > 360.0f) {
    angle_ -= 360.0f;
  }
  const glm::mat4 mvpMatrix =
      projectionMat * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.f, 8.0f)) *
      glm::rotate(
          glm::mat4(1.0f), glm::radians(angle_), glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f)));

  if (!framebuffer_) {
    framebufferDesc_.colorAttachments[0].texture = surfaceTextures.color;
    framebufferDesc_.depthAttachment.texture = surfaceTextures.depth;
    Result ret;
    framebuffer_ = device.createFramebuffer(framebufferDesc_, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  if (!pipelineState_) {
    Result ret;
    pipelineState_ = device.createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInput0_,
            .shaderStages = shaderStages_,
            .targetDesc =
                {
                    .colorAttachments =
                        {
                            {.textureFormat =
                                 framebuffer_->getColorAttachment(0)->getProperties().format},
                        },
                    .depthAttachmentFormat =
                        framebuffer_->getDepthAttachment()->getProperties().format,
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
  }

  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);

  const RenderPassDesc renderPass{
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
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass, framebuffer_);

  commands->bindTexture(0, texture_.get());
  commands->bindSamplerState(0, BindTarget::kFragment, sampler_.get());
  commands->bindRenderPipelineState(pipelineState_);
  if (device.getBackendType() == BackendType::Vulkan ||
      device.getBackendType() == BackendType::WebGPU) {
    commands->bindPushConstants(&mvpMatrix, sizeof(mvpMatrix));
  } else if (device.getBackendType() == BackendType::Metal) {
    commands->bindBytes(0, BindTarget::kVertex, &mvpMatrix, sizeof(mvpMatrix));
  } else if (device.getBackendType() == BackendType::D3D12) {
    commands->bindPushConstants(&mvpMatrix, sizeof(mvpMatrix));
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }
  commands->bindVertexBuffer(0, *vb_);
  commands->bindIndexBuffer(*ib_, IndexFormat::UInt16);
  commands->drawIndexed(36);

  imguiSession_->beginFrame(framebufferDesc_, getPlatform().getDisplayContext().pixelsPerPoint);
  imguiSession_->drawFPS(fps_.getAverageFPS());
  const ImGuiViewport* v = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(ImVec2(10, 10));
  ImGui::Begin("Mip-pyramid", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  const int width = static_cast<int>(v->WorkSize.x / 5.0f);
  for (uint32_t mip = 0; mip != textureViews_.size(); mip++) {
    ImGui::Image(ImTextureID(textureViews_[mip].get()), ImVec2(width >> mip, width >> mip));
  }
  ImGui::End();
  imguiSession_->endFrame(device, *commands);

  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(framebuffer_->getColorAttachment(0));
  }

  commandQueue_->submit(*buffer);
}

} // namespace igl::shell
