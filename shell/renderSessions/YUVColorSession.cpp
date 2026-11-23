/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/YUVColorSession.h>

#include <IGLU/simdtypes/SimdTypes.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <shell/shared/fileLoader/FileLoader.h>
#include <shell/shared/platform/DisplayContext.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/CommandBuffer.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>
#if IGL_BACKEND_OPENGL
#include <igl/opengl/Version.h>
#endif

namespace igl::shell {

struct VertexPosUv {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float2 uv;
};

namespace {

constexpr VertexPosUv kVertexData[] = {
    {.position = {-1.f, 1.f, 0.0}, .uv = {0.0, 0.0}},
    {.position = {1.f, 1.f, 0.0}, .uv = {1.0, 0.0}},
    {.position = {-1.f, -1.f, 0.0}, .uv = {0.0, 1.0}},
    {.position = {1.f, -1.f, 0.0}, .uv = {1.0, 1.0}},
};
constexpr uint16_t kIndexData[] = {0, 1, 2, 1, 3, 2};

std::string getOpenGLVertexShaderSource() {
  return R"(
                #version 300 es
                precision highp float;
                in vec3 position;
                in vec2 uv_in;

                out vec2 uv;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  uv = uv_in; // position.xy * 0.5 + 0.5;
                })";
}

std::string getOpenGLFragmentShaderSource() {
  return R"(
                #version 300 es
                #extension GL_EXT_YUV_target : require
                precision highp float;
                uniform __samplerExternal2DY2YEXT inputImage;

                in vec2 uv;
                layout (yuv) out vec4 outColor;

                void main() {
                  outColor = texture(inputImage, uv);
                })";
}

std::string getVulkanVertexShaderSource() {
  return R"(
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec2 uv_in;
                layout(location = 0) out vec2 uv;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  uv = uv_in;
                }
                )";
}

std::string getVulkanFragmentShaderSource() {
  return R"(
                layout(location = 0) in vec2 uv;
                layout(location = 0) out vec4 out_FragColor;

                layout(set = 0, binding = 0) uniform sampler2D in_texture;

                void main() {
                  out_FragColor = texture(in_texture, uv);
                }
                )";
}

// BT.709 full range, as the Vulkan YCbCr conversion. The source declares only the planes the
// entry point samples: every declared binding is part of the pipeline layout.
std::string getWgslShaderSource(bool isNV12) {
  std::string source = R"(
@group(0) @binding(0) var yTex : texture_2d<f32>;
@group(0) @binding(1) var ySampler : sampler;
@group(0) @binding(2) var uTex : texture_2d<f32>;
@group(0) @binding(3) var uSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vertexShader(@location(0) position : vec3f, @location(1) uv : vec2f) -> VertexOut {
  return VertexOut(vec4f(position, 1.0), uv);
}

fn yuvToRgb(y : f32, u : f32, v : f32) -> vec4f {
  let cb = u - 0.5;
  let cr = v - 0.5;
  return vec4f(y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb, 1.0);
}
)";
  source += isNV12 ? R"(
@fragment
fn fragmentShader(in : VertexOut) -> @location(0) vec4f {
  let uv = textureSample(uTex, uSampler, in.uv).rg;
  return yuvToRgb(textureSample(yTex, ySampler, in.uv).r, uv.x, uv.y);
}
)"
                   : R"(
@group(0) @binding(4) var vTex : texture_2d<f32>;
@group(0) @binding(5) var vSampler : sampler;

@fragment
fn fragmentShader(in : VertexOut) -> @location(0) vec4f {
  return yuvToRgb(textureSample(yTex, ySampler, in.uv).r,
                  textureSample(uTex, uSampler, in.uv).r,
                  textureSample(vTex, vSampler, in.uv).r);
}
)";
  return source;
}

std::string getMetalPlanarShaderSource(bool isNV12) {
  std::string source = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
  float3 position [[attribute(0)]];
  float2 uv [[attribute(1)]];
};

struct VertexOut {
  float4 position [[position]];
  float2 uv;
};

vertex VertexOut vertexShader(VertexIn in [[stage_in]]) {
  VertexOut out;
  out.position = float4(in.position, 1.0);
  out.uv = in.uv;
  return out;
}

float4 yuvToRgb(float y, float u, float v) {
  const float cb = u - 0.5;
  const float cr = v - 0.5;
  return float4(y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb, 1.0);
}
)";
  source += isNV12 ? R"(
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float> yTex [[texture(0)]],
                               sampler ySampler [[sampler(0)]],
                               texture2d<float> uTex [[texture(1)]],
                               sampler uSampler [[sampler(1)]]) {
  const float2 uv = uTex.sample(uSampler, in.uv).rg;
  return yuvToRgb(yTex.sample(ySampler, in.uv).r, uv.x, uv.y);
}
)"
                   : R"(
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float> yTex [[texture(0)]],
                               sampler ySampler [[sampler(0)]],
                               texture2d<float> uTex [[texture(1)]],
                               sampler uSampler [[sampler(1)]],
                               texture2d<float> vTex [[texture(2)]],
                               sampler vSampler [[sampler(2)]]) {
  return yuvToRgb(yTex.sample(ySampler, in.uv).r,
                  uTex.sample(uSampler, in.uv).r,
                  vTex.sample(vSampler, in.uv).r);
}
)";
  return source;
}

// GLSL ES 3.00 / GLSL 1.30 and up: single-channel R8/RG8 planes need them anyway.
std::unique_ptr<IShaderStages> createOpenGLPlanarShaderStages(IDevice& device, bool isNV12) {
#if IGL_BACKEND_OPENGL
  const auto shaderVersion = device.getShaderVersion();
  if (shaderVersion.majorVersion < 3 && shaderVersion.minorVersion < 30) {
    return nullptr;
  }
  const std::string prolog =
      igl::opengl::getStringFromShaderVersion(shaderVersion) + "\nprecision highp float;\n";
  const std::string vertexSource = prolog + R"(
in vec3 position;
in vec2 uv_in;
out vec2 uv;
void main() {
  gl_Position = vec4(position, 1.0);
  uv = uv_in;
})";
  std::string fragmentSource = prolog + R"(
uniform sampler2D yTex;
uniform sampler2D uTex;
in vec2 uv;
out vec4 fragColor;
vec4 yuvToRgb(float y, float u, float v) {
  float cb = u - 0.5;
  float cr = v - 0.5;
  return vec4(y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb, 1.0);
}
)";
  fragmentSource += isNV12 ? R"(
void main() {
  vec2 chroma = texture(uTex, uv).rg;
  fragColor = yuvToRgb(texture(yTex, uv).r, chroma.x, chroma.y);
})"
                           : R"(
uniform sampler2D vTex;
void main() {
  fragColor = yuvToRgb(texture(yTex, uv).r, texture(uTex, uv).r, texture(vTex, uv).r);
})";
  return igl::ShaderStagesCreator::fromModuleStringInput(
      device, vertexSource.c_str(), "main", "", fragmentSource.c_str(), "main", "", nullptr);
#else
  (void)device;
  (void)isNV12;
  return nullptr;
#endif // IGL_BACKEND_OPENGL
}

std::unique_ptr<IShaderStages> getPlanarShaderStages(IDevice& device, bool isNV12) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (device.getBackendType()) {
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(isNV12).c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device,
        getMetalPlanarShaderSource(isNV12).c_str(),
        "vertexShader",
        "fragmentShader",
        "",
        nullptr);
  case BackendType::OpenGL:
    return createOpenGLPlanarShaderStages(device, isNV12);
  default:
    return nullptr;
  }
}

std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  // @fb-only
  case BackendType::Invalid:
  case BackendType::Metal:
  case BackendType::Custom:
  case BackendType::D3D12: // D3D12 YUV shaders not yet implemented
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(false).c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getVulkanVertexShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           getVulkanFragmentShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getOpenGLVertexShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           getOpenGLFragmentShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           nullptr);
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

} // namespace

YUVColorSession::YUVColorSession(std::shared_ptr<Platform> platform) :
  RenderSession(std::move(platform)) {
  listener_ = std::make_shared<Listener>(*this);
  getPlatform().getInputDispatcher().addKeyListener(listener_);
  getPlatform().getInputDispatcher().addMouseListener(listener_);
  imguiSession_ = std::make_unique<iglu::imgui::Session>(getPlatform().getDevice(),
                                                         getPlatform().getInputDispatcher());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void YUVColorSession::initialize() noexcept {
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

  const VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes =
          {
              {
                  .bufferIndex = 1,
                  .format = VertexAttributeFormat::Float3,
                  .offset = offsetof(VertexPosUv, position),
                  .name = "position",
                  .location = 0,
              },
              {
                  .bufferIndex = 1,
                  .format = VertexAttributeFormat::Float2,
                  .offset = offsetof(VertexPosUv, uv),
                  .name = "uv_in",
                  .location = 1,
              },
          },
      .numInputBindings = 1,
      .inputBindings =
          {
              {},
              {
                  .stride = sizeof(VertexPosUv),
              },
          },
  };
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);
  IGL_DEBUG_ASSERT(vertexInput0_ != nullptr);

  // Samplers & Textures

  // Y, then U and V (420p) or interleaved UV (NV12) at half resolution, in single-plane textures.
  auto createPlanarYUVDemo = [this](IDevice& device,
                                    const char* demoName,
                                    TextureFormat yuvFormat,
                                    uint32_t width,
                                    uint32_t height,
                                    const uint8_t* data) {
    const bool isNV12 = yuvFormat == TextureFormat::YUV_NV12;
    const auto isSampleable = [&device](TextureFormat format) {
      return (device.getTextureFormatCapabilities(format) &
              ICapabilities::TextureFormatCapabilityBits::Sampled) != 0;
    };
    std::shared_ptr<IShaderStages> shaderStages =
        (isSampleable(TextureFormat::R_UNorm8) &&
         (!isNV12 || isSampleable(TextureFormat::RG_UNorm8)))
            ? getPlanarShaderStages(device, isNV12)
            : nullptr;
    if (!shaderStages) {
      IGL_LOG_INFO("YUVColorSession: %s is not supported on this device; skipping it\n", demoName);
      return;
    }
    auto createPlane = [&device](TextureFormat format,
                                 uint32_t planeWidth,
                                 uint32_t planeHeight,
                                 const uint8_t* planeData) {
      auto plane = device.createTexture(
          TextureDesc::new2D(
              format, planeWidth, planeHeight, TextureDesc::TextureUsageBits::Sampled, "YUV plane"),
          nullptr);
      IGL_DEBUG_ASSERT(plane);
      if (plane) {
        plane->upload(TextureRangeDesc::new2D(0, 0, planeWidth, planeHeight), planeData);
      }
      return plane;
    };
    const uint32_t chromaWidth = width / 2;
    const uint32_t chromaHeight = height / 2;
    const uint8_t* chroma = data + static_cast<size_t>(width) * height;
    YUVFormatDemo demo{
        .name = demoName,
        .sampler = device.createSamplerState(
            SamplerStateDesc{
                .minFilter = SamplerMinMagFilter::Linear,
                .magFilter = SamplerMinMagFilter::Linear,
                .addressModeU = SamplerAddressMode::Clamp,
                .addressModeV = SamplerAddressMode::Clamp,
                .debugName = "YUVPlaneSampler",
            },
            nullptr),
        .texture = createPlane(TextureFormat::R_UNorm8, width, height, data),
    };
    if (isNV12) {
      demo.chromaPlanes = {
          createPlane(TextureFormat::RG_UNorm8, chromaWidth, chromaHeight, chroma)};
    } else {
      const size_t chromaSize = static_cast<size_t>(chromaWidth) * chromaHeight;
      demo.chromaPlanes = {
          createPlane(TextureFormat::R_UNorm8, chromaWidth, chromaHeight, chroma),
          createPlane(TextureFormat::R_UNorm8, chromaWidth, chromaHeight, chroma + chromaSize),
      };
    }
    if (!demo.texture || std::find(demo.chromaPlanes.begin(), demo.chromaPlanes.end(), nullptr) !=
                             demo.chromaPlanes.end()) {
      IGL_LOG_ERROR("YUVColorSession: cannot create the %s planes; skipping it\n", demoName);
      return;
    }
    demo.shaderStages = std::move(shaderStages);
    this->yuvFormatDemos_.push_back(std::move(demo));
  };

  auto createYUVDemo =
      [this, &createPlanarYUVDemo](
          IDevice& device, const char* demoName, TextureFormat yuvFormat, const char* fileName) {
        constexpr uint32_t width = 1920;
        constexpr uint32_t height = 1080;

        auto& fileLoader = getPlatform().getFileLoader();
        const auto fileData = fileLoader.loadBinaryData(fileName);
        if (!fileData.data || fileData.length < width * height + width * height / 2) {
          IGL_LOG_ERROR("YUVColorSession: %s is missing or truncated; skipping it\n", fileName);
          return;
        }

        // Backends without sampleable multi-planar YUV textures (WebGPU, Metal, OpenGL) get one
        // texture per plane and convert in the shader.
        if (device.getShaderVersion().family == ShaderFamily::Wgsl ||
            (device.getTextureFormatCapabilities(yuvFormat) &
             ICapabilities::TextureFormatCapabilityBits::Sampled) == 0) {
          createPlanarYUVDemo(device, demoName, yuvFormat, width, height, fileData.data.get());
          return;
        }

        auto sampler =
            device.createSamplerState(SamplerStateDesc::newYUV(yuvFormat, "YUVSampler"), nullptr);
        IGL_DEBUG_ASSERT(sampler != nullptr);

        const TextureDesc textureDesc = igl::TextureDesc::new2D(
            yuvFormat, width, height, TextureDesc::TextureUsageBits::Sampled, "YUV texture");
        const auto texture = device.createTexture(textureDesc, nullptr);
        IGL_DEBUG_ASSERT(texture);
        texture->upload(TextureRangeDesc{.x = 0, .y = 0, .z = 0, .width = width, .height = height},
                        fileData.data.get());

        this->yuvFormatDemos_.push_back(YUVFormatDemo{
            .name = demoName, .sampler = sampler, .texture = texture, .pipelineState = nullptr});
      };

  createYUVDemo(device, "YUV 420p", TextureFormat::YUV_420p, "output_frame_900.420p.yuv");
  createYUVDemo(device, "YUV NV12", TextureFormat::YUV_NV12, "output_frame_900.nv12.yuv");

  if (std::any_of(yuvFormatDemos_.begin(), yuvFormatDemos_.end(), [](const YUVFormatDemo& demo) {
        return demo.shaderStages == nullptr;
      })) {
    shaderStages_ = getShaderStagesForBackend(device);
    IGL_DEBUG_ASSERT(shaderStages_ != nullptr);
  }

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
} // namespace igl::shell

// NOLINTNEXTLINE(bugprone-exception-escape)
void YUVColorSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  Result ret;
  framebufferDesc_.colorAttachments[0].texture = surfaceTextures.color;
  if (framebuffer_ == nullptr) {
    IGL_DEBUG_ASSERT(ret.isOk());
    framebufferDesc_.depthAttachment.texture = surfaceTextures.depth;
    framebuffer_ = getPlatform().getDevice().createFramebuffer(framebufferDesc_, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  YUVFormatDemo* demo = yuvFormatDemos_.empty() ? nullptr : &yuvFormatDemos_[currentDemo_];

  if (demo && !demo->pipelineState) {
    std::unordered_map<size_t, NameHandle> fragmentUnitSamplerMap = {
        {0, IGL_NAMEHANDLE("inputImage")}};
    if (!demo->chromaPlanes.empty()) {
      fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE("yTex")}, {1, IGL_NAMEHANDLE("uTex")}};
      if (demo->chromaPlanes.size() > 1) {
        fragmentUnitSamplerMap[2] = IGL_NAMEHANDLE("vTex");
      }
    }
    const RenderPipelineDesc desc = {
        .vertexInputState = vertexInput0_,
        .shaderStages = demo->shaderStages ? demo->shaderStages : shaderStages_,
        .targetDesc =
            {
                .colorAttachments =
                    {
                        {
                            .textureFormat =
                                framebuffer_->getColorAttachment(0)->getProperties().format,
                        },
                    },
                .depthAttachmentFormat = framebuffer_->getDepthAttachment()->getProperties().format,
            },
        .cullMode = CullMode::Back,
        .frontFaceWinding = WindingMode::Clockwise,
        .fragmentUnitSamplerMap = std::move(fragmentUnitSamplerMap),
        .immutableSamplers = {demo->chromaPlanes.empty() ? demo->sampler : nullptr}, // Ycbcr
                                                                                     // sampler
    };
    demo->pipelineState = getPlatform().getDevice().createRenderPipeline(desc, nullptr);
    IGL_DEBUG_ASSERT(demo->pipelineState != nullptr);
  }

  // Command Buffers
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  auto drawableSurface = framebuffer_->getColorAttachment(0);

  framebuffer_->updateDrawable(drawableSurface);

  // Submit commands
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (commands) {
    if (demo && demo->pipelineState) {
      commands->bindVertexBuffer(0, *vb0_);
      commands->bindVertexBuffer(1, *vb0_);
      commands->bindRenderPipelineState(demo->pipelineState);
      commands->bindTexture(0, BindTarget::kFragment, demo->texture.get());
      commands->bindSamplerState(0, BindTarget::kFragment, demo->sampler.get());
      for (size_t i = 0; i < demo->chromaPlanes.size(); ++i) {
        commands->bindTexture(i + 1, BindTarget::kFragment, demo->chromaPlanes[i].get());
        commands->bindSamplerState(i + 1, BindTarget::kFragment, demo->sampler.get());
      }
      commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
      commands->drawIndexed(6);
    }

    // draw the YUV format name using ImGui
    {
      imguiSession_->beginFrame(framebufferDesc_, getPlatform().getDisplayContext().pixelsPerPoint);
      constexpr ImGuiWindowFlags flags =
          ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
          ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
      ImGui::SetNextWindowPos({15.0f, 15.0f});
      ImGui::SetNextWindowBgAlpha(0.30f);
      ImGui::Begin("##FormatYUV", nullptr, flags);
      if (demo) {
        ImGui::Text("%s", demo->name);
        ImGui::Text("Press any key to change");
      } else {
        ImGui::Text("YUV textures are not supported on this device");
      }
      ImGui::End();
      imguiSession_->endFrame(getPlatform().getDevice(), *commands);
    }

    commands->endEncoding();
  }

  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer);
  RenderSession::update(surfaceTextures);
}

void YUVColorSession::nextFormatDemo() {
  if (!yuvFormatDemos_.empty()) {
    currentDemo_ = (currentDemo_ + 1) % yuvFormatDemos_.size();
  }
}

bool YUVColorSession::Listener::process(const KeyEvent& event) {
  if (!event.isDown) {
    session.nextFormatDemo();
  }
  return true;
}

bool YUVColorSession::Listener::process(const MouseButtonEvent& event) {
  if (!event.isDown) {
    session.nextFormatDemo();
  }
  return true;
}

bool YUVColorSession::Listener::process(const MouseMotionEvent& /*event*/) {
  return false;
}

bool YUVColorSession::Listener::process(const MouseWheelEvent& /*event*/) {
  return false;
}

} // namespace igl::shell
