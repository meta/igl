/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/MRTSession.h>

#include <IGLU/simdtypes/SimdTypes.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/CommandBuffer.h>
#include <igl/NameHandle.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/ShaderCreator.h>
#include <igl/VertexInputState.h>
#if IGL_BACKEND_OPENGL
#include <igl/opengl/Config.h>
#endif

namespace igl::shell {

namespace {

struct VertexPosUv {
  iglu::simdtypes::float3 position; // SIMD 128b aligned
  iglu::simdtypes::float2 uv; // SIMD 128b aligned
};
const VertexPosUv kVertexData0[] = {
    {.position = {-0.9f, 0.9f, 0.0}, .uv = {0.0, 1.0}},
    {.position = {-0.05f, 0.9f, 0.0}, .uv = {1.0, 1.0}},
    {.position = {-0.9f, -0.9f, 0.0}, .uv = {0.0, 0.0}},
    {.position = {-0.05f, -0.9f, 0.0}, .uv = {1.0, 0.0}},
};
const VertexPosUv kVertexData1[] = {
    {.position = {0.05f, 0.9f, 0.0}, .uv = {0.0, 1.0}},
    {.position = {0.90f, 0.9f, 0.0}, .uv = {1.0, 1.0}},
    {.position = {0.05f, -0.9f, 0.0}, .uv = {0.0, 0.0}},
    {.position = {0.90f, -0.9f, 0.0}, .uv = {1.0, 0.0}},
};
const uint16_t kIndexData[] = {
    0,
    1,
    2,
    1,
    3,
    2,
};

enum class ShaderPrecision { Low, Medium, High };

std::string getPrecisionProlog(ShaderPrecision precision) {
#if IGL_BACKEND_OPENGL && IGL_OPENGL_ES
  switch (precision) {
  case ShaderPrecision::Low:
    return {"precision lowp float;"};
  case ShaderPrecision::Medium:
    return {"precision mediump float;"};
  case ShaderPrecision::High:
    return {"precision highp float;"};
  }
#else
  return std::string();
#endif
}

std::string getVersionProlog() {
#if IGL_BACKEND_OPENGL
#if IGL_OPENGL_ES
  return "#version 300 es\n";
#else
  return "#version 410\n";
#endif
#else
  return "";
#endif
}

std::string getMetalShaderSource(int metalShaderIdx) {
  switch (metalShaderIdx) {
  case 0:
    return R"(
                    #include <metal_stdlib>
                    #include <simd/simd.h>
                    #line 0
                    using namespace metal;

                     struct VertexIn {
                       float3 position [[attribute(0)]];
                       float2 uv [[attribute(1)]];
                     };

                     struct VertexOut {
                       float4 position [[position]];
                       float2 uv;
                     };

                     vertex VertexOut vertexShader(
                         uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(0)]]) {
                       VertexOut out;
                       out.position = float4(vertices[vid].position, 1.0);
                       out.uv = vertices[vid].uv;
                       return out;
                     }

                     struct FragmentOutput {
                       float4 colorOutGreen [[color(0)]];
                       float4 colorOutRed [[color(1)]];
                     };

                     fragment FragmentOutput fragmentShader(VertexOut IN [[stage_in]],
                                                            texture2d<float> diffuseTex
                                                            [[texture(0)]]) {
                       constexpr sampler linearSampler(mag_filter::linear,
                                                       min_filter::linear);
                       FragmentOutput f;
                       float4 c = diffuseTex.sample(linearSampler, IN.uv);
                       f.colorOutRed.r = c.r;
                       f.colorOutRed.g = 0.0;
                       f.colorOutRed.b = 0.0;
                       f.colorOutRed.a = 1.0;
                       f.colorOutGreen.r = 0.0;
                       f.colorOutGreen.g = c.g;
                       f.colorOutGreen.b = 0.0;
                       f.colorOutGreen.a = 1.0;

                       return f;
                     })";
  default:
    return R"(
                      #include <metal_stdlib>
                      #include <simd/simd.h>
                      #line 0
                      using namespace metal;

                      struct VertexIn {
                        float3 position [[attribute(0)]];
                        float2 uv [[attribute(1)]];
                      };

                      struct VertexOut {
                        float4 position [[position]];
                        float2 uv;
                      };

                      vertex VertexOut vertexShader(
                          uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(0)]]) {
                        VertexOut out;
                        out.position = float4(vertices[vid].position, 1.0);
                        out.uv = vertices[vid].uv;
                        return out;
                      }

                      fragment float4 fragmentShader(VertexOut IN [[stage_in]],
                                                    texture2d<float> greenTex [[texture(0)]],
                                                    texture2d<float> redTex [[texture(1)]]) {
                        constexpr sampler linearSampler(mag_filter::linear,
                                                        min_filter::linear);
                        float4 c = greenTex.sample(linearSampler, IN.uv) +
                                  redTex.sample(linearSampler, IN.uv);
                        return c;
                      })";
  }
}

std::string getOpenGLVertexShaderSource() {
  return getVersionProlog() + getPrecisionProlog(ShaderPrecision::High) + R"(
                in vec3 position;
                in vec2 uv_in;
                out vec2 uv;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  uv = uv_in;
                })";
}

std::string getOpenGLFragmentShaderSource(int programIndex) {
  if (programIndex == 0) {
    return getVersionProlog() + getPrecisionProlog(ShaderPrecision::High) + R"(
                uniform sampler2D inputImage;
                in vec2 uv;
                layout(location = 0) out vec4 colorGreen;
                layout(location = 1) out vec4 colorRed;
                void main() {
                  vec4 c = texture(inputImage, uv);
                  colorGreen = vec4(0., c.g, 0., 1.0);
                  colorRed = vec4(c.r, 0., 0., 1.0);
                })";
  } else {
    return getVersionProlog() + getPrecisionProlog(ShaderPrecision::High) + R"(
                uniform sampler2D colorRed;
                uniform sampler2D colorGreen;
                in vec2 uv;
                out vec4 colorOut;
                void main() {
                  colorOut = texture(colorRed, uv) + texture(colorGreen, uv);
                })";
  }
}

std::string getVulkanVertexShaderSource() {
  return getPrecisionProlog(ShaderPrecision::High) + R"(
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec2 uv_in;
                layout(location = 0) out vec2 uv;

                void main() {
                  gl_Position = vec4(position, 1.0);
                  uv = uv_in;
                })";
}

std::string getVulkanFragmentShaderSource(int programIndex) {
  if (programIndex == 0) {
    return getPrecisionProlog(ShaderPrecision::High) + R"(
                layout(location = 0) in vec2 uv;
                layout(location = 0) out vec4 colorGreen;
                layout(location = 1) out vec4 colorRed;

                layout(set = 0, binding = 0) uniform sampler2D in_texture;

                void main() {
                  vec4 c = texture(in_texture, uv);
                  colorGreen = vec4(0., c.g, 0., 1.0);
                  colorRed = vec4(c.r, 0., 0., 1.0);
                })";
  } else {
    return getPrecisionProlog(ShaderPrecision::High) + R"(
                layout(location = 0) in vec2 uv;
                layout(location = 0) out vec4 out_FragColor;

                layout(set = 0, binding = 0) uniform sampler2D in_texture_green;
                layout(set = 0, binding = 1) uniform sampler2D in_texture_red;

                void main() {
                  vec2 uv1 = vec2(uv.x, 1.0-uv.y);
                  out_FragColor = texture(in_texture_green, uv1) + texture(in_texture_red, uv1);
                })";
  }
}

// Bind convention: texture unit i at @group(0) @binding(2i), its sampler at @binding(2i+1).
const char* getWgslShaderSource(int programIndex) {
  if (programIndex == 0) {
    return R"(
@group(0) @binding(0) var diffuseTex : texture_2d<f32>;
@group(0) @binding(1) var linearSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

struct FragmentOutput {
  @location(0) colorOutGreen : vec4f,
  @location(1) colorOutRed : vec4f,
};

@vertex
fn vertexShader(@location(0) position : vec3f, @location(1) uv : vec2f) -> VertexOut {
  return VertexOut(vec4f(position, 1.0), uv);
}

@fragment
fn fragmentShader(v : VertexOut) -> FragmentOutput {
  let c = textureSample(diffuseTex, linearSampler, v.uv);
  return FragmentOutput(vec4f(0.0, c.g, 0.0, 1.0), vec4f(c.r, 0.0, 0.0, 1.0));
}
)";
  }
  return R"(
@group(0) @binding(0) var greenTex : texture_2d<f32>;
@group(0) @binding(1) var greenSampler : sampler;
@group(0) @binding(2) var redTex : texture_2d<f32>;
@group(0) @binding(3) var redSampler : sampler;

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
  return textureSample(greenTex, greenSampler, v.uv) + textureSample(redTex, redSampler, v.uv);
}
)";
}

std::unique_ptr<IShaderStages> createShaderStagesForBackend(const IDevice& device,
                                                            int programIndex) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getVulkanVertexShaderSource().c_str(),
        "main",
        "",
        getVulkanFragmentShaderSource(programIndex).c_str(),
        "main",
        "",
        nullptr);
  case BackendType::Custom:
    IGL_DEBUG_ABORT("No Custom shader available");
    return nullptr;
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(programIndex), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::D3D12: {
    if (programIndex == 0) {
      // First pass: write to SV_Target0 and SV_Target1
      static const char* kVS = R"(
        struct VSIn { float3 position: POSITION; float2 uv: TEXCOORD0; };
        struct VSOut { float4 position: SV_POSITION; float2 uv: TEXCOORD0; };
        VSOut main(VSIn v){ VSOut o; o.position=float4(v.position,1); o.uv=v.uv; return o; }
      )";
      static const char* kPS = R"(
        Texture2D inputImage : register(t0); SamplerState s0 : register(s0);
        struct PSIn { float4 position: SV_POSITION; float2 uv: TEXCOORD0; };
        struct PSOut { float4 colorGreen: SV_Target0; float4 colorRed: SV_Target1; };
        PSOut main(PSIn i){ float4 c = inputImage.Sample(s0, i.uv);
          // Input PNG data arrives as BGRA, so route blue channel into the second target.
          PSOut o; o.colorGreen=float4(0,c.g,0,1); o.colorRed=float4(c.b,0,0,1); return o; }
      )";
      return igl::ShaderStagesCreator::fromModuleStringInput(
          device, kVS, "main", "", kPS, "main", "", nullptr);
    } else {
      // Second pass: sample two textures and output sum
      static const char* kVS = R"(
        struct VSIn { float3 position: POSITION; float2 uv: TEXCOORD0; };
        struct VSOut { float4 position: SV_POSITION; float2 uv: TEXCOORD0; };
        VSOut main(VSIn v){ VSOut o; o.position=float4(v.position,1); o.uv=v.uv; return o; }
      )";
      static const char* kPS = R"(
        Texture2D colorGreen : register(t0); Texture2D colorRed : register(t1); SamplerState s0 : register(s0);
        struct PSIn { float4 position: SV_POSITION; float2 uv: TEXCOORD0; };
        float4 main(PSIn i) : SV_Target { float2 uv1=float2(i.uv.x, 1.0 - i.uv.y);
          return colorGreen.Sample(s0, uv1) + colorRed.Sample(s0, uv1); }
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
        getOpenGLFragmentShaderSource(programIndex).c_str(),
        "main",
        "",
        nullptr);
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device,
        getMetalShaderSource(programIndex).c_str(),
        "vertexShader",
        "fragmentShader",
        "",
        nullptr);
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

bool isDeviceCompatible(IDevice& device) noexcept {
  return device.hasFeature(DeviceFeatures::MultipleRenderTargets);
}

} // namespace

void MRTSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

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

  vertexInput_ = device.createVertexInputState(
      VertexInputStateDesc{
          .numAttributes = 2,
          .attributes =
              {
                  VertexAttribute{.bufferIndex = 0,
                                  .format = VertexAttributeFormat::Float3,
                                  .offset = offsetof(VertexPosUv, position),
                                  .name = "position",
                                  .location = 0},
                  VertexAttribute{.bufferIndex = 0,
                                  .format = VertexAttributeFormat::Float2,
                                  .offset = offsetof(VertexPosUv, uv),
                                  .name = "uv_in",
                                  .location = 1},
              },
          .numInputBindings = 1,
          .inputBindings = {{.stride = sizeof(VertexPosUv)}},
      },
      nullptr);

  // Sampler & Texture
  samp0_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
          .debugName = "Sampler: linear",
      },
      nullptr);
  tex0_ = getPlatform().loadTexture("igl.png");

  shaderStagesMRT_ = createShaderStagesForBackend(device, 0);
  shaderStagesDisplayLast_ = createShaderStagesForBackend(device, 1);

  commandQueue_ = device.createCommandQueue({}, nullptr);

  tex0_->generateMipmap(*commandQueue_);

  renderPassMRT_ = {
      .colorAttachments = {{
          {
              .loadAction = LoadAction::Clear,
              .storeAction = StoreAction::Store,
              .clearColor = getPreferredClearColor(),
          },
          {
              .loadAction = LoadAction::Clear,
              .storeAction = StoreAction::Store,
              .clearColor = getPreferredClearColor(),
          },
      }},
  };

  renderPassDisplayLast_ = {
      .colorAttachments = {{
          {
              .loadAction = LoadAction::Clear,
              .storeAction = StoreAction::Store,
              .clearColor = getPreferredClearColor(),
          },
      }},
  };
}

// NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
void MRTSession::update(const SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  createOrUpdateFramebufferMRT(surfaceTextures);

  const size_t textureUnit = 0;

  // Graphics pipeline: state batch that fully configures GPU for rendering
  if (pipelineStateMRT_ == nullptr) {
    pipelineStateMRT_ = device.createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInput_,
            .shaderStages = shaderStagesMRT_,
            .targetDesc =
                {
                    .colorAttachments =
                        {
                            {
                                .textureFormat = surfaceTextures.color->getProperties().format,
                                .blendEnabled = true,
                                .rgbBlendOp = BlendOp::Add,
                                .alphaBlendOp = BlendOp::Add,
                                .srcRGBBlendFactor = BlendFactor::SrcAlpha,
                                .srcAlphaBlendFactor = BlendFactor::SrcAlpha,
                                .dstRGBBlendFactor = BlendFactor::OneMinusSrcAlpha,
                                .dstAlphaBlendFactor = BlendFactor::OneMinusSrcAlpha,
                            },
                            {
                                .textureFormat = surfaceTextures.color->getProperties().format,
                                .blendEnabled = true,
                                .rgbBlendOp = BlendOp::Add,
                                .alphaBlendOp = BlendOp::Add,
                                .srcRGBBlendFactor = BlendFactor::SrcAlpha,
                                .srcAlphaBlendFactor = BlendFactor::SrcAlpha,
                                .dstRGBBlendFactor = BlendFactor::OneMinusSrcAlpha,
                                .dstAlphaBlendFactor = BlendFactor::OneMinusSrcAlpha,
                            },
                        },
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
            .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputImage")}},
        },
        nullptr);
  }

  const std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer({}, nullptr);

  auto commands = buffer->createRenderCommandEncoder(renderPassMRT_, framebufferMRT_);

  commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);

  // Draw call 0
  // clang-format off
  commands->bindVertexBuffer(0, *vb0_);
  commands->bindRenderPipelineState(pipelineStateMRT_);
  commands->bindTexture(textureUnit, BindTarget::kFragment, tex0_.get());
  commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());
  commands->drawIndexed(6);
  // clang-format on

  // Draw call 1
  // clang-format off
  commands->bindVertexBuffer(0, *vb1_);
  commands->drawIndexed(6);
  // clang-format on

  commands->endEncoding();

  createOrUpdateFramebufferDisplayLast(surfaceTextures);

  if (pipelineStateLastDisplay_ == nullptr) {
    pipelineStateLastDisplay_ = device.createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInput_,
            .shaderStages = shaderStagesDisplayLast_,
            .targetDesc =
                {
                    .colorAttachments =
                        {
                            {
                                .textureFormat = surfaceTextures.color->getProperties().format,
                            },
                        },
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::Clockwise,
            .fragmentUnitSamplerMap =
                {
                    {textureUnit, IGL_NAMEHANDLE("colorRed")},
                    {textureUnit + 1, IGL_NAMEHANDLE("colorGreen")},
                },
        },
        nullptr);
  }

  // Command buffers (1-N per thread): create, submit and forget

  commands = buffer->createRenderCommandEncoder(renderPassDisplayLast_, framebufferDisplayLast_);

  commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);

  // Draw call 0
  // clang-format off
  commands->bindRenderPipelineState(pipelineStateLastDisplay_);
  const auto green = framebufferMRT_->getColorAttachment(0);
  commands->bindTexture(textureUnit, BindTarget::kFragment, green.get());
  commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());
  const auto red = framebufferMRT_->getColorAttachment(1);
  commands->bindTexture(textureUnit+1, BindTarget::kFragment, red.get());
  commands->bindSamplerState(textureUnit+1, BindTarget::kFragment, samp0_.get());

  commands->bindVertexBuffer(0,  *vb0_);
  commands->drawIndexed(6);

  commands->bindVertexBuffer(0, *vb1_);
  commands->drawIndexed(6);

  // clang-format on
  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(surfaceTextures.color);
  }

  commandQueue_->submit(*buffer); // Guarantees ordering between command buffers
}

std::shared_ptr<ITexture> MRTSession::createTexture2D(const std::shared_ptr<ITexture>& tex) {
  const auto dimensions = tex->getDimensions();
  TextureDesc desc = TextureDesc::new2D(tex->getProperties().format,
                                        dimensions.width,
                                        dimensions.height,
                                        TextureDesc::TextureUsageBits::Attachment |
                                            TextureDesc::TextureUsageBits::Sampled);
  desc.debugName = "shell/renderSessions/MRTSession.cpp:MRTSession::createTexture2D()";

  return getPlatform().getDevice().createTexture(desc, nullptr);
}

void MRTSession::createOrUpdateFramebufferDisplayLast(const SurfaceTextures& surfaceTextures) {
  if (framebufferDisplayLast_) {
    framebufferDisplayLast_->updateDrawable(surfaceTextures.color);
    return;
  }

  // Framebuffer & Texture
  const FramebufferDesc framebufferDesc = {
      .colorAttachments = {{.texture = surfaceTextures.color}},
  };

  framebufferDisplayLast_ = getPlatform().getDevice().createFramebuffer(framebufferDesc, nullptr);
}

void MRTSession::createOrUpdateFramebufferMRT(const SurfaceTextures& surfaceTextures) {
  if (framebufferMRT_) {
    return;
  }

  if (tex1_ == nullptr) {
    tex1_ = createTexture2D(surfaceTextures.color);
  }
  if (tex2_ == nullptr) {
    tex2_ = createTexture2D(surfaceTextures.color);
  }
  // Framebuffer & Texture
  const FramebufferDesc framebufferDesc = {
      .colorAttachments =
          {
              {.texture = tex1_},
              {.texture = tex2_},
          },
  };

  framebufferMRT_ = getPlatform().getDevice().createFramebuffer(framebufferDesc, nullptr);
}

} // namespace igl::shell
