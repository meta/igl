/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/Textured3DCubeSession.h>

#include <IGLU/managedUniformBuffer/ManagedUniformBuffer.h>
#include <cmath>
#include <cstddef>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>
#include <igl/opengl/Version.h> // IWYU pragma: keep

namespace igl::shell {

namespace {

struct VertexPosUvw {
  glm::vec3 position;
  glm::vec3 uvw;
};

const float kHalf = 1.0f;
const VertexPosUvw kVertexData0[] = {
    {.position = {-kHalf, kHalf, -kHalf}, .uvw = {0.0, 1.0, 0.0}},
    {.position = {kHalf, kHalf, -kHalf}, .uvw = {1.0, 1.0, 0.0}},
    {.position = {-kHalf, -kHalf, -kHalf}, .uvw = {0.0, 0.0, 0.0}},
    {.position = {kHalf, -kHalf, -kHalf}, .uvw = {1.0, 0.0, 0.0}},
    {.position = {kHalf, kHalf, kHalf}, .uvw = {1.0, 1.0, 1.0}},
    {.position = {-kHalf, kHalf, kHalf}, .uvw = {0.0, 1.0, 1.0}},
    {.position = {kHalf, -kHalf, kHalf}, .uvw = {1.0, 0.0, 1.0}},
    {.position = {-kHalf, -kHalf, kHalf}, .uvw = {0.0, 0.0, 1.0}},
};
const uint16_t kIndexData[] = {0, 1, 2, 1, 3, 2, 1, 4, 3, 4, 6, 3, 4, 5, 6, 5, 7, 6,
                               5, 0, 7, 0, 2, 7, 5, 4, 0, 4, 1, 0, 2, 3, 7, 3, 6, 7};

std::string getProlog(IDevice& device) {
#if IGL_BACKEND_OPENGL
  const auto shaderVersion = device.getShaderVersion();
  if (shaderVersion.majorVersion >= 3 || shaderVersion.minorVersion >= 30) {
    std::string prependVersionString = igl::opengl::getStringFromShaderVersion(shaderVersion);
    prependVersionString += "\nprecision highp float;\n";
    return prependVersionString;
  }
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
            float scaleZ;
          };

          struct VertexIn {
            float3 position [[attribute(0)]];
            float3 uvw [[attribute(1)]];
          };

          struct VertexOut {
            float4 position [[position]];
            float3 uvw;
          };

          vertex VertexOut vertexShader(VertexIn in [[stage_in]],
                 constant VertexUniformBlock &vUniform[[buffer(1)]]) {
            VertexOut out;
            out.position = vUniform.mvpMatrix * float4(in.position, 1.0);
            out.uvw = in.uvw;
            out.uvw = float3(
                         out.uvw.x, out.uvw.y, (out.uvw.z - 0.5f)*vUniform.scaleZ + 0.5f);
            return out;
           }

           fragment float4 fragmentShader(
                 VertexOut in[[stage_in]],
                 texture3d<float> diffuseTex [[texture(0)]],
                 sampler linearSampler [[sampler(0)]]) {
             constexpr sampler s(s_address::clamp_to_edge,
                                 t_address::clamp_to_edge,
                                 min_filter::linear,
                                 mag_filter::linear);
             float4 tex = diffuseTex.sample(s, in.uvw);
             return tex;
           }
        )";
}

std::string getOpenGLFragmentShaderSource(IDevice& device) {
  return getProlog(device) + std::string(R"(
                      precision highp float; precision highp sampler3D;
                      in vec3 uvw;
                      uniform sampler3D inputVolume;
                      out vec4 fragmentColor;
                      void main() {
                        fragmentColor = texture(inputVolume, uvw);
                      })");
}

std::string getOpenGLVertexShaderSource(IDevice& device) {
  return getProlog(device) + R"(
                      precision highp float;
                      uniform mat4 mvpMatrix;
                      uniform float scaleZ;
                      in vec3 position;
                      in vec3 uvw_in;
                      out vec3 uvw;

                      void main() {
                        gl_Position =  mvpMatrix * vec4(position, 1.0);
                        uvw = vec3(uvw_in.x, uvw_in.y, (uvw_in.z-0.5)*scaleZ+0.5);
                      })";
}

const char* getVulkanFragmentShaderSource() {
  return R"(
                      precision highp float;
                      layout(location = 0) in vec3 uvw;
                      layout(location = 0) out vec4 out_FragColor;

                      layout(set = 0, binding = 0) uniform sampler3D in_texture;

                      void main() {
                        out_FragColor = texture(in_texture, uvw);
                      })";
}

const char* getVulkanVertexShaderSource() {
  return R"(
                      precision highp float;

                      layout (set = 1, binding = 1, std140) uniform PerFrame {
                        mat4 mvpMatrix;
                        float scaleZ;
                      } perFrame;

                      layout(location = 0) in vec3 position;
                      layout(location = 1) in vec3 uvw_in;
                      layout(location = 0) out vec3 uvw;

                      void main() {
                        gl_Position =  perFrame.mvpMatrix * vec4(position, 1.0);
                        uvw = vec3(uvw_in.x, uvw_in.y, (uvw_in.z-0.5)*perFrame.scaleZ+0.5);
                      })";
}

// Bind convention: texture unit 0 at @group(0) @binding(0), its sampler at @binding(1), buffer 1
// at @group(1) @binding(1).
const char* getWgslShaderSource() {
  return R"(
struct VertexUniformBlock {
  mvpMatrix : mat4x4f,
  scaleZ : f32,
};

@group(1) @binding(1) var<uniform> vUniform : VertexUniformBlock;
@group(0) @binding(0) var diffuseTex : texture_3d<f32>;
@group(0) @binding(1) var linearSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uvw : vec3f,
};

@vertex
fn vertexShader(@location(0) position : vec3f, @location(1) uvw : vec3f) -> VertexOut {
  return VertexOut(vUniform.mvpMatrix * vec4f(position, 1.0),
                   vec3f(uvw.xy, (uvw.z - 0.5) * vUniform.scaleZ + 0.5));
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return textureSample(diffuseTex, linearSampler, v.uvw);
}
)";
}

std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getVulkanVertexShaderSource(),
                                                           "main",
                                                           "",
                                                           getVulkanFragmentShaderSource(),
                                                           "main",
                                                           "",
                                                           nullptr);
  case BackendType::D3D12: {
    static const char* kVS = R"(
      cbuffer VertexUniforms : register(b1) { float4x4 mvpMatrix; float scaleZ; };
      struct VSIn { float3 position : POSITION; float3 uvw : TEXCOORD0; };
      struct VSOut { float4 position : SV_POSITION; float3 uvw : TEXCOORD0; };
      VSOut main(VSIn v) {
        VSOut o; o.position = mul(mvpMatrix, float4(v.position,1.0));
        o.uvw = float3(v.uvw.x, v.uvw.y, (v.uvw.z - 0.5f)*scaleZ + 0.5f);
        return o; }
    )";
    static const char* kPS = R"(
      Texture3D<float4> inputVolume : register(t0);
      SamplerState linearSampler : register(s0);
      struct PSIn { float4 position : SV_POSITION; float3 uvw : TEXCOORD0; };
      float4 main(PSIn i) : SV_TARGET { return inputVolume.Sample(linearSampler, i.uvw); }
    )";
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kVS, "main", "", kPS, "main", "", nullptr);
  }
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

bool isDeviceCompatible(IDevice& device) noexcept {
  return device.hasFeature(DeviceFeatures::Texture3D);
}

} // namespace

void Textured3DCubeSession::createSamplerAndTextures(const IDevice& device) {
  // Sampler & Texture
  samp0_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
          .addressModeU = SamplerAddressMode::MirrorRepeat,
          .addressModeV = SamplerAddressMode::MirrorRepeat,
          .addressModeW = SamplerAddressMode::MirrorRepeat,
          .debugName = "Sampler: linear (MirrorRepeat)",
      },
      nullptr);

  const uint32_t width = 256;
  const uint32_t height = 256;
  const uint32_t depth = 256;
  const uint32_t bytesPerPixel = 4;
  auto textureData =
      std::vector<uint8_t>(static_cast<size_t>(width) * height * depth * bytesPerPixel);
  for (uint32_t k = 0; k < depth; ++k) {
    for (uint32_t j = 0; j < height; ++j) {
      for (uint32_t i = 0; i < width; ++i) {
        const uint32_t index = (i + width * j + width * height * k) * bytesPerPixel;
        const float d = std::sqrt((i - 128.0f) * (i - 128.0f) + (j - 128.0f) * (j - 128.0f) +
                                  (k - 128.0f) * (k - 128.0f)) /
                        16.0f;
        if (d > 7.0f) {
          textureData[index + 0] = 148;
          textureData[index + 1] = 0;
          textureData[index + 2] = 211;
          textureData[index + 3] = 255;
        } else if (d > 6.0f) {
          textureData[index + 0] = 75;
          textureData[index + 1] = 0;
          textureData[index + 2] = 130;
          textureData[index + 3] = 255;
        } else if (d > 5.0f) {
          textureData[index + 0] = 0;
          textureData[index + 1] = 0;
          textureData[index + 2] = 255;
          textureData[index + 3] = 255;
        } else if (d > 4.0f) {
          textureData[index + 0] = 0;
          textureData[index + 1] = 255;
          textureData[index + 2] = 0;
          textureData[index + 3] = 255;
        } else if (d > 3.0f) {
          textureData[index + 0] = 255;
          textureData[index + 1] = 255;
          textureData[index + 2] = 0;
          textureData[index + 3] = 255;
        } else if (d > 2.0f) {
          textureData[index + 0] = 255;
          textureData[index + 1] = 127;
          textureData[index + 2] = 0;
          textureData[index + 3] = 255;
        } else {
          textureData[index + 0] = 255;
          textureData[index + 1] = 0;
          textureData[index + 2] = 0;
          textureData[index + 3] = 255;
        }
      }
    }
  }

  TextureDesc texDesc = igl::TextureDesc::new3D(TextureFormat::RGBA_UNorm8,
                                                width,
                                                height,
                                                depth,
                                                igl::TextureDesc::TextureUsageBits::Sampled);
  texDesc.debugName = "shell/renderSessions/Textured3DCubeSession.cpp:tex0_";
  tex0_ = getPlatform().getDevice().createTexture(texDesc, nullptr);
  const auto range = TextureRangeDesc::new3D(0, 0, 0, width, height, depth);
  tex0_->upload(range, textureData.data());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void Textured3DCubeSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }
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
      .numAttributes = 2,
      .attributes = {{
                         .bufferIndex = 0,
                         .format = VertexAttributeFormat::Float3,
                         .offset = offsetof(VertexPosUvw, position),
                         .name = "position",
                         .location = 0,
                     },
                     {
                         .bufferIndex = 0,
                         .format = VertexAttributeFormat::Float3,
                         .offset = offsetof(VertexPosUvw, uvw),
                         .name = "uvw_in",
                         .location = 1,
                     }},
      .numInputBindings = 1,
      .inputBindings = {{.stride = sizeof(VertexPosUvw)}},
  };
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);

  createSamplerAndTextures(device);
  shaderStages_ = getShaderStagesForBackend(device);

  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = device.createCommandQueue({}, nullptr);

  // Set up vertex uniform data
  vertexParameters_.scaleZ = 1.0f;

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = getPreferredClearColor(),
      }},
      .depthAttachment = {.loadAction = LoadAction::Clear, .clearDepth = 1.0},
  };
}

void Textured3DCubeSession::setVertexParams(float aspectRatio) {
  // perspective projection
  const float fov = 45.0f * (M_PI / 180.0f);
  const glm::mat4 projectionMat = glm::perspectiveLH(fov, aspectRatio, 0.1f, 100.0f);
  // rotating animation
  static float angle = 0.0f, scaleZ = 1.0f, ss = 0.005f;
  angle += 0.005f;
  scaleZ += ss;
  scaleZ = scaleZ < 0.0f ? 0.0f : scaleZ > 1.0 ? 1.0f : scaleZ;
  if (scaleZ <= 0.05f || scaleZ >= 1.0f) {
    ss *= -1.0f;
  }
  const glm::mat4 xform = projectionMat *
                          glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.f, 8.0f)) *
                          glm::rotate(glm::mat4(1.0f), -0.2f, glm::vec3(1.0f, 0.0f, 0.0f)) *
                          glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 1.0f, 0.0f)) *
                          glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, scaleZ));

  vertexParameters_.mvpMatrix = xform;
  vertexParameters_.scaleZ = scaleZ;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void Textured3DCubeSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  // cube animation
  setVertexParams(surfaceTextures.color->getAspectRatio());

  Result ret;
  if (framebuffer_ == nullptr) {
    framebuffer_ = getPlatform().getDevice().createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color}},
            .depthAttachment = {.texture = surfaceTextures.depth},
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  const size_t textureUnit = 0;
  if (pipelineState_ == nullptr) {
    // Graphics pipeline: state batch that fully configures GPU for rendering

    const RenderPipelineDesc graphicsDesc = {
        .vertexInputState = vertexInput0_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments =
                           {{.textureFormat =
                                 framebuffer_->getColorAttachment(0)->getProperties().format}},
                       .depthAttachmentFormat =
                           framebuffer_->getDepthAttachment()->getProperties().format},
        .cullMode = CullMode::Back,
        .frontFaceWinding = WindingMode::Clockwise,
        .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputVolume")}},
    };
    pipelineState_ = getPlatform().getDevice().createRenderPipeline(graphicsDesc, nullptr);
  }

  // Command buffers (1-N per thread): create, submit and forget
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);

  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);

  commands->bindVertexBuffer(0, *vb0_);

  // Bind Vertex Uniform Data
  const iglu::ManagedUniformBufferInfo info = {
      .index = 1,
      .length = sizeof(VertexFormat),
      .uniforms =
          {
              {
                  .name = "mvpMatrix",
                  .location = -1,
                  .type = UniformType::Mat4x4,
                  .numElements = 1,
                  .offset = offsetof(VertexFormat, mvpMatrix),
                  .elementStride = 0,
              },
              {
                  .name = "scaleZ",
                  .location = -1,
                  .type = UniformType::Float,
                  .numElements = 1,
                  .offset = offsetof(VertexFormat, scaleZ),
                  .elementStride = 0,
              },
          },
  };

  const std::shared_ptr<iglu::ManagedUniformBuffer> vertUniformBuffer =
      std::make_shared<iglu::ManagedUniformBuffer>(device, info);
  IGL_DEBUG_ASSERT(vertUniformBuffer->result.isOk());
  *static_cast<VertexFormat*>(vertUniformBuffer->getData()) = vertexParameters_;
  vertUniformBuffer->bind(device, *pipelineState_, *commands);

  commands->bindTexture(textureUnit, BindTarget::kFragment, tex0_.get());
  commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());

  commands->bindRenderPipelineState(pipelineState_);

  commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
  commands->drawIndexed(static_cast<size_t>(3u * 6u * 2u));

  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(framebuffer_->getColorAttachment(0));
  }

  commandQueue_->submit(*buffer); // Guarantees ordering between command buffers
}

} // namespace igl::shell
