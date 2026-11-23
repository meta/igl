/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/CheckerboardMipmapSession.h>

#include <IGLU/simdtypes/SimdTypes.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <shell/shared/renderSession/RenderSession.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>

// @fb-only
// @fb-only
// @fb-only
// @fb-only
// @fb-only
// @fb-only

namespace igl::shell {

namespace {

struct VertexPosUv {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float2 uv;
};
const VertexPosUv kVertexData[] = {
    {.position = {-0.9f, 0.9f, 0.0}, .uv = {0.0, 1.0}},
    {.position = {0.9f, 0.9f, 0.0}, .uv = {1.0, 1.0}},
    {.position = {-0.9f, -0.9f, 0.0}, .uv = {0.0, 0.0}},
    {.position = {0.9f, -0.9f, 0.0}, .uv = {1.0, 0.0}},
};

const uint16_t kIndexData[] = {0, 1, 2, 1, 3, 2};

// @fb-only
// @fb-only
  // @fb-only
          // @fb-only
          // @fb-only
          // @fb-only
          // @fb-only
          // @fb-only
// @fb-only
// @fb-only

glm::mat4 getMVP(float aspectRatio) noexcept {
  const glm::vec3 eye = {0, 0, 2.5};
  const glm::vec3 ballCenter = {0, 0, 0};
  const float fov = M_PI / 4;

  const glm::mat4 view = glm::lookAt(glm::vec3(eye.x, eye.y, eye.z),
                                     glm::vec3(ballCenter.x, ballCenter.y, ballCenter.z),
                                     glm::vec3(0.0, 1.0, 0.0));
  const glm::mat4 projection = glm::perspective(fov, aspectRatio, 0.1f, 10.f);
  return projection * view;
}

BufferDesc getVertexBufferDesc(const IDevice& device) {
// @fb-only
  // @fb-only
    // @fb-only
                                    // @fb-only
                                    // @fb-only
                                    // @fb-only
    // @fb-only
                      // @fb-only
                      // @fb-only
                      // @fb-only
  // @fb-only
// @fb-only
  return {.type = BufferDesc::BufferTypeBits::Vertex,
          .data = kVertexData,
          .length = sizeof(kVertexData)};
}

uint32_t getVertexBufferIndex(const IDevice& device) {
// @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
// @fb-only
  return 1;
}

ResourceStorage getIndexBufferResourceStorage(const IDevice& device) {
// @fb-only
  // @fb-only
    // @fb-only
  // @fb-only
// @fb-only
  return BufferDesc{}.storage;
}

std::string getVersion() {
  return "#version 100";
}

std::string getMetalShaderSource() {
  return R"(
  using namespace metal;

  typedef struct {
      float4x4 mvp;
  } UniformBlock;

  typedef struct {
    float3 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
  } VertexIn;

  typedef struct {
    float4 position [[position]];
    float2 uv;
  } VertexOut;

  vertex VertexOut vertexShader(
      uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]],
      constant UniformBlock * ub [[buffer(0)]]) {
    VertexOut out;
    out.position = ub->mvp * float4(vertices[vid].position, 1.0);
    out.uv = vertices[vid].uv;
    return out;
  }

  fragment float4 fragmentShader(
      VertexOut IN [[stage_in]],
      texture2d<float> diffuseTex [[texture(0)]],
      sampler linearSampler [[sampler(0)]],
      constant UniformBlock * ub [[buffer(0)]]) {
    float4 tex = diffuseTex.sample(linearSampler, IN.uv);
    return tex;
  }
  )";
}

std::string getOpenGLVertexShaderSource() {
  return getVersion() + R"(
  precision highp float;
  attribute vec3 position;
  attribute vec2 uv_in;

  uniform mat4 mvp;
  uniform sampler2D inputImage;

  varying vec2 uv;

  void main() {
    gl_Position = mvp * vec4(position, 1.0);
    uv = uv_in;
  })";
}

std::string getOpenGLFragmentShaderSource() {
  return getVersion() + std::string(R"(
  precision highp float;
  uniform sampler2D inputImage;
  varying vec2 uv;

  void main() {
    gl_FragColor = texture2D(inputImage, uv);
  })");
}

std::string getVulkanVertexShaderSource() {
  return R"(
precision highp float;
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv_in;

layout(std140, set = 1, binding = 0) uniform Uniforms {
  mat4 mvpMatrix;
} perFrame;

layout(location = 0) out vec2 uv;

void main() {
  gl_Position = perFrame.mvpMatrix * vec4(position, 1.0);
  uv = uv_in;
}
)";
}

std::string getVulkanFragmentShaderSource() {
  return R"(
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;
layout(set = 0, binding = 0) uniform sampler2D inputImage;
void main() {
  out_FragColor = texture(inputImage, uv);
}
)";
}

// @fb-only

// Bind convention: texture unit 0 at @group(0) @binding(0), its sampler at @binding(1), buffer 0
// at @group(1) @binding(0).
const char* getWgslShaderSource() {
  return R"(
struct UniformBlock {
  mvp : mat4x4f,
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
  return VertexOut(ub.mvp * vec4f(position, 1.0), uv);
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return textureSample(diffuseTex, linearSampler, v.uv);
}
)";
}

std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
  default:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  // @fb-only
    // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
  case BackendType::Vulkan: {
    auto vertexSource = getVulkanVertexShaderSource();
    if (device.hasFeature(DeviceFeatures::Multiview)) {
      vertexSource = R"(#version 450
)" + vertexSource;
    }
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           vertexSource.c_str(),
                                                           "main",
                                                           "",
                                                           getVulkanFragmentShaderSource().c_str(),
                                                           "main",
                                                           "",
                                                           nullptr);
  }
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getMetalShaderSource().c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
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

// NOLINTNEXTLINE(bugprone-exception-escape)
void CheckerboardMipmapSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  const BufferDesc vb0Desc = getVertexBufferDesc(device);
  vb0_ = device.createBuffer(vb0Desc, nullptr);
  IGL_DEBUG_ASSERT(vb0_ != nullptr);

  const BufferDesc ibDesc{.type = BufferDesc::BufferTypeBits::Index,
                          .data = kIndexData,
                          .length = sizeof(kIndexData),
                          .storage = getIndexBufferResourceStorage(device),
                          .debugName = "index"};
  ib0_ = device.createBuffer(ibDesc, nullptr);
  IGL_DEBUG_ASSERT(ib0_ != nullptr);

  const auto vertexBufferIndex = getVertexBufferIndex(getPlatform().getDevice());
  VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes =
          {
              {
                  .bufferIndex = vertexBufferIndex,
                  .format = VertexAttributeFormat::Float3,
                  .offset = offsetof(VertexPosUv, position),
                  .name = "position",
                  .location = 0,
              },
              {
                  .bufferIndex = vertexBufferIndex,
                  .format = VertexAttributeFormat::Float2,
                  .offset = offsetof(VertexPosUv, uv),
                  .name = "uv_in",
                  .location = 1,
              },
          },
      .numInputBindings = 1,
  };
  inputDesc.inputBindings[vertexBufferIndex].stride = sizeof(VertexPosUv);
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);
  IGL_DEBUG_ASSERT(vertexInput0_ != nullptr);

  // Sampler & Texture with mipmaps enabled
  const SamplerStateDesc samplerDesc{
      .minFilter = SamplerMinMagFilter::Linear,
      .magFilter = SamplerMinMagFilter::Linear,
      .mipFilter = SamplerMipFilter::Linear, // Enable mipmap filtering
  };
  samp0_ = device.createSamplerState(samplerDesc, nullptr);
  IGL_DEBUG_ASSERT(samp0_ != nullptr);

  // Load checkerboard texture
  // OpenGL ES 2.0 (GL_EXT_sRGB) cannot generate mipmaps for sRGB textures.
  const auto shaderVersion = device.getShaderVersion();
  const bool isGLES2 = shaderVersion.family == ShaderFamily::GlslEs &&
                       shaderVersion.majorVersion < 3;
  tex0_ = getPlatform().loadTexture(
      "checker.png", true, isGLES2 ? TextureFormat::RGBA_UNorm8 : TextureFormat::RGBA_SRGB);
  IGL_DEBUG_ASSERT(tex0_ != nullptr);
  {
    Result result;

    const auto tempCommandQueue = device.createCommandQueue({}, &result);
    IGL_DEBUG_ASSERT(result.isOk(), "Error %d: %s", result.code, result.message.c_str());
    IGL_DEBUG_ASSERT(tempCommandQueue);
    if (tex0_->isRequiredGenerateMipmap()) {
      tex0_->generateMipmap(*tempCommandQueue, nullptr);
    }
  }

  shaderStages_ = getShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);

  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = getPreferredClearColor(),
      }},
      .depthAttachment =
          {
              .loadAction = LoadAction::Clear,
              .clearDepth = 1.0,
          },
  };
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void CheckerboardMipmapSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  Result ret;
  if (framebuffer_ == nullptr) {
    IGL_DEBUG_ASSERT(surfaceTextures.depth != nullptr);
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

  // Graphics pipeline: state batch that fully configures GPU for rendering
  if (pipelineState_ == nullptr) {
    const RenderPipelineDesc graphicsDesc = {
        .vertexInputState = vertexInput0_,
        .shaderStages = shaderStages_,
        .targetDesc =
            {
                .colorAttachments = {{
                    {.textureFormat = framebuffer_->getColorAttachment(0)->getProperties().format},
                }},
                .depthAttachmentFormat = framebuffer_->getDepthAttachment()->getProperties().format,
            },
        .cullMode = CullMode::Disabled,
        .frontFaceWinding = WindingMode::Clockwise,
        .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputImage")}},
    };
    pipelineState_ = getPlatform().getDevice().createRenderPipeline(graphicsDesc, nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  // Create uniform buffer for platforms that need it (Metal, Vulkan, etc.)
  if (getPlatform().getDevice().hasFeature(DeviceFeatures::UniformBlocks) && !mvpUniformBuffer_) {
    const BufferDesc bufDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                             .length = sizeof(glm::mat4),
                             .storage = ResourceStorage::Shared};
    mvpUniformBuffer_ = getPlatform().getDevice().createBuffer(bufDesc, &ret);
    IGL_DEBUG_ASSERT(mvpUniformBuffer_ != nullptr);
  }

  // Update the angle to rotate the plane. Value obtained empirically so that the rotation isn't too
  // fast/slow
#if ROTATE_PLANE
  planeAngle_ += 0.0016f;
#endif

  const auto aspectRatio = surfaceTextures.color->getAspectRatio();
  const glm::mat4 staticViewProjection = getMVP(aspectRatio);
  const glm::mat4 viewProjection =
      glm::rotate(staticViewProjection, planeAngle_, glm::vec3(0.0, 1.f, 0.0));

  // Command buffer: create, submit and forget
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  const auto commands = buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);

  if (commands) {
    commands->bindRenderPipelineState(pipelineState_);
    commands->bindVertexBuffer(getVertexBufferIndex(getPlatform().getDevice()), *vb0_);
    commands->bindTexture(textureUnit, BindTarget::kFragment, tex0_.get());
    commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());

    // Set the MVP matrix uniform
// @fb-only
    // @fb-only
      // @fb-only
      // @fb-only
      // @fb-only
      // @fb-only
             // @fb-only
             // @fb-only
      // @fb-only
                          // @fb-only
                          // @fb-only
                          // @fb-only
      // @fb-only
      // @fb-only
      // @fb-only
      // @fb-only
      // @fb-only
    } else { // @fb-only}
// @fb-only
    {
// @fb-only
      if (getPlatform().getDevice().hasFeature(DeviceFeatures::BindUniform)) {
        // OpenGL path: bind uniform directly
        const UniformDesc uniformDesc = {
            .location = pipelineState_->getIndexByName(IGL_NAMEHANDLE("mvp"), ShaderStage::Vertex),
            .type = UniformType::Mat4x4,
            .offset = 0,
        };
        commands->bindUniform(uniformDesc, &viewProjection[0][0]);
      } else if (getPlatform().getDevice().hasFeature(DeviceFeatures::UniformBlocks)) {
        // Metal/Vulkan path: upload to buffer and bind buffer
        mvpUniformBuffer_->upload(&viewProjection[0][0], BufferRange{sizeof(glm::mat4)});
        commands->bindBuffer(0, mvpUniformBuffer_.get(), 0);
      }
    }

    commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
    commands->drawIndexed(6);

    commands->endEncoding();
  }

  if (buffer) {
    if (shellParams().shouldPresent) {
      buffer->present(framebuffer_->getColorAttachment(0));
    }
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer);
  RenderSession::update(surfaceTextures);
}

} // namespace igl::shell
