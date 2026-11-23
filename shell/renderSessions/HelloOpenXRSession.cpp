/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/HelloOpenXRSession.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <glm/detail/qualifier.hpp>
#include <IGLU/shaderCross/ShaderCross.h>
#include <IGLU/shaderCross/ShaderCrossUniformBuffer.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/Device.h>
#include <igl/NameHandle.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/ShaderCreator.h>

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
constexpr uint16_t kIndexData[] = {0, 1, 2, 1, 3, 2, 1, 4, 3, 4, 6, 3, 4, 5, 6, 5, 7, 6,
                                   5, 0, 7, 0, 2, 7, 5, 4, 0, 4, 1, 0, 2, 3, 7, 3, 6, 7};

[[nodiscard]] const char* getVulkanFragmentShaderSource() {
  return R"(#version 450
            precision highp float;
            precision highp sampler2D;

            layout(location = 0) in vec3 uvw;
            layout(location = 1) in vec3 color;
            layout(set = 0, binding = 0) uniform sampler2D inputImage;
            layout(location = 0) out vec4 fragmentColor;

            void main() {
              fragmentColor = texture(inputImage, uvw.xy) * vec4(color, 1.0);
            })";
}

[[nodiscard]] std::string getVertexShaderProlog(bool stereoRendering) {
  return stereoRendering ? R"(#version 450
    #extension GL_OVR_multiview2 : require
    layout(num_views = 2) in;
    precision highp float;

    #define VIEW_ID int(gl_ViewID_OVR)
  )"
                         : R"(#version 450
    precision highp float;

    #define VIEW_ID perFrame.viewId
  )";
}

[[nodiscard]] std::string getVulkanVertexShaderSource(bool stereoRendering) {
  return getVertexShaderProlog(stereoRendering) + R"(
            layout (set = 1, binding = 1, std140) uniform PerFrame {
              mat4 modelMatrix;
              mat4 viewProjectionMatrix[2];
              float scaleZ;
              int viewId;
            } perFrame;

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 uvw_in;
            layout(location = 0) out vec3 uvw;
            layout(location = 1) out vec3 color;

            void main() {
              mat4 mvpMatrix = perFrame.viewProjectionMatrix[VIEW_ID] * perFrame.modelMatrix;
              gl_Position = mvpMatrix * vec4(position, 1.0);
              uvw = vec3(uvw_in.x, uvw_in.y, (uvw_in.z - 0.5) * perFrame.scaleZ + 0.5);
              color = vec3(1.0, 1.0, 0.0);
            })";
}

[[nodiscard]] std::unique_ptr<IShaderStages> getShaderStagesForBackend(
    IDevice& device,
    const iglu::ShaderCross& shaderCross,
    bool stereoRendering) noexcept {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (device.getBackendType()) {
  case BackendType::Metal:
    IGL_DEBUG_ABORT("Metal is not supported");
    return nullptr;
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getVulkanVertexShaderSource(stereoRendering).c_str(),
        "main",
        "",
        getVulkanFragmentShaderSource(),
        "main",
        "",
        nullptr);
  case BackendType::OpenGL: {
    Result res;
    const auto vs = shaderCross.crossCompileFromVulkanSource(
        getVulkanVertexShaderSource(stereoRendering).c_str(), ShaderStage::Vertex, &res);
    IGL_DEBUG_ASSERT(res.isOk(), res.message.c_str());

    const auto fs = shaderCross.crossCompileFromVulkanSource(
        getVulkanFragmentShaderSource(), ShaderStage::Fragment, &res);
    IGL_DEBUG_ASSERT(res.isOk(), res.message.c_str());

    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        vs.c_str(),
        shaderCross.entryPointName(ShaderStage::Vertex),
        "",
        fs.c_str(),
        shaderCross.entryPointName(ShaderStage::Fragment),
        "",
        nullptr);
  }
  default:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  }
}

[[nodiscard]] bool isDeviceCompatible(IDevice& device) noexcept {
  return device.hasFeature(DeviceFeatures::Multiview);
}

[[nodiscard]] glm::mat4 perspectiveAsymmetricFovRH(const Fov& fov,
                                                   float nearZ,
                                                   float farZ) noexcept {
  glm::mat4 mat;

  const float tanLeft = std::tan(fov.angleLeft);
  const float tanRight = std::tan(fov.angleRight);
  const float tanDown = std::tan(fov.angleDown);
  const float tanUp = std::tan(fov.angleUp);

  const float tanWidth = tanRight - tanLeft;
  const float tanHeight = tanUp - tanDown;

  mat[0][0] = 2.0f / tanWidth;
  mat[1][0] = 0.0f;
  mat[2][0] = (tanRight + tanLeft) / tanWidth;
  mat[3][0] = 0.0f;

  mat[0][1] = 0.0f;
  mat[1][1] = 2.0f / tanHeight;
  mat[2][1] = (tanUp + tanDown) / tanHeight;
  mat[3][1] = 0.0f;

  mat[0][2] = 0.0f;
  mat[1][2] = 0.0f;
  mat[2][2] = -(farZ + nearZ) / (farZ - nearZ);
  mat[3][2] = -2.0f * farZ * nearZ / (farZ - nearZ);

  mat[0][3] = 0.0f;
  mat[1][3] = 0.0f;
  mat[2][3] = -1.0f;
  mat[3][3] = 0.0f;

  return mat;
}
} // namespace

void HelloOpenXRSession::createSamplerAndTextures(const IDevice& device) {
  // Sampler & Texture
  samp0_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
          .addressModeU = SamplerAddressMode::MirrorRepeat,
          .addressModeV = SamplerAddressMode::MirrorRepeat,
          .addressModeW = SamplerAddressMode::MirrorRepeat,
      },
      nullptr);

  tex0_ = getPlatform().loadTexture("macbeth.png");
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void HelloOpenXRSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }
  // Vertex buffer, Index buffer and Vertex Input
  const BufferDesc vb0Desc{.type = BufferDesc::BufferTypeBits::Vertex,
                           .data = kVertexData0,
                           .length = sizeof(kVertexData0)};
  vb0_ = device.createBuffer(vb0Desc, nullptr);
  const BufferDesc ibDesc{
      .type = BufferDesc::BufferTypeBits::Index, .data = kIndexData, .length = sizeof(kIndexData)};
  ib0_ = device.createBuffer(ibDesc, nullptr);

  const VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes =
          {
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float3,
               .offset = offsetof(VertexPosUvw, position),
               .name = "position",
               .location = 0},
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float3,
               .offset = offsetof(VertexPosUvw, uvw),
               .name = "uvw_in",
               .location = 1},
          },
      .numInputBindings = 1,
      .inputBindings = {{.stride = sizeof(VertexPosUvw)}},
  };
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);

  const bool stereoRendering = shellParams().viewParams.size() > 1;

  createSamplerAndTextures(device);
  const iglu::ShaderCross shaderCross(device);
  shaderStages_ = getShaderStagesForBackend(device, shaderCross, stereoRendering);

  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = device.createCommandQueue({}, nullptr);

  // Set up vertex uniform data
  ub_.scaleZ = 1.0f;

  renderPass_ = {
      .colorAttachments =
          {
              {
                  .loadAction = LoadAction::Clear,
                  .storeAction = StoreAction::Store,
#if defined(IGL_OPENXR_MR_MODE)
                  .clearColor = {0.0, 0.0, 1.0, 0.0f},
#else
                  .clearColor = {0.0, 0.0, 1.0, 1.0f},
#endif
              },
          },
      .depthAttachment = {.loadAction = LoadAction::Clear, .clearDepth = 1.0},
  };
}

void HelloOpenXRSession::updateUniformBlock() {
  // rotating animation
  static float angle = 0.0f, scaleZ = 1.0f, ss = 0.005f;
  angle += 0.005f;
  scaleZ += ss;
  scaleZ = scaleZ < 0.0f ? 0.0f : (scaleZ > 1.0 ? 1.0f : scaleZ);
  if (scaleZ <= 0.05f || scaleZ >= 1.0f) {
    ss *= -1.0f;
  }

  const glm::mat4 rotMat = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 1.0f, 0.0f)) *
                           glm::rotate(glm::mat4(1.0f), -0.2f, glm::vec3(1.0f, 0.0f, 0.0f));
  ub_.modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.f, -8.0f)) * rotMat *
                    glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, scaleZ));
  for (size_t i = 0; i < std::min(shellParams().viewParams.size(), static_cast<size_t>(2)); ++i) {
    const auto viewIndex = shellParams().viewParams[i].viewIndex;

    ub_.viewProjectionMatrix[viewIndex] =
        perspectiveAsymmetricFovRH(shellParams().viewParams[i].fov, 0.1f, 100.0f) *
        shellParams().viewParams[i].viewMatrix;
    ub_.viewId = viewIndex;
  }

  ub_.scaleZ = scaleZ;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void HelloOpenXRSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  updateUniformBlock();

  IGL_DEBUG_ASSERT(!shellParams().viewParams.empty());
  const auto viewIndex = shellParams().viewParams[0].viewIndex;

  Result ret;
  if (framebuffer_[viewIndex] == nullptr) {
    const FramebufferMode mode = surfaceTextures.color->getNumLayers() > 1 ? FramebufferMode::Stereo
                                                                           : FramebufferMode::Mono;

    framebuffer_[viewIndex] = getPlatform().getDevice().createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color}},
            .depthAttachment = {.texture = surfaceTextures.depth},
            .mode = mode,
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_[viewIndex] != nullptr);
  } else {
    framebuffer_[viewIndex]->updateDrawable(surfaceTextures.color);
  }

  constexpr uint32_t textureUnit = 0;
  if (pipelineState_ == nullptr) {
    // Graphics pipeline: state batch that fully configures GPU for rendering

    pipelineState_ = getPlatform().getDevice().createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInput0_,
            .shaderStages = shaderStages_,
            .targetDesc =
                {
                    .colorAttachments =
                        {
                            {
                                .textureFormat = framebuffer_[viewIndex]
                                                     ->getColorAttachment(0)
                                                     ->getProperties()
                                                     .format,
                            },
                        },
                    .depthAttachmentFormat =
                        framebuffer_[viewIndex]->getDepthAttachment()->getProperties().format,
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::CounterClockwise,
            .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputImage")}},
        },
        nullptr);
  }

  // Command buffers (1-N per thread): create, submit and forget
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_[viewIndex]);
  commands->pushDebugGroupLabel("HelloOpenXRSession Commands", Color(0.0f, 1.0f, 0.0f));

  commands->bindVertexBuffer(0, *vb0_);

  const iglu::ManagedUniformBufferInfo info = {
      .index = 1,
      .length = sizeof(UniformBlock),
      .uniforms =
          {
              {
                  .name = "modelMatrix",
                  .location = -1,
                  .type = UniformType::Mat4x4,
                  .numElements = 1,
                  .offset = offsetof(UniformBlock, modelMatrix),
                  .elementStride = 0,
              },
              {
                  .name = "viewProjectionMatrix",
                  .location = -1,
                  .type = UniformType::Mat4x4,
                  .numElements = 2,
                  .offset = offsetof(UniformBlock, viewProjectionMatrix),
                  .elementStride = sizeof(glm::mat4),
              },
              {
                  .name = "scaleZ",
                  .location = -1,
                  .type = UniformType::Float,
                  .numElements = 1,
                  .offset = offsetof(UniformBlock, scaleZ),
                  .elementStride = 0,
              },
              {
                  .name = "viewId",
                  .location = -1,
                  .type = UniformType::Int,
                  .numElements = 1,
                  .offset = offsetof(UniformBlock, viewId),
                  .elementStride = 0,
              },
          },
  };

  const auto ubo = std::make_shared<iglu::ShaderCrossUniformBuffer>(device, "perFrame", info);
  IGL_DEBUG_ASSERT(ubo->result.isOk());
  *static_cast<UniformBlock*>(ubo->getData()) = ub_;

  ubo->bind(device, *pipelineState_, *commands);

  commands->bindTexture(textureUnit, BindTarget::kFragment, tex0_.get());
  commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());

  commands->bindRenderPipelineState(pipelineState_);

  commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
  commands->drawIndexed(static_cast<size_t>(3u * 6u * 2u));

  commands->popDebugGroupLabel();
  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(framebuffer_[viewIndex]->getColorAttachment(0));
  }

  commandQueue_->submit(*buffer); // Guarantees ordering between command buffers
}

} // namespace igl::shell
