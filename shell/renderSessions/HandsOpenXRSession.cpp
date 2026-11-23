/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/HandsOpenXRSession.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>
#include <IGLU/shaderCross/ShaderCross.h>
#include <IGLU/shaderCross/ShaderCrossUniformBuffer.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/Device.h>
#include <igl/DeviceFeatures.h>
#include <igl/NameHandle.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {

namespace {

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec4 weight;
  glm::vec4 joint;
};

[[nodiscard]] const char* getVulkanFragmentShaderSource() {
  return R"(#version 450
            precision highp float;
            layout(location = 0) in vec3 worldNormal;
            layout(location = 0) out vec4 fragmentColor;
            void main() {
              float att = max(dot(worldNormal, -normalize(vec3(-0.1, -1, 0))), 0.3);
              fragmentColor = vec4(att, att, att, 1.0);
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
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec4 weight;
            layout(location = 3) in vec4 joint;

            #define XR_HAND_JOINT_COUNT_EXT 26
            layout (set = 1, binding = 1, std140) uniform PerFrame {
              mat4 jointMatrices[XR_HAND_JOINT_COUNT_EXT];
              mat4 viewProjectionMatrix[2];
              int viewId;
            } perFrame;

            layout(location = 0) out vec3 worldNormal;

            void main() {
              mat4 world = perFrame.jointMatrices[int(joint.x)] * mat4(weight.x) +
                           perFrame.jointMatrices[int(joint.y)] * mat4(weight.y) +
                           perFrame.jointMatrices[int(joint.z)] * mat4(weight.z) +
                           perFrame.jointMatrices[int(joint.w)] * mat4(weight.w);
              worldNormal = (world * vec4(normal, 0.0)).xyz;
              vec4 worldPos = world * vec4(position, 1.0);
              gl_Position = perFrame.viewProjectionMatrix[VIEW_ID] * vec4(worldPos.xyz, 1.0);
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

[[nodiscard]] inline glm::mat4 poseToMat4(const Pose& pose) noexcept {
  return glm::translate(glm::mat4(1.0), glm::vec3(pose.position)) * glm::toMat4(pose.orientation);
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
void HandsOpenXRSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  if (shellParams().handMeshes[0].vertexCountOutput == 0 &&
      shellParams().handMeshes[1].vertexCountOutput == 0) {
    return;
  }

  const bool stereoRendering = shellParams().viewParams.size() > 1;

  const auto& handMeshes = shellParams().handMeshes;
  std::vector<Vertex> vertexData;
  std::vector<uint16_t> indices;
  vertexData.reserve(handMeshes[0].vertexCountOutput + handMeshes[1].vertexCountOutput);
  indices.reserve(handMeshes[0].indexCountOutput + handMeshes[1].indexCountOutput);
  for (uint8_t i = 0; i < 2; ++i) {
    handsDrawParams_[i].indexCount = handMeshes[i].indexCountOutput;
    handsDrawParams_[i].indexBufferOffset = indices.size() * sizeof(uint16_t);
    const uint16_t baseVertex = static_cast<uint16_t>(vertexData.size());

    for (size_t j = 0; j < handMeshes[i].vertexCountOutput; ++j) {
      Vertex v;
      v.position = handMeshes[i].vertexPositions[j];
      v.normal = handMeshes[i].vertexNormals[j];
      v.weight = handMeshes[i].vertexBlendWeights[j];
      v.joint = handMeshes[i].vertexBlendIndices[j];
      vertexData.push_back(v);
    }
    for (size_t j = 0; j < handMeshes[i].indexCountOutput; ++j) {
      indices.push_back(baseVertex + static_cast<uint16_t>(handMeshes[i].indices[j]));
    }

    for (size_t j = 0; j < handMeshes[i].jointBindPoses.size(); ++j) {
      jointInvBindMatrix_[i][j] = glm::inverse(poseToMat4(handMeshes[i].jointBindPoses[j]));
    }
  }

  const BufferDesc vb0Desc = BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                        .data = vertexData.data(),
                                        .length = sizeof(Vertex) * vertexData.size()};
  vb0_ = device.createBuffer(vb0Desc, nullptr);
  const BufferDesc ibDesc = BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                       .data = indices.data(),
                                       .length = sizeof(uint16_t) * indices.size()};
  ib0_ = device.createBuffer(ibDesc, nullptr);

  const VertexInputStateDesc inputDesc = {
      .numAttributes = 4,
      .attributes =
          {
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float3,
               .offset = offsetof(Vertex, position),
               .name = "position",
               .location = 0},
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float3,
               .offset = offsetof(Vertex, normal),
               .name = "normal",
               .location = 1},
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float4,
               .offset = offsetof(Vertex, weight),
               .name = "weight",
               .location = 2},
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float4,
               .offset = offsetof(Vertex, joint),
               .name = "joint",
               .location = 3},
          },
      .numInputBindings = 1,
      .inputBindings = {{.stride = sizeof(Vertex)}},
  };
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);

  const iglu::ShaderCross shaderCross(device);
  shaderStages_ = getShaderStagesForBackend(device, shaderCross, stereoRendering);

  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = device.createCommandQueue({}, nullptr);

  renderPass_.colorAttachments.resize(1);
  renderPass_.colorAttachments[0].loadAction = LoadAction::Clear;
  renderPass_.colorAttachments[0].storeAction = StoreAction::Store;
#if defined(IGL_OPENXR_MR_MODE)
  renderPass_.colorAttachments[0].clearColor = {0.0, 0.0, 1.0, 0.0f};
#else
  renderPass_.colorAttachments[0].clearColor = {0.0, 0.0, 1.0, 1.0f};
#endif
  renderPass_.depthAttachment.loadAction = LoadAction::Clear;
  renderPass_.depthAttachment.clearDepth = 1.0;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void HandsOpenXRSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  if (shellParams().handMeshes[0].vertexCountOutput == 0 &&
      shellParams().handMeshes[1].vertexCountOutput == 0) {
    return;
  }

  // Update uniforms.
  for (size_t i = 0; i < std::min(shellParams().viewParams.size(), static_cast<size_t>(2)); ++i) {
    const auto currentViewId = shellParams().viewParams[i].viewIndex;
    ub_.viewProjectionMatrix[currentViewId] =
        perspectiveAsymmetricFovRH(shellParams().viewParams[i].fov, 0.1f, 100.0f) *
        shellParams().viewParams[i].viewMatrix;
    ub_.viewId = currentViewId;
  }

  IGL_DEBUG_ASSERT(!shellParams().viewParams.empty());
  const auto viewIndex = shellParams().viewParams[0].viewIndex;

  Result ret;
  if (framebuffer_[viewIndex] == nullptr) {
    framebuffer_[viewIndex] = getPlatform().getDevice().createFramebuffer(
        FramebufferDesc{
            .colorAttachments = {{.texture = surfaceTextures.color}},
            .depthAttachment = {.texture = surfaceTextures.depth},
            .mode = surfaceTextures.color->getNumLayers() > 1 ? FramebufferMode::Stereo
                                                              : FramebufferMode::Mono,
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_[viewIndex] != nullptr);
  } else {
    framebuffer_[viewIndex]->updateDrawable(surfaceTextures.color);
  }

  if (pipelineState_ == nullptr) {
    const RenderPipelineDesc graphicsDesc = {
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
    };
    pipelineState_ = getPlatform().getDevice().createRenderPipeline(graphicsDesc, nullptr);
  }

  if (depthStencilState_ == nullptr) {
    const DepthStencilStateDesc depthStencilDesc{
        .compareFunction = CompareFunction::LessEqual,
        .isDepthWriteEnabled = true,
    };
    depthStencilState_ =
        getPlatform().getDevice().createDepthStencilState(depthStencilDesc, nullptr);
  }

  // Command buffers (1-N per thread): create, submit and forget
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_[viewIndex]);
  commands->pushDebugGroupLabel("HandsOpenXRSession Commands", Color(0.0f, 1.0f, 0.0f));

  commands->bindVertexBuffer(0, *vb0_);

  commands->bindRenderPipelineState(pipelineState_);
  commands->bindDepthStencilState(depthStencilState_);

  iglu::ManagedUniformBufferInfo info;
  info.index = 1;
  info.length = sizeof(UniformBlock);
  info.uniforms =
      std::vector<UniformDesc>{UniformDesc{
                                   .name = "jointMatrices",
                                   .location = -1,
                                   .type = UniformType::Mat4x4,
                                   .numElements = kMaxJoints,
                                   .offset = offsetof(UniformBlock, jointMatrices),
                                   .elementStride = sizeof(glm::mat4),
                               },
                               UniformDesc{.name = "viewProjectionMatrix",
                                           .location = -1,
                                           .type = UniformType::Mat4x4,
                                           .numElements = 2,
                                           .offset = offsetof(UniformBlock, viewProjectionMatrix),
                                           .elementStride = sizeof(glm::mat4)},
                               UniformDesc{
                                   .name = "viewId",
                                   .location = -1,
                                   .type = UniformType::Int,
                                   .numElements = 1,
                                   .offset = offsetof(UniformBlock, viewId),
                                   .elementStride = 0,
                               }};

  std::shared_ptr<iglu::ShaderCrossUniformBuffer> ubos[2];
  for (int i = 0; i < 2; ++i) {
    const auto& handTracking = shellParams().handTracking[i];
    IGL_DEBUG_ASSERT(handTracking.jointPose.size() <= kMaxJoints);
    for (size_t j = 0; j < handTracking.jointPose.size(); ++j) {
      ub_.jointMatrices[j] = poseToMat4(handTracking.jointPose[j]) * jointInvBindMatrix_[i][j];
    }

    ubos[i] = std::make_shared<iglu::ShaderCrossUniformBuffer>(device, "perFrame", info);
    IGL_DEBUG_ASSERT(ubos[i]->result.isOk());
    *static_cast<UniformBlock*>(ubos[i]->getData()) = ub_;

    ubos[i]->bind(device, *pipelineState_, *commands);

    commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16, handsDrawParams_[i].indexBufferOffset);
    commands->drawIndexed(handsDrawParams_[i].indexCount);
  }

  commands->popDebugGroupLabel();
  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(framebuffer_[viewIndex]->getColorAttachment(0));
  }

  commandQueue_->submit(*buffer); // Guarantees ordering between command buffers
}

} // namespace igl::shell
