/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/MeshShaderTriangleSession.h>

#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {

namespace {

std::string getMetalTaskShaderSource() {
  return R"(
using namespace metal;

[[object]]
void taskMain(mesh_grid_properties meshGridProperties) {
  meshGridProperties.set_threadgroups_per_grid(uint3(1, 1, 1));
}
)";
}

std::string getMetalMeshShaderSource() {
  return R"(
using namespace metal;

struct VertexOut{
  float4 position [[position]];
  float4 color [[user(locn0)]];
};

struct UniformBlock {
  float4x4 mvpMatrix;
};

using TriangleMeshType = metal::mesh<VertexOut, void, 64, 64, metal::topology::triangle>;

constant float4 vertexData[3] = {{-0.6f, -0.4f, 0.0, 1.0}, {0.6f, -0.4f, 0.0, 1.0}, {0.0f, 0.6f, 0.0, 1.0}};
constant float4 colorData[3]  = {{1.0, 0.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}};

[[mesh]]
void meshMain(TriangleMeshType output, constant UniformBlock &vUniform[[buffer(1)]]) {
  output.set_primitive_count(1);

  for (int i = 0; i != 3; ++i){
    VertexOut v;
    v.position = vUniform.mvpMatrix * vertexData[i];
    v.color = colorData[i];
    output.set_vertex(i, v);
  }

  output.set_index(0, 0);
  output.set_index(1, 1);
  output.set_index(2, 2);
}
)";
}

std::string getMetalFragmentShaderSource() {
  return R"(
using namespace metal;

struct FS_IN{
  float4 color [[user(locn0)]];
};

fragment float4 fragmentMain(FS_IN in [[stage_in]]) {
  return in.color;
}
)";
}

std::string getVulkanTaskShaderSource() {
  return R"(
#version 460
#extension GL_EXT_mesh_shader : enable

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main(){
  EmitMeshTasksEXT(1,1,1);
}
)";
}

std::string getVulkanMeshShaderSource() {
  return R"(
#version 460
#extension GL_EXT_mesh_shader : enable

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 1, binding = 1, std140) uniform UniformBlock {
  mat4 mvpMatrix;
};

layout(location = 0) out PerVertexData {vec4 color; } v_out[];

layout(triangles, max_vertices = 3, max_primitives = 1) out;

const vec4 vertexData[3] = {{-0.6f, -0.4f, 0.0, 1.0}, {0.6f, -0.4f, 0.0, 1.0}, {0.0f, 0.6f, 0.0, 1.0}};
const vec4 colorData[3]  = {{1.0, 0.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}};

void main(){
  SetMeshOutputsEXT(3, 1);

  for (int i = 0; i != 3; ++i) {
    gl_MeshVerticesEXT[i].gl_Position = mvpMatrix * vertexData[i];
    v_out[i].color = colorData[i];
  }

  gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
)";
}

std::string getVulkanFragmentShaderSource() {
  return R"(
#version 460

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 out_FragColor;

void main() {
  out_FragColor = color;
}
)";
}

std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  std::string taskShader, meshShader, fragmentShader;
  std::string taskShaderEntryPoint, meshShaderEntryPoint, fragmentShaderEntryPoint;

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (device.getBackendType()) {
  case BackendType::Metal:
    taskShader = getMetalTaskShaderSource();
    meshShader = getMetalMeshShaderSource();
    fragmentShader = getMetalFragmentShaderSource();
    taskShaderEntryPoint = "taskMain";
    meshShaderEntryPoint = "meshMain";
    fragmentShaderEntryPoint = "fragmentMain";
    break;

  case BackendType::Vulkan:
    taskShader = getVulkanTaskShaderSource();
    meshShader = getVulkanMeshShaderSource();
    fragmentShader = getVulkanFragmentShaderSource();
    taskShaderEntryPoint = "main";
    meshShaderEntryPoint = "main";
    fragmentShaderEntryPoint = "main";
    break;

  default:
    break;
  }

  if (taskShader.empty() || meshShader.empty() || fragmentShader.empty()) {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    return nullptr;
  }

  auto taskModule = igl::ShaderModuleCreator::fromStringInput(
      device,
      taskShader.c_str(),
      {.stage = ShaderStage::Task, .entryPoint = taskShaderEntryPoint},
      "task shader",
      nullptr);
  auto meshModule = igl::ShaderModuleCreator::fromStringInput(
      device,
      meshShader.c_str(),
      {.stage = ShaderStage::Mesh, .entryPoint = meshShaderEntryPoint},
      "mesh shader",
      nullptr);
  auto fragmentModule = igl::ShaderModuleCreator::fromStringInput(
      device,
      fragmentShader.c_str(),
      {.stage = ShaderStage::Fragment, .entryPoint = fragmentShaderEntryPoint},
      "fragment shader",
      nullptr);

  Result result;
  auto shaderStages = igl::ShaderStagesCreator::fromMeshRenderModules(
      device, taskModule, meshModule, fragmentModule, &result);

  IGL_DEBUG_ASSERT(shaderStages);

  return shaderStages;
}

// WebGPU has no mesh shaders: a vertex shader emits the same triangle from the vertex index.
const char* getWgslVertexFallbackShaderSource() {
  return R"(
struct UniformBlock {
  mvpMatrix : mat4x4f,
};

@group(1) @binding(1) var<uniform> vUniform : UniformBlock;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
};

@vertex
fn vertexMain(@builtin(vertex_index) i : u32) -> VertexOut {
  var vertexData = array<vec4f, 3>(
      vec4f(-0.6, -0.4, 0.0, 1.0), vec4f(0.6, -0.4, 0.0, 1.0), vec4f(0.0, 0.6, 0.0, 1.0));
  var colorData = array<vec4f, 3>(
      vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0, 1.0, 0.0, 1.0), vec4f(0.0, 0.0, 1.0, 1.0));
  return VertexOut(vUniform.mvpMatrix * vertexData[i], colorData[i]);
}

@fragment
fn fragmentMain(v : VertexOut) -> @location(0) vec4f {
  return v.color;
}
)";
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
void MeshShaderTriangleSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  useMeshShaders_ = device.hasFeature(DeviceFeatures::MeshShaders);
  if (useMeshShaders_) {
    shaderStages_ = getShaderStagesForBackend(device);
  } else if (device.getShaderVersion().family == ShaderFamily::Wgsl) {
    shaderStages_ = ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslVertexFallbackShaderSource(), "vertexMain", "fragmentMain", "", nullptr);
  } else {
    IGL_LOG_INFO("MeshShaderTriangleSession: mesh shaders are not supported; only clearing\n");
  }

  const BufferDesc uboDesc{
      .type = igl::BufferDesc::BufferTypeBits::Uniform,
      .length = sizeof(glm::mat4),
      .storage = ResourceStorage::Shared,
  };
  ubo_ = device.createBuffer(uboDesc, nullptr);
  IGL_DEBUG_ASSERT(ubo_ != nullptr);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  renderPass_.colorAttachments = {{
      .loadAction = LoadAction::Clear,
      .storeAction = StoreAction::Store,
      .clearColor = getPreferredClearColor(),
  }};
  renderPass_.depthAttachment = {
      .loadAction = LoadAction::Clear,
      .clearDepth = 1.0,
  };
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void MeshShaderTriangleSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  Result ret;
  if (!framebuffer_) {
    const FramebufferDesc framebufferDesc = {
        .colorAttachments = {{.texture = surfaceTextures.color}},
        .depthAttachment = {.texture = surfaceTextures.depth},
        .stencilAttachment =
            (surfaceTextures.depth && surfaceTextures.depth->getProperties().hasStencil())
                ? FramebufferDesc::AttachmentDesc{.texture = surfaceTextures.depth}
                : FramebufferDesc::AttachmentDesc{},
    };
    framebuffer_ = getPlatform().getDevice().createFramebuffer(framebufferDesc, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures);
  }

  // Graphics pipeline
  if (!pipelineState_ && shaderStages_) {
    const RenderPipelineDesc graphicsDesc = {
        .shaderStages = shaderStages_,
        .targetDesc =
            {
                .colorAttachments = {{.textureFormat =
                                          framebuffer_->getColorAttachment(0)->getFormat()}},
                .depthAttachmentFormat = framebuffer_->getDepthAttachment()->getFormat(),
                .stencilAttachmentFormat = framebuffer_->getStencilAttachment()
                                               ? framebuffer_->getStencilAttachment()->getFormat()
                                               : TextureFormat::Invalid,
            },
        .cullMode = CullMode::Disabled,
        .frontFaceWinding = WindingMode::Clockwise,
    };
    pipelineState_ = getPlatform().getDevice().createRenderPipeline(graphicsDesc, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  // Command Buffers
  const auto buffer = commandQueue_->createCommandBuffer({}, &ret);
  IGL_DEBUG_ASSERT(ret.isOk());
  IGL_DEBUG_ASSERT(buffer != nullptr);
  auto drawableSurface = framebuffer_->getColorAttachment(0);

  frameNum_ = (++frameNum_) % 360;
  const float angle = static_cast<float>(frameNum_) * M_PI / 180.0f;
  const glm::mat4 matrix = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f));
  if (pipelineState_ && ubo_) {
    ubo_->upload(&matrix, {sizeof(matrix)});
  }

  // Submit commands
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  if (pipelineState_) {
    commands->bindRenderPipelineState(pipelineState_);
    if (useMeshShaders_) {
      commands->bindBuffer(1, BindTarget::kMesh, ubo_.get());
      commands->drawMeshTasks({1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    } else {
      commands->bindBuffer(1, BindTarget::kVertex, ubo_.get());
      commands->draw(3);
    }
  }
  commands->endEncoding();

  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer);
  RenderSession::update(surfaceTextures);
}

} // namespace igl::shell
