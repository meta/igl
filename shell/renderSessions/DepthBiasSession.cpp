/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/DepthBiasSession.h>

#include <shell/shared/renderSession/ShellParams.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>

namespace igl::shell {

namespace {

// Shadow map resolution
constexpr uint32_t kShadowMapSize = 1024;

struct VertexPosNormal {
  iglu::simdtypes::float3 position;
  iglu::simdtypes::float3 normal;
};

// clang-format off
// Floor quad (two triangles) lying on the Y=-0.5 plane
// Triangle (floating above the floor) casting a shadow
VertexPosNormal vertexData[] = {
    // Floor quad vertices (indices 0-3)
    {.position = {-1.0f, -0.5f, -1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
    {.position = {1.0f, -0.5f, -1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
    {.position = {1.0f, -0.5f, 1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
    {.position = {-1.0f, -0.5f, 1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
    // Triangle vertices (indices 4-6), floating above the floor
    {.position = {-0.3f, 0.3f, 0.0f}, .normal = {0.0f, 0.0f, -1.0f}},
    {.position = {0.3f, 0.3f, 0.0f}, .normal = {0.0f, 0.0f, -1.0f}},
    {.position = {0.0f, 0.7f, 0.0f}, .normal = {0.0f, 0.0f, -1.0f}},
};

uint16_t indexData[] = {
    // Floor quad (two triangles)
    0, 1, 2,
    0, 2, 3,
    // Floating triangle
    4, 5, 6,
};
// clang-format on

constexpr size_t kFloorIndexCount = 6;
constexpr size_t kTriangleIndexCount = 3;
constexpr size_t kTotalIndexCount = kFloorIndexCount + kTriangleIndexCount;

// ===========================================================================
// Shadow pass shaders: depth-only, simple transform from light's perspective
// ===========================================================================

// A simple orthographic light view-projection applied to each vertex.
// The light looks down along -Y with a slight tilt.

std::string getShadowMetalShaderSource() {
  return R"(
              using namespace metal;

              typedef struct {
                float3 position [[attribute(0)]];
                float3 normal [[attribute(1)]];
              } VertexIn;

              typedef struct {
                float4 position [[position]];
              } VertexOut;

              vertex VertexOut shadowVertexShader(
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]]) {
                VertexOut out;
                // Simple orthographic projection from light's point of view
                // Light is above, looking down along -Y
                float3 p = vertices[vid].position;
                out.position = float4(p.x * 0.5, -p.z * 0.5, (p.y + 1.0) * 0.5, 1.0);
                return out;
              }

              fragment float4 shadowFragmentShader(
                  VertexOut IN [[stage_in]]) {
                  return float4(0.0);
              }
    )";
}

std::string getShadowOpenGLVertexShaderSource() {
  return R"(#version 100
                precision highp float;
                attribute vec3 position;
                attribute vec3 normal;

                void main() {
                  // Simple orthographic projection from light's point of view
                  gl_Position = vec4(position.x * 0.5, -position.z * 0.5,
                                     (position.y + 1.0) * 0.5, 1.0);
                })";
}

std::string getShadowOpenGLFragmentShaderSource() {
  return R"(#version 100
                precision highp float;

                void main() {
                  gl_FragColor = vec4(0.0);
                })";
}

std::string getShadowVulkanVertexShaderSource() {
  return R"(
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;

                void main() {
                  // Simple orthographic projection from light's point of view
                  gl_Position = vec4(position.x * 0.5, -position.z * 0.5,
                                     (position.y + 1.0) * 0.5, 1.0);
                }
                )";
}

std::string getShadowVulkanFragmentShaderSource() {
  return R"(
                layout(location = 0) out vec4 out_FragColor;

                void main() {
                  out_FragColor = vec4(0.0);
                }
                )";
}

std::string getShadowWgslShaderSource() {
  return R"(
@vertex
fn shadowVertexShader(@location(0) position : vec3f, @location(1) normal : vec3f)
    -> @builtin(position) vec4f {
  return vec4f(position.x * 0.5, -position.z * 0.5, (position.y + 1.0) * 0.5, 1.0);
}

@fragment
fn shadowFragmentShader() {}
)";
}

std::unique_ptr<IShaderStages> getShadowShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(device,
                                                            getShadowWgslShaderSource().c_str(),
                                                            "shadowVertexShader",
                                                            "shadowFragmentShader",
                                                            "",
                                                            nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getShadowVulkanVertexShaderSource().c_str(),
        "main",
        "",
        getShadowVulkanFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(device,
                                                            getShadowMetalShaderSource().c_str(),
                                                            "shadowVertexShader",
                                                            "shadowFragmentShader",
                                                            "",
                                                            nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getShadowOpenGLVertexShaderSource().c_str(),
        "main",
        "",
        getShadowOpenGLFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  case BackendType::D3D12: {
    static const char* kVS = R"(
      struct VSIn { float3 position : POSITION; float3 normal : NORMAL; };
      struct VSOut { float4 position : SV_POSITION; };
      VSOut main(VSIn v) {
        VSOut o;
        o.position = float4(v.position.x * 0.5, -v.position.z * 0.5,
                            (v.position.y + 1.0) * 0.5, 1.0);
        return o;
      }
    )";
    static const char* kPS = R"(
      struct PSIn { float4 position : SV_POSITION; };
      float4 main(PSIn i) : SV_TARGET { return float4(0.0, 0.0, 0.0, 0.0); }
    )";
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kVS, "main", "", kPS, "main", "", nullptr);
  }
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

// ===========================================================================
// Main pass shaders: render scene with basic shadow testing
// ===========================================================================

// The main pass transforms vertices for the camera view, and also computes
// shadow-map texture coordinates (the same transform used in the shadow pass).
// The fragment shader samples the shadow map to determine if a fragment is
// in shadow and darkens it accordingly.

std::string getMainMetalShaderSource() {
  return R"(
              using namespace metal;

              typedef struct {
                float3 position [[attribute(0)]];
                float3 normal [[attribute(1)]];
              } VertexIn;

              typedef struct {
                float4 position [[position]];
                float3 normal;
                float3 shadowCoord;
              } VertexOut;

              vertex VertexOut vertexShader(
                  uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]]) {
                VertexOut out;
                float3 p = vertices[vid].position;
                // Camera: simple perspective-like view
                out.position = float4(p.x * 0.8, p.y * 0.8 + 0.1, p.z * 0.1 + 0.5, 1.0);
                out.normal = vertices[vid].normal;
                // Shadow map coords: same transform as shadow pass, mapped to [0,1]
                out.shadowCoord = float3(p.x * 0.5 * 0.5 + 0.5,
                                         -p.z * 0.5 * 0.5 + 0.5,
                                         (p.y + 1.0) * 0.5);
                return out;
              }

              fragment float4 fragmentShader(
                  VertexOut IN [[stage_in]],
                  depth2d<float> shadowMap [[texture(0)]],
                  sampler shadowSampler [[sampler(0)]]) {
                // Base color: light gray for floor, blue for triangle
                float3 baseColor = (IN.normal.y > 0.5)
                                       ? float3(0.8, 0.8, 0.8)
                                       : float3(0.2, 0.5, 1.0);

                // Simple diffuse lighting from above
                float3 lightDir = normalize(float3(0.3, 1.0, 0.5));
                float ndotl = max(dot(IN.normal, lightDir), 0.3);

                // Shadow test
                float shadowDepth = shadowMap.sample(shadowSampler, IN.shadowCoord.xy);
                float shadow = (IN.shadowCoord.z > shadowDepth + 0.005) ? 0.4 : 1.0;

                return float4(baseColor * ndotl * shadow, 1.0);
              }
    )";
}

std::string getMainOpenGLVertexShaderSource() {
  return R"(#version 100
                precision highp float;
                attribute vec3 position;
                attribute vec3 normal;

                varying vec3 vNormal;
                varying vec3 vShadowCoord;

                void main() {
                  // Camera transform
                  gl_Position = vec4(position.x * 0.8, position.y * 0.8 + 0.1,
                                     position.z * 0.1 + 0.5, 1.0);
                  vNormal = normal;
                  // Shadow map coords
                  vShadowCoord = vec3(position.x * 0.5 * 0.5 + 0.5,
                                      -position.z * 0.5 * 0.5 + 0.5,
                                      (position.y + 1.0) * 0.5);
                })";
}

std::string getMainOpenGLFragmentShaderSource() {
  return R"(#version 100
                precision highp float;

                varying vec3 vNormal;
                varying vec3 vShadowCoord;

                uniform sampler2D shadowMap;

                void main() {
                  // Base color
                  vec3 baseColor = (vNormal.y > 0.5)
                                       ? vec3(0.8, 0.8, 0.8)
                                       : vec3(0.2, 0.5, 1.0);

                  // Simple diffuse lighting
                  vec3 lightDir = normalize(vec3(0.3, 1.0, 0.5));
                  float ndotl = max(dot(vNormal, lightDir), 0.3);

                  // Shadow test
                  float shadowDepth = texture2D(shadowMap, vShadowCoord.xy).r;
                  float shadow = (vShadowCoord.z > shadowDepth + 0.005) ? 0.4 : 1.0;

                  gl_FragColor = vec4(baseColor * ndotl * shadow, 1.0);
                })";
}

std::string getMainVulkanVertexShaderSource() {
  return R"(
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;
                layout(location = 0) out vec3 vNormal;
                layout(location = 1) out vec3 vShadowCoord;

                void main() {
                  // Camera transform
                  gl_Position = vec4(position.x * 0.8, position.y * 0.8 + 0.1,
                                     position.z * 0.1 + 0.5, 1.0);
                  vNormal = normal;
                  // Shadow map coords
                  vShadowCoord = vec3(position.x * 0.5 * 0.5 + 0.5,
                                      -position.z * 0.5 * 0.5 + 0.5,
                                      (position.y + 1.0) * 0.5);
                }
                )";
}

std::string getMainVulkanFragmentShaderSource() {
  return R"(
                layout(location = 0) in vec3 vNormal;
                layout(location = 1) in vec3 vShadowCoord;
                layout(location = 0) out vec4 out_FragColor;
                layout(set = 0, binding = 0) uniform sampler2D shadowMap;

                void main() {
                  // Base color
                  vec3 baseColor = (vNormal.y > 0.5)
                                       ? vec3(0.8, 0.8, 0.8)
                                       : vec3(0.2, 0.5, 1.0);

                  // Simple diffuse lighting
                  vec3 lightDir = normalize(vec3(0.3, 1.0, 0.5));
                  float ndotl = max(dot(vNormal, lightDir), 0.3);

                  // Shadow test
                  float shadowDepth = texture(shadowMap, vShadowCoord.xy).r;
                  float shadow = (vShadowCoord.z > shadowDepth + 0.005) ? 0.4 : 1.0;

                  out_FragColor = vec4(baseColor * ndotl * shadow, 1.0);
                }
                )";
}

std::string getMainWgslShaderSource() {
  return R"(
@group(0) @binding(0) var shadowMap : texture_depth_2d;
@group(0) @binding(1) var shadowSampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) normal : vec3f,
  @location(1) shadowCoord : vec3f,
};

@vertex
fn vertexShader(@location(0) position : vec3f, @location(1) normal : vec3f) -> VertexOut {
  var out : VertexOut;
  out.position = vec4f(position.x * 0.8, position.y * 0.8 + 0.1, position.z * 0.1 + 0.5, 1.0);
  out.normal = normal;
  out.shadowCoord = vec3f(position.x * 0.5 * 0.5 + 0.5,
                          -position.z * 0.5 * 0.5 + 0.5,
                          (position.y + 1.0) * 0.5);
  return out;
}

@fragment
fn fragmentShader(in : VertexOut) -> @location(0) vec4f {
  let baseColor = select(vec3f(0.2, 0.5, 1.0), vec3f(0.8, 0.8, 0.8), in.normal.y > 0.5);
  let lightDir = normalize(vec3f(0.3, 1.0, 0.5));
  let ndotl = max(dot(in.normal, lightDir), 0.3);
  let shadowDepth = textureSample(shadowMap, shadowSampler, in.shadowCoord.xy);
  let shadow = select(1.0, 0.4, in.shadowCoord.z > shadowDepth + 0.005);
  return vec4f(baseColor * ndotl * shadow, 1.0);
}
)";
}

std::unique_ptr<IShaderStages> getMainShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getMainWgslShaderSource().c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getMainVulkanVertexShaderSource().c_str(),
        "main",
        "",
        getMainVulkanFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getMainMetalShaderSource().c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::OpenGL:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getMainOpenGLVertexShaderSource().c_str(),
        "main",
        "",
        getMainOpenGLFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  case BackendType::D3D12: {
    static const char* kVS = R"(
      struct VSIn { float3 position : POSITION; float3 normal : NORMAL; };
      struct VSOut {
        float4 position : SV_POSITION;
        float3 normal : NORMAL;
        float3 shadowCoord : TEXCOORD0;
      };
      VSOut main(VSIn v) {
        VSOut o;
        o.position = float4(v.position.x * 0.8, v.position.y * 0.8 + 0.1,
                            v.position.z * 0.1 + 0.5, 1.0);
        o.normal = v.normal;
        o.shadowCoord = float3(v.position.x * 0.5 * 0.5 + 0.5,
                               -v.position.z * 0.5 * 0.5 + 0.5,
                               (v.position.y + 1.0) * 0.5);
        return o;
      }
    )";
    static const char* kPS = R"(
      Texture2D shadowMap : register(t0);
      SamplerState shadowSampler : register(s0);

      struct PSIn {
        float4 position : SV_POSITION;
        float3 normal : NORMAL;
        float3 shadowCoord : TEXCOORD0;
      };

      float4 main(PSIn i) : SV_TARGET {
        // Base color
        float3 baseColor = (i.normal.y > 0.5)
                               ? float3(0.8, 0.8, 0.8)
                               : float3(0.2, 0.5, 1.0);

        // Simple diffuse lighting
        float3 lightDir = normalize(float3(0.3, 1.0, 0.5));
        float ndotl = max(dot(i.normal, lightDir), 0.3);

        // Shadow test
        float shadowDepth = shadowMap.Sample(shadowSampler, i.shadowCoord.xy).r;
        float shadow = (i.shadowCoord.z > shadowDepth + 0.005) ? 0.4 : 1.0;

        return float4(baseColor * ndotl * shadow, 1.0);
      }
    )";
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kVS, "main", "", kPS, "main", "", nullptr);
  }
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

} // namespace

void DepthBiasSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();

  // Vertex & Index buffer
  vertexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                                 .data = vertexData,
                                                 .length = sizeof(vertexData)},
                                      nullptr);
  IGL_DEBUG_ASSERT(vertexBuffer_ != nullptr);
  indexBuffer_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                                .data = indexData,
                                                .length = sizeof(indexData)},
                                     nullptr);
  IGL_DEBUG_ASSERT(indexBuffer_ != nullptr);

  // Vertex input state: position (float3) + normal (float3)
  vertexInputState_ = device.createVertexInputState(
      VertexInputStateDesc{
          .numAttributes = 2,
          .attributes =
              {
                  VertexAttribute{.bufferIndex = 1,
                                  .format = VertexAttributeFormat::Float3,
                                  .offset = offsetof(VertexPosNormal, position),
                                  .name = "position",
                                  .location = 0},
                  VertexAttribute{.bufferIndex = 1,
                                  .format = VertexAttributeFormat::Float3,
                                  .offset = offsetof(VertexPosNormal, normal),
                                  .name = "normal",
                                  .location = 1},
              },
          .numInputBindings = 1,
          .inputBindings = {{}, {.stride = sizeof(VertexPosNormal)}},
      },
      nullptr);
  IGL_DEBUG_ASSERT(vertexInputState_ != nullptr);

  // Shader stages for both passes
  shadowShaderStages_ = getShadowShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shadowShaderStages_ != nullptr);

  shaderStages_ = getMainShaderStagesForBackend(device);
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);

  // Command queue
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  // Depth stencil state: depth testing and depth writes enabled
  {
    const DepthStencilStateDesc desc{
        .compareFunction = CompareFunction::Less,
        .isDepthWriteEnabled = true,
    };
    depthStencilState_ = device.createDepthStencilState(desc, nullptr);
    IGL_DEBUG_ASSERT(depthStencilState_ != nullptr);
  }

  // Sampler for the shadow map
  shadowSampler_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Nearest,
          .magFilter = SamplerMinMagFilter::Nearest,
          .addressModeU = SamplerAddressMode::Clamp,
          .addressModeV = SamplerAddressMode::Clamp,
          .debugName = "Shadow Sampler",
      },
      nullptr);
  IGL_DEBUG_ASSERT(shadowSampler_ != nullptr);

  // Shadow render pass descriptor: depth-only, no color attachment
  shadowRenderPass_ = {
      .depthAttachment =
          {
              .loadAction = LoadAction::Clear,
              .storeAction = StoreAction::Store,
              .clearDepth = 1.0,
          },
  };

  // Main render pass descriptor
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
void DepthBiasSession::update(SurfaceTextures textures) noexcept {
  // Per IGL guidelines, textures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!textures.color) {
    return;
  }
  Result ret;
  auto& device = getPlatform().getDevice();

  // Create the shadow map texture (depth-only, also sampled for shadow testing)
  if (shadowMap_ == nullptr) {
    TextureDesc shadowMapDesc = TextureDesc::new2D(TextureFormat::Z_UNorm24,
                                                   kShadowMapSize,
                                                   kShadowMapSize,
                                                   TextureDesc::TextureUsageBits::Attachment |
                                                       TextureDesc::TextureUsageBits::Sampled,
                                                   "Shadow Map");
    // Metal on the iOS simulator rejects depth textures in shared storage.
    shadowMapDesc.storage = ResourceStorage::Private;
    shadowMap_ = device.createTexture(shadowMapDesc, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(shadowMap_ != nullptr);
  }

  // Create shadow framebuffer (depth-only, no color attachment)
  if (shadowFramebuffer_ == nullptr) {
    shadowFramebuffer_ = device.createFramebuffer(
        FramebufferDesc{
            .depthAttachment = {.texture = shadowMap_},
        },
        &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(shadowFramebuffer_ != nullptr);
  }

  // Create or update main framebuffer
  if (framebuffer_ == nullptr) {
    framebuffer_ = device.createFramebuffer(
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

  // Shadow pipeline state (lazy, cached)
  if (shadowPipelineState_ == nullptr) {
    shadowPipelineState_ = device.createRenderPipeline(
        RenderPipelineDesc{
            .vertexInputState = vertexInputState_,
            .shaderStages = shadowShaderStages_,
            .targetDesc =
                {
                    .depthAttachmentFormat = shadowMap_->getFormat(),
                },
            .cullMode = CullMode::Back,
            .frontFaceWinding = WindingMode::CounterClockwise,
        },
        nullptr);
    IGL_DEBUG_ASSERT(shadowPipelineState_ != nullptr);
  }

  // Main pipeline state (lazy, cached)
  if (pipelineState_ == nullptr) {
    pipelineState_ = device.createRenderPipeline(
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
            .frontFaceWinding = WindingMode::CounterClockwise,
            .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE("shadowMap")}},
        },
        nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  // Create command buffer
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  auto drawableSurface = framebuffer_->getColorAttachment(0);

  // -----------------------------------------------------------------------
  // Pass 1: Render scene to shadow map (depth-only) with depth bias
  // -----------------------------------------------------------------------
  {
    const std::shared_ptr<IRenderCommandEncoder> commands =
        buffer->createRenderCommandEncoder(shadowRenderPass_, shadowFramebuffer_);
    IGL_DEBUG_ASSERT(commands != nullptr);
    if (commands) {
      commands->bindVertexBuffer(1, *vertexBuffer_);
      commands->bindRenderPipelineState(shadowPipelineState_);
      commands->bindDepthStencilState(depthStencilState_);
      commands->bindIndexBuffer(*indexBuffer_, IndexFormat::UInt16);

      // KEY API CALL: Apply depth bias to prevent shadow acne.
      // depthBias: constant offset added to fragment depth
      // slopeScale: scales the bias based on the polygon slope relative to the light
      // clamp: maximum absolute depth bias value (0 = unclamped)
      commands->setDepthBias(0.005f, 1.5f, 0.0f);

      commands->drawIndexed(kTotalIndexCount);
      commands->endEncoding();
    }
  }

  // -----------------------------------------------------------------------
  // Pass 2: Render scene with shadow testing
  // -----------------------------------------------------------------------
  {
    const std::shared_ptr<IRenderCommandEncoder> commands =
        buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
    IGL_DEBUG_ASSERT(commands != nullptr);
    if (commands) {
      commands->bindVertexBuffer(1, *vertexBuffer_);
      commands->bindRenderPipelineState(pipelineState_);
      commands->bindDepthStencilState(depthStencilState_);
      commands->bindIndexBuffer(*indexBuffer_, IndexFormat::UInt16);

      // Bind the shadow map texture for shadow testing in the fragment shader
      commands->bindTexture(0, BindTarget::kFragment, shadowMap_.get());
      commands->bindSamplerState(0, BindTarget::kFragment, shadowSampler_.get());

      commands->drawIndexed(kTotalIndexCount);
      commands->endEncoding();
    }
  }

  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(drawableSurface);
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer);
  RenderSession::update(textures);
}

} // namespace igl::shell
