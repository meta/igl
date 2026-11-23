/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/DrawInstancedSession.h>

#include <shell/shared/renderSession/ShellParams.h>
#include <igl/RenderCommandEncoder.h>
#if IGL_BACKEND_OPENGL
#include <igl/opengl/Device.h>
#endif
namespace igl::shell {

namespace {

[[maybe_unused]] void stringReplaceAll(std::string& s,
                                       const std::string& searchString,
                                       const std::string& replaceString) {
  size_t pos = 0;
  while ((pos = s.find(searchString, pos)) != std::string::npos) {
    s.replace(pos, searchString.length(), replaceString);
  }
}

const char* getMetalShaderSource() {
  return R"(
          #include <metal_stdlib>
          #include <simd/simd.h>
          using namespace metal;

          constant float2 pos[6] = {
            float2(-0.05f,  0.05f),
            float2( 0.05f, -0.05f),
            float2( -0.05f, -0.05f),
            float2(-0.05f,  0.05f),
            float2(0.05f, -0.05f),
            float2(0.05f,  0.05f)
          };
          constant float3 col[6] = {
            float3(1.0, 0.0, 0.0),
            float3(0.0, 1.0, 0.0),
            float3(0.0, 0.0, 1.0),
            float3(1.0, 0.0, 0.0),
            float3(0.0, 1.0, 0.0),
            float3(0.0, 0.0, 1.0)
          };

         struct VertexIn{
            float2 offset [[attribute(0)]];
         };

          struct VertexOut {
            float4 position [[position]];
            float3 uvw;
          };

          vertex VertexOut vertexShader(uint vid [[vertex_id]], constant VertexIn * vertices [[buffer(1)]],
                                        VertexIn in [[stage_in]]) {
            VertexOut out;
            out.position = float4(pos[vid] + in.offset, 0.0, 1.0);
            out.uvw = col[vid];
            return out;
           }

           fragment float4 fragmentShader(
                 VertexOut in[[stage_in]]) {

             float4 tex = float4(in.uvw,1.0);
             return tex;
           }
        )";
}

const char* getVulkanVertexShaderSource() {
  return R"(#version 460
layout (location=0) in vec2 offset;
layout (location=0) out vec3 color;
const vec2 pos[6] = vec2[6](
    vec2(-0.05f,  0.05f),
    vec2( 0.05f, -0.05f),
    vec2( -0.05f, -0.05f),
    vec2(-0.05f,  0.05f),
    vec2(0.05f, -0.05f),
    vec2(0.05f,  0.05f)
);
const vec3 col[6] = vec3[6](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);
void main() {
	gl_Position = vec4(pos[gl_VertexIndex] + offset , 0.0, 1.0);
	color = col[gl_VertexIndex];
}
)";
}

const char* getVulkanFragmentShaderSource() {
  return R"(#version 460
precision mediump float;
precision highp int;
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor;
void main() {
	out_FragColor = vec4(color, 1.0);
}
)";
}

const char* getWgslShaderSource() {
  return R"(
var<private> pos : array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-0.05, 0.05), vec2f(0.05, -0.05), vec2f(-0.05, -0.05),
  vec2f(-0.05, 0.05), vec2f(0.05, -0.05), vec2f(0.05, 0.05));
var<private> col : array<vec3f, 6> = array<vec3f, 6>(
  vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0),
  vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec3f,
};

@vertex
fn vertexShader(@builtin(vertex_index) vid : u32, @location(0) offset : vec2f) -> VertexOut {
  return VertexOut(vec4f(pos[vid] + offset, 0.0, 1.0), col[vid]);
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  return vec4f(v.color, 1.0);
}
)";
}

std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  switch (device.getBackendType()) {
  case BackendType::Invalid:
  case BackendType::Custom:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getVulkanVertexShaderSource(),
                                                           "main",
                                                           "",
                                                           getVulkanFragmentShaderSource(),
                                                           "main",
                                                           "",
                                                           nullptr);
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getMetalShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::OpenGL: {
#if IGL_BACKEND_OPENGL
    auto glVersion =
        static_cast<opengl::Device&>(device).getContext().deviceFeatures().getGLVersion();
    if (glVersion > opengl::GLVersion::v2_1) {
      auto usesOpenGLES = opengl::DeviceFeatureSet::usesOpenGLES();
      std::string codeVS(getVulkanVertexShaderSource());
      stringReplaceAll(codeVS, "gl_VertexIndex", "gl_VertexID");
      stringReplaceAll(codeVS, "460", usesOpenGLES ? "300 es" : "410");

      std::string codeFS(getVulkanFragmentShaderSource());
      stringReplaceAll(codeFS, "460", usesOpenGLES ? "300 es" : "410");

      if (usesOpenGLES) {
        stringReplaceAll(codeVS, "layout (location=0) out", "out");
        stringReplaceAll(codeFS, "layout (location=0) out", "out");
        stringReplaceAll(codeFS, "layout (location=0) in", "in");
      }

      return igl::ShaderStagesCreator::fromModuleStringInput(
          device, codeVS.c_str(), "main", "", codeFS.c_str(), "main", "", nullptr);
    } else {
      IGL_LOG_INFO("DrawInstancedSession: needs OpenGL 3.0 / OpenGL ES 3.0; skipping\n");
      return nullptr;
    }
#else
    return nullptr;
#endif // IGL_BACKEND_OPENGL
  }
  case BackendType::D3D12: {
    // D3D12 instanced drawing shader
    // Note: D3D12 clip space Y is flipped vs Vulkan, so we negate Y to match Vulkan behavior
    static const char* kVS = R"(
      static const float2 pos[6] = {
        float2(-0.05f,  0.05f), float2( 0.05f, -0.05f), float2(-0.05f, -0.05f),
        float2(-0.05f,  0.05f), float2( 0.05f, -0.05f), float2( 0.05f,  0.05f)
      };
      static const float3 col[6] = {
        float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0)
      };
      struct VSIn { float2 offset : TEXCOORD0; };
      struct VSOut { float4 position : SV_POSITION; float3 color : TEXCOORD0; };
      VSOut main(VSIn v, uint vid : SV_VertexID) {
        VSOut o;
        float2 p = pos[vid] + v.offset;
        o.position = float4(p.x, -p.y, 0.0, 1.0);  // Flip Y for D3D12 clip space
        o.color = col[vid];
        return o;
      }
    )";
    static const char* kPS = R"(
      struct PSIn { float4 position : SV_POSITION; float3 color : TEXCOORD0; };
      float4 main(PSIn i) : SV_TARGET { return float4(i.color, 1.0); }
    )";
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, kVS, "main", "", kPS, "main", "", nullptr);
  }
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
void DrawInstancedSession::initialize() noexcept {
  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = getPlatform().getDevice().createCommandQueue({}, nullptr);

  renderPass_.colorAttachments = {{
      .loadAction = LoadAction::Clear,
      .storeAction = StoreAction::Store,
      .clearColor = getPreferredClearColor(),
  }};
  renderPass_.depthAttachment = {.loadAction = LoadAction::DontCare};

  // Create Index Buffer
  const int16_t indexes[6] = {0, 1, 2, 3, 4, 5};
  indexBuffer_ = getPlatform().getDevice().createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Index, .data = &indexes, .length = sizeof(indexes)},
      nullptr);
  IGL_DEBUG_ASSERT(indexBuffer_);

  shaderStages_ = getShaderStagesForBackend(getPlatform().getDevice());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void DrawInstancedSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  const auto dimensions = surfaceTextures.color->getDimensions();
  framebuffer_ = getPlatform().getDevice().createFramebuffer(
      FramebufferDesc{.colorAttachments = {{.texture = surfaceTextures.color}}}, nullptr);
  IGL_DEBUG_ASSERT(framebuffer_);

  if (!renderPipelineStateTriangle_ && shaderStages_) {
    const VertexInputStateDesc inputDesc = {
        .numAttributes = 1,
        .attributes =
            {
                {
                    .bufferIndex = 1,
                    .format = VertexAttributeFormat::Float2,
                    .offset = 0,
                    .name = "offset",
                    .location = 0,
                },
            },
        .numInputBindings = 1,
        .inputBindings =
            {
                {},
                {
                    .stride = sizeof(float) * 2,
                    .sampleFunction = VertexSampleFunction::Instance,
                },
            },
    };
    auto vertexInput0 = getPlatform().getDevice().createVertexInputState(inputDesc, nullptr);
    IGL_DEBUG_ASSERT(vertexInput0 != nullptr);

    const RenderPipelineDesc desc = {
        .vertexInputState = vertexInput0,
        .shaderStages = shaderStages_,
        .targetDesc =
            {
                .colorAttachments =
                    {
                        {
                            .textureFormat =
                                framebuffer_->getColorAttachment(0)
                                    ? framebuffer_->getColorAttachment(0)->getProperties().format
                                    : TextureFormat::Invalid,
                        },
                    },
                .depthAttachmentFormat =
                    framebuffer_->getDepthAttachment()
                        ? framebuffer_->getDepthAttachment()->getProperties().format
                        : TextureFormat::Invalid,
            },
    };
    renderPipelineStateTriangle_ = getPlatform().getDevice().createRenderPipeline(desc, nullptr);
    IGL_DEBUG_ASSERT(renderPipelineStateTriangle_);
  }

  if (!vertexBuffer_) {
    glm::vec2 translations[100];
    int index = 0;
    const float offset = 0.1f;
    for (int y = -10; y < 10; y += 2) {
      for (int x = -10; x < 10; x += 2) {
        glm::vec2 translation;
        translation.x = static_cast<float>(x) / 10.0f + offset;
        translation.y = static_cast<float>(y) / 10.0f + offset;
        translations[index++] = translation;
      }
    }

    vertexBuffer_ = getPlatform().getDevice().createBuffer(
        BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                   .data = translations,
                   .length = sizeof(glm::vec2) * 100},
        nullptr);
    IGL_DEBUG_ASSERT(vertexBuffer_);
  }

  framebuffer_->updateDrawable(surfaceTextures.color);

  // Command buffers (1-N per thread): create, submit and forget
  const std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer({}, nullptr);

  const Viewport viewport = {.x = 0.0f,
                             .y = 0.0f,
                             .width = static_cast<float>(dimensions.width),
                             .height = static_cast<float>(dimensions.height),
                             .minDepth = 0.0f,
                             .maxDepth = +1.0f};
  const ScissorRect scissor = {.x = 0,
                               .y = 0,
                               .width = static_cast<uint32_t>(dimensions.width),
                               .height = static_cast<uint32_t>(dimensions.height)};

  // This will clear the framebuffer
  const auto commands = buffer->createRenderCommandEncoder(renderPass_, framebuffer_);

  if (renderPipelineStateTriangle_) {
    commands->bindRenderPipelineState(renderPipelineStateTriangle_);
    commands->bindViewport(viewport);
    commands->bindScissorRect(scissor);
    commands->pushDebugGroupLabel("Render Triangle", Color(1, 0, 0));
    commands->bindVertexBuffer(1, *vertexBuffer_);
    commands->bindIndexBuffer(*indexBuffer_, IndexFormat::UInt16);
    commands->drawIndexed(6, 100);
    commands->popDebugGroupLabel();
  }
  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(surfaceTextures.color);
  }

  commandQueue_->submit(*buffer);
  RenderSession::update(surfaceTextures);
}

} // namespace igl::shell
