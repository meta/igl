/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include "Session.h"

#include <IGLU/simple_renderer/Drawable.h>
#include <IGLU/simple_renderer/Material.h>
#include <igl/Macros.h>
#include <igl/ShaderCreator.h>

// D3D12 FXC precompiled shaders
#include "imgui_ps_d3d12_fxc.h"
#include "imgui_vs_d3d12_fxc.h"

namespace iglu::imgui {

/* internal renderer -- based on imgui_impl_metal.mm */

#define PLAIN_SHADER_STRINGIFY(...) #__VA_ARGS__
#define PLAIN_SHADER(...) PLAIN_SHADER_STRINGIFY(__VA_ARGS__)

namespace {

const char* metalShaderStr() {
  return PLAIN_SHADER(
      using namespace metal;

      struct Uniforms { float4x4 projectionMatrix; };

      struct VertexIn {
        float2 position [[attribute(0)]];
        float2 texCoords [[attribute(1)]];
        float4 color [[attribute(2)]];
      };

      struct VertexOut {
        float4 position [[position]];
        float2 texCoords;
        float4 color;
      };

      vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                                   constant Uniforms & uniforms [[buffer(1)]]) {
        VertexOut out;
        out.position = uniforms.projectionMatrix * float4(in.position, 0, 1);
        out.texCoords = in.texCoords;
        out.color = in.color;
        return out;
      }

      fragment half4 fragment_main(VertexOut in [[stage_in]],
                                   texture2d<half, access::sample> texture [[texture(0)]]) {
        constexpr sampler linearSampler(
            coord::normalized, min_filter::linear, mag_filter::linear, mip_filter::linear);
        half4 texColor = texture.sample(linearSampler, in.texCoords);
        return half4(in.color) * texColor;
      }

  );
}

std::string getOpenGLVertexShaderSource(igl::ShaderVersion shaderVersion) {
  std::string shader;
  if (shaderVersion.majorVersion > 1 || shaderVersion.minorVersion > 30 ||
      shaderVersion.family == igl::ShaderFamily::GlslEs) {
#if IGL_PLATFORM_MACOSX
    shader += "#version 100\n";
#endif
    shader += "precision mediump float;";
  }
  shader += PLAIN_SHADER(attribute vec2 position; attribute vec2 texCoords; attribute vec4 color;
                         uniform mat4 projectionMatrix;
                         varying vec2 Frag_UV;
                         varying vec4 Frag_Color;
                         void main() {
                           Frag_UV = texCoords;
                           Frag_Color = color;
                           gl_Position = projectionMatrix * vec4(position.xy, 0, 1);
                         });

  return shader;
}
const char* getVulkanVertexShaderSource() {
  return R"(
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoords;
layout(location = 2) in vec4 col;

layout (location = 0) out vec4 color;
layout (location = 1) out vec2 uv;

layout(push_constant) uniform PushConstants {
    mat4 proj;
} pc;

out gl_PerVertex { vec4 gl_Position; };

void main() {
    color = col;
    uv = texCoords;
    gl_Position = pc.proj * vec4(position.xy, 0, 1);
})";
}

std::string getOpenGLFragmentShaderSource(igl::ShaderVersion shaderVersion) {
  std::string shader;
  if (shaderVersion.majorVersion > 1 || shaderVersion.minorVersion > 30 ||
      shaderVersion.family == igl::ShaderFamily::GlslEs) {
#if IGL_PLATFORM_MACOSX
    shader += "#version 100\n";
#endif
    shader += "precision mediump float;";
  }
  shader +=
      PLAIN_SHADER(uniform sampler2D texture; varying vec2 Frag_UV; varying vec4 Frag_Color;
                   void main() { gl_FragColor = Frag_Color * texture2D(texture, Frag_UV.st); });
  return shader;
}
const char* getVulkanFragmentShaderSource() {
  return R"(
layout(location = 0) out vec4 fColor;
layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout (set = 0, binding = 0) uniform sampler2D uTex;

void main() {
  fColor = color * texture(uTex, uv);
})";
}

// The Vulkan shaders under the WebGPU bind convention; push constants are @group(3) @binding(0).
const char* getWgslVertexShaderSource() {
  return R"(
struct PushConstants {
  proj : mat4x4f,
}

@group(3) @binding(0) var<uniform> pc : PushConstants;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) uv : vec2f,
}

@vertex
fn main(@location(0) position : vec2f,
        @location(1) texCoords : vec2f,
        @location(2) col : vec4f) -> VertexOut {
  return VertexOut(pc.proj * vec4f(position, 0.0, 1.0), col, texCoords);
})";
}

const char* getWgslFragmentShaderSource() {
  return R"(
@group(0) @binding(0) var uTex : texture_2d<f32>;
@group(0) @binding(1) var uSampler : sampler;

@fragment
fn main(@location(0) color : vec4f, @location(1) uv : vec2f) -> @location(0) vec4f {
  return color * textureSample(uTex, uSampler, uv);
})";
}

// Note: D3D12 shader source functions are kept for reference but not used.
// The D3D12 backend uses pre-compiled binary shaders.
#if IGL_PLATFORM_WINDOWS
const char* getD3D12VertexShaderSource() {
  return R"(
cbuffer Uniforms : register(b0) {
  float4x4 projectionMatrix;
};

struct VSInput {
  float2 position : POSITION;
  float2 uv : TEXCOORD0;
  float4 color : COLOR;
};

struct PSInput {
  float4 position : SV_Position;
  float4 color : COLOR;
  float2 uv : TEXCOORD0;
};

PSInput main(VSInput input) {
  PSInput output;
  // Column-major multiplication to match the CPU-side matrix format
  // In HLSL: mul(vector, matrix) treats matrix as column-major
  output.position = mul(float4(input.position.xy, 0, 1), projectionMatrix);
  output.color = input.color;
  output.uv = input.uv;
  return output;
})";
}

const char* getD3D12FragmentShaderSource() {
  return R"(
struct PSInput {
  float4 position : SV_Position;
  float4 color : COLOR;
  float2 uv : TEXCOORD0;
};

Texture2D tex : register(t0);
SamplerState uSampler : register(s0);

float4 main(PSInput input) : SV_Target {
  return input.color * tex.Sample(uSampler, input.uv);
})";
}
#endif

std::unique_ptr<igl::IShaderStages> getShaderStagesForBackend(igl::IDevice& device) {
  igl::Result result;
  switch (device.getBackendType()) {
  case igl::BackendType::Invalid:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  case igl::BackendType::Vulkan: {
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getVulkanVertexShaderSource(),
                                                           "main",
                                                           "Shader Module: imgui::vertex",
                                                           getVulkanFragmentShaderSource(),
                                                           "main",
                                                           "Shader Module: imgui::fragment",
                                                           &result);
  }
  // @fb-only
    // @fb-only
    // @fb-only
  case igl::BackendType::Custom:
    IGL_DEBUG_ABORT("IGLSamples not set up for Custom");
    return nullptr;
  case igl::BackendType::WebGPU: {
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getWgslVertexShaderSource(),
                                                           "main",
                                                           "Shader Module: imgui::vertex",
                                                           getWgslFragmentShaderSource(),
                                                           "main",
                                                           "Shader Module: imgui::fragment",
                                                           &result);
  }
  case igl::BackendType::Metal: {
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, metalShaderStr(), "vertex_main", "fragment_main", "", &result);
  }
  case igl::BackendType::OpenGL: {
    auto shaderVersion = device.getShaderVersion();
    const std::string vertexStr = getOpenGLVertexShaderSource(shaderVersion);
    const std::string fragmentStr = getOpenGLFragmentShaderSource(shaderVersion);
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device, vertexStr.c_str(), "main", "", fragmentStr.c_str(), "main", "", &result);
  }
  case igl::BackendType::D3D12: {
    return igl::ShaderStagesCreator::fromModuleBinaryInput(device,
                                                           tmpImguiVsFxcCso,
                                                           tmpImguiVsFxcCsoLen,
                                                           "main",
                                                           "Shader Module: imgui::vertex (D3D12)",
                                                           tmpImguiPsFxcCso,
                                                           tmpImguiPsFxcCsoLen,
                                                           "main",
                                                           "Shader Module: imgui::fragment (D3D12)",
                                                           &result);
  }
  }
  IGL_UNREACHABLE_RETURN(nullptr)
}

} // namespace

namespace {
struct DrawableData {
  std::shared_ptr<vertexdata::VertexData> vertexData;
  std::shared_ptr<iglu::drawable::Drawable> drawable;

  DrawableData(igl::IDevice& device,
               const std::shared_ptr<igl::IVertexInputState>& inputState,
               const std::shared_ptr<material::Material>& material) {
    IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
    IGL_DEBUG_ASSERT(sizeof(ImDrawIdx) == 2,
                     "The constants below may not work with the ImGui data.");
    const size_t kMaxVertices = (1l << 16);
    const size_t kMaxVertexBufferSize = kMaxVertices * sizeof(ImDrawVert);
    const size_t kMaxIndexBufferSize = kMaxVertices * sizeof(ImDrawIdx);

    const igl::BufferDesc vbDesc{.type = igl::BufferDesc::BufferTypeBits::Vertex,
                                 .data = nullptr,
                                 .length = kMaxVertexBufferSize,
                                 .storage = igl::ResourceStorage::Shared,
                                 .hint = 0,
                                 .debugName = "vertex (" + material->name + ")"};
    const igl::BufferDesc ibDesc{.type = igl::BufferDesc::BufferTypeBits::Index,
                                 .data = nullptr,
                                 .length = kMaxIndexBufferSize,
                                 .storage = igl::ResourceStorage::Shared,
                                 .hint = 0,
                                 .debugName = "index (" + material->name + ")"};

    vertexdata::PrimitiveDesc primitiveDesc;
    primitiveDesc.numEntries = 0;

    // Per IGL Error Handling rule #24, every resource creation call must pass
    // a Result* and check it; passing nullptr silently swallows errors.
    igl::Result result;
    auto vb = device.createBuffer(vbDesc, &result);
    IGL_DEBUG_ASSERT(result.isOk(), "createBuffer(vertex) failed: %s", result.message.c_str());
    IGL_DEBUG_ASSERT(vb != nullptr);
    auto ib = device.createBuffer(ibDesc, &result);
    IGL_DEBUG_ASSERT(result.isOk(), "createBuffer(index) failed: %s", result.message.c_str());
    IGL_DEBUG_ASSERT(ib != nullptr);

    vertexData = std::make_shared<vertexdata::VertexData>(
        inputState,
        std::move(vb),
        std::move(ib),
        sizeof(ImDrawIdx) == sizeof(uint16_t) ? igl::IndexFormat::UInt16 : igl::IndexFormat::UInt32,
        primitiveDesc);

    drawable = std::make_shared<iglu::drawable::Drawable>(vertexData, material);
  }
};
} // namespace

class Session::Renderer {
 public:
  explicit Renderer(igl::IDevice& device);
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;

  void newFrame(const igl::FramebufferDesc& desc);
  void renderDrawData(igl::IDevice& device,
                      igl::IRenderCommandEncoder& cmdEncoder,
                      ImDrawData* drawData);

  bool dumpRenderDrawDataInfo{false};

 private:
  std::shared_ptr<igl::IVertexInputState> vertexInputState_;
  std::shared_ptr<material::Material> material_;
  std::vector<DrawableData> drawables_[3]; // list of drawables to be reused every 3 frames
  size_t nextBufferingIndex_ = 0;

  igl::RenderPipelineDesc renderPipelineDesc_;
  std::shared_ptr<igl::ITexture> fontTexture_;
  std::shared_ptr<igl::ISamplerState> linearSampler_;
};

Session::Renderer::Renderer(igl::IDevice& device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  ImGuiIO& io = ImGui::GetIO();
  io.BackendRendererName = "imgui_impl_igl";

  // Per IGL Error Handling rule #24, every resource creation call must pass a
  // Result* and check it; passing nullptr silently swallows errors.
  igl::Result result;
  linearSampler_ = device.createSamplerState(igl::SamplerStateDesc::newLinear(), &result);
  IGL_DEBUG_ASSERT(result.isOk(), "createSamplerState() failed: %s", result.message.c_str());
  IGL_DEBUG_ASSERT(linearSampler_ != nullptr);

  { // init fonts
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    igl::TextureDesc desc = igl::TextureDesc::new2D(igl::TextureFormat::RGBA_UNorm8,
                                                    width,
                                                    height,
                                                    igl::TextureDesc::TextureUsageBits::Sampled);
    desc.debugName = "IGLU/imgui/Session.cpp:Session::Renderer::_fontTexture";
    fontTexture_ = device.createTexture(desc, &result);
    IGL_DEBUG_ASSERT(result.isOk(), "createTexture() failed: %s", result.message.c_str());
    IGL_DEBUG_ASSERT(fontTexture_ != nullptr);
    if (!fontTexture_) {
      return;
    }
    fontTexture_->upload(igl::TextureRangeDesc::new2D(0, 0, width, height), pixels);

    io.Fonts->TexID = reinterpret_cast<ImTextureID>(fontTexture_.get());
  }

  {
    const igl::VertexInputStateDesc inputDesc = {
        .numAttributes = 3,
        .attributes = {{
                           .bufferIndex = 0,
                           .format = igl::VertexAttributeFormat::Float2,
                           .offset = offsetof(ImDrawVert, pos),
                           .name = "position",
                           .location = 0,
                       },
                       {
                           .bufferIndex = 0,
                           .format = igl::VertexAttributeFormat::Float2,
                           .offset = offsetof(ImDrawVert, uv),
                           .name = "texCoords",
                           .location = 1,
                       },
                       {
                           .bufferIndex = 0,
                           .format = igl::VertexAttributeFormat::UByte4Norm,
                           .offset = offsetof(ImDrawVert, col),
                           .name = "color",
                           .location = 2,
                       }},
        .numInputBindings = 1,
        .inputBindings = {{.stride = sizeof(ImDrawVert)}},
    };
    vertexInputState_ = device.createVertexInputState(inputDesc, &result);
    IGL_DEBUG_ASSERT(result.isOk(), "createVertexInputState() failed: %s", result.message.c_str());
    IGL_DEBUG_ASSERT(vertexInputState_ != nullptr);
  }

  {
    auto stages = getShaderStagesForBackend(device);
    auto program =
        std::make_shared<material::ShaderProgram>(device, std::move(stages), vertexInputState_);

    material_ = std::make_shared<material::Material>(device, "imgui");
    material_->setShaderProgram(device, program);
    material_->cullMode = igl::CullMode::Disabled;
    material_->blendMode = material::BlendMode::Translucent();

    // D3D12, Vulkan and WebGPU use direct slot binding, OpenGL/Metal use named binding
    const bool usesDirectBinding = (device.getBackendType() == igl::BackendType::Vulkan ||
                                    device.getBackendType() == igl::BackendType::D3D12 ||
                                    device.getBackendType() == igl::BackendType::WebGPU);
    if (!usesDirectBinding) {
      material_->shaderUniforms().setTexture("texture", fontTexture_.get(), linearSampler_);
    }
  }
}

Session::Renderer::~Renderer() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  const ImGuiIO& io = ImGui::GetIO();
  fontTexture_ = nullptr;
  io.Fonts->TexID = static_cast<ImTextureID>(0);
}

void Session::Renderer::newFrame(const igl::FramebufferDesc& desc) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(desc.colorAttachments[0].texture);
  renderPipelineDesc_.targetDesc.colorAttachments.resize(1);
  renderPipelineDesc_.targetDesc.colorAttachments[0].textureFormat =
      desc.colorAttachments[0].texture->getFormat();
  renderPipelineDesc_.targetDesc.depthAttachmentFormat =
      desc.depthAttachment.texture ? desc.depthAttachment.texture->getFormat()
                                   : igl::TextureFormat::Invalid;
  renderPipelineDesc_.targetDesc.stencilAttachmentFormat =
      desc.stencilAttachment.texture ? desc.stencilAttachment.texture->getFormat()
                                     : igl::TextureFormat::Invalid;
  renderPipelineDesc_.sampleCount = desc.colorAttachments[0].texture->getSamples();
}

void Session::Renderer::renderDrawData(igl::IDevice& device,
                                       igl::IRenderCommandEncoder& cmdEncoder,
                                       ImDrawData* drawData) {
  IGL_PROFILER_FUNCTION();
  // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates !=
  // framebuffer coordinates)
  const int fbWidth = static_cast<int>(drawData->DisplaySize.x * drawData->FramebufferScale.x);
  const int fbHeight = static_cast<int>(drawData->DisplaySize.y * drawData->FramebufferScale.y);

  if (dumpRenderDrawDataInfo) {
    // throttle or turn off due to swamping other messages
    IGL_LOG_INFO(
        "ImGui renderDrawData: DisplaySize=(%.1f,%.1f), FramebufferScale=(%.1f,%.1f), fb=(%d,%d), "
        "CmdLists=%d, TotalVtx=%d, TotalIdx=%d\n",
        drawData->DisplaySize.x,
        drawData->DisplaySize.y,
        drawData->FramebufferScale.x,
        drawData->FramebufferScale.y,
        fbWidth,
        fbHeight,
        drawData->CmdListsCount,
        drawData->TotalVtxCount,
        drawData->TotalIdxCount);
  }

  if (fbWidth <= 0 || fbHeight <= 0 || drawData->CmdListsCount == 0) {
    IGL_LOG_INFO("ImGui renderDrawData: Early return (invalid dimensions or no command lists)\n");
    return;
  }

  cmdEncoder.pushDebugGroupLabel("ImGui Rendering", igl::Color(0, 1, 0));

  const igl::Viewport viewport = {
      /*.x = */ .x = 0.0,
      /*.y = */ .y = 0.0,
      /*.width = */ .width = (drawData->DisplaySize.x * drawData->FramebufferScale.x),
      /*.height = */ .height = (drawData->DisplaySize.y * drawData->FramebufferScale.y),
  };
  cmdEncoder.bindViewport(viewport);

  using namespace iglu::simdtypes;
  float4x4 orthoProjection{};

  { // setup projection matrix
    const float l = drawData->DisplayPos.x;
    const float r = drawData->DisplayPos.x + drawData->DisplaySize.x;
    const float t = drawData->DisplayPos.y;
    const float b = drawData->DisplayPos.y + drawData->DisplaySize.y;
    orthoProjection.columns[0] = float4{2.0f / (r - l), 0.0f, 0.0f, 0.0f};
    orthoProjection.columns[1] = float4{0.0f, 2.0f / (t - b), 0.0f, 0.0f};
    orthoProjection.columns[2] = float4{0.0f, 0.0f, -1.0f, 0.0f};
    orthoProjection.columns[3] = float4{(r + l) / (l - r), (t + b) / (b - t), 0.0f, 1.0f};
    // D3D12, Vulkan and WebGPU use direct slot binding, OpenGL/Metal use named binding
    const bool usesDirectBinding = (device.getBackendType() == igl::BackendType::Vulkan ||
                                    device.getBackendType() == igl::BackendType::D3D12 ||
                                    device.getBackendType() == igl::BackendType::WebGPU);
    if (!usesDirectBinding) {
      material_->shaderUniforms().setFloat4x4(igl::genNameHandle("projectionMatrix"),
                                              orthoProjection);
    }
  }

  const ImVec2 clipOff = drawData->DisplayPos; // (0,0) unless using multi-viewports
  const ImVec2 clipScale =
      drawData->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

  // Since vertex buffers are updated every frame, we must use triple buffering for Metal to work
  std::vector<DrawableData>& curFrameDrawables = drawables_[nextBufferingIndex_];
  nextBufferingIndex_ = (nextBufferingIndex_ + 1) % 3;

  const bool isOpenGL = device.getBackendType() == igl::BackendType::OpenGL;
  const bool isVulkan = device.getBackendType() == igl::BackendType::Vulkan;
  const bool isD3D12 = device.getBackendType() == igl::BackendType::D3D12;
  const bool isWebGPU = device.getBackendType() == igl::BackendType::WebGPU;
  const bool usesDirectBinding = isVulkan || isD3D12 || isWebGPU;

  auto lastBoundTextureId = static_cast<ImTextureID>(0);

  for (int n = 0; n < drawData->CmdListsCount; n++) {
    const ImDrawList* cmdList = drawData->CmdLists[n];

    if (n >= curFrameDrawables.size()) {
      curFrameDrawables.emplace_back(device, vertexInputState_, material_);
    }
    const DrawableData& drawableData = curFrameDrawables[n];

    // Upload vertex/index buffers
    drawableData.vertexData->vertexBuffer().upload(
        cmdList->VtxBuffer.Data, {cmdList->VtxBuffer.Size * sizeof(ImDrawVert), 0});
    drawableData.vertexData->indexBuffer().upload(cmdList->IdxBuffer.Data,
                                                  {cmdList->IdxBuffer.Size * sizeof(ImDrawIdx), 0});

    for (int cmdI = 0; cmdI < cmdList->CmdBuffer.Size; cmdI++) {
      const ImDrawCmd cmd = cmdList->CmdBuffer[cmdI];
      IGL_DEBUG_ASSERT(cmd.UserCallback == nullptr);

      const ImVec2 clipMin((cmd.ClipRect.x - clipOff.x) * clipScale.x,
                           (cmd.ClipRect.y - clipOff.y) * clipScale.y);
      const ImVec2 clipMax((cmd.ClipRect.z - clipOff.x) * clipScale.x,
                           (cmd.ClipRect.w - clipOff.y) * clipScale.y);

      if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y) {
        continue;
      }

      // OpenGL Y-axis goes up (Vulkan and Metal are good)
      // https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
      const igl::ScissorRect rect{.x = static_cast<uint32_t>(clipMin.x),
                                  .y = isOpenGL ? static_cast<uint32_t>(viewport.height - clipMax.y)
                                                : static_cast<uint32_t>(clipMin.y),
                                  .width = static_cast<uint32_t>(clipMax.x - clipMin.x),
                                  .height = static_cast<uint32_t>(clipMax.y - clipMin.y)};
      cmdEncoder.bindScissorRect(rect);

      if (cmd.GetTexID() != lastBoundTextureId) {
        lastBoundTextureId = cmd.GetTexID();
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* tex = reinterpret_cast<igl::ITexture*>(static_cast<intptr_t>(cmd.GetTexID()));
        if (usesDirectBinding) {
          // D3D12, Vulkan and WebGPU use direct slot binding
          // Add Vulkan support for texture reflection info in ShaderUniforms so we don't need to
          // bind the texture directly
          cmdEncoder.bindTexture(0, igl::BindTarget::kFragment, tex);
          cmdEncoder.bindSamplerState(0, igl::BindTarget::kFragment, linearSampler_.get());
        } else {
          material_->shaderUniforms().setTexture(
              "texture", tex ? tex : fontTexture_.get(), linearSampler_);
        }
      }

      drawableData.vertexData->primitiveDesc().numEntries = cmd.ElemCount;
      drawableData.vertexData->primitiveDesc().offset = cmd.IdxOffset * sizeof(ImDrawIdx);

      drawableData.drawable->draw(device,
                                  cmdEncoder,
                                  renderPipelineDesc_,
                                  usesDirectBinding ? sizeof(orthoProjection) : 0,
                                  &orthoProjection);
    }
  }

  if (isOpenGL) {
    // disable scissor
    cmdEncoder.bindScissorRect(igl::ScissorRect());
  }

  cmdEncoder.popDebugGroupLabel();
}

/* public API */

Session::Session(igl::IDevice& device,
                 igl::shell::InputDispatcher& inputDispatcher,
                 bool needInitializeSession /* = true */) :
  inputDispatcher_(inputDispatcher) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  context_ = ImGui::CreateContext();
  makeCurrentContext();

  ImGuiStyle& style = ImGui::GetStyle();
  style.TouchExtraPadding = ImVec2(5, 5); // adjust to make touches more accurate

  if (needInitializeSession) {
    initialize(device);
  }
}

void Session::initialize(igl::IDevice& device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (!isInitialized_) {
    inputListener_ = std::make_shared<InputListener>(context_);
    renderer_ = std::make_unique<Renderer>(device);
    inputDispatcher_.addMouseListener(inputListener_);
    inputDispatcher_.addTouchListener(inputListener_);
    inputDispatcher_.addKeyListener(inputListener_);
    isInitialized_ = true;
  }
}

Session::~Session() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  makeCurrentContext();

  inputDispatcher_.removeTouchListener(inputListener_);
  inputDispatcher_.removeMouseListener(inputListener_);
  inputDispatcher_.removeKeyListener(inputListener_);
  renderer_ = nullptr;
  inputListener_ = nullptr;
  ImGui::DestroyContext();
}

void Session::beginFrame(const igl::FramebufferDesc& desc, float displayScale) {
  IGL_PROFILER_FUNCTION();
  makeCurrentContext();

  IGL_DEBUG_ASSERT(desc.colorAttachments[0].texture);

  const igl::Size size = desc.colorAttachments[0].texture->getSize();

  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(size.width / displayScale, size.height / displayScale);
  io.DisplayFramebufferScale = ImVec2(displayScale, displayScale);
  io.IniFilename = nullptr;

  renderer_->newFrame(desc);
  ImGui::NewFrame();
}

void Session::endFrame(igl::IDevice& device, igl::IRenderCommandEncoder& cmdEncoder) {
  IGL_PROFILER_FUNCTION();
  makeCurrentContext();

  ImGui::EndFrame();
  ImGui::Render();
  renderer_->renderDrawData(device, cmdEncoder, ImGui::GetDrawData());
}

void Session::makeCurrentContext() const {
  ImGui::SetCurrentContext(context_);
}

void Session::drawFPS(float fps) const {
  IGL_PROFILER_FUNCTION();
  // a nice FPS counter
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                 ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                 ImGuiWindowFlags_NoMove;
  const ImGuiViewport* v = ImGui::GetMainViewport();
  IGL_DEBUG_ASSERT(v);
  ImGui::SetNextWindowPos(
      {
          v->WorkPos.x + v->WorkSize.x - 15.0f,
          v->WorkPos.y + 15.0f,
      },
      ImGuiCond_Always,
      {1.0f, 0.0f});
  ImGui::SetNextWindowBgAlpha(0.30f);
  ImGui::SetNextWindowSize(ImVec2(ImGui::CalcTextSize("FPS : _______").x, 0));
  if (ImGui::Begin("##FPS", nullptr, flags)) {
    ImGui::Text("FPS : %i", static_cast<int>(fps));
    ImGui::Text("Ms  : %.1f", 1000.0 / fps);
  }
  ImGui::End();
}

} // namespace iglu::imgui
