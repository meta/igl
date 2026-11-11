/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/TinyMeshSession.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <vector>
#include <shell/shared/imageLoader/ImageLoader.h>
#include <shell/shared/platform/DisplayContext.h>
#include <igl/CommandBuffer.h>
#include <igl/FPSCounter.h>

#if IGL_BACKEND_OPENGL
#include <igl/opengl/Device.h>
#endif // IGL_BACKEND_OPENGL
#if IGL_BACKEND_VULKAN
#include <igl/vulkan/PlatformDevice.h>
#endif // IGL_BACKEND_VULKAN

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#endif // __clang__
#define TINY_TEST_USE_DEPTH_BUFFER 1

namespace igl::shell {

namespace {

[[maybe_unused, nodiscard]] std::string stringReplaceAll(const char* input,
                                                         const char* searchString,
                                                         const char* replaceString) {
  std::string s(input);
  const size_t len = std::strlen(searchString);
  size_t pos = 0;
  while ((pos = s.find(searchString, pos)) != std::string::npos) {
    s.replace(pos, len, replaceString);
  }
  return s;
}

constexpr uint32_t kNumBufferedFrames = 3;

int width = 0;
int height = 0;

constexpr uint32_t kNumCubes = 256;

struct VertexPosUvw {
  glm::vec3 position;
  glm::vec3 color;
  glm::vec2 uv;
};

struct UniformsPerFrame {
  glm::mat4 proj;
  glm::mat4 view;
};
struct UniformsPerObject {
  glm::mat4 model;
};

// from igl/shell/renderSessions/Textured3DCubeSession.cpp
constexpr float kHalf = 1.0f;

// UV-mapped cube with indices: 24 vertices, 36 indices
const VertexPosUvw kVertexData0[] = {
    // top
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.0, 0.0, 1.0}, .uv = {0, 0}}, // 0
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.0, 1.0}, .uv = {1, 0}}, // 1
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 2
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.0, 1.0, 1.0}, .uv = {0, 1}}, // 3
    // bottom
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 4
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.0, 1.0, 0.0}, .uv = {0, 1}}, // 5
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.0}, .uv = {1, 1}}, // 6
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.0, 0.0}, .uv = {1, 0}}, // 7
    // left
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.0}, .uv = {1, 0}}, // 8
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.0, 1.0, 0.0}, .uv = {0, 0}}, // 9
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.0, 1.0, 1.0}, .uv = {0, 1}}, // 10
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 11
    // right
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 12
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.0, 0.0}, .uv = {1, 0}}, // 13
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.0, 1.0}, .uv = {1, 1}}, // 14
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.0, 0.0, 1.0}, .uv = {0, 1}}, // 15
    // front
    {.position = {+kHalf, -kHalf, -kHalf}, .color = {1.0, 0.0, 0.0}, .uv = {0, 0}}, // 16
    {.position = {+kHalf, +kHalf, -kHalf}, .color = {1.0, 1.0, 0.0}, .uv = {1, 0}}, // 17
    {.position = {+kHalf, +kHalf, +kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {1, 1}}, // 18
    {.position = {+kHalf, -kHalf, +kHalf}, .color = {1.0, 0.0, 1.0}, .uv = {0, 1}}, // 19
    // back
    {.position = {-kHalf, +kHalf, -kHalf}, .color = {0.0, 1.0, 0.0}, .uv = {1, 0}}, // 20
    {.position = {-kHalf, -kHalf, -kHalf}, .color = {1.0, 1.0, 1.0}, .uv = {0, 0}}, // 21
    {.position = {-kHalf, -kHalf, +kHalf}, .color = {0.0, 0.0, 1.0}, .uv = {0, 1}}, // 22
    {.position = {-kHalf, +kHalf, +kHalf}, .color = {0.0, 1.0, 1.0}, .uv = {1, 1}}, // 23
};

constexpr uint16_t kIndexData[] = {0,  1,  2,  2,  3,  0,  4,  5,  6,  6,  7,  4,
                                   8,  9,  10, 10, 11, 8,  12, 13, 14, 14, 15, 12,
                                   16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};

UniformsPerFrame perFrame;
UniformsPerObject perObject[kNumCubes];
glm::vec3 axis[kNumCubes];

// Per-object uniforms are bound at offsets that must honor the device's uniform buffer offset
// alignment (256 bytes on WebGPU), so each object's block is padded to that stride.
[[nodiscard]] size_t getPerObjectStride(const IDevice& device) {
  size_t alignment = 0;
  if (!device.getFeatureLimits(DeviceFeatureLimits::BufferAlignment, alignment) || alignment == 0) {
    return sizeof(UniformsPerObject);
  }
  return (sizeof(UniformsPerObject) + alignment - 1) / alignment * alignment;
}

void uploadPerObject(IBuffer& buffer, size_t stride) {
  if (stride == sizeof(UniformsPerObject)) {
    buffer.upload(perObject, BufferRange(sizeof(perObject)));
    return;
  }
  std::vector<uint8_t> data(kNumCubes * stride);
  for (uint32_t i = 0; i != kNumCubes; i++) {
    std::memcpy(data.data() + i * stride, &perObject[i], sizeof(UniformsPerObject));
  }
  buffer.upload(data.data(), BufferRange(data.size()));
}

#if IGL_BACKEND_METAL
[[nodiscard]] const char* getMetalShaderSource() {
  return R"(
          #include <metal_stdlib>
          #include <simd/simd.h>
          using namespace metal;

          constant float2 pos[3] = {
            float2(-0.6, -0.4),
            float2( 0.6, -0.4),
            float2( 0.0,  0.6)
          };
          constant float3 col[3] = {
            float3(1.0, 0.0, 0.0),
            float3(0.0, 1.0, 0.0),
            float3(0.0, 0.0, 1.0)
          };

          struct VertexOut {
            float4 position [[position]];
            float3 uvw;
          };

          vertex VertexOut vertexShader(uint vid [[vertex_id]]) {
            VertexOut out;
            out.position = float4(pos[vid], 0.0, 1.0);
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
#endif // IGL_BACKEND_METAL

[[nodiscard]] const char* getVulkanVertexShaderSource() {
  return R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec3 col;
layout (location=2) in vec2 st;
layout (location=0) out vec3 color;
layout (location=1) out vec2 uv;

#if VULKAN
layout (set = 1, binding = 0, std140)
#else
layout (binding = 0, std140)
#endif
uniform UniformsPerFrame {
  mat4 proj;
  mat4 view;
} perFrame;

#if VULKAN
layout (set = 1, binding = 1, std140)
#else
layout (binding = 1, std140)
#endif
uniform UniformsPerObject {
  mat4 model;
} perObject;

void main() {
  mat4 proj = perFrame.proj;
  mat4 view = perFrame.view;
  mat4 model = perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
  color = col;
  uv = st;
}
)";
}

[[nodiscard]] const char* getVulkanFragmentShaderSource() {
  return R"(
layout (location=0) in vec3 color;
layout (location=1) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

#if VULKAN
layout (set = 0, binding = 0) uniform sampler2D uTex0;
layout (set = 0, binding = 1) uniform sampler2D uTex1;
#else
layout (binding = 0) uniform sampler2D uTex0;
layout (binding = 1) uniform sampler2D uTex1;
#endif

void main() {
  vec4 t0 = texture(uTex0, 2.0 * uv);
  vec4 t1 = texture(uTex1,  uv);
  out_FragColor = vec4(2.0 * color * (t0.rgb * t1.rgb), 1.0);
};
)";
}

[[nodiscard]] const char* getWgslShaderSource() {
  return R"(
struct UniformsPerFrame {
  proj : mat4x4f,
  view : mat4x4f,
};

struct UniformsPerObject {
  model : mat4x4f,
};

@group(1) @binding(0) var<uniform> perFrame : UniformsPerFrame;
@group(1) @binding(1) var<uniform> perObject : UniformsPerObject;
@group(0) @binding(0) var uTex0 : texture_2d<f32>;
@group(0) @binding(1) var uSampler0 : sampler;
@group(0) @binding(2) var uTex1 : texture_2d<f32>;
@group(0) @binding(3) var uSampler1 : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec3f,
  @location(1) uv : vec2f,
};

@vertex
fn vertexShader(@location(0) pos : vec3f,
                @location(1) col : vec3f,
                @location(2) st : vec2f) -> VertexOut {
  var out : VertexOut;
  out.position = perFrame.proj * perFrame.view * perObject.model * vec4f(pos, 1.0);
  out.color = col;
  out.uv = st;
  return out;
}

@fragment
fn fragmentShader(in : VertexOut) -> @location(0) vec4f {
  let t0 = textureSample(uTex0, uSampler0, 2.0 * in.uv);
  let t1 = textureSample(uTex1, uSampler1, in.uv);
  return vec4f(2.0 * in.color * (t0.rgb * t1.rgb), 1.0);
}
)";
}

[[nodiscard]] std::unique_ptr<IShaderStages> getShaderStagesForBackend(IDevice& device) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (device.getBackendType()) {
  case igl::BackendType::Invalid:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;

  case igl::BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource(), "vertexShader", "fragmentShader", "", nullptr);

#if IGL_BACKEND_VULKAN
  case igl::BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(device,
                                                           getVulkanVertexShaderSource(),
                                                           "main",
                                                           "",
                                                           getVulkanFragmentShaderSource(),
                                                           "main",
                                                           "",
                                                           nullptr);
#endif // IGL_BACKEND_VULKAN

// @fb-only
  // @fb-only
    // @fb-only
    // @fb-only
// @fb-only

#if IGL_BACKEND_METAL
  case igl::BackendType::Metal:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getMetalShaderSource(), "vertexShader", "fragmentShader", "", nullptr);
#endif // IGL_BACKEND_METAL

#if IGL_BACKEND_OPENGL
  case igl::BackendType::OpenGL: {
    const auto glVersion =
        static_cast<igl::opengl::Device&>(device).getContext().deviceFeatures().getGLVersion();

    // The shaders are the Vulkan GLSL under `#version 460`.
    if (glVersion >= igl::opengl::GLVersion::v4_6) {
      const std::string codeVS1 =
          stringReplaceAll(getVulkanVertexShaderSource(), "gl_VertexIndex", "gl_VertexID");
      const auto codeVS2 = "#version 460\n" + codeVS1;
      const auto codeFS = "#version 460\n" + std::string(getVulkanFragmentShaderSource());

      return igl::ShaderStagesCreator::fromModuleStringInput(
          device, codeVS2.c_str(), "main", "", codeFS.c_str(), "main", "", nullptr);
    } else {
      IGL_LOG_INFO("TinyMeshSession: needs OpenGL 4.6; only clearing\n");
      return nullptr;
    }
  }
#endif // IGL_BACKEND_OPENGL

  default:
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    return nullptr;
  }
}

} // namespace

TinyMeshSession::TinyMeshSession(std::shared_ptr<Platform> platform) :
  RenderSession(std::move(platform)) {
  listener_ = std::make_shared<Listener>(*this);
  getPlatform().getInputDispatcher().addKeyListener(listener_);
  imguiSession_ = std::make_unique<iglu::imgui::Session>(getPlatform().getDevice(),
                                                         getPlatform().getInputDispatcher());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void TinyMeshSession::initialize() noexcept {
  device_ = &getPlatform().getDevice();

  // Vertex buffer, Index buffer and Vertex Input.
  // Shared, not Private: virtualized Metal GPUs fail to create Private buffers with data.
  vb0_ = device_->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                          .data = kVertexData0,
                                          .length = sizeof(kVertexData0),
                                          .storage = ResourceStorage::Shared,
                                          .debugName = "Buffer: vertex"},
                               nullptr);
  ib0_ = device_->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                          .data = kIndexData,
                                          .length = sizeof(kIndexData),
                                          .storage = ResourceStorage::Shared,
                                          .debugName = "Buffer: index"},
                               nullptr);
  const size_t perObjectStride = getPerObjectStride(*device_);
  // create Uniform buffers to store uniforms for 2 objects
  // Uniform blocks are missing on OpenGL ES 2.0; the mesh is not drawn there anyway.
  if (device_->getBackendType() != BackendType::OpenGL ||
      device_->hasFeature(DeviceFeatures::UniformBlocks)) {
    for (uint32_t i = 0; i != kNumBufferedFrames; i++) {
      ubPerFrame_.push_back(
          device_->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                                           .data = &perFrame,
                                           .length = sizeof(UniformsPerFrame),
                                           .storage = ResourceStorage::Shared,
                                           .hint = BufferDesc::BufferAPIHintBits::UniformBlock,
                                           .debugName = "Buffer: uniforms (per frame)"},
                                nullptr));
      ubPerObject_.push_back(device_->createBuffer(
          BufferDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                     .data = perObjectStride == sizeof(UniformsPerObject) ? perObject : nullptr,
                     .length = kNumCubes * perObjectStride,
                     .storage = ResourceStorage::Shared,
                     .hint = BufferDesc::BufferAPIHintBits::UniformBlock,
                     .debugName = "Buffer: uniforms (per object)"},
          nullptr));
    }
  }

  vertexInput0_ = device_->createVertexInputState(
      VertexInputStateDesc{
          .numAttributes = 3,
          .attributes =
              {
                  {.bufferIndex = 0,
                   .format = VertexAttributeFormat::Float3,
                   .offset = offsetof(VertexPosUvw, position),
                   .name = "pos",
                   .location = 0},
                  {.bufferIndex = 0,
                   .format = VertexAttributeFormat::Float3,
                   .offset = offsetof(VertexPosUvw, color),
                   .name = "col",
                   .location = 1},
                  {.bufferIndex = 0,
                   .format = VertexAttributeFormat::Float2,
                   .offset = offsetof(VertexPosUvw, uv),
                   .name = "st",
                   .location = 2},
              },
          .numInputBindings = 1,
          .inputBindings = {{.stride = sizeof(VertexPosUvw)}},
      },
      nullptr);

  depthStencilState_ = device_->createDepthStencilState(
      DepthStencilStateDesc{
          .compareFunction = igl::CompareFunction::Less,
          .isDepthWriteEnabled = true,
      },
      nullptr);

  {
    const uint32_t texWidth = 256;
    const uint32_t texHeight = 256;
    // OpenGL ES 2.0 cannot sample BGRA sRGB; the gray XOR pattern reads the same as RGBA.
    const TextureFormat xorFormat =
        (device_->getTextureFormatCapabilities(TextureFormat::BGRA_SRGB) &
         ICapabilities::TextureFormatCapabilityBits::Sampled) != 0
            ? TextureFormat::BGRA_SRGB
            : TextureFormat::RGBA_UNorm8;
    const TextureDesc desc = TextureDesc::new2D(
        xorFormat, texWidth, texHeight, TextureDesc::TextureUsageBits::Sampled, "XOR pattern");
    texture0_ = device_->createTexture(desc, nullptr);
    std::vector<uint32_t> pixels(
        static_cast<std::vector<unsigned int>::size_type>(texWidth * texHeight));

    for (uint32_t y = 0; y != texHeight; y++) {
      for (uint32_t x = 0; x != texWidth; x++) {
        // create a XOR pattern
        pixels[y * texWidth + x] = 0xFF000000 + ((x ^ y) << 16) + ((x ^ y) << 8) + (x ^ y);
      }
    }
    texture0_->upload(TextureRangeDesc::new2D(0, 0, texWidth, texHeight), pixels.data());
  }
  {
    const auto imageData = getPlatform().getImageLoader().loadImageData("marble.png");
    IGL_DEBUG_ASSERT(imageData.data, "Cannot load marble.png");
    const TextureDesc desc = TextureDesc::new2D(igl::TextureFormat::RGBA_SRGB,
                                                imageData.desc.width,
                                                imageData.desc.height,
                                                TextureDesc::TextureUsageBits::Sampled,
                                                "marble.png");
    texture1_ = device_->createTexture(desc, nullptr);
    if (texture1_ && imageData.data) {
      texture1_->upload(texture1_->getFullRange(), imageData.data->data());
    }
  }
  sampler_ = device_->createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
          .addressModeU = igl::SamplerAddressMode::Repeat,
          .addressModeV = igl::SamplerAddressMode::Repeat,
          .debugName = "Sampler: linear",
      },
      nullptr);

  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = device_->createCommandQueue({}, nullptr);

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = {1.0f, 0.0f, 0.0f, 1.0f},
      }},
#if TINY_TEST_USE_DEPTH_BUFFER
      .depthAttachment = {.loadAction = LoadAction::Clear, .clearDepth = 1.0},
#else
      .depthAttachment = {.loadAction = LoadAction::DontCare},
#endif // TINY_TEST_USE_DEPTH_BUFFER
  };

  // initialize random rotation axes for all cubes
  for (glm::vec3& axi : axis) {
    axi = glm::sphericalRand(1.0f);
  }
}

std::shared_ptr<ITexture> TinyMeshSession::getVulkanNativeDepth() {
#if IGL_BACKEND_VULKAN
  if (device_->getBackendType() == BackendType::Vulkan) {
    const auto& vkPlatformDevice = device_->getPlatformDevice<igl::vulkan::PlatformDevice>();

    IGL_DEBUG_ASSERT(vkPlatformDevice != nullptr);

    Result ret;
    std::shared_ptr<ITexture> drawable =
        vkPlatformDevice->createTextureFromNativeDepth(width, height, &ret);

    IGL_DEBUG_ASSERT(ret.isOk());
    return drawable;
  }
#endif // IGL_BACKEND_VULKAN

  // TODO: unhardcode Vulkan assumption above
  return nullptr;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void TinyMeshSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }

  width = surfaceTextures.color->getSize().width;
  height = surfaceTextures.color->getSize().height;

  const float deltaSeconds = getDeltaSeconds();

  fps_.updateFPS(deltaSeconds);
  currentTime_ += deltaSeconds;

  if (!framebuffer_) {
    framebufferDesc_.colorAttachments[0].texture = surfaceTextures.color;

#if TINY_TEST_USE_DEPTH_BUFFER
    framebufferDesc_.depthAttachment.texture = device_->getBackendType() == BackendType::WebGPU
                                                   ? surfaceTextures.depth
                                                   : getVulkanNativeDepth();
#endif // TINY_TEST_USE_DEPTH_BUFFER
    framebuffer_ = device_->createFramebuffer(framebufferDesc_, nullptr);
    IGL_DEBUG_ASSERT(framebuffer_);

    const TextureFormat depthFormat =
        framebuffer_->getDepthAttachment()
            ? framebuffer_->getDepthAttachment()->getProperties().format
            : TextureFormat::Invalid;
    std::shared_ptr<IShaderStages> shaderStages = getShaderStagesForBackend(*device_);
    if (shaderStages) {
      renderPipelineStateMesh_ = device_->createRenderPipeline(
          RenderPipelineDesc{
              .vertexInputState = vertexInput0_,
              .shaderStages = shaderStages,
              .targetDesc =
                  {
                      .colorAttachments = {{
                          .textureFormat =
                              framebuffer_->getColorAttachment(0)->getProperties().format,
                      }},
                      .depthAttachmentFormat = depthFormat,
                  },
#if !TINY_TEST_USE_DEPTH_BUFFER
              .cullMode = igl::CullMode::Back,
#endif // TINY_TEST_USE_DEPTH_BUFFER
              .frontFaceWinding = igl::WindingMode::Clockwise,
              .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE("uTex0")},
                                         {1, IGL_NAMEHANDLE("uTex1")}},
              .debugName = igl::genNameHandle("Pipeline: mesh"),
          },
          nullptr);
    }
  }

  framebuffer_->updateDrawable(surfaceTextures.color);

  // from igl/shell/renderSessions/Textured3DCubeSession.cpp
  const float fov = static_cast<float>(45.0f * (M_PI / 180.0f));
  const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
  perFrame.proj = glm::perspectiveLH(fov, aspectRatio, 0.1f, 500.0f);
  // place a "camera" behind the cubes, the distance depends on the total number of cubes
  perFrame.view = glm::translate(
      glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, std::sqrt(kNumCubes / 16.0f) * 20.0f * kHalf));
  // The uniform buffers exist only where the mesh can be drawn (not on OpenGL ES 2.0).
  const bool drawMesh = renderPipelineStateMesh_ && !ubPerFrame_.empty();
  if (drawMesh) {
    ubPerFrame_[frameIndex_]->upload(&perFrame, BufferRange(sizeof(perFrame)));
  }

  // rotate cubes around random axes
  for (uint32_t i = 0; i != kNumCubes; i++) {
    const float direction = std::pow(-1, static_cast<float>(i + 1));
    const uint32_t cubesInLine = static_cast<uint32_t>(std::sqrt(kNumCubes));
    const glm::vec3 offset = glm::vec3(
        -1.5f * std::sqrt(kNumCubes) + 4.0f * static_cast<float>(i % cubesInLine),
        -1.5f * std::sqrt(kNumCubes) + 4.0f * std::floor(static_cast<float>(i) / cubesInLine),
        0);
    perObject[i].model = glm::rotate(glm::translate(glm::mat4(1.0f), offset),
                                     static_cast<float>(direction * currentTime_),
                                     axis[i]);
  }

  const size_t perObjectStride = getPerObjectStride(*device_);
  if (drawMesh) {
    uploadPerObject(*ubPerObject_[frameIndex_], perObjectStride);
  }

  // Command buffers (1-N per thread): create, submit and forget
  const std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer({}, nullptr);

  const Viewport viewport = {.x = 0.0f,
                             .y = 0.0f,
                             .width = static_cast<float>(width),
                             .height = static_cast<float>(height),
                             .minDepth = 0.0f,
                             .maxDepth = +1.0f};
  const ScissorRect scissor = {.x = 0,
                               .y = 0,
                               .width = static_cast<uint32_t>(width),
                               .height = static_cast<uint32_t>(height)};

  // This will clear the framebuffer
  const auto commands = buffer->createRenderCommandEncoder(renderPass_, framebuffer_);

  if (drawMesh) {
    commands->bindRenderPipelineState(renderPipelineStateMesh_);
    commands->bindViewport(viewport);
    commands->bindScissorRect(scissor);
    commands->pushDebugGroupLabel("Render Mesh", Color(1, 0, 0));
    commands->bindVertexBuffer(0, *vb0_);
    commands->bindDepthStencilState(depthStencilState_);
    commands->bindBuffer(0, ubPerFrame_[frameIndex_].get());
    commands->bindTexture(0, igl::BindTarget::kFragment, texture0_.get());
    commands->bindTexture(1, igl::BindTarget::kFragment, texture1_.get());
    commands->bindSamplerState(0, igl::BindTarget::kFragment, sampler_.get());
    commands->bindSamplerState(1, igl::BindTarget::kFragment, sampler_.get());
    // Draw 2 cubes: we use uniform buffer to update matrices
    commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);
    for (uint32_t i = 0; i != kNumCubes; i++) {
      commands->bindBuffer(
          1, ubPerObject_[frameIndex_].get(), i * perObjectStride, sizeof(UniformsPerObject));
      commands->drawIndexed(3u * 6u * 2u);
    }
    commands->popDebugGroupLabel();
  }
  {
    imguiSession_->beginFrame(framebufferDesc_, getPlatform().getDisplayContext().pixelsPerPoint);
    ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Image(ImTextureID(texture1_.get()), ImVec2(512, 512));
    ImGui::End();
    imguiSession_->drawFPS(fps_.getAverageFPS());
    imguiSession_->endFrame(getPlatform().getDevice(), *commands);
  }
  commands->endEncoding();

  buffer->present(surfaceTextures.color);

  commandQueue_->submit(*buffer);

  frameIndex_ = (frameIndex_ + 1) % kNumBufferedFrames;
}

bool TinyMeshSession::Listener::process(const CharEvent& event) {
  if (event.character == 't') {
    session.texture1_.reset();
    return true;
  }
  return false;
}

} // namespace igl::shell
