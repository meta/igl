/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * A brief tutorial how to run this beast:
 *
 * 1) Run the script "deploy_deps.py" from the IGL root folder.
 * 2) Run the script "deploy_content.py" from the IGL root folder.
 * 3) Run this app.
 *
 */

// @fb-only

#define GLFW_INCLUDE_NONE

// NOLINTNEXTLINE(bugprone-reserved-identifier)
#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <Compress.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <ktx.h>
#include <meshoptimizer.h>
#include <mutex>
#include <shared/Camera.h>
#include <shared/UtilsCubemap.h>
#include <stb/stb_image.h>
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>
#include <taskflow/core/async.hpp>
#include <thread>
#include <tiny_obj_loader.h>
#include <igl/FPSCounter.h>
#include <igl/IGL.h>
#include <igl/vulkan/util/TextureFormat.h>

#define USE_TEXTURE_LOADER 0
#define USE_OPENGL_BACKEND 0

#if IGL_BACKEND_OPENGL && !IGL_BACKEND_VULKAN
// no IGL/Vulkan was compiled in, switch to IGL/OpenGL
#undef USE_OPENGL_BACKEND
#define USE_OPENGL_BACKEND 1
#endif

#if USE_OPENGL_BACKEND
// for KTX textures
#define VK_FORMAT_R32G32B32A32_SFLOAT 109
#define VK_FORMAT_BC7_UNORM_BLOCK 145
#endif // USE_OPENGL_BACKEND

#if defined(__cpp_lib_format) && !IGL_PLATFORM_APPLE
#include <format>
#define IGL_FORMAT std::format
#else
#include <fmt/format.h>
#define IGL_FORMAT fmt::format
#endif // __cpp_lib_format

// NOLINTBEGIN(facebook-unused-include-check)
// clang-format off
#if USE_OPENGL_BACKEND
  #include <igl/RenderCommandEncoder.h>
  #include <igl/opengl/RenderCommandEncoder.h>
  #include <igl/opengl/RenderPipelineState.h>
  #if IGL_PLATFORM_WINDOWS
    #include <igl/opengl/wgl/Context.h>
    #include <igl/opengl/wgl/Device.h>
    #include <igl/opengl/wgl/HWDevice.h>
    #include <igl/opengl/wgl/PlatformDevice.h>
  #elif IGL_PLATFORM_LINUX
    #include <igl/opengl/glx/Context.h>
    #include <igl/opengl/glx/Device.h>
    #include <igl/opengl/glx/HWDevice.h>
    #include <igl/opengl/glx/PlatformDevice.h>
  #endif
#else
  #include <igl/vulkan/Common.h>
  #include <igl/vulkan/Device.h>
  #include <igl/vulkan/HWDevice.h>
  #include <igl/vulkan/PlatformDevice.h>
  #include <igl/vulkan/VulkanContext.h>
#endif
// clang-format on
// NOLINTEND(facebook-unused-include-check)

#include <GLFW/glfw3.h>

// NOLINTBEGIN(facebook-unused-include-check)
#if defined(_XLESS_GLFW_)
// do nothing
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#else
#error Unsupported OS
#endif
#include <GLFW/glfw3native.h>
// NOLINTEND(facebook-unused-include-check)

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

#if IGL_WITH_IGLU
#include <IGLU/imgui/Session.h>
#endif // IGL_WITH_IGLU

#if USE_TEXTURE_LOADER
#include <IGLU/texture_loader/ktx1/TextureLoaderFactory.h>
#include <IGLU/texture_loader/ktx2/TextureLoaderFactory.h>
static_assert(IGL_WITH_IGLU != 0,
              "USE_TEXTURE_LOADER can be enabled only together with IGL_WITH_IGLU");
#endif // USE_TEXTURE_LOADER

namespace {

constexpr uint32_t kMeshCacheVersion = 0xC0DE0009;
#if USE_OPENGL_BACKEND
constexpr bool kEnableCompression = false;
#else
constexpr bool kEnableCompression = true;
constexpr bool kPreferIntegratedGPU = false;
#endif // USE_OPENGL_BACKEND

// NOLINTNEXTLINE(facebook-static-object-destructor-check)
std::string contentRootFolder;

#if IGL_WITH_IGLU
// NOLINTNEXTLINE(facebook-static-object-destructor-check)
std::unique_ptr<iglu::imgui::Session> imguiSession;
// NOLINTNEXTLINE(facebook-static-object-destructor-check)
igl::shell::InputDispatcher inputDispatcher;
#endif // IGL_WITH_IGLU

#if USE_TEXTURE_LOADER
void loadKtxTexture(const igl::IDevice& device,
                    igl::ICommandQueue& commandQueue,
                    const std::string filename,
                    std::shared_ptr<igl::ITexture>& texture,
                    bool generateMipmaps) {
  igl::Result result;
  if (!filename.empty() && filename.back() != '2' && std::filesystem::exists(filename + "2")) {
    loadKtxTexture(device, commandQueue, filename + "2", texture, generateMipmaps);
    return;
  }

  const auto size = std::filesystem::file_size(filename);
  FILE* file = std::fopen(filename.c_str(), "rb");
  if (!IGL_DEBUG_VERIFY(file)) {
    return;
  }
  std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(size);

  std::fread(data.get(), 1, size, file);

  std::fclose(file);

  iglu::textureloader::ktx1::TextureLoaderFactory factory1;
  iglu::textureloader::ktx2::TextureLoaderFactory factory2;
  iglu::textureloader::ITextureLoaderFactory* factory =
      filename.back() == '2' ? static_cast<iglu::textureloader::ITextureLoaderFactory*>(&factory2)
                             : &factory1;

  auto loader = factory->tryCreate(data.get(), size, &result);
  if (!IGL_DEBUG_VERIFY(loader && result.isOk())) {
    return;
  }

  if (!texture) {
    IGL_DEBUG_ASSERT(loader->isSupported(device));
    texture = loader->create(device, &result);
    if (!IGL_DEBUG_VERIFY(texture && result.isOk())) {
      return;
    }
  }
  IGL_DEBUG_ASSERT(loader->isSupported(device, texture->getUsage()));
  loader->upload(*texture, &result);
  if (IGL_DEBUG_VERIFY(result.isOk())) {
    if (generateMipmaps) {
      texture->generateMipmap(commandQueue);
    }
  }
}
#endif // USE_TEXTURE_LOADER

const char* kCodeComputeTest = R"(
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#ifdef VULKAN
// kBinding_StorageImages in VulkanContext.cpp
layout (set = 3, binding = 6, rgba8) uniform readonly  image2D kTextures2Din[];
layout (set = 3, binding = 6, rgba8) uniform writeonly image2D kTextures2Dout[];

layout(push_constant) uniform PushConstants {
  uint textureId;
} pc;
#else
layout (binding = 2, rgba8) uniform readonly  image2D kTextures2Din;
layout (binding = 2, rgba8) uniform writeonly image2D kTextures2Dout;
#endif

vec4 imageLoad2D(ivec2 uv) {
#ifdef VULKAN
  return imageLoad(kTextures2Din[pc.textureId], uv);
#else
  return imageLoad(kTextures2Din, uv);
#endif
}

void imageStore2D(ivec2 uv, vec4 data) {
#ifdef VULKAN
  imageStore(kTextures2Dout[pc.textureId], uv, data);
#else
  imageStore(kTextures2Dout, uv, data);
#endif
}

void main() {
   vec4 pixel = imageLoad2D(ivec2(gl_GlobalInvocationID.xy));
   float luminance = dot(pixel, vec4(0.299, 0.587, 0.114, 0.0)); // https://www.w3.org/TR/AERT/#color-contrast
   imageStore2D(ivec2(gl_GlobalInvocationID.xy), vec4(vec3(luminance), 1.0));
}
)";

const char* kCodeFullscreenVS = R"(
layout (location=0) out vec2 uv;
void main() {
  // generate a triangle covering the entire screen
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
#ifdef VULKAN
  gl_Position = vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
#else
  gl_Position = vec4(uv * vec2(2, 2) + vec2(-1, -1), 0.0, 1.0);
#endif
}
)";

const char* kCodeFullscreenFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

#ifdef VULKAN
layout(set = 0, binding = 0) uniform sampler2D texFullScreen;
#else
uniform sampler2D texFullScreen;
#endif

void main() {
  out_FragColor = texture(texFullScreen, uv);
}
)";

const char* kCodeVS = R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;
#ifdef VULKAN
layout (location=3) in uint mtlIndex;
#else
layout (location=3) in float mtlIndex;
#endif

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  int bDrawNormals;
  int bDebugLines;
  vec2 padding;
};

struct UniformsPerObject {
  mat4 model;
};

struct Material {
   vec4 ambient;
   vec4 diffuse;
   int texAmbient;
   int texDiffuse;
   int texAlpha;
   int padding;
};

#ifdef VULKAN
layout(set = 1, binding = 0, std140) uniform PerFrame {
  UniformsPerFrame perFrame;
};

layout(set = 1, binding = 1, std140) uniform PerObject {
  UniformsPerObject perObject;
};

layout(set = 1, binding = 2, std430) readonly buffer Materials {
  Material mtl[];
} mat;
#else
uniform PerFrame  {
  UniformsPerFrame perFrame;
};
uniform PerObject {
  UniformsPerObject perObject;
};
uniform MeshMaterials {
  Material materials[132];
};
#endif

// output
struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
};
layout (location=0) out PerVertex vtx;
layout (location=5) flat out Material mtl;
//

void main() {
  mat4 proj = perFrame.proj;
  mat4 view = perFrame.view;
  mat4 model = perObject.model;
  mat4 light = perFrame.light;
#ifdef VULKAN
  mtl = mat.mtl[uint(mtlIndex)];
#else
  mtl = materials[int(mtlIndex)];
#endif
  gl_Position = proj * view * model * vec4(pos, 1.0);

  // Compute the normal in world-space
  mat3 norm_matrix = transpose(inverse(mat3(model)));
  vtx.normal = normalize(norm_matrix * normal);
  vtx.uv = uv;
  vtx.shadowCoords = light * model * vec4(pos, 1.0);
}
)";

const char* kCodeVS_Wireframe = R"(
layout (location=0) in vec3 pos;

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
};

struct UniformsPerObject {
  mat4 model;
};

#ifdef VULKAN
layout(set = 1, binding = 0, std140)
#endif
uniform PerFrame {
  UniformsPerFrame perFrame;
};
#ifdef VULKAN
layout(set = 1, binding = 1, std140)
#endif
uniform PerObject{
  UniformsPerObject perObject;
};

void main() {
  mat4 proj = perFrame.proj;
  mat4 view = perFrame.view;
  mat4 model = perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kCodeFS_Wireframe = R"(
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = vec4(1.0);
};
)";

const char* kCodeFS = R"(
struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  int bDrawNormals;
  int bDebugLines;
  vec2 padding;
};
#ifdef VULKAN
layout(set = 1, binding = 0, std140)
#endif
uniform PerFrame {
  UniformsPerFrame perFrame;
};

struct Material {
  vec4 ambient;
  vec4 diffuse;
  int texAmbient;
  int texDiffuse;
  int texAlpha;
  int padding;
};
struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
};

layout (location=0) in PerVertex vtx;
layout (location=5) flat in Material mtl;

layout (location=0) out vec4 out_FragColor;

#ifdef VULKAN
layout(set = 0, binding = 0) uniform sampler2DShadow texShadow;
layout(set = 0, binding = 4) uniform samplerCube texSkyboxIrradiance;

vec4 textureBindless2D(uint textureid, vec2 uv) {
  return texture(sampler2D(kTextures2D[textureid], kSamplers[1]), uv);
}
#else
  layout(binding = 0) uniform sampler2D texShadow;
  layout(binding = 1) uniform sampler2D texAmbient;
  layout(binding = 2) uniform sampler2D texDiffuse;
  layout(binding = 3) uniform sampler2D texAlpha;
  layout(binding = 4) uniform samplerCube texSkyboxIrradiance;
#endif // VULKAN

float PCF3(vec3 uvw) {
  float size = 1.0 / float( textureSize(texShadow, 0).x );
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
#ifdef VULKAN
      shadow += texture(texShadow, uvw + size * vec3(u, v, 0));
#else
      shadow += (uvw.z <= texture(texShadow, uvw.xy + size * vec2(u, v) ).r) ? 1.0 : 0.0;
#endif
  return shadow / 9;
}

float shadow(vec4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float depthBias = -0.00005;
#ifdef VULKAN
    s.y = 1.0 - s.y;
#endif
    float shadowSample = PCF3(vec3(s.x, s.y, s.z + depthBias));
    return mix(0.3, 1.0, shadowSample);
  }
  return 1.0;
}

void main() {
#ifdef VULKAN
  vec4 alpha = textureBindless2D(mtl.texAlpha, vtx.uv);
  if (mtl.texAlpha > 0 && alpha.r < 0.5)
    discard;
  vec4 Ka = mtl.ambient * textureBindless2D(mtl.texAmbient, vtx.uv);
  vec4 Kd = mtl.diffuse * textureBindless2D(mtl.texDiffuse, vtx.uv);
#else
  vec4 alpha = texture(texAlpha, vtx.uv);
  // check it is not a dummy 1x1 texture
  if (textureSize(texAlpha, 0).x > 1 && alpha.r < 0.5)
    discard;
  vec4 Ka = mtl.ambient * texture(texAmbient, vtx.uv);
  vec4 Kd = mtl.diffuse * texture(texDiffuse, vtx.uv);
#endif
  bool drawNormals = perFrame.bDrawNormals > 0;
  if (Kd.a < 0.5)
    discard;
  vec3 n = normalize(vtx.normal);
  float NdotL1 = clamp(dot(n, normalize(vec3(-1, 1,+1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(vec3(-1, 1,-1))), 0.0, 1.0);
  float NdotL = 0.5 * (NdotL1+NdotL2);
  // IBL diffuse
  const vec4 f0 = vec4(0.04);
  vec4 diffuse = texture(texSkyboxIrradiance, n) * Kd * (vec4(1.0) - f0);
  out_FragColor = drawNormals ?
    vec4(0.5 * (n+vec3(1.0)), 1.0) :
    Ka + diffuse * shadow(vtx.shadowCoords);
};
)";

const char* kShadowVS = R"(
layout (location=0) in vec3 pos;

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  int bDrawNormals;
  int bDebugLines;
  vec2 padding;
};

struct UniformsPerObject {
  mat4 model;
};

#ifdef VULKAN
layout(set = 1, binding = 0, std140) uniform PerFrame {
  UniformsPerFrame perFrame;
};
layout(set = 1, binding = 1, std140) uniform PerObject {
  UniformsPerObject perObject;
};
#else
uniform ShadowFrameUniforms {
   UniformsPerFrame perFrame;
};
uniform ShadowObjectUniforms {
  UniformsPerObject perObject;
};

#endif
void main() {
  mat4 proj = perFrame.proj;
  mat4 view = perFrame.view;
  mat4 model = perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kShadowFS = R"(
void main() {
};
)";

const char* kSkyboxVS = R"(
layout (location=0) out vec3 textureCoords;

const vec3 positions[8] = vec3[8](
	vec3(-1.0,-1.0, 1.0), vec3( 1.0,-1.0, 1.0), vec3( 1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0),
	vec3(-1.0,-1.0,-1.0), vec3( 1.0,-1.0,-1.0), vec3( 1.0, 1.0,-1.0), vec3(-1.0, 1.0,-1.0)
);

const int indices[36] = int[36](
	0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3
);

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  int bDrawNormals;
  int bDebugLines;
  vec2 padding;
};

#ifdef VULKAN
layout(set = 1, binding = 0, std140) uniform PerFrame {
  UniformsPerFrame perFrame;
};
#else
uniform SkyboxFrameUniforms {
  UniformsPerFrame perFrame;
};

#endif
void main() {
  mat4 proj = perFrame.proj;
  mat4 view = perFrame.view;
  // discard translation
  view = mat4(view[0], view[1], view[2], vec4(0, 0, 0, 1));
  mat4 transform = proj * view;
  vec3 pos = positions[indices[gl_VertexIndex]];
  gl_Position = (transform * vec4(pos, 1.0)).xyww;

  // skybox
  textureCoords = pos;
}

)";
const char* kSkyboxFS = R"(
layout (location=0) in vec3 textureCoords;
layout (location=0) out vec4 out_FragColor;

#ifdef VULKAN
layout(set = 0, binding = 1) uniform samplerCube texSkybox;
#else
uniform samplerCube texSkybox;
#endif

void main() {
  out_FragColor = texture(texSkybox, textureCoords);
}
)";

// @fb-only
// @fb-only
// @fb-only

using namespace igl;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

int width_ = 1920;
int height_ = 1080;
igl::FPSCounter fps_;

constexpr uint32_t kNumBufferedFrames = 3;

// NOLINTBEGIN(facebook-static-object-destructor-check)
std::unique_ptr<IDevice> device_; // NOLINT(readability-identifier-naming)
std::shared_ptr<ICommandQueue> commandQueue_;
RenderPassDesc renderPassOffscreen_;
RenderPassDesc renderPassMain_;
RenderPassDesc renderPassShadow_;
std::shared_ptr<IFramebuffer> fbMain_; // swapchain
std::shared_ptr<IFramebuffer> fbOffscreen_;
std::shared_ptr<IFramebuffer> fbShadowMap_;
std::shared_ptr<IComputePipelineState> computePipelineState_Grayscale_;
std::shared_ptr<IRenderPipelineState> renderPipelineState_Mesh_;
std::shared_ptr<IRenderPipelineState> renderPipelineState_MeshWireframe_;
std::shared_ptr<IRenderPipelineState> renderPipelineState_Shadow_;
std::shared_ptr<IRenderPipelineState> renderPipelineState_Skybox_;
std::shared_ptr<IRenderPipelineState> renderPipelineState_Fullscreen_;
std::shared_ptr<IBuffer> vb0_, ib0_; // buffers for vertices and indices
std::shared_ptr<IBuffer> sbMaterials_; // storage buffer for materials
std::vector<std::shared_ptr<IBuffer>> ubPerFrame_, ubPerFrameShadow_, ubPerObject_;
std::shared_ptr<IVertexInputState> vertexInput0_;
std::shared_ptr<IVertexInputState> vertexInputShadows_;
std::shared_ptr<IDepthStencilState> depthStencilState_;
std::shared_ptr<IDepthStencilState> depthStencilStateLEqual_;
std::shared_ptr<ISamplerState> sampler;
std::shared_ptr<ISamplerState> samplerShadow;
std::shared_ptr<ITexture> textureDummyWhite_;
#if USE_OPENGL_BACKEND
std::shared_ptr<ITexture> textureDummyBlack_;
#endif // USE_OPENGL_BACKEND
std::shared_ptr<ITexture> skyboxTextureReference_;
std::shared_ptr<ITexture> skyboxTextureIrradiance_;
// scene navigation
CameraPositioner_FirstPerson positioner_(vec3(-100, 40, -47), vec3(0, 35, 0), vec3(0, 1, 0));
Camera camera(positioner_);
// NOLINTEND(facebook-static-object-destructor-check)
glm::vec2 mousePos_ = glm::vec2(0.0f);
bool mousePressed_ = false;
bool enableComputePass_ = false;
bool enableWireframe_ = false;

bool isShadowMapDirty_ = true;

struct VertexData {
  vec3 position;
  uint32_t normal{}; // Int_2_10_10_10_REV
  uint32_t uv{}; // hvec2
  uint32_t mtlIndex{};
};

// NOLINTBEGIN(facebook-static-object-destructor-check)
std::vector<VertexData> vertexData_;
std::vector<uint32_t> indexData_;
std::vector<uint32_t> shapeVertexCnt_;
// NOLINTEND(facebook-static-object-destructor-check)

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  int bDrawNormals = 0;
  int bDebugLines = 0;
  vec2 padding;
} perFrame_;

struct UniformsPerObject {
  mat4 model;
};
#define MAX_MATERIAL_NAME 128

struct CachedMaterial {
  char name[MAX_MATERIAL_NAME] = {};
  vec3 ambient = vec3(0.0f);
  vec3 diffuse = vec3(0.0f);
  char ambient_texname[MAX_MATERIAL_NAME] = {};
  char diffuse_texname[MAX_MATERIAL_NAME] = {};
  char alpha_texname[MAX_MATERIAL_NAME] = {};
};

// this goes into our GLSL shaders
struct GPUMaterial {
  vec4 ambient = vec4(0.0f);
  vec4 diffuse = vec4(0.0f);
  uint32_t texAmbient = 0;
  uint32_t texDiffuse = 0;
  uint32_t texAlpha = 0;
  uint32_t padding[1];
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

// NOLINTBEGIN(facebook-static-object-destructor-check)
std::vector<CachedMaterial> cachedMaterials_;
std::vector<GPUMaterial> materials_;
// NOLINTEND(facebook-static-object-destructor-check)

struct MaterialTextures {
  std::shared_ptr<ITexture> ambient;
  std::shared_ptr<ITexture> diffuse;
  std::shared_ptr<ITexture> alpha;
};

// NOLINTNEXTLINE(facebook-static-object-destructor-check)
std::vector<MaterialTextures> textures_; // same indexing as in materials_

struct LoadedImage {
  int w = 0;
  int h = 0;
  uint8_t* pixels = nullptr;
  int channels = 0;
  std::string debugName;
  std::string compressedFileName;
};

struct LoadedMaterial {
  size_t idx = 0;
  LoadedImage ambient;
  LoadedImage diffuse;
  LoadedImage alpha;
};

// file name -> LoadedImage
// NOLINTBEGIN(facebook-static-object-destructor-check)
std::mutex imagesCacheMutex_;
std::unordered_map<std::string, LoadedImage> imagesCache_; // accessible only from the loader thread
                                                           // pool (multiple threads)
std::unordered_map<std::string, std::shared_ptr<ITexture>> texturesCache_; // accessible only from
                                                                           // the main thread
std::vector<LoadedMaterial> loadedMaterials_;
std::mutex loadedMaterialsMutex_;
std::atomic<bool> loaderShouldExit_ = false;
std::atomic<uint32_t> remainingMaterialsToLoad_ = 0;
std::unique_ptr<tf::Executor> loaderPool_ =
    std::make_unique<tf::Executor>(std::max(2u, std::thread::hardware_concurrency() / 2));
// NOLINTEND(facebook-static-object-destructor-check)

std::string convertFileName(std::string fileName) {
  // generate compressed filename
  const std::string compressedPathPrefix = contentRootFolder;

  if (fileName.starts_with(compressedPathPrefix)) {
    // remove leading path
    fileName = fileName.substr(compressedPathPrefix.length());
  }

  std::replace(fileName.begin(), fileName.end(), ':', '_');
  std::replace(fileName.begin(), fileName.end(), '.', '_');
  std::replace(fileName.begin(), fileName.end(), '/', '_');
  std::replace(fileName.begin(), fileName.end(), '\\', '_');

  // return absolute compressed filename
  return compressedPathPrefix + fileName + ".ktx2";
}
[[maybe_unused]] void stringReplaceAll(std::string& s,
                                       const std::string& searchString,
                                       const std::string& replaceString) {
  size_t pos = 0;
  while ((pos = s.find(searchString, pos)) != std::string::npos) {
    s.replace(pos, searchString.length(), replaceString);
  }
}

GLFWwindow* FOLLY_NULLABLE initIGL(bool isHeadless, bool enableVulkanValidationLayers) {
  if (!glfwInit()) {
    printf("glfwInit() failed");
    return nullptr;
  }

#if USE_OPENGL_BACKEND
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, true);
  glfwWindowHint(GLFW_DOUBLEBUFFER, true);
  glfwWindowHint(GLFW_SRGB_CAPABLE, true);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  const char* title = "OpenGL Mesh";
#else
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  const char* title = "Vulkan Mesh";
#endif
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  GLFWwindow* window = nullptr;

  if (!isHeadless) {
    // render full screen without overlapping taskbar
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    int posX = 0;
    int posY = 0;
    int width = mode->width;
    int height = mode->height;

    glfwGetMonitorWorkarea(monitor, &posX, &posY, &width, &height);

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);

    glfwSetWindowPos(window, posX, posY);

    glfwSetErrorCallback([](int error, const char* description) {
      printf("GLFW Error (%i): %s\n", error, description);
    });

    glfwSetCursorPosCallback(window, [](auto* window, double x, double y) {
      int fbWidth = 0, fbHeight = 0;
      glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
      mousePos_ = vec2(x / fbWidth, 1.0f - y / fbHeight);
#if IGL_WITH_IGLU
      inputDispatcher.queueEvent(igl::shell::MouseMotionEvent(x, y, 0, 0));
#endif // IGL_WITH_IGLU
    });

    glfwSetMouseButtonCallback(window, [](auto* window, int button, int action, int /*mods*/) {
#if IGL_WITH_IGLU
      if (!ImGui::GetIO().WantCaptureMouse) {
#endif // IGL_WITH_IGLU
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
          mousePressed_ = (action == GLFW_PRESS);
        }
#if IGL_WITH_IGLU
      } else {
        // release the mouse
        mousePressed_ = false;
      }
      double xpos = NAN, ypos = NAN;
      glfwGetCursorPos(window, &xpos, &ypos);
      using igl::shell::MouseButton;
      const MouseButton iglButton =
          (button == GLFW_MOUSE_BUTTON_LEFT)
              ? MouseButton::Left
              : (button == GLFW_MOUSE_BUTTON_RIGHT ? MouseButton::Right : MouseButton::Middle);
      inputDispatcher.queueEvent(igl::shell::MouseButtonEvent(
          iglButton, action == GLFW_PRESS, static_cast<float>(xpos), static_cast<float>(ypos)));
#endif // IGL_WITH_IGLU
    });

    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int, int action, int mods) {
      const bool pressed = action != GLFW_RELEASE;
      if (key == GLFW_KEY_ESCAPE && pressed) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
      if (key == GLFW_KEY_N && pressed) {
        perFrame_.bDrawNormals = (perFrame_.bDrawNormals + 1) % 2;
      }
      if (key == GLFW_KEY_C && pressed) {
        enableComputePass_ = !enableComputePass_;
      }
      if (key == GLFW_KEY_T && pressed) {
        enableWireframe_ = !enableWireframe_;
      }
      if (key == GLFW_KEY_ESCAPE && pressed) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
      if (key == GLFW_KEY_W) {
        positioner_.movement_.forward_ = pressed;
      }
      if (key == GLFW_KEY_S) {
        positioner_.movement_.backward_ = pressed;
      }
      if (key == GLFW_KEY_A) {
        positioner_.movement_.left_ = pressed;
      }
      if (key == GLFW_KEY_D) {
        positioner_.movement_.right_ = pressed;
      }
      if (key == GLFW_KEY_1) {
        positioner_.movement_.up_ = pressed;
      }
      if (key == GLFW_KEY_2) {
        positioner_.movement_.down_ = pressed;
      }
      if (mods & GLFW_MOD_SHIFT) {
        positioner_.movement_.fastSpeed_ = pressed;
      }
      if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
        positioner_.movement_.fastSpeed_ = pressed;
      }
      if (key == GLFW_KEY_SPACE) {
        positioner_.setUpVector(vec3(0.0f, 1.0f, 0.0f));
      }
      if (key == GLFW_KEY_L && pressed) {
        perFrame_.bDebugLines = (perFrame_.bDebugLines + 1) % 2;
      }
    });

    glfwGetWindowSize(window, &width_, &height_);
  }

  // create a device
  {
    {
      const Result result;
#if USE_OPENGL_BACKEND
#if IGL_PLATFORM_WINDOWS
      auto ctx = std::make_unique<igl::opengl::wgl::Context>(GetDC(glfwGetWin32Window(window_)),
                                                             glfwGetWGLContext(window_));
      device_ = std::make_unique<igl::opengl::wgl::Device>(std::move(ctx));
#elif IGL_PLATFORM_LINUX
      auto ctx = std::make_unique<igl::opengl::glx::Context>(
          nullptr,
          glfwGetX11Display(),
          (igl::opengl::glx::GLXDrawable)glfwGetX11Window(window_),
          (igl::opengl::glx::GLXContext)glfwGetGLXContext(window_));
      device_ = std::make_unique<igl::opengl::glx::Device>(std::move(ctx));

#endif
#else
      const igl::vulkan::VulkanContextConfig cfg = {
          .terminateOnValidationError = false,
          .enableValidation = enableVulkanValidationLayers,
          .enableDescriptorIndexing = true,
          .headless = isHeadless,
      };
#if defined(_XLESS_GLFW_)
      auto ctx = vulkan::HWDevice::createContext(cfg, nullptr);
#elif defined(_WIN32)
      auto ctx = vulkan::HWDevice::createContext(
          cfg, window ? (void*)glfwGetWin32Window(window) : nullptr);

#elif defined(__APPLE__)
      auto ctx = vulkan::HWDevice::createContext(
          cfg, window ? (void*)glfwGetCocoaWindow(window) : nullptr);

#elif defined(__linux__)
      auto ctx = vulkan::HWDevice::createContext(
          cfg, window ? (void*)glfwGetX11Window(window) : nullptr, (void*)glfwGetX11Display());

#else
#error Unsupported OS
#endif
      if (!ctx) {
        IGL_LOG_ERROR("Vulkan 1.1 or newer is required.\n");
        return nullptr;
      }
      const HWDeviceType hardwareType = kPreferIntegratedGPU ? HWDeviceType::IntegratedGpu
                                                             : HWDeviceType::DiscreteGpu;
      std::vector<HWDeviceDesc> devices =
          vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(hardwareType), nullptr);
      if (devices.empty()) {
        const HWDeviceType fallbackHardwareType =
            !kPreferIntegratedGPU ? HWDeviceType::IntegratedGpu : HWDeviceType::DiscreteGpu;
        devices =
            vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(fallbackHardwareType), nullptr);
      }
      if (devices.empty() || cfg.headless) {
        // LavaPipe etc
        devices = vulkan::HWDevice::queryDevices(
            *ctx, HWDeviceQueryDesc(HWDeviceType::SoftwareGpu), nullptr);
      }
      if (devices.empty()) {
        IGL_LOG_ERROR("No suitable Vulkan device.\n");
        return nullptr;
      }
      device_ = vulkan::HWDevice::create(std::move(ctx),
                                         devices[0],
                                         static_cast<uint32_t>(width_),
                                         static_cast<uint32_t>(height_));
#endif
      if (!device_) {
        IGL_LOG_ERROR("Failed to create a device.\n");
        return nullptr;
      }
    }
  }

  return window;
}

namespace {
void normalizeName(std::string& name) {
#if defined(__linux__)
  std::replace(name.begin(), name.end(), '\\', '/');
#endif
}
} // namespace

bool loadAndCache(const char* cacheFileName) {
  // load 3D model and cache it
  IGL_LOG_INFO("Loading `exterior.obj`... It can take a while in debug builds...\n");

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;

  const bool ret =
      tinyobj::LoadObj(&attrib,
                       &shapes,
                       &materials,
                       &warn,
                       &err,
                       (contentRootFolder + "src/bistro/Exterior/exterior.obj").c_str(),
                       (contentRootFolder + "src/bistro/Exterior/").c_str());

  if (!IGL_DEBUG_VERIFY(ret)) {
    IGL_DEBUG_ASSERT(ret, "Did you read the tutorial at the top of this file?");
    return false;
  }

  // loop over shapes as described in https://github.com/tinyobjloader/tinyobjloader
  std::vector<std::vector<VertexData>> resplitShapes;
  std::vector<VertexData> shapeData;
  resplitShapes.resize(materials.size());
  int prevIndex = shapes[0].mesh.material_ids[0];
  for (auto& shape : shapes) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
      IGL_DEBUG_ASSERT(shape.mesh.num_face_vertices[f] == 3);

      for (size_t v = 0; v < 3; v++) {
        const tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

        const vec3 pos(attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                       attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                       attrib.vertices[3 * size_t(idx.vertex_index) + 2]);

        const bool hasNormal = (idx.normal_index >= 0);

        const vec3 normal = hasNormal ? vec3(attrib.normals[3 * size_t(idx.normal_index) + 0],
                                             attrib.normals[3 * size_t(idx.normal_index) + 1],
                                             attrib.normals[3 * size_t(idx.normal_index) + 2])
                                      : vec3(0, 0, 1);

        const bool hasUV = (idx.texcoord_index >= 0);

        const vec2 uv = hasUV ? vec2(attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                                     attrib.texcoords[2 * size_t(idx.texcoord_index) + 1])
                              : vec2(0);

        const int mtlIndex = shape.mesh.material_ids[f];

        IGL_DEBUG_ASSERT(mtlIndex >= 0 && mtlIndex < materials.size());

        if (prevIndex != mtlIndex) {
          resplitShapes[prevIndex].insert(
              resplitShapes[prevIndex].end(), shapeData.begin(), shapeData.end());
          shapeData.clear();
          prevIndex = mtlIndex;
        }
        vertexData_.push_back({pos,
                               glm::packSnorm3x10_1x2(vec4(normal, 0)),
                               glm::packHalf2x16(uv),
                               static_cast<uint32_t>(mtlIndex)});
        shapeData.push_back({pos,
                             glm::packSnorm3x10_1x2(vec4(normal, 0)),
                             glm::packHalf2x16(uv),
                             static_cast<uint32_t>(mtlIndex)});
      }
      index_offset += 3;
    }
  }
  resplitShapes[prevIndex].insert(
      resplitShapes[prevIndex].end(), shapeData.begin(), shapeData.end());
  shapeData.clear();
  for (auto shape : resplitShapes) {
    shapeData.insert(shapeData.end(), shape.begin(), shape.end());
    shapeVertexCnt_.emplace_back(static_cast<uint32_t>(shape.size()));
  }

  // repack the mesh as described in https://github.com/zeux/meshoptimizer
  {
    // 1. Generate an index buffer
    const size_t indexCount = vertexData_.size();
    std::vector<uint32_t> remap(indexCount);
    const size_t vertexCount = meshopt_generateVertexRemap(
        remap.data(), nullptr, indexCount, vertexData_.data(), indexCount, sizeof(VertexData));
    // 2. Remap vertices
    std::vector<VertexData> remappedVertices;
    indexData_.resize(indexCount);
    remappedVertices.resize(vertexCount);
    meshopt_remapIndexBuffer(indexData_.data(), nullptr, indexCount, remap.data());
    meshopt_remapVertexBuffer(
        remappedVertices.data(), vertexData_.data(), indexCount, sizeof(VertexData), remap.data());
    vertexData_ = remappedVertices;
    // 3. Optimize for the GPU vertex cache reuse and overdraw
    meshopt_optimizeVertexCache(indexData_.data(), indexData_.data(), indexCount, vertexCount);
    meshopt_optimizeOverdraw(indexData_.data(),
                             indexData_.data(),
                             indexCount,
                             &vertexData_[0].position.x,
                             vertexCount,
                             sizeof(VertexData),
                             1.05f);
    meshopt_optimizeVertexFetch(vertexData_.data(),
                                indexData_.data(),
                                indexCount,
                                vertexData_.data(),
                                vertexCount,
                                sizeof(VertexData));
  }

  // loop over materials
  for (auto& m : materials) {
    CachedMaterial mtl;
    mtl.ambient = vec3(m.ambient[0], m.ambient[1], m.ambient[2]);
    mtl.diffuse = vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
    IGL_DEBUG_ASSERT(m.name.length() < MAX_MATERIAL_NAME);
    IGL_DEBUG_ASSERT(m.ambient_texname.length() < MAX_MATERIAL_NAME);
    IGL_DEBUG_ASSERT(m.diffuse_texname.length() < MAX_MATERIAL_NAME);
    IGL_DEBUG_ASSERT(m.alpha_texname.length() < MAX_MATERIAL_NAME);
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.strcpy)
    strcat(mtl.name, m.name.c_str());
    normalizeName(m.ambient_texname);
    normalizeName(m.diffuse_texname);
    normalizeName(m.alpha_texname);
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.strcpy)
    strcat(mtl.ambient_texname, m.ambient_texname.c_str());
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.strcpy)
    strcat(mtl.diffuse_texname, m.diffuse_texname.c_str());
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.strcpy)
    strcat(mtl.alpha_texname, m.alpha_texname.c_str());
    cachedMaterials_.push_back(mtl);
  }

  IGL_LOG_INFO("Caching mesh...\n");

  FILE* cacheFile = fopen(cacheFileName, "wb");
  if (!cacheFile) {
    return false;
  }
  const uint32_t numMaterials = static_cast<uint32_t>(cachedMaterials_.size());
  const uint32_t numVertices = static_cast<uint32_t>(vertexData_.size());
  const uint32_t numIndices = static_cast<uint32_t>(indexData_.size());
  fwrite(&kMeshCacheVersion, sizeof(kMeshCacheVersion), 1, cacheFile);
  fwrite(&numMaterials, sizeof(numMaterials), 1, cacheFile);
  fwrite(&numVertices, sizeof(numVertices), 1, cacheFile);
  fwrite(&numIndices, sizeof(numIndices), 1, cacheFile);
  fwrite(cachedMaterials_.data(), sizeof(CachedMaterial), numMaterials, cacheFile);
  fwrite(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile);
  fwrite(indexData_.data(), sizeof(uint32_t), numIndices, cacheFile);
  const uint32_t numShapes = static_cast<uint32_t>(shapeData.size());
  fwrite(&numShapes, sizeof(numShapes), 1, cacheFile);
  fwrite(shapeData.data(), sizeof(VertexData), numShapes, cacheFile);
  const uint32_t numShapeVertices = static_cast<uint32_t>(shapeVertexCnt_.size());
  fwrite(&numShapeVertices, sizeof(numShapeVertices), 1, cacheFile);
  fwrite(shapeVertexCnt_.data(), sizeof(uint32_t), numShapeVertices, cacheFile);
#if USE_OPENGL_BACKEND
  vertexData_.clear();
  vertexData_.assign(shapeData.begin(), shapeData.end());
#endif
  return fclose(cacheFile) == 0;
}

bool loadFromCache(const char* cacheFileName) {
  FILE* cacheFile = fopen(cacheFileName, "rb");
  IGL_SCOPE_EXIT {
    if (cacheFile) {
      fclose(cacheFile);
    }
  };
  if (!cacheFile) {
    return false;
  }
#define CHECK_READ(expected, read) \
  if ((read) != (expected)) {      \
    return false;                  \
  }
  uint32_t versionProbe = 0;
  CHECK_READ(1, fread(&versionProbe, sizeof(versionProbe), 1, cacheFile));
  if (versionProbe != kMeshCacheVersion) {
    IGL_LOG_INFO("Cache file has wrong version id\n");
    return false;
  }
  uint32_t numMaterials = 0;
  uint32_t numVertices = 0;
  uint32_t numIndices = 0;
  CHECK_READ(1, fread(&numMaterials, sizeof(numMaterials), 1, cacheFile));
  CHECK_READ(1, fread(&numVertices, sizeof(numVertices), 1, cacheFile));
  CHECK_READ(1, fread(&numIndices, sizeof(numIndices), 1, cacheFile));
  cachedMaterials_.resize(numMaterials);
  vertexData_.resize(numVertices);
  indexData_.resize(numIndices);
  CHECK_READ(numMaterials,
             fread(cachedMaterials_.data(), sizeof(CachedMaterial), numMaterials, cacheFile));
#if !USE_OPENGL_BACKEND
  CHECK_READ(numVertices, fread(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile));
  CHECK_READ(numIndices, fread(indexData_.data(), sizeof(uint32_t), numIndices, cacheFile));
#else
  fseek(cacheFile, sizeof(VertexData) * numVertices + sizeof(uint32_t) * numIndices, SEEK_CUR);
  CHECK_READ(1, fread(&numVertices, sizeof(numVertices), 1, cacheFile));
  vertexData_.resize(numVertices);
  CHECK_READ(numVertices, fread(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile));
  uint32_t numShapeVertices = 0;
  CHECK_READ(1, fread(&numShapeVertices, sizeof(numShapeVertices), 1, cacheFile));
  shapeVertexCnt_.resize(numShapeVertices);
  CHECK_READ(numShapeVertices,
             fread(shapeVertexCnt_.data(), sizeof(uint32_t), numShapeVertices, cacheFile));
#endif
#undef CHECK_READ
  return true;
}

void initModel(int numSamplesMSAA) {
// @fb-only
  // @fb-only
      // @fb-only
// @fb-only

  {
    const TextureDesc desc = TextureDesc::new2D(igl::TextureFormat::RGBA_UNorm8,
                                                1,
                                                1,
                                                TextureDesc::TextureUsageBits::Sampled,
                                                "dummy 1x1 (white)");
    textureDummyWhite_ = device_->createTexture(desc, nullptr);
    const uint32_t pixel = 0xFFFFFFFF;
    textureDummyWhite_->upload(TextureRangeDesc::new2D(0, 0, 1, 1), &pixel);
  }

  {
#if USE_OPENGL_BACKEND
    {
      const TextureDesc desc = TextureDesc::new2D(igl::TextureFormat::RGBA_UNorm8,
                                                  1,
                                                  1,
                                                  TextureDesc::TextureUsageBits::Sampled,
                                                  "dummy 1x1 (black)");
      textureDummyBlack_ = device_->createTexture(desc, nullptr);
      const uint32_t pixel = 0xFF000000;
      textureDummyBlack_->upload(TextureRangeDesc::new2D(0, 0, 1, 1), &pixel);
    }

    const auto bufType = BufferDesc::BufferTypeBits::Uniform;
    const auto hint = BufferDesc::BufferAPIHintBits::UniformBlock;
#else
    const auto bufType = BufferDesc::BufferTypeBits::Uniform;
    const auto hint = 0;
#endif
    // create Uniform buffers to store uniforms for 2 objects
    for (uint32_t i = 0; i != kNumBufferedFrames; i++) {
      ubPerFrame_.push_back(device_->createBuffer(
          BufferDesc{.type = bufType,
                     .data = nullptr,
                     .length = sizeof(UniformsPerFrame),
                     .storage = ResourceStorage::Shared,
                     .hint = hint,
                     .debugName = "uniforms (per frame) " + std::to_string(i)},
          nullptr));
      ubPerFrameShadow_.push_back(device_->createBuffer(
          BufferDesc{.type = bufType,
                     .data = nullptr,
                     .length = sizeof(UniformsPerFrame),
                     .storage = ResourceStorage::Shared,
                     .hint = hint,
                     .debugName = "uniforms (per frame shadow) " + std::to_string(i)},
          nullptr));
      ubPerObject_.push_back(device_->createBuffer(
          BufferDesc{.type = bufType,
                     .data = nullptr,
                     .length = sizeof(UniformsPerObject),
                     .storage = ResourceStorage::Shared,
                     .hint = hint,
                     .debugName = "uniforms (per object) " + std::to_string(i)},
          nullptr));
    }
  }

  {
    const VertexInputStateDesc desc = {
        .numAttributes = 4,
        .attributes =
            {
                {
                    .format = VertexAttributeFormat::Float3,
                    .offset = offsetof(VertexData, position),
                    .name = "pos",
                    .location = 0,
                },
                {
                    .format = VertexAttributeFormat::Int_2_10_10_10_REV,
                    .offset = offsetof(VertexData, normal),
                    .name = "normal",
                    .location = 1,
                },
                {
                    .format = VertexAttributeFormat::HalfFloat2,
                    .offset = offsetof(VertexData, uv),
                    .name = "uv",
                    .location = 2,
                },
                {
                    .format = VertexAttributeFormat::UInt1,
                    .offset = offsetof(VertexData, mtlIndex),
                    .name = "mtlIndex",
                    .location = 3,
                },
            },
        .numInputBindings = 1,
        .inputBindings = {{.stride = sizeof(VertexData)}},
    };
    vertexInput0_ = device_->createVertexInputState(desc, nullptr);
  }

  {
    const VertexInputStateDesc desc = {
        .numAttributes = 1,
        .attributes = {{
            .format = VertexAttributeFormat::Float3,
            .offset = offsetof(VertexData, position),
            .name = "pos",
            .location = 0,
        }},
        .numInputBindings = 1,
        .inputBindings = {{.stride = sizeof(VertexData)}},
    };
    vertexInputShadows_ = device_->createVertexInputState(desc, nullptr);
  }

  depthStencilState_ = device_->createDepthStencilState(
      {.compareFunction = igl::CompareFunction::Less, .isDepthWriteEnabled = true}, nullptr);

  depthStencilStateLEqual_ = device_->createDepthStencilState(
      {.compareFunction = igl::CompareFunction::LessEqual, .isDepthWriteEnabled = true}, nullptr);

  {
    igl::SamplerStateDesc desc = igl::SamplerStateDesc::newLinear();
    desc.addressModeU = igl::SamplerAddressMode::Repeat;
    desc.addressModeV = igl::SamplerAddressMode::Repeat;
    desc.mipFilter = igl::SamplerMipFilter::Linear;
    desc.debugName = "Sampler: linear";
    sampler = device_->createSamplerState(desc, nullptr);

    desc.addressModeU = igl::SamplerAddressMode::Clamp;
    desc.addressModeV = igl::SamplerAddressMode::Clamp;
    desc.mipFilter = igl::SamplerMipFilter::Disabled;
    desc.debugName = "Sampler: shadow";
    desc.depthCompareEnabled = true;
    desc.depthCompareFunction = igl::CompareFunction::LessEqual;
    samplerShadow = device_->createSamplerState(desc, nullptr);
  }

  commandQueue_ = device_->createCommandQueue({}, nullptr);

  renderPassOffscreen_.colorAttachments.emplace_back();
  renderPassOffscreen_.colorAttachments.back().loadAction = LoadAction::Clear;
  renderPassOffscreen_.colorAttachments.back().storeAction =
      numSamplesMSAA > 1 ? StoreAction::MsaaResolve : StoreAction::Store;
  renderPassOffscreen_.colorAttachments.back().clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
  renderPassOffscreen_.depthAttachment.loadAction = LoadAction::Clear;
  renderPassOffscreen_.depthAttachment.storeAction = StoreAction::DontCare;
  renderPassOffscreen_.depthAttachment.clearDepth = 1.0f;

  renderPassMain_.colorAttachments.emplace_back();
  renderPassMain_.colorAttachments.back().loadAction = LoadAction::Clear;
  renderPassMain_.colorAttachments.back().storeAction = StoreAction::Store;
  renderPassMain_.colorAttachments.back().clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
  renderPassMain_.depthAttachment.loadAction = LoadAction::Clear;
  renderPassMain_.depthAttachment.storeAction = StoreAction::DontCare;
  renderPassMain_.depthAttachment.clearDepth = 1.0f;

#if USE_OPENGL_BACKEND
  renderPassShadow_.colorAttachments.push_back(igl::RenderPassDesc::ColorAttachmentDesc{});
  renderPassShadow_.colorAttachments.back().loadAction = LoadAction::Clear;
  renderPassShadow_.colorAttachments.back().storeAction = StoreAction::Store;
  renderPassShadow_.colorAttachments.back().clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
#endif
  renderPassShadow_.depthAttachment.loadAction = LoadAction::Clear;
  renderPassShadow_.depthAttachment.storeAction = StoreAction::Store;
  renderPassShadow_.depthAttachment.clearDepth = 1.0f;

  const std::string cacheFileName = contentRootFolder + "cache.data";

  if (!loadFromCache(cacheFileName.c_str())) {
    if (!IGL_DEBUG_VERIFY(loadAndCache(cacheFileName.c_str()))) {
      IGL_DEBUG_ABORT("Cannot load 3D model");
    }
  }

#if USE_OPENGL_BACKEND
  const uint32_t id = 0;
#else
  const uint32_t id = static_cast<uint32_t>(textureDummyWhite_->getTextureId());
#endif

  for (const auto& mtl : cachedMaterials_) {
    materials_.push_back(GPUMaterial{.ambient = vec4(mtl.ambient, 1.0f),
                                     .diffuse = vec4(mtl.diffuse, 1.0f),
                                     .texAmbient = id,
                                     .texDiffuse = id});
  }

  {
#if USE_OPENGL_BACKEND
    const auto bufType = BufferDesc::BufferTypeBits::Uniform;
    const auto hint = BufferDesc::BufferAPIHintBits::UniformBlock;
#else
    const auto bufType = BufferDesc::BufferTypeBits::Storage;
    const auto hint = 0;
#endif
    sbMaterials_ =
        device_->createBuffer(BufferDesc{.type = bufType,
                                         .data = materials_.data(),
                                         .length = sizeof(GPUMaterial) * materials_.size(),
                                         .storage = ResourceStorage::Private,
                                         .hint = hint,
                                         .debugName = "materials"},
                              nullptr);

    vb0_ = device_->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                            .data = vertexData_.data(),
                                            .length = sizeof(VertexData) * vertexData_.size(),
                                            .storage = ResourceStorage::Private,
                                            .hint = hint,
                                            .debugName = "vertex"},
                                 nullptr);
    ib0_ = device_->createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                            .data = indexData_.data(),
                                            .length = sizeof(uint32_t) * indexData_.size(),
                                            .storage = ResourceStorage::Private,
                                            .hint = hint,
                                            .debugName = "index"},
                                 nullptr);
  }
}

void createComputePipeline() {
  if (computePipelineState_Grayscale_) {
    return;
  }

  ComputePipelineDesc desc;
#if USE_OPENGL_BACKEND
  std::string computeCode = std::string("#version 460") + kCodeComputeTest;
  kCodeComputeTest = computeCode.c_str();
#endif
  desc.shaderStages = ShaderStagesCreator::fromModuleStringInput(
      *device_, kCodeComputeTest, "main", "Shader Module: grayscale (comp)", nullptr);

  computePipelineState_Grayscale_ = device_->createComputePipeline(desc, nullptr);
}

void createRenderPipelines(int numSamplesMSAA) {
  if (renderPipelineState_Mesh_) {
    return;
  }

  IGL_DEBUG_ASSERT(fbMain_);

  {
    RenderPipelineDesc desc;

    desc.targetDesc.colorAttachments.resize(1);
    desc.targetDesc.colorAttachments[0].textureFormat = fbMain_->getColorAttachment(0)->getFormat();

    if (fbMain_->getDepthAttachment()) {
      desc.targetDesc.depthAttachmentFormat = fbMain_->getDepthAttachment()->getFormat();
    }

    desc.vertexInputState = vertexInput0_;

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
              // @fb-only
              // @fb-only
              // @fb-only
              // @fb-only
              // @fb-only
              // @fb-only
              // @fb-only
    // @fb-only
// @fb-only
#if USE_OPENGL_BACKEND
    std::string vsCode = std::string("#version 460") + kCodeVS;
    kCodeVS = vsCode.c_str();
    std::string fsCode = std::string("#version 460") + kCodeFS;
    kCodeFS = fsCode.c_str();
#endif
    desc.shaderStages = ShaderStagesCreator::fromModuleStringInput(*device_,
                                                                   kCodeVS,
                                                                   "main",
                                                                   "Shader Module: main (vert)",
                                                                   kCodeFS,
                                                                   "main",
                                                                   "Shader Module: main (frag)",
                                                                   nullptr);
// @fb-only
#if USE_OPENGL_BACKEND
    desc.uniformBlockBindingMap.emplace(
        0,
        std::vector<std::pair<igl::NameHandle, igl::NameHandle>>{
            std::make_pair(IGL_NAMEHANDLE("PerFrame"), igl::NameHandle{})});
    desc.uniformBlockBindingMap.emplace(
        1,
        std::vector<std::pair<igl::NameHandle, igl::NameHandle>>{
            std::make_pair(IGL_NAMEHANDLE("PerObject"), igl::NameHandle{})});
    desc.uniformBlockBindingMap.emplace(
        2,
        std::vector<std::pair<igl::NameHandle, igl::NameHandle>>{
            std::make_pair(IGL_NAMEHANDLE("MeshMaterials"), igl::NameHandle{})});
#endif
    desc.cullMode = igl::CullMode::Back;
    desc.frontFaceWinding = igl::WindingMode::CounterClockwise;
    desc.sampleCount = numSamplesMSAA;
    desc.debugName = IGL_NAMEHANDLE("Pipeline: mesh");
#if USE_OPENGL_BACKEND
    desc.fragmentUnitSamplerMap[0] = IGL_NAMEHANDLE("texShadow");
    desc.fragmentUnitSamplerMap[1] = IGL_NAMEHANDLE("texAmbient");
    desc.fragmentUnitSamplerMap[2] = IGL_NAMEHANDLE("texDiffuse");
    desc.fragmentUnitSamplerMap[3] = IGL_NAMEHANDLE("texAlpha");
    desc.fragmentUnitSamplerMap[4] = IGL_NAMEHANDLE("texSkyboxIrradiance");
#endif
    renderPipelineState_Mesh_ = device_->createRenderPipeline(desc, nullptr);
    desc.polygonFillMode = igl::PolygonFillMode::Line;
    desc.vertexInputState = vertexInputShadows_; // positions-only
#if USE_OPENGL_BACKEND
    const std::string vsCodeWireframe = std::string("#version 460") + kCodeVS_Wireframe;
    kCodeVS_Wireframe = vsCodeWireframe.c_str();
    const std::string fsCodeWireframe = std::string("#version 460") + kCodeFS_Wireframe;
    kCodeFS_Wireframe = fsCodeWireframe.c_str();
    desc.fragmentUnitSamplerMap.clear();
    desc.uniformBlockBindingMap.clear();
    desc.uniformBlockBindingMap.emplace(
        0,
        std::vector<std::pair<igl::NameHandle, igl::NameHandle>>{
            std::make_pair(IGL_NAMEHANDLE("PerFrame"), igl::NameHandle{})});
    desc.uniformBlockBindingMap.emplace(
        1,
        std::vector<std::pair<igl::NameHandle, igl::NameHandle>>{
            std::make_pair(IGL_NAMEHANDLE("PerObject"), igl::NameHandle{})});
#endif
    desc.shaderStages =
        ShaderStagesCreator::fromModuleStringInput(*device_,
                                                   kCodeVS_Wireframe,
                                                   "main",
                                                   "Shader Module: main wireframe (vert)",
                                                   kCodeFS_Wireframe,
                                                   "main",
                                                   "Shader Module: main wireframe (frag)",
                                                   nullptr);
    renderPipelineState_MeshWireframe_ = device_->createRenderPipeline(desc, nullptr);
  }

  // shadow
  {
    RenderPipelineDesc desc;
    desc.targetDesc.colorAttachments.clear();
    desc.targetDesc.depthAttachmentFormat = fbShadowMap_->getDepthAttachment()->getFormat();
    desc.vertexInputState = vertexInputShadows_;
#if USE_OPENGL_BACKEND
    std::string vsCode = std::string("#version 460") + kShadowVS;
    kShadowVS = vsCode.c_str();
    std::string fsCode = std::string("#version 460") + kShadowFS;
    kShadowFS = fsCode.c_str();
#endif
    desc.shaderStages = ShaderStagesCreator::fromModuleStringInput(*device_,
                                                                   kShadowVS,
                                                                   "main",
                                                                   "Shader Module: shadow (vert)",
                                                                   kShadowFS,
                                                                   "main",
                                                                   "Shader Module: shadow (frag)",
                                                                   nullptr);
#if USE_OPENGL_BACKEND
    size_t bindingPoint = 0;
    desc.uniformBlockBindingMap[bindingPoint++].emplace_back(
        std::make_pair(IGL_NAMEHANDLE("ShadowFrameUniforms"), igl::NameHandle{}));
    desc.uniformBlockBindingMap[bindingPoint++].emplace_back(
        std::make_pair(IGL_NAMEHANDLE("ShadowObjectUniforms"), igl::NameHandle{}));
#endif
    desc.cullMode = igl::CullMode::Disabled;
    desc.debugName = IGL_NAMEHANDLE("Pipeline: shadow");
    renderPipelineState_Shadow_ = device_->createRenderPipeline(desc, nullptr);
  }

  // fullscreen
  {
    RenderPipelineDesc desc;
    desc.targetDesc.colorAttachments.resize(1);
    desc.targetDesc.colorAttachments[0].textureFormat = fbMain_->getColorAttachment(0)->getFormat();
    if (fbMain_->getDepthAttachment()) {
      desc.targetDesc.depthAttachmentFormat = fbMain_->getDepthAttachment()->getFormat();
    }
#if USE_OPENGL_BACKEND
    std::string vsCode = std::string("#version 460") + kCodeFullscreenVS;
    stringReplaceAll(vsCode, "gl_VertexIndex", "gl_VertexID");
    kCodeFullscreenVS = vsCode.c_str();
    std::string fsCode = std::string("#version 460") + kCodeFullscreenFS;
    kCodeFullscreenFS = fsCode.c_str();
#endif
    desc.shaderStages =
        ShaderStagesCreator::fromModuleStringInput(*device_,
                                                   kCodeFullscreenVS,
                                                   "main",
                                                   "Shader Module: fullscreen (vert)",
                                                   kCodeFullscreenFS,
                                                   "main",
                                                   "Shader Module: fullscreen (frag)",
                                                   nullptr);
    desc.cullMode = igl::CullMode::Disabled;
    desc.debugName = IGL_NAMEHANDLE("Pipeline: fullscreen");
    desc.fragmentUnitSamplerMap[0] = IGL_NAMEHANDLE("texFullScreen");
    renderPipelineState_Fullscreen_ = device_->createRenderPipeline(desc, nullptr);
  }
}

void createRenderPipelineSkybox(int numSamplesMSAA) {
  if (renderPipelineState_Skybox_) {
    return;
  }

  IGL_DEBUG_ASSERT(fbMain_);

  RenderPipelineDesc desc;
  desc.targetDesc.colorAttachments.resize(1);
  desc.targetDesc.colorAttachments[0].textureFormat = fbMain_->getColorAttachment(0)->getFormat();

  if (fbMain_->getDepthAttachment()) {
    desc.targetDesc.depthAttachmentFormat = fbMain_->getDepthAttachment()->getFormat();
  }

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
            // @fb-only
            // @fb-only
            // @fb-only
            // @fb-only
            // @fb-only
  // @fb-only
// @fb-only
#if USE_OPENGL_BACKEND
  std::string vsCode = std::string("#version 460") + kSkyboxVS;
  stringReplaceAll(vsCode, "gl_VertexIndex", "gl_VertexID");
  kSkyboxVS = vsCode.c_str();
  std::string fsCode = std::string("#version 460") + kSkyboxFS;
  kSkyboxFS = fsCode.c_str();
#endif
  desc.shaderStages = ShaderStagesCreator::fromModuleStringInput(*device_,
                                                                 kSkyboxVS,
                                                                 "main",
                                                                 "Shader Module: skybox (vert)",
                                                                 kSkyboxFS,
                                                                 "main",
                                                                 "Shader Module: skybox (frag)",
                                                                 nullptr);
// @fb-only
#if USE_OPENGL_BACKEND
  size_t bindingPoint = 0;
  desc.uniformBlockBindingMap[bindingPoint++].emplace_back(
      std::make_pair(IGL_NAMEHANDLE("SkyboxFrameUniforms"), igl::NameHandle{}));
#endif
  desc.cullMode = igl::CullMode::Front;
  desc.frontFaceWinding = igl::WindingMode::CounterClockwise;
  desc.sampleCount = numSamplesMSAA;
  desc.debugName = IGL_NAMEHANDLE("Pipeline: skybox");
#if USE_OPENGL_BACKEND
  desc.fragmentUnitSamplerMap[1] = IGL_NAMEHANDLE("texSkybox");
#endif
  renderPipelineState_Skybox_ = device_->createRenderPipeline(desc, nullptr);
}

std::shared_ptr<ITexture> getNativeDrawable() {
  IGL_PROFILER_FUNCTION();
  Result ret;
  std::shared_ptr<ITexture> drawable;
#if USE_OPENGL_BACKEND
#if IGL_PLATFORM_WINDOWS
  const auto& platformDevice = device_->getPlatformDevice<opengl::wgl::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDrawable(&ret);
#elif IGL_PLATFORM_LINUX
  const auto& platformDevice = device_->getPlatformDevice<opengl::glx::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDrawable(width_, height_, &ret);
#endif
#else
  const auto& platformDevice = device_->getPlatformDevice<igl::vulkan::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDrawable(&ret);
#endif
  IGL_DEBUG_ASSERT(ret.isOk(), ret.message.c_str());
  IGL_DEBUG_ASSERT(drawable != nullptr);
  return drawable;
}

std::shared_ptr<ITexture> getNativeDepthDrawable() {
  IGL_PROFILER_FUNCTION();
  Result ret;
  std::shared_ptr<ITexture> drawable;
#if USE_OPENGL_BACKEND
#if IGL_PLATFORM_WINDOWS
  const auto& platformDevice = device_->getPlatformDevice<opengl::wgl::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDepth(width_, height_, &ret);
#elif IGL_PLATFORM_LINUX
  const auto& platformDevice = device_->getPlatformDevice<opengl::glx::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDepth(width_, height_, &ret);
#endif
#else
  const auto& platformDevice = device_->getPlatformDevice<igl::vulkan::PlatformDevice>();
  IGL_DEBUG_ASSERT(platformDevice != nullptr);
  drawable = platformDevice->createTextureFromNativeDepth(width_, height_, &ret);
#endif
  IGL_DEBUG_ASSERT(ret.isOk(), ret.message.c_str());
  IGL_DEBUG_ASSERT(drawable != nullptr);
  return drawable;
}

void createFramebuffer(const std::shared_ptr<ITexture>& nativeDrawable) {
  const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = nativeDrawable}},
                                        .depthAttachment = {.texture = getNativeDepthDrawable()}};
  fbMain_ = device_->createFramebuffer(framebufferDesc, nullptr);
  IGL_DEBUG_ASSERT(fbMain_);
}

void createShadowMap() {
  const uint32_t w = 4096;
  const uint32_t h = 4096;
  auto desc = TextureDesc::new2D(igl::TextureFormat::Z_UNorm16,
                                 w,
                                 h,
                                 TextureDesc::TextureUsageBits::Attachment |
                                     TextureDesc::TextureUsageBits::Sampled,
                                 "Shadow map");
  desc.numMipLevels = TextureDesc::calcNumMipLevels(w, h);
  Result ret;
  const std::shared_ptr<ITexture> shadowMap = device_->createTexture(desc, &ret);
  IGL_DEBUG_ASSERT(ret.isOk());

  FramebufferDesc framebufferDesc;
  framebufferDesc.depthAttachment.texture = shadowMap;
  fbShadowMap_ = device_->createFramebuffer(framebufferDesc, nullptr);
  IGL_DEBUG_ASSERT(fbShadowMap_);
}

void createOffscreenFramebuffer(int numSamplesMSAA) {
  const uint32_t w = width_;
  const uint32_t h = height_;
  Result ret;
  auto descDepth = TextureDesc::new2D(igl::TextureFormat::Z_UNorm24,
                                      w,
                                      h,
                                      TextureDesc::TextureUsageBits::Attachment |
                                          TextureDesc::TextureUsageBits::Sampled,
                                      "Offscreen framebuffer (d)");
  descDepth.numMipLevels = TextureDesc::calcNumMipLevels(w, h);
  if (numSamplesMSAA > 1) {
    descDepth.usage = TextureDesc::TextureUsageBits::Attachment;
    descDepth.numSamples = numSamplesMSAA;
    descDepth.numMipLevels = 1;
    descDepth.storage = ResourceStorage::Memoryless;
  }
  const std::shared_ptr<ITexture> texDepth = device_->createTexture(descDepth, &ret);
  IGL_DEBUG_ASSERT(ret.isOk());

  TextureDesc::TextureUsage usage =
      TextureDesc::TextureUsageBits::Attachment | TextureDesc::TextureUsageBits::Sampled;
  const TextureFormat format = igl::TextureFormat::RGBA_UNorm8;
#if !USE_OPENGL_BACKEND
  usage |= TextureDesc::TextureUsageBits::Storage; // compute shader postprocessing
#endif

  auto descColor = TextureDesc::new2D(format, w, h, usage, "Offscreen framebuffer (c)");
  descColor.numMipLevels = TextureDesc::calcNumMipLevels(w, h);
  if (numSamplesMSAA > 1) {
    descColor.usage = TextureDesc::TextureUsageBits::Attachment;
    descColor.numSamples = numSamplesMSAA;
    descColor.numMipLevels = 1;
    descColor.storage = ResourceStorage::Memoryless;
  }
  const std::shared_ptr<ITexture> texColor = device_->createTexture(descColor, &ret);
  IGL_DEBUG_ASSERT(ret.isOk());

  FramebufferDesc framebufferDesc;
  framebufferDesc.colorAttachments[0].texture = texColor;
  framebufferDesc.depthAttachment.texture = texDepth;
  if (numSamplesMSAA > 1) {
    auto descColorResolve =
        TextureDesc::new2D(format, w, h, usage, "Offscreen framebuffer (c - resolve)");
    descColorResolve.usage = usage;
    const std::shared_ptr<ITexture> texResolveColor =
        device_->createTexture(descColorResolve, &ret);
    IGL_DEBUG_ASSERT(ret.isOk());
    framebufferDesc.colorAttachments[0].resolveTexture = texResolveColor;
  }
  fbOffscreen_ = device_->createFramebuffer(framebufferDesc, nullptr);
  IGL_DEBUG_ASSERT(fbOffscreen_);
}

void render(const std::shared_ptr<ITexture>& nativeDrawable,
            uint32_t frameIndex,
            int numSamplesMSAA) {
  IGL_PROFILER_FUNCTION();

  fbMain_->updateDrawable(nativeDrawable);

  // from igl/shell/renderSessions/Textured3DCubeSession.cpp
  const float fov = float(45.0f * (M_PI / 180.0f));
  const float aspectRatio = static_cast<float>(width_) / static_cast<float>(height_);

  const mat4 shadowProj = glm::perspective(float(60.0f * (M_PI / 180.0f)), 1.0f, 10.0f, 4000.0f);
  const mat4 shadowView = mat4(vec4(0.772608519f, 0.532385886f, -0.345892131f, 0),
                               vec4(0, 0.544812560f, 0.838557839f, 0),
                               vec4(0.634882748f, -0.647876859f, 0.420926809f, 0),
                               vec4(-58.9244843f, -30.4530792f, -508.410126f, 1.0f));
#if USE_OPENGL_BACKEND
  const mat4 scaleBias =
      mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0);
#else
  const mat4 scaleBias =
      mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);
#endif

  perFrame_.proj = glm::perspective(fov, aspectRatio, 0.5f, 500.0f);
  perFrame_.view = camera.getViewMatrix();
  perFrame_.light = scaleBias * shadowProj * shadowView;

  ubPerFrame_[frameIndex]->upload(&perFrame_, igl::BufferRange(sizeof(perFrame_), 0));

  {
    UniformsPerFrame perFrameShadow;
    perFrameShadow.proj = shadowProj;
    perFrameShadow.view = shadowView;
    ubPerFrameShadow_[frameIndex]->upload(&perFrameShadow,
                                          igl::BufferRange(sizeof(perFrameShadow), 0));
  }

  UniformsPerObject perObject;

  perObject.model = glm::scale(mat4(1.0f), vec3(0.05f));

  ubPerObject_[frameIndex]->upload(&perObject, igl::BufferRange(sizeof(perObject), 0));

  // Command buffers (1-N per thread): create, submit and forget

  // Pass 1: shadows
  if (isShadowMapDirty_) {
    const std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer({}, nullptr);

    auto commands = buffer->createRenderCommandEncoder(renderPassShadow_, fbShadowMap_);

    commands->bindRenderPipelineState(renderPipelineState_Shadow_);
    commands->pushDebugGroupLabel("Render Shadows", igl::Color(1, 0, 0));
    commands->bindDepthStencilState(depthStencilState_);
    commands->bindVertexBuffer(0, *vb0_);
#if USE_OPENGL_BACKEND
    const auto& glPipelineState =
        static_cast<const igl::opengl::RenderPipelineState*>(renderPipelineState_Shadow_.get());
    const int ubPerFrameShadowIdx =
        glPipelineState->getUniformBlockBindingPoint(IGL_NAMEHANDLE(("ShadowFrameUniforms")));

    const int ubPerObjectIdx =
        glPipelineState->getUniformBlockBindingPoint(IGL_NAMEHANDLE(("ShadowObjectUniforms")));
#else
    const int ubPerFrameShadowIdx = 0;
    const int ubPerObjectIdx = 1;
#endif
    commands->bindBuffer(ubPerFrameShadowIdx, ubPerFrameShadow_[frameIndex].get());
    commands->bindBuffer(ubPerObjectIdx, ubPerObject_[frameIndex].get());

#if USE_OPENGL_BACKEND
    int start = 0;
    for (auto numVertices : shapeVertexCnt_) {
      commands->draw(numVertices, 1, start);
      start += numVertices;
    }
#else
    commands->bindIndexBuffer(*ib0_, igl::IndexFormat::UInt32);
    commands->drawIndexed(indexData_.size());
#endif
    commands->popDebugGroupLabel();
    commands->endEncoding();

    buffer->present(fbShadowMap_->getDepthAttachment());

    commandQueue_->submit(*buffer);

    fbShadowMap_->getDepthAttachment()->generateMipmap(*commandQueue_);

    isShadowMapDirty_ = false;
  }

  // Pass 2: mesh
  {
    const std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer({}, nullptr);

    // This will clear the framebuffer
    auto commands = buffer->createRenderCommandEncoder(renderPassOffscreen_, fbOffscreen_);
    // Scene
    commands->bindRenderPipelineState(renderPipelineState_Mesh_);
    commands->pushDebugGroupLabel("Render Mesh", igl::Color(1, 0, 0));
    commands->bindDepthStencilState(depthStencilState_);
    commands->bindVertexBuffer(0, *vb0_);

#if USE_OPENGL_BACKEND
    const auto& glPipelineState =
        static_cast<const igl::opengl::RenderPipelineState*>(renderPipelineState_Mesh_.get());
    const int ubPerFrameIdx =
        glPipelineState->getUniformBlockBindingPoint(IGL_NAMEHANDLE(("PerFrame")));

    const int ubPerObjectIdx =
        glPipelineState->getUniformBlockBindingPoint(IGL_NAMEHANDLE(("PerObject")));

    const int sbIdx =
        glPipelineState->getUniformBlockBindingPoint(IGL_NAMEHANDLE(("MeshMaterials")));
#else
    const int ubPerFrameIdx = 0;
    const int ubPerObjectIdx = 1;
    const int sbIdx = 2;
#endif
    commands->bindBuffer(ubPerFrameIdx, ubPerFrame_[frameIndex].get());
    commands->bindBuffer(ubPerObjectIdx, ubPerObject_[frameIndex].get());
    commands->bindBuffer(sbIdx, sbMaterials_.get());
    commands->bindTexture(0, igl::BindTarget::kFragment, fbShadowMap_->getDepthAttachment().get());
    commands->bindTexture(4, igl::BindTarget::kFragment, skyboxTextureIrradiance_.get());
    commands->bindSamplerState(0, igl::BindTarget::kFragment, samplerShadow.get());
    commands->bindSamplerState(1, igl::BindTarget::kFragment, sampler.get());
    commands->bindSamplerState(2, igl::BindTarget::kFragment, sampler.get());
    commands->bindSamplerState(3, igl::BindTarget::kFragment, sampler.get());
    commands->bindSamplerState(4, igl::BindTarget::kFragment, sampler.get());

#if USE_OPENGL_BACKEND
    commands->bindVertexBuffer(0, *vb0_);
    int shapeStart = 0;
    for (auto numVertices : shapeVertexCnt_) {
      const uint32_t imageIdx = static_cast<uint32_t>(vertexData_[shapeStart].mtlIndex);
      const auto ambientTextureReference =
          strstr(cachedMaterials_[imageIdx].name, "MASTER_Glass_") ? textureDummyWhite_
          : textures_[imageIdx].ambient                            ? textures_[imageIdx].ambient
                                                                   : textureDummyBlack_;
      const auto diffuseTextureReference =
          strstr(cachedMaterials_[imageIdx].name, "MASTER_Glass_Clean") ? textureDummyWhite_
          : textures_[imageIdx].diffuse ? textures_[imageIdx].diffuse
                                        : textureDummyBlack_;
      const auto alphaTextureReference = strstr(cachedMaterials_[imageIdx].name, "MASTER_Glass_")
                                             ? textureDummyWhite_
                                         : textures_[imageIdx].alpha ? textures_[imageIdx].alpha
                                                                     : textureDummyBlack_;

      commands->bindTexture(1, igl::BindTarget::kFragment, ambientTextureReference.get());
      commands->bindTexture(2, igl::BindTarget::kFragment, diffuseTextureReference.get());
      commands->bindTexture(3, igl::BindTarget::kFragment, alphaTextureReference.get());
      commands->draw(numVertices, 1, shapeStart);
      if (enableWireframe_) {
        commands->bindRenderPipelineState(renderPipelineState_MeshWireframe_);
        commands->bindVertexBuffer(0, *vb0_);
        commands->draw(numVertices, 1, shapeStart);

        // Bind the non-wireframe pipeline and the vertex buffer
        commands->bindRenderPipelineState(renderPipelineState_Mesh_);
        commands->bindVertexBuffer(0, *vb0_);
      }

      shapeStart += numVertices;
    }
#else
    commands->bindTexture(0, igl::BindTarget::kFragment, fbShadowMap_->getDepthAttachment().get());
    commands->bindTexture(1, igl::BindTarget::kFragment, skyboxTextureIrradiance_.get());
    commands->bindSamplerState(0, igl::BindTarget::kFragment, samplerShadow.get());
    commands->bindSamplerState(1, igl::BindTarget::kFragment, sampler.get());
    commands->bindIndexBuffer(*ib0_, IndexFormat::UInt32);
    commands->drawIndexed(indexData_.size());
    if (enableWireframe_) {
      commands->bindRenderPipelineState(renderPipelineState_MeshWireframe_);
      commands->drawIndexed(indexData_.size());
    }
#endif
    commands->popDebugGroupLabel();

    // Skybox
    commands->bindRenderPipelineState(renderPipelineState_Skybox_);
    commands->bindTexture(1, igl::BindTarget::kFragment, skyboxTextureReference_.get());
    commands->bindSamplerState(1, igl::BindTarget::kFragment, sampler.get());
    commands->pushDebugGroupLabel("Render Skybox", igl::Color(0, 1, 0));
    commands->bindDepthStencilState(depthStencilStateLEqual_);
    commands->draw(3u * 6u * 2u);
    commands->popDebugGroupLabel();
    commands->endEncoding();

#if !USE_OPENGL_BACKEND
    buffer->present(fbOffscreen_->getColorAttachment(0));
#endif
    commandQueue_->submit(*buffer);
#if USE_OPENGL_BACKEND
    if (kNumSamplesMSAA == 1) {
      fbOffscreen_->getColorAttachment(0)->generateMipmap(*commandQueue_.get());
    }
#endif
  }

  // Pass 3: compute shader post-processing
  if (enableComputePass_) {
    const std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer(
        CommandBufferDesc{.debugName = "computeBuffer"}, nullptr);

    auto commands = buffer->createComputeCommandEncoder();
    commands->bindComputePipelineState(computePipelineState_Grayscale_);
    ITexture* tex = numSamplesMSAA > 1 ? fbOffscreen_->getResolveColorAttachment(0).get()
                                       : fbOffscreen_->getColorAttachment(0).get();
#if !USE_OPENGL_BACKEND
    const uint32_t textureId = tex->getTextureId();
    commands->bindPushConstants(&textureId, sizeof(textureId));
#endif
    commands->bindTexture(0, tex);
    commands->dispatchThreadGroups(igl::Dimensions(width_, height_, 1), igl::Dimensions());
    commands->endEncoding();

    commandQueue_->submit(*buffer);
  }

  // Pass 4: render into the swapchain image
  {
    const std::shared_ptr<ICommandBuffer> buffer = commandQueue_->createCommandBuffer({}, nullptr);

    // This will clear the framebuffer
    auto commands = buffer->createRenderCommandEncoder(renderPassMain_, fbMain_);
    commands->bindRenderPipelineState(renderPipelineState_Fullscreen_);
    commands->pushDebugGroupLabel("Swapchain Output", igl::Color(1, 0, 0));
    commands->bindTexture(0,
                          igl::BindTarget::kFragment,
                          numSamplesMSAA > 1 ? fbOffscreen_->getResolveColorAttachment(0).get()
                                             : fbOffscreen_->getColorAttachment(0).get());
    commands->bindSamplerState(0, igl::BindTarget::kFragment, sampler.get());
    commands->draw(3);
    commands->popDebugGroupLabel();

#if IGL_WITH_IGLU
    imguiSession->endFrame(*device_, *commands);
#endif // IGL_WITH_IGLU

    commands->endEncoding();

    buffer->present(fbMain_->getColorAttachment(0));

    commandQueue_->submit(*buffer);
  }

#if !USE_OPENGL_BACKEND
  fbMain_->getDepthAttachment()->generateMipmap(*commandQueue_);
#endif
}

void generateCompressedTexture(const LoadedImage& img) {
  if (loaderShouldExit_.load(std::memory_order_acquire)) {
    return;
  }

  printf("...compressing texture to %s\n", img.compressedFileName.c_str());

  const auto numMipLevels = TextureDesc::calcNumMipLevels(img.w, img.h);

  // Go over all generated mipmap and create a compressed texture

  // Create ktx2 texture hard coded and support only BC7 format
  ktxTextureCreateInfo createInfo = {
      .vkFormat = VK_FORMAT_BC7_UNORM_BLOCK,
      .baseWidth = static_cast<uint32_t>(img.w),
      .baseHeight = static_cast<uint32_t>(img.h),
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = numMipLevels,
      .numLayers = 1u,
      .numFaces = 1u,
      .generateMipmaps = KTX_FALSE, // Mipmaps are explicitly provided
  };

  ktxTexture2* texture = nullptr;
  auto error = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
  IGL_DEBUG_ASSERT(error == KTX_SUCCESS);

  IGL_SCOPE_EXIT {
    ktxTexture_Destroy(ktxTexture(texture));
  };

  uint32_t w = img.w;
  uint32_t h = img.h;
  for (uint32_t i = 0; i < numMipLevels; ++i) {
    std::vector<uint8_t> destPixels(w * h * img.channels);

    // resize
    stbir_resize_uint8(img.pixels,
                       static_cast<int>(img.w),
                       static_cast<int>(img.h),
                       0,
                       destPixels.data(),
                       w,
                       h,
                       0,
                       static_cast<int>(img.channels));
    // compress
    auto packedImage16 = Compress::getCompressedImage(
        destPixels.data(), w, h, img.channels, false, &loaderShouldExit_);
    ktxTexture_SetImageFromMemory(ktxTexture(texture),
                                  i,
                                  0,
                                  0,
                                  reinterpret_cast<const uint8_t*>(packedImage16.data()),
                                  sizeof(block16) * packedImage16.size());

    h = h > 1 ? h >> 1 : 1;
    w = w > 1 ? w >> 1 : 1;

    if (loaderShouldExit_.load(std::memory_order_acquire)) {
      return;
    }
  }

  ktxTexture_WriteToNamedFile(ktxTexture(texture), img.compressedFileName.c_str());
}

LoadedImage loadImage(const char* fileName, int channels) {
  if (!fileName || !*fileName) {
    return {};
  }

  const std::string debugName = IGL_FORMAT("{} ({})", fileName, channels).c_str();

  {
    const std::lock_guard lock(imagesCacheMutex_);

    const auto it = imagesCache_.find(debugName);

    if (it != imagesCache_.end()) {
      IGL_DEBUG_ASSERT(channels == it->second.channels);
      return it->second;
    }
  }

  LoadedImage img;
  img.compressedFileName = convertFileName(fileName);
  img.pixels = stbi_load(fileName, &img.w, &img.h, nullptr, channels);
  img.channels = channels;
  img.debugName = debugName;

  if ((img.pixels != nullptr) && kEnableCompression && (channels != 1) &&
      !std::filesystem::exists(img.compressedFileName.c_str())) {
    generateCompressedTexture(img);
  }

  const std::lock_guard lock(imagesCacheMutex_);

  imagesCache_[fileName] = img;

  return img;
}

void loadMaterial(size_t i) {
  // NOLINTNEXTLINE(facebook-static-object-destructor-check)
  static const std::string pathPrefix = contentRootFolder + "src/bistro/Exterior/";

  IGL_SCOPE_EXIT {
    remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
  };

#define LOAD_TEX(result, tex, channels)                                                     \
  std::string tmp##result = cachedMaterials_[i].tex;                                        \
  std::replace(tmp##result.begin(), tmp##result.end(), '\\', '/');                          \
  const LoadedImage result = std::string(cachedMaterials_[i].tex).empty()                   \
                                 ? LoadedImage()                                            \
                                 : loadImage((pathPrefix + tmp##result).c_str(), channels); \
  if (loaderShouldExit_.load(std::memory_order_acquire)) {                                  \
    return;                                                                                 \
  }

  LOAD_TEX(ambient, ambient_texname, 4);
  LOAD_TEX(diffuse, diffuse_texname, 4);
  LOAD_TEX(alpha, alpha_texname, 1);

#undef LOAD_TEX

  const LoadedMaterial mtl{.idx = i, .ambient = ambient, .diffuse = diffuse, .alpha = alpha};

  if (!mtl.ambient.pixels && !mtl.diffuse.pixels) {
    // skip missing textures
    materials_[i].texDiffuse = 0;
  } else {
    const std::lock_guard guard(loadedMaterialsMutex_);
    loadedMaterials_.push_back(mtl);
    remainingMaterialsToLoad_.fetch_add(1u, std::memory_order_release);
  }
}

void loadMaterials(bool isHeadless) {
  stbi_set_flip_vertically_on_load(1);

  textures_.resize(cachedMaterials_.size());

  remainingMaterialsToLoad_ = static_cast<uint32_t>(cachedMaterials_.size());

  for (size_t i = 0; i != cachedMaterials_.size(); i++) {
    loaderPool_->silent_async([i]() { loadMaterial(i); });
  }
}

void loadCubemapTexture(const std::string& fileNameKTX, std::shared_ptr<ITexture>& tex) {
  if (!std::filesystem::exists(fileNameKTX)) {
    return;
  }

#if USE_TEXTURE_LOADER
  loadKtxTexture(*device_, *commandQueue_, fileNameKTX, tex, !kEnableCompression);
#else
  ktxTexture2* texture = nullptr;
  auto error = ktxTexture2_CreateFromNamedFile(
      fileNameKTX.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  IGL_DEBUG_ASSERT(error == KTX_SUCCESS);

  IGL_SCOPE_EXIT {
    ktxTexture_Destroy(ktxTexture(texture));
  };

  if (!IGL_DEBUG_VERIFY(texture->vkFormat == VK_FORMAT_R32G32B32A32_SFLOAT)) {
    IGL_DEBUG_ABORT("Texture format not supported");
    return;
  }

  auto texRefRange = TextureRangeDesc::new2D(
      0, 0, static_cast<size_t>(texture->baseWidth), static_cast<size_t>(texture->baseHeight));

  // If compression is enabled, upload all mip levels
  if (kEnableCompression) {
    texRefRange.numMipLevels = TextureDesc::calcNumMipLevels(texRefRange.width, texRefRange.height);
  }

  auto desc =
      TextureDesc::newCube(igl::vulkan::util::vkTextureFormatToTextureFormat(texture->vkFormat),
                           texRefRange.width,
                           texRefRange.height,
                           TextureDesc::TextureUsageBits::Sampled,
                           fileNameKTX.c_str());
  desc.numMipLevels = texRefRange.numMipLevels;
  tex = device_->createTexture(desc, nullptr);
  IGL_DEBUG_ASSERT(tex);
  for (uint8_t face = 0; face < 6; ++face) {
    for (size_t i = 0; i < desc.numMipLevels; ++i) {
      size_t offset = 0;
      error = ktxTexture_GetImageOffset(ktxTexture(texture), i, 0, face, &offset);
      IGL_DEBUG_ASSERT(error == KTX_SUCCESS);

      tex->upload(tex->getCubeFaceRange(face, i), texture->pData + offset);
    }
  }

  if (!kEnableCompression) {
    tex->generateMipmap(*commandQueue_);
  }
#endif // IGL_WITH_IGLU
}

ktxTexture2* FOLLY_NULLABLE bitmapToCube(Bitmap& bmp) {
  IGL_DEBUG_ASSERT(bmp.comp_ == 3); // RGB
  IGL_DEBUG_ASSERT(bmp.type_ == eBitmapType_Cube);
  IGL_DEBUG_ASSERT(bmp.fmt_ == eBitmapFormat_Float);

  const int w = bmp.w_;
  const int h = bmp.h_;

  const auto numMipLevels = TextureDesc::calcNumMipLevels(w, h);

  // Create ktx2 texture
  // hard coded and support only BC7 format
  ktxTextureCreateInfo createInfo = {
      .vkFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
      .baseWidth = static_cast<uint32_t>(w),
      .baseHeight = static_cast<uint32_t>(h),
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = numMipLevels,
      .numLayers = 1u,
      .numFaces = 6u,
      .generateMipmaps = KTX_FALSE, // Mipmaps are explicitly provided
  };

  ktxTexture2* texture = nullptr;
  auto error = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
  IGL_DEBUG_ASSERT(error == KTX_SUCCESS);

  const size_t numFacePixels = static_cast<size_t>(w) * static_cast<size_t>(h);

  const std::unique_ptr<float[]> facePixels =
      std::make_unique<float[]>(numFacePixels * static_cast<size_t>(4));
  for (size_t face = 0; face != 6; face++) {
    float* dst = facePixels.get();
    const vec3* src = reinterpret_cast<vec3*>(bmp.data_.data()) + face * numFacePixels;
    for (int y = 0; y != h; y++) {
      for (int x = 0; x != w; x++) {
        const vec3& rgb = src[x + y * w];
        *dst++ = rgb.x;
        *dst++ = rgb.y;
        *dst++ = rgb.z;
        *dst++ = 0.0f;
      }
    }
    ktxTexture_SetImageFromMemory(ktxTexture(texture),
                                  0,
                                  0,
                                  face,
                                  reinterpret_cast<const uint8_t*>(facePixels.get()),
                                  sizeof(float) * numFacePixels * static_cast<size_t>(4));
  }

  return texture;
}

void generateMipmaps(const std::string& outFilename, ktxTexture2* cubemap) {
  IGL_LOG_INFO("Generating mipmaps");

  const std::unique_ptr<float[]> buffer =
      std::make_unique<float[]>(static_cast<size_t>(cubemap->baseWidth) *
                                static_cast<size_t>(cubemap->baseHeight) * static_cast<size_t>(4));
  auto prevWidth = cubemap->baseWidth;
  auto prevHeight = cubemap->baseHeight;
  for (size_t face = 0; face < 6; ++face) {
    IGL_LOG_INFO(".");
    for (size_t miplevel = 1; miplevel < cubemap->numLevels; ++miplevel) {
      IGL_LOG_INFO(":");
      const auto width = prevWidth > 1 ? prevWidth >> 1 : 1;
      const auto height = prevHeight > 1 ? prevWidth >> 1 : 1;

      size_t prevOffset = 0;
      auto error =
          ktxTexture_GetImageOffset(ktxTexture(cubemap), miplevel - 1, 0, face, &prevOffset);
      IGL_DEBUG_ASSERT(error == KTX_SUCCESS);

      stbir_resize_float(reinterpret_cast<const float*>(cubemap->pData + prevOffset),
                         prevWidth,
                         prevHeight,
                         0,
                         buffer.get(),
                         width,
                         height,
                         0,
                         4);

      ktxTexture_SetImageFromMemory(ktxTexture(cubemap),
                                    miplevel,
                                    0,
                                    face,
                                    reinterpret_cast<const uint8_t*>(buffer.get()),
                                    sizeof(float) * width * height * 4);

      prevWidth = width;
      prevHeight = height;
    }
    prevWidth = cubemap->baseWidth;
    prevHeight = cubemap->baseHeight;
  }

  IGL_LOG_INFO("\n");
  ktxTexture_WriteToNamedFile(ktxTexture(cubemap), outFilename.c_str());
}

void processCubemap(const std::string& inFilename,
                    const std::string& outFilenameEnv,
                    const std::string& outFilenameIrr) {
  int sourceWidth = 0, sourceHeight = 0;
  float* pxs = stbi_loadf(inFilename.c_str(), &sourceWidth, &sourceHeight, nullptr, 3);
  IGL_SCOPE_EXIT {
    if (pxs) {
      stbi_image_free(pxs);
    }
  };

  if (!IGL_DEBUG_VERIFY(pxs != nullptr)) {
    IGL_DEBUG_ABORT("Did you read the tutorial at the top of Tiny_MeshLarge.cpp?");
    return;
  }

  // Environment map
  {
    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(
        Bitmap(sourceWidth, sourceHeight, 3, eBitmapFormat_Float, pxs));
    ktxTexture2* cube = bitmapToCube(bmp);
    // NOLINTNEXTLINE(facebook-hte-NullableDereference)
    generateMipmaps(outFilenameEnv, cube);
  }

  // Irradiance map
  {
    constexpr int dstW = 256;
    constexpr int dstH = 128;

    std::vector<vec3> out(dstW * dstH);
    convolveDiffuse((vec3*)pxs, sourceWidth, sourceHeight, dstW, dstH, out.data(), 1024);

    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(
        Bitmap(dstW, dstH, 3, eBitmapFormat_Float, out.data()));
    ktxTexture2* cube = bitmapToCube(bmp);
    // NOLINTNEXTLINE(facebook-hte-NullableDereference)
    generateMipmaps(outFilenameIrr, cube);

    IGL_SCOPE_EXIT {
      ktxTexture_Destroy(ktxTexture(cube));
    };
  }
}

void loadSkyboxTexture() {
  // NOLINTBEGIN(facebook-static-object-destructor-check)
  static const std::string skyboxFileName{"immenstadter_horn_2k"};
  static const std::string skyboxSubdir{"src/skybox_hdr/"};

  // NOLINTNEXTLINE(facebook-static-object-destructor-check)
  static const std::string fileNameRefKTX =
      contentRootFolder + skyboxFileName + "_ReferenceMap.ktx2";
  static const std::string fileNameIrrKTX =
      contentRootFolder + skyboxFileName + "_IrradianceMap.ktx2";
  // NOLINTEND(facebook-static-object-destructor-check)

  if (!std::filesystem::exists(fileNameRefKTX) || !std::filesystem::exists(fileNameIrrKTX)) {
    IGL_LOG_INFO("Cubemap in KTX format not found. Extracting from HDR file...\n");
    // NOLINTNEXTLINE(facebook-static-object-destructor-check)
    static const std::string inFilename =
        contentRootFolder + skyboxSubdir + skyboxFileName + ".hdr";

    processCubemap(inFilename, fileNameRefKTX, fileNameIrrKTX);
  }

  loadCubemapTexture(fileNameRefKTX, skyboxTextureReference_);
  loadCubemapTexture(fileNameIrrKTX, skyboxTextureIrradiance_);
}

std::shared_ptr<ITexture> createTexture(const LoadedImage& img) {
  if (!img.pixels) {
    return nullptr;
  }

  const auto it = texturesCache_.find(img.debugName);

  if (it != texturesCache_.end()) {
    return it->second;
  }

  igl::TextureFormat fmt = igl::TextureFormat::Invalid;
  if (img.channels == 1) {
    fmt = igl::TextureFormat::R_UNorm8;
  } else if (img.channels == 4) {
    fmt = kEnableCompression ? igl::TextureFormat::RGBA_BC7_UNORM_4x4
                             : igl::TextureFormat::RGBA_UNorm8;
  }

  TextureDesc desc = TextureDesc::new2D(
      fmt, img.w, img.h, TextureDesc::TextureUsageBits::Sampled, img.debugName.c_str());
  desc.numMipLevels = TextureDesc::calcNumMipLevels(img.w, img.h);
  auto tex = device_->createTexture(desc, nullptr);

  if (kEnableCompression && img.channels == 4 &&
      std::filesystem::exists(img.compressedFileName.c_str())) {
#if USE_TEXTURE_LOADER
    loadKtxTexture(*device_, *commandQueue_, img.compressedFileName, tex, false);
#else
    // Uploading the texture
    const auto rangeDesc = TextureRangeDesc::new2D(0, 0, img.w, img.h, 0, desc.numMipLevels);
    ktxTexture* texture = nullptr;
    auto error = ktxTexture_CreateFromNamedFile(
        img.compressedFileName.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
    if (IGL_DEBUG_VERIFY_NOT(error != KTX_SUCCESS)) {
      printf("Failed to load %s\n", img.compressedFileName.c_str());
    }
    IGL_SCOPE_EXIT {
      ktxTexture_Destroy(ktxTexture(texture));
    };

    for (size_t i = 0; i < desc.numMipLevels; ++i) {
      size_t offset = 0;
      error = ktxTexture_GetImageOffset(ktxTexture(texture), i, 0, 0, &offset);
      IGL_DEBUG_ASSERT(error == KTX_SUCCESS);

      tex->upload(rangeDesc.atMipLevel(i), texture->pData + offset);
    }
#endif // USE_TEXTURE_LOADER
  } else {
    tex->upload(TextureRangeDesc::new2D(0, 0, img.w, img.h), img.pixels);
    tex->generateMipmap(*commandQueue_);
  }
  texturesCache_[img.debugName] = tex;
  return tex;
}

void processLoadedMaterials() {
  int numLoaded = 0;

  const int kMaxMaterialsPerFrame = 5;

  for (; numLoaded < kMaxMaterialsPerFrame; numLoaded++) {
    LoadedMaterial mtl;

    {
      const std::lock_guard guard(loadedMaterialsMutex_);
      if (loadedMaterials_.empty()) {
        break;
      } else {
        mtl = loadedMaterials_.back();
        loadedMaterials_.pop_back();
        remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
      }
    }

    MaterialTextures tex;

    tex.ambient = createTexture(mtl.ambient);
    tex.diffuse = createTexture(mtl.diffuse);
    tex.alpha = createTexture(mtl.alpha);

    // update GPU materials
    textures_[mtl.idx] = tex;
#if !USE_OPENGL_BACKEND
    materials_[mtl.idx].texAmbient =
        tex.ambient ? static_cast<uint32_t>(tex.ambient->getTextureId()) : 0;
    materials_[mtl.idx].texDiffuse =
        tex.diffuse ? static_cast<uint32_t>(tex.diffuse->getTextureId()) : 0;
    materials_[mtl.idx].texAlpha = tex.alpha ? static_cast<uint32_t>(tex.alpha->getTextureId()) : 0;
    IGL_DEBUG_ASSERT(materials_[mtl.idx].texAmbient >= 0);
    IGL_DEBUG_ASSERT(materials_[mtl.idx].texDiffuse >= 0);
    IGL_DEBUG_ASSERT(materials_[mtl.idx].texAlpha >= 0);
#endif
  }
  if (numLoaded) {
    sbMaterials_->upload(materials_.data(), BufferRange(sizeof(GPUMaterial) * materials_.size()));
  }
}

} // namespace

int main(int argc, char* argv[]) {
  bool isHeadless = false;
  bool enableVulkanValidationLayers = true;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--headless")) {
      isHeadless = true;
    } else if (!strcmp(argv[i], "--disable-vulkan-validation-layers")) {
      enableVulkanValidationLayers = false;
    }
  }

#if defined(IGL_USE_STATIC_LAVAPIPE)
  const int kNumSamplesMSAA = 1;
#elif defined(IGL_USE_STATIC_KOSMICKRISP)
  // KosmicKrisp maps Vulkan MSAA onto Metal. Apple GPUs support at most 4x MSAA, so requesting
  // 8x triggers an MTLTextureDescriptor validation assertion ("sampleCount (8) is not supported
  // by device"). Cap MSAA at 4x on KosmicKrisp.
  const int kNumSamplesMSAA = isHeadless ? 1 : 4;
#else
  const int kNumSamplesMSAA = isHeadless ? 1 : 8;
#endif
  // find the content folder
  {
    using namespace std::filesystem;
    path subdir("third-party/content/");
    // @fb-only
    path dir = current_path();
    // find the content somewhere above our current build directory
    while (dir != current_path().root_path() && !exists(dir / subdir)) {
      dir = dir.parent_path();
    }
    if (!exists(dir / subdir)) {
      // Use stderr (unbuffered) + trailing newline + explicit flush so the message survives the
      // immediate process termination below. A buffered stdout printf without a newline is lost
      // when IGL_DEBUG_ASSERT_NOT_REACHED() aborts (or the early return exits) before the C
      // runtime flushes stdout, which makes the failure look like a silent crash.
      fprintf(stderr,
              "Cannot find the content directory '%s'. Run `deploy_content.py` before running this "
              "app, and launch it from the repository root.\n",
              subdir.string().c_str());
      fflush(stderr);
      IGL_DEBUG_ASSERT_NOT_REACHED();
      return EXIT_FAILURE;
    }
    contentRootFolder = (dir / subdir).string();
  }

  GLFWwindow* window = initIGL(isHeadless, enableVulkanValidationLayers);
  if (!device_) {
    glfwTerminate();
    return EXIT_FAILURE;
  }
  initModel(kNumSamplesMSAA);

  if (kEnableCompression) {
    printf(
        "Compressing textures... It can take a while in debug builds...(needs to be done once)\n");
  }

  loadSkyboxTexture();
  loadMaterials(isHeadless);

  createFramebuffer(getNativeDrawable());
  createShadowMap();
  createOffscreenFramebuffer(kNumSamplesMSAA);
  createRenderPipelines(kNumSamplesMSAA);
  createRenderPipelineSkybox(kNumSamplesMSAA);
  createComputePipeline();

#if IGL_WITH_IGLU
  imguiSession = std::make_unique<iglu::imgui::Session>(*device_, inputDispatcher);
#endif // IGL_WITH_IGLU

  // In headless mode, wait for all textures to be loaded before rendering
  if (isHeadless) {
    printf("Waiting for all textures to load...\n");
    while (remainingMaterialsToLoad_.load(std::memory_order_acquire) > 0) {
      processLoadedMaterials();
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    printf("All textures loaded.\n");
  }

  double prevTime = glfwGetTime();

  uint32_t frameIndex = 0;

  // Main loop
  while (!window || !glfwWindowShouldClose(window)) {
    {
      const FramebufferDesc framebufferDesc{
          .colorAttachments = {{.texture = getNativeDrawable()}},
          .depthAttachment = {.texture = getNativeDepthDrawable()}};
#if IGL_WITH_IGLU
      imguiSession->beginFrame(framebufferDesc, 1.0f);
      ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
      ImGui::ShowDemoWindow();

      ImGui::Begin("Keyboard hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::Text("W/S/A/D - camera movement");
      ImGui::Text("1/2 - camera up/down");
      ImGui::Text("Shift - fast movement");
      ImGui::Text("C - toggle compute shader postprocessing");
      ImGui::Text("N - toggle normals");
      ImGui::Text("T - toggle wireframe");
      ImGui::End();

      if (textures_[1].diffuse) {
        ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Image(ImTextureID(textures_[1].diffuse.get()), ImVec2(256, 256));
        ImGui::End();
      }

      if (const uint32_t num = remainingMaterialsToLoad_.load(std::memory_order_acquire)) {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::Begin(
            "Loading...", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs);
        ImGui::ProgressBar(1.0f - float(num) / cachedMaterials_.size(),
                           ImVec2(ImGui::GetIO().DisplaySize.x, 32));
        ImGui::End();
      }

      imguiSession->drawFPS(fps_.getAverageFPS());
#endif // IGL_WITH_IGLU
    }

    processLoadedMaterials();
    const double newTime = glfwGetTime();
    const double delta = newTime - prevTime;
    fps_.updateFPS(delta);
    positioner_.update(delta, mousePos_, mousePressed_);
    prevTime = newTime;
#if IGL_WITH_IGLU
    inputDispatcher.processEvents();
#endif // IGL_WITH_IGLU
    render(getNativeDrawable(), frameIndex, kNumSamplesMSAA);
    frameIndex = (frameIndex + 1) % kNumBufferedFrames;
    if (window) {
      glfwPollEvents();
    } else {
      printf("We are running headless - breaking after 1 frame\n");
      std::shared_ptr<ITexture> texture = fbMain_->getColorAttachment(0);
      const Dimensions dim = texture->getDimensions();
      std::vector<uint8_t> pixelsRGBA(dim.width * dim.height * 4);
      std::vector<uint8_t> pixelsRGB(dim.width * dim.height * 3);
      fbMain_->copyBytesColorAttachment(*commandQueue_,
                                        0,
                                        pixelsRGBA.data(),
                                        TextureRangeDesc::new2D(0, 0, dim.width, dim.height));
      if (texture->getFormat() == igl::TextureFormat::BGRA_UNorm8 ||
          texture->getFormat() == igl::TextureFormat::BGRA_SRGB) {
        // swap R-B
        for (uint32_t i = 0; i < pixelsRGBA.size(); i += 4) {
          std::swap(pixelsRGBA[i + 0], pixelsRGBA[i + 2]);
        }
      }
      // convert to RGB
      for (uint32_t i = 0; i < pixelsRGB.size() / 3; i++) {
        pixelsRGB[3 * i + 0] = pixelsRGBA[4 * i + 0];
        pixelsRGB[3 * i + 1] = pixelsRGBA[4 * i + 1];
        pixelsRGB[3 * i + 2] = pixelsRGBA[4 * i + 2];
      }
      const char* fileName = "TinyMeshLarge.png";
      IGLLog(IGLLogInfo, "Writing screenshot to: '%s'\n", fileName);
      stbi_flip_vertically_on_write(1);
      stbi_write_png(fileName,
                     static_cast<int>(dim.width),
                     static_cast<int>(dim.height),
                     3,
                     pixelsRGB.data(),
                     0);
      break;
    }
  }

  loaderShouldExit_.store(true, std::memory_order_release);

#if IGL_WITH_IGLU
  imguiSession = nullptr;
#endif // IGL_WITH_IGLU
  // destroy all the Vulkan stuff before closing the window
  vb0_ = nullptr;
  ib0_ = nullptr;
  sbMaterials_ = nullptr;
  ubPerFrame_.clear();
  ubPerFrameShadow_.clear();
  ubPerObject_.clear();
  renderPipelineState_Mesh_ = nullptr;
  renderPipelineState_MeshWireframe_ = nullptr;
  renderPipelineState_Shadow_ = nullptr;
  renderPipelineState_Skybox_ = nullptr;
  renderPipelineState_Fullscreen_ = nullptr;
  computePipelineState_Grayscale_ = nullptr;
  textureDummyWhite_ = nullptr;
#if USE_OPENGL_BACKEND
  textureDummyBlack_ = nullptr;
#endif // USE_OPENGL_BACKEND
  skyboxTextureReference_ = nullptr;
  skyboxTextureIrradiance_ = nullptr;
  textures_.clear();
  texturesCache_.clear();
  sampler = nullptr;
  samplerShadow = nullptr;
  fbMain_ = nullptr;
  fbShadowMap_ = nullptr;
  fbOffscreen_ = nullptr;
  device_.reset(nullptr);

  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  glfwDestroyWindow(window);
  glfwTerminate();

  printf("Waiting for the loader thread to exit...\n");

  loaderPool_ = nullptr;

  return 0;
}
