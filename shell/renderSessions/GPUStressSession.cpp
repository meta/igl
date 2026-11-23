/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/GPUStressSession.h>

#include <IGLU/imgui/Session.h>
#include <IGLU/managedUniformBuffer/ManagedUniformBuffer.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/fwd.hpp>
#include <memory>
#include <random>
#include <utility>
#include <shell/shared/platform/DisplayContext.h>
#include <shell/shared/renderSession/AppParams.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/CommandBuffer.h>
#include <igl/NameHandle.h>
#include <igl/ShaderCreator.h>

namespace {
uint32_t customArc4random() {
  // NOLINTNEXTLINE(cert-msc50-cpp, facebook-hte-BadCall-rand)
  return static_cast<uint32_t>(std::rand()) * (0xffffffff / RAND_MAX);
}
} // namespace

#if IGL_PLATFORM_ANDROID

#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace igl::shell {

namespace {

constexpr uint32_t kMsaaSamples = 4u; // this is the max number possible
constexpr float kScaleFill = 1.f;

constexpr float kHalf = .5f;

} // namespace

GPUStressSession::GPUStressSession(std::shared_ptr<Platform> platform) :
  RenderSession(std::move(platform)),
  fps_(false),
  vertexData0_{
      VertexPosUvw{.position = {-kHalf, kHalf, -kHalf},
                   .uvw = {0.0, 1.0, 0.0, 1.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
      VertexPosUvw{.position = {kHalf, kHalf, -kHalf},
                   .uvw = {1.0, 1.0, 1.0, 1.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
      VertexPosUvw{.position = {-kHalf, -kHalf, -kHalf},
                   .uvw = {0.0, 0.0, 0.0, 0.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
      VertexPosUvw{.position = {kHalf, -kHalf, -kHalf},
                   .uvw = {1.0, 0.0, 1.0, 0.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
      VertexPosUvw{.position = {kHalf, kHalf, kHalf},
                   .uvw = {1.0, 1.0, 1.0, 1.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
      VertexPosUvw{.position = {-kHalf, kHalf, kHalf},
                   .uvw = {0.0, 1.0, 0.0, 1.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
      VertexPosUvw{.position = {kHalf, -kHalf, kHalf},
                   .uvw = {1.0, 0.0, 1.0, 0.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
      VertexPosUvw{.position = {-kHalf, -kHalf, kHalf},
                   .uvw = {0.0, 0.0, 0.0, 0.0},
                   .baseColor = {1.0, 1.0, 1.0, 1.0}},
  },
  indexData0_{0, 1, 2, 1, 3, 2, 1, 4, 3, 4, 6, 3, 4, 5, 6, 5, 7, 6,
              5, 0, 7, 0, 2, 7, 5, 4, 0, 4, 1, 0, 2, 3, 7, 3, 6, 7},
  indexData_{indexData0_.begin(), indexData0_.end()} {};

namespace {
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::string getLightingFunc(const char* matrixProj, const char* matrixMod) {
  const std::string var1 = matrixProj;
  const std::string var2 = matrixMod;
  auto func = std::string(
      R"(

      vec3 calcLighting(vec3 lightDir, vec3 lightPosition,  vec3 normal, float attenuation, vec3 color)
      {
        normal.xyz = ()" +
      var1 + "*" + var2 +
      R"(* vec4(normal, 0.f)).xyz;
        normal = normalize(normal);
        float angle = dot(normalize(lightDir), normal);
        float distance = length(lightPosition - screen_pos);
        float intensity = smoothstep(attenuation, 0.f, distance);
        intensity = clamp(intensity, 0.0, 1.0);
        return intensity * color * angle;
      }
      )");

  return func;
}
} // namespace

std::string GPUStressSession::getLightingCalc() const {
  std::string params = "\nvec4 lightFactor = color;\n";
  if (lightCount_) {
    params = "\nvec4 lightFactor = vec4(0.2, 0.2, 0.2, 1.0);\n";
  }
  for (int i = 0; i < lightCount_; ++i) {
    char tmp[256];
    std::snprintf(tmp,
                  sizeof(tmp),
                  "const vec3 lightColor%d = vec3(%f, %f, %f);\n",
                  i,
                  i % 3 == 0 ? 1.0 : static_cast<float>(customArc4random() % 32) / 32.f,
                  i % 3 == 1 ? 1.0 : static_cast<float>(customArc4random() % 32) / 32.f,
                  i % 3 == 2 ? 1.0 : static_cast<float>(customArc4random() % 32) / 32.f);
    params += tmp;
    std::snprintf(tmp,
                  sizeof(tmp),
                  "const vec3 lightPos%d = vec3(%f, %f, %f);\n",
                  i,
                  -1.f + static_cast<float>(customArc4random() % 32) / 16.f,
                  -1.f + static_cast<float>(customArc4random() % 32) / 16.f,
                  -1.f + static_cast<float>(customArc4random() % 32) / 16.f);
    params += tmp;
    std::snprintf(
        tmp,
        sizeof(tmp),
        "lightFactor.xyz += calcLighting(-lightPos%d, lightPos%d, color.xyz, 1.0, lightColor%d);\n",
        i,
        i,
        i);
    params += tmp;
  }
  return params;
}

namespace {
std::string getVulkanVertexShaderSource(bool multiView) {
  return std::string(multiView ? "\n#define MULTIVIEW 1\n" : "") + R"(
#ifdef MULTIVIEW
#extension GL_EXT_multiview : enable
#endif
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 uvw_in;
layout(location = 2) in vec4 base_color;

layout (location = 0) out vec4 color;
layout (location = 1) out vec4 uv;
layout (location = 2) out vec3 screen_pos;

layout(push_constant) uniform PushConstants {
    mat4 projectionMatrix;
    mat4 modelViewMatrix;
} pc;

out gl_PerVertex { vec4 gl_Position; };

void main() {
  #ifdef MULTIVIEW
    color = vec4(base_color.x, abs(float(gl_ViewIndex)-1.f) * base_color.y, base_color.z, base_color.w);
  #elif
    color = base_color;
  #endif

    uv = uvw_in;
    gl_Position = pc.projectionMatrix * pc.modelViewMatrix * vec4(position.xyz, 1.0);
    screen_pos = gl_Position.xyz/gl_Position.w;
})";
}
} // namespace

std::string GPUStressSession::getVulkanFragmentShaderSource() const {
  return R"(
layout(location = 0) out vec4 fColor;
layout(location = 0) in vec4 color;
layout(location = 1) in vec4 uv;
layout(location = 2) in vec3 screen_pos;

layout (set = 0, binding = 0) uniform sampler2D uTex;
layout (set = 0, binding = 1) uniform sampler2D uTex2;

layout(push_constant) uniform PushConstants {
    mat4 projectionMatrix;
    mat4 modelViewMatrix;
} pc;
)" + getLightingFunc("pc.projectionMatrix", "pc.modelViewMatrix") +
         R"(
                      void main() {)" +
         getLightingCalc() +
         R"(
  fColor = lightFactor * texture(uTex2, uv.xy) * texture(uTex, uv.zw);
})";
}

std::string GPUStressSession::getWgslShaderSource() const {
  // getLightingCalc() emits GLSL statements; they map to WGSL with calcLighting() returning a vec4
  // whose w is 0.
  std::string lighting = getLightingCalc();
  const std::pair<const char*, const char*> kGlslToWgsl[] = {
      {"vec4 lightFactor = ", "var lightFactor = "},
      {"const vec3 ", "let "},
      {"vec4(", "vec4f("},
      {"vec3(", "vec3f("},
      {"lightFactor.xyz += ", "lightFactor += "},
  };
  for (const auto& [from, to] : kGlslToWgsl) {
    for (size_t pos = 0; (pos = lighting.find(from, pos)) != std::string::npos;
         pos += std::strlen(to)) {
      lighting.replace(pos, std::strlen(from), to);
    }
  }
  return R"(
struct PushConstants {
  projectionMatrix : mat4x4f,
  modelViewMatrix : mat4x4f,
};

@group(3) @binding(0) var<uniform> pc : PushConstants;
@group(0) @binding(0) var uTex : texture_2d<f32>;
@group(0) @binding(1) var uTexSampler : sampler;
@group(0) @binding(2) var uTex2 : texture_2d<f32>;
@group(0) @binding(3) var uTex2Sampler : sampler;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) uv : vec4f,
  @location(2) screen_pos : vec3f,
};

@vertex
fn vertexShader(@location(0) position : vec3f,
                @location(1) uvw_in : vec4f,
                @location(2) base_color : vec4f) -> VertexOut {
  let pos = pc.projectionMatrix * pc.modelViewMatrix * vec4f(position, 1.0);
  return VertexOut(pos, base_color, uvw_in, pos.xyz / pos.w);
}

var<private> screen_pos : vec3f;

fn calcLighting(lightDir : vec3f, lightPosition : vec3f, normalIn : vec3f, attenuation : f32,
                color : vec3f) -> vec4f {
  let n = normalize((pc.projectionMatrix * pc.modelViewMatrix * vec4f(normalIn, 0.0)).xyz);
  let angle = dot(normalize(lightDir), n);
  let dist = length(lightPosition - screen_pos);
  let intensity = clamp(smoothstep(attenuation, 0.0, dist), 0.0, 1.0);
  return vec4f(intensity * color * angle, 0.0);
}

@fragment
fn fragmentShader(v : VertexOut) -> @location(0) vec4f {
  screen_pos = v.screen_pos;
  let color = v.color;
)" + lighting +
         R"(
  return lightFactor * textureSample(uTex2, uTex2Sampler, v.uv.xy) *
         textureSample(uTex, uTexSampler, v.uv.zw);
}
)";
}

std::unique_ptr<IShaderStages> GPUStressSession::getShaderStagesForBackend(
    IDevice& device) const noexcept {
  const bool multiView = device.hasFeature(DeviceFeatures::Multiview);
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (device.getBackendType()) {
  // @fb-only
    // @fb-only
    // @fb-only
  case BackendType::WebGPU:
    return igl::ShaderStagesCreator::fromLibraryStringInput(
        device, getWgslShaderSource().c_str(), "vertexShader", "fragmentShader", "", nullptr);
  case BackendType::Vulkan:
    return igl::ShaderStagesCreator::fromModuleStringInput(
        device,
        getVulkanVertexShaderSource(multiView).c_str(),
        "main",
        "",
        getVulkanFragmentShaderSource().c_str(),
        "main",
        "",
        nullptr);
  default:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  }
}

void GPUStressSession::addNormalsToCube() {
  if (!lightCount_) {
    return;
  }

  const size_t faceCount = indexData_.size() / 6;
  bool normalSet[36] = {false};
  for (size_t j = 0; j < faceCount; j++) {
    const size_t offset = j * 6;
    auto vec1 = vertexData0_.at(indexData_[offset + 1]).position -
                vertexData0_.at(indexData_[offset + 2]).position;
    auto vec2 = vertexData0_.at(indexData_[offset + 1]).position -
                vertexData0_.at(indexData_.at(offset + 0)).position;
    auto normal = glm::normalize(glm::cross(vec1, vec2));
    std::vector<int> indexremap;
    indexremap.resize(24, -1);

    for (size_t i = offset; i < offset + 6; i++) {
      const size_t oldIndex = indexData_[i];
      if (indexremap.at(oldIndex) != -1) {
        indexData_.at(i) = indexremap[oldIndex];
      } else if (!normalSet[oldIndex]) {
        vertexData_.at(oldIndex).baseColor = glm::vec4(normal, 1.0);
        normalSet[oldIndex] = true;
        indexremap.at(oldIndex) = static_cast<int>(oldIndex);
      } else {
        auto vertex = vertexData0_.at(oldIndex);
        vertex.baseColor = glm::vec4(normal, 1.0);
        vertexData_.push_back(vertex);
        const size_t nextIndex = (vertexData_.size() - 1);
        indexData_.at(i) = nextIndex;
        normalSet[nextIndex] = true;
        indexremap.at(oldIndex) = static_cast<int>(nextIndex);
      }
    }
  }
}

namespace {
bool isDeviceCompatible(IDevice& device) noexcept {
  // Only Vulkan and WebGPU shaders exist.
  const auto backendtype = device.getBackendType();
  return backendtype == BackendType::Vulkan || backendtype == BackendType::WebGPU;
}

int setCurrentThreadAffinityMask(int mask) {
#if IGL_PLATFORM_ANDROID
  int err = 0, syscallres = 0;
  const pid_t pid = gettid();
  syscallres = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (syscallres) {
    err = errno;
    IGL_LOG_ERROR("Set thread affinity failed. with mask 0x%x and error 0x%x\n", mask, err);
    return err;
  }
  return 0;
#else
  IGL_LOG_ERROR("Set thread affinity not supported on this platorm");
  return -1;
#endif
}

double calcPi(int numberOfDivisions, int core) {
  double pi = 0.0;

  if (core >= 0) {
    setCurrentThreadAffinityMask((1 << core));
  }
  for (int i = 0; i <= numberOfDivisions; ++i) {
    const double numerator = 1.0;
    const double denominator = std::sqrt(1.0 + std::pow(-1.0, i));
    if (denominator > 0.f) {
      pi += numerator / denominator;
    }
  }
  return pi * 4.0;
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
void GPUStressSession::thrashCPU() noexcept {
  // @fb-only
  static std::vector<std::future<double>> futures;
  static unsigned int threadSpawnId = 0;
  if (goSlowOnCpu_) {
    // don't fall off the array
    while (threadIds_.size() < threadCount_) {
      threadIds_.push_back(-1);
    }
    if (!threadCount_) {
      pi_ = calcPi(goSlowOnCpu_, -1);
    }
    while (futures.size() < threadCount_) {
      auto future = std::async(std::launch::async, [this] {
        return calcPi(goSlowOnCpu_, threadIds_[threadSpawnId % threadCount_]);
      });

      futures.push_back(std::move(future));
      threadSpawnId++;
    }

    for (int i = static_cast<int>(futures.size()) - 1; i > -1; i--) {
      auto& future = futures.at(i);

      // Use wait_for() with zero milliseconds to check thread status.
      auto status = future.wait_for(std::chrono::milliseconds(0));

      if (status == std::future_status::ready) {
        pi_ += future.get();
        futures.erase(futures.begin() + i);
      }
    }
  }
}

float GPUStressSession::doReadWrite(std::vector<std::vector<std::vector<float>>>& memBlock,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    int numBlocks,
                                    int numRows,
                                    int numCols,
                                    int threadId) {
  if (threadId != -1) {
    setCurrentThreadAffinityMask(1 << threadId);
  }
  std::mt19937 gen(0);
  std::uniform_int_distribution<> randBlocks(0, numBlocks - 1);
  std::uniform_int_distribution<> randRows(0, numRows - 1);
  std::uniform_int_distribution<> randCols(0, numCols - 1);
  float sum = 0.f;
  for (int i = 0; i < memoryWrites_; i++) {
    const int block = randBlocks(gen);
    const int row = randRows(gen);
    const int col = randCols(gen);
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    memBlock[block].at(row)[col] = customArc4random();
  }

  for (int i = 0; i < memoryReads_; i++) {
    const int block = randBlocks(gen);
    const int row = randRows(gen);
    const int col = randCols(gen);
    sum += i % 1 ? -1.f : 1.f * memBlock.at(block)[row][col];
  }

  return sum;
}

void GPUStressSession::allocateMemory() {
  if (thrashMemory_) {
    const static size_t kBlocks = memorySize_;
    const static size_t kRows = 1024;
    const static size_t kCols = 1024;
    if (memBlock_.empty()) {
      memBlock_.resize((kBlocks));
      for (auto& block : memBlock_) {
        block.resize(kRows);
        for (auto& row : block) {
          row.resize(kCols, 0);
          for (int i = 0; i < kCols; i++) {
            row.at(i) = (i);
          }
        }
      }
    }
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void GPUStressSession::thrashMemory() noexcept {
  if (!thrashMemory_) {
    return;
  }

  const static size_t kBlocks = memorySize_;
  const static size_t kRows = 1024;
  const static size_t kCols = 1024;

  if (!threadCount_) {
    memoryVal_.store(doReadWrite(memBlock_,
                                 static_cast<int>(kBlocks),
                                 static_cast<int>(kRows),
                                 static_cast<int>(kCols),
                                 -1));
  } else {
    // @fb-only
    static std::vector<std::future<float>> futures;
    static int memoryThreadId = 0;

    while (futures.size() < threadCount_) {
      auto future = std::async(std::launch::async, [this] {
        return doReadWrite(memBlock_,
                           static_cast<int>(kBlocks),
                           static_cast<int>(kRows),
                           static_cast<int>(kCols),
                           threadIds_[memoryThreadId % threadCount_]);
      });

      futures.push_back(std::move(future));
      memoryThreadId++;
    }

    for (int i = static_cast<int>(futures.size()) - 1; i > -1; i--) {
      auto& future = futures.at(i);

      // Use wait_for() with zero milliseconds to check thread status.
      auto status = future.wait_for(std::chrono::milliseconds(0));

      if (status == std::future_status::ready) {
        memoryVal_.store(future.get());
        futures.erase(futures.begin() + i);
      }
    }
  }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void GPUStressSession::getOffset(int counter, float& x, float& y, float& z) {
  if (testOverdraw_) {
    x = 0.f;
    y = 0.f;
    z = counter % 2 ? -kHalf / static_cast<float>(cubeCount_)
                    : kHalf / static_cast<float>(cubeCount_);
    z *= counter / 2.f;
    return;
  }
  const float grid = std::ceil(std::pow(cubeCount_, 1.0f / 3.0f));
  const int igrid = static_cast<int>(grid);
  // const float fgrid = static_cast<float>(igrid);
  x = static_cast<float>((counter % igrid) - grid / 2);
  // NOLINTNEXTLINE(bugprone-integer-division)
  z = (static_cast<float>(counter / (igrid * igrid)) - grid / 2.f);
  // NOLINTNEXTLINE(bugprone-integer-division)
  y = (static_cast<float>((counter % (igrid * igrid)) / igrid) - grid / 2.f);
}

glm::vec3 GPUStressSession::animateCube(int counter,
                                        float x,
                                        float y,
                                        float scale,
                                        int frameCount) {
  struct AnimationInfo {
    glm::vec3 velocity;
    glm::vec3 lastPos;
  };

  // @fb-only
  static std::vector<AnimationInfo> animations;
  if (animations.size() < counter) {
    AnimationInfo info;
    info.velocity =
        glm::vec3(1.f * (counter % 2 ? 1.0 : -1.0), 1.f - static_cast<float>(counter % 3), 0.f);
    info.lastPos = glm::vec3(x, y, 0);
    animations.push_back(info);
  }

  float velocityScale = 1.f;
  if (dropFrameX_ && (frameCount % dropFrameX_) < dropFrameCount_) {
    velocityScale = 0.f;
  } else if (dropFrameX_ && (frameCount % dropFrameX_) == dropFrameCount_) {
    velocityScale = 1.f + static_cast<float>(dropFrameCount_);
  }
  const glm::vec3 pos =
      animations[counter].lastPos + animations[counter].velocity * velocityScale * scale * .005f;
  // check for collisons;
  const float radius = .75 * scale;
  if (pos.x + radius > 1.f) {
    animations[counter].velocity.x = -1.f;
  }
  if (pos.x - radius < -1.f) {
    animations[counter].velocity.x = 1.f;
  }

  if (pos.y + radius > 1.f) {
    animations[counter].velocity.y = -1.f;
  }
  if (pos.y - radius < -1.f) {
    animations[counter].velocity.y = 1.f;
  }

  animations[counter].lastPos = pos;
  return pos;
}

void GPUStressSession::createSamplerAndTextures(const IDevice& device) {
  // Sampler & Texture
  const SamplerStateDesc samplerDesc{
      .minFilter = SamplerMinMagFilter::Linear,
      .magFilter = SamplerMinMagFilter::Linear,
      .addressModeU = SamplerAddressMode::MirrorRepeat,
      .addressModeV = SamplerAddressMode::MirrorRepeat,
      .addressModeW = SamplerAddressMode::MirrorRepeat,
  };
  samp0_ = device.createSamplerState(samplerDesc, nullptr);
  samp1_ = device.createSamplerState(samplerDesc, nullptr);

  tex0_ = getPlatform().loadTexture("macbeth.png");
  tex1_ = getPlatform().loadTexture("igl.png");
}

void GPUStressSession::createCubes() {
  // only reset once - on mac we hit this path multiple times for different
  // devices
  vertexData_ = vertexData0_;
  indexData_ = indexData0_;

  addNormalsToCube(); // setup for lighting if appropriate

  const float grid = std::ceil(std::pow(cubeCount_, 1.0f / 3.0f));

  const int vertexCount = static_cast<int>(vertexData_.size());
  const int indexCount = static_cast<int>(indexData_.size());

  std::mt19937 gen(0);
  std::uniform_real_distribution<> dis(0, 1.f);
  const float scale = 1.f / grid;

  const int uvScale = 1.f / grid;
  glm::vec2 offset = glm::vec2(0.f, 0.f);

  // Vertex buffer, Index buffer and Vertex Input
  for (int i = 1; i < cubeCount_; i++) {
    float x = NAN, y = NAN, z = NAN;
    getOffset(i, x, y, z);
    glm::vec4 color(1.0, 1.0, 1.0, 1.f);
    color[0] = (dis(gen));
    color[1] = (dis(gen));
    color[2] = (dis(gen));

    for (int j = 0; j < vertexCount; j++) {
      VertexPosUvw newPoint = vertexData_.at(j);
      newPoint.position += (glm::vec3(x, y, z));
      newPoint.uvw *= glm::vec4(uvScale, uvScale, 1.f, 1.f);
      newPoint.uvw += glm::vec4(offset.x, offset.y, 0.f, 0.f);
      if (!lightCount_) {
        newPoint.baseColor = color;
      }
      vertexData_.push_back(newPoint);
    }
    for (int j = 0; j < indexCount; j++) {
      indexData_.push_back(static_cast<uint16_t>(indexData_.at(j) + i * (vertexCount)));
    }

    offset.x += 1.f / grid;
    if (offset.x > 1.f) {
      offset.x = 0.f;
      offset.y += 1.f / grid;
    }
  }

  if (!testOverdraw_) // we want to fill up the screen here
  {
    for (auto& i : vertexData_) {
      i.position.x *= scale;
      i.position.y *= scale;
      i.position.z *= scale;
    }
  }

  auto& device = getPlatform().getDevice();
  const BufferDesc vb0Desc{.type = BufferDesc::BufferTypeBits::Vertex,
                           .data = vertexData_.data(),
                           .length = sizeof(VertexPosUvw) * vertexData_.size()};
  vb0_ = device.createBuffer(vb0Desc, nullptr);
  const BufferDesc ibDesc{.type = BufferDesc::BufferTypeBits::Index,
                          .data = indexData_.data(),
                          .length = sizeof(uint16_t) * indexData_.size()};
  ib0_ = device.createBuffer(ibDesc, nullptr);

  const VertexInputStateDesc inputDesc = {
      .numAttributes = 3,
      .attributes =
          {
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float3,
               .offset = offsetof(VertexPosUvw, position),
               .name = "position",
               .location = 0},
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float4,
               .offset = offsetof(VertexPosUvw, uvw),
               .name = "uvw_in",
               .location = 1},
              {.bufferIndex = 0,
               .format = VertexAttributeFormat::Float4,
               .offset = offsetof(VertexPosUvw, baseColor),
               .name = "base_color",
               .location = 2},
          },
      .numInputBindings = 1,
      .inputBindings = {{.stride = sizeof(VertexPosUvw)}},
  };
  vertexInput0_ = device.createVertexInputState(inputDesc, nullptr);
}

void GPUStressSession::processCustomParameter(const std::string& key, const std::string& value) {
  auto toLower = [](std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
  };

  const std::string lowerKey = toLower(key);

  if (lowerKey == "numthreads") {
    setNumThreads(std::stoi(value));
  } else if (lowerKey == "thrashmemory") {
    setThrashMemory(toLower(value) == "true" || value == "1");
  } else if (lowerKey == "memorysize") {
    setMemorySize(static_cast<size_t>(std::stoull(value)));
  } else if (lowerKey == "memoryreads") {
    setMemoryReads(static_cast<size_t>(std::stoull(value)));
  } else if (lowerKey == "memorywrites") {
    setMemoryWrites(static_cast<size_t>(std::stoull(value)));
  } else if (lowerKey == "goslowoncpu") {
    setGoSlowOnCpu(std::stoi(value));
  } else if (lowerKey == "cubecount") {
    setCubeCount(std::stoi(value));
  } else if (lowerKey == "drawcount") {
    setDrawCount(std::stoi(value));
  } else if (lowerKey == "testoverdraw") {
    setTestOverdraw(toLower(value) == "true" || value == "1");
  } else if (lowerKey == "enableblending") {
    setEnableBlending(toLower(value) == "true" || value == "1");
  } else if (lowerKey == "usemsaa") {
    setUseMSAA(toLower(value) == "true" || value == "1");
  } else if (lowerKey == "lightcount") {
    setLightCount(std::stoi(value));
  } else if (lowerKey == "threadcore") {
    const size_t commaPos = value.find(',');
    if (commaPos != std::string::npos) {
      int thread = std::stoi(value.substr(0, commaPos));
      int core = std::stoi(value.substr(commaPos + 1));
      if (thread >= 0 && threadCount_ > 0) {
        setThreadCore(thread, core);
      } else {
        IGL_LOG_ERROR(
            "Invalid threadCore parameter: thread=%d, threadCount=%d. Ensure numThreads > 0 is "
            "set before threadCore.",
            thread,
            threadCount_.load());
      }
    }
  } else if (lowerKey == "dropframeinterval") {
    setDropFrameInterval(std::stoi(value));
  } else if (lowerKey == "dropframecount") {
    setDropFrameCount(std::stoi(value));
  } else if (lowerKey == "rotatecubes") {
    setRotateCubes(toLower(value) == "true" || value == "1");
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void GPUStressSession::initialize() noexcept {
  pipelineState_ = nullptr;
  vertexInput0_ = nullptr;
  vb0_ = nullptr;
  ib0_ = nullptr; // Buffers for vertices and indices (or constants)
  samp0_ = nullptr;
  samp1_ = nullptr;
  framebuffer_ = nullptr;
  vertexData_.resize(0); // recalc verts
  indexData_.resize(36); // keep the first 36 indices

  //  this is sets the size of our 'app window' so we can shrink the number of
  //  changed pixels we send to the delphi.
  appParamsRef().sizeX = .5f;
  appParamsRef().sizeY = .5f;
  // Process custom parameters from ShellParams
  if (shellParams().benchmarkParams.has_value()) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto& benchmarkParams = shellParams().benchmarkParams.value();
    const auto& customParams = benchmarkParams.customParams;
    for (const auto& [key, value] : customParams) {
      IGL_LOG_DEBUG("Processing custom parameter: '%s' = '%s'", key.c_str(), value.c_str());
      processCustomParameter(key, value);
    }
  }

  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  createCubes();
  if (!imguiSession_) {
    imguiSession_ = std::make_unique<iglu::imgui::Session>(getPlatform().getDevice(),
                                                           getPlatform().getInputDispatcher());
  }

  createSamplerAndTextures(device);
  shaderStages_ = getShaderStagesForBackend(device);

  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = device.createCommandQueue({}, nullptr);

  tex0_->generateMipmap(*commandQueue_);
  tex1_->generateMipmap(*commandQueue_);

  // Set up vertex uniform data
  vertexParameters_.scaleZ = 1.0f;

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = {0.0, 0.0, 0.0f, 0.0f},
      }},
      .depthAttachment =
          {
              .loadAction = LoadAction::Clear,
              .clearDepth = 1.0,
          },
  };

  if (useMSAA_) {
    renderPass_.colorAttachments[0].storeAction = StoreAction::MsaaResolve;
  }

  const DepthStencilStateDesc depthDesc{
      .compareFunction = CompareFunction::Less,
      .isDepthWriteEnabled = true,
  };
  depthStencilState_ = device.createDepthStencilState(depthDesc, nullptr);
}

void GPUStressSession::setProjectionMatrix(float aspectRatio) {
  // perspective projection
  constexpr float fov = 45.0f * (M_PI / 180.0f);
  glm::mat4 projectionMat = glm::perspectiveLH(fov, aspectRatio, .1f, 2.1f);
  if (testOverdraw_ || !rotateCubes_) {
    projectionMat =
        glm::orthoLH_ZO(-kHalf, kHalf, -kHalf / aspectRatio, kHalf / aspectRatio, .1f, 2.1f);
  }
  vertexParameters_.projectionMatrix = projectionMat;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void GPUStressSession::setModelViewMatrix(float angle,
                                          float scaleZ,
                                          float offsetX,
                                          float offsetY,
                                          float offsetZ) {
  float divisor = std::ceil(std::sqrt(static_cast<float>(drawCount_))) / (kHalf * kScaleFill);

  if (testOverdraw_) {
    divisor = 1.f;
    offsetX = 0.f;
    offsetY = 0.f;
  }

  const float cosAngle = std::cos(angle);
  const float sinAngle = std::sin(angle);
  const glm::vec4 v0(cosAngle / divisor, 0.f, -sinAngle / divisor, 0.f);
  const glm::vec4 v1(0.f, 1.f / divisor, 0.f, 0.f);
  const glm::vec4 v2(sinAngle / divisor, 0.f, cosAngle / divisor, 0.f);
  const glm::vec4 v3(offsetX, offsetY, 1.f + offsetZ, 1.f);
  const glm::mat4 test(v0, v1, v2, v3);

  vertexParameters_.modelViewMatrix = test;
  vertexParameters_.scaleZ = scaleZ;
}

void GPUStressSession::initState(const SurfaceTextures& surfaceTextures) {
  Result ret;

  // TODO: fix framebuffers so you can update the resolve texture
  if (framebuffer_ == nullptr) {
    FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = surfaceTextures.color}},
                                    .depthAttachment = {.texture = surfaceTextures.depth},
                                    .mode = surfaceTextures.color->getNumLayers() > 1
                                                ? FramebufferMode::Stereo
                                                : FramebufferMode::Mono};

    if (useMSAA_) {
      const auto dimensions = surfaceTextures.color->getDimensions();

      const TextureDesc fbTexDesc = {.width = dimensions.width,
                                     .height = dimensions.height,
                                     .depth = 1,
                                     .numLayers = surfaceTextures.color->getNumLayers(),
                                     .numSamples = kMsaaSamples,
                                     .usage = TextureDesc::TextureUsageBits::Attachment,
                                     .numMipLevels = 1,
                                     .type = surfaceTextures.color->getNumLayers() > 1
                                                 ? TextureType::TwoDArray
                                                 : TextureType::TwoD,
                                     .format = surfaceTextures.color->getFormat(),
                                     .storage = ResourceStorage::Private};

      framebufferDesc.colorAttachments[0].texture =
          getPlatform().getDevice().createTexture(fbTexDesc, nullptr);

      framebufferDesc.colorAttachments[0].resolveTexture = surfaceTextures.color;

      // Per IGL guidelines, surfaceTextures.depth may be null on configurations
      // without a depth buffer (e.g., 2D overlay sessions). Only build the MSAA
      // depth attachment if the platform actually provided a depth surface.
      if (surfaceTextures.depth) {
        const TextureDesc depthDesc = {.width = dimensions.width,
                                       .height = dimensions.height,
                                       .depth = 1,
                                       .numLayers = surfaceTextures.depth->getNumLayers(),
                                       .numSamples = kMsaaSamples,
                                       .usage = TextureDesc::TextureUsageBits::Attachment,
                                       .numMipLevels = 1,
                                       .type = surfaceTextures.depth->getNumLayers() > 1
                                                   ? TextureType::TwoDArray
                                                   : TextureType::TwoD,
                                       .format = surfaceTextures.depth->getFormat(),
                                       .storage = ResourceStorage::Private};

        framebufferDesc.depthAttachment.texture =
            getPlatform().getDevice().createTexture(depthDesc, nullptr);
      }
    }

    framebuffer_ = getPlatform().getDevice().createFramebuffer(framebufferDesc, &ret);

    IGL_DEBUG_ASSERT(ret.isOk());
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  }

  if (useMSAA_) {
    framebuffer_->updateResolveAttachment(surfaceTextures.color);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  constexpr uint32_t textureUnit = 0;
  if (pipelineState_ == nullptr) {
    const RenderPipelineDesc graphicsDesc = {
        .vertexInputState = vertexInput0_,
        .shaderStages = shaderStages_,
        .targetDesc =
            {
                .colorAttachments = {{
                    .textureFormat = framebuffer_->getColorAttachment(0)->getProperties().format,
                    .blendEnabled = enableBlending_,
                    .rgbBlendOp = BlendOp::Add,
                    .alphaBlendOp = BlendOp::Add,
                    .srcRGBBlendFactor = BlendFactor::SrcAlpha,
                    .srcAlphaBlendFactor = BlendFactor::SrcAlpha,
                    .dstRGBBlendFactor = BlendFactor::OneMinusSrcAlpha,
                    .dstAlphaBlendFactor = BlendFactor::OneMinusSrcAlpha,
                }},
                .depthAttachmentFormat = framebuffer_->getDepthAttachment()->getProperties().format,
            },
        .cullMode = CullMode::Back,
        .frontFaceWinding = WindingMode::Clockwise,
        .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE("inputImage")}},
        .sampleCount = useMSAA_ ? kMsaaSamples : 1u,
    };

    pipelineState_ = getPlatform().getDevice().createRenderPipeline(graphicsDesc, nullptr);
  }
}

void GPUStressSession::drawCubes(const SurfaceTextures& surfaceTextures,
                                 std::shared_ptr<IRenderCommandEncoder> commands) {
  static float angle = 0.0f;
  static int frameCount = 0;
  frameCount++;

  angle += 0.005f;

  // rotating animation
  static float scaleZ = 1.0f, ss = 0.005f;
  scaleZ += ss;
  scaleZ = scaleZ < 0.0f ? 0.0f : scaleZ > 1.0 ? 1.0f : scaleZ;
  if (scaleZ <= 0.05f || scaleZ >= 1.0f) {
    ss *= -1.0f;
  }

  auto& device = getPlatform().getDevice();
  // cube animation
  constexpr uint32_t textureUnit = 0;
  constexpr uint32_t textureUnit1 = 1;
  const int grid = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(drawCount_))));
  const float divisor = .5 / static_cast<float>(grid);
  const float scale = 1.f / std::ceil(std::pow(cubeCount_, 1.0f / 3.0f));

  int counter = 0;
  setProjectionMatrix(surfaceTextures.color->getAspectRatio());

  commands->bindIndexBuffer(*ib0_, IndexFormat::UInt16);

  std::shared_ptr<iglu::ManagedUniformBuffer> vertUniformBuffer = nullptr;
  for (int i = -grid / 2; i < grid / 2 + grid % 2; i++) {
    for (int j = -grid / 2; j < grid / 2 + grid % 2; j++) {
      if (counter >= drawCount_) {
        break;
      }
      counter++;
      float x = static_cast<float>(j) * divisor;
      float y = static_cast<float>(i) * divisor;
      if (dropFrameX_) {
        auto offset = animateCube(counter, x, y, scale, frameCount);
        x = offset.x;
        y = offset.y;
      }

      setModelViewMatrix((testOverdraw_ || !rotateCubes_) ? 0.f : angle, scaleZ, x, y, 0.f);

      // note that we are deliberately binding redundant state - the goal here
      // is to tax the driver.  The giant vertex buffer (cubeCount_) will stress
      // just the gpu
      commands->bindVertexBuffer(0, *vb0_);
      commands->bindTexture(textureUnit, BindTarget::kFragment, tex0_.get());
      commands->bindSamplerState(textureUnit, BindTarget::kFragment, samp0_.get());
      commands->bindTexture(textureUnit1, BindTarget::kFragment, tex1_.get());
      commands->bindSamplerState(textureUnit1, BindTarget::kFragment, samp1_.get());
      commands->bindRenderPipelineState(pipelineState_);
      commands->bindDepthStencilState(depthStencilState_);

      // Bind Vertex Uniform Data
      if (device.getBackendType() == BackendType::Vulkan ||
          device.getBackendType() == BackendType::WebGPU) {
        commands->bindPushConstants(&vertexParameters_,
                                    sizeof(vertexParameters_) - sizeof(float)); // z isn't used
      } else {
        if (!vertUniformBuffer) {
          iglu::ManagedUniformBufferInfo info;
          info.index = 1;
          info.length = sizeof(VertexFormat);
          info.uniforms = std::vector<UniformDesc>{
              UniformDesc{.name = "projectionMatrix",
                          .location = -1,
                          .type = UniformType::Mat4x4,
                          .numElements = 1,
                          .offset = offsetof(VertexFormat, projectionMatrix),
                          .elementStride = 0},
              UniformDesc{.name = "modelViewMatrix",
                          .location = -1,
                          .type = UniformType::Mat4x4,
                          .numElements = 1,
                          .offset = offsetof(VertexFormat, modelViewMatrix),
                          .elementStride = 0},
              UniformDesc{.name = "scaleZ",
                          .location = -1,
                          .type = UniformType::Float,
                          .numElements = 1,
                          .offset = offsetof(VertexFormat, scaleZ),
                          .elementStride = 0}};

          vertUniformBuffer = std::make_shared<iglu::ManagedUniformBuffer>(device, info);
          IGL_DEBUG_ASSERT(vertUniformBuffer->result.isOk());
        }
        *static_cast<VertexFormat*>(vertUniformBuffer->getData()) = vertexParameters_;
        vertUniformBuffer->bind(device, *pipelineState_, *commands);
      }

      commands->drawIndexed(indexData_.size());
    }
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void GPUStressSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }
  if (forceReset_) {
    memBlock_.resize(0);
    forceReset_ = false;
    initialize();
  }

  allocateMemory();
  thrashCPU();
  thrashMemory();

  fps_.updateFPS(getDeltaSeconds());

  initState(surfaceTextures);

  // Command buffers (1-N per thread): create, submit and forget
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  const std::shared_ptr<IRenderCommandEncoder> commands =
      buffer->createRenderCommandEncoder(renderPass_, framebuffer_);

  const FramebufferDesc framebufferDesc{
      .colorAttachments = {{.texture = framebuffer_->getColorAttachment(0)}},
      .depthAttachment = {.texture = framebuffer_->getDepthAttachment()},
  };

  // setup UI
  const ImGuiViewport* v = ImGui::GetMainViewport();
  imguiSession_->beginFrame(framebufferDesc, getPlatform().getDisplayContext().pixelsPerPoint);
  bool open = false;
  ImGui::SetNextWindowPos(
      {
          v->WorkPos.x + v->WorkSize.x - 60.0f,
          v->WorkPos.y + v->WorkSize.y * .25f + 15.0f,
      },
      ImGuiCond_Always,
      {1.0f, 0.0f});
  ImGui::Begin("GPU", &open, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground);
  ImGui::SetWindowFontScale(2.f);

  // draw stuff
  drawCubes(surfaceTextures, commands);

  { // Draw using ImGui every frame

    ImGui::TextColored(ImVec4(1.f, 0.f, 0.f, 1.f),
                       "FPS: (%f)   PI: (%lf)  Memory (%f)",
                       fps_.getAverageFPS(),
                       pi_,
                       memoryVal_.load());
    ImGui::End();
    imguiSession_->endFrame(getPlatform().getDevice(), *commands);
  }

  commands->endEncoding();

  if (shellParams().shouldPresent) {
    buffer->present(useMSAA_ ? framebuffer_->getResolveColorAttachment(0)
                             : framebuffer_->getColorAttachment(0));
  }

  commandQueue_->submit(*buffer); // Guarantees ordering between command buffers
}

void GPUStressSession::setNumThreads(int numThreads) {
  threadCount_ = numThreads;
}

void GPUStressSession::setThrashMemory(bool thrashMemory) {
  thrashMemory_ = thrashMemory;
}
void GPUStressSession::setMemorySize(size_t memorySize) {
  if (memorySize != memorySize_) {
    memorySize_ = memorySize;
    forceReset_ = true;
  }
}
void GPUStressSession::setMemoryReads(size_t memoryReads) {
  memoryReads_ = memoryReads;
}
void GPUStressSession::setMemoryWrites(size_t memoryWrites) {
  memoryWrites_ = memoryWrites;
}
void GPUStressSession::setGoSlowOnCpu(int goSlowOnCpu) {
  goSlowOnCpu_ = goSlowOnCpu;
}
void GPUStressSession::setCubeCount(int count) {
  if (cubeCount_ != count) {
    forceReset_ = true;
    cubeCount_ = count;
  }
}
void GPUStressSession::setDrawCount(int count) {
  drawCount_ = count;
}
void GPUStressSession::setTestOverdraw(bool testOverdraw) {
  if (testOverdraw != testOverdraw_) {
    testOverdraw_ = testOverdraw;
    forceReset_ = true;
  }
}
void GPUStressSession::setEnableBlending(bool enableBlending) {
  if (enableBlending != enableBlending_) {
    enableBlending_ = enableBlending;
    forceReset_ = true;
  }
}
void GPUStressSession::setUseMSAA(bool useMSAA) {
  if (useMSAA_ != useMSAA) {
    useMSAA_ = useMSAA;
    forceReset_ = true;
  }
}
void GPUStressSession::setLightCount(int lightCount) {
  if (lightCount_ != lightCount) {
    lightCount_ = lightCount;
    forceReset_ = true;
  }
}

void GPUStressSession::setThreadCore(int thread, int core) {
  threadIds_[thread % threadCount_] = core;
}

int GPUStressSession::getNumThreads() const {
  return threadCount_;
}
bool GPUStressSession::getThrashMemory() const {
  return thrashMemory_;
}
size_t GPUStressSession::getMemorySize() const {
  return memorySize_;
}
size_t GPUStressSession::getMemoryReads() const {
  return memoryReads_;
}
size_t GPUStressSession::getMemoryWrites() const {
  return memoryWrites_;
}
bool GPUStressSession::getGoSlowOnCpu() const {
  return goSlowOnCpu_ != 0;
}
int GPUStressSession::getCubeCount() const {
  return cubeCount_;
}
int GPUStressSession::getDrawCount() const {
  return drawCount_;
}
bool GPUStressSession::getTestOverdraw() const {
  return testOverdraw_;
}
bool GPUStressSession::getEnableBlending() const {
  return enableBlending_;
}
bool GPUStressSession::getUseMSAA() const {
  return useMSAA_;
}
int GPUStressSession::getLightCount() const {
  return lightCount_;
}
std::vector<int> GPUStressSession::getThreadsCores() const {
  return threadIds_;
}

void GPUStressSession::setDropFrameInterval(int numberOfFramesBetweenDrops) {
  dropFrameX_ = numberOfFramesBetweenDrops;
}

int GPUStressSession::getDropFrameInterval() const {
  return dropFrameX_;
}

void GPUStressSession::setDropFrameCount(int numberOfFramesToDrop) {
  dropFrameCount_ = numberOfFramesToDrop;
}

int GPUStressSession::getDropFrameCount() const {
  return dropFrameCount_;
}

void GPUStressSession::setRotateCubes(bool bRotate) {
  rotateCubes_ = bRotate;
}

bool GPUStressSession::getRotateCubes() const {
  return rotateCubes_;
}

std::string GPUStressSession::getCurrentUsageString() const {
  char output[2048];

  std::snprintf(output,
                sizeof(output),
                "cubes: %d, draws: %d, lights: %d, threads: %d,  cpu load: %d, memory reads: %lu , "
                "memory writes: %lu, "
                "msaa %s , blending %s, framerate: %.2f,",
                cubeCount_.load(),
                drawCount_.load(),
                lightCount_.load(),
                threadCount_.load(),
                goSlowOnCpu_.load(),
                memoryReads_.load() * (thrashMemory_ ? 1 : 0),
                memoryWrites_.load() * (thrashMemory_ ? 1 : 0),
                useMSAA_ ? "on" : "off",
                enableBlending_ ? "on" : "off ",
                fps_.getAverageFPS());

  return output;
}
void GPUStressSession::setNumLayers(size_t numLayers) {
#if !defined(IGL_PLATFORM_WINDOWS)
  igl::shell::QuadLayerParams params;
  params.layerInfo.reserve(numLayers);
  for (int i = 0; i < numLayers; i++) {
    params.layerInfo.emplace_back({.position = {0.f, 0.f, 0.f},
                                   .size = {1.f, 1.f},
                                   .blendMode = igl::shell::LayerBlendMode::AlphaBlend});
  }

  appParamsRef().quadLayerParamsGetter = [params]() -> igl::shell::QuadLayerParams {
    return params;
  };
#endif
}
} // namespace igl::shell
