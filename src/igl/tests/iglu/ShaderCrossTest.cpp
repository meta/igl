/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IGLU/shaderCross/ShaderCross.h>

#include "../util/Common.h"

#include <IGLU/shaderCross/ShaderCrossUniformBuffer.h>

namespace igl::tests {

namespace {
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

[[nodiscard]] const char* getVulkanSubgroupArithmeticComputeShaderSource() {
  return R"(#version 450
              #extension GL_KHR_shader_subgroup_arithmetic : require

              layout(local_size_x = 64) in;

              layout(set = 0, binding = 0, std430) buffer Values {
                uint values[];
              } data;

              void main() {
                const uint index = gl_GlobalInvocationID.x;
                data.values[index] = subgroupExclusiveAdd(data.values[index]);
              })";
}

[[nodiscard]] const char* getVulkanSubpassInputFragmentShaderSource() {
  return R"(#version 450
              layout(input_attachment_index = 1, set = 2, binding = 17) uniform subpassInput albedo;
              layout(location = 0) out vec4 fragmentColor;

              void main() {
                fragmentColor = subpassLoad(albedo);
              })";
}
} // namespace

class ShaderCrossTest : public ::testing::Test {
 public:
  ShaderCrossTest() = default;

  // Set up common resources. This will create a device and a command queue
  void SetUp() override {
    // Turn off debug break so unit tests can run
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

TEST_F(ShaderCrossTest, Construction) {
  const iglu::ShaderCross shaderCross(*iglDev_);
}

TEST_F(ShaderCrossTest, EntryPointName) {
  const iglu::ShaderCross shaderCross(*iglDev_);
  if (iglDev_->getBackendType() == BackendType::Metal) {
    EXPECT_EQ(shaderCross.entryPointName(igl::ShaderStage::Vertex), "main0");
  } else if (iglDev_->getBackendType() == BackendType::OpenGL) {
    EXPECT_EQ(shaderCross.entryPointName(igl::ShaderStage::Vertex), "main");
  } else {
    EXPECT_EQ(shaderCross.entryPointName(igl::ShaderStage::Vertex), "");
  }
}

TEST_F(ShaderCrossTest, CrossCompile) {
  const iglu::ShaderCross shaderCross(*iglDev_);
  Result res;
  if (iglDev_->getBackendType() == BackendType::Metal) {
    const auto vs = shaderCross.crossCompileFromVulkanSource(
        getVulkanVertexShaderSource(false).c_str(), ShaderStage::Vertex, &res);
    EXPECT_TRUE(res.isOk());
    EXPECT_TRUE(!vs.empty());

    const auto fs = shaderCross.crossCompileFromVulkanSource(
        getVulkanFragmentShaderSource(), ShaderStage::Fragment, &res);
    EXPECT_TRUE(res.isOk());
    EXPECT_TRUE(!fs.empty());

  } else if (iglDev_->getBackendType() == BackendType::OpenGL) {
    const auto vs = shaderCross.crossCompileFromVulkanSource(
        getVulkanVertexShaderSource(true).c_str(), ShaderStage::Vertex, &res);
    EXPECT_TRUE(res.isOk());
    EXPECT_TRUE(!vs.empty());

    const auto fs = shaderCross.crossCompileFromVulkanSource(
        getVulkanFragmentShaderSource(), ShaderStage::Fragment, &res);
    EXPECT_TRUE(res.isOk());
    EXPECT_TRUE(!fs.empty());
  }
}

TEST_F(ShaderCrossTest, CrossCompileSubgroupArithmetic) {
  if (iglDev_->getBackendType() != BackendType::Metal) {
    GTEST_SKIP() << "Subgroup arithmetic cross-compilation is only gated on the Metal path";
  }

  // subgroupExclusiveAdd() has no quadgroup equivalent, so SPIRV-Cross rejects it on iOS unless
  // the target is MSL 2.3+ *and* SIMD-group functions are enabled. Both halves are required; the
  // error text only mentions the version.
  const iglu::ShaderCross shaderCross(*iglDev_, {.iosUseSimdgroupFunctions = true});

  Result res;
  const auto cs = shaderCross.crossCompileFromVulkanSource(
      getVulkanSubgroupArithmeticComputeShaderSource(), ShaderStage::Compute, &res);
  EXPECT_TRUE(res.isOk()) << res.message;
  EXPECT_TRUE(!cs.empty());
}

TEST_F(ShaderCrossTest, ShaderCrossUniformBuffer) {
  iglu::ShaderCrossUniformBuffer buffer(*iglDev_,
                                        "perFrame",
                                        {.index = 0,
                                         .length = 10,
                                         .uniforms = {{.name = "myUniform",
                                                       .location = 0,
                                                       .type = UniformType::Float,
                                                       .numElements = 1,
                                                       .offset = 0,
                                                       .elementStride = 0}}});
  EXPECT_EQ(buffer.uniformInfo.uniforms.size(), 1);
  EXPECT_EQ(buffer.uniformInfo.uniforms[0].name, "perFrame.myUniform");
  // The block name is recorded so the OpenGL bind path can fall back to a real UBO buffer binding
  // when the program keeps the block native (GLSL ES 3.x) instead of flattening it to the
  // "perFrame.myUniform" plain uniform above.
  EXPECT_EQ(buffer.uniformInfo.blockName, "perFrame");
}

TEST_F(ShaderCrossTest, CrossCompileRejectsSubpassInputs) {
  const iglu::ShaderCross shaderCross(*iglDev_);
  Result result;
  const std::string out = shaderCross.crossCompileFromVulkanSource(
      getVulkanSubpassInputFragmentShaderSource(), ShaderStage::Fragment, &result);

  if (iglDev_->getBackendType() == BackendType::Vulkan) {
    EXPECT_TRUE(result.isOk());
    EXPECT_EQ(out, getVulkanSubpassInputFragmentShaderSource());
  } else if (iglDev_->getBackendType() == BackendType::Metal ||
             iglDev_->getBackendType() == BackendType::OpenGL) {
    EXPECT_EQ(result.code, igl::Result::Code::Unsupported) << result.message;
    EXPECT_TRUE(out.empty());
    EXPECT_NE(result.message.find("subpass"), std::string::npos) << result.message;
  } else {
    GTEST_SKIP();
  }
}

} // namespace igl::tests
