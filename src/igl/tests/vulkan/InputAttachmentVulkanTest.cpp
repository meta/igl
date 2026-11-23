/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/TestDevice.h"

#include <igl/Device.h>
#include <igl/Framebuffer.h>
#include <igl/RenderPipelineState.h>
#include <igl/ShaderCreator.h>
#include <igl/Texture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class InputAttachmentVulkanTest : public ::testing::Test {
 public:
  InputAttachmentVulkanTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
};

TEST_F(InputAttachmentVulkanTest, RenderPipelineWithInputAttachmentCreates) {
  Result ret;

  // Create color attachment textures
  TextureDesc colorDesc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, 256, 256, TextureDesc::TextureUsageBits::Attachment);
  auto colorTex0 = iglDev_->createTexture(colorDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(colorTex0, nullptr);

  auto colorTex1 = iglDev_->createTexture(colorDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(colorTex1, nullptr);

  // Create framebuffer with two color attachments
  const FramebufferDesc fbDesc{
      .colorAttachments = {{.texture = colorTex0}, {.texture = colorTex1}}};
  auto fb = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(fb, nullptr);

  // Vertex shader for fullscreen triangle from gl_VertexIndex
  const char* vertexSource = R"(#version 460
out gl_PerVertex {
  vec4 gl_Position;
};
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

  // Fragment shader with input attachment
  const char* fragmentSource = R"(#version 460
layout(input_attachment_index = 1, set = 2, binding = 17) uniform subpassInput inA;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = subpassLoad(inA);
})";

  // Create shader stages
  auto shaderStages = ShaderStagesCreator::fromModuleStringInput(*iglDev_,
                                                                 vertexSource,
                                                                 "main",
                                                                 "VertexShader",
                                                                 fragmentSource,
                                                                 "main",
                                                                 "FragmentShader",
                                                                 &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(shaderStages, nullptr);

  // Create render pipeline with input attachments
  RenderPipelineDesc pipelineDesc;
  pipelineDesc.shaderStages = std::move(shaderStages);
  pipelineDesc.targetDesc.colorAttachments.resize(2);
  pipelineDesc.targetDesc.colorAttachments[0].textureFormat = TextureFormat::RGBA_UNorm8;
  pipelineDesc.targetDesc.colorAttachments[1].textureFormat = TextureFormat::RGBA_UNorm8;

  auto pipeline = iglDev_->createRenderPipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);
}

TEST_F(InputAttachmentVulkanTest, RejectsInputAttachmentWrongBinding) {
  Result ret;

  const char* vertexSource = R"(#version 460
out gl_PerVertex {
  vec4 gl_Position;
};
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

  // Index 1 at binding 18 (wrong: should be 17 = 16 + 1)
  const char* fragmentSource = R"(#version 460
layout(input_attachment_index = 1, set = 2, binding = 18) uniform subpassInput inA;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = subpassLoad(inA);
})";

  auto shaderStages = ShaderStagesCreator::fromModuleStringInput(*iglDev_,
                                                                 vertexSource,
                                                                 "main",
                                                                 "VertexShader",
                                                                 fragmentSource,
                                                                 "main",
                                                                 "FragmentShader",
                                                                 &ret);
  EXPECT_FALSE(ret.isOk());
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

TEST_F(InputAttachmentVulkanTest, RejectsInputAttachmentWrongSet) {
  Result ret;

  const char* vertexSource = R"(#version 460
out gl_PerVertex {
  vec4 gl_Position;
};
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

  // Set 0 instead of 2
  const char* fragmentSource = R"(#version 460
layout(input_attachment_index = 1, set = 0, binding = 17) uniform subpassInput inA;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = subpassLoad(inA);
})";

  auto shaderStages = ShaderStagesCreator::fromModuleStringInput(*iglDev_,
                                                                 vertexSource,
                                                                 "main",
                                                                 "VertexShader",
                                                                 fragmentSource,
                                                                 "main",
                                                                 "FragmentShader",
                                                                 &ret);
  EXPECT_FALSE(ret.isOk());
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

TEST_F(InputAttachmentVulkanTest, RejectsInputAttachmentIndexOutOfRange) {
  Result ret;

  const char* vertexSource = R"(#version 460
out gl_PerVertex {
  vec4 gl_Position;
};
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

  // Index 4 (out of range, max is 3)
  const char* fragmentSource = R"(#version 460
layout(input_attachment_index = 4, set = 2, binding = 20) uniform subpassInput inA;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = subpassLoad(inA);
})";

  auto shaderStages = ShaderStagesCreator::fromModuleStringInput(*iglDev_,
                                                                 vertexSource,
                                                                 "main",
                                                                 "VertexShader",
                                                                 fragmentSource,
                                                                 "main",
                                                                 "FragmentShader",
                                                                 &ret);
  EXPECT_FALSE(ret.isOk());
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

TEST_F(InputAttachmentVulkanTest, RejectsDuplicateInputAttachmentIndex) {
  Result ret;

  const char* vertexSource = R"(#version 460
out gl_PerVertex {
  vec4 gl_Position;
};
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

  // Two input attachments with same index
  const char* fragmentSource = R"(#version 460
layout(input_attachment_index = 1, set = 2, binding = 17) uniform subpassInput inA;
layout(input_attachment_index = 1, set = 2, binding = 18) uniform subpassInput inB;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = subpassLoad(inA);
})";

  auto shaderStages = ShaderStagesCreator::fromModuleStringInput(*iglDev_,
                                                                 vertexSource,
                                                                 "main",
                                                                 "VertexShader",
                                                                 fragmentSource,
                                                                 "main",
                                                                 "FragmentShader",
                                                                 &ret);
  EXPECT_FALSE(ret.isOk());
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

TEST_F(InputAttachmentVulkanTest, RejectsSubpassInputArray) {
  Result ret;

  const char* vertexSource = R"(#version 460
out gl_PerVertex {
  vec4 gl_Position;
};
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

  // Array of subpass inputs
  const char* fragmentSource = R"(#version 460
layout(input_attachment_index = 0, set = 2, binding = 16) uniform subpassInput inA[2];
layout(location = 0) out vec4 outColor;
void main() {
  outColor = subpassLoad(inA[0]);
})";

  auto shaderStages = ShaderStagesCreator::fromModuleStringInput(*iglDev_,
                                                                 vertexSource,
                                                                 "main",
                                                                 "VertexShader",
                                                                 fragmentSource,
                                                                 "main",
                                                                 "FragmentShader",
                                                                 &ret);
  EXPECT_FALSE(ret.isOk());
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

TEST_F(InputAttachmentVulkanTest, RejectsSetTwoStorageImageAtInputBinding) {
  Result ret;

  const char* vertexSource = R"(#version 460
out gl_PerVertex {
  vec4 gl_Position;
};
void main() {
  vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";

  // Storage image at set 2 binding 16 (input attachment binding range)
  const char* fragmentSource = R"(#version 460
layout(set = 2, binding = 16, rgba8) uniform image2D img;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(0.0);
})";

  auto shaderStages = ShaderStagesCreator::fromModuleStringInput(*iglDev_,
                                                                 vertexSource,
                                                                 "main",
                                                                 "VertexShader",
                                                                 fragmentSource,
                                                                 "main",
                                                                 "FragmentShader",
                                                                 &ret);
  EXPECT_FALSE(ret.isOk());
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

} // namespace igl::tests

#endif
