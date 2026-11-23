/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/RenderPipelineReflection.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"

#include <igl/Device.h>
#include <igl/RenderPipelineState.h>
#include <igl/VertexInputState.h>

namespace igl::tests {

// Use a 4x4 texture for this test
#define OFFSCREEN_TEX_WIDTH 4
#define OFFSCREEN_TEX_HEIGHT 4

class RenderPipelineReflectionTest : public ::testing::Test {
 private:
 public:
  RenderPipelineReflectionTest() = default;

  //
  // SetUp()
  //
  void SetUp() override {
    // Turn off debug breaks, only use in debug mode
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    // Create an offscreen texture to render to
    const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                   OFFSCREEN_TEX_WIDTH,
                                                   OFFSCREEN_TEX_HEIGHT,
                                                   TextureDesc::TextureUsageBits::Sampled |
                                                       TextureDesc::TextureUsageBits::Attachment);

    Result ret;
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(offscreenTexture_ != nullptr);

    // Initialize input to vertex shader
    // numAttributes has to equal numInputBindings when using more than one buffer
    const VertexInputStateDesc inputDesc{
        .numAttributes = 1,
        .attributes = {{.bufferIndex = data::shader::kSimplePosIndex,
                        .format = VertexAttributeFormat::Float4,
                        .offset = 0,
                        .name = std::string(data::shader::kSimplePos),
                        .location = 0}},
        .numInputBindings = 1,
        .inputBindings = {{.stride = sizeof(float) * 4}},
    };

    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(vertexInputState_ != nullptr);

    std::unique_ptr<IShaderStages> stages;
    util::createShaderStages(iglDev_,
                             data::shader::kOglSimpleVertShaderCube,
                             "vertexShader",
                             data::shader::kOglSimpleFragShaderCube,
                             "fragmentShader",
                             stages);
    ASSERT_TRUE(stages != nullptr);
    std::shared_ptr<IShaderStages> shaderStages;
    shaderStages = std::move(stages);

    // Initialize Render Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    const RenderPipelineDesc renderPipelineDesc{.vertexInputState = vertexInputState_,
                                                .shaderStages = shaderStages};

    //----------------
    // Create Pipeline
    //----------------
    pipelineState_ = iglDev_->createRenderPipeline(renderPipelineDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(pipelineState_ != nullptr);

    pipeRef_ = static_cast<opengl::RenderPipelineReflection*>(
        pipelineState_->renderPipelineReflection().get());
    ASSERT_TRUE(pipeRef_ != nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  const std::string backend_ = IGL_BACKEND_TYPE;

  std::shared_ptr<ITexture> offscreenTexture_;

  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IRenderPipelineState> pipelineState_;
  opengl::RenderPipelineReflection* pipeRef_{};
};

TEST_F(RenderPipelineReflectionTest, GetIndexByName) {
  auto index = pipeRef_->getIndexByName(IGL_NAMEHANDLE(data::shader::kSimpleCubeView));
  ASSERT_TRUE(index >= 0);
}

TEST_F(RenderPipelineReflectionTest, GetNonexistentIndexByName) {
  auto index = pipeRef_->getIndexByName(IGL_NAMEHANDLE("ZYA"));
  ASSERT_EQ(index, -1);
}

TEST_F(RenderPipelineReflectionTest, CheckUniformDictionary) {
  ASSERT_EQ(pipeRef_->allUniformBuffers().size(), 1);
  ASSERT_EQ(pipeRef_->allSamplers().size(), 1);
  ASSERT_EQ(pipeRef_->allTextures().size(), 1);
}

TEST_F(RenderPipelineReflectionTest, VerifyBuffers) {
  auto buffers = pipeRef_->allUniformBuffers();
  ASSERT_EQ(buffers.size(), 1);
  for (const auto& buffer : buffers) {
    ASSERT_EQ(buffer.shaderStage,
              ShaderStage::Fragment); // all uniforms are set to Fragment stage in OpenGL
    EXPECT_TRUE(buffer.name.toString() == data::shader::kSimpleCubeView);
  }
}

TEST_F(RenderPipelineReflectionTest, VerifyTextures) {
  auto textures = pipeRef_->allTextures();
  ASSERT_EQ(textures.size(), 1);
  const auto& theOneTexture = textures.front();
  ASSERT_EQ(theOneTexture.name, "inputImage");
}

TEST_F(RenderPipelineReflectionTest, VerifySamplers) {
  auto samplers = pipeRef_->allSamplers();
  ASSERT_EQ(samplers.size(), 1);
  const auto& theOneSampler = samplers.front();
  ASSERT_EQ(theOneSampler.name, "inputImage");
}

TEST_F(RenderPipelineReflectionTest, UniformBlocks) {
  bool useBlocks = iglDev_->hasFeature(DeviceFeatures::UniformBlocks);
  const bool isGles3 = iglDev_->getBackendVersion().flavor == BackendFlavor::OpenGL_ES &&
                       iglDev_->getBackendVersion().majorVersion >= 3;

#if defined(IGL_PLATFORM_LINUX) && IGL_PLATFORM_LINUX
  useBlocks = !isGles3;
#endif
  if (!useBlocks) {
    GTEST_SKIP() << "Uniform blocks not supported";
  }

  if (!isGles3) {
    return;
  }
  std::unique_ptr<IShaderStages> stages;
  util::createShaderStages(iglDev_,
                           data::shader::kOglSimpleVertShaderUniformBlocks,
                           "vertexShader",
                           data::shader::kOglSimpleFragShaderUniformBlocks,
                           "fragmentShader",
                           stages);
  ASSERT_TRUE(stages != nullptr);
  std::shared_ptr<IShaderStages> shaderStages;
  shaderStages = std::move(stages);

  // Initialize Render Pipeline Descriptor, but leave the creation
  // to the individual tests in case further customization is required
  const RenderPipelineDesc renderPipelineDesc{.vertexInputState = vertexInputState_,
                                              .shaderStages = shaderStages};
  //----------------
  // Create Pipeline
  //----------------
  Result ret;
  auto pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(pipelineState != nullptr);

  auto* pipeRef = static_cast<opengl::RenderPipelineReflection*>(
      pipelineState->renderPipelineReflection().get());
  ASSERT_TRUE(pipeRef != nullptr);

  ASSERT_TRUE(pipeRef->getIndexByName(IGL_NAMEHANDLE("block_without_instance_name")) >= 0);
  ASSERT_TRUE(pipeRef->getIndexByName(IGL_NAMEHANDLE("block_with_instance_name")) >= 0);
  ASSERT_EQ(pipeRef->allSamplers().size(), 1);
  ASSERT_EQ(pipeRef->allTextures().size(), 1);

  const auto& uniformDict = pipeRef->getUniformDictionary();
  ASSERT_EQ(uniformDict.size(), 2);
  const auto& uniformBlocksDict = pipeRef->getUniformBlocksDictionary();
  ASSERT_EQ(uniformBlocksDict.size(), 2);

  auto buffers = pipeRef->allUniformBuffers();
  ASSERT_EQ(buffers.size(), 3);
  for (auto& buffer : buffers) {
    if (buffer.name.toString() == "block_without_instance_name") {
      ASSERT_EQ(buffer.isUniformBlock, true);
      ASSERT_EQ(buffer.members.size(), 1);
      ASSERT_EQ(buffer.members[0].type, igl::UniformType::Float);
      ASSERT_EQ(buffer.members[0].offset, 0);
      ASSERT_EQ(buffer.members[0].arrayLength, 1);
    } else if (buffer.name.toString() == "block_with_instance_name") {
      ASSERT_EQ(buffer.isUniformBlock, true);
      ASSERT_EQ(buffer.members.size(), 2);
      for (const auto& member : buffer.members) {
        if (member.name.toString() == "view") {
          ASSERT_EQ(member.type, igl::UniformType::Float3);
          ASSERT_EQ(member.offset, 0);
          ASSERT_EQ(member.arrayLength, 1);
        } else if (member.name.toString() == "testArray") {
          ASSERT_EQ(member.type, igl::UniformType::Float4);
          ASSERT_EQ(member.offset, 16);
          ASSERT_EQ(member.arrayLength, 2);
        }
      }
    } else if (buffer.name.toString() == "non_uniform_block_bool") {
      ASSERT_EQ(buffer.isUniformBlock, false);
      ASSERT_EQ(buffer.members.size(), 1);
      ASSERT_EQ(buffer.members[0].type, igl::UniformType::Boolean);
      ASSERT_EQ(buffer.members[0].offset, 0);
      ASSERT_EQ(buffer.members[0].arrayLength, 1);
    }
  }
}
} // namespace igl::tests
