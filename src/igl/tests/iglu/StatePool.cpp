/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"

#include <IGLU/state_pool/DepthStencilStatePool.h>
#include <IGLU/state_pool/RenderPipelineStatePool.h>
#include <string>
#include <igl/CommandBuffer.h>
#include <igl/NameHandle.h>
#include <igl/VertexInputState.h>

namespace igl::tests {

//
// StatePoolTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class StatePoolTest : public ::testing::Test {
 private:
 public:
  StatePoolTest() = default;

  //
  // SetUp()
  //
  // This function sets up two identical graphics pipeline descriptors
  // so that they can be modified by individual tests to exercise the
  // state pool caching capabilities
  //
  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    Result ret;

    // Initialize shader stages
    std::unique_ptr<IShaderStages> stages;
    igl::tests::util::createSimpleShaderStages(iglDev_, stages);
    shaderStages_ = std::move(stages);

    // Initialize input to vertex shader
    VertexInputStateDesc inputDesc;

    inputDesc.attributes[0].format = VertexAttributeFormat::Float4;
    inputDesc.attributes[0].offset = 0;
    inputDesc.attributes[0].location = 0;
    inputDesc.attributes[0].bufferIndex = data::shader::kSimplePosIndex;
    inputDesc.attributes[0].name = std::string(data::shader::kSimplePos);
    inputDesc.inputBindings[0].stride = sizeof(float) * 4;

    inputDesc.attributes[1].format = VertexAttributeFormat::Float2;
    inputDesc.attributes[1].offset = 0;
    inputDesc.attributes[1].location = 1;
    inputDesc.attributes[1].bufferIndex = data::shader::kSimpleUvIndex;
    inputDesc.attributes[1].name = std::string(data::shader::kSimpleUv);
    inputDesc.inputBindings[1].stride = sizeof(float) * 2;

    // numAttributes has to equal to bindings when using more than 1 buffer
    inputDesc.numAttributes = inputDesc.numInputBindings = 2;

    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(vertexInputState_ != nullptr);

    // Initialize Graphics Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    renderPipelineDesc1_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
    };

    renderPipelineDesc2_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
    };

    renderPipelineDesc3_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
    };
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ICommandBuffer> cmdBuf_;
  CommandBufferDesc cbDesc_ = {};

  std::shared_ptr<IShaderStages> shaderStages_;

  std::shared_ptr<IVertexInputState> vertexInputState_;

  RenderPipelineDesc renderPipelineDesc1_, renderPipelineDesc2_, renderPipelineDesc3_;
  iglu::state_pool::RenderPipelineStatePool graphicsPool_;
};

//
// renderPipelineDescCaching1 Test
//
// Tests to see if RenderPipelineDesc caching works
//
TEST_F(StatePoolTest, renderPipelineDescCaching1) {
  Result ret;
  std::shared_ptr<IRenderPipelineState> ps1, ps2;

  //---------------------------------------------------------------------
  // Create two pipelines without cache. Should get two different objects
  //---------------------------------------------------------------------
  ps1 = iglDev_->createRenderPipeline(renderPipelineDesc1_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps1 != nullptr);

  ps2 = iglDev_->createRenderPipeline(renderPipelineDesc2_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps2 != nullptr);

  ASSERT_TRUE(ps1.get() != ps2.get());

  //------------------------------------------------------------
  // Create two pipelines with cache. Should get the same object
  //------------------------------------------------------------
  ps1 = graphicsPool_.getOrCreate(*iglDev_, renderPipelineDesc1_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps1 != nullptr);

  ps2 = graphicsPool_.getOrCreate(*iglDev_, renderPipelineDesc2_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps2 != nullptr);

  ASSERT_TRUE(ps1 == ps2);

  //------------------------------------------------------------
  // Modify one of the descriptors, should get different objects
  //------------------------------------------------------------
  renderPipelineDesc2_.cullMode = CullMode::Front;

  ps2 = graphicsPool_.getOrCreate(*iglDev_, renderPipelineDesc2_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps2 != nullptr);

  ASSERT_TRUE(ps1 != ps2);
  renderPipelineDesc2_.cullMode = renderPipelineDesc1_.cullMode; // restore change
}

//
// renderPipelineDescCachingLRU1 Test
//
// Tests to see if RenderPipelineDesc LRU caching works
//
TEST_F(StatePoolTest, renderPipelineDescCachingLRU1) {
  Result ret;
  std::shared_ptr<IRenderPipelineState> ps1, ps2, ps3;
  //------------------------------------------------------------
  // Ensure cache releases objects after it gets full
  //------------------------------------------------------------
  iglu::state_pool::RenderPipelineStatePool smallCachePool;
  smallCachePool.setCacheSize(2);
  renderPipelineDesc2_.cullMode = CullMode::Front;
  renderPipelineDesc3_.cullMode = CullMode::Back;

  // Fill up the cache
  ps1 = iglDev_->createRenderPipeline(renderPipelineDesc1_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps1 != nullptr);

  ps2 = iglDev_->createRenderPipeline(renderPipelineDesc2_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps2 != nullptr);

  // Add new element (PS1 should no longer be stored in the cache)
  ps3 = iglDev_->createRenderPipeline(renderPipelineDesc3_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps3 != nullptr);

  // Adding renderPipelineDesc1_ again should produce a new state than before
  ps2 = iglDev_->createRenderPipeline(renderPipelineDesc1_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ps2 != nullptr);
  ASSERT_TRUE(ps1 != ps2);

  renderPipelineDesc2_.cullMode = renderPipelineDesc1_.cullMode; // restore change
  renderPipelineDesc3_.cullMode = renderPipelineDesc1_.cullMode; // restore change
}

//
// depthStencilStateCaching Test
//
// Tests that DepthStencilStatePool returns the same cached object for identical
// descriptors and distinct objects for differing descriptors.
//
TEST_F(StatePoolTest, depthStencilStateCaching) {
  Result ret;
  iglu::state_pool::DepthStencilStatePool pool;

  const DepthStencilStateDesc descA{.compareFunction = CompareFunction::Less,
                                    .isDepthWriteEnabled = true};

  // Identical to descA - should map to the same cached state object
  const DepthStencilStateDesc& descB = descA;

  const DepthStencilStateDesc descC{.compareFunction = CompareFunction::Greater,
                                    .isDepthWriteEnabled = false};

  //------------------------------------------------------------
  // Identical descriptors should return the same cached object
  //------------------------------------------------------------
  std::shared_ptr<IDepthStencilState> s1 = pool.getOrCreate(*iglDev_, descA, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(s1 != nullptr);

  std::shared_ptr<IDepthStencilState> s2 = pool.getOrCreate(*iglDev_, descB, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(s2 != nullptr);

  ASSERT_TRUE(s1 == s2);

  //------------------------------------------------------------
  // A different descriptor should return a different object
  //------------------------------------------------------------
  std::shared_ptr<IDepthStencilState> s3 = pool.getOrCreate(*iglDev_, descC, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(s3 != nullptr);

  ASSERT_TRUE(s1 != s3);
}

//
// depthStencilStateCachingLRU Test
//
// Tests that DepthStencilStatePool evicts the least-recently-used entry once the
// cache is full, so re-fetching an evicted descriptor produces a fresh object.
//
TEST_F(StatePoolTest, depthStencilStateCachingLRU) {
  Result ret;
  iglu::state_pool::DepthStencilStatePool pool;
  pool.setCacheSize(2);

  const DepthStencilStateDesc descA{.compareFunction = CompareFunction::Less};
  const DepthStencilStateDesc descB{.compareFunction = CompareFunction::Greater};
  const DepthStencilStateDesc descC{.compareFunction = CompareFunction::Equal};

  // Insert A, then confirm a second lookup hits the cache (same object)
  std::shared_ptr<IDepthStencilState> a1 = pool.getOrCreate(*iglDev_, descA, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(a1 != nullptr);

  std::shared_ptr<IDepthStencilState> a2 = pool.getOrCreate(*iglDev_, descA, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(a1 == a2);

  // Cache holds {A}. Insert B -> cache holds {B, A}.
  std::shared_ptr<IDepthStencilState> b1 = pool.getOrCreate(*iglDev_, descB, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(b1 != nullptr);

  // Cache is full (size 2). Inserting C evicts the least-recently-used entry (A).
  std::shared_ptr<IDepthStencilState> c1 = pool.getOrCreate(*iglDev_, descC, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(c1 != nullptr);

  // A was evicted, so re-fetching it must produce a fresh object.
  std::shared_ptr<IDepthStencilState> a3 = pool.getOrCreate(*iglDev_, descA, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(a3 != nullptr);
  ASSERT_TRUE(a1 != a3);
}

} // namespace igl::tests
