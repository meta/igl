/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/metal/RenderPipelineReflection.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"

#include <igl/Device.h>
#include <igl/IGL.h>
#include <igl/metal/Shader.h>
#include <igl/metal/VertexInputState.h>

namespace igl::tests {

class RenderPipelineReflectionMTLTest : public ::testing::Test {
 public:
  RenderPipelineReflectionMTLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);
    Result ret;

    auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
        *iglDev_,
        std::string(data::shader::kMtlSimpleShader).c_str(),
        {
            {.stage = ShaderStage::Vertex,
             .entryPoint = std::string(data::shader::kSimpleVertFunc)},
            {.stage = ShaderStage::Fragment,
             .entryPoint = std::string(data::shader::kSimpleFragFunc)},
        },
        "",
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    // Initialize vertex Shader
    vertShader_ = shaderLibrary->getShaderModule(ShaderStage::Vertex,
                                                 std::string(data::shader::kSimpleVertFunc));
    ASSERT_TRUE(vertShader_ != nullptr);

    // Initialize Fragment Shader
    fragShader_ = shaderLibrary->getShaderModule(ShaderStage::Fragment,
                                                 std::string(data::shader::kSimpleFragFunc));
    ASSERT_TRUE(fragShader_ != nullptr);

    // Initialize input to vertex shader
    // numAttributes has to equal numInputBindings when using more than one buffer
    const VertexInputStateDesc inputDesc{
        .numAttributes = 2,
        .attributes = {{.bufferIndex = data::shader::kSimplePosIndex,
                        .format = VertexAttributeFormat::Float4,
                        .offset = 0,
                        .name = std::string(data::shader::kSimplePos),
                        .location = 0},
                       {.bufferIndex = data::shader::kSimpleUvIndex,
                        .format = VertexAttributeFormat::Float2,
                        .offset = 0,
                        .name = std::string(data::shader::kSimpleUv),
                        .location = 1}},
        .numInputBindings = 2,
        .inputBindings = {{.stride = sizeof(float) * 4}, {.stride = sizeof(float) * 2}},
    };

    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(vertexInputState_ != nullptr);

    NSError* error = nullptr;
    MTLRenderPipelineDescriptor* metalDesc = [MTLRenderPipelineDescriptor new];
    metalDesc.vertexDescriptor =
        static_cast<metal::VertexInputState*>(vertexInputState_.get())->get();
    metalDesc.vertexFunction = static_cast<metal::ShaderModule*>(vertShader_.get())->get();
    IGL_DEBUG_ASSERT(metalDesc.vertexFunction, "RenderPipeline requires non-null vertex function");
    metalDesc.fragmentFunction = static_cast<metal::ShaderModule*>(fragShader_.get())->get();
    metalDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    metalDesc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;

    MTLRenderPipelineReflection* reflection = nil;
    auto device = MTLCreateSystemDefaultDevice();

// Suppress warnings about MTLPipelineOptionArgumentInfo being deprecated
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    // Create reflection for use later in binding, etc.
    [device newRenderPipelineStateWithDescriptor:metalDesc
                                         options:MTLPipelineOptionArgumentInfo
                                      reflection:&reflection
                                           error:&error];
#pragma GCC diagnostic pop

    pipeRef_ = std::make_shared<metal::RenderPipelineReflection>(reflection);
    ASSERT_NE(pipeRef_, nullptr);
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<metal::RenderPipelineReflection> pipeRef_;

 private:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;

  std::shared_ptr<IShaderModule> vertShader_;
  std::shared_ptr<IShaderModule> fragShader_;
  std::shared_ptr<IVertexInputState> vertexInputState_;
};

TEST_F(RenderPipelineReflectionMTLTest, GetIndexByName) {
  auto index = pipeRef_->getIndexByName(std::string(data::shader::kSimpleUv), ShaderStage::Vertex);
  ASSERT_EQ(index, 1);
}

TEST_F(RenderPipelineReflectionMTLTest, GetNonexistentIndexByName) {
  auto index = pipeRef_->getIndexByName("", ShaderStage::Fragment);
  ASSERT_EQ(index, -1);
}

TEST_F(RenderPipelineReflectionMTLTest, VerifyBuffers) {
  auto buffers = pipeRef_->allUniformBuffers();
  ASSERT_EQ(buffers.size(), 2);
  for (const auto& buffer : buffers) {
    ASSERT_EQ(buffer.shaderStage, ShaderStage::Vertex);
    EXPECT_TRUE(buffer.name.toString() == data::shader::kSimplePos ||
                buffer.name.toString() == data::shader::kSimpleUv);
  }
}

TEST_F(RenderPipelineReflectionMTLTest, VerifyTextures) {
  auto textures = pipeRef_->allTextures();
  ASSERT_EQ(textures.size(), 1);
  const auto& theOneTexture = textures.front();
  ASSERT_EQ(theOneTexture.name, "diffuseTex");
}

TEST_F(RenderPipelineReflectionMTLTest, VerifySamplers) {
  auto samplers = pipeRef_->allSamplers();
  ASSERT_EQ(samplers.size(), 1);
  const auto& theOneSampler = samplers.front();
  ASSERT_EQ(theOneSampler.name, "linearSampler");
}

// Verifies that array-typed uniform-struct members reflect with the correct
// element type, array length, and per-element stride. This mirrors what shaderc
// emits for `uniform vec3 x[N]` (a native MSL C-array `float3 x[N]`), which
// reflects as MTLDataTypeArray with a mappable element type and a real stride
// (16 for float3/float4, 4 for float). See D-plan iglu_metal_array_uniforms.
// The struct also holds `metal::array<T, N>` members, which Slang emits for arrays.
class RenderPipelineReflectionMTLArrayTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);
    Result ret;

    // Fragment shader declares a uniform struct with array members of three
    // element types, each referenced so the buffer argument stays active.
    const char* kArrayShader = R"(using namespace metal;

      typedef struct {
        float3 palette[5];
        float4 rects[3];
        float weights[4];
        array<float4, 2> slots;
        array<float, 3> scalars;
      } ArrayUniforms;

      typedef struct {
        float4 position [[position]];
        float2 uv;
      } VertexOut;

      vertex VertexOut vertexShader(uint vid [[vertex_id]],
                                    constant float4* position_in [[buffer(0)]],
                                    constant float2* uv_in [[buffer(1)]]) {
        VertexOut out;
        out.position = position_in[vid];
        out.uv = uv_in[vid];
        return out;
      }

      fragment float4 fragmentShader(VertexOut IN [[stage_in]],
                                     constant ArrayUniforms& u [[buffer(2)]]) {
        float3 c = u.palette[0] + u.palette[4];
        float4 r = u.rects[1];
        float w = u.weights[2];
        return float4(c * w, r.w) + u.slots[1] * u.scalars[2];
      })";

    auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
        *iglDev_,
        kArrayShader,
        {
            {.stage = ShaderStage::Vertex,
             .entryPoint = std::string(data::shader::kSimpleVertFunc)},
            {.stage = ShaderStage::Fragment,
             .entryPoint = std::string(data::shader::kSimpleFragFunc)},
        },
        "",
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    vertShader_ = shaderLibrary->getShaderModule(ShaderStage::Vertex,
                                                 std::string(data::shader::kSimpleVertFunc));
    ASSERT_TRUE(vertShader_ != nullptr);
    fragShader_ = shaderLibrary->getShaderModule(ShaderStage::Fragment,
                                                 std::string(data::shader::kSimpleFragFunc));
    ASSERT_TRUE(fragShader_ != nullptr);

    // numAttributes has to equal numInputBindings when using more than one buffer
    const VertexInputStateDesc inputDesc{
        .numAttributes = 2,
        .attributes = {{.bufferIndex = data::shader::kSimplePosIndex,
                        .format = VertexAttributeFormat::Float4,
                        .offset = 0,
                        .name = std::string(data::shader::kSimplePos),
                        .location = 0},
                       {.bufferIndex = data::shader::kSimpleUvIndex,
                        .format = VertexAttributeFormat::Float2,
                        .offset = 0,
                        .name = std::string(data::shader::kSimpleUv),
                        .location = 1}},
        .numInputBindings = 2,
        .inputBindings = {{.stride = sizeof(float) * 4}, {.stride = sizeof(float) * 2}},
    };

    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_TRUE(vertexInputState_ != nullptr);

    NSError* error = nullptr;
    MTLRenderPipelineDescriptor* metalDesc = [MTLRenderPipelineDescriptor new];
    metalDesc.vertexDescriptor =
        static_cast<metal::VertexInputState*>(vertexInputState_.get())->get();
    metalDesc.vertexFunction = static_cast<metal::ShaderModule*>(vertShader_.get())->get();
    metalDesc.fragmentFunction = static_cast<metal::ShaderModule*>(fragShader_.get())->get();
    metalDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

    MTLRenderPipelineReflection* reflection = nil;
    auto device = MTLCreateSystemDefaultDevice();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    [device newRenderPipelineStateWithDescriptor:metalDesc
                                         options:MTLPipelineOptionArgumentInfo
                                      reflection:&reflection
                                           error:&error];
#pragma GCC diagnostic pop

    pipeRef_ = std::make_shared<metal::RenderPipelineReflection>(reflection);
    ASSERT_NE(pipeRef_, nullptr);
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<metal::RenderPipelineReflection> pipeRef_;

 private:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<IShaderModule> vertShader_;
  std::shared_ptr<IShaderModule> fragShader_;
  std::shared_ptr<IVertexInputState> vertexInputState_;
};

TEST_F(RenderPipelineReflectionMTLArrayTest, VerifyArrayStrides) {
  auto buffers = pipeRef_->allUniformBuffers();
  const BufferArgDesc* arrayBuffer = nullptr;
  for (const auto& buffer : buffers) {
    if (buffer.shaderStage == ShaderStage::Fragment && !buffer.members.empty()) {
      arrayBuffer = &buffer;
      break;
    }
  }
  ASSERT_NE(arrayBuffer, nullptr) << "fragment uniform buffer not found in reflection";

  const BufferArgDesc::BufferMemberDesc* palette = nullptr;
  const BufferArgDesc::BufferMemberDesc* rects = nullptr;
  const BufferArgDesc::BufferMemberDesc* weights = nullptr;
  for (const auto& member : arrayBuffer->members) {
    const auto name = member.name.toString();
    if (name == "palette") {
      palette = &member;
    } else if (name == "rects") {
      rects = &member;
    } else if (name == "weights") {
      weights = &member;
    }
  }

  ASSERT_NE(palette, nullptr);
  EXPECT_EQ(palette->type, UniformType::Float3);
  EXPECT_EQ(palette->arrayLength, 5u);
  EXPECT_EQ(palette->arrayStride, 16u);

  ASSERT_NE(rects, nullptr);
  EXPECT_EQ(rects->type, UniformType::Float4);
  EXPECT_EQ(rects->arrayLength, 3u);
  EXPECT_EQ(rects->arrayStride, 16u);

  ASSERT_NE(weights, nullptr);
  EXPECT_EQ(weights->type, UniformType::Float);
  EXPECT_EQ(weights->arrayLength, 4u);
  EXPECT_EQ(weights->arrayStride, 4u);
}

// `metal::array<T, N>` reflects as a struct wrapping a C array, and maps to the C array's element
// type, length and stride.
TEST_F(RenderPipelineReflectionMTLArrayTest, VerifyMetalArrayTemplateMembers) {
  const BufferArgDesc::BufferMemberDesc* slots = nullptr;
  const BufferArgDesc::BufferMemberDesc* scalars = nullptr;
  for (const auto& buffer : pipeRef_->allUniformBuffers()) {
    if (buffer.shaderStage != ShaderStage::Fragment) {
      continue;
    }
    for (const auto& member : buffer.members) {
      const auto name = member.name.toString();
      if (name == "slots") {
        slots = &member;
      } else if (name == "scalars") {
        scalars = &member;
      }
    }
  }

  ASSERT_NE(slots, nullptr);
  EXPECT_EQ(slots->type, UniformType::Float4);
  EXPECT_EQ(slots->arrayLength, 2u);
  EXPECT_EQ(slots->arrayStride, 16u);

  ASSERT_NE(scalars, nullptr);
  EXPECT_EQ(scalars->type, UniformType::Float);
  EXPECT_EQ(scalars->arrayLength, 3u);
  EXPECT_EQ(scalars->arrayStride, 4u);
}

} // namespace igl::tests
