/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"

#include <array>
#include <cstring>
#include <igl/Device.h>
#include <igl/RenderPipelineState.h>
#include <igl/VertexInputState.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/RenderPipelineState.h>
#include <igl/opengl/UniformBuffer.h>

namespace igl::tests {

namespace {
// Minimal shader with various uniform types to exercise bindUniform/bindUniformArray
// clang-format off
const char kVertShader[] =
    IGL_TO_STRING(LEGACY_VERSION
               attribute vec4 position_in;
               void main() {
                 gl_Position = position_in;
               });

const char kFragShaderWithUniforms[] =
    IGL_TO_STRING(LEGACY_VERSION PROLOG
               uniform float uFloat;
               uniform vec2 uVec2;
               uniform vec3 uVec3;
               uniform vec4 uVec4;
               uniform int uInt;
               uniform ivec2 uIVec2;
               uniform ivec3 uIVec3;
               uniform ivec4 uIVec4;
               uniform bool uBool;
               uniform mat2 uMat2;
               uniform mat3 uMat3;
               uniform mat4 uMat4;

               void main() {
                 float f = uFloat + uVec2.x + uVec3.x + uVec4.x;
                 f += float(uInt) + float(uIVec2.x) + float(uIVec3.x) + float(uIVec4.x);
                 f += uBool ? 1.0 : 0.0;
                 f += uMat2[0][0] + uMat3[0][0] + uMat4[0][0];
                 gl_FragColor = vec4(f, 0.0, 0.0, 1.0);
               });

const char kFragShaderWithUniformArrays[] =
    IGL_TO_STRING(LEGACY_VERSION PROLOG
               uniform float uFloatArr[2];
               uniform int uIntArr[2];
               uniform bool uBoolArr[2];
               uniform vec4 uVec4Arr[2];
               uniform mat4 uMat4Arr[2];

               void main() {
                 float f = uFloatArr[0] + uFloatArr[1];
                 f += float(uIntArr[0] + uIntArr[1]);
                 f += (uBoolArr[0] ? 1.0 : 0.0) + (uBoolArr[1] ? 1.0 : 0.0);
                 f += uVec4Arr[0].x + uVec4Arr[1].x;
                 f += uMat4Arr[0][0][0] + uMat4Arr[1][0][0];
                 gl_FragColor = vec4(f, 0.0, 0.0, 1.0);
               });
// clang-format on
} // namespace

///
/// UniformBufferBindTest
///
/// Directly exercises UniformBuffer::bindUniform and UniformBuffer::bindUniformArray
/// with a real OpenGL context and shader program, covering all uniform type dispatch
/// paths including reinterpret_cast and static_cast usage.
///
class UniformBufferBindTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);
    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
  }

 protected:
  void createPipeline(const char* fragShader, std::shared_ptr<IRenderPipelineState>& outPipeline) {
    const VertexInputStateDesc inputDesc = {
        .numAttributes = 1,
        .attributes = {{
            .bufferIndex = data::shader::kSimplePosIndex,
            .format = VertexAttributeFormat::Float4,
            .offset = 0,
            .name = std::string(data::shader::kSimplePos),
            .location = 0,
        }},
        .numInputBindings = 1,
        .inputBindings = {{.stride = sizeof(float) * 4}},
    };

    Result ret;
    std::shared_ptr<IVertexInputState> vertexInputState =
        iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    std::unique_ptr<IShaderStages> stages;
    util::createShaderStages(iglDev_, kVertShader, "main", fragShader, "main", stages);
    ASSERT_NE(stages, nullptr);

    RenderPipelineDesc renderPipelineDesc = {
        .vertexInputState = std::move(vertexInputState),
        .shaderStages = std::move(stages),
    };

    outPipeline = iglDev_->createRenderPipeline(renderPipelineDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  }

  [[nodiscard]] GLint getLocation(const std::shared_ptr<IRenderPipelineState>& pipeline,
                                  const char* name) const {
    return pipeline->getIndexByName(genNameHandle(name), ShaderStage::Fragment);
  }

  [[nodiscard]] GLuint usePipeline(const std::shared_ptr<IRenderPipelineState>& pipeline) {
    const auto& glPipeline = static_cast<const opengl::RenderPipelineState&>(*pipeline);
    const GLuint programID = glPipeline.getShaderStages()->getProgramID();
    context_->useProgram(programID);
    return programID;
  }

  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
};

///
/// BindUniformAllTypes
///
/// Exercises bindUniform for every UniformType: Float, Float2..4, Int, Int2..4,
/// Boolean, Mat2x2, Mat3x3, Mat4x4. Covers the reinterpret_cast<const GLfloat*>
/// and reinterpret_cast<const GLint*> paths.
///
TEST_F(UniformBufferBindTest, BindUniformAllTypes) {
  std::shared_ptr<IRenderPipelineState> pipeline;
  createPipeline(kFragShaderWithUniforms, pipeline);
  ASSERT_NE(pipeline, nullptr);
  const GLuint programID = usePipeline(pipeline);

  // Float
  {
    const GLint loc = getLocation(pipeline, "uFloat");
    ASSERT_GE(loc, 0);
    const float val = 1.0f;
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Float, reinterpret_cast<const uint8_t*>(&val), 1);
  }

  // Float2
  {
    const GLint loc = getLocation(pipeline, "uVec2");
    ASSERT_GE(loc, 0);
    const std::array<float, 2> val = {1.0f, 2.0f};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Float2, reinterpret_cast<const uint8_t*>(val.data()), 1);
  }

  // Float3
  {
    const GLint loc = getLocation(pipeline, "uVec3");
    ASSERT_GE(loc, 0);
    const std::array<float, 3> val = {1.0f, 2.0f, 3.0f};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Float3, reinterpret_cast<const uint8_t*>(val.data()), 1);
  }

  // Float4
  {
    const GLint loc = getLocation(pipeline, "uVec4");
    ASSERT_GE(loc, 0);
    const std::array<float, 4> val = {1.0f, 2.0f, 3.0f, 4.0f};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Float4, reinterpret_cast<const uint8_t*>(val.data()), 1);
  }

  // Int
  {
    const GLint loc = getLocation(pipeline, "uInt");
    ASSERT_GE(loc, 0);
    const int val = 42;
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Int, reinterpret_cast<const uint8_t*>(&val), 1);
    GLint readback = 0;
    context_->getUniformiv(programID, loc, &readback);
    EXPECT_EQ(readback, val);
  }

  // Int2
  {
    const GLint loc = getLocation(pipeline, "uIVec2");
    ASSERT_GE(loc, 0);
    const std::array<int, 2> val = {1, 2};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Int2, reinterpret_cast<const uint8_t*>(val.data()), 1);
    std::array<GLint, 2> readback = {};
    context_->getUniformiv(programID, loc, readback.data());
    EXPECT_EQ(readback[0], val[0]);
    EXPECT_EQ(readback[1], val[1]);
  }

  // Int3
  {
    const GLint loc = getLocation(pipeline, "uIVec3");
    ASSERT_GE(loc, 0);
    const std::array<int, 3> val = {1, 2, 3};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Int3, reinterpret_cast<const uint8_t*>(val.data()), 1);
    std::array<GLint, 3> readback = {};
    context_->getUniformiv(programID, loc, readback.data());
    EXPECT_EQ(readback[0], val[0]);
    EXPECT_EQ(readback[1], val[1]);
    EXPECT_EQ(readback[2], val[2]);
  }

  // Int4
  {
    const GLint loc = getLocation(pipeline, "uIVec4");
    ASSERT_GE(loc, 0);
    const std::array<int, 4> val = {1, 2, 3, 4};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Int4, reinterpret_cast<const uint8_t*>(val.data()), 1);
    std::array<GLint, 4> readback = {};
    context_->getUniformiv(programID, loc, readback.data());
    EXPECT_EQ(readback[0], val[0]);
    EXPECT_EQ(readback[1], val[1]);
    EXPECT_EQ(readback[2], val[2]);
    EXPECT_EQ(readback[3], val[3]);
  }

  // Boolean
  {
    const GLint loc = getLocation(pipeline, "uBool");
    ASSERT_GE(loc, 0);
    const uint8_t val = 1;
    opengl::UniformBuffer::bindUniform(*context_, loc, UniformType::Boolean, &val, 1);
    GLint readback = 0;
    context_->getUniformiv(programID, loc, &readback);
    EXPECT_EQ(readback, static_cast<GLint>(val));
  }

  // Mat2x2
  {
    const GLint loc = getLocation(pipeline, "uMat2");
    ASSERT_GE(loc, 0);
    const std::array<float, 4> val = {1.0f, 0.0f, 0.0f, 1.0f};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Mat2x2, reinterpret_cast<const uint8_t*>(val.data()), 1);
  }

  // Mat3x3
  {
    const GLint loc = getLocation(pipeline, "uMat3");
    ASSERT_GE(loc, 0);
    const std::array<float, 9> val = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Mat3x3, reinterpret_cast<const uint8_t*>(val.data()), 1);
  }

  // Mat4x4
  {
    const GLint loc = getLocation(pipeline, "uMat4");
    ASSERT_GE(loc, 0);
    const std::array<float, 16> val = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    opengl::UniformBuffer::bindUniform(
        *context_, loc, UniformType::Mat4x4, reinterpret_cast<const uint8_t*>(val.data()), 1);
  }
}

///
/// BindUniformArrayStridedPacking
///
/// Exercises bindUniformArray with stride != packed size, triggering the manual
/// packing paths with reinterpret_cast<const uint8_t*>(packedArray.get()) for
/// Int, Float, Boolean, and FloatMatrix base types.
///
TEST_F(UniformBufferBindTest, BindUniformArrayStridedPacking) {
  std::shared_ptr<IRenderPipelineState> pipeline;
  createPipeline(kFragShaderWithUniformArrays, pipeline);
  ASSERT_NE(pipeline, nullptr);
  const GLuint programID = usePipeline(pipeline);

  // Float array with stride > sizeof(float) — triggers Float packing path
  {
    const GLint loc = getLocation(pipeline, "uFloatArr");
    ASSERT_GE(loc, 0);
    struct PaddedFloat {
      float val = 0.0f;
      float padding = 0.0f;
    };
    const std::array<PaddedFloat, 2> data = {
        PaddedFloat{.val = 1.0f, .padding = 0.0f},
        PaddedFloat{.val = 2.0f, .padding = 0.0f},
    };
    opengl::UniformBuffer::bindUniformArray(*context_,
                                            loc,
                                            UniformType::Float,
                                            reinterpret_cast<const uint8_t*>(data.data()),
                                            2,
                                            sizeof(PaddedFloat));
  }

  // Int array with stride > sizeof(int) — triggers Int packing path
  {
    const GLint loc = getLocation(pipeline, "uIntArr");
    ASSERT_GE(loc, 0);
    struct PaddedInt {
      int val = 0;
      int padding = 0;
    };
    const std::array<PaddedInt, 2> data = {
        PaddedInt{.val = 10, .padding = 0},
        PaddedInt{.val = 20, .padding = 0},
    };
    opengl::UniformBuffer::bindUniformArray(*context_,
                                            loc,
                                            UniformType::Int,
                                            reinterpret_cast<const uint8_t*>(data.data()),
                                            2,
                                            sizeof(PaddedInt));
    std::array<GLint, 2> readback = {};
    context_->getUniformiv(programID, loc, &readback[0]);
    context_->getUniformiv(programID, loc + 1, &readback[1]);
    EXPECT_EQ(readback[0], data[0].val);
    EXPECT_EQ(readback[1], data[1].val);
  }

  // Boolean array with stride > 1 — triggers Boolean packing path
  {
    const GLint loc = getLocation(pipeline, "uBoolArr");
    ASSERT_GE(loc, 0);
    struct PaddedBool {
      uint8_t val = 0;
      uint8_t padding[7] = {};
    };
    const std::array<PaddedBool, 2> data = {
        PaddedBool{.val = 1, .padding = {}},
        PaddedBool{.val = 0, .padding = {}},
    };
    opengl::UniformBuffer::bindUniformArray(*context_,
                                            loc,
                                            UniformType::Boolean,
                                            reinterpret_cast<const uint8_t*>(data.data()),
                                            2,
                                            sizeof(PaddedBool));
    std::array<GLint, 2> readback = {};
    context_->getUniformiv(programID, loc, &readback[0]);
    context_->getUniformiv(programID, loc + 1, &readback[1]);
    EXPECT_EQ(readback[0], static_cast<GLint>(data[0].val));
    EXPECT_EQ(readback[1], static_cast<GLint>(data[1].val));
  }

  // Float4 array with stride > sizeof(float)*4 — triggers Float packing path
  {
    const GLint loc = getLocation(pipeline, "uVec4Arr");
    ASSERT_GE(loc, 0);
    struct PaddedVec4 {
      std::array<float, 4> val;
      std::array<float, 4> padding;
    };
    const std::array<PaddedVec4, 2> data = {
        PaddedVec4{.val = {1, 2, 3, 4}, .padding = {}},
        PaddedVec4{.val = {5, 6, 7, 8}, .padding = {}},
    };
    opengl::UniformBuffer::bindUniformArray(*context_,
                                            loc,
                                            UniformType::Float4,
                                            reinterpret_cast<const uint8_t*>(data.data()),
                                            2,
                                            sizeof(PaddedVec4));
  }

  // Mat4x4 array with stride > sizeof(float)*16 — triggers FloatMatrix packing path
  // bindUniformArray advances per-column by stride/primitivesPerElement, so padding
  // must be per-column (not after the entire 4x4 block)
  {
    const GLint loc = getLocation(pipeline, "uMat4Arr");
    ASSERT_GE(loc, 0);
    struct PaddedMat4Column {
      std::array<float, 4> val;
      std::array<float, 4> padding;
    };
    struct PaddedMat4 {
      std::array<PaddedMat4Column, 4> columns;
    };
    const std::array<PaddedMat4, 2> data = {
        PaddedMat4{.columns = {{
                       {.val = {1, 0, 0, 0}, .padding = {}},
                       {.val = {0, 1, 0, 0}, .padding = {}},
                       {.val = {0, 0, 1, 0}, .padding = {}},
                       {.val = {0, 0, 0, 1}, .padding = {}},
                   }}},
        PaddedMat4{.columns = {{
                       {.val = {2, 0, 0, 0}, .padding = {}},
                       {.val = {0, 2, 0, 0}, .padding = {}},
                       {.val = {0, 0, 2, 0}, .padding = {}},
                       {.val = {0, 0, 0, 2}, .padding = {}},
                   }}},
    };
    opengl::UniformBuffer::bindUniformArray(*context_,
                                            loc,
                                            UniformType::Mat4x4,
                                            reinterpret_cast<const uint8_t*>(data.data()),
                                            2,
                                            sizeof(PaddedMat4));
  }
}

///
/// BindUniformArrayPackedPath
///
/// Exercises bindUniformArray when stride == packed size, which delegates
/// directly to bindUniform (no manual packing). Covers the early-return path.
///
TEST_F(UniformBufferBindTest, BindUniformArrayPackedPath) {
  std::shared_ptr<IRenderPipelineState> pipeline;
  createPipeline(kFragShaderWithUniformArrays, pipeline);
  ASSERT_NE(pipeline, nullptr);
  const GLuint programID = usePipeline(pipeline);

  // Float array, tightly packed (stride == sizeof(float))
  {
    const GLint loc = getLocation(pipeline, "uFloatArr");
    ASSERT_GE(loc, 0);
    const std::array<float, 2> data = {3.14f, 2.71f};
    opengl::UniformBuffer::bindUniformArray(*context_,
                                            loc,
                                            UniformType::Float,
                                            reinterpret_cast<const uint8_t*>(data.data()),
                                            2,
                                            sizeof(float));
  }

  // Int array, tightly packed
  {
    const GLint loc = getLocation(pipeline, "uIntArr");
    ASSERT_GE(loc, 0);
    const std::array<int, 2> data = {100, 200};
    opengl::UniformBuffer::bindUniformArray(*context_,
                                            loc,
                                            UniformType::Int,
                                            reinterpret_cast<const uint8_t*>(data.data()),
                                            2,
                                            sizeof(int));
    std::array<GLint, 2> readback = {};
    context_->getUniformiv(programID, loc, &readback[0]);
    context_->getUniformiv(programID, loc + 1, &readback[1]);
    EXPECT_EQ(readback[0], data[0]);
    EXPECT_EQ(readback[1], data[1]);
  }
}

} // namespace igl::tests
