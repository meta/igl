/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IGLU/managedUniformBuffer/ManagedUniformBuffer.h>

#include "../util/Common.h"
#include "RecordingDevice.h"

namespace igl::tests {

//
// ManagedUniformBufferTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class ManagedUniformBufferTest : public ::testing::Test {
 public:
  ManagedUniformBufferTest() = default;

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

TEST_F(ManagedUniformBufferTest, Construction) {
  iglu::ManagedUniformBuffer buffer(*iglDev_, {.index = 0, .length = 10});
  EXPECT_TRUE(buffer.getData() != nullptr);
}

TEST_F(ManagedUniformBufferTest, ConstructionWithBlockName) {
  // When a block name is supplied, the uniforms are backed by an interface block. Construction must
  // succeed on every backend and expose writable data; on OpenGL it also allocates a uniform-block
  // buffer used by the native-UBO bind path (GLSL ES 3.x).
  iglu::ManagedUniformBuffer buffer(*iglDev_,
                                    {.index = 12,
                                     .length = 16,
                                     .uniforms = {{.name = "params.color",
                                                   .location = 0,
                                                   .type = UniformType::Float4,
                                                   .numElements = 1,
                                                   .offset = 0,
                                                   .elementStride = 0}},
                                     .blockName = "params"});
  EXPECT_TRUE(buffer.result.isOk());
  EXPECT_TRUE(buffer.getData() != nullptr);
  EXPECT_EQ(buffer.uniformInfo.blockName, "params");
}

TEST_F(ManagedUniformBufferTest, RequestsAPlainUniformBuffer) {
  RecordingDevice device;
  const iglu::ManagedUniformBuffer buffer(device, {.index = 0, .length = 256});

  ASSERT_EQ(device.createBufferCount, 1u);
  // BufferAPIHintBits::NoCopy and BufferTypeBits::Command are both bit 5, so an API hint ORed into
  // desc.type lands in the type mask: it is dropped as a hint and misreports the uniforms as a
  // command buffer (Uniform|Command == 0x24).
  EXPECT_EQ(device.recordedType, BufferDesc::BufferTypeBits::Uniform);
  // NoCopy would alias the allocation this object frees in its destructor; see the comment in
  // ManagedUniformBuffer's constructor.
  EXPECT_EQ(device.recordedHint & BufferDesc::BufferAPIHintBits::NoCopy, 0);
}

TEST_F(ManagedUniformBufferTest, RequestsARingBufferWhenTheDeviceReportsOne) {
  RecordingDevice device;
  device.reportsBufferRing = true;
  const iglu::ManagedUniformBuffer buffer(device, {.index = 0, .length = 256});

  ASSERT_EQ(device.createBufferCount, 1u);
  EXPECT_NE(device.recordedHint & BufferDesc::BufferAPIHintBits::Ring, 0);
  EXPECT_EQ(device.recordedType, BufferDesc::BufferTypeBits::Uniform);
}

TEST_F(ManagedUniformBufferTest, AsksForNoRingBufferWithoutTheCapability) {
  RecordingDevice device;
  ASSERT_FALSE(device.hasFeature(DeviceFeatures::BufferRing));
  const iglu::ManagedUniformBuffer buffer(device, {.index = 0, .length = 256});

  // A device that does not implement the hint keeps the single-slot descriptor it got before.
  ASSERT_EQ(device.createBufferCount, 1u);
  EXPECT_EQ(device.recordedHint & BufferDesc::BufferAPIHintBits::Ring, 0);
  EXPECT_EQ(device.recordedType, BufferDesc::BufferTypeBits::Uniform);
}

TEST_F(ManagedUniformBufferTest, UpdateData) {
  // Using LUT to update data
  {
    iglu::ManagedUniformBuffer buffer(*iglDev_,
                                      {.index = 0,
                                       .length = 10,
                                       .uniforms = {{.name = "myUniform",
                                                     .location = 0,
                                                     .type = UniformType::Float,
                                                     .numElements = 1,
                                                     .offset = 0,
                                                     .elementStride = 0}}});
    float data = 1000.0f;

    buffer.buildUniformLUT();
    buffer.updateData("myUniform", &data, sizeof(float));
    EXPECT_TRUE(*static_cast<float*>(buffer.getData()) == data);
  }

  // Not using LUT to update data
  {
    iglu::ManagedUniformBuffer buffer(*iglDev_,
                                      {.index = 0,
                                       .length = 10,
                                       .uniforms = {{.name = "myUniform",
                                                     .location = 0,
                                                     .type = UniformType::Float,
                                                     .numElements = 1,
                                                     .offset = 0,
                                                     .elementStride = 0}}});
    float data = 1000.0f;

    buffer.updateData("myUniform", &data, sizeof(float));
    EXPECT_TRUE(*static_cast<float*>(buffer.getData()) == data);
  }

  // Capped data size
  {
    iglu::ManagedUniformBuffer buffer(*iglDev_,
                                      {.index = 0,
                                       .length = 10,
                                       .uniforms = {{.name = "myUniform",
                                                     .location = 0,
                                                     .type = UniformType::Float,
                                                     .numElements = 1,
                                                     .offset = 0,
                                                     .elementStride = 0}}});
    float data[2] = {1000.0f, 1.0f};

    buffer.updateData("myUniform", &data, 2 * sizeof(float));
    EXPECT_TRUE(*static_cast<float*>(buffer.getData()) == data[0]);
  }

  // Wrong index
  {
    iglu::ManagedUniformBuffer buffer(*iglDev_,
                                      {.index = 0,
                                       .length = 10,
                                       .uniforms = {{.name = "nonExistingUniform",
                                                     .location = -1,
                                                     .type = UniformType::Float,
                                                     .numElements = 1,
                                                     .offset = 0,
                                                     .elementStride = 0}}});
    float data = 1000.0f;

    EXPECT_FALSE(buffer.updateData("myUniform", &data, sizeof(float)));
  }
}

TEST_F(ManagedUniformBufferTest, GetUniformDataSize) {
  iglu::ManagedUniformBuffer buffer(*iglDev_,
                                    {.index = 0,
                                     .length = 10,
                                     .uniforms = {{.name = "myUniform",
                                                   .location = 0,
                                                   .type = UniformType::Float,
                                                   .numElements = 1,
                                                   .offset = 0,
                                                   .elementStride = 0}}});
  EXPECT_EQ(buffer.getUniformDataSize("myUniform"), sizeof(float));
  EXPECT_EQ(buffer.getUniformDataSize("nonExistingUniform"), 0);
}

TEST_F(ManagedUniformBufferTest, GetUniformType) {
  iglu::ManagedUniformBuffer buffer(*iglDev_,
                                    {.index = 0,
                                     .length = 64,
                                     .uniforms = {{.name = "floatUniform",
                                                   .location = 0,
                                                   .type = UniformType::Float,
                                                   .numElements = 1,
                                                   .offset = 0,
                                                   .elementStride = 0},
                                                  {.name = "int3Uniform",
                                                   .location = 1,
                                                   .type = UniformType::Int3,
                                                   .numElements = 1,
                                                   .offset = sizeof(float),
                                                   .elementStride = 0}}});
  EXPECT_EQ(buffer.getUniformType("floatUniform"), UniformType::Float);
  EXPECT_EQ(buffer.getUniformType("int3Uniform"), UniformType::Int3);
  EXPECT_EQ(buffer.getUniformType("nonExistingUniform"), UniformType::Invalid);
}

TEST_F(ManagedUniformBufferTest, GetIndexLinearSearch) {
  // Without calling buildUniformLUT(), getIndex() falls back to a linear search
  // over the uniform descriptors.
  iglu::ManagedUniformBuffer buffer(*iglDev_,
                                    {.index = 0,
                                     .length = 64,
                                     .uniforms = {{.name = "first",
                                                   .location = 0,
                                                   .type = UniformType::Float,
                                                   .numElements = 1,
                                                   .offset = 0,
                                                   .elementStride = 0},
                                                  {.name = "second",
                                                   .location = 1,
                                                   .type = UniformType::Float,
                                                   .numElements = 1,
                                                   .offset = sizeof(float),
                                                   .elementStride = 0}}});
  EXPECT_EQ(buffer.getIndex("first"), 0);
  EXPECT_EQ(buffer.getIndex("second"), 1);
  EXPECT_EQ(buffer.getIndex("missing"), -1);
}

TEST_F(ManagedUniformBufferTest, GetIndexWithLUT) {
  // After buildUniformLUT(), getIndex() resolves names via the lookup table and
  // must return the same indices as the linear-search path.
  iglu::ManagedUniformBuffer buffer(*iglDev_,
                                    {.index = 0,
                                     .length = 64,
                                     .uniforms = {{.name = "first",
                                                   .location = 0,
                                                   .type = UniformType::Float,
                                                   .numElements = 1,
                                                   .offset = 0,
                                                   .elementStride = 0},
                                                  {.name = "second",
                                                   .location = 1,
                                                   .type = UniformType::Float,
                                                   .numElements = 1,
                                                   .offset = sizeof(float),
                                                   .elementStride = 0}}});
  buffer.buildUniformLUT();
  EXPECT_EQ(buffer.getIndex("first"), 0);
  EXPECT_EQ(buffer.getIndex("second"), 1);
  EXPECT_EQ(buffer.getIndex("missing"), -1);
}

} // namespace igl::tests
