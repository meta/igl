/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/opengl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/VertexArrayObject.h>

namespace igl::tests {

//
// VertexArrayObjectOGLTest
//
// Tests for the OpenGL VertexArrayObject.
//
class VertexArrayObjectOGLTest : public ::testing::Test {
 public:
  VertexArrayObjectOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
};

//
// CreateBindUnbind
//
// Create a VAO, verify it is valid, bind and unbind with no GL errors.
//
TEST_F(VertexArrayObjectOGLTest, CreateBindUnbind) {
  if (!context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::VertexArrayObject)) {
    GTEST_SKIP() << "VertexArrayObject not supported";
  }

  opengl::VertexArrayObject vao(*context_);

  Result ret = vao.create();
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(vao.isValid());

  // Bind should succeed without errors
  vao.bind();
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  // Unbind should succeed without errors
  vao.unbind();
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
