/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// MemoryObjectImportOGLTest
//
// Tests for external memory object import in OpenGL.
//
class MemoryObjectImportOGLTest : public ::testing::Test {
 public:
  MemoryObjectImportOGLTest() = default;

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
// MemoryObjectCreation
//
// Test that external memory object APIs are accessible (skip if not supported).
//
TEST_F(MemoryObjectImportOGLTest, MemoryObjectCreation) {
  if (!iglDev_->hasFeature(DeviceFeatures::ExternalMemoryObjects)) {
    GTEST_SKIP() << "External memory objects not supported";
  }

  // Create a memory object
  GLuint memObject = 0;
  context_->createMemoryObjects(1, &memObject);

  // The memory object should be created (non-zero)
  ASSERT_NE(memObject, 0u);

  // Clean up
  context_->deleteMemoryObjects(1, &memObject);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
