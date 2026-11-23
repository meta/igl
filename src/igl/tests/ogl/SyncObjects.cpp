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

namespace igl::tests {

//
// SyncObjectsOGLTest
//
// Tests for OpenGL sync (fence) objects.
//
class SyncObjectsOGLTest : public ::testing::Test {
 public:
  SyncObjectsOGLTest() = default;

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
// FenceSyncAndWait
//
// Create a fence sync object, flush, and query its status.
//
TEST_F(SyncObjectsOGLTest, FenceSyncAndWait) {
  if (!context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::Sync)) {
    GTEST_SKIP() << "Sync objects not supported";
  }

  // Create a fence sync
  GLsync sync = context_->fenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  ASSERT_NE(sync, nullptr);

  // Flush to ensure the sync is submitted
  context_->flush();

  // Query the sync status
  GLint syncStatus = GL_UNSIGNALED;
  GLsizei length = 0;
  context_->getSynciv(sync, GL_SYNC_STATUS, sizeof(GLint), &length, &syncStatus);

  // Status should be either signaled or unsignaled (both are valid)
  ASSERT_TRUE(syncStatus == GL_SIGNALED || syncStatus == GL_UNSIGNALED);

  // Wait by calling finish which ensures all commands complete
  context_->finish();

  // After finish, query again - should be signaled
  context_->getSynciv(sync, GL_SYNC_STATUS, sizeof(GLint), &length, &syncStatus);
  ASSERT_EQ(syncStatus, GL_SIGNALED);

  // Clean up
  context_->deleteSync(sync);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
