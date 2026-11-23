/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/DestructionGuard.h>

#include "../util/Common.h"

#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/PlatformDevice.h>

namespace igl::tests {

//
// DestructionGuardOGLTest
//
// Tests for the OpenGL DestructionGuard.
//
class DestructionGuardOGLTest : public ::testing::Test {
 public:
  DestructionGuardOGLTest() = default;

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
// InitiallyAllowed
//
// Destruction should be allowed before any guard is active.
//
TEST_F(DestructionGuardOGLTest, InitiallyAllowed) {
  ASSERT_TRUE(context_->isDestructionAllowed());
}

//
// GuardPreventsDestruction
//
// While a DestructionGuard is alive, destruction should not be allowed.
//
TEST_F(DestructionGuardOGLTest, GuardPreventsDestruction) {
  auto& device = static_cast<opengl::Device&>(*iglDev_);
  const auto& platformDevice =
      static_cast<const opengl::PlatformDevice&>(device.getPlatformDevice());

  {
    auto guard = platformDevice.getDestructionGuard();
    ASSERT_FALSE(context_->isDestructionAllowed());
  }
}

//
// GuardScopeRestore
//
// After a DestructionGuard goes out of scope, destruction should be allowed again.
//
TEST_F(DestructionGuardOGLTest, GuardScopeRestore) {
  auto& device = static_cast<opengl::Device&>(*iglDev_);
  const auto& platformDevice =
      static_cast<const opengl::PlatformDevice&>(device.getPlatformDevice());

  {
    auto guard = platformDevice.getDestructionGuard();
    ASSERT_FALSE(context_->isDestructionAllowed());
  }

  // After guard scope, destruction should be allowed again
  ASSERT_TRUE(context_->isDestructionAllowed());
}

//
// NestedGuards
//
// Nested guards should work correctly - destruction stays blocked until all guards are released.
//
TEST_F(DestructionGuardOGLTest, NestedGuards) {
  auto& device = static_cast<opengl::Device&>(*iglDev_);
  const auto& platformDevice =
      static_cast<const opengl::PlatformDevice&>(device.getPlatformDevice());

  {
    auto outerGuard = platformDevice.getDestructionGuard();
    ASSERT_FALSE(context_->isDestructionAllowed());

    {
      auto innerGuard = platformDevice.getDestructionGuard();
      ASSERT_FALSE(context_->isDestructionAllowed());
    }

    // Inner guard released, but outer guard still active
    ASSERT_FALSE(context_->isDestructionAllowed());
  }

  // Both guards released
  ASSERT_TRUE(context_->isDestructionAllowed());
}

} // namespace igl::tests
