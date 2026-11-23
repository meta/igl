/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/metal/Device.h>

#include "../util/TestDevice.h"

namespace igl::tests {

class DeviceMetalTest : public ::testing::Test {
 public:
  DeviceMetalTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);

    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
};

// Test Cases
TEST_F(DeviceMetalTest, GetShaderVersion) {
  auto iglShaderVersion = iglDev_->getShaderVersion();
  ASSERT_EQ(iglShaderVersion.family, igl::ShaderFamily::Metal);
  ASSERT_GT(iglShaderVersion.majorVersion, 0);
  ASSERT_GT(iglShaderVersion.minorVersion, 0);
}

TEST_F(DeviceMetalTest, GetMostRecentCommandQueue) {
  auto* metalDevice = static_cast<metal::Device*>(iglDev_.get());

  // Initially, no command queue should exist
  auto initialCmdQueue = metalDevice->getMostRecentCommandQueue();
  ASSERT_EQ(initialCmdQueue, nullptr);

  // Create a command queue
  Result result;
  CommandQueueDesc desc{};
  auto cmdQueue1 = iglDev_->createCommandQueue(desc, &result);
  ASSERT_EQ(result.code, igl::Result::Code::Ok);
  ASSERT_NE(cmdQueue1, nullptr);

  // Get the most recent command queue and verify it matches
  auto mostRecentCmdQueue1 = metalDevice->getMostRecentCommandQueue();
  ASSERT_EQ(mostRecentCmdQueue1, cmdQueue1);

  // Create another command queue
  auto cmdQueue2 = iglDev_->createCommandQueue(desc, &result);
  ASSERT_EQ(result.code, igl::Result::Code::Ok);
  ASSERT_NE(cmdQueue2, nullptr);

  // Verify the most recent command queue is now the second one
  auto mostRecentCmdQueue2 = metalDevice->getMostRecentCommandQueue();
  ASSERT_EQ(mostRecentCmdQueue2, cmdQueue2);
  ASSERT_NE(mostRecentCmdQueue2, cmdQueue1);
}

} // namespace igl::tests
