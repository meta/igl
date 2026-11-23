/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/metal/SamplerState.h>

namespace igl::tests {

class SamplerStateMTLTest : public ::testing::Test {
 public:
  SamplerStateMTLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);

    samplerState_ = std::make_shared<metal::SamplerState>(nil);
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<metal::SamplerState> samplerState_;
};

TEST_F(SamplerStateMTLTest, ConvertMinMagFilter) {
  MTLSamplerMinMagFilter res = MTLSamplerMinMagFilterNearest;

  res = igl::metal::SamplerState::convertMinMagFilter(SamplerMinMagFilter::Linear);
  ASSERT_EQ(res, MTLSamplerMinMagFilterLinear);

  res = igl::metal::SamplerState::convertMinMagFilter(SamplerMinMagFilter::Nearest);
  ASSERT_EQ(res, MTLSamplerMinMagFilterNearest);
}

TEST_F(SamplerStateMTLTest, ConvertMipFilter) {
  MTLSamplerMipFilter res = MTLSamplerMipFilterNotMipmapped;

  res = igl::metal::SamplerState::convertMipFilter(SamplerMipFilter::Disabled);
  ASSERT_EQ(res, MTLSamplerMipFilterNotMipmapped);

  res = igl::metal::SamplerState::convertMipFilter(SamplerMipFilter::Nearest);
  ASSERT_EQ(res, MTLSamplerMipFilterNearest);

  res = igl::metal::SamplerState::convertMipFilter(SamplerMipFilter::Linear);
  ASSERT_EQ(res, MTLSamplerMipFilterLinear);
}

TEST_F(SamplerStateMTLTest, ConvertAddressMode) {
  MTLSamplerAddressMode res = MTLSamplerAddressModeRepeat;

  res = igl::metal::SamplerState::convertAddressMode(SamplerAddressMode::Repeat);
  ASSERT_EQ(res, MTLSamplerAddressModeRepeat);

  res = igl::metal::SamplerState::convertAddressMode(SamplerAddressMode::Clamp);
  ASSERT_EQ(res, MTLSamplerAddressModeClampToEdge);

  res = igl::metal::SamplerState::convertAddressMode(SamplerAddressMode::MirrorRepeat);
  ASSERT_EQ(res, MTLSamplerAddressModeMirrorRepeat);

  res = igl::metal::SamplerState::convertAddressMode(SamplerAddressMode::ClampToBorder);
  if (@available(macOS 10.12, iOS 14.0, *)) {
    ASSERT_EQ(res, MTLSamplerAddressModeClampToBorderColor);
  } else {
    ASSERT_EQ(res, MTLSamplerAddressModeClampToZero);
  }
}

} // namespace igl::tests
