/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/metal/DepthStencilState.h>

namespace igl::tests {

// DepthStencilStateMTLTest
//
// This test covers metal::DepthStencilState.
class DepthStencilStateMTLTest : public ::testing::Test {
 public:
  DepthStencilStateMTLTest() = default;
  void SetUp() override {
    setDebugBreakEnabled(false);
  }
  void TearDown() override {}
};

//
// Compare Function to MTL Test
//
// Check expected outputs for metal::DepthStencilState::convertStencilOperation
//
TEST_F(DepthStencilStateMTLTest, StencilOpConversionToMTL) {
  struct StencilOpConversion {
    StencilOperation igl = StencilOperation::Keep;
    MTLStencilOperation mtl = MTLStencilOperationKeep;
  };
  const MTLStencilOperation mtlStencilOperation =
      metal::DepthStencilState::convertStencilOperation(StencilOperation::Keep);
  ASSERT_EQ(mtlStencilOperation, MTLStencilOperationKeep);

  const std::vector<StencilOpConversion> conversions{
      StencilOpConversion{.igl = StencilOperation::Keep, .mtl = MTLStencilOperationKeep},
      StencilOpConversion{.igl = StencilOperation::Zero, .mtl = MTLStencilOperationZero},
      StencilOpConversion{.igl = StencilOperation::Replace, .mtl = MTLStencilOperationReplace},
      StencilOpConversion{.igl = StencilOperation::IncrementClamp,
                          .mtl = MTLStencilOperationIncrementClamp},
      StencilOpConversion{.igl = StencilOperation::DecrementClamp,
                          .mtl = MTLStencilOperationDecrementClamp},
      StencilOpConversion{.igl = StencilOperation::Invert, .mtl = MTLStencilOperationInvert},
      StencilOpConversion{.igl = StencilOperation::IncrementWrap,
                          .mtl = MTLStencilOperationIncrementWrap},
      StencilOpConversion{.igl = StencilOperation::DecrementWrap,
                          .mtl = MTLStencilOperationDecrementWrap},
  };

  for (auto data : conversions) {
    enum MTLStencilOperation mtl = metal::DepthStencilState::convertStencilOperation(data.igl);
    ASSERT_EQ(mtl, data.mtl);
  }
}

//
// Compare Function to MTL Test
//
// Check expected outputs for metal::DepthStencilState::convertCompareFunction
//
TEST_F(DepthStencilStateMTLTest, CompareFunctionToMTL) {
  struct CompareFuncConversion {
    CompareFunction igl = CompareFunction::Never;
    MTLCompareFunction mtl = MTLCompareFunctionNever;
  };

  const std::vector<CompareFuncConversion> conversions{
      CompareFuncConversion{.igl = CompareFunction::Never, .mtl = MTLCompareFunctionNever},
      CompareFuncConversion{.igl = CompareFunction::Less, .mtl = MTLCompareFunctionLess},
      CompareFuncConversion{.igl = CompareFunction::Equal, .mtl = MTLCompareFunctionEqual},
      CompareFuncConversion{.igl = CompareFunction::LessEqual, .mtl = MTLCompareFunctionLessEqual},
      CompareFuncConversion{.igl = CompareFunction::Greater, .mtl = MTLCompareFunctionGreater},
      CompareFuncConversion{.igl = CompareFunction::NotEqual, .mtl = MTLCompareFunctionNotEqual},
      CompareFuncConversion{.igl = CompareFunction::GreaterEqual,
                            .mtl = MTLCompareFunctionGreaterEqual},
      CompareFuncConversion{.igl = CompareFunction::AlwaysPass, .mtl = MTLCompareFunctionAlways},
  };

  for (auto data : conversions) {
    const MTLCompareFunction mtl = metal::DepthStencilState::convertCompareFunction(data.igl);
    ASSERT_EQ(mtl, data.mtl);
  }
}

} // namespace igl::tests
