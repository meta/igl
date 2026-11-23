/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <igl/Common.h>

namespace igl::tests {

//
// MemcpyTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class MemcpyOGLTest : public ::testing::Test {
 public:
  MemcpyOGLTest() = default;

  // Set up common resources. This will create a device and a command queue
  void SetUp() override {
    setDebugBreakEnabled(false);
  }

  void TearDown() override {}
};

//
// opengl::optimizedMemcpy
//
// Verify opengl::optimizedMemcpy works under various alignment cases.
//
TEST_F(MemcpyOGLTest, optimizedMemcpyAlignmentPermutation) {
  // Source buffer, destination buffer and its clearing pattern
  const char src[] = "0123456789ABCDEF+-*/";
  const char clr[] = "abcdefghijklmnopqrst";
  char dst[sizeof(src)];

  ASSERT_EQ(sizeof(src), sizeof(clr));

  // We do a series of optimizedMemcpy() from src to dst buffer,
  // we advance the src pointer at twice the speed of the dst pointer,
  // so we cover many kinds of alignment cases wrt both src and dst buffer.
  for (unsigned i = 0; i < sizeof(dst); ++i) {
    const unsigned di = i / 2;
    const unsigned len = sizeof(dst) - i;

    // Always clear the dst buffer before each optimizedMemcpy() call.
    std::memcpy(dst, clr, sizeof(dst));
    optimizedMemcpy(dst + di, src + i, len);

    // Verify optimizedMemcpy() has done the job correctly.
    // We divide the dst buffer into up to 3 sections,
    // then we copy into the middle section from src buffer,
    // in the end, we check all of them to have the correct content.
    ASSERT_EQ(0, std::memcmp(dst, clr, di));
    ASSERT_EQ(0, std::memcmp(dst + di, src + i, len));
    ASSERT_EQ(0, std::memcmp(dst + di + len, clr + di + len, sizeof(dst) - di - len));
  }
}

} // namespace igl::tests
