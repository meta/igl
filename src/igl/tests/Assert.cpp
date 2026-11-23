/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

// Only include Assert.h and ensure it is configured to enable both soft errors and debug aborts
#define IGL_DEBUG_ABORT_ENABLED 1
#define IGL_SOFT_ERROR_ENABLED 1
#define IGL_COMMON_SKIP_CHECK 1
#include <igl/Assert.h>

namespace igl::tests {
namespace {
bool sAbort = false;
bool sSoftError = false;
} // namespace

class AssertTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
    iglSetDebugAbortListener([](const char* /*category*/,
                                const char* /*reason*/,
                                const char* /*file*/,
                                const char* /*func*/,
                                int /*line*/,
                                const char* /*format*/,
                                va_list /*ap*/) { sAbort = true; });
    iglSetSoftErrorHandler([](const char* /*category*/,
                              const char* /*reason*/,
                              const char* /*file*/,
                              const char* /*func*/,
                              int /*line*/,
                              const char* /*format*/,
                              va_list /*ap*/) { sSoftError = true; });
  }

  void TearDown() override {
    iglSetDebugAbortListener(nullptr);
    iglSetSoftErrorHandler(nullptr);
  }
};

TEST_F(AssertTest, DebugAbort) {
  sAbort = false;
  EXPECT_FALSE(sAbort);
  IGL_DEBUG_ABORT("Aborting");
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  IGL_DEBUG_ABORT("Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  IGL_DEBUG_ASSERT(false);
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  IGL_DEBUG_ASSERT(false, "Aborting");
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  IGL_DEBUG_ASSERT(false, "Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  std::ignore = IGL_DEBUG_VERIFY(false);
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  std::ignore = IGL_DEBUG_VERIFY(false, "Aborting");
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  std::ignore = IGL_DEBUG_VERIFY(false, "Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  std::ignore = IGL_DEBUG_VERIFY_NOT(true);
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  std::ignore = IGL_DEBUG_VERIFY_NOT(true, "Aborting");
  EXPECT_TRUE(sAbort);

  sAbort = false;
  EXPECT_FALSE(sAbort);
  std::ignore = IGL_DEBUG_VERIFY_NOT(true, "Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);
}

TEST_F(AssertTest, SoftError) {
  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  IGL_SOFT_ERROR("Aborting");
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  IGL_SOFT_ERROR("Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  IGL_SOFT_ASSERT(false);
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  IGL_SOFT_ASSERT(false, "Aborting");
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  IGL_SOFT_ASSERT(false, "Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  std::ignore = IGL_SOFT_VERIFY(false);
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  std::ignore = IGL_SOFT_VERIFY(false, "Aborting");
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  std::ignore = IGL_SOFT_VERIFY(false, "Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  std::ignore = IGL_SOFT_VERIFY_NOT(true);
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  std::ignore = IGL_SOFT_VERIFY_NOT(true, "Aborting");
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);

  sAbort = false;
  sSoftError = false;
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
  std::ignore = IGL_SOFT_VERIFY_NOT(true, "Aborting with arg %d", 1);
  EXPECT_TRUE(sAbort);
  EXPECT_TRUE(sSoftError);
}

TEST_F(AssertTest, DebugAssertPassDoesNotAbort) {
  sAbort = false;
  IGL_DEBUG_ASSERT(true);
  EXPECT_FALSE(sAbort);

  IGL_DEBUG_ASSERT(true, "Should not fire");
  EXPECT_FALSE(sAbort);

  IGL_DEBUG_ASSERT(true, "Should not fire %d", 1);
  EXPECT_FALSE(sAbort);
}

TEST_F(AssertTest, DebugVerifyReturnValues) {
  sAbort = false;
  EXPECT_TRUE(IGL_DEBUG_VERIFY(true));
  EXPECT_FALSE(sAbort);

  EXPECT_TRUE(IGL_DEBUG_VERIFY(true, "pass msg"));
  EXPECT_FALSE(sAbort);

  EXPECT_FALSE(IGL_DEBUG_VERIFY(false));
  sAbort = false;

  EXPECT_FALSE(IGL_DEBUG_VERIFY_NOT(false));
  EXPECT_FALSE(sAbort);

  EXPECT_FALSE(IGL_DEBUG_VERIFY_NOT(false, "pass msg"));
  EXPECT_FALSE(sAbort);

  EXPECT_TRUE(IGL_DEBUG_VERIFY_NOT(true));
  sAbort = false;
}

TEST_F(AssertTest, SoftAssertPassDoesNotFire) {
  sAbort = false;
  sSoftError = false;
  IGL_SOFT_ASSERT(true);
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);

  IGL_SOFT_ASSERT(true, "Should not fire");
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);

  EXPECT_TRUE(IGL_SOFT_VERIFY(true));
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);

  EXPECT_FALSE(IGL_SOFT_VERIFY_NOT(false));
  EXPECT_FALSE(sAbort);
  EXPECT_FALSE(sSoftError);
}

TEST_F(AssertTest, DebugBreakEnabledRoundTrip) {
  // SetUp() disables debug break, so the getter should reflect that initially.
  EXPECT_FALSE(isDebugBreakEnabled());

  setDebugBreakEnabled(true);
  EXPECT_TRUE(isDebugBreakEnabled());

  setDebugBreakEnabled(false);
  EXPECT_FALSE(isDebugBreakEnabled());
}

TEST_F(AssertTest, SoftErrorHandlerRoundTrip) {
  // SetUp() installs a non-null soft error handler.
  EXPECT_NE(iglGetSoftErrorHandler(), nullptr);

  const IGLErrorHandlerFunc handler = [](const char* /*category*/,
                                         const char* /*reason*/,
                                         const char* /*file*/,
                                         const char* /*func*/,
                                         int /*line*/,
                                         const char* /*format*/,
                                         va_list /*ap*/) {};
  iglSetSoftErrorHandler(handler);
  EXPECT_EQ(iglGetSoftErrorHandler(), handler);

  iglSetSoftErrorHandler(nullptr);
  EXPECT_EQ(iglGetSoftErrorHandler(), nullptr);
}

TEST_F(AssertTest, DebugAbortListenerRoundTrip) {
  // SetUp() installs a non-null debug abort listener.
  EXPECT_NE(iglGetDebugAbortListener(), nullptr);

  const IGLErrorHandlerFunc listener = [](const char* /*category*/,
                                          const char* /*reason*/,
                                          const char* /*file*/,
                                          const char* /*func*/,
                                          int /*line*/,
                                          const char* /*format*/,
                                          va_list /*ap*/) {};
  iglSetDebugAbortListener(listener);
  EXPECT_EQ(iglGetDebugAbortListener(), listener);

  iglSetDebugAbortListener(nullptr);
  EXPECT_EQ(iglGetDebugAbortListener(), nullptr);
}

} // namespace igl::tests
