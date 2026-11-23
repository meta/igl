/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <shell/shared/fileLoader/linux/FileLoaderLinux.h>

#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <igl/Core.h>

namespace igl::shell::tests {

namespace {
bool sAborted = false;
} // namespace

class FileLoaderLinuxTest : public ::testing::Test {
 public:
  void SetUp() override {
    sAborted = false;
    debugBreakEnabled_ = isDebugBreakEnabled();
    setDebugBreakEnabled(false);
    iglSetDebugAbortListener([](const char* /*category*/,
                                const char* /*reason*/,
                                const char* /*file*/,
                                const char* /*func*/,
                                int /*line*/,
                                const char* /*format*/,
                                va_list /*ap*/) { sAborted = true; });

    originalCwd_ = std::filesystem::current_path();
    tempDir_ = std::filesystem::temp_directory_path() /
               ("FileLoaderLinuxTest_" + std::to_string(::getpid()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directories(tempDir_);
    std::filesystem::current_path(tempDir_);
  }

  void TearDown() override {
    std::filesystem::current_path(originalCwd_);
    std::error_code ec;
    std::filesystem::remove_all(tempDir_, ec);
    iglSetDebugAbortListener(nullptr);
    setDebugBreakEnabled(debugBreakEnabled_);
  }

 protected:
  static void writeFile(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << "data";
  }

  bool debugBreakEnabled_ = true;
  std::filesystem::path originalCwd_;
  std::filesystem::path tempDir_;
  FileLoaderLinux loader_;
};

TEST_F(FileLoaderLinuxTest, ExactPathExists) {
  const auto path = (tempDir_ / "exact.txt").string();
  writeFile(path);

  EXPECT_TRUE(loader_.fileExists(path));
  EXPECT_EQ(loader_.fullPath(path), path);
  EXPECT_FALSE(sAborted);
}

TEST_F(FileLoaderLinuxTest, BareNameInResourceFolderExists) {
  writeFile(tempDir_ / "samples/resources/models/bare.txt");

  EXPECT_TRUE(loader_.fileExists("bare.txt"));
  EXPECT_EQ(std::filesystem::path(loader_.fullPath("bare.txt")),
            tempDir_ / "samples/resources/models/bare.txt");
  EXPECT_FALSE(sAborted);
}

TEST_F(FileLoaderLinuxTest, EmptyNameDoesNotExist) {
  EXPECT_FALSE(loader_.fileExists(""));
  EXPECT_FALSE(sAborted);
}

TEST_F(FileLoaderLinuxTest, MissingFileDoesNotExistWithoutAsserting) {
  EXPECT_FALSE(loader_.fileExists("FileLoaderLinuxTest_missing.txt"));
  EXPECT_FALSE(sAborted);
}

TEST_F(FileLoaderLinuxTest, FullPathOfMissingFileAsserts) {
  EXPECT_EQ(loader_.fullPath("FileLoaderLinuxTest_missing.txt"), "");
#if IGL_DEBUG_ABORT_ENABLED
  EXPECT_TRUE(sAborted);
#endif
}

} // namespace igl::shell::tests
