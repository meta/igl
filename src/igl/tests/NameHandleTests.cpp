/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <set>
#include <unordered_map>
#include <utility>
#include <igl/NameHandle.h>

// NOLINTBEGIN(misc-use-internal-linkage,facebook-static-object-destructor-check)
IGL_NAMEHANDLE_ACCESSOR(myTestAccessorHandle)
IGL_NAMEHANDLE_ACCESSOR_IMPL(myTestAccessorHandle, "myTestStr")
// NOLINTEND(misc-use-internal-linkage,facebook-static-object-destructor-check)

namespace igl::tests {
namespace {
// NOLINTBEGIN(clang-diagnostic-global-constructors,facebook-static-object-destructor-check)
NameHandle a = IGL_NAMEHANDLE("a");
NameHandle b = IGL_NAMEHANDLE("b");
NameHandle c = IGL_NAMEHANDLE("c");
NameHandle someLongerString = IGL_NAMEHANDLE("someLongerString");
// NOLINTEND(clang-diagnostic-global-constructors,facebook-static-object-destructor-check)
} // namespace

TEST(NameHandleTests, string) {
  EXPECT_EQ(a.toString(), "a");
  EXPECT_EQ(b.toString(), "b");
  EXPECT_EQ(c.toString(), "c");
  EXPECT_EQ(someLongerString.toString(), "someLongerString");
}

TEST(NameHandleTests, crc32) {
  EXPECT_EQ(a.getCrc32(), 3904355907);
  EXPECT_EQ(b.getCrc32(), 1908338681);
  EXPECT_EQ(c.getCrc32(), 112844655);
  EXPECT_EQ(someLongerString.getCrc32(), 3994903871);
}

TEST(NameHandleTests, equality) {
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);

  EXPECT_NE(a, b);
}

TEST(NameHandleTests, ordering) {
  EXPECT_GT(a, b);
  EXPECT_GT(b, c);

  EXPECT_GE(a, a);
  EXPECT_GE(a, b);
  EXPECT_GE(b, c);

  EXPECT_LT(c, b);
  EXPECT_LT(b, a);

  EXPECT_LE(c, c);
  EXPECT_LE(c, b);
  EXPECT_LE(b, a);
}

TEST(NameHandleTests, set) {
  std::set s{a, b};
  EXPECT_EQ(s.size(), 2);
  EXPECT_NE(s.find(a), s.end());
  EXPECT_NE(s.find(b), s.end());
  EXPECT_EQ(s.find(c), s.end());
}

// NOLINTBEGIN(google-readability-avoid-underscore-in-googletest-name)
TEST(NameHandleTests, DefaultConstructor) {
  NameHandle empty;
  EXPECT_EQ(empty.getCrc32(), 0u);
  EXPECT_TRUE(empty.toString().empty());
  EXPECT_STREQ(empty.c_str(), "");
}

TEST(NameHandleTests, CStr) {
  EXPECT_STREQ(a.c_str(), "a");
  EXPECT_STREQ(someLongerString.c_str(), "someLongerString");
}

TEST(NameHandleTests, ImplicitConstCharPtr) {
  const char* sa = a;
  EXPECT_STREQ(sa, "a");
  const char* sl = someLongerString;
  EXPECT_STREQ(sl, "someLongerString");
}

TEST(NameHandleTests, CopyAssignment) {
  NameHandle h;
  h = a;
  EXPECT_EQ(h.getCrc32(), a.getCrc32());
  EXPECT_EQ(h.toString(), a.toString());
  // Assign same CRC (no-op path in operator=)
  NameHandle a2 = IGL_NAMEHANDLE("a");
  h = a2;
  EXPECT_EQ(h.getCrc32(), a.getCrc32());
}

TEST(NameHandleTests, MoveConstructor) {
  NameHandle src = IGL_NAMEHANDLE("moveSource");
  const uint32_t crc = src.getCrc32();
  const NameHandle dst(std::move(src));
  EXPECT_EQ(dst.getCrc32(), crc);
  EXPECT_EQ(dst.toString(), "moveSource");
}

TEST(NameHandleTests, MoveAssignment) {
  NameHandle src = IGL_NAMEHANDLE("moveAssign");
  const uint32_t crc = src.getCrc32();
  NameHandle dst;
  dst = std::move(src);
  EXPECT_EQ(dst.getCrc32(), crc);
  EXPECT_EQ(dst.toString(), "moveAssign");
}

TEST(NameHandleTests, GenNameHandleRuntimeConsistency) {
  const NameHandle compileTime = IGL_NAMEHANDLE("hello");
  const NameHandle runtime = genNameHandle("hello");
  EXPECT_EQ(compileTime.getCrc32(), runtime.getCrc32());
  EXPECT_EQ(compileTime.toString(), runtime.toString());
}

TEST(NameHandleTests, Crc32StringViewConsistency) {
  constexpr std::string_view sv = "testString";
  constexpr uint32_t crcSv = igl::iglCrc32ConstExpr(sv);
  constexpr uint32_t crcCstr = iglCrc32ConstExpr("testString");
  static_assert(crcSv == crcCstr, "CRC32 string_view and const char* overloads must agree");
  EXPECT_EQ(crcSv, crcCstr);
}

TEST(NameHandleTests, UnorderedMap) {
  std::unordered_map<NameHandle, int> m;
  m[a] = 1;
  m[b] = 2;
  EXPECT_EQ(m.at(a), 1);
  EXPECT_EQ(m.at(b), 2);
  EXPECT_EQ(m.find(c), m.end());
  EXPECT_EQ(m.size(), 2u);
}

TEST(NameHandleTests, PairHash) {
  std::unordered_map<std::pair<NameHandle, NameHandle>, int> m;
  m[{a, b}] = 42;
  EXPECT_EQ(m.at({a, b}), 42);
  EXPECT_EQ(m.find({a, c}), m.end());
}

TEST(NameHandleTests, AccessorMacros) {
  const NameHandle& h1 = myTestAccessorHandle();
  const NameHandle& h2 = myTestAccessorHandle();
  EXPECT_EQ(&h1, &h2);
  EXPECT_EQ(h1.toString(), "myTestStr");
  EXPECT_EQ(h1.getCrc32(), IGL_NAMEHANDLE("myTestStr").getCrc32());
}

TEST(NameHandleTests, StringViewConstructor) {
  constexpr std::string_view sv = "viewTest";
  const NameHandle h(sv, iglCrc32ConstExpr(sv));
  EXPECT_EQ(h.toString(), "viewTest");
  EXPECT_EQ(h.getCrc32(), iglCrc32ConstExpr("viewTest"));
}

TEST(NameHandleTests, VectorHash) {
  const std::hash<std::vector<NameHandle>> hasher;
  const std::vector<NameHandle> key1 = {a, b};
  const std::vector<NameHandle> key1Copy = {a, b};
  const std::vector<NameHandle> key3 = {a, c};

  EXPECT_EQ(hasher(key1), hasher(key1Copy));
  EXPECT_NE(hasher(key1), hasher(key3));

  std::unordered_map<std::vector<NameHandle>, int> m;
  m[key1] = 10;
  m[key3] = 30;
  EXPECT_EQ(m.at(key1), 10);
  EXPECT_EQ(m.at(key3), 30);
  EXPECT_EQ(m.at(key1Copy), 10);
  EXPECT_EQ(m.size(), 2u);
}
// NOLINTEND(google-readability-avoid-underscore-in-googletest-name)

} // namespace igl::tests
