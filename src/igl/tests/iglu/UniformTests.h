/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>

#include <IGLU/uniform/Descriptor.h>
#include <string>
#include <igl/Shader.h>

namespace iglu::tests {

namespace {

constexpr int kVertexIndex = 9;
constexpr int kFragmentIndex = 51;

template<typename T>
struct Compare {
  static void verifyAligned(const T* expected, const void* aligned);
};

template<typename T>
void Compare<T>::verifyAligned(const T* expected, const void* aligned) {
  const T& alignedValue = *(static_cast<const T*>(aligned));
  ASSERT_EQ(*expected, alignedValue);
}

template<>
struct Compare<glm::mat3> {
  static void verifyAligned(const glm::mat3* expected, const void* aligned) {
    const auto& alignedValue =
        *static_cast<const typename uniform::Trait<glm::mat3>::Aligned*>(aligned);
    const auto* expectedPtr = static_cast<const float*>(glm::value_ptr(*expected));
    for (int i = 0; i < 3; i++) {
      // base of each row
      const auto* alignedRowPtr = static_cast<const float*>(glm::value_ptr(alignedValue[i]));
      for (int j = 0; j < 3; j++) {
        ASSERT_EQ(*expectedPtr++, *alignedRowPtr++);
      }
    }
  }
};

inline void testIndex(uniform::Descriptor& uniform) {
  ASSERT_EQ(uniform.getIndex(igl::ShaderStage::Fragment), -1);
  ASSERT_EQ(uniform.getIndex(igl::ShaderStage::Vertex), -1);

  uniform.setIndex(igl::ShaderStage::Fragment, kFragmentIndex);
  uniform.setIndex(igl::ShaderStage::Vertex, kVertexIndex);

  ASSERT_EQ(uniform.getIndex(igl::ShaderStage::Fragment), kFragmentIndex);
  ASSERT_EQ(uniform.getIndex(igl::ShaderStage::Vertex), kVertexIndex);
}

template<typename T>
void testUniformData(const T& expected, uniform::DescriptorValue<T>& uniform) {
  ASSERT_EQ(uniform.numBytes(uniform::Alignment::Packed), sizeof(T));
  ASSERT_EQ(uniform.numBytes(uniform::Alignment::Aligned), sizeof(T) + uniform::Trait<T>::kPadding);
  ASSERT_EQ(uniform.size(), 1);

  const void* uniformPackedPtr = uniform.data(uniform::Alignment::Packed);
  const void* uniformAlignedPtr = uniform.data(uniform::Alignment::Aligned);

  if (uniform::Trait<T>::kPadding == 0) {
    ASSERT_EQ(uniformPackedPtr, uniformAlignedPtr);
  }

  const T* uniformPacked = static_cast<const T*>(uniformPackedPtr);
  ASSERT_EQ(expected, *uniformPacked);

  if (uniform::Trait<T>::kPadding > 0) {
    Compare<T>::verifyAligned(&expected, uniformAlignedPtr);
  }
}

template<typename T>
void testUniformData(const std::vector<T>& expected, uniform::DescriptorVector<T>& uniform) {
  ASSERT_EQ(uniform.numBytes(uniform::Alignment::Packed), uniform.size() * sizeof(T));
  ASSERT_EQ(uniform.numBytes(uniform::Alignment::Aligned),
            uniform.size() * (sizeof(T) + uniform::Trait<T>::kPadding));
  ASSERT_EQ(uniform.size(), expected.size());

  constexpr size_t kNumBytesPacked = sizeof(T);
  constexpr size_t kNumBytesAligned = kNumBytesPacked + uniform::Trait<T>::kPadding;

  const auto* uniformPackedBytePtr =
      static_cast<const uint8_t*>(uniform.data(uniform::Alignment::Packed));
  const auto* uniformAlignedBytePtr =
      static_cast<const uint8_t*>(uniform.data(uniform::Alignment::Aligned));

  if (uniform::Trait<T>::kPadding == 0) {
    ASSERT_EQ(uniformPackedBytePtr, uniformAlignedBytePtr);
  }

  for (size_t i = 0, iLen = uniform.size(); i < iLen; ++i) {
    const T* uniformPacked = reinterpret_cast<const T*>(uniformPackedBytePtr);
    ASSERT_EQ(expected[i], *uniformPacked);

    if (uniform::Trait<T>::kPadding > 0) {
      Compare<T>::verifyAligned(&(expected[i]), uniformAlignedBytePtr);
    }

    uniformPackedBytePtr += kNumBytesPacked;
    uniformAlignedBytePtr += kNumBytesAligned;
  }
}

} // namespace
} // namespace iglu::tests
