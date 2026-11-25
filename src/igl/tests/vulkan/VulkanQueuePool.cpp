/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanQueuePool.h>

#include <igl/Macros.h>

namespace igl::vulkan {
namespace {
[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const VulkanQueueDescriptor& queue) {
  return os << "VulkanQueueDescriptor" << "\n\tQueue Index        : " << queue.queueIndex
            << "\n\tQueue Family Index : " << queue.familyIndex;
}
} // namespace
} // namespace igl::vulkan

namespace igl::tests {

using namespace vulkan;

TEST(VulkanQueuePoolTest, ReturnDedicatedComputeQueueWhenComputeQueueIsRequested) {
  // Given a dedicated compute queue
  const VulkanQueueDescriptor computeQueueDescriptor{
      .queueFlags = VK_QUEUE_COMPUTE_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueuePool queuePool({computeQueueDescriptor});

  // When compute queue is requested
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);

  // Then return compute queue
  ASSERT_TRUE(queueDescriptor.isValid());
  EXPECT_EQ(queueDescriptor, computeQueueDescriptor);
}

TEST(VulkanQueuePoolTest, ReturnDedicatedTransferQueueWhenTransferQueueIsRequested) {
  // Given a dedicated transfer queue
  const VulkanQueueDescriptor transferQueueDescriptor{
      .queueFlags = VK_QUEUE_TRANSFER_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueuePool queuePool({transferQueueDescriptor});

  // When transfer queue is requested
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_TRANSFER_BIT);

  // Then return transfer queue
  ASSERT_TRUE(queueDescriptor.isValid());
  EXPECT_EQ(queueDescriptor, transferQueueDescriptor);
}

TEST(VulkanQueuePoolTest, ReturnAllInOneQueueWhenComputeQueueIsRequested) {
  // Given an all in one queue
  const VulkanQueueDescriptor allInOneQueueDescriptor{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 1};
  const VulkanQueuePool queuePool({allInOneQueueDescriptor});

  // When compute queue is requested
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);

  // Then return all in one queue
  ASSERT_TRUE(queueDescriptor.isValid());
  EXPECT_EQ(queueDescriptor, allInOneQueueDescriptor);
}

TEST(VulkanQueuePoolTest, ReturnAllInOneQueueWhenTransferQueueIsRequested) {
  // Given an all in one queue
  const VulkanQueueDescriptor allInOneQueueDescriptor{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 1};
  const VulkanQueuePool queuePool({allInOneQueueDescriptor});

  // When transfer queue is requested
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_TRANSFER_BIT);

  // Then return all in one queue
  ASSERT_TRUE(queueDescriptor.isValid());
  EXPECT_EQ(queueDescriptor, allInOneQueueDescriptor);
}

TEST(VulkanQueuePoolTest, PreferDedicatedComputeQueueOverAllInOneQueue) {
  // Given a dedicated compute queue and an all in one queue
  const VulkanQueueDescriptor computeQueueDescriptor{
      .queueFlags = VK_QUEUE_COMPUTE_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueueDescriptor allInOneQueueDescriptor{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 2};
  const VulkanQueuePool queuePool({allInOneQueueDescriptor, computeQueueDescriptor});

  // When compute queue is requested
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);

  // Then return dedicated compute queue
  ASSERT_TRUE(queueDescriptor.isValid());
  EXPECT_EQ(queueDescriptor, computeQueueDescriptor);
}

TEST(VulkanQueuePoolTest, PreferDedicatedTransferQueueOverAllInOneQueue) {
  // Given a dedicated transfer queue and an all in one queue
  const VulkanQueueDescriptor transferQueueDescriptor{
      .queueFlags = VK_QUEUE_TRANSFER_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueueDescriptor allInOneQueueDescriptor{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 2};
  const VulkanQueuePool queuePool({allInOneQueueDescriptor, transferQueueDescriptor});

  // When transfer queue is requested
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_TRANSFER_BIT);

  // Then return dedicated transfer queue
  ASSERT_TRUE(queueDescriptor.isValid());
  EXPECT_EQ(queueDescriptor, transferQueueDescriptor);
}

TEST(VulkanQueuePoolTest, IfUnreservedUseSameQueueForQueueRequests) {
  // Given 2 all in one queues
  const VulkanQueueDescriptor allInOneQueueDescriptor1{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 1};
  const VulkanQueueDescriptor allInOneQueueDescriptor2{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 2};
  const VulkanQueuePool queuePool({allInOneQueueDescriptor1, allInOneQueueDescriptor2});

  // When graphics queue and compute queue are requested
  auto graphicsQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  auto computeQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);

  // Then return same all in one queue
  ASSERT_TRUE(graphicsQueueDescriptor.isValid());
  ASSERT_TRUE(computeQueueDescriptor.isValid());
  EXPECT_EQ(graphicsQueueDescriptor, allInOneQueueDescriptor1);
  EXPECT_EQ(computeQueueDescriptor, allInOneQueueDescriptor1);
}

TEST(VulkanQueuePoolTest, DoNotUseReservedQueuesForFurtherQueueRequests) {
  // Given a 2 all in one queues
  const VulkanQueueDescriptor allInOneQueueDescriptor1{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 1};
  const VulkanQueueDescriptor allInOneQueueDescriptor2{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      .queueIndex = 0,
      .familyIndex = 2};
  VulkanQueuePool queuePool({allInOneQueueDescriptor1, allInOneQueueDescriptor2});

  // When the first queue is reserved
  auto graphicsQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  ASSERT_TRUE(graphicsQueueDescriptor.isValid());
  EXPECT_EQ(graphicsQueueDescriptor, allInOneQueueDescriptor1);
  queuePool.reserveQueue(graphicsQueueDescriptor);
  auto computeQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);

  // Then return an unreserved queue for further requests
  ASSERT_TRUE(computeQueueDescriptor.isValid());
  EXPECT_EQ(computeQueueDescriptor, allInOneQueueDescriptor2);
}

TEST(VulkanQueuePoolTest, DoNotReturnQueueCreationInfosIfNothingIsReserved) {
  // Given a queue
  const VulkanQueueDescriptor graphicsDescriptor{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueuePool queuePool({graphicsDescriptor});

  // When a queue is requested but not reserved
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  ASSERT_TRUE(queueDescriptor.isValid());

  // Then do not return any creation infos
  auto qcis = queuePool.getQueueCreationInfos();
  ASSERT_TRUE(qcis.empty());
}

TEST(VulkanQueuePoolTest, ReturnQueueCreationInfoIfAnyQueueIsReserved) {
  // Given a queue
  const VulkanQueueDescriptor graphicsQueueDescriptor{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  VulkanQueuePool queuePool({graphicsQueueDescriptor});

  // When a queue is requested and reserved
  auto queueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  ASSERT_TRUE(queueDescriptor.isValid());
  queuePool.reserveQueue(queueDescriptor);

  // Then return reserved queue creation info
  auto qcis = queuePool.getQueueCreationInfos();
  ASSERT_EQ(qcis.size(), 1);
  EXPECT_EQ(qcis[0].sType, VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
  EXPECT_EQ(qcis[0].queueFamilyIndex, graphicsQueueDescriptor.familyIndex);
  EXPECT_EQ(qcis[0].queueCount, 1);
  EXPECT_EQ(*qcis[0].pQueuePriorities, 1.0);
}

TEST(VulkanQueuePoolTest, ReturnSingleQueueCreationInfoForSameQueueFamily) {
  // Given 2 queues from same family
  const VulkanQueueDescriptor graphicsQueueDescriptor1{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueueDescriptor graphicsQueueDescriptor2{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 1, .familyIndex = 1};
  VulkanQueuePool queuePool({graphicsQueueDescriptor1, graphicsQueueDescriptor2});

  // When 2 queues are requested and reserved
  auto queueDescriptor1 = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  ASSERT_TRUE(queueDescriptor1.isValid());
  queuePool.reserveQueue(queueDescriptor1);
  auto queueDescriptor2 = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  ASSERT_TRUE(queueDescriptor2.isValid());
  queuePool.reserveQueue(queueDescriptor2);

  // Then return a single queue creation info with queue count 2
  auto qcis = queuePool.getQueueCreationInfos();
  ASSERT_EQ(qcis.size(), 1);
  EXPECT_EQ(qcis[0].queueFamilyIndex, graphicsQueueDescriptor1.familyIndex);
  EXPECT_EQ(qcis[0].queueCount, 2);
}

TEST(VulkanQueueDescriptorTest, DefaultConstructedIsInvalid) {
  const VulkanQueueDescriptor desc;
  EXPECT_FALSE(desc.isValid());
  EXPECT_EQ(desc.queueIndex, VulkanQueueDescriptor::kInvalid);
  EXPECT_EQ(desc.familyIndex, VulkanQueueDescriptor::kInvalid);
  EXPECT_EQ(desc.queueFlags, 0u);
}

TEST(VulkanQueueDescriptorTest, LessThanOrdering) {
  const VulkanQueueDescriptor a{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueueDescriptor b{
      .queueFlags = VK_QUEUE_COMPUTE_BIT, .queueIndex = 0, .familyIndex = 2};
  const VulkanQueueDescriptor c{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 1, .familyIndex = 1};

  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_TRUE(a < c);
  EXPECT_FALSE(c < a);
  EXPECT_FALSE(a < a);
}

TEST(VulkanQueueDescriptorTest, EqualityIgnoresQueueFlags) {
  // operator== considers two descriptors equal when familyIndex and queueIndex
  // match, regardless of queueFlags (see the uniqueness comment on the struct).
  const VulkanQueueDescriptor graphics{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueueDescriptor compute{
      .queueFlags = VK_QUEUE_COMPUTE_BIT, .queueIndex = 0, .familyIndex = 1};

  EXPECT_TRUE(graphics == compute);

  // Differing queueIndex makes the descriptors unequal even with identical flags.
  const VulkanQueueDescriptor graphicsOtherIndex{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 1, .familyIndex = 1};
  EXPECT_FALSE(graphics == graphicsOtherIndex);
}

TEST(VulkanQueueDescriptorTest, OrderingIgnoresQueueFlags) {
  // Descriptors sharing familyIndex and queueIndex are equivalent under
  // operator<, so a std::set treats them as the same element regardless of flags.
  const VulkanQueueDescriptor graphics{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueueDescriptor compute{
      .queueFlags = VK_QUEUE_COMPUTE_BIT, .queueIndex = 0, .familyIndex = 1};

  EXPECT_FALSE(graphics < compute);
  EXPECT_FALSE(compute < graphics);
}

TEST(VulkanQueuePoolTest, FindQueueDescriptorReturnsInvalidWhenNoMatch) {
  const VulkanQueueDescriptor graphicsOnly{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueuePool queuePool({graphicsOnly});

  const auto result = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);
  EXPECT_FALSE(result.isValid());
}

TEST(VulkanQueuePoolTest, ReturnMultipleQueueCreationInfosForDifferentQueueFamilies) {
  // Given 2 queues from different families
  const VulkanQueueDescriptor graphicsQueueDescriptor1{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 1};
  const VulkanQueueDescriptor graphicsQueueDescriptor2{
      .queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueIndex = 0, .familyIndex = 2};
  VulkanQueuePool queuePool({graphicsQueueDescriptor1, graphicsQueueDescriptor2});

  // When 2 queues are requested and reserved
  auto queueDescriptor1 = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  ASSERT_TRUE(queueDescriptor1.isValid());
  queuePool.reserveQueue(queueDescriptor1);
  auto queueDescriptor2 = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  ASSERT_TRUE(queueDescriptor2.isValid());
  queuePool.reserveQueue(queueDescriptor2);

  // Then return 2 queue creation infos with queue count 1
  auto qcis = queuePool.getQueueCreationInfos();
  ASSERT_EQ(qcis.size(), 2);
  EXPECT_EQ(qcis[0].queueFamilyIndex, queueDescriptor1.familyIndex);
  EXPECT_EQ(qcis[0].queueCount, 1);
  EXPECT_EQ(qcis[1].queueFamilyIndex, queueDescriptor2.familyIndex);
  EXPECT_EQ(qcis[1].queueCount, 1);
}

} // namespace igl::tests
