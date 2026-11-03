/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanHelpers.h>

#include <array>
#include <cstring>
#include <string>

namespace igl::tests {

//
// VulkanHelpersTest
//
// Unit tests for functions in VulkanHelpers.{h|c}.
//

// ivkGetVulkanResultString **********************************************************************
class GetVulkanResultString : public ::testing::Test {};
TEST_F(GetVulkanResultString, VulkanHelpersTest) {
  EXPECT_STREQ(ivkGetVulkanResultString(VK_SUCCESS), "VK_SUCCESS");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_NOT_READY), "VK_NOT_READY");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_TIMEOUT), "VK_TIMEOUT");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_EVENT_SET), "VK_EVENT_SET");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_EVENT_RESET), "VK_EVENT_RESET");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_INCOMPLETE), "VK_INCOMPLETE");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_OUT_OF_HOST_MEMORY),
               "VK_ERROR_OUT_OF_HOST_MEMORY");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_OUT_OF_DEVICE_MEMORY),
               "VK_ERROR_OUT_OF_DEVICE_MEMORY");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_INITIALIZATION_FAILED),
               "VK_ERROR_INITIALIZATION_FAILED");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_DEVICE_LOST), "VK_ERROR_DEVICE_LOST");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_MEMORY_MAP_FAILED), "VK_ERROR_MEMORY_MAP_FAILED");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_LAYER_NOT_PRESENT), "VK_ERROR_LAYER_NOT_PRESENT");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_EXTENSION_NOT_PRESENT),
               "VK_ERROR_EXTENSION_NOT_PRESENT");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_FEATURE_NOT_PRESENT),
               "VK_ERROR_FEATURE_NOT_PRESENT");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_INCOMPATIBLE_DRIVER),
               "VK_ERROR_INCOMPATIBLE_DRIVER");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_TOO_MANY_OBJECTS), "VK_ERROR_TOO_MANY_OBJECTS");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_FORMAT_NOT_SUPPORTED),
               "VK_ERROR_FORMAT_NOT_SUPPORTED");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_SURFACE_LOST_KHR), "VK_ERROR_SURFACE_LOST_KHR");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_OUT_OF_DATE_KHR), "VK_ERROR_OUT_OF_DATE_KHR");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR),
               "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR),
               "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_VALIDATION_FAILED_EXT),
               "VK_ERROR_VALIDATION_FAILED_EXT");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_FRAGMENTED_POOL), "VK_ERROR_FRAGMENTED_POOL");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_UNKNOWN), "VK_ERROR_UNKNOWN");
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_OUT_OF_POOL_MEMORY),
               "VK_ERROR_OUT_OF_POOL_MEMORY"); // 1.1
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_INVALID_EXTERNAL_HANDLE),
               "VK_ERROR_INVALID_EXTERNAL_HANDLE"); // 1.1
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_FRAGMENTATION), "VK_ERROR_FRAGMENTATION"); // 1.2
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS),
               "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS"); // 1.2
  EXPECT_STREQ(ivkGetVulkanResultString(VK_SUBOPTIMAL_KHR),
               "VK_SUBOPTIMAL_KHR"); // VK_KHR_swapchain
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_INVALID_SHADER_NV),
               "VK_ERROR_INVALID_SHADER_NV"); // VK_NV_glsl_shader
#ifdef VK_ENABLE_BETA_EXTENSIONS
                                              // Provided by VK_KHR_video_queue
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR),
               "VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR");
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
  // Provided by VK_KHR_video_queue
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR),
               "VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR");
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
  // Provided by VK_KHR_video_queue
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR),
               "VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR");
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
  // Provided by VK_KHR_video_queue
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR),
               "VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR");
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
  // Provided by VK_KHR_video_queue
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR),
               "VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR");
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
  // Provided by VK_KHR_video_queue
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR),
               "VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR");
#endif
  EXPECT_STREQ(
      ivkGetVulkanResultString(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT),
      "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT"); // VK_EXT_image_drm_format_modifier
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_NOT_PERMITTED_KHR),
               "VK_ERROR_NOT_PERMITTED_KHR"); // VK_KHR_global_priority
  EXPECT_STREQ(ivkGetVulkanResultString(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT),
               "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT"); // VK_EXT_full_screen_exclusive
  EXPECT_STREQ(ivkGetVulkanResultString(VK_THREAD_IDLE_KHR),
               "VK_THREAD_IDLE_KHR"); // VK_KHR_deferred_host_operations
  EXPECT_STREQ(ivkGetVulkanResultString(VK_THREAD_DONE_KHR),
               "VK_THREAD_DONE_KHR"); // VK_KHR_deferred_host_operations
  EXPECT_STREQ(ivkGetVulkanResultString(VK_OPERATION_DEFERRED_KHR),
               "VK_OPERATION_DEFERRED_KHR"); // VK_KHR_deferred_host_operations
  EXPECT_STREQ(ivkGetVulkanResultString(VK_OPERATION_NOT_DEFERRED_KHR),
               "VK_OPERATION_NOT_DEFERRED_KHR"); // VK_KHR_deferred_host_operations
}

// ivkGetWriteDescriptorSetImageInfo *******************************
class GetWriteDescriptorSetImageInfoTest
  : public ::testing::TestWithParam<std::tuple<uint32_t, VkDescriptorType, uint32_t>> {};

TEST_P(GetWriteDescriptorSetImageInfoTest, GetWriteDescriptorSetImageInfo) {
  constexpr VkDescriptorSet descSet = VK_NULL_HANDLE;
  const uint32_t dstBinding = std::get<0>(GetParam());
  const VkDescriptorType descType = std::get<1>(GetParam());
  const uint32_t numDescs = std::get<2>(GetParam());

  const std::array<VkDescriptorImageInfo, 2> pImageInfo = {
      VkDescriptorImageInfo{.sampler = VK_NULL_HANDLE,
                            .imageView = VK_NULL_HANDLE,
                            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED},
      VkDescriptorImageInfo{.sampler = VK_NULL_HANDLE,
                            .imageView = VK_NULL_HANDLE,
                            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED}};

  const VkWriteDescriptorSet imageDescSet =
      ivkGetWriteDescriptorSetImageInfo(descSet, dstBinding, descType, numDescs, pImageInfo.data());

  EXPECT_EQ(imageDescSet.sType, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
  EXPECT_EQ(imageDescSet.pNext, nullptr);
  EXPECT_EQ(imageDescSet.dstSet, descSet);
  EXPECT_EQ(imageDescSet.dstBinding, dstBinding);
  EXPECT_EQ(imageDescSet.dstArrayElement, 0);
  EXPECT_EQ(imageDescSet.descriptorCount, numDescs);
  EXPECT_EQ(imageDescSet.descriptorType, descType);
  EXPECT_EQ(imageDescSet.pImageInfo, pImageInfo.data());
  EXPECT_EQ(imageDescSet.pBufferInfo, nullptr);
  EXPECT_EQ(imageDescSet.pTexelBufferView, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    GetWriteDescriptorSetImageInfoTest,
    ::testing::Combine(::testing::Values(0, 1),
                       ::testing::Values(VK_DESCRIPTOR_TYPE_SAMPLER,
                                         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
                       ::testing::Values(1, 2)),
    [](const testing::TestParamInfo<GetWriteDescriptorSetImageInfoTest::ParamType>& info) {
      const std::string name = std::to_string(std::get<0>(info.param)) + "_" +
                               std::to_string(std::get<1>(info.param)) + "_" +
                               std::to_string(std::get<2>(info.param));
      return name;
    });

// ivkGetWriteDescriptorSetBufferInfo *******************************
class GetWriteDescriptorSetBufferInfoTest
  : public ::testing::TestWithParam<std::tuple<uint32_t, VkDescriptorType, uint32_t>> {};

TEST_P(GetWriteDescriptorSetBufferInfoTest, GetWriteDescriptorSetBufferInfo) {
  constexpr VkDescriptorSet descSet = VK_NULL_HANDLE;
  const uint32_t dstBinding = std::get<0>(GetParam());
  const VkDescriptorType descType = std::get<1>(GetParam());
  const uint32_t numDescs = std::get<2>(GetParam());

  const std::array<VkDescriptorBufferInfo, 2> pBufferInfo = {
      VkDescriptorBufferInfo{.buffer = VK_NULL_HANDLE, .offset = 0, .range = VK_WHOLE_SIZE},
      VkDescriptorBufferInfo{.buffer = VK_NULL_HANDLE, .offset = 0, .range = VK_WHOLE_SIZE}};

  const VkWriteDescriptorSet bufferDescSet = ivkGetWriteDescriptorSetBufferInfo(
      descSet, dstBinding, descType, numDescs, pBufferInfo.data());

  EXPECT_EQ(bufferDescSet.sType, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
  EXPECT_EQ(bufferDescSet.pNext, nullptr);
  EXPECT_EQ(bufferDescSet.dstSet, descSet);
  EXPECT_EQ(bufferDescSet.dstBinding, dstBinding);
  EXPECT_EQ(bufferDescSet.dstArrayElement, 0);
  EXPECT_EQ(bufferDescSet.descriptorCount, numDescs);
  EXPECT_EQ(bufferDescSet.descriptorType, descType);
  EXPECT_EQ(bufferDescSet.pImageInfo, nullptr);
  EXPECT_EQ(bufferDescSet.pBufferInfo, pBufferInfo.data());
  EXPECT_EQ(bufferDescSet.pTexelBufferView, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    GetWriteDescriptorSetBufferInfoTest,
    ::testing::Combine(::testing::Values(0, 1),
                       ::testing::Values(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                       ::testing::Values(1, 2)),
    [](const testing::TestParamInfo<GetWriteDescriptorSetBufferInfoTest::ParamType>& info) {
      const std::string name = std::to_string(std::get<0>(info.param)) + "_" +
                               std::to_string(std::get<1>(info.param)) + "_" +
                               std::to_string(std::get<2>(info.param));
      return name;
    });

// ivkGetVertexInputBindingDescription *******************************
class GetVertexInputBindingDescriptionTest
  : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, VkVertexInputRate>> {};

TEST_P(GetVertexInputBindingDescriptionTest, GetVertexInputBindingDescription) {
  const auto binding = std::get<0>(GetParam());
  const auto stride = std::get<1>(GetParam());
  const auto inputRate = std::get<2>(GetParam());

  const VkVertexInputBindingDescription vtxInputBindingDesc =
      ivkGetVertexInputBindingDescription(binding, stride, inputRate);

  EXPECT_EQ(vtxInputBindingDesc.binding, binding);
  EXPECT_EQ(vtxInputBindingDesc.stride, stride);
  EXPECT_EQ(vtxInputBindingDesc.inputRate, inputRate);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    GetVertexInputBindingDescriptionTest,
    ::testing::Combine(::testing::Values(0, 1),
                       ::testing::Values(0, 16),
                       ::testing::Values(VK_VERTEX_INPUT_RATE_VERTEX,
                                         VK_VERTEX_INPUT_RATE_INSTANCE)),
    [](const testing::TestParamInfo<GetVertexInputBindingDescriptionTest::ParamType>& info) {
      const std::string name = std::to_string(std::get<0>(info.param)) + "_" +
                               std::to_string(std::get<1>(info.param)) + "_" +
                               std::to_string(std::get<2>(info.param));
      return name;
    });

// ivkGetVertexInputAttributeDescription *******************************
class GetVertexInputAttributeDescriptionTest
  : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, VkFormat, uint32_t>> {};

TEST_P(GetVertexInputAttributeDescriptionTest, GetVertexInputAttributeDescription) {
  const auto location = std::get<0>(GetParam());
  const auto binding = std::get<1>(GetParam());
  const auto format = std::get<2>(GetParam());
  const auto offset = std::get<3>(GetParam());

  const VkVertexInputAttributeDescription vtxInputAttrDesc =
      ivkGetVertexInputAttributeDescription(location, binding, format, offset);

  EXPECT_EQ(vtxInputAttrDesc.location, location);
  EXPECT_EQ(vtxInputAttrDesc.binding, binding);
  EXPECT_EQ(vtxInputAttrDesc.format, format);
  EXPECT_EQ(vtxInputAttrDesc.offset, offset);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    GetVertexInputAttributeDescriptionTest,
    ::testing::Combine(::testing::Values(0, 1),
                       ::testing::Values(0, 1),
                       ::testing::Values(VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM),
                       ::testing::Values(0, 16)),
    [](const testing::TestParamInfo<GetVertexInputAttributeDescriptionTest::ParamType>& info) {
      const std::string name = std::to_string(std::get<0>(info.param)) + "_" +
                               std::to_string(std::get<1>(info.param)) + "_" +
                               std::to_string(std::get<2>(info.param)) + "_" +
                               std::to_string(std::get<3>(info.param));
      return name;
    });

// ivkIsHostVisibleSingleHeapMemory *******************************
class IsHostVisibleSingleHeapMemoryTest : public ::testing::Test {};

TEST_F(IsHostVisibleSingleHeapMemoryTest, MultipleHeapsReturnsFalse) {
  const VkPhysicalDeviceMemoryProperties memProps{
      .memoryTypeCount = 1,
      .memoryTypes = {{.propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}},
      .memoryHeapCount = 2};
  EXPECT_FALSE(ivkIsHostVisibleSingleHeapMemory(&memProps));
}

TEST_F(IsHostVisibleSingleHeapMemoryTest, SingleHeapHostVisibleDeviceLocalReturnsTrue) {
  const VkPhysicalDeviceMemoryProperties memProps{
      .memoryTypeCount = 1,
      .memoryTypes = {{.propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}},
      .memoryHeapCount = 1};
  EXPECT_TRUE(ivkIsHostVisibleSingleHeapMemory(&memProps));
}

TEST_F(IsHostVisibleSingleHeapMemoryTest, SingleHeapNoDeviceLocalReturnsFalse) {
  const VkPhysicalDeviceMemoryProperties memProps{
      .memoryTypeCount = 1,
      .memoryTypes = {{.propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT}},
      .memoryHeapCount = 1};
  EXPECT_FALSE(ivkIsHostVisibleSingleHeapMemory(&memProps));
}

// ivkFindMemoryType *******************************
class FindMemoryTypeTest : public ::testing::Test {};

TEST_F(FindMemoryTypeTest, FindsFirstMatchingType) {
  const VkPhysicalDeviceMemoryProperties memProps{
      .memoryTypeCount = 2,
      .memoryTypes = {{.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
                      {.propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT}}};

  const uint32_t result = ivkFindMemoryType(&memProps, 0b11, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  EXPECT_EQ(result, 1u);
}

TEST_F(FindMemoryTypeTest, RespectsMemoryTypeBitsMask) {
  const VkPhysicalDeviceMemoryProperties memProps{
      .memoryTypeCount = 3,
      .memoryTypes = {{.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
                      {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
                      {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}}};

  const uint32_t result = ivkFindMemoryType(&memProps, 0b100, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  EXPECT_EQ(result, 2u);
}

TEST_F(FindMemoryTypeTest, MatchesExactPropertyFlags) {
  const VkPhysicalDeviceMemoryProperties memProps{
      .memoryTypeCount = 2,
      .memoryTypes = {{.propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT},
                      {.propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT}}};

  const uint32_t result = ivkFindMemoryType(
      &memProps, 0b11, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  EXPECT_EQ(result, 1u);
}

// ivkAddNext *******************************
class AddNextTest : public ::testing::Test {};

TEST_F(AddNextTest, AppendToEmptyChain) {
  VkBaseOutStructure node = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pNext = nullptr};
  VkBaseOutStructure next = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = nullptr};

  ivkAddNext(&node, &next);

  EXPECT_EQ(node.pNext, &next);
  EXPECT_EQ(next.pNext, nullptr);
}

TEST_F(AddNextTest, AppendToExistingChain) {
  VkBaseOutStructure second = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = nullptr};
  VkBaseOutStructure first = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pNext = &second};
  VkBaseOutStructure appended = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = nullptr};

  ivkAddNext(&first, &appended);

  EXPECT_EQ(first.pNext, &second);
  EXPECT_EQ(second.pNext, &appended);
  EXPECT_EQ(appended.pNext, nullptr);
}

TEST_F(AddNextTest, NullArgumentsAreNoop) {
  VkBaseOutStructure node = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pNext = nullptr};

  ivkAddNext(nullptr, &node);
  EXPECT_EQ(node.pNext, nullptr);

  ivkAddNext(&node, nullptr);
  EXPECT_EQ(node.pNext, nullptr);
}

// ivkUpdateGlslangResource *******************************
class UpdateGlslangResourceTest : public ::testing::Test {};

TEST_F(UpdateGlslangResourceTest, CopiesDeviceLimits) {
  VkPhysicalDeviceProperties props = {};
  props.limits.maxVertexInputAttributes = 16;
  props.limits.maxClipDistances = 8;
  props.limits.maxComputeWorkGroupCount[0] = 65535;
  props.limits.maxComputeWorkGroupCount[1] = 65534;
  props.limits.maxComputeWorkGroupCount[2] = 65533;
  props.limits.maxComputeWorkGroupSize[0] = 1024;
  props.limits.maxViewports = 16;
  props.limits.maxCullDistances = 8;

  glslang_resource_t res = {};
  ivkUpdateGlslangResource(&res, &props, nullptr);

  EXPECT_EQ(res.max_vertex_attribs, 16);
  EXPECT_EQ(res.max_clip_distances, 8);
  EXPECT_EQ(res.max_compute_work_group_count_x, 65535);
  EXPECT_EQ(res.max_compute_work_group_count_y, 65534);
  EXPECT_EQ(res.max_compute_work_group_count_z, 65533);
  EXPECT_EQ(res.max_compute_work_group_size_x, 1024);
  EXPECT_EQ(res.max_viewports, 16);
  EXPECT_EQ(res.max_cull_distances, 8);
  // Mesh shader fields stay zero-initialized when no mesh shader props are passed.
  EXPECT_EQ(res.max_mesh_output_vertices_ext, 0);
  EXPECT_EQ(res.max_task_work_group_size_x_ext, 0);
}

TEST_F(UpdateGlslangResourceTest, CopiesMeshShaderProps) {
  VkPhysicalDeviceProperties props = {};
  VkPhysicalDeviceMeshShaderPropertiesEXT meshProps = {};
  meshProps.maxMeshOutputVertices = 256;
  meshProps.maxMeshOutputPrimitives = 512;
  meshProps.maxMeshWorkGroupSize[0] = 128;
  meshProps.maxTaskWorkGroupSize[0] = 64;
  meshProps.maxMeshMultiviewViewCount = 4;

  glslang_resource_t res = {};
  ivkUpdateGlslangResource(&res, &props, &meshProps);

  EXPECT_EQ(res.max_mesh_output_vertices_ext, 256);
  EXPECT_EQ(res.max_mesh_output_primitives_ext, 512);
  EXPECT_EQ(res.max_mesh_work_group_size_x_ext, 128);
  EXPECT_EQ(res.max_task_work_group_size_x_ext, 64);
  EXPECT_EQ(res.max_mesh_view_count_ext, 4);
}

TEST_F(UpdateGlslangResourceTest, NullArgumentsAreNoop) {
  VkPhysicalDeviceProperties props = {};
  props.limits.maxVertexInputAttributes = 16;

  // Null resource: must not dereference, must not crash.
  ivkUpdateGlslangResource(nullptr, &props, nullptr);

  // Null properties: resource is left untouched.
  glslang_resource_t res = {};
  res.max_vertex_attribs = 7;
  ivkUpdateGlslangResource(&res, nullptr, nullptr);
  EXPECT_EQ(res.max_vertex_attribs, 7);
}

// ivkCreateDevice *******************************
namespace {
VkPhysicalDeviceFeatures gCapturedFeatures;
bool gFeaturesCaptured = false;
const void* gCapturedPNext = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL fakeCreateDevice(VkPhysicalDevice /*physicalDevice*/,
                                                const VkDeviceCreateInfo* createInfo,
                                                const VkAllocationCallbacks* /*allocator*/,
                                                VkDevice* device) {
  if (createInfo && createInfo->pEnabledFeatures) {
    gCapturedFeatures = *createInfo->pEnabledFeatures;
    gFeaturesCaptured = true;
  } else {
    gFeaturesCaptured = false;
  }
  gCapturedPNext = createInfo ? createInfo->pNext : nullptr;
  *device = VK_NULL_HANDLE;
  return VK_SUCCESS;
}
} // namespace

class IvkCreateDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    gCapturedFeatures = {};
    gFeaturesCaptured = false;
    gCapturedPNext = nullptr;
  }
};

TEST_F(IvkCreateDeviceTest, EnablesIndependentBlendWhenSupported) {
  const VkDeviceQueueCreateInfo queueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = nullptr,
  };

  const VkPhysicalDeviceFeatures2 supported = {.sType =
                                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                               .features = {.independentBlend = VK_TRUE}};

  const VulkanFunctionTable vt = {.vkCreateDevice = &fakeCreateDevice};
  VkDevice device = VK_NULL_HANDLE;

  const VkResult result =
      ivkCreateDevice(&vt, VK_NULL_HANDLE, 1, &queueCreateInfo, 0, nullptr, &supported, &device);

  EXPECT_EQ(result, VK_SUCCESS);
  EXPECT_TRUE(gFeaturesCaptured);
  EXPECT_EQ(gCapturedFeatures.independentBlend, VK_TRUE);
}

TEST_F(IvkCreateDeviceTest, DoesNotEnableIndependentBlendWhenUnsupported) {
  const VkDeviceQueueCreateInfo queueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = nullptr,
  };

  const VkPhysicalDeviceFeatures2 supported = {.sType =
                                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                               .features = {.independentBlend = VK_FALSE}};

  const VulkanFunctionTable vt = {.vkCreateDevice = &fakeCreateDevice};
  VkDevice device = VK_NULL_HANDLE;

  const VkResult result =
      ivkCreateDevice(&vt, VK_NULL_HANDLE, 1, &queueCreateInfo, 0, nullptr, &supported, &device);

  EXPECT_EQ(result, VK_SUCCESS);
  EXPECT_TRUE(gFeaturesCaptured);
  EXPECT_EQ(gCapturedFeatures.independentBlend, VK_FALSE);
}

TEST_F(IvkCreateDeviceTest, ForwardsSelectedCoreFeaturesExactly) {
  const VkDeviceQueueCreateInfo queueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = nullptr,
  };
  const VkPhysicalDeviceFeatures2 selected = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .features = {.independentBlend = VK_TRUE, .shaderFloat64 = VK_TRUE, .shaderInt64 = VK_TRUE},
  };
  const VulkanFunctionTable vt = {.vkCreateDevice = &fakeCreateDevice};
  VkDevice device = VK_NULL_HANDLE;
  const VkResult result =
      ivkCreateDevice(&vt, VK_NULL_HANDLE, 1, &queueCreateInfo, 0, nullptr, &selected, &device);
  EXPECT_EQ(result, VK_SUCCESS);
  ASSERT_TRUE(gFeaturesCaptured);
  EXPECT_EQ(std::memcmp(&gCapturedFeatures, &selected.features, sizeof(VkPhysicalDeviceFeatures)),
            0);
}

TEST_F(IvkCreateDeviceTest, PassesTheFeatureChain) {
  const VkDeviceQueueCreateInfo queueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = nullptr,
  };

  VkPhysicalDeviceMultiviewFeatures multiviewFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES};

  const VkPhysicalDeviceFeatures2 supported = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &multiviewFeatures};

  const VulkanFunctionTable vt = {.vkCreateDevice = &fakeCreateDevice};
  VkDevice device = VK_NULL_HANDLE;

  const VkResult result =
      ivkCreateDevice(&vt, VK_NULL_HANDLE, 1, &queueCreateInfo, 0, nullptr, &supported, &device);

  EXPECT_EQ(result, VK_SUCCESS);
  EXPECT_EQ(gCapturedPNext, &multiviewFeatures);
}

} // namespace igl::tests
