/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <unordered_set>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanHelpers.h>

namespace igl::vulkan {

class VulkanContext;

/// @brief Creates and maintains a list of feature structures for checking feature availability and
/// feature selection. This class provides a way to quickly enable the default and required
/// features. This class also manages instance and device extensions in Vulkan by enumerating all
/// extensions available for either object and storing the names of the available ones as
/// std::strings. A call to either `enumerate()` or `enumerate(VkPhysicalDevice)` must be performed
/// before the class can be used. After enumeration, this class allows users to enable an object's
/// extension by name by first checking them against all available extensions of that type. Only
/// available extensions are stored as enabled internally. The class also provides helper functions
/// to return all available extensions of a type, checking whether an extension is available without
/// modifying the internal storage of the object, checking if an extension has been enabled for an
/// object and, finally, a method to return a list of all enabled extensions of a type as `const
/// char *`, which is accepted by the Vulkan API
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class VulkanFeatures final {
 public:
  /// @brief Helper enumeration to determine which extension is being used. It's converted to a
  /// size_t internally to help access the right list of enumerations
  enum class ExtensionType { Instance = 0, Device };

  explicit VulkanFeatures(VulkanContextConfig config) noexcept;
  VulkanFeatures(const VulkanFeatures& other) noexcept;

  /// @brief Adds requested bits from a Vulkan feature structure. Owns a copy; ignores pNext.
  /// The structure must contain only sType, pNext, and VkBool32 feature members.
  /// It is queried only if extensionName is enabled or coreVersion is supported (0 means none).
  template<typename Feature>
  void requestFeature(const Feature& feature,
                      const char* IGL_NONNULL extensionName,
                      uint32_t coreVersion = 0) {
    static_assert(std::is_trivially_copyable_v<Feature>);
    static_assert(std::is_standard_layout_v<Feature>);
    static_assert(offsetof(Feature, pNext) == offsetof(VkBaseOutStructure, pNext));
    static_assert(alignof(Feature) <= alignof(uint64_t));
    requestFeature(&feature, sizeof(Feature), extensionName, coreVersion);
  }

  void requestFeature(const VkPhysicalDeviceFeatures& features) {
    requestFeature(VkPhysicalDeviceFeatures2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                             .features = features},
                   "",
                   VK_API_VERSION_1_0);
  }

  /// @brief After device creation, returns the enabled structure; nullptr means its bits are off.
  [[nodiscard]] const void* IGL_NULLABLE feature(VkStructureType type) const noexcept;

  /// @brief Returns whether an extension was enabled for the specified object type.
  [[nodiscard]] bool enabled(const char* extensionName, ExtensionType extensionType) const;

  /// @brief Populates the VkPhysicalDeviceFeatures2 and its pNext chain for a Vulkan context
  void populateWithAvailablePhysicalDeviceFeatures(const VulkanContext& context,
                                                   VkPhysicalDevice physicalDevice) noexcept;

  /// @brief Checks the features enabled in this class against the ones passed
  /// in as a parameter in 'availableFeatures'. If a requested feature is not present, the class
  /// logs the message and returns a failure
  [[nodiscard]] Result checkSelectedFeatures(
      const VulkanFeatures& availableFeatures) const noexcept;

  /// @brief Resolves mutually exclusive requested features after
  /// populateWithAvailablePhysicalDeviceFeatures(). That query overwrites every chained struct
  /// with the device's reported values, so a GPU supporting both fragmentDensityMap and any
  /// fragment shading rate would request both, violating
  /// VUID-VkDeviceCreateInfo-fragmentDensityMap-04481 (pipeline) and -04482 (primitive).
  /// IGL uses density maps and never uses shading rates, so the latter are dropped whenever
  /// the former is enabled. Call after populateWithAvailablePhysicalDeviceFeatures() and
  /// before vkCreateDevice().
  void resolveMutuallyExclusiveFeatures() noexcept;

  VkPhysicalDeviceFeatures2 vkPhysicalDeviceFeatures2{};

  // Vulkan 1.1
  VkPhysicalDeviceSamplerYcbcrConversionFeatures featuresSamplerYcbcrConversion{};
  VkPhysicalDeviceShaderDrawParametersFeatures featuresShaderDrawParameters{};
  VkPhysicalDeviceMultiviewFeatures featuresMultiview{};

  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR featuresBufferDeviceAddress{};
  VkPhysicalDeviceDescriptorIndexingFeaturesEXT featuresDescriptorIndexing{};
  VkPhysicalDevice16BitStorageFeatures features16BitStorage{};

  // Vulkan 1.2
  VkPhysicalDeviceShaderFloat16Int8Features featuresShaderFloat16Int8{};
  VkPhysicalDeviceIndexTypeUint8FeaturesEXT featuresIndexTypeUint8{};
  VkPhysicalDeviceSynchronization2FeaturesKHR featuresSynchronization2{};
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR featuresTimelineSemaphore{};
  VkPhysicalDeviceFragmentDensityMapFeaturesEXT featuresFragmentDensityMap{};
  VkPhysicalDeviceVulkanMemoryModelFeaturesKHR featuresVulkanMemoryModel{};
  VkPhysicalDevice8BitStorageFeaturesKHR features8BitStorage{};
  VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR featuresUniformBufferStandardLayout{};
  VkPhysicalDeviceScalarBlockLayoutFeaturesEXT featuresScalarBlockLayout{};
  VkPhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM featuresMultiviewPerViewViewports{};
  VkPhysicalDeviceMeshShaderFeaturesEXT featuresMeshShader{};
  VkPhysicalDeviceFragmentShadingRateFeaturesKHR featuresFragmentShadingRate{};

  VkPhysicalDeviceDescriptorBufferFeaturesEXT featuresDescriptorBuffer{};
  VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT featuresTextureCompressionAstcHdr{};

  // VK_KHR_shader_integer_dot_product (promoted to Vulkan 1.3). Opt-in: it defaults to
  // VK_FALSE and is chained into the feature list only when a caller has set it, so device
  // creation is unchanged for everyone who does not ask for it. Enabling this feature is what
  // makes the DotProduct SPIR-V capabilities legal to declare, so a caller that emits
  // dotPacked4x8EXT() (or any other OpSDot/OpUDot form) must set it.
  //
  // SET IT AFTER populateWithAvailablePhysicalDeviceFeatures(), and only for a device that
  // reports the feature. Ordering matters and neither half is enforced here: populate()
  // overwrites every CHAINED struct with the device's reported values, so a request made
  // before it is silently replaced by whatever the driver answers -- and this struct is not
  // chained until requested, so it is not even queried then. A request made after populate()
  // does reach vkCreateDevice(), where an unsupported one fails whole-device creation.
  // checkSelectedFeatures() does not cover this struct, so neither mistake is reported: the
  // first silently disables the feature and the second loses the device. Query the physical
  // device yourself and request only what it reports.
  // requestFeature() queries and selects this bit when the device is created.
  VkPhysicalDeviceShaderIntegerDotProductFeatures featuresShaderIntegerDotProduct{};

  // VK_EXT_extended_dynamic_state (promoted to Vulkan 1.3)
  VkPhysicalDeviceExtendedDynamicStateFeaturesEXT featuresExtendedDynamicState{};
  // VK_EXT_extended_dynamic_state2 (promoted to Vulkan 1.3)
  VkPhysicalDeviceExtendedDynamicState2FeaturesEXT featuresExtendedDynamicState2{};

  // We need to reassemble the feature chain because of the pNext pointers
  VulkanFeatures& operator=(const VulkanFeatures& other) noexcept;

  /// @brief Enumerates all instance extensions and stores their names internally in a vector of
  /// std::strings
  void enumerate(const VulkanFunctionTable& vf);

  /// @brief Enumerates all physical device extensions and stores their names internally in a vector
  /// of std::strings
  /// @param The physical device to use as a reference for the enumeration
  void enumerate(const VulkanFunctionTable& vf, VkPhysicalDevice device);

  /// @brief Returns all available extensions of a type
  /// @param extensionType The type of the extensions
  [[nodiscard]] const std::vector<std::string>& allAvailableExtensions(
      ExtensionType extensionType) const;

  /// @brief Returns true if the extension with name equal to `extensionName` of the type
  /// `extensionType` is available and false otherwise
  /// @param extensionName The name of the extension
  /// @param extensionType The type of the extensions to return
  bool available(const char* extensionName, ExtensionType extensionType) const;

  /// @brief Enables the common instance extensions used in IGL.
  /// @param contextConfig The VulkanContextConfig used to create the VulkanContext
  void enableCommonInstanceExtensions(const VulkanContextConfig& contextConfig);

  /// @brief Enables the common device extensions used in IGL.
  /// @param contextConfig The VulkanContextConfig used to create the VulkanContext
  void enableCommonDeviceExtensions(const VulkanContextConfig& contextConfig);

 public:
  friend class Device;
  friend class VulkanContext;

  // A copy of the config used by the VulkanContext
  VulkanContextConfig config{};

  // NOLINTBEGIN(readability-identifier-naming)
  bool has_VK_EXT_descriptor_buffer = false;
  bool has_VK_EXT_descriptor_indexing = false; // promoted to Vulkan 1.2
  bool has_VK_EXT_fragment_density_map = false;
  bool has_VK_EXT_headless_surface = false;
  bool has_VK_EXT_index_type_uint8 = false; // promoted to Vulkan 1.4
  bool has_VK_EXT_mesh_shader = false;
  bool has_VK_EXT_queue_family_foreign = false;
  bool has_VK_EXT_scalar_block_layout = false; // promoted to Vulkan 1.2
  bool has_VK_KHR_8bit_storage = false; // promoted to Vulkan 1.2
  bool has_VK_KHR_buffer_device_address = false; // promoted to Vulkan 1.2
  bool has_VK_KHR_create_renderpass2 = false; // promoted to Vulkan 1.2
  bool has_VK_KHR_get_surface_capabilities2 = false;
  bool has_VK_KHR_portability_enumeration = false;
  bool has_VK_KHR_shader_non_semantic_info = false; // promoted to Vulkan 1.3
  bool has_VK_KHR_synchronization2 = false; // promoted to Vulkan 1.3
  bool has_VK_KHR_timeline_semaphore = false; // promoted to Vulkan 1.2
  bool has_VK_KHR_uniform_buffer_standard_layout = false; // promoted to Vulkan 1.2
  bool has_VK_KHR_vulkan_memory_model = false; // promoted to Vulkan 1.2
  bool has_VK_QCOM_multiview_per_view_viewports = false;
  bool has_VK_EXT_extended_dynamic_state = false; // promoted to Vulkan 1.3
  bool has_VK_EXT_extended_dynamic_state2 = false; // promoted to Vulkan 1.3
  // NOLINTEND(readability-identifier-naming)

 private:
  static constexpr size_t kNumberOfExtensionTypes = 2;
  /// @brief a Vector of a vector of strings. The outer vector stores two vectors, one for each
  /// object type (instance and physical device). The inner vector is the list of all available
  /// extensions for the type
  std::vector<std::vector<std::string>> extensions_;

  /// @brief a Vector of unordered_sets of string. The outer vector stores two sets, one for each
  /// object type (instance and device). The inner set is the list of all extensions enabled for the
  /// type
  std::vector<std::unordered_set<std::string>> enabledExtensions_;

  /// @brief Assembles the feature chain for the VkPhysicalDeviceFeatures2 structure by connecting
  /// the existing/required feature structures and their pNext chain.
  void assembleFeatureChain(const VulkanContextConfig& contextConfig) noexcept;
  bool hasExtension(const char* ext) const;

  /// @brief Enables the extension with name `extensionName` of the type `extensionType` if the
  /// extension is available. If an instance or physical device deoesn't support the
  /// extension, this method is a no-op
  /// @param extensionName The name of the extension
  /// @param extensionType The type of the extension
  /// @return True if the extension is available, false otherwise
  bool enable(const char* extensionName, ExtensionType extensionType);

  /// @brief Returns true if an extension with name `extensionName` is enabled and false otherwise.
  /// This method will check the extension against the list of enabled ones for the
  /// instance and the physical device
  /// @param extensionName The name of the extension
  /// @return True if the extension has been enabled, false otherwise
  [[nodiscard]] bool enabled(const char* extensionName) const;

  /// @brief Returns a vector of `const char *` of all enabled extensions for an instance or phyical
  /// device. This method is particularly useful because Vulkan expects an
  /// array of `const char *` with the names of the extensions to enable
  /// @param extensionType The type of the extensions
  /// @return A vector of `const char *` with all enabled extensions of type `extensionType`. The
  /// return value must not outlive the instance of this class, as the pointers in the returned
  /// vector point to the strings stored internally in this class
  [[nodiscard]] std::vector<const char*> allEnabled(ExtensionType extensionType) const;

  std::vector<VkExtensionProperties> extensionProps_;

  struct alignas(uint64_t) FeatureWord {
    std::byte bytes[sizeof(uint64_t)]{};
  };

  struct FeatureRequest {
    std::string extensionName;
    uint32_t coreVersion;
    size_t size;
    std::vector<FeatureWord> requested;
    std::vector<FeatureWord> selected;
    bool chained = false;
    bool addedToChain = false;
  };
  std::vector<FeatureRequest> featureRequests_;

  void requestFeature(const void* IGL_NONNULL feature,
                      size_t size,
                      const char* IGL_NONNULL extensionName,
                      uint32_t coreVersion);
  void enableRequestedFeatures(const VulkanContext& context,
                               VkPhysicalDevice physicalDevice,
                               uint32_t apiVersion);
  void resetRequestedFeatures() noexcept;
  [[nodiscard]] std::array<VkBaseOutStructure*, 24> featureStructures() noexcept;
  [[nodiscard]] VkBaseOutStructure* requestedFeatureStructure(FeatureRequest& request) noexcept;
};

} // namespace igl::vulkan
