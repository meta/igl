/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @brief Return type for type safety with C factory function
 */
struct IGLShellExtension {};

/**
 * @brief C factory function for creating new function
 *
 * @return New instance of Extension
 *
 * @note Callers usually use ExtensionLoader instead of calling this directly.
 * @note Callers should call `delete` to free the extension
 *
 * @see ExtensionLoader
 */
using IGLShellExtension_NewCFunction = IGLShellExtension* (*)();

namespace igl::shell {

class Platform;

/**
 * @brief Base class for shell extensions loaded by ExtensionLoader
 */
class Extension : public IGLShellExtension {
 public:
  virtual ~Extension() = default;
  virtual bool initialize(Platform& /*platform*/) noexcept;
};

inline bool Extension::initialize(Platform& /*platform*/) noexcept {
  return true;
}

} // namespace igl::shell
