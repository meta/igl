/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

package com.facebook.igl.shell;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.hardware.display.DisplayManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.view.Display;
import android.view.Surface;
import android.view.WindowManager;

// Wrapper for our native C++ library, which implements the actual rendering.
public class SampleLib {

  static {
    System.loadLibrary("sampleJni");
  }

  // Must match igl/Common.h
  public static enum BackendFlavor {
    Invalid,
    OpenGL,
    OpenGL_ES,
    Metal,
    Vulkan,
    D3D12,
    // @fb-only
    WebGPU
  }

  // Must match igl/DeviceFeatures.h
  public static class BackendVersion {
    BackendFlavor flavor;
    byte majorVersion;
    byte minorVersion;

    public BackendVersion(BackendFlavor flavor, byte majorVersion, byte minorVersion) {
      this.flavor = flavor;
      this.majorVersion = majorVersion;
      this.minorVersion = minorVersion;
    }
  }

  public static native RenderSessionConfig[] getRenderSessionConfigs(Intent intent);

  public static native boolean init(
      BackendVersion backendVersion,
      int swapchainColorTextureFormat,
      int swapchainColorSpace,
      AssetManager assetManager,
      Surface surface,
      Intent intent,
      float displayCurrentRefreshRateHz,
      float displayMaxRefreshRateHz);

  /**
   * The rate the display {@code context} is showing on is refreshing at <em>right now</em>, in Hz,
   * or 0 when it cannot be determined. The NDK has no equivalent query, so this is the only source
   * the native presentation-rate seam has for the number; without it every rate request is refused
   * rather than granted against a guessed refresh rate.
   *
   * <p>This is the mode currently in force, not the panel's best mode — every cadence the seam can
   * hold is a whole division of this one, so a divisor derived from anything else paces wrong. See
   * {@link #displayMaxRefreshRateHz} for the ceiling.
   */
  public static float displayCurrentRefreshRateHz(Context context) {
    Display display = display(context);
    return display == null ? 0.0f : display.getRefreshRate();
  }

  /**
   * The highest refresh rate any mode of {@code context}'s display supports, in Hz, or 0 when it
   * cannot be determined.
   *
   * <p>{@link Display#getRefreshRate} answers for the mode in force, which on a 120 Hz panel
   * sitting at 60 is 60. Reporting that as the ceiling would make the "max" rung unreachable on
   * exactly the hardware it exists for, so the maximum is taken across {@link
   * Display#getSupportedModes}. Only the compositor can actually switch modes; this is the number
   * it is asked for.
   */
  public static float displayMaxRefreshRateHz(Context context) {
    Display display = display(context);
    if (display == null) {
      return 0.0f;
    }
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
      // No supported-modes query exists below M: report unknown (0) rather
      // than the current rate as the ceiling. Native stands in the current
      // rate when the maximum is unknown, so nothing is lost.
      return 0.0f;
    }
    float maxHz = display.getRefreshRate();
    Display.Mode[] modes = display.getSupportedModes();
    if (modes != null) {
      for (Display.Mode mode : modes) {
        if (mode != null && mode.getRefreshRate() > maxHz) {
          maxHz = mode.getRefreshRate();
        }
      }
    }
    return maxHz;
  }

  /**
   * Hands the native presentation-rate seam a fresh pair of rates and reinstalls the backend
   * against them.
   *
   * <p>Must be called on the render thread of the active backend — it rebuilds the frame divider,
   * which on GLES reads and writes the EGL surface that thread owns. {@link DisplayRateWatcher}
   * exists to do that hop; nothing should call this straight from the main thread.
   */
  public static native void setDisplayRates(
      float displayCurrentRefreshRateHz, float displayMaxRefreshRateHz);

  /**
   * Reports refresh-rate changes that arrive without a surface callback.
   *
   * <p>The rates are otherwise sampled only when a surface is created or changed, and a panel can
   * change mode without either happening — 60 to 120 and back is routine on a variable-refresh
   * display. The seam would then keep dividing a rate the panel has left, so a grant of 60 on a
   * panel that has moved to 120 would pace at 120 while still reporting 60.
   *
   * <p>{@link DisplayManager.DisplayListener} fires on the main thread, but the rates have to reach
   * native on the render thread that owns the backend. Each view knows how to schedule work on its
   * own render thread and this class does not, so the hop is the caller's, supplied as a {@link
   * Sink}.
   */
  public static final class DisplayRateWatcher {

    /**
     * Receives new rates on the main thread. Implementations are responsible for getting them onto
     * their backend's render thread before calling {@link SampleLib#setDisplayRates}.
     */
    public interface Sink {
      void onDisplayRatesChanged(float currentHz, float maxHz);
    }

    private final Context context;
    private final Sink sink;
    private final DisplayManager displayManager;
    private final DisplayManager.DisplayListener listener;
    private boolean started = false;

    public DisplayRateWatcher(Context context, Sink sink) {
      this.context = context;
      this.sink = sink;
      this.displayManager = (DisplayManager) context.getSystemService(Context.DISPLAY_SERVICE);
      this.listener =
          new DisplayManager.DisplayListener() {
            @Override
            public void onDisplayAdded(int displayId) {}

            @Override
            public void onDisplayRemoved(int displayId) {}

            @Override
            public void onDisplayChanged(int displayId) {
              // The rates are re-read rather than filtered on displayId: a mode switch, a
              // rotation and a move to another screen all arrive here looking the same, and in
              // the last case the display this context resolves to is a different one than the
              // id would have been matched against. Two cheap queries beat getting that wrong.
              onDisplayRatesChanged();
            }
          };
    }

    private void onDisplayRatesChanged() {
      sink.onDisplayRatesChanged(
          displayCurrentRefreshRateHz(context), displayMaxRefreshRateHz(context));
    }

    /** Idempotent — registering the same listener twice would double every callback. */
    public void start() {
      if (started || displayManager == null) {
        return;
      }
      displayManager.registerDisplayListener(listener, new Handler(Looper.getMainLooper()));
      started = true;
    }

    /** Idempotent, and safe to call when {@link #start} never ran or already failed. */
    public void stop() {
      if (!started) {
        return;
      }
      displayManager.unregisterDisplayListener(listener);
      started = false;
    }
  }

  private static Display display(Context context) {
    Display display = null;
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
      // Context.getDisplay() throws on a context that is not showing anything; both callers
      // pass the Activity, but a future one may not, and a thrown exception here would take
      // down a render thread over a profiling convenience.
      try {
        display = context.getDisplay();
      } catch (UnsupportedOperationException e) {
        display = null;
      }
    }
    if (display == null) {
      WindowManager windowManager =
          (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
      if (windowManager != null) {
        display = windowManager.getDefaultDisplay();
      }
    }
    return display;
  }

  public static native void setActiveBackendVersion(BackendVersion backendVersion);

  public static native void surfaceChanged(
      Surface surface,
      int width,
      int height,
      float displayCurrentRefreshRateHz,
      float displayMaxRefreshRateHz);

  public static native boolean render(float displayScale);

  public static native void touchEvent(boolean isDown, float x, float y, float dx, float dy);

  public static native void setClearColorValue(float r, float g, float b, float a);

  public static native boolean isSRGBTextureFormat(int textureFormat);

  public static native void surfaceDestroyed(Surface surface);

  /// @brief Returns true if the active renderer is in headless mode.
  /// Must be called after init().
  public static native boolean isHeadless();

  public static class RenderSessionConfig {
    String displayName;
    BackendVersion version;
    int swapchainColorTextureFormat;
    int swapchainColorSpace;

    public RenderSessionConfig(
        String displayName,
        BackendVersion version,
        int swapchainColorTextureFormat,
        int swapchainColorSpace) {
      this.displayName = displayName;
      this.version = version;
      this.swapchainColorTextureFormat = swapchainColorTextureFormat;
      this.swapchainColorSpace = swapchainColorSpace;
    }
  }
}
