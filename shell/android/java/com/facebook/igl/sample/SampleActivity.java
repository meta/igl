/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

package com.facebook.igl.shell;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

public class SampleActivity extends Activity implements View.OnClickListener {
  private static final String TAG = "SampleActivity";
  private static final String FILL_VIEWPORT_METADATA = "com.facebook.igl.shell.FILL_VIEWPORT";
  private static final String VULKAN_ONLY_METADATA = "com.facebook.igl.shell.VULKAN_ONLY";
  private static final int REQUEST_CAMERA_PERMISSION = 1;

  // UI
  LinearLayout mMainView;
  LinearLayout mTabBar;
  FrameLayout mBackendViewFrame;

  // initialize runtime backend configuration / context
  private int curConfig = 0;
  private SampleLib.RenderSessionConfig[] mConfigs;
  private SurfaceView[] mTabViews;

  private final int selectedTabColor = Color.BLUE;
  private final int unSelectedTabColor = Color.GRAY;
  private boolean mViewsInitialized = false;

  protected boolean mEnableStencilBuffer = false;

  @Override
  protected void onCreate(Bundle icicle) {
    super.onCreate(icicle);

    // Keep screen on during benchmark - prevents the activity from going to sleep
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    if (shouldRequestCameraPermission()) {
      requestPermissions(new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
      return;
    }

    initializeViews();
  }

  private void initializeViews() {
    if (mViewsInitialized) {
      return;
    }
    mViewsInitialized = true;

    // configure the mainview
    mMainView = new LinearLayout(this);
    mMainView.setOrientation(LinearLayout.VERTICAL);

    // build tab bar from a horizontal linearlayout
    mTabBar = new LinearLayout(this);
    mTabBar.setOrientation(LinearLayout.HORIZONTAL);
    mTabBar.setGravity(Gravity.CENTER);
    mTabBar.setPadding(10, 5, 10, 5);

    // set up tab and cached sample view for different backend types
    mBackendViewFrame = new FrameLayout(this);
    mConfigs = SampleLib.getRenderSessionConfigs(getIntent());
    final int vulkanConfig = shouldUseVulkanOnly() ? findVulkanConfig() : -1;
    final boolean vulkanOnly = vulkanConfig >= 0;
    curConfig = vulkanOnly ? vulkanConfig : 0;
    mTabViews = new SurfaceView[mConfigs.length];
    for (int i = 0; i < mConfigs.length; i++) {
      if (vulkanOnly && i != curConfig) {
        continue;
      }

      TextView item = null;
      if (!vulkanOnly) {
        item = new TextView(this);
        item.setId(i);
        item.setText(mConfigs[i].displayName);
        item.setPadding(20, 0, 20, 0);
        item.setOnClickListener(this);
        mTabBar.addView(item);
      }

      // initialize sampleView for each backend type
      SurfaceView backendView = null;
      if (mConfigs[i].version.flavor == SampleLib.BackendFlavor.Vulkan) {
        backendView =
            new VulkanView(
                getApplication(),
                mConfigs[i].version,
                mConfigs[i].swapchainColorTextureFormat,
                mConfigs[i].swapchainColorSpace,
                getIntent());
      } else if (mConfigs[i].version.flavor == SampleLib.BackendFlavor.OpenGL_ES) {
        backendView =
            new SampleView(
                getApplication(),
                mConfigs[i].version,
                mConfigs[i].swapchainColorTextureFormat,
                mConfigs[i].swapchainColorSpace,
                mEnableStencilBuffer,
                getIntent());
        ((SampleView) backendView).onPause();
      }

      // set current backend tab as selected
      if (curConfig == i) {
        if (item != null) {
          item.setTextColor(selectedTabColor);
        }
        mBackendViewFrame.addView(backendView);
      } else {
        item.setTextColor(unSelectedTabColor);
      }

      // cache sampleView in map for reference
      mTabViews[i] = backendView;
    }

    // setup and display the mainview
    if (shouldFillViewport()) {
      if (!vulkanOnly) {
        mMainView.addView(
            mTabBar,
            new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
      }
      mMainView.addView(
          mBackendViewFrame,
          new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1.0f));
    } else {
      if (!vulkanOnly) {
        mMainView.addView(mTabBar);
      }
      mMainView.addView(mBackendViewFrame);
    }
    setContentView(mMainView);
  }

  private int findVulkanConfig() {
    for (int i = 0; i < mConfigs.length; i++) {
      if (mConfigs[i].version.flavor == SampleLib.BackendFlavor.Vulkan) {
        return i;
      }
    }
    Log.e(TAG, "No Vulkan configuration available; showing all configurations");
    return -1;
  }

  private boolean shouldFillViewport() {
    return isMetadataEnabled(FILL_VIEWPORT_METADATA);
  }

  private boolean shouldUseVulkanOnly() {
    return isMetadataEnabled(VULKAN_ONLY_METADATA);
  }

  private boolean isMetadataEnabled(String name) {
    try {
      Bundle metadata =
          getPackageManager()
              .getActivityInfo(getComponentName(), PackageManager.GET_META_DATA)
              .metaData;
      return metadata != null && metadata.getBoolean(name, false);
    } catch (PackageManager.NameNotFoundException e) {
      Log.w(TAG, "Could not inspect activity metadata for " + getPackageName(), e);
      return false;
    }
  }

  private boolean shouldRequestCameraPermission() {
    return Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
        && isPermissionDeclared(Manifest.permission.CAMERA)
        && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED;
  }

  private boolean isPermissionDeclared(String permission) {
    try {
      PackageInfo packageInfo =
          getPackageManager().getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
      String[] requestedPermissions = packageInfo.requestedPermissions;
      if (requestedPermissions == null) {
        return false;
      }
      for (String requestedPermission : requestedPermissions) {
        if (permission.equals(requestedPermission)) {
          return true;
        }
      }
    } catch (PackageManager.NameNotFoundException e) {
      Log.w(TAG, "Could not inspect requested permissions for " + getPackageName(), e);
    }
    return false;
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == REQUEST_CAMERA_PERMISSION) {
      if (grantResults.length == 0 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
        // Camera input is optional: the demo still runs without it, so initialize
        // the UI regardless of the user's choice.
        Log.w(TAG, "CAMERA permission not granted; continuing without camera input.");
      }
      initializeViews();
      // The permission result arrives after the activity's onResume() has already
      // run (and early-returned while mViewsInitialized was false), so the freshly
      // created OpenGL_ES view would stay paused. Resume it here.
      resumeCurrentView();
    }
  }

  @Override
  public void onClick(View view) {
    int prevConfig = curConfig;
    for (int i = 0; i < mTabBar.getChildCount(); i++) {
      TextView item = (TextView) mTabBar.getChildAt(i);

      // if not the selected tab, reset the tab status and skip
      if (view.getId() != item.getId()) {
        item.setTextColor(unSelectedTabColor);
        continue;
      }

      // reset main view if the selected backend is not the current
      if (prevConfig != i) {
        if (mConfigs[prevConfig].version.flavor != SampleLib.BackendFlavor.Vulkan) {
          ((SampleView) mTabViews[prevConfig]).onPause();
        }
        item.setTextColor(selectedTabColor);
        curConfig = i;
        mBackendViewFrame.removeAllViews();
        SurfaceView surfaceView = mTabViews[curConfig];
        mBackendViewFrame.addView(surfaceView);
        SampleLib.setActiveBackendVersion(mConfigs[curConfig].version);
        if (mConfigs[curConfig].version.flavor != SampleLib.BackendFlavor.Vulkan) {
          ((SampleView) mTabViews[curConfig]).onResume();
        }
      }
    }
  }

  @Override
  protected void onPause() {
    super.onPause();
    if (!mViewsInitialized) {
      return;
    }
    if (mConfigs[curConfig].version.flavor != SampleLib.BackendFlavor.Vulkan) {
      ((SampleView) mTabViews[curConfig]).onPause();
    }
  }

  @Override
  protected void onResume() {
    super.onResume();
    resumeCurrentView();
  }

  // Resumes the current OpenGL_ES tab view. Vulkan views manage their own render
  // thread and are not resumed here. Called both from the normal onResume()
  // lifecycle and from onRequestPermissionsResult() (where onResume() has already
  // run before the views existed); SampleView.onResume() is idempotent.
  private void resumeCurrentView() {
    if (!mViewsInitialized) {
      return;
    }
    if (mConfigs[curConfig].version.flavor != SampleLib.BackendFlavor.Vulkan) {
      ((SampleView) mTabViews[curConfig]).onResume();
    }
  }
}
