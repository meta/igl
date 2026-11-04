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
import android.graphics.PixelFormat;
import android.opengl.EGL15;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import java.util.concurrent.CountDownLatch;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;
import javax.microedition.khronos.opengles.GL10;

/// Simple view that sets up a GLES 2.0 rendering context
public class SampleView extends GLSurfaceView {
  private static String TAG = "SampleView";
  private float lastTouchX = 0.0f;
  private float lastTouchY = 0.0f;
  private CountDownLatch renderSessionInitLatch = new CountDownLatch(1);
  private Intent mIntent;
  private SampleLib.DisplayRateWatcher mDisplayRateWatcher;

  @SuppressWarnings("this-escape")
  public SampleView(
      Context context,
      SampleLib.BackendVersion backendVersion,
      int swapchainColorTextureFormat,
      boolean enableStencilBuffer,
      Intent intent) {
    super(context);
    init(context, backendVersion, swapchainColorTextureFormat, enableStencilBuffer, intent);
  }

  @SuppressWarnings("this-escape")
  public SampleView(
      Context context,
      SampleLib.BackendVersion backendVersion,
      int swapchainColorTextureFormat,
      Intent intent) {
    super(context);
    init(context, backendVersion, swapchainColorTextureFormat, false, intent);
  }

  @SuppressWarnings("this-escape")
  private void init(
      Context context,
      SampleLib.BackendVersion backendVersion,
      int swapchainColorTextureFormat,
      boolean enableStencilBuffer,
      Intent intent) {

    // Uncomment to attach debugging
    // android.os.Debug.waitForDebugger();
    mIntent = intent;

    setEGLContextFactory(new ContextFactory(backendVersion));

    // Set the view to be transluscent since we provide an alpha channel below.
    this.getHolder().setFormat(PixelFormat.TRANSLUCENT);

    setEGLWindowSurfaceFactory(
        new SurfaceFactory(SampleLib.isSRGBTextureFormat(swapchainColorTextureFormat)));

    setEGLConfigChooser(new ConfigChooser(backendVersion, enableStencilBuffer));

    setRenderer(
        new Renderer(
            context,
            backendVersion,
            swapchainColorTextureFormat,
            renderSessionInitLatch,
            mIntent,
            this.getHolder()));

    // queueEvent() is the only hop onto the GL thread this view has, and the rates must land
    // there because reinstalling the backend re-applies the EGL swap interval on the surface
    // that thread owns.
    mDisplayRateWatcher =
        new SampleLib.DisplayRateWatcher(
            context,
            (currentHz, maxHz) -> queueEvent(() -> SampleLib.setDisplayRates(currentHz, maxHz)));
  }

  @Override
  protected void onAttachedToWindow() {
    super.onAttachedToWindow();
    mDisplayRateWatcher.start();
  }

  @Override
  protected void onDetachedFromWindow() {
    mDisplayRateWatcher.stop();
    super.onDetachedFromWindow();
  }

  @Override
  public void surfaceDestroyed(SurfaceHolder holder) {
    // The native renderers are process-lifetime, so nothing else ever drops the ANativeWindow
    // reference taken from this Surface: without this the destroyed window stays retained for
    // the life of the process and the next init() reinstalls the presentation-rate backend
    // against it. GLSurfaceView delivers this callback on the main thread while the reference
    // is owned by the GL thread, so the release is queued onto that thread. super's
    // implementation then blocks until the GL thread acknowledges the destruction, and the GL
    // thread drains its event queue while waiting, so the release runs before it idles.
    final Surface surface = holder.getSurface();
    queueEvent(() -> SampleLib.surfaceDestroyed(surface));
    super.surfaceDestroyed(holder);
  }

  public boolean isRenderSessionInitialized() {
    return renderSessionInitLatch.getCount() == 0;
  }

  public void awaitRenderSessionInitialization() throws InterruptedException {
    renderSessionInitLatch.await();
  }

  @Override
  public void setBackgroundColor(int color) {
    int A = (color >> 24) & 0xff;
    int R = (color >> 16) & 0xff;
    int G = (color >> 8) & 0xff;
    int B = (color) & 0xff;
    SampleLib.setClearColorValue(R, G, B, A);
  }

  @Override
  public boolean onTouchEvent(MotionEvent e) {
    float x = e.getX();
    float y = e.getY();
    float dx = x - lastTouchX;
    float dy = y - lastTouchY;

    lastTouchX = x;
    lastTouchY = y;

    switch (e.getAction()) {
      case MotionEvent.ACTION_DOWN:
        SampleLib.touchEvent(true, x, y, 0, 0);
        return true;

      case MotionEvent.ACTION_MOVE:
        SampleLib.touchEvent(true, x, y, dx, dy);
        return true;

      case MotionEvent.ACTION_UP:
        SampleLib.touchEvent(false, x, y, 0, 0);
        return true;
    }

    return false;
  }

  /// Context factory: handles creating the EGL context for this view with the correct settings.
  private static class ContextFactory implements GLSurfaceView.EGLContextFactory {

    private final SampleLib.BackendVersion mBackendVersion;

    public ContextFactory(SampleLib.BackendVersion version) {
      mBackendVersion = version;
    }

    public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig eglConfig) {
      final int EGL_CONTEXT_CLIENT_VERSION = 0x3098;
      int[] attrib_list = {
        EGL_CONTEXT_CLIENT_VERSION, mBackendVersion.majorVersion, EGL10.EGL_NONE
      };
      EGLContext context =
          egl.eglCreateContext(display, eglConfig, EGL10.EGL_NO_CONTEXT, attrib_list);
      checkEglError("Error creating EGL context", egl);
      return context;
    }

    public void destroyContext(EGL10 egl, EGLDisplay display, EGLContext context) {
      egl.eglDestroyContext(display, context);
    }
  }

  private static void checkEglError(String prompt, EGL10 egl) {
    int error;
    while ((error = egl.eglGetError()) != EGL10.EGL_SUCCESS) {
      Log.e(TAG, String.format("%s: EGL error: 0x%x", prompt, error));
    }
  }

  private static class SurfaceFactory implements GLSurfaceView.EGLWindowSurfaceFactory {
    final int EGL_GL_COLORSPACE_KHR = 0x309D;
    final int EGL_GL_COLORSPACE_SRGB_KHR = 0x3089;
    final int EGL_GL_COLORSPACE_LINEAR_KHR = 0x308A;

    private boolean mIsSRGBColorSpace;

    SurfaceFactory(boolean isSRGB) {
      mIsSRGBColorSpace = isSRGB;
    }

    @Override
    @SuppressWarnings("static")
    public EGLSurface createWindowSurface(
        EGL10 egl10, EGLDisplay eglDisplay, EGLConfig eglConfig, Object nativeWindow) {
      String eglExtensionString = egl10.eglQueryString(eglDisplay, egl10.EGL_EXTENSIONS);
      if (!eglExtensionString.contains("EGL_KHR_gl_colorspace")) {
        return egl10.eglCreateWindowSurface(eglDisplay, eglConfig, nativeWindow, null);
      }
      int[] configAttribs = {
        EGL_GL_COLORSPACE_KHR,
        (mIsSRGBColorSpace ? EGL_GL_COLORSPACE_SRGB_KHR : EGL_GL_COLORSPACE_LINEAR_KHR),
        EGL10.EGL_NONE
      };

      return egl10.eglCreateWindowSurface(eglDisplay, eglConfig, nativeWindow, configAttribs);
    }

    @Override
    public void destroySurface(EGL10 egl10, EGLDisplay eglDisplay, EGLSurface eglSurface) {
      egl10.eglDestroySurface(eglDisplay, eglSurface);
    }
  }

  /// Config chooser: handles specifying the requirements for the EGL config and choosing the
  // correct one.
  private static class ConfigChooser implements GLSurfaceView.EGLConfigChooser {

    private final SampleLib.BackendVersion mBackendVersion;

    private boolean mEnableStencilBuffer = false;

    public ConfigChooser(SampleLib.BackendVersion version, boolean enableStencilBuffer) {
      mBackendVersion = version;
      mEnableStencilBuffer = enableStencilBuffer;
    }

    public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display) {
      final int EGL_OPENGL_ES2_BIT = 4;

      // Set ourselves a strict configuration: RGBA8888, 16-bit depth buffer, no stencil.
      final int[] configAttribs = {
        EGL10.EGL_RED_SIZE, 8,
        EGL10.EGL_GREEN_SIZE, 8,
        EGL10.EGL_BLUE_SIZE, 8,
        EGL10.EGL_ALPHA_SIZE, 8,
        EGL10.EGL_DEPTH_SIZE, 16,
        EGL10.EGL_STENCIL_SIZE, mEnableStencilBuffer ? 8 : 0,
        EGL10.EGL_RENDERABLE_TYPE,
            (mBackendVersion.majorVersion == (byte) 3)
                ? EGL15.EGL_OPENGL_ES3_BIT
                : EGL_OPENGL_ES2_BIT,
        EGL10.EGL_NONE
      };

      int[] numConfigs = new int[1];
      egl.eglChooseConfig(display, configAttribs, null, 0, numConfigs);

      if (numConfigs[0] <= 0) {
        throw new IllegalArgumentException("Couldn't find an appropriate EGL config");
      }

      EGLConfig[] configs = new EGLConfig[1];
      egl.eglChooseConfig(display, configAttribs, configs, 1, numConfigs);

      return configs[0];
    }
  }

  /// Renderer: This class communicates with our JNI library to implement the OpenGL rendering.
  private static class Renderer implements GLSurfaceView.Renderer {
    private final Context mContext;
    private final Intent mIntent;
    private final SampleLib.BackendVersion mBackendVersion;
    private final int mSwapchainColorTextureFormat;
    private CountDownLatch mRenderSessionInitLatch;
    /// GLSurfaceView drives GL through its own EGL surface and hands the Renderer nothing that
    /// identifies the window, but the native side needs the Surface to hold an ANativeWindow
    /// and ask the compositor for the panel's best mode. This is the only route to it.
    private final SurfaceHolder mSurfaceHolder;

    Renderer(
        Context context,
        SampleLib.BackendVersion backendVersion,
        int swapchainColorTextureFormat,
        CountDownLatch renderSessionInitLatch,
        Intent intent,
        SurfaceHolder surfaceHolder) {
      mContext = context;
      mIntent = intent;
      mBackendVersion = backendVersion;
      mSwapchainColorTextureFormat = swapchainColorTextureFormat;
      mRenderSessionInitLatch = renderSessionInitLatch;
      mSurfaceHolder = surfaceHolder;
    }

    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
      SampleLib.init(
          mBackendVersion,
          mSwapchainColorTextureFormat,
          mContext.getAssets(),
          mSurfaceHolder.getSurface(),
          mIntent,
          SampleLib.displayCurrentRefreshRateHz(mContext),
          SampleLib.displayMaxRefreshRateHz(mContext));

      // Signal that application has being started.
      mRenderSessionInitLatch.countDown();
    }

    public void onSurfaceChanged(GL10 gl, int width, int height) {
      SampleLib.surfaceChanged(
          mSurfaceHolder.getSurface(),
          width,
          height,
          SampleLib.displayCurrentRefreshRateHz(mContext),
          SampleLib.displayMaxRefreshRateHz(mContext));
    }

    public void onDrawFrame(GL10 gl) {
      if (SampleLib.isHeadless()) {
        // Headless mode: run an unrestricted render loop directly on the GL thread.
        // This prevents GLSurfaceView from calling eglSwapBuffers (which blocks on vsync)
        // between frames, allowing maximum throughput for benchmarks.
        android.util.Log.i("igl", "Starting unrestricted render loop (headless GLES mode)");
        float density = mContext.getResources().getDisplayMetrics().density;
        while (true) {
          if (renderAndCheckExit(density)) {
            return;
          }
        }
      }
      renderAndCheckExit(mContext.getResources().getDisplayMetrics().density);
    }

    private boolean renderAndCheckExit(float density) {
      boolean shouldExit = SampleLib.render(density);
      if (shouldExit) {
        android.util.Log.i(
            "igl", "[IGL Benchmark] Java: Benchmark complete, waiting for logs to flush...");
        try {
          Thread.sleep(2000);
        } catch (InterruptedException e) {
          // Ignore
        }
        android.util.Log.i("igl", "[IGL Benchmark] Java: Exiting process");
        System.exit(0);
      }
      return shouldExit;
    }
  }
}
