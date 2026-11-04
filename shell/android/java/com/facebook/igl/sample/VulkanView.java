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
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.Choreographer;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

// Simple view that sets up a Vulkan Render view
public class VulkanView extends SurfaceView
    implements SurfaceHolder.Callback2, Choreographer.FrameCallback {
  private static String TAG = "VulkanView";
  private float lastTouchX = 0.0f;
  private float lastTouchY = 0.0f;
  Context mContext;
  RenderThread mRenderThread;
  private final SampleLib.BackendVersion mBackendVersion;
  private final int mSwapchainColorTextureFormat;
  private Intent mIntent;
  private final SampleLib.DisplayRateWatcher mDisplayRateWatcher;

  @SuppressWarnings("this-escape")
  public VulkanView(
      Context context,
      SampleLib.BackendVersion backendVersion,
      int swapchainColorTextureFormat,
      Intent intent) {

    super(context);

    // Set the view to be translucent since we provide an alpha channel below.
    this.getHolder().setFormat(PixelFormat.TRANSLUCENT);
    this.getHolder().addCallback(this);

    mContext = context;
    mBackendVersion = backendVersion;
    mSwapchainColorTextureFormat = swapchainColorTextureFormat;
    mIntent = intent;

    // The render thread's Handler is the hop onto it. A null thread or handler means there is
    // no backend installed to update, and the next surfaceCreated() samples the rates anyway.
    mDisplayRateWatcher =
        new SampleLib.DisplayRateWatcher(
            context,
            (currentHz, maxHz) -> {
              RenderThread renderThread = mRenderThread;
              RenderHandler rh = renderThread == null ? null : renderThread.getHandler();
              if (rh != null) {
                rh.sendDisplayRatesChanged(currentHz, maxHz);
              }
            });
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

  @Override
  public void surfaceRedrawNeeded(SurfaceHolder surfaceHolder) {}

  @Override
  public void surfaceCreated(SurfaceHolder surfaceHolder) {
    // Start rendering on the RenderThread
    mRenderThread =
        new RenderThread(mContext, surfaceHolder, mBackendVersion, mSwapchainColorTextureFormat);
    mRenderThread.setName("Vulkan Render Thread");
    mRenderThread.start();
    mRenderThread.waitUntilReady();

    RenderHandler rh = mRenderThread.getHandler();
    if (rh != null) {
      // Send surface created — the RenderThread will call init() and then decide
      // whether to enter headless (unrestricted) or vsync (Choreographer) mode.
      // We cannot check isHeadless() here because init() runs asynchronously on
      // the RenderThread and hasn't read shell params yet.
      rh.sendSurfaceCreated();
    }
  }

  @Override
  public void surfaceChanged(SurfaceHolder surfaceHolder, int format, int width, int height) {
    RenderHandler rh = mRenderThread.getHandler();
    if (rh != null) {
      rh.sendSurfaceChanged(format, width, height);
    }
  }

  @Override
  public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
    // Stop render thread
    RenderHandler rh = mRenderThread.getHandler();
    if (rh != null) {
      rh.sendShutdown();
      try {
        mRenderThread.join();
      } catch (InterruptedException ie) {
        // not expected
        throw new RuntimeException("join was interrupted", ie);
      }
    }
    mRenderThread = null;

    // If the callback was posted, remove it.  Without this, we could get one more
    // call on doFrame().
    Choreographer.getInstance().removeFrameCallback(this);
  }

  /*
   * Choreographer callback, called near vsync.
   *
   * @see android.view.Choreographer.FrameCallback#doFrame(long)
   */
  @Override
  public void doFrame(long frameTimeNanos) {
    RenderHandler rh = mRenderThread.getHandler();
    if (rh != null) {
      Choreographer.getInstance().postFrameCallback(this);
      rh.sendDoFrame(frameTimeNanos);
    }
  }

  // Rendering thread
  private class RenderThread extends Thread {
    private Context mContext;
    private SurfaceHolder mSurfaceHolder;
    private volatile RenderHandler mHandler;

    private Object mStartLock = new Object();
    private boolean mReady = false;
    private final SampleLib.BackendVersion mBackendVersion;
    private final int mSwapchainColorTextureFormat;

    public RenderThread(
        Context context,
        SurfaceHolder surfaceHolder,
        SampleLib.BackendVersion backendVersion,
        int swapchainColorTextureformat) {
      mContext = context;
      mSurfaceHolder = surfaceHolder;
      mBackendVersion = backendVersion;
      mSwapchainColorTextureFormat = swapchainColorTextureformat;
    }

    public RenderHandler getHandler() {
      return mHandler;
    }

    @Override
    public void run() {
      Looper.prepare();
      mHandler = new RenderHandler(this);
      synchronized (mStartLock) {
        mReady = true;
        mStartLock.notify(); // signal waitUntilReady()
      }

      Looper.loop();

      // surface has been destroyed
      synchronized (mStartLock) {
        mReady = false;
      }
      Surface surface = mSurfaceHolder.getSurface();
      SampleLib.surfaceDestroyed(surface);
    }

    public void waitUntilReady() {
      synchronized (mStartLock) {
        while (!mReady) {
          try {
            mStartLock.wait();
          } catch (InterruptedException ie) {
            /* not expected */
          }
        }
      }
    }

    public void surfaceCreated() {
      Log.d(TAG, "SurfaceCreated");
      Surface surface = mSurfaceHolder.getSurface();
      SampleLib.init(
          mBackendVersion,
          mSwapchainColorTextureFormat,
          mContext.getAssets(),
          surface,
          mIntent,
          SampleLib.displayCurrentRefreshRateHz(mContext),
          SampleLib.displayMaxRefreshRateHz(mContext));

      // Now that init() has completed and shell params are read, decide render mode.
      if (SampleLib.isHeadless()) {
        // Headless: enter unrestricted render loop directly on this thread
        startUnrestrictedLoop();
      } else {
        // Normal: kick off vsync-driven rendering via Choreographer on the UI thread
        new Handler(Looper.getMainLooper())
            .post(
                () -> {
                  if (mRenderThread != null) {
                    Choreographer.getInstance().postFrameCallback(VulkanView.this);
                  }
                });
      }
    }

    public void surfaceChanged(int width, int height) {
      Surface surface = mSurfaceHolder.getSurface();
      SampleLib.surfaceChanged(
          surface,
          width,
          height,
          SampleLib.displayCurrentRefreshRateHz(mContext),
          SampleLib.displayMaxRefreshRateHz(mContext));
    }

    /**
     * Applies a display mode change that arrived without a surface callback. Runs on the render
     * thread, which is where reinstalling the backend has to happen.
     */
    public void displayRatesChanged(float currentHz, float maxHz) {
      SampleLib.setDisplayRates(currentHz, maxHz);
    }

    /** draw frame in response to a vsync event. */
    private void doFrame(long timeStampNanos) {
      long diff = System.nanoTime() - timeStampNanos;
      long max = 16666667l - 2000000; // if we're within 2ms, don't bother
      if (diff > max) {
        // too much, drop a frame
        Log.d(
            TAG,
            "diff is "
                + (diff / 1000000.0)
                + " ms, max "
                + (max / 1000000.0)
                + ", skipping render");
        return;
      }

      boolean shouldExit = SampleLib.render(mContext.getResources().getDisplayMetrics().density);
      if (shouldExit) {
        android.util.Log.i(
            "igl", "[IGL Benchmark] Java: Benchmark complete, waiting for logs to flush...");
        // Give logcat time to flush the final report before killing the process
        try {
          Thread.sleep(2000);
        } catch (InterruptedException e) {
          // Ignore
        }
        android.util.Log.i("igl", "[IGL Benchmark] Java: Exiting process");
        System.exit(0);
      }
    }

    /** Run an unrestricted render loop (no vsync gating) for headless benchmarks. */
    private void startUnrestrictedLoop() {
      Log.i(TAG, "Starting unrestricted render loop (headless mode)");
      float density = mContext.getResources().getDisplayMetrics().density;
      while (true) {
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
      }
    }

    private void shutdown() {
      Log.d(TAG, "shutdown");
      Looper.myLooper().quit();
    }
  }

  /**
   * Handler for RenderThread. Used for messages sent from the UI thread to the render thread.
   *
   * <p>The object is created on the render thread, and the various "send" methods are called from
   * the UI thread.
   */
  private static class RenderHandler extends Handler {
    private static final int MSG_SURFACE_CREATED = 0;
    private static final int MSG_SURFACE_CHANGED = 1;
    private static final int MSG_DO_FRAME = 2;
    private static final int MSG_SHUTDOWN = 3;
    private static final int MSG_DISPLAY_RATES_CHANGED = 4;

    private RenderThread mRenderThread;

    public RenderHandler(RenderThread rt) {
      mRenderThread = rt;
    }

    public void sendSurfaceCreated() {
      sendMessage(obtainMessage(RenderHandler.MSG_SURFACE_CREATED));
    }

    public void sendSurfaceChanged(@SuppressWarnings("unused") int format, int width, int height) {
      // ignore format
      sendMessage(obtainMessage(RenderHandler.MSG_SURFACE_CHANGED, width, height));
    }

    public void sendDoFrame(long frameTimeNanos) {
      sendMessage(
          obtainMessage(
              RenderHandler.MSG_DO_FRAME, (int) (frameTimeNanos >> 32), (int) frameTimeNanos));
    }

    public void sendShutdown() {
      sendMessage(obtainMessage(RenderHandler.MSG_SHUTDOWN));
    }

    public void sendDisplayRatesChanged(float currentHz, float maxHz) {
      // A Message carries two ints and these are floats, so they travel as their bit patterns
      // — the same trick sendDoFrame() uses to get a long across.
      sendMessage(
          obtainMessage(
              RenderHandler.MSG_DISPLAY_RATES_CHANGED,
              Float.floatToRawIntBits(currentHz),
              Float.floatToRawIntBits(maxHz)));
    }

    @Override // runs on RenderThread
    public void handleMessage(Message msg) {
      int what = msg.what;
      // Log.d(TAG, "RenderHandler [" + this + "]: what=" + what);

      if (mRenderThread == null) {
        Log.w(TAG, "RenderHandler.handleMessage: mRenderThread is null");
        return;
      }

      switch (what) {
        case MSG_SURFACE_CREATED:
          mRenderThread.surfaceCreated();
          break;
        case MSG_SURFACE_CHANGED:
          mRenderThread.surfaceChanged(msg.arg1, msg.arg2);
          break;
        case MSG_DO_FRAME:
          long timestamp = (((long) msg.arg1) << 32) | (((long) msg.arg2) & 0xffffffffL);
          mRenderThread.doFrame(timestamp);
          break;
        case MSG_SHUTDOWN:
          mRenderThread.shutdown();
          break;
        case MSG_DISPLAY_RATES_CHANGED:
          mRenderThread.displayRatesChanged(
              Float.intBitsToFloat(msg.arg1), Float.intBitsToFloat(msg.arg2));
          break;
        default:
          throw new RuntimeException("unknown message " + what);
      }
    }
  }
}
