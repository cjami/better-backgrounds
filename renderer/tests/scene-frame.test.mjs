import assert from 'node:assert/strict';
import test from 'node:test';

import {
  SceneRenderer,
  bytesToBase64,
  flipPixelRows,
  hasPixelVariation,
} from '../src/main.mjs';

test('scene capture reads the WebGL framebuffer before publishing its bitmap', () => {
  const operations = [];
  const source = { width: 2, height: 2 };
  const previousOffscreenCanvas = globalThis.OffscreenCanvas;
  const previousImageData = globalThis.ImageData;
  globalThis.OffscreenCanvas = class {
    constructor(width, height) {
      this.width = width;
      this.height = height;
    }

    getContext() {
      return { putImageData: (image) => operations.push(['copy', [...image.data]]) };
    }

  };
  globalThis.ImageData = class {
    constructor(data, width, height) {
      this.data = data;
      this.width = width;
      this.height = height;
    }
  };

  try {
    const renderer = Object.create(SceneRenderer.prototype);
    renderer.app = {
      graphicsDevice: {
        canvas: source,
        readPixels: (_x, _y, _width, _height, pixels) => {
          operations.push(['read']);
          pixels.set([
            1, 2, 3, 255, 4, 5, 6, 255,
            20, 30, 40, 255, 50, 60, 70, 255,
          ]);
        },
      },
    };
    renderer.sceneSnapshot = null;
    renderer.sceneSnapshotContext = null;
    renderer.scenePixels = null;

    assert.deepEqual(renderer.captureSceneFrame(), {
      hasContent: true,
    });
    assert.deepEqual(operations, [
      ['read'],
      ['copy', [20, 30, 40, 255, 50, 60, 70, 255, 1, 2, 3, 255, 4, 5, 6, 255]],
    ]);
  } finally {
    globalThis.OffscreenCanvas = previousOffscreenCanvas;
    globalThis.ImageData = previousImageData;
  }
});

test('snapshot pixels are encoded for direct delivery to the native compositor', () => {
  assert.equal(bytesToBase64(new Uint8Array([1, 2, 3, 254, 255])), 'AQID/v8=');
});

test('pixel helpers flip framebuffer rows and reject a uniform clear frame', () => {
  const pixels = new Uint8ClampedArray([
    1, 1, 1, 255, 2, 2, 2, 255,
    10, 10, 10, 255, 20, 20, 20, 255,
  ]);

  assert.deepEqual(
    flipPixelRows(pixels, 2, 2),
    new Uint8ClampedArray([
      10, 10, 10, 255, 20, 20, 20, 255,
      1, 1, 1, 255, 2, 2, 2, 255,
    ]),
  );
  assert.equal(hasPixelVariation(new Uint8ClampedArray([9, 9, 11, 255, 9, 9, 11, 255])), false);
  assert.equal(hasPixelVariation(pixels), true);
});

test('cached scene requests render on demand until a valid frame can be captured', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.cacheSceneFrames = true;
  renderer.sceneFramePending = false;
  renderer.sceneFramesRemaining = 0;
  renderer.app = {
    autoRender: false,
    once: () => {},
    renderNextFrame: false,
  };

  renderer.requestSceneFrame();

  assert.equal(renderer.app.autoRender, true);
  assert.equal(renderer.app.renderNextFrame, true);
  assert.equal(renderer.sceneFramePending, true);
  assert.equal(renderer.sceneFramesRemaining, 2);
});

test('interactive scene save exports the currently visible framebuffer', () => {
  const published = [];
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.assetId = 'room-v1';
  renderer.snapshotRevision = 4;
  renderer.sceneEntity = {};
  renderer.captureSceneFrame = () => ({ hasContent: true });
  renderer.publishSceneSnapshot = (...args) => published.push(args);
  renderer.bridge = { report_scene_error: () => {} };

  renderer.publishCurrentSnapshot();

  assert.deepEqual(published, [['room-v1', 4, 'background']]);
});

test('inactive interactive renderer does not schedule hidden frames', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.rendererActive = false;
  renderer.cacheSceneFrames = false;
  renderer.app = { autoRender: false, renderNextFrame: false };

  renderer.requestSceneFrame();

  assert.equal(renderer.app.autoRender, false);
  assert.equal(renderer.app.renderNextFrame, false);
});

test('cached scene publishes after a bounded GSplat settling window', () => {
  const published = [];
  let postrender;
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.assetId = 'room-v1';
  renderer.cacheSceneFrames = true;
  renderer.pendingSnapshotKind = 'background';
  renderer.pendingSnapshotRevision = 7;
  renderer.sceneCaptureAttempts = 12;
  renderer.sceneEntity = {};
  renderer.sceneFramePending = false;
  renderer.sceneFramesRemaining = 1;
  renderer.sceneSettled = false;
  renderer.firstPersonNavigation = false;
  renderer.snapshotRequiresSettlement = false;
  renderer.captureSceneFrame = () => ({ hasContent: true });
  renderer.publishSceneSnapshot = (...args) => published.push(args);
  renderer.bridge = { report_scene_error: () => {} };
  renderer.app = {
    autoRender: true,
    once: (_event, callback) => { postrender = callback; },
    renderNextFrame: false,
  };

  renderer.renderSceneFrame();
  postrender();

  assert.deepEqual(published, [['room-v1', 7, 'background']]);
  assert.equal(renderer.pendingSnapshotKind, null);
  assert.equal(renderer.sceneFramePending, false);
  assert.equal(renderer.app.autoRender, false);
});

test('non-SHARP scene waits for GSplat settlement before publishing its snapshot', () => {
  const published = [];
  let postrender;
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.assetId = 'streamed-room';
  renderer.cacheSceneFrames = true;
  renderer.pendingSnapshotKind = 'background';
  renderer.pendingSnapshotRevision = 3;
  renderer.sceneCaptureAttempts = 12;
  renderer.sceneEntity = {};
  renderer.sceneFramePending = false;
  renderer.sceneFramesRemaining = 1;
  renderer.sceneSettled = false;
  renderer.firstPersonNavigation = false;
  renderer.snapshotRequiresSettlement = true;
  renderer.captureSceneFrame = () => ({ hasContent: true });
  renderer.publishSceneSnapshot = (...args) => published.push(args);
  renderer.bridge = { report_scene_error: () => {} };
  renderer.app = {
    autoRender: true,
    once: (_event, callback) => { postrender = callback; },
    renderNextFrame: false,
  };

  renderer.renderSceneFrame();
  postrender();

  assert.deepEqual(published, []);
  assert.equal(renderer.pendingSnapshotKind, 'background');
  assert.equal(renderer.sceneFramePending, false);
  assert.equal(renderer.app.autoRender, false);
});

test('cached viewpoint refresh captures a sharp reference before the DOF background', () => {
  const calls = [];
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.snapshotRevision = 9;
  renderer.snapshotQueue = [];
  renderer.pendingSnapshotKind = null;
  renderer.sceneFramePending = false;
  renderer.configureNormalFrame = (strength) => calls.push(['configure', strength]);
  renderer.requestSceneFrame = (frames, kind, revision) => {
    calls.push(['request', frames, kind, revision]);
    renderer.pendingSnapshotKind = kind;
  };

  renderer.queueSnapshotRefresh(true);
  renderer.pendingSnapshotKind = null;
  renderer.startNextSnapshot();

  assert.deepEqual(calls, [
    ['configure', 0],
    ['request', 2, 'harmonization', 9],
    ['configure', null],
    ['request', 2, 'background', 9],
  ]);
});

test('held flight keys move the camera and orbit target through the scene', () => {
  const cameraCalls = [];
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.viewpoint = {
    position: { x: 0, y: 1, z: 0 },
    orbit_target: { x: 0, y: 1, z: -1 },
    horizon: 0,
    safe_camera_region: {
      minimum: { x: -10, y: -10, z: -10 },
      maximum: { x: 10, y: 10, z: 10 },
    },
  };
  renderer.flightKeys = new Set(['w']);
  renderer.flightDirty = false;
  renderer.warning = { hidden: true };
  renderer.camera = {
    forward: { x: 0, y: 0, z: -1 },
    right: { x: 1, y: 0, z: 0 },
    setPosition: (...values) => cameraCalls.push(['position', ...values]),
    lookAt: (...values) => cameraCalls.push(['target', ...values]),
    rotateLocal: () => {},
  };
  renderer.requestSceneFrame = () => cameraCalls.push(['render']);

  renderer.updateFlight(0.1);

  assert.ok(renderer.viewpoint.position.z < 0);
  assert.equal(renderer.viewpoint.orbit_target.z, renderer.viewpoint.position.z - 1);
  assert.equal(renderer.flightDirty, true);
  assert.equal(cameraCalls.at(-1)[0], 'render');
});
