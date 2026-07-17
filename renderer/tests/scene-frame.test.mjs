import assert from 'node:assert/strict';
import test from 'node:test';

import { SceneRenderer, flipPixelRows, hasPixelVariation } from '../src/main.mjs';

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

    transferToImageBitmap() {
      operations.push(['transfer']);
      return { width: this.width, height: this.height };
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
      frame: { width: 2, height: 2 },
      hasContent: true,
    });
    assert.deepEqual(operations, [
      ['read'],
      ['copy', [20, 30, 40, 255, 50, 60, 70, 255, 1, 2, 3, 255, 4, 5, 6, 255]],
      ['transfer'],
    ]);
  } finally {
    globalThis.OffscreenCanvas = previousOffscreenCanvas;
    globalThis.ImageData = previousImageData;
  }
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

test('cached scene requests resume rendering until a valid frame can be captured', () => {
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
