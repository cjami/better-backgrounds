import assert from 'node:assert/strict';
import test from 'node:test';

import {
  SceneRenderer,
  buildDepthProxyGeometry,
  buildViewDepthProxyGeometry,
  collectDepthProxyCenters,
  depthOfFieldForStrength,
  harmonizationViewpointKey,
} from '../src/main.mjs';

const viewpoint = {
  depth_of_field: { blur_strength: 0.5 },
  field_of_view: 42,
};

test('normal depth of field stays locked to subject depth', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.sceneEntity = {};
  renderer.depthProxyEntity = {};
  renderer.viewpoint = viewpoint;
  renderer.cameraFrame = {
    debug: null,
    dof: {},
    rendering: {},
    updateCalled: false,
    update() { this.updateCalled = true; },
  };

  renderer.configureNormalFrame();

  assert.equal(renderer.cameraFrame.dof.enabled, true);
  assert.equal(renderer.cameraFrame.dof.focusDistance, 1.5);
  assert.ok(Math.abs(renderer.cameraFrame.dof.focusRange - 1.7) < 1e-10);
  assert.equal(renderer.cameraFrame.dof.blurRadius, 4);
  assert.equal(renderer.cameraFrame.dof.nearBlur, true);
  assert.equal(renderer.cameraFrame.dof.highQuality, true);
  assert.equal(renderer.cameraFrame.dof.blurRings, 4);
  assert.equal(renderer.cameraFrame.dof.blurRingPoints, 5);
  assert.equal(renderer.cameraFrame.rendering.sharpness, 0.24);
  assert.equal(renderer.cameraFrame.updateCalled, true);
});

test('depth-of-field blur opens the aperture and narrows the focus band', () => {
  const gentle = depthOfFieldForStrength(0.1);
  const strong = depthOfFieldForStrength(0.9);

  assert.ok(gentle.focusRange > 6);
  assert.ok(strong.focusRange < 1);
  assert.equal(gentle.blurRadius, 0.8);
  assert.ok(strong.blurRadius > gentle.blurRadius);
});

test('one percent blur remains subpixel while retaining spatial depth', () => {
  const effect = depthOfFieldForStrength(0.01, 42);

  assert.ok(effect.focusRange > 7.7);
  assert.ok(effect.blurRadius < 0.1);
});

test('field of view adjusts the effective aperture', () => {
  const telephoto = depthOfFieldForStrength(0.5, 24);
  const reference = depthOfFieldForStrength(0.5, 42);
  const wide = depthOfFieldForStrength(0.5, 90);

  assert.ok(telephoto.blurRadius > reference.blurRadius);
  assert.ok(telephoto.focusRange < reference.focusRange);
  assert.ok(wide.blurRadius < reference.blurRadius);
  assert.ok(wide.focusRange > reference.focusRange);
});

test('circle of confusion grows gradually with scene distance', () => {
  const effect = depthOfFieldForStrength(0.25, 42);
  const farFocusEdge = 1.5 + effect.focusRange / 2;
  const circleOfConfusion = (depth) => Math.min(
    1,
    Math.max(0, (depth - farFocusEdge) / effect.focusRange),
  );

  assert.ok(circleOfConfusion(3.5) > 0);
  assert.ok(circleOfConfusion(3.5) < circleOfConfusion(5));
  assert.ok(circleOfConfusion(5) < circleOfConfusion(8));
});

test('depth-of-field changes retain the same sharp harmonization reference', () => {
  const first = { field_of_view: 42, depth_of_field: { blur_strength: 0.1 } };
  const second = { field_of_view: 42, depth_of_field: { blur_strength: 0.9 } };

  assert.equal(harmonizationViewpointKey(first), harmonizationViewpointKey(second));
  assert.notEqual(
    harmonizationViewpointKey(first),
    harmonizationViewpointKey({ ...second, field_of_view: 55 }),
  );
});

test('SHARP raster Gaussians produce a bounded depth-only proxy', () => {
  const properties = {
    x: new Float32Array([0, 2, 0, 2]),
    y: new Float32Array([0, 0, 2, 2]),
    z: new Float32Array([3, 3, 3, 3]),
    image_size: new Uint32Array([2, 2]),
    intrinsic: new Float32Array([1.5, 0, 0.5, 0, 1.5, 0.5, 0, 0, 1]),
  };
  const gsplatData = {
    getProp(name, element = 'vertex') {
      if (name === 'image_size' && element !== 'image_size') return undefined;
      return properties[name];
    },
  };

  const geometry = buildDepthProxyGeometry(gsplatData, 2);

  assert.equal(geometry.columns, 2);
  assert.equal(geometry.rows, 2);
  assert.deepEqual(Array.from(geometry.positions), [
    0, 0, 3,
    2, 0, 3,
    0, 2, 3,
    2, 2, 3,
  ]);
  assert.deepEqual(Array.from(geometry.indices), [0, 2, 1, 1, 2, 3]);
});

test('generic splat centers produce a view-dependent depth proxy', () => {
  const centers = new Float32Array([
    -1, -1, 3,
    1, -1, 3,
    -1, 1, 3,
    1, 1, 3,
  ]);
  const geometry = buildViewDepthProxyGeometry(
    centers,
    (x, y, depth) => ({ x: (x + 2) / 2, y: (y + 2) / 2, depth }),
    (x, y, depth) => ({ x: x - 0.5, y: y - 0.5, z: depth }),
    2,
    2,
    2,
  );

  assert.equal(geometry.columns, 2);
  assert.equal(geometry.rows, 2);
  assert.deepEqual(Array.from(geometry.indices), [0, 2, 1, 1, 2, 3]);
});

test('streamed SOG depth sampling combines currently resident LOD chunks', () => {
  const first = { centers: new Float32Array([0, 0, 1, 1, 0, 1]) };
  const second = { centers: new Float32Array([0, 1, 2, 1, 1, 2]) };
  const environment = { centers: new Float32Array([0, 2, 4]) };
  const resource = {
    octree: {
      fileResources: new Map([[0, first], [1, second]]),
      environmentResource: environment,
    },
  };

  const centers = collectDepthProxyCenters(resource, 5);

  assert.deepEqual(Array.from(centers), [
    0, 0, 1,
    1, 0, 1,
    0, 1, 2,
    1, 1, 2,
    0, 2, 4,
  ]);
});

test('generic depth proxies do not bridge large depth discontinuities', () => {
  const centers = new Float32Array([
    -1, -1, 1,
    1, -1, 1,
    -1, 1, 1,
    1, 1, 10,
  ]);
  const geometry = buildViewDepthProxyGeometry(
    centers,
    (x, y, depth) => ({ x: (x + 2) / 2, y: (y + 2) / 2, depth }),
    (x, y, depth) => ({ x, y, z: depth }),
    2,
    2,
    2,
  );

  assert.deepEqual(Array.from(geometry.indices), [0, 2, 1]);
});

test('zero background blur keeps a stable depth pipeline with a zero radius', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.sceneEntity = {};
  renderer.depthProxyEntity = {};
  renderer.viewpoint = { depth_of_field: { blur_strength: 0 } };
  renderer.cameraFrame = { debug: null, dof: {}, rendering: {}, update() {} };

  renderer.configureNormalFrame();

  assert.equal(renderer.cameraFrame.dof.enabled, true);
  assert.equal(renderer.cameraFrame.dof.blurRadius, 0);
  assert.equal(renderer.cameraFrame.rendering.sharpness, 0.24);
});

test('interactive renderer remains live when requesting a scene frame', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.cacheSceneFrames = false;
  renderer.app = { autoRender: false, renderNextFrame: false };

  renderer.requestSceneFrame();

  assert.equal(renderer.app.autoRender, true);
  assert.equal(renderer.app.renderNextFrame, true);
});

test('interactive depth-proxy refreshes are debounced while cached snapshots rebuild now', async () => {
  const interactive = Object.create(SceneRenderer.prototype);
  interactive.cacheSceneFrames = false;
  interactive.depthProxyRefreshTimer = null;
  const interactiveCalls = [];
  interactive.rebuildViewDepthProxy = (force) => interactiveCalls.push(force);

  interactive.refreshViewDepthProxy(false);
  interactive.refreshViewDepthProxy(true);
  await new Promise((resolve) => setTimeout(resolve, 140));

  assert.deepEqual(interactiveCalls, [true]);

  const cached = Object.create(SceneRenderer.prototype);
  cached.cacheSceneFrames = true;
  const cachedCalls = [];
  cached.rebuildViewDepthProxy = (force) => cachedCalls.push(force);
  cached.refreshViewDepthProxy(true);
  assert.deepEqual(cachedCalls, [true]);
});
