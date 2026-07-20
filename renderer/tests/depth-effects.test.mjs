import assert from 'node:assert/strict';
import test from 'node:test';

import * as pc from 'playcanvas';

import {
  SceneRenderer,
  configureGsplatStreaming,
  depthOfFieldForStrength,
  focusDistanceAtCamera,
  harmonizationViewpointKey,
  isStreamedSceneUrl,
} from '../src/main.mjs';
import {
  SPLAT_PREPASS_ALPHA_THRESHOLD,
  configureGpuSplatDepth,
  installSplatPrepass,
  renderSplatPrepass,
  repairSplatDepth,
  smoothCircleOfConfusion,
} from '../src/gpu-depth.mjs';

test('streamed scenes use the responsive LOD settings and GPU depth threshold', () => {
  const scene = { gsplat: {} };

  configureGsplatStreaming(scene);

  assert.equal(scene.gsplat.lodUpdateAngle, 2);
  assert.equal(scene.gsplat.lodUpdateDistance, 0.1);
  assert.equal(scene.gsplat.alphaClip, 0.1);
  assert.equal(isStreamedSceneUrl('scene://room/lod-meta.json'), true);
  assert.equal(isStreamedSceneUrl('scene://room/scene.ply'), false);
});

test('GPU splat depth uses the engine prepass alpha threshold', () => {
  const scene = { gsplat: { alphaClip: 0 } };

  configureGpuSplatDepth(scene);

  assert.equal(scene.gsplat.alphaClip, SPLAT_PREPASS_ALPHA_THRESHOLD);
});

test('the GPU prepass enables opaque depth state only for its draw', () => {
  const originalBlendState = { name: 'premultiplied', clone() { return { ...this }; } };
  const originalDepthState = { name: 'read-only', clone() { return { ...this }; } };
  const material = {
    blendState: originalBlendState,
    depthState: originalDepthState,
    depthWrite: false,
  };
  let rendered = false;

  renderSplatPrepass([material], () => {
    rendered = true;
    assert.equal(material.blendState, pc.BlendState.NOBLEND);
    assert.equal(material.depthWrite, true);
  });

  assert.equal(rendered, true);
  assert.equal(material.blendState.name, 'premultiplied');
  assert.equal(material.depthState.name, 'read-only');
});

test('the GPU prepass restores material state after a failed draw', () => {
  const material = {
    blendState: { name: 'premultiplied', clone() { return { ...this }; } },
    depthState: { name: 'read-only', clone() { return { ...this }; } },
    depthWrite: false,
  };

  assert.throws(
    () => renderSplatPrepass([material], () => { throw new Error('draw failed'); }),
    /draw failed/,
  );
  assert.equal(material.blendState.name, 'premultiplied');
  assert.equal(material.depthState.name, 'read-only');
});

test('the native prepass draws the current unified splat manager', () => {
  let draws = 0;
  const splatMeshInstance = { material: null };
  const material = {
    blendState: { clone() { return { ...this }; } },
    depthState: { clone() { return { ...this }; } },
    depthWrite: false,
  };
  splatMeshInstance.material = material;
  const camera = {};
  const layer = {
    id: 0,
    enabled: true,
    camerasSet: new Set([camera]),
    getCulledInstances: () => ({ opaque: [], transparent: [] }),
  };
  const depthLayer = { id: pc.LAYERID_DEPTH };
  const manager = { material, renderer: { meshInstance: splatMeshInstance } };
  const prePass = {
    camera: { camera },
    renderTarget: {},
    scene: {
      layers: {
        layerList: [layer, depthLayer, layer],
        subLayerEnabled: [true, true, true],
        subLayerList: [false, false, true],
      },
    },
    renderer: {
      renderForwardLayer(...args) {
        draws += 1;
        assert.deepEqual(args.at(-1).meshInstances, [splatMeshInstance]);
        assert.equal(material.depthWrite, true);
      },
    },
    viewBindGroups: [],
  };
  const cameraFrame = {
    app: {
      renderer: {
        gsplatDirector: {
          camerasMap: new Map([[camera, {
            layersMap: new Map([[layer, { gsplatManager: manager }]]),
          }]]),
        },
      },
    },
    cameraComponent: { camera },
    renderPassCamera: { prePass },
  };

  assert.equal(installSplatPrepass(cameraFrame), true);
  prePass.execute();

  assert.equal(draws, 1);
  assert.equal(material.depthWrite, false);
  assert.equal(prePass.betterBackgroundsGpuDepth, true);
});

test('depth edge repair fills only well-supported continuous splat holes', () => {
  const continuous = [3, 3.02, 3.01, 3, 3.03, 100, 100, 100, 100];
  const discontinuous = [2, 2, 2, 8, 8, 8, 100, 100, 100];

  assert.ok(Math.abs(repairSplatDepth(100, continuous, 100) - 3.012) < 1e-6);
  assert.equal(repairSplatDepth(100, discontinuous, 100), 100);
  assert.equal(repairSplatDepth(3, continuous, 100), 3);
  assert.equal(repairSplatDepth(100, [3, 3, 3, 3], 100), 100);
});

test('circle of confusion eases continuously away from the focus band', () => {
  const values = [1.9, 2.1, 2.3, 2.5, 2.7]
    .map((depth) => smoothCircleOfConfusion(depth, 1.5, 0.8));

  assert.equal(values[0], 0);
  assert.ok(values[0] < values[1]);
  assert.ok(values[1] < values[2]);
  assert.ok(values[2] < values[3]);
  assert.equal(values[4], 1);
});

test('depth-of-field strength retains metric focus and reaches full aperture', () => {
  const gentle = depthOfFieldForStrength(0.1);
  const strong = depthOfFieldForStrength(0.9);
  const maximum = depthOfFieldForStrength(1, 90);

  assert.ok(gentle.focusRange > 6);
  assert.ok(strong.focusRange < 1);
  assert.ok(strong.blurRadius > gentle.blurRadius);
  assert.equal(maximum.focusRange, 0.8);
  assert.equal(maximum.blurRadius, 8);
});

test('depth of field focuses immediately in front of the camera', () => {
  assert.equal(focusDistanceAtCamera(0.03), 0.03);
  assert.equal(focusDistanceAtCamera(0.05), 0.05);
  assert.equal(focusDistanceAtCamera(Number.NaN), 0.001);
});

test('zero blur and held navigation both use direct rendering', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.sceneEntity = {};
  renderer.navigationInputs = new Set();
  renderer.viewpoint = { depth_of_field: { blur_strength: 0 }, field_of_view: 42 };
  renderer.cameraFrame = { enabled: true };

  renderer.configureNormalFrame();
  assert.equal(renderer.cameraFrame.enabled, false);

  renderer.viewpoint.depth_of_field.blur_strength = 0.5;
  renderer.navigationInputs.add('pointer:1');
  renderer.cameraFrame.enabled = true;
  renderer.configureNormalFrame();
  assert.equal(renderer.cameraFrame.enabled, false);
});

test('a held pointer keeps DOF disabled through move-stop-rotate', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.sceneEntity = {};
  renderer.navigationInputs = new Set();
  renderer.viewpoint = { depth_of_field: { blur_strength: 0.5 }, field_of_view: 42 };
  renderer.cameraFrame = { enabled: false };
  renderer.requestSceneFrame = () => {};

  renderer.beginNavigation('pointer:7');
  renderer.beginNavigation('pointer:7');

  assert.deepEqual([...renderer.navigationInputs], ['pointer:7']);
  assert.equal(renderer.cameraFrame.enabled, false);
});

test('depth-of-field changes retain the same harmonization reference', () => {
  const first = { field_of_view: 42, depth_of_field: { blur_strength: 0.1 } };
  const second = { field_of_view: 42, depth_of_field: { blur_strength: 0.9 } };

  assert.equal(harmonizationViewpointKey(first), harmonizationViewpointKey(second));
  assert.notEqual(
    harmonizationViewpointKey(first),
    harmonizationViewpointKey({ ...second, field_of_view: 55 }),
  );
});

test('interactive renderer remains live when requesting a scene frame', () => {
  const renderer = Object.create(SceneRenderer.prototype);
  renderer.cacheSceneFrames = false;
  renderer.rendererActive = true;
  renderer.app = { autoRender: false, renderNextFrame: false };

  renderer.requestSceneFrame();

  assert.equal(renderer.app.autoRender, true);
  assert.equal(renderer.app.renderNextFrame, true);
});
