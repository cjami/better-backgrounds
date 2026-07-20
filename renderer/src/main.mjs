import * as pc from 'playcanvas';

import {
  clampPosition,
  flySpeed,
  lookTarget,
  orbitPosition,
  positionsEqual,
  translateViewpoint,
  zoomPosition,
} from './viewpoint.mjs';
import {
  configureGpuDepthPipeline,
  configureGpuSplatDepth,
} from './gpu-depth.mjs';

const MAX_SCENE_CAPTURE_ATTEMPTS = 12;
const STREAMED_LOD_UPDATE_ANGLE = 2;
const STREAMED_LOD_UPDATE_DISTANCE = 0.1;
const MIN_DOF_FOCUS_DISTANCE = 0.001;
const MIN_SUBJECT_FOCUS_RANGE = 0.8;
const MAX_SUBJECT_FOCUS_RANGE = 8;
const MAX_DOF_BLUR_RADIUS = 8;
const REFERENCE_FIELD_OF_VIEW = 42;
const MIN_PERSPECTIVE_SCALE = 0.75;
const MAX_PERSPECTIVE_SCALE = 1.25;
const CAMERA_FRAME_GSPLAT_OUTPUT_GLSL = `
vec3 prepareOutputFromGamma(vec3 gammaColor, float depth) {
  return gammaColor;
}
`;
const CAMERA_FRAME_GSPLAT_OUTPUT_WGSL = `
fn prepareOutputFromGamma(gammaColor: vec3f, depth: f32) -> vec3f {
  return gammaColor;
}
`;
const focusDistanceAtCamera = (nearClip) => Math.max(
  MIN_DOF_FOCUS_DISTANCE,
  Number.isFinite(nearClip) ? nearClip : MIN_DOF_FOCUS_DISTANCE,
);

const depthOfFieldForStrength = (strength, fieldOfView = REFERENCE_FIELD_OF_VIEW) => {
  const blurStrength = Math.min(1, Math.max(0, strength));
  const safeFieldOfView = Math.min(90, Math.max(24, fieldOfView));
  const perspectiveScale = Math.min(
    MAX_PERSPECTIVE_SCALE,
    Math.max(
      MIN_PERSPECTIVE_SCALE,
      Math.tan(REFERENCE_FIELD_OF_VIEW * Math.PI / 360)
        / Math.tan(safeFieldOfView * Math.PI / 360),
    ),
  );
  const scaledAperture = blurStrength * perspectiveScale;
  const apertureDenominator = 1 - blurStrength + scaledAperture;
  const aperture = apertureDenominator > 0 ? scaledAperture / apertureDenominator : 0;
  const focusFalloff = (1 - aperture) ** 3;
  return {
    blurRadius: aperture * MAX_DOF_BLUR_RADIUS,
    focusRange: MIN_SUBJECT_FOCUS_RANGE
      + (MAX_SUBJECT_FOCUS_RANGE - MIN_SUBJECT_FOCUS_RANGE) * focusFalloff,
  };
};

const harmonizationViewpointKey = (viewpoint) => {
  const reference = structuredClone(viewpoint);
  delete reference.depth_of_field;
  return JSON.stringify(reference);
};

const bytesToBase64 = (bytes) => {
  let binary = '';
  const chunkSize = 32_768;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
};

const flipPixelRows = (pixels, width, height) => {
  const rowLength = width * 4;
  const temporary = new Uint8ClampedArray(rowLength);
  for (let top = 0; top < Math.floor(height / 2); top += 1) {
    const bottom = height - top - 1;
    const topOffset = top * rowLength;
    const bottomOffset = bottom * rowLength;
    temporary.set(pixels.subarray(topOffset, topOffset + rowLength));
    pixels.copyWithin(topOffset, bottomOffset, bottomOffset + rowLength);
    pixels.set(temporary, bottomOffset);
  }
  return pixels;
};

const hasPixelVariation = (pixels) => {
  let minimum = 765;
  let maximum = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    const brightness = pixels[index] + pixels[index + 1] + pixels[index + 2];
    minimum = Math.min(minimum, brightness);
    maximum = Math.max(maximum, brightness);
    if (maximum - minimum > 12) return true;
  }
  return false;
};

const configureGsplatStreaming = (scene) => {
  scene.gsplat.lodUpdateAngle = STREAMED_LOD_UPDATE_ANGLE;
  scene.gsplat.lodUpdateDistance = STREAMED_LOD_UPDATE_DISTANCE;
  configureGpuSplatDepth(scene);
};

const isStreamedSceneUrl = (url) => /(?:^|\/)lod-meta\.json(?:[?#]|$)/i.test(url);

class SceneRenderer {
  constructor(bridge, { cacheSceneFrames = false } = {}) {
    this.bridge = bridge;
    this.cacheSceneFrames = cacheSceneFrames;
    this.rendererActive = true;
    this.app = null;
    this.camera = null;
    this.cameraFrame = null;
    this.cameraFrameColorPipeline = null;
    this.sceneEntity = null;
    this.sceneAsset = null;
    this.sceneTransformKey = '';
    this.assetId = '';
    this.viewpoint = null;
    this.resetViewpoint = null;
    this.drag = null;
    this.flightKeys = new Set();
    this.flightDirty = false;
    this.navigationInputs = new Set();
    this.wheelIdleTimer = null;
    this.firstPersonNavigation = false;
    this.sceneFramePending = false;
    this.sceneFramesRemaining = 0;
    this.sceneSnapshot = null;
    this.sceneSnapshotContext = null;
    this.scenePixels = null;
    this.sceneCaptureAttempts = 0;
    this.sceneSettled = false;
    this.snapshotRequiresSettlement = false;
    this.snapshotRevision = 0;
    this.pendingSnapshotKind = null;
    this.pendingSnapshotRevision = 0;
    this.snapshotQueue = [];
    this.harmonizationViewpointKey = '';
    this.warning = document.getElementById('safe-warning');
    this.status = document.getElementById('status');
    this.subjectGuide = document.getElementById('subject-guide');
  }

  async initialize() {
    const canvas = document.getElementById('scene-canvas');
    try {
      const device = await pc.createGraphicsDevice(canvas, {
        deviceTypes: [pc.DEVICETYPE_WEBGL2],
        antialias: false,
        preserveDrawingBuffer: true,
        powerPreference: 'high-performance',
      });
      const options = new pc.AppOptions();
      options.graphicsDevice = device;
      options.componentSystems = [
        pc.RenderComponentSystem,
        pc.CameraComponentSystem,
        pc.GSplatComponentSystem,
      ];
      options.resourceHandlers = [pc.TextureHandler, pc.GSplatHandler];
      this.app = new pc.AppBase(canvas);
      this.app.init(options);
      this.app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
      this.app.setCanvasResolution(pc.RESOLUTION_AUTO);
      this.app.scene.toneMapping = pc.TONEMAP_NONE;
      this.app.scene.gammaCorrection = pc.GAMMA_SRGB;
      configureGsplatStreaming(this.app.scene);

      this.camera = new pc.Entity('virtual-camera');
      this.camera.addComponent('camera', {
        clearColor: new pc.Color(0.035, 0.037, 0.043),
      });
      this.app.root.addChild(this.camera);
      this.cameraFrame = new pc.CameraFrame(this.app, this.camera.camera);
      this.cameraFrame.enabled = false;
      this.app.start();
      this.app.on('update', (deltaTime) => this.updateFlight(deltaTime));
      this.app.autoRender = !this.cacheSceneFrames;
      this.requestSceneFrame();
      if (this.status) {
        this.status.textContent = device.deviceType === pc.DEVICETYPE_WEBGPU ? 'WEBGPU' : 'WEBGL 2';
      }
      this.bindInput(canvas);
      window.addEventListener('resize', () => {
        this.app?.resizeCanvas();
        this.requestSceneFrame();
      });
      this.connectBridge();
      this.app.systems.gsplat.on('frame:ready', (_camera, _layer, ready, loadingCount) => {
        this.sceneSettled = ready && loadingCount === 0;
        if (this.sceneSettled) {
          if (this.pendingSnapshotKind) this.requestSceneFrame(1);
        }
      });
      this.app.systems.gsplat.on('frame:request', () => this.requestSceneFrame(1));
    } catch (error) {
      if (this.status) this.status.textContent = 'GPU UNAVAILABLE';
      this.bridge.report_scene_error('renderer', 'gpu_unavailable', this.safeMessage(error));
    }
  }

  connectBridge() {
    this.bridge.output_size_requested.connect((width, height) => {
      this.setOutputSize(width, height);
    });
    this.bridge.snapshot_requested.connect(() => {
      this.publishCurrentSnapshot();
    });
    this.bridge.renderer_active_requested.connect((active) => {
      this.setRendererActive(active);
    });
    this.bridge.scene_requested.connect((assetId, url, payload) => {
      this.loadScene(assetId, url, payload);
    });
    this.bridge.viewpoint_requested.connect((payload) => this.applyViewpoint(payload));
    this.bridge.reset_requested.connect(() => {
      if (this.resetViewpoint) this.applyViewpoint(JSON.stringify(this.resetViewpoint));
    });
    this.bridge.scene_cleared.connect(() => {
      this.removeScene();
      this.assetId = '';
      this.setLoading(false, 'No spatial scene selected');
      this.requestSceneFrame();
    });
  }

  setOutputSize(width, height) {
    if (!this.cacheSceneFrames || !this.app) return;
    if (!Number.isInteger(width) || !Number.isInteger(height)) return;
    if (width < 1 || height < 1 || width > 8192 || height > 8192) return;
    const canvas = this.app.graphicsDevice.canvas;
    if (canvas.width === width && canvas.height === height) return;
    this.app.setCanvasResolution(pc.RESOLUTION_FIXED, width, height);
    this.snapshotRevision += 1;
    if (this.sceneEntity) this.queueSnapshotRefresh(true);
    else this.requestSceneFrame();
  }

  setRendererActive(active) {
    this.rendererActive = Boolean(active);
    if (!this.app) return;
    if (!this.rendererActive) {
      this.flightKeys.clear();
      this.flightDirty = false;
      this.app.autoRender = false;
      return;
    }
    this.requestSceneFrame();
  }

  loadScene(assetId, url, payload) {
    if (!this.app || !this.camera) return;
    this.assetId = assetId;
    this.removeScene();
    this.sceneSettled = false;
    this.snapshotRequiresSettlement = true;
    this.firstPersonNavigation = isStreamedSceneUrl(url);
    this.applyViewpoint(payload, true);
    this.setLoading(true, 'Loading spatial scene…');
    this.bridge.report_scene_progress(assetId, 0, 100);

    const asset = new pc.Asset(assetId, 'gsplat', { url }, { reorder: false });
    this.sceneAsset = asset;
    this.app.assets.add(asset);
    asset.ready((loadedAsset) => {
      if (asset !== this.sceneAsset) return;
      const entity = new pc.Entity(assetId);
      entity.addComponent('gsplat', { asset: loadedAsset, unified: true });
      this.sceneEntity = entity;
      this.sceneCaptureAttempts = 0;
      this.app.root.addChild(entity);
      this.applySceneTransform();
      this.configureNormalFrame();
      if (this.cacheSceneFrames) this.queueSnapshotRefresh(true);
      else this.requestSceneFrame();
      this.bridge.report_scene_progress(assetId, 100, 100);
      this.setLoading(false, 'Scene ready');
    });
    asset.on('error', (error) => {
      if (asset !== this.sceneAsset) return;
      this.setLoading(false, 'Scene failed to load');
      this.bridge.report_scene_error(assetId, 'scene_load_failed', this.safeMessage(error));
    });
    this.app.assets.load(asset);
  }

  removeScene() {
    this.navigationInputs.clear();
    if (this.wheelIdleTimer) clearTimeout(this.wheelIdleTimer);
    this.wheelIdleTimer = null;
    this.disableCameraFrame();
    this.sceneEntity?.destroy();
    this.sceneEntity = null;
    if (this.sceneAsset && this.app) {
      this.app.assets.remove(this.sceneAsset);
      this.sceneAsset.unload();
    }
    this.sceneAsset = null;
    this.sceneTransformKey = '';
    this.sceneSettled = false;
    this.firstPersonNavigation = false;
    this.snapshotRequiresSettlement = false;
    this.snapshotQueue.length = 0;
    this.harmonizationViewpointKey = '';
  }

  activeGsplatManagers() {
    const director = this.app?.renderer?.gsplatDirector;
    const camera = this.camera?.camera?.camera;
    const cameraData = director?.camerasMap?.get(camera);
    if (!cameraData) return [];
    return [...cameraData.layersMap.values()]
      .map((layerData) => layerData.gsplatManager)
      .filter(Boolean);
  }

  invalidateGsplatMaterials() {
    this.app?.scene?.gsplat?.material?.update?.();
    for (const manager of this.activeGsplatManagers()) {
      manager.renderer.forceCopyMaterial = true;
      manager.material?.update?.();
    }
  }

  installCameraFrameColorPipeline() {
    if (this.cameraFrameColorPipeline || !this.app) return;
    const device = this.app.graphicsDevice;
    const glslChunks = pc.ShaderChunks.get(device, pc.SHADERLANGUAGE_GLSL);
    const wgslChunks = pc.ShaderChunks.get(device, pc.SHADERLANGUAGE_WGSL);
    const originalGlsl = glslChunks.get('gsplatOutputVS');
    const originalWgsl = wgslChunks.get('gsplatOutputVS');
    glslChunks.set('gsplatOutputVS', CAMERA_FRAME_GSPLAT_OUTPUT_GLSL);
    wgslChunks.set('gsplatOutputVS', CAMERA_FRAME_GSPLAT_OUTPUT_WGSL);

    const backBuffer = device.backBuffer;
    const originalIsColorBufferSrgb = backBuffer.isColorBufferSrgb;
    const patchedIsColorBufferSrgb = () => true;
    backBuffer.isColorBufferSrgb = patchedIsColorBufferSrgb;
    this.cameraFrameColorPipeline = {
      backBuffer,
      glslChunks,
      wgslChunks,
      originalGlsl,
      originalWgsl,
      originalIsColorBufferSrgb,
      patchedIsColorBufferSrgb,
    };
    this.invalidateGsplatMaterials();
  }

  restoreCameraFrameColorPipeline() {
    const pipeline = this.cameraFrameColorPipeline;
    if (!pipeline) return;
    pipeline.glslChunks.set('gsplatOutputVS', pipeline.originalGlsl);
    pipeline.wgslChunks.set('gsplatOutputVS', pipeline.originalWgsl);
    if (pipeline.backBuffer.isColorBufferSrgb === pipeline.patchedIsColorBufferSrgb) {
      pipeline.backBuffer.isColorBufferSrgb = pipeline.originalIsColorBufferSrgb;
    }
    this.cameraFrameColorPipeline = null;
    this.invalidateGsplatMaterials();
  }

  disableCameraFrame() {
    if (!this.cameraFrame) return;
    this.cameraFrame.enabled = false;
    this.restoreCameraFrameColorPipeline();
  }

  applyViewpoint(payload, rememberAsReset = false) {
    if (!this.camera) return;
    try {
      const next = JSON.parse(payload);
      if (rememberAsReset) this.resetViewpoint = structuredClone(next);
      if (rememberAsReset || !this.viewpoint) this.viewpoint = structuredClone(next);
      else this.viewpoint = { ...this.viewpoint, ...next };
      const current = this.viewpoint;
      const safePosition = clampPosition(current.position, current.safe_camera_region);
      current.position = safePosition;
      this.camera.setPosition(safePosition.x, safePosition.y, safePosition.z);
      const target = current.orbit_target;
      this.camera.lookAt(target.x, target.y, target.z);
      this.camera.rotateLocal(0, 0, current.horizon);
      this.camera.camera.fov = current.field_of_view;
      this.camera.camera.nearClip = current.near_clip;
      this.camera.camera.farClip = current.far_clip;
      this.camera.camera.aspectRatioMode = pc.ASPECT_MANUAL;
      this.camera.camera.aspectRatio = current.aspect_ratio;
      const crop = current.crop;
      this.camera.camera.rect = new pc.Vec4(
        crop.left,
        1 - crop.bottom,
        crop.right - crop.left,
        crop.bottom - crop.top,
      );
      this.applySceneTransform();
      this.updateSubjectGuide();
      this.snapshotRevision += 1;
      const nextReferenceKey = harmonizationViewpointKey(current);
      const referenceChanged = nextReferenceKey !== this.harmonizationViewpointKey;
      this.harmonizationViewpointKey = nextReferenceKey;
      if (this.cacheSceneFrames && this.sceneEntity) {
        this.queueSnapshotRefresh(referenceChanged);
      } else {
        this.configureNormalFrame();
        this.requestSceneFrame();
      }
    } catch (error) {
      this.bridge.report_scene_error(this.assetId || 'renderer', 'viewpoint_invalid', this.safeMessage(error));
    }
  }

  beginNavigation(input) {
    if (this.navigationInputs.has(input)) return;
    this.navigationInputs.add(input);
    this.configureNormalFrame();
    this.requestSceneFrame();
  }

  endNavigation(input) {
    if (!this.navigationInputs.delete(input) || this.navigationInputs.size) return;
    this.configureNormalFrame();
    this.requestSceneFrame(2);
  }

  bindInput(canvas) {
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
      canvas.focus();
      canvas.setPointerCapture(event.pointerId);
      this.drag = { x: event.clientX, y: event.clientY, pan: event.button === 2 || event.shiftKey };
      this.beginNavigation(`pointer:${event.pointerId}`);
    });
    canvas.addEventListener('pointerup', (event) => {
      this.drag = null;
      this.endNavigation(`pointer:${event.pointerId}`);
      this.publishViewpoint();
    });
    canvas.addEventListener('pointercancel', (event) => {
      this.drag = null;
      this.endNavigation(`pointer:${event.pointerId}`);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.drag || !this.viewpoint) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      if (this.drag.pan) this.pan(dx, dy);
      else if (this.firstPersonNavigation) this.look(dx, dy);
      else this.orbit(dx, dy);
    });
    canvas.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        if (!this.viewpoint) return;
        this.beginNavigation('wheel');
        if (this.wheelIdleTimer) clearTimeout(this.wheelIdleTimer);
        this.wheelIdleTimer = setTimeout(() => {
          this.wheelIdleTimer = null;
          this.endNavigation('wheel');
        }, 120);
        if (this.firstPersonNavigation) this.dolly(event.deltaY);
        else {
          const position = zoomPosition(
            this.viewpoint.position,
            this.viewpoint.orbit_target,
            Math.exp(event.deltaY * 0.001),
          );
          this.setInteractivePosition(position);
        }
        this.publishViewpoint();
      },
      { passive: false },
    );
    canvas.addEventListener('keydown', (event) => {
      if (!this.viewpoint) return;
      const key = event.key.toLowerCase();
      if ('wasdqe'.includes(key) || key === 'shift') {
        event.preventDefault();
        this.flightKeys.add(key);
        this.beginNavigation('flight');
        return;
      }
      const orbitKeys = { ArrowLeft: [-12, 0], ArrowRight: [12, 0], ArrowUp: [0, -12], ArrowDown: [0, 12] };
      if (orbitKeys[event.key]) {
        event.preventDefault();
        if (this.firstPersonNavigation) this.look(...orbitKeys[event.key]);
        else this.orbit(...orbitKeys[event.key]);
        this.publishViewpoint();
      }
      if (event.key === '0') this.bridge.reset_requested();
    });
    canvas.addEventListener('keyup', (event) => {
      const key = event.key.toLowerCase();
      if (!this.flightKeys.delete(key)) return;
      event.preventDefault();
      this.finishFlight();
      if (!this.flightKeys.size) this.endNavigation('flight');
    });
    const stopFlight = () => {
      this.flightKeys.clear();
      this.finishFlight();
      this.endNavigation('flight');
      for (const input of [...this.navigationInputs]) {
        if (input.startsWith('pointer:')) this.endNavigation(input);
      }
    };
    canvas.addEventListener('blur', stopFlight);
    window.addEventListener('blur', stopFlight);
  }

  updateFlight(deltaTime) {
    if (!this.viewpoint || !this.camera || !this.flightKeys.size) return;
    const forwardAxis = Number(this.flightKeys.has('w')) - Number(this.flightKeys.has('s'));
    const rightAxis = Number(this.flightKeys.has('d')) - Number(this.flightKeys.has('a'));
    const verticalAxis = Number(this.flightKeys.has('e')) - Number(this.flightKeys.has('q'));
    if (!forwardAxis && !rightAxis && !verticalAxis) return;
    const forward = this.camera.forward;
    const right = this.camera.right;
    const direction = {
      x: forward.x * forwardAxis + right.x * rightAxis,
      y: forward.y * forwardAxis + right.y * rightAxis + verticalAxis,
      z: forward.z * forwardAxis + right.z * rightAxis,
    };
    const length = Math.hypot(direction.x, direction.y, direction.z);
    if (length <= 1e-6) return;
    const distance = flySpeed(
      this.viewpoint.safe_camera_region,
      this.flightKeys.has('shift'),
    ) * Math.min(Math.max(deltaTime, 0), 0.1);
    const moved = translateViewpoint(
      this.viewpoint.position,
      this.viewpoint.orbit_target,
      {
        x: direction.x / length * distance,
        y: direction.y / length * distance,
        z: direction.z / length * distance,
      },
      this.viewpoint.safe_camera_region,
    );
    this.viewpoint.position = moved.position;
    this.viewpoint.orbit_target = moved.target;
    this.warning.hidden = !moved.clamped;
    this.camera.setPosition(moved.position.x, moved.position.y, moved.position.z);
    this.camera.lookAt(moved.target.x, moved.target.y, moved.target.z);
    this.camera.rotateLocal(0, 0, this.viewpoint.horizon);
    this.flightDirty = true;
    this.requestSceneFrame();
  }

  finishFlight() {
    if (!this.flightDirty) return;
    this.flightDirty = false;
    this.publishViewpoint();
  }

  orbit(dx, dy) {
    const position = orbitPosition(
      this.viewpoint.position,
      this.viewpoint.orbit_target,
      -dx * 0.006,
      -dy * 0.006,
    );
    this.setInteractivePosition(position);
  }

  look(dx, dy) {
    const target = lookTarget(
      this.viewpoint.position,
      this.viewpoint.orbit_target,
      -dx * 0.006,
      -dy * 0.006,
    );
    this.viewpoint.orbit_target = target;
    this.camera.lookAt(target.x, target.y, target.z);
    this.camera.rotateLocal(0, 0, this.viewpoint.horizon);
    this.requestSceneFrame();
  }

  dolly(deltaY) {
    const position = this.viewpoint.position;
    const target = this.viewpoint.orbit_target;
    const direction = {
      x: target.x - position.x,
      y: target.y - position.y,
      z: target.z - position.z,
    };
    const length = Math.max(0.05, Math.hypot(direction.x, direction.y, direction.z));
    const distance = -deltaY * flySpeed(this.viewpoint.safe_camera_region) * 0.001;
    const moved = translateViewpoint(
      position,
      target,
      {
        x: direction.x / length * distance,
        y: direction.y / length * distance,
        z: direction.z / length * distance,
      },
      this.viewpoint.safe_camera_region,
    );
    this.viewpoint.orbit_target = moved.target;
    this.setInteractivePosition(moved.position);
  }

  pan(dx, dy) {
    const distance = Math.max(
      0.1,
      Math.hypot(
        this.viewpoint.position.x - this.viewpoint.orbit_target.x,
        this.viewpoint.position.y - this.viewpoint.orbit_target.y,
        this.viewpoint.position.z - this.viewpoint.orbit_target.z,
      ),
    );
    const scale = distance * 0.0015;
    const right = this.camera.right;
    const up = this.camera.up;
    const translation = {
      x: -right.x * dx * scale + up.x * dy * scale,
      y: -right.y * dx * scale + up.y * dy * scale,
      z: -right.z * dx * scale + up.z * dy * scale,
    };
    this.viewpoint.orbit_target = {
      x: this.viewpoint.orbit_target.x + translation.x,
      y: this.viewpoint.orbit_target.y + translation.y,
      z: this.viewpoint.orbit_target.z + translation.z,
    };
    this.setInteractivePosition({
      x: this.viewpoint.position.x + translation.x,
      y: this.viewpoint.position.y + translation.y,
      z: this.viewpoint.position.z + translation.z,
    });
  }

  setInteractivePosition(candidate) {
    const safe = clampPosition(candidate, this.viewpoint.safe_camera_region);
    const wasClamped = !positionsEqual(candidate, safe);
    this.viewpoint.position = safe;
    this.warning.hidden = !wasClamped;
    this.camera.setPosition(safe.x, safe.y, safe.z);
    const target = this.viewpoint.orbit_target;
    this.camera.lookAt(target.x, target.y, target.z);
    this.camera.rotateLocal(0, 0, this.viewpoint.horizon);
    this.requestSceneFrame();
  }

  publishViewpoint() {
    if (!this.viewpoint || !this.camera) return;
    const rotation = this.camera.getRotation();
    this.viewpoint.orientation = { x: rotation.x, y: rotation.y, z: rotation.z, w: rotation.w };
    this.bridge.submit_viewpoint(JSON.stringify(this.viewpoint));
  }

  updateSubjectGuide() {
    const region = this.viewpoint.subject_region;
    if (!this.subjectGuide) return;
    Object.assign(this.subjectGuide.style, {
      left: `${region.x * 100}%`,
      top: `${region.y * 100}%`,
      width: `${region.width * 100}%`,
      height: `${region.height * 100}%`,
    });
  }

  configureNormalFrame(blurStrengthOverride = null) {
    if (!this.cameraFrame || !this.viewpoint) return;
    const settings = this.viewpoint.depth_of_field;
    const requestedStrength = blurStrengthOverride ?? settings.blur_strength;
    const blurStrength = Math.min(1, Math.max(0, requestedStrength));
    const effect = depthOfFieldForStrength(blurStrength, this.viewpoint.field_of_view);
    const enabled = Boolean(this.sceneEntity && blurStrength > 0 && !this.navigationInputs?.size);
    if (!enabled) {
      this.disableCameraFrame();
      return;
    }
    this.installCameraFrameColorPipeline();
    this.cameraFrame.enabled = true;
    this.cameraFrame.debug = null;
    this.cameraFrame.dof.enabled = true;
    this.cameraFrame.dof.nearBlur = true;
    this.cameraFrame.dof.highQuality = true;
    this.cameraFrame.dof.focusDistance = focusDistanceAtCamera(this.camera.camera.nearClip);
    this.cameraFrame.dof.focusRange = effect.focusRange;
    this.cameraFrame.dof.blurRadius = effect.blurRadius;
    this.cameraFrame.dof.blurRings = 4;
    this.cameraFrame.dof.blurRingPoints = 5;
    this.cameraFrame.rendering.renderFormats = [pc.PIXELFORMAT_RGBA8];
    this.cameraFrame.rendering.sharpness = 0;
    this.cameraFrame.update();
    if (!configureGpuDepthPipeline(this.cameraFrame)) {
      throw new Error('GPU splat depth prepass could not be created');
    }
  }

  queueSnapshotRefresh(referenceChanged) {
    this.snapshotQueue.length = 0;
    const revision = this.snapshotRevision;
    if (referenceChanged) this.snapshotQueue.push({ kind: 'harmonization', revision });
    this.snapshotQueue.push({ kind: 'background', revision });
    this.startNextSnapshot();
  }

  startNextSnapshot() {
    if (this.pendingSnapshotKind || this.sceneFramePending || !this.snapshotQueue?.length) return;
    const next = this.snapshotQueue.shift();
    this.configureNormalFrame(next.kind === 'harmonization' ? 0 : null);
    this.requestSceneFrame(2, next.kind, next.revision);
  }

  applySceneTransform(force = false) {
    if (!this.sceneEntity || !this.viewpoint) return;
    const transform = this.viewpoint.scene_transform;
    const key = JSON.stringify(transform);
    if (!force && key === this.sceneTransformKey) return;
    const translation = transform.translation;
    const orientation = transform.orientation;
    this.sceneEntity.setLocalPosition(translation.x, translation.y, translation.z);
    this.sceneEntity.setLocalRotation(
      orientation.x,
      orientation.y,
      orientation.z,
      orientation.w,
    );
    this.sceneEntity.setLocalScale(transform.scale, transform.scale, transform.scale);
    this.sceneTransformKey = key;
  }

  requestSceneFrame(frameCount = 2, snapshotKind = null, revision = this.snapshotRevision) {
    if (!this.app || this.rendererActive === false) return;
    if (!this.cacheSceneFrames) {
      this.app.autoRender = true;
      this.app.renderNextFrame = true;
      return;
    }
    this.app.autoRender = true;
    if (snapshotKind) {
      this.pendingSnapshotKind = snapshotKind;
      this.pendingSnapshotRevision = revision;
    }
    this.sceneFramesRemaining = Math.max(this.sceneFramesRemaining, frameCount);
    if (this.sceneFramePending) return;
    this.renderSceneFrame();
  }

  renderSceneFrame() {
    this.sceneFramePending = true;
    this.app.once('postrender', () => {
      try {
        this.sceneFramesRemaining -= 1;
        if (this.sceneFramesRemaining === 0) {
          if (this.cacheSceneFrames && this.pendingSnapshotKind) {
            const capture = this.captureSceneFrame();
            if (this.snapshotRequiresSettlement && !this.sceneSettled) {
              this.sceneCaptureAttempts = 0;
            } else if (
              this.sceneEntity
              && (!capture.hasContent || !this.sceneSettled)
              && this.sceneCaptureAttempts < MAX_SCENE_CAPTURE_ATTEMPTS
            ) {
              this.sceneCaptureAttempts += 1;
              this.sceneFramesRemaining = 1;
            } else {
              this.sceneCaptureAttempts = 0;
              const kind = this.pendingSnapshotKind;
              const revision = this.pendingSnapshotRevision;
              const assetId = this.assetId;
              this.pendingSnapshotKind = null;
              void Promise.resolve(this.publishSceneSnapshot(assetId, revision, kind))
                .finally(() => this.startNextSnapshot());
            }
          }
        }
      } catch (error) {
        this.bridge.report_scene_error(
          this.assetId || 'renderer',
          'scene_frame_capture_failed',
          this.safeMessage(error),
        );
      } finally {
        this.sceneFramePending = false;
        if (this.sceneFramesRemaining > 0) {
          this.renderSceneFrame();
        } else if (this.cacheSceneFrames) {
          this.app.autoRender = false;
        }
      }
    });
    this.app.renderNextFrame = true;
  }

  captureSceneFrame() {
    const source = this.app.graphicsDevice.canvas;
    if (
      !this.sceneSnapshot ||
      this.sceneSnapshot.width !== source.width ||
      this.sceneSnapshot.height !== source.height
    ) {
      this.sceneSnapshot = new OffscreenCanvas(source.width, source.height);
      this.sceneSnapshotContext = this.sceneSnapshot.getContext('2d', { alpha: false });
      this.scenePixels = new Uint8ClampedArray(source.width * source.height * 4);
    }
    this.app.graphicsDevice.readPixels(0, 0, source.width, source.height, this.scenePixels);
    flipPixelRows(this.scenePixels, source.width, source.height);
    this.sceneSnapshotContext.putImageData(
      new ImageData(this.scenePixels, source.width, source.height),
      0,
      0,
    );
    return {
      hasContent: hasPixelVariation(this.scenePixels),
    };
  }

  async publishSceneSnapshot(assetId, revision, kind) {
    try {
      const blob = await this.sceneSnapshot.convertToBlob({ type: 'image/png' });
      const payload = bytesToBase64(new Uint8Array(await blob.arrayBuffer()));
      this.bridge.report_snapshot_ready(assetId, revision, kind, payload);
    } catch (error) {
      this.bridge.report_scene_error(
        assetId || 'renderer',
        'scene_frame_encode_failed',
        this.safeMessage(error),
      );
    }
  }

  publishCurrentSnapshot() {
    if (!this.sceneEntity) return;
    try {
      if (!this.captureSceneFrame().hasContent) return;
      void this.publishSceneSnapshot(this.assetId, this.snapshotRevision, 'background');
    } catch (error) {
      this.bridge.report_scene_error(
        this.assetId || 'renderer',
        'scene_frame_capture_failed',
        this.safeMessage(error),
      );
    }
  }

  setLoading(loading, message) {
    const overlay = document.getElementById('loading');
    if (!overlay) return;
    overlay.hidden = !loading;
    overlay.querySelector('span').textContent = message;
  }

  safeMessage(error) {
    const message = error instanceof Error ? error.message : String(error);
    return message.slice(0, 300) || 'Unknown renderer error';
  }
}

const start = async (bridge, options = {}) => {
  const renderer = new SceneRenderer(bridge, options);
  await renderer.initialize();
  bridge.renderer_ready();
};

export {
  SceneRenderer,
  configureGsplatStreaming,
  bytesToBase64,
  depthOfFieldForStrength,
  focusDistanceAtCamera,
  harmonizationViewpointKey,
  flipPixelRows,
  hasPixelVariation,
  isStreamedSceneUrl,
  start,
};
