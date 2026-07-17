import * as pc from 'playcanvas';

import { clampPosition, orbitPosition, positionsEqual, zoomPosition } from './viewpoint.mjs';

const MAX_SCENE_CAPTURE_ATTEMPTS = 600;

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

class SceneRenderer {
  constructor(bridge, { cacheSceneFrames = false } = {}) {
    this.bridge = bridge;
    this.cacheSceneFrames = cacheSceneFrames;
    this.app = null;
    this.camera = null;
    this.sceneEntity = null;
    this.sceneAsset = null;
    this.sceneTransformKey = '';
    this.assetId = '';
    this.viewpoint = null;
    this.resetViewpoint = null;
    this.drag = null;
    this.sceneFramePending = false;
    this.sceneFramesRemaining = 0;
    this.sceneSnapshot = null;
    this.sceneSnapshotContext = null;
    this.scenePixels = null;
    this.sceneCaptureAttempts = 0;
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

      this.camera = new pc.Entity('virtual-camera');
      this.camera.addComponent('camera', {
        clearColor: new pc.Color(0.035, 0.037, 0.043),
      });
      this.app.root.addChild(this.camera);
      this.app.start();
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
    } catch (error) {
      if (this.status) this.status.textContent = 'GPU UNAVAILABLE';
      this.bridge.report_scene_error('renderer', 'gpu_unavailable', this.safeMessage(error));
    }
  }

  connectBridge() {
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

  loadScene(assetId, url, payload) {
    if (!this.app || !this.camera) return;
    this.assetId = assetId;
    this.applyViewpoint(payload, true);
    this.removeScene();
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
      this.applySceneTransform();
      this.app.root.addChild(entity);
      this.requestSceneFrame();
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
    this.sceneEntity?.destroy();
    this.sceneEntity = null;
    if (this.sceneAsset && this.app) {
      this.app.assets.remove(this.sceneAsset);
      this.sceneAsset.unload();
    }
    this.sceneAsset = null;
    this.sceneTransformKey = '';
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
      this.requestSceneFrame();
    } catch (error) {
      this.bridge.report_scene_error(this.assetId || 'renderer', 'viewpoint_invalid', this.safeMessage(error));
    }
  }

  bindInput(canvas) {
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
      canvas.setPointerCapture(event.pointerId);
      this.drag = { x: event.clientX, y: event.clientY, pan: event.button === 2 || event.shiftKey };
    });
    canvas.addEventListener('pointerup', () => {
      this.drag = null;
      this.publishViewpoint();
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.drag || !this.viewpoint) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      if (this.drag.pan) this.pan(dx, dy);
      else this.orbit(dx, dy);
    });
    canvas.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        if (!this.viewpoint) return;
        const position = zoomPosition(
          this.viewpoint.position,
          this.viewpoint.orbit_target,
          Math.exp(event.deltaY * 0.001),
        );
        this.setInteractivePosition(position);
        this.publishViewpoint();
      },
      { passive: false },
    );
    canvas.addEventListener('keydown', (event) => {
      if (!this.viewpoint) return;
      const orbitKeys = { ArrowLeft: [-12, 0], ArrowRight: [12, 0], ArrowUp: [0, -12], ArrowDown: [0, 12] };
      if (orbitKeys[event.key]) {
        event.preventDefault();
        this.orbit(...orbitKeys[event.key]);
        this.publishViewpoint();
      }
      if (event.key === '0') this.bridge.reset_requested();
    });
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

  applySceneTransform() {
    if (!this.sceneEntity || !this.viewpoint) return;
    const transform = this.viewpoint.scene_transform;
    const key = JSON.stringify(transform);
    if (key === this.sceneTransformKey) return;
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

  requestSceneFrame(frameCount = 2) {
    if (!this.app) return;
    if (!this.cacheSceneFrames) {
      this.app.autoRender = true;
      this.app.renderNextFrame = true;
      return;
    }
    this.app.autoRender = true;
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
          const capture = this.captureSceneFrame();
          if (
            this.sceneEntity &&
            !capture.hasContent &&
            this.sceneCaptureAttempts < MAX_SCENE_CAPTURE_ATTEMPTS
          ) {
            capture.frame.close();
            this.sceneCaptureAttempts += 1;
            this.sceneFramesRemaining = 1;
          } else {
            this.sceneCaptureAttempts = 0;
            this.app.autoRender = false;
            window.dispatchEvent(new CustomEvent('bb-scene-frame', { detail: capture.frame }));
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
        if (this.sceneFramesRemaining > 0) this.renderSceneFrame();
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
      frame: this.sceneSnapshot.transferToImageBitmap(),
      hasContent: hasPixelVariation(this.scenePixels),
    };
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

const start = async (bridge) => {
  const renderer = new SceneRenderer(bridge);
  await renderer.initialize();
  bridge.renderer_ready();
};

export { SceneRenderer, flipPixelRows, hasPixelVariation, start };
