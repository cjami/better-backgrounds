import * as pc from 'playcanvas';

import { clampPosition, orbitPosition, positionsEqual, zoomPosition } from './viewpoint.mjs';

class SceneRenderer {
  constructor(bridge) {
    this.bridge = bridge;
    this.app = null;
    this.camera = null;
    this.sceneEntity = null;
    this.sceneAsset = null;
    this.assetId = '';
    this.viewpoint = null;
    this.resetViewpoint = null;
    this.drag = null;
    this.warning = document.getElementById('safe-warning');
    this.status = document.getElementById('status');
    this.subjectGuide = document.getElementById('subject-guide');
  }

  async initialize() {
    const canvas = document.getElementById('scene-canvas');
    try {
      const device = await pc.createGraphicsDevice(canvas, {
        deviceTypes: [pc.DEVICETYPE_WEBGPU],
        antialias: false,
        powerPreference: 'high-performance',
      });
      const options = new pc.AppOptions();
      options.graphicsDevice = device;
      options.componentSystems = [pc.CameraComponentSystem, pc.GSplatComponentSystem];
      options.resourceHandlers = [pc.GSplatHandler];
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
      this.status.textContent = device.deviceType === pc.DEVICETYPE_WEBGPU ? 'WEBGPU' : 'WEBGL 2';
      this.bindInput(canvas);
      window.addEventListener('resize', () => this.app?.resizeCanvas());
      this.connectBridge();
      this.bridge.renderer_ready();
    } catch (error) {
      this.status.textContent = 'GPU UNAVAILABLE';
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
      this.app.root.addChild(entity);
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
      this.updateSubjectGuide();
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
  }

  publishViewpoint() {
    if (!this.viewpoint || !this.camera) return;
    const rotation = this.camera.getRotation();
    this.viewpoint.orientation = { x: rotation.x, y: rotation.y, z: rotation.z, w: rotation.w };
    this.bridge.submit_viewpoint(JSON.stringify(this.viewpoint));
  }

  updateSubjectGuide() {
    const region = this.viewpoint.subject_region;
    Object.assign(this.subjectGuide.style, {
      left: `${region.x * 100}%`,
      top: `${region.y * 100}%`,
      width: `${region.width * 100}%`,
      height: `${region.height * 100}%`,
    });
  }

  setLoading(loading, message) {
    const overlay = document.getElementById('loading');
    overlay.hidden = !loading;
    overlay.querySelector('span').textContent = message;
  }

  safeMessage(error) {
    const message = error instanceof Error ? error.message : String(error);
    return message.slice(0, 300) || 'Unknown renderer error';
  }
}

const start = (bridge) => new SceneRenderer(bridge).initialize();

export { start };
