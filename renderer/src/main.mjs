import * as pc from 'playcanvas';

import { clampPosition, orbitPosition, positionsEqual, zoomPosition } from './viewpoint.mjs';

const MAX_SCENE_CAPTURE_ATTEMPTS = 12;
const SUBJECT_FOCUS_DISTANCE = 1.5;
const MIN_SUBJECT_FOCUS_RANGE = 0.8;
const MAX_SUBJECT_FOCUS_RANGE = 8;
const MAX_DOF_BLUR_RADIUS = 8;
const REFERENCE_FIELD_OF_VIEW = 42;
const MIN_PERSPECTIVE_SCALE = 0.75;
const MAX_PERSPECTIVE_SCALE = 1.25;
const DEPTH_PROXY_COLUMNS = 384;
const DEPTH_PROXY_SAMPLE_RANK = 3;
const DEPTH_PROXY_MAX_RELATIVE_STEP = 0.2;
const DEPTH_PROXY_MIN_STEP_METRES = 0.5;

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
  const aperture = Math.min(1, blurStrength * perspectiveScale);
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

const buildDepthProxyGeometry = (gsplatData, requestedColumns = DEPTH_PROXY_COLUMNS) => {
  const x = gsplatData?.getProp?.('x');
  const y = gsplatData?.getProp?.('y');
  const z = gsplatData?.getProp?.('z');
  const imageSize = gsplatData?.getProp?.('image_size', 'image_size');
  const intrinsic = gsplatData?.getProp?.('intrinsic', 'intrinsic');
  const sourceWidth = Math.trunc(imageSize?.[0] ?? 0);
  const sourceHeight = Math.trunc(imageSize?.[1] ?? 0);
  const focalX = intrinsic?.[0];
  const focalY = intrinsic?.[4];
  const principalX = intrinsic?.[2];
  const principalY = intrinsic?.[5];
  if (
    sourceWidth < 2
    || sourceHeight < 2
    || !x
    || !y
    || !z
    || y.length !== x.length
    || z.length !== x.length
    || !Number.isFinite(focalX)
    || !Number.isFinite(focalY)
    || focalX <= 0
    || focalY <= 0
    || !Number.isFinite(principalX)
    || !Number.isFinite(principalY)
  ) return null;

  const columns = Math.max(2, Math.min(sourceWidth, Math.trunc(requestedColumns)));
  const rows = Math.max(2, Math.min(
    sourceHeight,
    Math.round((sourceHeight / sourceWidth) * columns),
  ));
  const positions = new Float32Array(columns * rows * 3);
  const valid = new Uint8Array(columns * rows);
  const depths = new Float32Array(columns * rows);
  const rankedDepths = Array.from(
    { length: DEPTH_PROXY_SAMPLE_RANK },
    () => new Float32Array(columns * rows).fill(Number.POSITIVE_INFINITY),
  );

  for (let sourceIndex = 0; sourceIndex < z.length; sourceIndex += 1) {
    const depth = z[sourceIndex];
    if (
      !Number.isFinite(x[sourceIndex])
      || !Number.isFinite(y[sourceIndex])
      || !Number.isFinite(depth)
      || depth <= 0
    ) continue;
    const pixelX = focalX * x[sourceIndex] / depth + principalX;
    const pixelY = focalY * y[sourceIndex] / depth + principalY;
    const column = Math.floor(pixelX * columns / sourceWidth);
    const row = Math.floor(pixelY * rows / sourceHeight);
    if (column < 0 || column >= columns || row < 0 || row >= rows) continue;
    const targetIndex = row * columns + column;
    for (let rank = 0; rank < DEPTH_PROXY_SAMPLE_RANK; rank += 1) {
      if (depth >= rankedDepths[rank][targetIndex]) continue;
      for (let move = DEPTH_PROXY_SAMPLE_RANK - 1; move > rank; move -= 1) {
        rankedDepths[move][targetIndex] = rankedDepths[move - 1][targetIndex];
      }
      rankedDepths[rank][targetIndex] = depth;
      break;
    }
  }

  for (let row = 0; row < rows; row += 1) {
    const pixelY = (row + 0.5) * sourceHeight / rows;
    for (let column = 0; column < columns; column += 1) {
      const targetIndex = row * columns + column;
      const rankedDepth = rankedDepths[DEPTH_PROXY_SAMPLE_RANK - 1][targetIndex];
      const depth = Number.isFinite(rankedDepth) ? rankedDepth : rankedDepths[0][targetIndex];
      if (!Number.isFinite(depth)) continue;
      const pixelX = (column + 0.5) * sourceWidth / columns;
      const positionOffset = targetIndex * 3;
      positions[positionOffset] = (pixelX - principalX) * depth / focalX;
      positions[positionOffset + 1] = (pixelY - principalY) * depth / focalY;
      positions[positionOffset + 2] = depth;
      depths[targetIndex] = depth;
      valid[targetIndex] = 1;
    }
  }

  const indices = [];
  const addTriangle = (first, second, third) => {
    if (!valid[first] || !valid[second] || !valid[third]) return;
    const nearest = Math.min(depths[first], depths[second], depths[third]);
    const farthest = Math.max(depths[first], depths[second], depths[third]);
    const maximumStep = Math.max(
      DEPTH_PROXY_MIN_STEP_METRES,
      nearest * DEPTH_PROXY_MAX_RELATIVE_STEP,
    );
    if (farthest - nearest <= maximumStep) indices.push(first, second, third);
  };
  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const topLeft = row * columns + column;
      const topRight = topLeft + 1;
      const bottomLeft = topLeft + columns;
      const bottomRight = bottomLeft + 1;
      addTriangle(topLeft, bottomLeft, topRight);
      addTriangle(topRight, bottomLeft, bottomRight);
    }
  }
  if (!indices.length) return null;
  return { columns, rows, positions, indices: new Uint32Array(indices) };
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

class SceneRenderer {
  constructor(bridge, { cacheSceneFrames = false } = {}) {
    this.bridge = bridge;
    this.cacheSceneFrames = cacheSceneFrames;
    this.app = null;
    this.camera = null;
    this.cameraFrame = null;
    this.sceneEntity = null;
    this.sceneAsset = null;
    this.depthProxyEntity = null;
    this.depthProxyMesh = null;
    this.depthProxyMaterial = null;
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
    this.sceneSettled = false;
    this.metricDepthAvailable = false;
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

      this.camera = new pc.Entity('virtual-camera');
      this.camera.addComponent('camera', {
        clearColor: new pc.Color(0.035, 0.037, 0.043),
      });
      this.app.root.addChild(this.camera);
      this.cameraFrame = new pc.CameraFrame(this.app, this.camera.camera);
      this.cameraFrame.update();
      this.app.start();
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
        if (this.sceneSettled && this.pendingSnapshotKind) this.requestSceneFrame(1);
      });
      this.app.systems.gsplat.on('frame:request', () => this.requestSceneFrame(1));
    } catch (error) {
      if (this.status) this.status.textContent = 'GPU UNAVAILABLE';
      this.bridge.report_scene_error('renderer', 'gpu_unavailable', this.safeMessage(error));
    }
  }

  connectBridge() {
    this.bridge.scene_requested.connect((assetId, url, payload, metricDepthAvailable) => {
      this.loadScene(assetId, url, payload, metricDepthAvailable);
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

  loadScene(assetId, url, payload, metricDepthAvailable = false) {
    if (!this.app || !this.camera) return;
    this.assetId = assetId;
    this.metricDepthAvailable = metricDepthAvailable;
    this.sceneSettled = false;
    this.removeScene();
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
      this.createDepthProxy(loadedAsset.resource?.gsplatData);
      this.sceneCaptureAttempts = 0;
      this.applySceneTransform();
      this.app.root.addChild(entity);
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
    this.depthProxyEntity?.destroy();
    this.depthProxyMesh?.destroy();
    this.depthProxyMaterial?.destroy();
    this.depthProxyEntity = null;
    this.depthProxyMesh = null;
    this.depthProxyMaterial = null;
    this.sceneEntity?.destroy();
    this.sceneEntity = null;
    if (this.sceneAsset && this.app) {
      this.app.assets.remove(this.sceneAsset);
      this.sceneAsset.unload();
    }
    this.sceneAsset = null;
    this.sceneTransformKey = '';
    this.sceneSettled = false;
    this.snapshotQueue.length = 0;
    this.harmonizationViewpointKey = '';
  }

  createDepthProxy(gsplatData) {
    if (!this.metricDepthAvailable || !this.app) return;
    const geometry = buildDepthProxyGeometry(gsplatData);
    if (!geometry) return;

    const mesh = new pc.Mesh(this.app.graphicsDevice);
    mesh.setPositions(geometry.positions);
    mesh.setIndices(geometry.indices);
    mesh.update(pc.PRIMITIVE_TRIANGLES);

    const material = new pc.StandardMaterial();
    material.cull = pc.CULLFACE_NONE;
    material.depthWrite = true;
    material.update();

    const entity = new pc.Entity('sharp-depth-proxy');
    const meshInstance = new pc.MeshInstance(mesh, material, entity);
    meshInstance.shaderPassMask &= ~(1 << pc.SHADER_FORWARD);
    entity.addComponent('render', { meshInstances: [meshInstance] });
    this.depthProxyEntity = entity;
    this.depthProxyMesh = mesh;
    this.depthProxyMaterial = material;
    this.app.root.addChild(entity);
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

  configureNormalFrame(blurStrengthOverride = null) {
    if (!this.cameraFrame || !this.viewpoint) return;
    const settings = this.viewpoint.depth_of_field;
    const requestedStrength = blurStrengthOverride ?? settings.blur_strength;
    const blurStrength = Math.min(1, Math.max(0, requestedStrength));
    const effect = depthOfFieldForStrength(blurStrength, this.viewpoint.field_of_view);
    const enabled = this.metricDepthAvailable;
    this.cameraFrame.debug = null;
    this.cameraFrame.dof.enabled = enabled;
    this.cameraFrame.dof.nearBlur = true;
    this.cameraFrame.dof.highQuality = true;
    this.cameraFrame.dof.focusDistance = SUBJECT_FOCUS_DISTANCE;
    this.cameraFrame.dof.focusRange = effect.focusRange;
    this.cameraFrame.dof.blurRadius = effect.blurRadius;
    this.cameraFrame.dof.blurRings = 4;
    this.cameraFrame.dof.blurRingPoints = 5;
    this.cameraFrame.rendering.sharpness = enabled ? 0.24 : 0;
    this.cameraFrame.update();
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

  applySceneTransform() {
    if (!this.sceneEntity || !this.viewpoint) return;
    const transform = this.viewpoint.scene_transform;
    const key = JSON.stringify(transform);
    if (key === this.sceneTransformKey) return;
    const translation = transform.translation;
    const orientation = transform.orientation;
    for (const entity of [this.sceneEntity, this.depthProxyEntity]) {
      if (!entity) continue;
      entity.setLocalPosition(translation.x, translation.y, translation.z);
      entity.setLocalRotation(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
      );
      entity.setLocalScale(transform.scale, transform.scale, transform.scale);
    }
    this.sceneTransformKey = key;
  }

  requestSceneFrame(frameCount = 2, snapshotKind = null, revision = this.snapshotRevision) {
    if (!this.app) return;
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
            if (
              this.sceneEntity &&
              (!capture.hasContent || !this.sceneSettled) &&
              this.sceneCaptureAttempts < MAX_SCENE_CAPTURE_ATTEMPTS
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
  buildDepthProxyGeometry,
  bytesToBase64,
  depthOfFieldForStrength,
  harmonizationViewpointKey,
  flipPixelRows,
  hasPixelVariation,
  start,
};
