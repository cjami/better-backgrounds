import { SceneRenderer } from './main.mjs';
import { LatestFrameScheduler } from './matting.mjs';

const PROCESSING_WIDTH = 256;
const PROCESSING_HEIGHT = 144;

class LivePipeline {
  constructor(bridge) {
    this.bridge = bridge;
    this.video = document.getElementById('camera-source');
    this.output = document.getElementById('live-output');
    this.outputContext = this.output.getContext('2d', { alpha: false });
    this.foreground = new OffscreenCanvas(1, 1);
    this.foregroundContext = this.foreground.getContext('2d', { willReadFrequently: false });
    this.mask = new OffscreenCanvas(PROCESSING_WIDTH, PROCESSING_HEIGHT);
    this.maskContext = this.mask.getContext('2d');
    this.stream = null;
    this.worker = null;
    this.workerReady = false;
    this.frameCallback = null;
    this.mode = 'show';
    this.wipe = 50;
    this.mirrored = true;
    this.maskTimestamp = -Infinity;
    this.maskCount = 0;
    this.displayCount = 0;
    this.lastRateTime = performance.now();
    this.displayRate = 0;
    this.maskRate = 0;
    this.workerTime = 0;
    this.capturePending = false;
    this.backgroundFrame = null;
    this.maskPreview = document.getElementById('mask-preview');
    this.maskPreviewContext = this.maskPreview.getContext('2d');
    this.scheduler = new LatestFrameScheduler((timestamp, capture) => {
      this.processScheduledFrame(timestamp, capture);
    });
    window.addEventListener('bb-scene-frame', (event) => {
      this.backgroundFrame?.close();
      this.backgroundFrame = event.detail;
      this.drawBackground();
    });
  }

  connectBridge() {
    this.bridge.camera_start_requested.connect((preferredLabel, mirrored) => {
      this.startCamera(preferredLabel, mirrored);
    });
    this.bridge.camera_stop_requested.connect(() => this.stopCamera('idle'));
    this.bridge.presentation_requested.connect((mode, wipe) => {
      this.mode = mode;
      this.wipe = Math.min(100, Math.max(0, wipe));
      document.body.dataset.mode = mode;
    });
    this.bridge.mirroring_requested.connect((mirrored) => { this.mirrored = mirrored; });
    this.bridge.matting_settings_requested.connect((payload) => {
      try {
        this.worker?.postMessage({ type: 'settings', settings: JSON.parse(payload) });
      } catch {
        this.setStatus('error', 'Invalid matting settings');
      }
    });
  }

  async startCamera(preferredLabel, mirrored) {
    if (this.stream) return;
    this.mirrored = mirrored;
    this.setStatus('starting', 'Requesting camera permission…');
    try {
      await this.ensureWorker();
      let stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } },
        audio: false,
      });
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === 'videoinput');
      this.bridge.report_camera_devices(JSON.stringify(cameras.map((device) => ({
        device_id: device.deviceId,
        description: device.label || 'Camera',
      }))));
      const preferred = cameras.find((device) => device.label === preferredLabel);
      const currentLabel = stream.getVideoTracks()[0]?.label;
      if (preferred && preferred.label !== currentLabel) {
        stream.getTracks().forEach((track) => track.stop());
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: preferred.deviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 },
          },
          audio: false,
        });
      }
      this.stream = stream;
      const track = stream.getVideoTracks()[0];
      track.addEventListener('ended', () => this.deviceLost());
      this.video.srcObject = stream;
      await this.video.play();
      this.resizeOutput();
      this.scheduler.reset();
      this.setStatus('live', `Live · ${track.label || 'Camera'}`);
      this.requestNextFrame();
    } catch (error) {
      this.stopCamera('error');
      this.setStatus('error', this.cameraError(error));
    }
  }

  async ensureWorker() {
    if (this.worker) return;
    const response = await fetch('bbapp://renderer/matting-worker.js');
    if (!response.ok) throw new Error('Unable to load the packaged segmentation worker');
    const workerUrl = URL.createObjectURL(await response.blob());
    this.worker = new Worker(workerUrl);
    URL.revokeObjectURL(workerUrl);
    this.worker.onmessage = (event) => this.handleWorkerMessage(event.data);
    this.worker.onerror = () => this.setStatus('error', 'Segmentation worker failed');
    this.worker.postMessage({ type: 'initialize' });
    await new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error('Segmentation model timed out')), 15000);
      const listener = (event) => {
        if (event.data.type === 'ready') {
          window.clearTimeout(timeout);
          this.worker.removeEventListener('message', listener);
          this.workerReady = true;
          resolve();
        } else if (event.data.type === 'initialization-error') {
          window.clearTimeout(timeout);
          this.worker.removeEventListener('message', listener);
          reject(new Error(event.data.message));
        }
      };
      this.worker.addEventListener('message', listener);
    });
  }

  requestNextFrame() {
    if (!this.stream) return;
    this.frameCallback = this.video.requestVideoFrameCallback((_now, metadata) => {
      this.renderFrame(metadata);
      this.requestNextFrame();
    });
  }

  async renderFrame(metadata) {
    if (!this.stream) return;
    const timestamp = metadata.mediaTime * 1000;
    if (!this.workerReady || this.capturePending) {
      this.scheduler.droppedFrames += 1;
      return;
    }
    this.capturePending = true;
    try {
      const frame = await createImageBitmap(this.video);
      const capture = {
        frame,
        capturedAt: performance.now(),
        close: () => frame.close(),
      };
      if (this.stream) this.scheduler.submit(timestamp, capture);
      else capture.close();
    } catch {
      this.scheduler.droppedFrames += 1;
    } finally {
      this.capturePending = false;
    }
  }

  async processScheduledFrame(timestamp, capture) {
    try {
      const frame = await createImageBitmap(capture.frame, {
        resizeWidth: PROCESSING_WIDTH,
        resizeHeight: PROCESSING_HEIGHT,
        resizeQuality: 'medium',
      });
      if (!this.stream || !this.worker) throw new Error('Camera stopped');
      this.worker.postMessage({ type: 'frame', timestamp, frame }, [frame]);
    } catch {
      this.scheduler.complete(timestamp)?.close();
      this.scheduler.droppedFrames += 1;
    }
  }

  handleWorkerMessage(message) {
    if (message.type === 'mask') {
      const capture = this.scheduler.complete(message.timestamp);
      if (capture === null) return;
      this.maskTimestamp = message.timestamp;
      this.workerTime = message.workerTime;
      this.maskCount += 1;
      const alpha = new Uint8ClampedArray(message.alpha);
      const pixels = new Uint8ClampedArray(alpha.length * 4);
      for (let index = 0; index < alpha.length; index += 1) {
        const offset = index * 4;
        pixels[offset] = 255;
        pixels[offset + 1] = 255;
        pixels[offset + 2] = 255;
        pixels[offset + 3] = alpha[index];
      }
      this.mask.width = message.width;
      this.mask.height = message.height;
      this.maskContext.putImageData(new ImageData(pixels, message.width, message.height), 0, 0);
      this.maskPreview.width = message.width;
      this.maskPreview.height = message.height;
      this.maskPreviewContext.putImageData(
        new ImageData(new Uint8ClampedArray(pixels), message.width, message.height),
        0,
        0,
      );
      this.compose(capture.frame);
      capture.close();
      this.displayCount += 1;
      this.updateDiagnostics(performance.now() - capture.capturedAt);
    } else if (message.type === 'frame-error') {
      this.scheduler.complete(message.timestamp)?.close();
    } else if (message.type === 'initialization-error') {
      this.setStatus('error', `Segmentation unavailable: ${message.message}`);
    }
  }

  compose(sourceFrame) {
    const width = this.output.width;
    const height = this.output.height;
    if (width === 0 || height === 0) return;
    this.drawBackground();
    if (this.maskTimestamp > -Infinity) {
      this.foregroundContext.clearRect(0, 0, width, height);
      this.foregroundContext.globalCompositeOperation = 'source-over';
      this.foregroundContext.drawImage(sourceFrame, 0, 0, width, height);
      this.foregroundContext.globalCompositeOperation = 'destination-in';
      this.foregroundContext.drawImage(this.mask, 0, 0, width, height);
      this.foregroundContext.globalCompositeOperation = 'source-over';
      this.drawMirrored(this.foreground, 0, width);
    }
    if (this.mode === 'compare') {
      const split = Math.round(width * this.wipe / 100);
      this.outputContext.save();
      this.outputContext.beginPath();
      this.outputContext.rect(0, 0, split, height);
      this.outputContext.clip();
      this.drawMirrored(sourceFrame, 0, width, false);
      this.outputContext.restore();
      this.outputContext.fillStyle = '#e0a34a';
      this.outputContext.fillRect(split - 1, 0, 2, height);
    }
  }

  drawMirrored(source, x, width, suppressSpill = true) {
    this.outputContext.save();
    if (suppressSpill) this.outputContext.filter = 'saturate(0.98)';
    if (this.mirrored) {
      this.outputContext.translate(x + width, 0);
      this.outputContext.scale(-1, 1);
      this.outputContext.drawImage(source, 0, 0, width, this.output.height);
    } else {
      this.outputContext.drawImage(source, x, 0, width, this.output.height);
    }
    this.outputContext.restore();
  }

  resizeOutput() {
    const width = this.video.videoWidth || 1280;
    const height = this.video.videoHeight || 720;
    this.output.width = width;
    this.output.height = height;
    this.foreground.width = width;
    this.foreground.height = height;
    this.drawBackground();
  }

  drawBackground() {
    const width = this.output.width;
    const height = this.output.height;
    if (width === 0 || height === 0) return;
    if (this.backgroundFrame) {
      this.outputContext.drawImage(this.backgroundFrame, 0, 0, width, height);
    } else {
      this.outputContext.fillStyle = '#090a0c';
      this.outputContext.fillRect(0, 0, width, height);
    }
  }

  stopCamera(state = 'idle') {
    if (this.frameCallback !== null && this.video.cancelVideoFrameCallback) {
      this.video.cancelVideoFrameCallback(this.frameCallback);
    }
    this.frameCallback = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.worker?.postMessage({ type: 'close' });
    this.worker?.terminate();
    this.worker = null;
    this.workerReady = false;
    this.maskTimestamp = -Infinity;
    this.scheduler.reset();
    if (state === 'idle') this.setStatus('idle', 'Camera stopped');
  }

  deviceLost() {
    if (!this.stream) return;
    this.stopCamera('lost');
    this.setStatus('lost', 'Camera disconnected — reconnect it and restart');
  }

  setStatus(state, message) {
    document.body.dataset.camera = state;
    document.getElementById('camera-state').textContent = message;
    this.bridge.report_camera_state(state, message);
  }

  updateDiagnostics(maskAge) {
    const now = performance.now();
    const elapsed = now - this.lastRateTime;
    if (elapsed < 500) return;
    this.displayRate = this.displayCount * 1000 / elapsed;
    this.maskRate = this.maskCount * 1000 / elapsed;
    this.displayCount = 0;
    this.maskCount = 0;
    this.lastRateTime = now;
    const diagnostics = {
      display_fps: this.displayRate,
      mask_fps: this.maskRate,
      mask_age_ms: Number.isFinite(maskAge) ? Math.max(0, maskAge) : 0,
      dropped_frames: this.scheduler.droppedFrames,
      worker_time_ms: this.workerTime,
      processing_width: PROCESSING_WIDTH,
      processing_height: PROCESSING_HEIGHT,
    };
    document.getElementById('diagnostics').textContent =
      `${this.displayRate.toFixed(0)} display fps · ${this.maskRate.toFixed(0)} masks/s · ` +
      `${diagnostics.mask_age_ms.toFixed(0)} ms mask age · ${this.scheduler.droppedFrames} dropped · ` +
      `${this.workerTime.toFixed(1)} ms worker · ${PROCESSING_WIDTH}×${PROCESSING_HEIGHT}`;
    this.bridge.report_diagnostics(JSON.stringify(diagnostics));
  }

  cameraError(error) {
    const name = error instanceof DOMException ? error.name : '';
    if (name === 'NotAllowedError') return 'Camera permission was denied or dismissed';
    if (name === 'NotFoundError') return 'No camera is available';
    if (name === 'NotReadableError') return 'Camera is already in use or unavailable';
    return `Camera failed: ${(error?.message ?? String(error)).slice(0, 240)}`;
  }
}

const startLive = async (bridge) => {
  const pipeline = new LivePipeline(bridge);
  pipeline.connectBridge();
  pipeline.setStatus('starting', 'Preparing camera preview…');
  const sceneRenderer = new SceneRenderer(bridge, { cacheSceneFrames: true });
  await sceneRenderer.initialize();
  bridge.renderer_ready();
  window.addEventListener('beforeunload', () => pipeline.stopCamera('closed'));
};

export { LivePipeline, startLive };
