import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

import { refineConfidenceMask, temporalWeightForInterval } from './matting.mjs';

let segmenter = null;
let history = null;
let historyTimestamp = null;
let settings = { threshold: 0.55, temporal: 0.65, feather: 0.06, edgeRadius: 0 };
let runtimeUrls = [];

const safeMessage = (error) => {
  const message = error instanceof Error ? error.message : String(error);
  return message.slice(0, 300) || 'Unknown segmentation error';
};

const initialize = async () => {
  const simd = await FilesetResolver.isSimdSupported(false);
  const suffix = simd ? '' : '_nosimd';
  const loaderResponse = await fetch(`bbapp://matting/wasm/vision_wasm${suffix}_internal.js`);
  const binaryResponse = await fetch(`bbapp://matting/wasm/vision_wasm${suffix}_internal.wasm`);
  const modelResponse = await fetch('bbapp://matting/selfie_segmenter_landscape.tflite');
  if (!loaderResponse.ok || !binaryResponse.ok || !modelResponse.ok) {
    throw new Error('A packaged segmentation asset could not be loaded');
  }
  const loaderUrl = URL.createObjectURL(await loaderResponse.blob());
  const binaryUrl = URL.createObjectURL(await binaryResponse.blob());
  runtimeUrls = [loaderUrl, binaryUrl];
  const fileset = { wasmLoaderPath: loaderUrl, wasmBinaryPath: binaryUrl };
  const modelAssetBuffer = new Uint8Array(await modelResponse.arrayBuffer());
  segmenter = await ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer },
    runningMode: 'VIDEO',
    outputConfidenceMasks: true,
    outputCategoryMask: false,
  });
  self.postMessage({ type: 'ready', labels: segmenter.getLabels() });
  runtimeUrls.forEach((url) => URL.revokeObjectURL(url));
  runtimeUrls = [];
};

const segment = (message) => {
  const started = performance.now();
  let result = null;
  try {
    result = segmenter.segmentForVideo(message.frame, message.timestamp);
    const masks = result.confidenceMasks ?? [];
    if (masks.length === 0) throw new Error('The person model returned no confidence mask');
    const labels = segmenter.getLabels();
    const labelledIndex = labels.findIndex((label) => label.toLowerCase() === 'person');
    const personIndex = labelledIndex >= 0 ? labelledIndex : masks.length - 1;
    const mask = masks[personIndex];
    const confidence = mask.getAsFloat32Array();
    const elapsed = historyTimestamp === null ? 1000 / 30 : message.timestamp - historyTimestamp;
    const temporal = temporalWeightForInterval(settings.temporal, elapsed);
    const refined = refineConfidenceMask(
      confidence,
      history,
      mask.width,
      mask.height,
      { ...settings, temporal },
    );
    history = refined.history;
    historyTimestamp = message.timestamp;
    self.postMessage(
      {
        type: 'mask',
        timestamp: message.timestamp,
        width: mask.width,
        height: mask.height,
        alpha: refined.alpha.buffer,
        workerTime: performance.now() - started,
      },
      [refined.alpha.buffer],
    );
  } catch (error) {
    self.postMessage({ type: 'frame-error', timestamp: message.timestamp, message: safeMessage(error) });
  } finally {
    result?.close();
    message.frame.close?.();
  }
};

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === 'initialize') {
    try {
      await initialize();
    } catch (error) {
      runtimeUrls.forEach((url) => URL.revokeObjectURL(url));
      runtimeUrls = [];
      self.postMessage({ type: 'initialization-error', message: safeMessage(error) });
    }
  } else if (message.type === 'settings') {
    settings = { ...settings, ...message.settings };
    history = null;
    historyTimestamp = null;
  } else if (message.type === 'frame' && segmenter) {
    segment(message);
  } else if (message.type === 'close') {
    segmenter?.close();
    segmenter = null;
    history = null;
    historyTimestamp = null;
    runtimeUrls.forEach((url) => URL.revokeObjectURL(url));
    runtimeUrls = [];
    self.close();
  }
};
