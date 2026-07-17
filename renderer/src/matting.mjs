const clamp01 = (value) => Math.min(1, Math.max(0, value));

const smoothstep = (minimum, maximum, value) => {
  if (maximum <= minimum) return value >= maximum ? 1 : 0;
  const position = clamp01((value - minimum) / (maximum - minimum));
  return position * position * (3 - 2 * position);
};

const temporalWeightForInterval = (weight, elapsedMs, referenceMs = 1000 / 30) => {
  const boundedWeight = clamp01(weight);
  const boundedElapsed = Math.max(0, elapsedMs);
  return boundedWeight ** (boundedElapsed / referenceMs);
};

const boxBlur = (values, width, height, radius) => {
  if (radius <= 0) return values;
  const horizontal = new Float32Array(values.length);
  const output = new Float32Array(values.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      let count = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sample = x + offset;
        if (sample >= 0 && sample < width) {
          total += values[y * width + sample];
          count += 1;
        }
      }
      horizontal[y * width + x] = total / count;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      let count = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sample = y + offset;
        if (sample >= 0 && sample < height) {
          total += horizontal[sample * width + x];
          count += 1;
        }
      }
      output[y * width + x] = total / count;
    }
  }
  return output;
};

const refineConfidenceMask = (confidence, previous, width, height, options = {}) => {
  if (confidence.length !== width * height) throw new RangeError('Mask dimensions do not match');
  if (previous && previous.length !== confidence.length) throw new RangeError('Mask history does not match');
  const threshold = options.threshold ?? 0.5;
  const temporal = clamp01(options.temporal ?? 0.65);
  const feather = Math.max(0.001, options.feather ?? 0.12);
  const stableValues = new Float32Array(confidence.length);
  const shaped = new Float32Array(confidence.length);
  for (let index = 0; index < confidence.length; index += 1) {
    const stable = previous
      ? confidence[index] * (1 - temporal) + previous[index] * temporal
      : confidence[index];
    stableValues[index] = stable;
    shaped[index] = smoothstep(threshold - feather, threshold + feather, stable);
  }
  const blurred = boxBlur(shaped, width, height, Math.max(0, Math.round(options.edgeRadius ?? 1)));
  const alpha = new Uint8ClampedArray(confidence.length);
  for (let index = 0; index < alpha.length; index += 1) alpha[index] = Math.round(blurred[index] * 255);
  return { alpha, history: stableValues };
};

const alphaComposite = (foreground, background, alpha) => {
  if (foreground.length !== background.length || foreground.length % 4 !== 0) {
    throw new RangeError('RGBA inputs must have matching dimensions');
  }
  if (alpha.length * 4 !== foreground.length) throw new RangeError('Alpha dimensions do not match');
  const output = new Uint8ClampedArray(foreground.length);
  for (let pixel = 0; pixel < alpha.length; pixel += 1) {
    const opacity = alpha[pixel] / 255;
    const offset = pixel * 4;
    for (let channel = 0; channel < 3; channel += 1) {
      output[offset + channel] = Math.round(
        foreground[offset + channel] * opacity + background[offset + channel] * (1 - opacity),
      );
    }
    output[offset + 3] = 255;
  }
  return output;
};

class OneInFlightScheduler {
  constructor(dispatch) {
    this.dispatch = dispatch;
    this.activeTimestamp = null;
    this.latestSourceTimestamp = -Infinity;
    this.latestMaskTimestamp = -Infinity;
    this.droppedFrames = 0;
  }

  submit(timestamp, resource) {
    this.latestSourceTimestamp = Math.max(this.latestSourceTimestamp, timestamp);
    if (this.activeTimestamp !== null || timestamp <= this.latestMaskTimestamp) {
      resource?.close?.();
      this.droppedFrames += 1;
      return false;
    }
    this.activeTimestamp = timestamp;
    this.dispatch(timestamp, resource);
    return true;
  }

  complete(timestamp) {
    this.activeTimestamp = null;
    if (timestamp > this.latestSourceTimestamp || timestamp < this.latestMaskTimestamp) return false;
    this.latestMaskTimestamp = timestamp;
    return true;
  }

  reset() {
    this.activeTimestamp = null;
    this.latestSourceTimestamp = -Infinity;
    this.latestMaskTimestamp = -Infinity;
    this.droppedFrames = 0;
  }
}

class LatestFrameScheduler {
  constructor(dispatch) {
    this.dispatch = dispatch;
    this.active = null;
    this.pending = null;
    this.latestMaskTimestamp = -Infinity;
    this.droppedFrames = 0;
  }

  submit(timestamp, resource) {
    if (timestamp <= this.latestMaskTimestamp) {
      resource?.close?.();
      this.droppedFrames += 1;
      return false;
    }
    if (this.active === null) {
      this.start(timestamp, resource);
      return true;
    }
    this.pending?.resource?.close?.();
    if (this.pending !== null) this.droppedFrames += 1;
    this.pending = { timestamp, resource };
    return false;
  }

  complete(timestamp) {
    if (this.active?.timestamp !== timestamp) return null;
    const completed = this.active.resource;
    this.active = null;
    this.latestMaskTimestamp = timestamp;
    const pending = this.pending;
    this.pending = null;
    if (pending !== null) this.start(pending.timestamp, pending.resource);
    return completed;
  }

  start(timestamp, resource) {
    this.active = { timestamp, resource };
    this.dispatch(timestamp, resource);
  }

  reset() {
    this.active?.resource?.close?.();
    this.pending?.resource?.close?.();
    this.active = null;
    this.pending = null;
    this.latestMaskTimestamp = -Infinity;
    this.droppedFrames = 0;
  }
}

export {
  OneInFlightScheduler,
  LatestFrameScheduler,
  alphaComposite,
  refineConfidenceMask,
  smoothstep,
  temporalWeightForInterval,
};
