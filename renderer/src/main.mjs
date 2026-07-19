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

const MAX_SCENE_CAPTURE_ATTEMPTS = 12;
const STREAMED_LOD_UPDATE_ANGLE = 2;
const STREAMED_LOD_UPDATE_DISTANCE = 0.1;
const SUBJECT_FOCUS_DISTANCE = 1.5;
const MIN_SUBJECT_FOCUS_RANGE = 0.8;
const MAX_SUBJECT_FOCUS_RANGE = 8;
const MAX_DOF_BLUR_RADIUS = 8;
const REFERENCE_FIELD_OF_VIEW = 42;
const MIN_PERSPECTIVE_SCALE = 0.75;
const MAX_PERSPECTIVE_SCALE = 1.25;
const DEPTH_PROXY_COLUMNS = 384;
const DEPTH_PROXY_SAMPLE_RANK = 3;
const GENERIC_DEPTH_PROXY_SAMPLE_RANK = 2;
const DEPTH_PROXY_MAX_RELATIVE_STEP = 0.2;
const DEPTH_PROXY_MIN_STEP_METRES = 0.5;
const DEPTH_PROXY_MAX_CENTERS = 1_000_000;
const DEPTH_PROXY_REFRESH_DELAY_MS = 120;
const DEPTH_PROXY_ALPHA_THRESHOLD = 1 / 255;
const DEPTH_PROXY_MIN_FOOTPRINT_CELLS = 0.65;
const DEPTH_PROXY_MAX_FOOTPRINT_CELLS = 6;
const DEPTH_PROXY_HOLE_FILL_PASSES = 2;
const DEPTH_PROXY_HOLE_FILL_NEIGHBORS = 5;
const GENERIC_DEPTH_PROXY_NEAR_QUANTILE = 0.01;
const GENERIC_MIN_FOCUS_RANGE = 0.001;

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

const buildViewDepthProxyGeometry = (
  samples,
  projectCenter,
  unprojectDepth,
  viewportWidth,
  viewportHeight,
  requestedColumns = DEPTH_PROXY_COLUMNS,
) => {
  const centers = samples?.centers ?? samples;
  const radii = samples?.radii ?? null;
  const opacities = samples?.opacities ?? null;
  if (!centers || centers.length < 9 || viewportWidth < 2 || viewportHeight < 2) return null;
  const columns = Math.max(2, Math.min(viewportWidth, Math.trunc(requestedColumns)));
  const rows = Math.max(2, Math.min(
    viewportHeight,
    Math.round((viewportHeight / viewportWidth) * columns),
  ));
  const depths = new Float32Array(columns * rows);
  const valid = new Uint8Array(columns * rows);
  const positions = new Float32Array(columns * rows * 3);
  const rankedDepths = Array.from(
    { length: GENERIC_DEPTH_PROXY_SAMPLE_RANK },
    () => new Float32Array(columns * rows).fill(Number.POSITIVE_INFINITY),
  );
  const insertDepth = (column, row, depth) => {
    if (column < 0 || column >= columns || row < 0 || row >= rows) return;
    const targetIndex = row * columns + column;
    for (let rank = 0; rank < GENERIC_DEPTH_PROXY_SAMPLE_RANK; rank += 1) {
      if (depth >= rankedDepths[rank][targetIndex]) continue;
      for (let move = GENERIC_DEPTH_PROXY_SAMPLE_RANK - 1; move > rank; move -= 1) {
        rankedDepths[move][targetIndex] = rankedDepths[move - 1][targetIndex];
      }
      rankedDepths[rank][targetIndex] = depth;
      break;
    }
  };
  const centerCount = Math.floor(centers.length / 3);
  const stride = Math.max(1, Math.ceil(centerCount / DEPTH_PROXY_MAX_CENTERS));
  for (let centerIndex = 0; centerIndex < centerCount; centerIndex += stride) {
    const offset = centerIndex * 3;
    const projected = projectCenter(
      centers[offset],
      centers[offset + 1],
      centers[offset + 2],
      radii?.[centerIndex] ?? 0,
    );
    if (
      !projected
      || !Number.isFinite(projected.x)
      || !Number.isFinite(projected.y)
      || !Number.isFinite(projected.depth)
      || projected.depth <= 0
    ) continue;
    const centerColumn = projected.x * columns / viewportWidth;
    const centerRow = projected.y * rows / viewportHeight;
    const opacity = Math.min(1, Math.max(0, opacities?.[centerIndex] ?? 1));
    const alphaScale = Math.max(
      1,
      opacity > DEPTH_PROXY_ALPHA_THRESHOLD
        ? Math.sqrt(2 * Math.log(opacity / DEPTH_PROXY_ALPHA_THRESHOLD))
        : 0,
    );
    const projectedRadius = Math.max(0, projected.radius ?? 0) * alphaScale;
    const radiusColumns = projectedRadius > 0
      ? Math.min(
        DEPTH_PROXY_MAX_FOOTPRINT_CELLS,
        Math.max(DEPTH_PROXY_MIN_FOOTPRINT_CELLS, projectedRadius * columns / viewportWidth),
      )
      : 0;
    const radiusRows = projectedRadius > 0
      ? Math.min(
        DEPTH_PROXY_MAX_FOOTPRINT_CELLS,
        Math.max(DEPTH_PROXY_MIN_FOOTPRINT_CELLS, projectedRadius * rows / viewportHeight),
      )
      : 0;
    if (radiusColumns === 0 || radiusRows === 0) {
      insertDepth(Math.floor(centerColumn), Math.floor(centerRow), projected.depth);
      continue;
    }
    const minimumColumn = Math.floor(centerColumn - radiusColumns);
    const maximumColumn = Math.floor(centerColumn + radiusColumns);
    const minimumRow = Math.floor(centerRow - radiusRows);
    const maximumRow = Math.floor(centerRow + radiusRows);
    for (let row = minimumRow; row <= maximumRow; row += 1) {
      for (let column = minimumColumn; column <= maximumColumn; column += 1) {
        const dx = (column + 0.5 - centerColumn) / radiusColumns;
        const dy = (row + 0.5 - centerRow) / radiusRows;
        if (dx * dx + dy * dy <= 1) insertDepth(column, row, projected.depth);
      }
    }
  }
  const surfaceDepths = new Float32Array(columns * rows).fill(Number.POSITIVE_INFINITY);
  for (let index = 0; index < surfaceDepths.length; index += 1) {
    const rankedDepth = rankedDepths[GENERIC_DEPTH_PROXY_SAMPLE_RANK - 1][index];
    surfaceDepths[index] = Number.isFinite(rankedDepth) ? rankedDepth : rankedDepths[0][index];
  }
  for (let pass = 0; pass < DEPTH_PROXY_HOLE_FILL_PASSES; pass += 1) {
    const filled = surfaceDepths.slice();
    let changed = false;
    for (let row = 1; row < rows - 1; row += 1) {
      for (let column = 1; column < columns - 1; column += 1) {
        const targetIndex = row * columns + column;
        if (Number.isFinite(surfaceDepths[targetIndex])) continue;
        const neighbors = [];
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0) continue;
            const depth = surfaceDepths[(row + dy) * columns + column + dx];
            if (Number.isFinite(depth)) neighbors.push(depth);
          }
        }
        if (neighbors.length < DEPTH_PROXY_HOLE_FILL_NEIGHBORS) continue;
        neighbors.sort((left, right) => left - right);
        const nearest = neighbors[0];
        const farthest = neighbors[neighbors.length - 1];
        const maximumStep = Math.max(
          DEPTH_PROXY_MIN_STEP_METRES,
          nearest * DEPTH_PROXY_MAX_RELATIVE_STEP,
        );
        if (farthest - nearest > maximumStep) continue;
        filled[targetIndex] = neighbors[Math.floor(neighbors.length / 2)];
        changed = true;
      }
    }
    surfaceDepths.set(filled);
    if (!changed) break;
  }
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const targetIndex = row * columns + column;
      const depth = surfaceDepths[targetIndex];
      if (!Number.isFinite(depth)) continue;
      const point = unprojectDepth(
        (column + 0.5) * viewportWidth / columns,
        (row + 0.5) * viewportHeight / rows,
        depth,
      );
      if (!point) continue;
      const positionOffset = targetIndex * 3;
      positions[positionOffset] = point.x;
      positions[positionOffset + 1] = point.y;
      positions[positionOffset + 2] = point.z;
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
  const finiteDepths = Array.from(surfaceDepths)
    .filter((depth) => Number.isFinite(depth))
    .sort((left, right) => left - right);
  const nearDepth = finiteDepths[
    Math.floor((finiteDepths.length - 1) * GENERIC_DEPTH_PROXY_NEAR_QUANTILE)
  ];
  return { columns, rows, positions, indices: new Uint32Array(indices), nearDepth };
};

const collectDepthProxyCenters = (
  resource,
  maximumCenters = DEPTH_PROXY_MAX_CENTERS,
) => {
  if (!resource || maximumCenters < 1) return null;
  const resources = [];
  if (resource.centers) resources.push(resource);
  const octree = resource.octree;
  if (octree) {
    for (const loaded of octree.fileResources?.values?.() ?? []) {
      if (loaded?.centers) resources.push(loaded);
    }
    if (octree.environmentResource?.centers) resources.push(octree.environmentResource);
  }
  const unique = [...new Set(resources)];
  const arrays = unique.map((item) => item.centers).filter((centers) => centers?.length >= 3);
  if (!arrays.length) return null;
  const totalCenters = arrays.reduce((total, centers) => total + Math.floor(centers.length / 3), 0);
  if (arrays.length === 1 && totalCenters <= maximumCenters) return arrays[0];

  const sampled = new Float32Array(Math.min(totalCenters, maximumCenters) * 3);
  let outputCenter = 0;
  for (const centers of arrays) {
    const centerCount = Math.floor(centers.length / 3);
    const quota = Math.max(1, Math.floor(maximumCenters * centerCount / totalCenters));
    const stride = Math.max(1, centerCount / quota);
    for (let sample = 0; sample < quota && outputCenter < maximumCenters; sample += 1) {
      const sourceCenter = Math.min(centerCount - 1, Math.floor(sample * stride));
      sampled.set(
        centers.subarray(sourceCenter * 3, sourceCenter * 3 + 3),
        outputCenter * 3,
      );
      outputCenter += 1;
    }
  }
  return outputCenter === sampled.length / 3
    ? sampled
    : sampled.slice(0, outputCenter * 3);
};

const sogDepthEvidenceCache = new WeakMap();

const textureSourceBytes = (texture) => {
  const source = texture?.getSource?.() ?? texture?._levels?.[0];
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  }
  if (ArrayBuffer.isView(source?.data)) {
    return new Uint8Array(source.data.buffer, source.data.byteOffset, source.data.byteLength);
  }
  if (
    typeof OffscreenCanvas !== 'undefined'
    && source
    && Number.isFinite(texture.width)
    && Number.isFinite(texture.height)
  ) {
    try {
      const canvas = new OffscreenCanvas(texture.width, texture.height);
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.drawImage(source, 0, 0);
      return context.getImageData(0, 0, texture.width, texture.height).data;
    } catch (_error) {
      return null;
    }
  }
  return null;
};

const sogDepthEvidence = (data) => {
  if (!data?.meta?.scales || !data.scales) return null;
  if (sogDepthEvidenceCache.has(data)) return sogDepthEvidenceCache.get(data);
  const scales = textureSourceBytes(data.scales);
  if (!scales) {
    sogDepthEvidenceCache.set(data, null);
    return null;
  }
  const evidence = {
    scales,
    colors: textureSourceBytes(data.sh0),
    meta: data.meta,
  };
  sogDepthEvidenceCache.set(data, evidence);
  return evidence;
};

const depthProxyRadiusSource = (resource) => {
  const data = resource?.gsplatData;
  const scaleX = data?.getProp?.('scale_0');
  const scaleY = data?.getProp?.('scale_1');
  const scaleZ = data?.getProp?.('scale_2');
  const opacity = data?.getProp?.('opacity');
  if (scaleX && scaleY && scaleZ) return { scaleX, scaleY, scaleZ, opacity };
  const sog = sogDepthEvidence(data);
  if (sog) return { sog };
  const codebook = data?.meta?.scales?.codebook;
  const finite = Array.isArray(codebook)
    ? codebook.filter((value) => Number.isFinite(value)).sort((left, right) => left - right)
    : [];
  const logRadius = finite.length ? finite[Math.floor(finite.length * 0.75)] : null;
  return { logRadius };
};

const collectDepthProxySamples = (
  resource,
  maximumCenters = DEPTH_PROXY_MAX_CENTERS,
) => {
  if (!resource || maximumCenters < 1) return null;
  const resources = [];
  if (resource.centers) resources.push(resource);
  const octree = resource.octree;
  if (octree) {
    for (const loaded of octree.fileResources?.values?.() ?? []) {
      if (loaded?.centers) resources.push(loaded);
    }
    if (octree.environmentResource?.centers) resources.push(octree.environmentResource);
  }
  const unique = [...new Set(resources)];
  const descriptors = unique
    .filter((item) => item.centers?.length >= 3)
    .map((item) => ({ item, source: depthProxyRadiusSource(item) }));
  if (!descriptors.length) return null;
  const totalCenters = descriptors.reduce(
    (total, descriptor) => total + Math.floor(descriptor.item.centers.length / 3),
    0,
  );
  const sampleCount = Math.min(totalCenters, maximumCenters);
  const centers = new Float32Array(sampleCount * 3);
  const radii = new Float32Array(sampleCount);
  const opacities = new Float32Array(sampleCount).fill(1);
  let outputCenter = 0;
  for (const descriptor of descriptors) {
    const sourceCenters = descriptor.item.centers;
    const centerCount = Math.floor(sourceCenters.length / 3);
    const quota = Math.max(1, Math.floor(maximumCenters * centerCount / totalCenters));
    const stride = Math.max(1, centerCount / quota);
    for (let sample = 0; sample < quota && outputCenter < sampleCount; sample += 1) {
      const sourceCenter = Math.min(centerCount - 1, Math.floor(sample * stride));
      centers.set(
        sourceCenters.subarray(sourceCenter * 3, sourceCenter * 3 + 3),
        outputCenter * 3,
      );
      const source = descriptor.source;
      if (source.scaleX) {
        const logScales = [
          source.scaleX[sourceCenter],
          source.scaleY[sourceCenter],
          source.scaleZ[sourceCenter],
        ].sort((left, right) => right - left);
        radii[outputCenter] = Math.exp((logScales[0] + logScales[1]) / 2);
        const logit = source.opacity?.[sourceCenter];
        if (Number.isFinite(logit)) opacities[outputCenter] = 1 / (1 + Math.exp(-logit));
      } else if (source.sog) {
        const byteOffset = sourceCenter * 4;
        const scaleBytes = source.sog.scales;
        const meta = source.sog.meta;
        const logScales = meta.version === 2
          ? [
            meta.scales.codebook[scaleBytes[byteOffset]],
            meta.scales.codebook[scaleBytes[byteOffset + 1]],
            meta.scales.codebook[scaleBytes[byteOffset + 2]],
          ]
          : [0, 1, 2].map((axis) => (
            meta.scales.mins[axis]
            + (meta.scales.maxs[axis] - meta.scales.mins[axis])
              * scaleBytes[byteOffset + axis] / 255
          ));
        logScales.sort((left, right) => right - left);
        radii[outputCenter] = Math.exp((logScales[0] + logScales[1]) / 2);
        const colorBytes = source.sog.colors;
        if (colorBytes) {
          if (meta.version === 2) {
            opacities[outputCenter] = colorBytes[byteOffset + 3] / 255;
          } else {
            const logit = meta.sh0.mins[3]
              + (meta.sh0.maxs[3] - meta.sh0.mins[3]) * colorBytes[byteOffset + 3] / 255;
            opacities[outputCenter] = 1 / (1 + Math.exp(-logit));
          }
        }
      } else if (Number.isFinite(source.logRadius)) {
        radii[outputCenter] = Math.exp(source.logRadius);
      }
      outputCenter += 1;
    }
  }
  if (outputCenter === sampleCount) return { centers, radii, opacities };
  return {
    centers: centers.slice(0, outputCenter * 3),
    radii: radii.slice(0, outputCenter),
    opacities: opacities.slice(0, outputCenter),
  };
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
    this.sceneEntity = null;
    this.sceneAsset = null;
    this.sceneResource = null;
    this.depthProxyEntity = null;
    this.depthProxyMesh = null;
    this.depthProxyMaterial = null;
    this.depthProxyUsesSceneTransform = false;
    this.depthProxyNearDepth = null;
    this.gsplatData = null;
    this.depthProxyViewpointKey = '';
    this.depthProxyRefreshTimer = null;
    this.depthProxyRefreshForce = false;
    this.sceneTransformKey = '';
    this.assetId = '';
    this.viewpoint = null;
    this.resetViewpoint = null;
    this.drag = null;
    this.flightKeys = new Set();
    this.flightDirty = false;
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
      this.cameraFrame.update();
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
        this.refreshViewDepthProxy(true);
        this.requestSceneFrame();
      });
      this.connectBridge();
      this.app.systems.gsplat.on('frame:ready', (_camera, _layer, ready, loadingCount) => {
        this.sceneSettled = ready && loadingCount === 0;
        if (this.sceneSettled) {
          this.refreshViewDepthProxy();
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
    this.refreshViewDepthProxy(true);
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
      this.sceneResource = loadedAsset.resource ?? null;
      this.gsplatData = this.sceneResource?.gsplatData ?? null;
      this.sceneCaptureAttempts = 0;
      this.app.root.addChild(entity);
      this.applySceneTransform();
      this.createDepthProxy(this.sceneResource);
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
    this.depthProxyUsesSceneTransform = false;
    this.depthProxyNearDepth = null;
    this.gsplatData = null;
    this.sceneResource = null;
    this.depthProxyViewpointKey = '';
    if (this.depthProxyRefreshTimer) clearTimeout(this.depthProxyRefreshTimer);
    this.depthProxyRefreshTimer = null;
    this.depthProxyRefreshForce = false;
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

  createDepthProxy(resource) {
    if (!this.app || !this.sceneEntity) return;
    const intrinsicGeometry = buildDepthProxyGeometry(resource?.gsplatData);
    this.snapshotRequiresSettlement = this.firstPersonNavigation || !intrinsicGeometry;
    if (intrinsicGeometry) {
      this.installDepthProxy(intrinsicGeometry, true);
      return;
    }
    if ((this.viewpoint?.depth_of_field?.blur_strength ?? 0) > 0) {
      this.rebuildViewDepthProxy(true);
    }
  }

  installDepthProxy(geometry, usesSceneTransform) {
    this.depthProxyEntity?.destroy();
    this.depthProxyMesh?.destroy();
    this.depthProxyMaterial?.destroy();

    const mesh = new pc.Mesh(this.app.graphicsDevice);
    mesh.setPositions(geometry.positions);
    mesh.setIndices(geometry.indices);
    mesh.update(pc.PRIMITIVE_TRIANGLES);

    const material = new pc.StandardMaterial();
    material.cull = pc.CULLFACE_NONE;
    material.depthWrite = true;
    material.update();

    const entity = new pc.Entity('splat-depth-proxy');
    const meshInstance = new pc.MeshInstance(mesh, material, entity);
    meshInstance.shaderPassMask &= ~(1 << pc.SHADER_FORWARD);
    entity.addComponent('render', { meshInstances: [meshInstance] });
    this.depthProxyEntity = entity;
    this.depthProxyMesh = mesh;
    this.depthProxyMaterial = material;
    this.depthProxyUsesSceneTransform = usesSceneTransform;
    this.depthProxyNearDepth = Number.isFinite(geometry.nearDepth) ? geometry.nearDepth : null;
    this.app.root.addChild(entity);
    if (usesSceneTransform) this.applySceneTransform(true);
    this.configureNormalFrame();
  }

  rebuildViewDepthProxy(force = false) {
    if (!this.sceneResource || !this.sceneEntity || !this.camera || !this.app) return;
    if ((this.viewpoint?.depth_of_field?.blur_strength ?? 0) <= 0) {
      if (this.depthProxyEntity && !this.depthProxyUsesSceneTransform) {
        this.depthProxyEntity.destroy();
        this.depthProxyMesh?.destroy();
        this.depthProxyMaterial?.destroy();
        this.depthProxyEntity = null;
        this.depthProxyMesh = null;
        this.depthProxyMaterial = null;
        this.depthProxyNearDepth = null;
        this.depthProxyViewpointKey = '';
        this.configureNormalFrame();
      }
      return;
    }
    const octree = this.sceneResource.octree;
    const sourceKey = octree
      ? `${[...(octree.fileResources?.keys?.() ?? [])].sort((a, b) => a - b).join(',')}:${Boolean(octree.environmentResource)}`
      : `single:${this.sceneResource.centersVersion ?? 0}`;
    const key = `${harmonizationViewpointKey(this.viewpoint)}:${sourceKey}`;
    if (!force && key === this.depthProxyViewpointKey) return;
    this.depthProxyViewpointKey = key;
    const samples = collectDepthProxySamples(this.sceneResource);
    if (!samples) return;
    const worldTransform = this.sceneEntity.getWorldTransform();
    const cameraPosition = this.camera.getPosition();
    const cameraForward = this.camera.forward;
    const local = new pc.Vec3();
    const world = new pc.Vec3();
    const delta = new pc.Vec3();
    const screen = new pc.Vec3();
    const radiusPoint = new pc.Vec3();
    const radiusScreen = new pc.Vec3();
    const output = new pc.Vec3();
    const device = this.app.graphicsDevice;
    const width = device.clientRect.width;
    const height = device.clientRect.height;
    const geometry = buildViewDepthProxyGeometry(
      samples,
      (x, y, z, radius) => {
        local.set(x, y, z);
        worldTransform.transformPoint(local, world);
        delta.sub2(world, cameraPosition);
        if (delta.dot(cameraForward) <= 0) return null;
        this.camera.camera.worldToScreen(world, screen);
        let screenRadius = 0;
        if (Number.isFinite(radius) && radius > 0) {
          radiusPoint.copy(this.camera.right).mulScalar(
            radius * this.viewpoint.scene_transform.scale,
          ).add(world);
          this.camera.camera.worldToScreen(radiusPoint, radiusScreen);
          screenRadius = Math.hypot(radiusScreen.x - screen.x, radiusScreen.y - screen.y);
        }
        return { x: screen.x, y: screen.y, depth: delta.length(), radius: screenRadius };
      },
      (x, y, depth) => this.camera.camera.screenToWorld(x, y, depth, output),
      width,
      height,
    );
    if (geometry) this.installDepthProxy(geometry, false);
  }

  refreshViewDepthProxy(force = false) {
    if (this.depthProxyUsesSceneTransform) return;
    if (this.cacheSceneFrames) {
      this.rebuildViewDepthProxy(force);
      return;
    }
    this.depthProxyRefreshForce ||= force;
    if (this.depthProxyRefreshTimer) return;
    this.depthProxyRefreshTimer = setTimeout(() => {
      this.depthProxyRefreshTimer = null;
      const refreshForce = this.depthProxyRefreshForce;
      this.depthProxyRefreshForce = false;
      this.rebuildViewDepthProxy(refreshForce);
    }, DEPTH_PROXY_REFRESH_DELAY_MS);
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
      this.refreshViewDepthProxy();
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
      canvas.focus();
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
      else if (this.firstPersonNavigation) this.look(dx, dy);
      else this.orbit(dx, dy);
    });
    canvas.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        if (!this.viewpoint) return;
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
    });
    const stopFlight = () => {
      this.flightKeys.clear();
      this.finishFlight();
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
    this.refreshViewDepthProxy();
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
    const enabled = Boolean(this.sceneEntity && this.depthProxyEntity);
    const genericNearDepth = !this.depthProxyUsesSceneTransform
      && Number.isFinite(this.depthProxyNearDepth)
      && this.depthProxyNearDepth > 0
      ? this.depthProxyNearDepth
      : null;
    const focusDistance = genericNearDepth === null
      ? SUBJECT_FOCUS_DISTANCE
      : Math.max(this.viewpoint.near_clip, this.camera.camera.nearClip);
    const focusRange = genericNearDepth === null
      ? effect.focusRange
      : Math.max(
        GENERIC_MIN_FOCUS_RANGE,
        genericNearDepth * effect.focusRange / MAX_SUBJECT_FOCUS_RANGE,
      );
    this.cameraFrame.debug = null;
    this.cameraFrame.dof.enabled = enabled && blurStrength > 0;
    this.cameraFrame.dof.nearBlur = true;
    this.cameraFrame.dof.highQuality = true;
    this.cameraFrame.dof.focusDistance = focusDistance;
    this.cameraFrame.dof.focusRange = focusRange;
    this.cameraFrame.dof.blurRadius = effect.blurRadius;
    this.cameraFrame.dof.blurRings = 4;
    this.cameraFrame.dof.blurRingPoints = 5;
    this.cameraFrame.rendering.sharpness = 0;
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

  applySceneTransform(force = false) {
    if (!this.sceneEntity || !this.viewpoint) return;
    const transform = this.viewpoint.scene_transform;
    const key = JSON.stringify(transform);
    if (!force && key === this.sceneTransformKey) return;
    const translation = transform.translation;
    const orientation = transform.orientation;
    const entities = [this.sceneEntity];
    if (this.depthProxyUsesSceneTransform) entities.push(this.depthProxyEntity);
    for (const entity of entities) {
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
  buildDepthProxyGeometry,
  buildViewDepthProxyGeometry,
  collectDepthProxyCenters,
  collectDepthProxySamples,
  configureGsplatStreaming,
  bytesToBase64,
  depthOfFieldForStrength,
  harmonizationViewpointKey,
  flipPixelRows,
  hasPixelVariation,
  isStreamedSceneUrl,
  start,
};
