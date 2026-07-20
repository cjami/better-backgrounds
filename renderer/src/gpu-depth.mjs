import * as pc from 'playcanvas';

const SPLAT_PREPASS_ALPHA_THRESHOLD = 0.1;

const GPU_COC_GLSL = `
#include "screenDepthPS"
varying vec2 uv0;
uniform vec3 params;

float repairSplatDepth(vec2 uv) {
  float centerDepth = getLinearScreenDepth(clamp(uv, vec2(0.0), vec2(1.0)));
  float farGate = camera_params.y * 0.9999;
  if (centerDepth < farGate) return centerDepth;

  vec2 texelSize = 1.0 / vec2(textureSize(uSceneDepthMap, 0));
  float nearest = camera_params.y;
  float farthest = 0.0;
  float depthSum = 0.0;
  int coverage = 0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      float depth = getLinearScreenDepth(clamp(
        uv + vec2(float(x), float(y)) * texelSize,
        vec2(0.0),
        vec2(1.0)
      ));
      if (depth < farGate) {
        nearest = min(nearest, depth);
        farthest = max(farthest, depth);
        depthSum += depth;
        coverage++;
      }
    }
  }

  float maximumStep = max(0.25, nearest * 0.08);
  if (coverage >= 5 && farthest - nearest <= maximumStep) {
    return depthSum / float(coverage);
  }
  return centerDepth;
}

vec2 circleOfConfusion(float depth) {
  float focusDistance = params.x;
  float focusRange = params.y;
  float invRange = params.z;
  float farEdge = focusDistance + focusRange * 0.5;
  float farCoc = smoothstep(0.0, 1.0, clamp((depth - farEdge) * invRange, 0.0, 1.0));

#ifdef NEAR_BLUR
  float nearEdge = focusDistance - focusRange * 0.5;
  float nearCoc = smoothstep(0.0, 1.0, clamp((nearEdge - depth) * invRange, 0.0, 1.0));
#else
  float nearCoc = 0.0;
#endif

  return vec2(farCoc, nearCoc);
}

float cocKernelWeight(int offset) {
  int distance = abs(offset);
  if (distance == 0) return 20.0;
  if (distance == 1) return 15.0;
  if (distance == 2) return 6.0;
  return 1.0;
}

vec2 smoothSplatCircleOfConfusion(vec2 uv) {
  vec2 texelSize = 1.0 / vec2(textureSize(uSceneDepthMap, 0));
  vec2 cocSum = vec2(0.0);
  float weightSum = 0.0;
  for (int y = -3; y <= 3; y++) {
    for (int x = -3; x <= 3; x++) {
      vec2 sampleUv = clamp(
        uv + vec2(float(x), float(y)) * texelSize,
        vec2(0.0),
        vec2(1.0)
      );
      float depth = getLinearScreenDepth(sampleUv);
      if (x == 0 && y == 0) depth = repairSplatDepth(sampleUv);
      float weight = cocKernelWeight(x) * cocKernelWeight(y);
      cocSum += circleOfConfusion(depth) * weight;
      weightSum += weight;
    }
  }
  return cocSum / weightSum;
}

void main() {
  gl_FragColor = vec4(smoothSplatCircleOfConfusion(uv0), 0.0, 0.0);
}
`;

const GPU_COC_WGSL = `
#include "screenDepthPS"
varying uv0: vec2f;
uniform params: vec3f;

fn repairSplatDepth(uv: vec2f) -> f32 {
  let centerDepth = getLinearScreenDepth(clamp(uv, vec2f(0.0), vec2f(1.0)));
  let farGate = uniform.camera_params.y * 0.9999;
  if (centerDepth < farGate) {
    return centerDepth;
  }

  let texelSize = 1.0 / vec2f(textureDimensions(uSceneDepthMap, 0));
  var nearest = uniform.camera_params.y;
  var farthest = 0.0;
  var depthSum = 0.0;
  var coverage = 0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      let depth = getLinearScreenDepth(clamp(
        uv + vec2f(f32(x), f32(y)) * texelSize,
        vec2f(0.0),
        vec2f(1.0)
      ));
      if (depth < farGate) {
        nearest = min(nearest, depth);
        farthest = max(farthest, depth);
        depthSum = depthSum + depth;
        coverage = coverage + 1;
      }
    }
  }

  let maximumStep = max(0.25, nearest * 0.08);
  if (coverage >= 5 && farthest - nearest <= maximumStep) {
    return depthSum / f32(coverage);
  }
  return centerDepth;
}

fn circleOfConfusion(depth: f32) -> vec2f {
  let focusDistance = uniform.params.x;
  let focusRange = uniform.params.y;
  let invRange = uniform.params.z;
  let farEdge = focusDistance + focusRange * 0.5;
  let farCoc = smoothstep(0.0, 1.0, clamp((depth - farEdge) * invRange, 0.0, 1.0));

#ifdef NEAR_BLUR
  let nearEdge = focusDistance - focusRange * 0.5;
  let nearCoc = smoothstep(0.0, 1.0, clamp((nearEdge - depth) * invRange, 0.0, 1.0));
#else
  let nearCoc = 0.0;
#endif

  return vec2f(farCoc, nearCoc);
}

fn cocKernelWeight(offset: i32) -> f32 {
  let distance = abs(offset);
  if (distance == 0) {
    return 20.0;
  }
  if (distance == 1) {
    return 15.0;
  }
  if (distance == 2) {
    return 6.0;
  }
  return 1.0;
}

fn smoothSplatCircleOfConfusion(uv: vec2f) -> vec2f {
  let texelSize = 1.0 / vec2f(textureDimensions(uSceneDepthMap, 0));
  var cocSum = vec2f(0.0);
  var weightSum = 0.0;
  for (var y = -3; y <= 3; y = y + 1) {
    for (var x = -3; x <= 3; x = x + 1) {
      let sampleUv = clamp(
        uv + vec2f(f32(x), f32(y)) * texelSize,
        vec2f(0.0),
        vec2f(1.0)
      );
      var depth = getLinearScreenDepth(sampleUv);
      if (x == 0 && y == 0) {
        depth = repairSplatDepth(sampleUv);
      }
      let weight = cocKernelWeight(x) * cocKernelWeight(y);
      cocSum = cocSum + circleOfConfusion(depth) * weight;
      weightSum = weightSum + weight;
    }
  }
  return cocSum / weightSum;
}

@fragment
fn fragmentMain(input: FragmentInput) -> FragmentOutput {
  var output: FragmentOutput;
  output.color = vec4f(smoothSplatCircleOfConfusion(input.uv0), 0.0, 0.0);
  return output;
}
`;

const smoothCircleOfConfusion = (depth, focusDistance, focusRange) => {
  const safeRange = Math.max(Number.EPSILON, focusRange);
  const focusDelta = Math.abs(depth - focusDistance) - safeRange / 2;
  const linear = Math.min(1, Math.max(0, focusDelta / safeRange));
  return linear * linear * (3 - 2 * linear);
};

const repairSplatDepth = (centerDepth, neighborhood, farDepth) => {
  const farGate = farDepth * 0.9999;
  if (centerDepth < farGate) return centerDepth;
  const covered = neighborhood.filter((depth) => Number.isFinite(depth) && depth < farGate);
  if (covered.length < 5) return centerDepth;
  const nearest = Math.min(...covered);
  const farthest = Math.max(...covered);
  if (farthest - nearest > Math.max(0.25, nearest * 0.08)) return centerDepth;
  return covered.reduce((sum, depth) => sum + depth, 0) / covered.length;
};

const splatManagerForLayer = (cameraFrame, layer) => {
  const director = cameraFrame.app?.renderer?.gsplatDirector;
  const camera = cameraFrame.cameraComponent?.camera;
  const cameraData = director?.camerasMap?.get(camera);
  return cameraData?.layersMap?.get(layer)?.gsplatManager ?? null;
};

const renderSplatPrepass = (materials, render) => {
  const states = materials.map((material) => ({
    material,
    blendState: material.blendState.clone(),
    depthState: material.depthState.clone(),
    depthWrite: material.depthWrite,
  }));
  try {
    for (const { material } of states) {
      material.blendState = pc.BlendState.NOBLEND;
      material.depthWrite = true;
    }
    render();
  } finally {
    for (const { material, blendState, depthState, depthWrite } of states) {
      material.blendState = blendState;
      material.depthState = depthState;
      material.depthWrite = depthWrite;
    }
  }
};

const executeSplatPrepass = (cameraFrame, prePass) => {
  const { renderer, scene, renderTarget } = prePass;
  const camera = prePass.camera.camera;
  const composition = scene.layers;
  const layers = composition.layerList;
  const renderedManagers = new Set();
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index];
    if (layer.id === pc.LAYERID_DEPTH) break;
    if (!layer.enabled || !composition.subLayerEnabled[index]) continue;
    if (!layer.camerasSet.has(camera)) continue;

    const transparent = composition.subLayerList[index];
    const culled = layer.getCulledInstances(camera);
    const source = transparent ? culled.transparent : culled.opaque;
    const meshInstances = source.filter((meshInstance) => meshInstance.material?.depthWrite);
    const candidate = splatManagerForLayer(cameraFrame, layer);
    const manager = candidate && !renderedManagers.has(candidate) ? candidate : null;
    const splatMeshInstance = manager?.renderer?.meshInstance;
    if (splatMeshInstance && !meshInstances.includes(splatMeshInstance)) {
      meshInstances.push(splatMeshInstance);
      renderedManagers.add(manager);
    }
    if (!meshInstances.length) continue;

    const materials = manager?.material ? [manager.material] : [];
    renderSplatPrepass(materials, () => {
      renderer.renderForwardLayer(
        camera,
        renderTarget,
        null,
        undefined,
        pc.SHADER_PREPASS,
        prePass.viewBindGroups,
        { meshInstances },
      );
    });
  }
};

const installSplatPrepass = (cameraFrame) => {
  const prePass = cameraFrame.renderPassCamera?.prePass;
  if (!prePass) return false;
  if (prePass.betterBackgroundsGpuDepth) return true;
  prePass.execute = () => executeSplatPrepass(cameraFrame, prePass);
  prePass.betterBackgroundsGpuDepth = true;
  return true;
};

const installCircleOfConfusion = (cameraFrame) => {
  const cocPass = cameraFrame.renderPassCamera?.dofPass?.cocPass;
  const device = cameraFrame.app?.graphicsDevice;
  const cameraComponent = cocPass?.cameraComponent;
  if (!cocPass || !device || !cameraComponent) return false;
  if (cocPass.betterBackgroundsGpuCoc) return true;
  const language = device.deviceType === pc.DEVICETYPE_WEBGPU
    ? pc.SHADERLANGUAGE_WGSL
    : pc.SHADERLANGUAGE_GLSL;
  const chunks = pc.ShaderChunks.get(device, language);
  const originalChunk = chunks.get('cocPS');
  const defines = new Map();
  if (cameraFrame.dof.nearBlur) defines.set('NEAR_BLUR', '');
  pc.ShaderUtils.addScreenDepthChunkDefines(cameraComponent.shaderParams, defines);

  try {
    chunks.set('cocPS', language === pc.SHADERLANGUAGE_WGSL ? GPU_COC_WGSL : GPU_COC_GLSL);
    cocPass.shader = pc.ShaderUtils.createShader(device, {
      uniqueName: `BetterBackgroundsGpuCoc-${language}-${cameraFrame.dof.nearBlur}`,
      attributes: { aPosition: pc.SEMANTIC_POSITION },
      vertexChunk: 'quadVS',
      fragmentChunk: 'cocPS',
      fragmentDefines: defines,
    });
    cocPass.betterBackgroundsGpuCoc = true;
  } finally {
    chunks.set('cocPS', originalChunk);
  }
  return true;
};

const configureGpuDepthPipeline = (cameraFrame) => (
  installSplatPrepass(cameraFrame) && installCircleOfConfusion(cameraFrame)
);

const configureGpuSplatDepth = (scene) => {
  scene.gsplat.alphaClip = SPLAT_PREPASS_ALPHA_THRESHOLD;
};

export {
  SPLAT_PREPASS_ALPHA_THRESHOLD,
  configureGpuDepthPipeline,
  configureGpuSplatDepth,
  executeSplatPrepass,
  installSplatPrepass,
  renderSplatPrepass,
  repairSplatDepth,
  smoothCircleOfConfusion,
};
