import { copyFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

import { nodeResolve } from '@rollup/plugin-node-resolve';

const output = (file, name) => ({
  file: `src/better_backgrounds/desktop/assets/${file}`,
  format: 'iife',
  name,
  generatedCode: 'es2015',
  inlineDynamicImports: true,
});

const copyMediaPipeWasm = () => ({
  name: 'copy-mediapipe-wasm',
  async writeBundle() {
    const source = 'node_modules/@mediapipe/tasks-vision/wasm';
    const destination = 'src/better_backgrounds/desktop/assets/matting/wasm';
    await mkdir(destination, { recursive: true });
    for (const name of [
      'vision_wasm_internal.js',
      'vision_wasm_internal.wasm',
      'vision_wasm_module_internal.js',
      'vision_wasm_module_internal.wasm',
      'vision_wasm_nosimd_internal.js',
      'vision_wasm_nosimd_internal.wasm',
    ]) {
      await copyFile(path.join(source, name), path.join(destination, name));
    }
  },
});

export default [
  {
    input: 'renderer/src/main.mjs',
    output: output('renderer.js', 'BetterBackgroundsRenderer'),
    plugins: [nodeResolve({ browser: true })],
  },
  {
    input: 'renderer/src/live.mjs',
    output: output('live-renderer.js', 'BetterBackgroundsLive'),
    plugins: [nodeResolve({ browser: true })],
  },
  {
    input: 'renderer/src/matting-worker.mjs',
    output: output('matting-worker.js', 'BetterBackgroundsMattingWorker'),
    plugins: [nodeResolve({ browser: true }), copyMediaPipeWasm()],
  },
];
