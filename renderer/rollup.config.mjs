import { nodeResolve } from '@rollup/plugin-node-resolve';

export default {
  input: 'renderer/src/main.mjs',
  output: {
    file: 'src/better_backgrounds/desktop/assets/renderer.js',
    format: 'iife',
    name: 'BetterBackgroundsRenderer',
    generatedCode: 'es2015',
    inlineDynamicImports: true,
  },
  plugins: [nodeResolve({ browser: true })],
};
