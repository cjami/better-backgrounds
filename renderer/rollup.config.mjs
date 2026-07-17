import { nodeResolve } from '@rollup/plugin-node-resolve';

const output = (file, name) => ({
  file: `src/better_backgrounds/desktop/assets/${file}`,
  format: 'iife',
  name,
  generatedCode: 'es2015',
  inlineDynamicImports: true,
});

export default {
  input: 'renderer/src/main.mjs',
  output: output('renderer.js', 'BetterBackgroundsRenderer'),
  plugins: [nodeResolve({ browser: true })],
};
