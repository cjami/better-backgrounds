import assert from 'node:assert/strict';
import test from 'node:test';

import { clampPosition, orbitPosition, zoomPosition } from '../src/viewpoint.mjs';

const bounds = {
  minimum: { x: -1, y: 0, z: -3 },
  maximum: { x: 1, y: 2, z: 1 },
};

test('camera movement is clamped to the registered safe region', () => {
  assert.deepEqual(clampPosition({ x: -4, y: 1, z: 9 }, bounds), { x: -1, y: 1, z: 1 });
});

test('orbit preserves distance from the target', () => {
  const position = { x: 0, y: 1, z: -2 };
  const target = { x: 0, y: 1, z: 0 };
  const moved = orbitPosition(position, target, Math.PI / 4, 0.2);

  assert.ok(Math.abs(Math.hypot(moved.x, moved.y - 1, moved.z) - 2) < 1e-10);
});

test('zoom changes distance monotonically', () => {
  const position = { x: 0, y: 0, z: -2 };
  const target = { x: 0, y: 0, z: 0 };

  assert.equal(zoomPosition(position, target, 0.5).z, -1);
  assert.equal(zoomPosition(position, target, 2).z, -4);
});
