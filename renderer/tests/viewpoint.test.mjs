import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clampPosition,
  flySpeed,
  lookTarget,
  orbitPosition,
  translateViewpoint,
  zoomPosition,
} from '../src/viewpoint.mjs';

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

test('first-person look rotates the target without moving the camera', () => {
  const position = { x: 1, y: 1, z: 1 };
  const target = { x: 1, y: 1, z: -1 };
  const moved = lookTarget(position, target, -Math.PI / 2, 0.2);

  assert.ok(Math.abs(moved.x - (1 + 2 * Math.cos(0.2))) < 1e-10);
  assert.ok(Math.abs(moved.y - (1 + 2 * Math.sin(0.2))) < 1e-10);
  assert.ok(Math.abs(moved.z - 1) < 1e-10);
});

test('zoom changes distance monotonically', () => {
  const position = { x: 0, y: 0, z: -2 };
  const target = { x: 0, y: 0, z: 0 };

  assert.equal(zoomPosition(position, target, 0.5).z, -1);
  assert.equal(zoomPosition(position, target, 2).z, -4);
});

test('fly translation moves the camera and target together', () => {
  const moved = translateViewpoint(
    { x: 0, y: 1, z: -2 },
    { x: 0, y: 1, z: 0 },
    { x: 0.5, y: 0.25, z: 1 },
    bounds,
  );

  assert.deepEqual(moved, {
    position: { x: 0.5, y: 1.25, z: -1 },
    target: { x: 0.5, y: 1.25, z: 1 },
    clamped: false,
  });
});

test('fly translation preserves direction when the camera reaches a boundary', () => {
  const moved = translateViewpoint(
    { x: 0.8, y: 1, z: 0 },
    { x: 0.8, y: 1, z: -1 },
    { x: 1, y: 0, z: 0 },
    bounds,
  );

  assert.deepEqual(moved.position, { x: 1, y: 1, z: 0 });
  assert.deepEqual(moved.target, { x: 1, y: 1, z: -1 });
  assert.equal(moved.clamped, true);
});

test('fly speed follows scene scale and supports acceleration', () => {
  const normal = flySpeed(bounds);

  assert.ok(normal >= 0.5);
  assert.equal(flySpeed(bounds, true), normal * 3);
});
