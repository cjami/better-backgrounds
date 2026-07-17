import assert from 'node:assert/strict';
import test from 'node:test';

import {
  OneInFlightScheduler,
  LatestFrameScheduler,
  alphaComposite,
  refineConfidenceMask,
  temporalWeightForInterval,
} from '../src/matting.mjs';

test('temporal refinement retains confidence rather than shaped alpha as history', () => {
  const first = refineConfidenceMask(new Float32Array([0.75]), null, 1, 1, {
    threshold: 0.5,
    temporal: 0.5,
    feather: 0.5,
    edgeRadius: 0,
  });
  const second = refineConfidenceMask(new Float32Array([0.25]), first.history, 1, 1, {
    threshold: 0.5,
    temporal: 0.5,
    feather: 0.5,
    edgeRadius: 0,
  });

  assert.equal(first.history[0], 0.75);
  assert.equal(second.alpha[0], 128);
});

test('latest-frame scheduling keeps one pending frame and preserves mask pairing', () => {
  const dispatched = [];
  const closed = [];
  const scheduler = new LatestFrameScheduler((timestamp) => dispatched.push(timestamp));
  const frame = (timestamp) => ({ timestamp, close: () => closed.push(timestamp) });

  scheduler.submit(10, frame(10));
  scheduler.submit(11, frame(11));
  scheduler.submit(12, frame(12));

  assert.deepEqual(dispatched, [10]);
  assert.deepEqual(closed, [11]);
  assert.equal(scheduler.complete(10).timestamp, 10);
  assert.deepEqual(dispatched, [10, 12]);
  assert.equal(scheduler.complete(12).timestamp, 12);
  assert.equal(scheduler.droppedFrames, 1);
});

test('temporal history decays by elapsed source time', () => {
  assert.equal(temporalWeightForInterval(0.5, 1000 / 30), 0.5);
  assert.equal(temporalWeightForInterval(0.5, 2000 / 30), 0.25);
});

test('alpha compositing selects exact foreground and background at mask extremes', () => {
  const foreground = new Uint8ClampedArray([200, 100, 50, 255, 20, 30, 40, 255]);
  const background = new Uint8ClampedArray([1, 2, 3, 255, 210, 220, 230, 255]);

  assert.deepEqual(
    alphaComposite(foreground, background, new Uint8ClampedArray([255, 0])),
    new Uint8ClampedArray([200, 100, 50, 255, 210, 220, 230, 255]),
  );
});

test('one-in-flight scheduling drops queued work and rejects future masks', () => {
  const dispatched = [];
  let closed = 0;
  const scheduler = new OneInFlightScheduler((timestamp) => dispatched.push(timestamp));

  assert.equal(scheduler.submit(10, {}), true);
  assert.equal(scheduler.submit(11, { close: () => { closed += 1; } }), false);
  assert.equal(scheduler.complete(12), false);
  assert.equal(scheduler.submit(13, {}), true);

  assert.deepEqual(dispatched, [10, 13]);
  assert.equal(closed, 1);
  assert.equal(scheduler.droppedFrames, 1);
});
