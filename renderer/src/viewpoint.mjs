const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));

const clampPosition = (position, bounds) => ({
  x: clamp(position.x, bounds.minimum.x, bounds.maximum.x),
  y: clamp(position.y, bounds.minimum.y, bounds.maximum.y),
  z: clamp(position.z, bounds.minimum.z, bounds.maximum.z),
});

const orbitPosition = (position, target, deltaYaw, deltaPitch) => {
  const offset = {
    x: position.x - target.x,
    y: position.y - target.y,
    z: position.z - target.z,
  };
  const radius = Math.max(0.05, Math.hypot(offset.x, offset.y, offset.z));
  const yaw = Math.atan2(offset.x, offset.z) + deltaYaw;
  const currentPitch = Math.asin(clamp(offset.y / radius, -1, 1));
  const pitch = clamp(currentPitch + deltaPitch, -Math.PI * 0.47, Math.PI * 0.47);
  const planar = radius * Math.cos(pitch);
  return {
    x: target.x + planar * Math.sin(yaw),
    y: target.y + radius * Math.sin(pitch),
    z: target.z + planar * Math.cos(yaw),
  };
};

const lookTarget = (position, target, deltaYaw, deltaPitch) => {
  const offset = {
    x: target.x - position.x,
    y: target.y - position.y,
    z: target.z - position.z,
  };
  const radius = Math.max(0.05, Math.hypot(offset.x, offset.y, offset.z));
  const yaw = Math.atan2(offset.x, offset.z) + deltaYaw;
  const currentPitch = Math.asin(clamp(offset.y / radius, -1, 1));
  const pitch = clamp(currentPitch + deltaPitch, -Math.PI * 0.47, Math.PI * 0.47);
  const planar = radius * Math.cos(pitch);
  return {
    x: position.x + planar * Math.sin(yaw),
    y: position.y + radius * Math.sin(pitch),
    z: position.z + planar * Math.cos(yaw),
  };
};

const zoomPosition = (position, target, scale) => ({
  x: target.x + (position.x - target.x) * scale,
  y: target.y + (position.y - target.y) * scale,
  z: target.z + (position.z - target.z) * scale,
});

const positionsEqual = (left, right) =>
  left.x === right.x && left.y === right.y && left.z === right.z;

const flySpeed = (bounds, accelerated = false) => {
  const width = bounds.maximum.x - bounds.minimum.x;
  const height = bounds.maximum.y - bounds.minimum.y;
  const depth = bounds.maximum.z - bounds.minimum.z;
  const base = clamp(Math.hypot(width, height, depth) * 0.08, 0.5, 12);
  return base * (accelerated ? 3 : 1);
};

const translateViewpoint = (position, target, translation, bounds) => {
  const candidate = {
    x: position.x + translation.x,
    y: position.y + translation.y,
    z: position.z + translation.z,
  };
  const nextPosition = clampPosition(candidate, bounds);
  const applied = {
    x: nextPosition.x - position.x,
    y: nextPosition.y - position.y,
    z: nextPosition.z - position.z,
  };
  return {
    position: nextPosition,
    target: {
      x: target.x + applied.x,
      y: target.y + applied.y,
      z: target.z + applied.z,
    },
    clamped: !positionsEqual(candidate, nextPosition),
  };
};

export {
  clampPosition,
  flySpeed,
  lookTarget,
  orbitPosition,
  positionsEqual,
  translateViewpoint,
  zoomPosition,
};
