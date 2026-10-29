import { ArticulatedBody, Vec3 } from '../physics/Physics3D.js';

/**
 * Computes the local 3D coordinate frame { center, forward, up, right }
 * and sensor vector for a creature.
 */
export class CreatureInstance {
  constructor({ id, morphologyKey, body, frameRefs, footNodes, hipNodes, legControllers, color, spawnOrigin }) {
    this.id = id;
    this.morphologyKey = morphologyKey;
    this.body = body;
    this.frameRefs = frameRefs; // { top, bottom, front, tail, backLeft, backRight }
    this.footNodes = footNodes;
    this.hipNodes = hipNodes || [];
    this.legControllers = legControllers || [];
    this.color = color;
    this.spawnOrigin = spawnOrigin.clone();

    // Task state (Food foraging)
    this.foodPos = new Vec3(0, 0.16, 0);
    this.foodRadius = 0.24;
    this.foodCollected = 0;
    this.prevFoodDist = 0;

    // Step/Episode telemetry
    this.ageTime = 0;
    this.phase = (id * 0.39) % (Math.PI * 2);
    this.episodeSteps = 0;
    this.episodeReward = 0;
    this.cumulativeReward = 0;
    this.lastStepReward = 0;
    this.lastVelocityTowardFood = 0;
    this.lastMechanicalPower = 0;
    this.lastState = null;
    this.lastAction = null;
  }

  /**
   * Computes orthonormal local frame of the creature's rigid torso:
   * - center: 3D center of mass
   * - up: dorsal unit vector (bottom -> top)
   * - forward: anterior unit vector (tail -> front nose)
   * - right: lateral unit vector (forward x up)
   */
  getTorsoFrame() {
    const center = this.body.getCenterOfMass();
    const { top, bottom, front, tail, backLeft, backRight } = this.frameRefs;

    const baseX = bottom ? bottom.pos.x : (front.pos.x + backLeft.pos.x + backRight.pos.x) / 3;
    const baseY = bottom ? bottom.pos.y : (front.pos.y + backLeft.pos.y + backRight.pos.y) / 3;
    const baseZ = bottom ? bottom.pos.z : (front.pos.z + backLeft.pos.z + backRight.pos.z) / 3;

    const up = new Vec3(
      top.pos.x - baseX,
      top.pos.y - baseY,
      top.pos.z - baseZ
    ).normalize();

    const rearX = tail ? tail.pos.x : 0.5 * (backLeft.pos.x + backRight.pos.x);
    const rearY = tail ? tail.pos.y : 0.5 * (backLeft.pos.y + backRight.pos.y);
    const rearZ = tail ? tail.pos.z : 0.5 * (backLeft.pos.z + backRight.pos.z);

    const rawFx = front.pos.x - rearX;
    const rawFy = front.pos.y - rearY;
    const rawFz = front.pos.z - rearZ;
    const dotFU = rawFx * up.x + rawFy * up.y + rawFz * up.z;

    const forward = new Vec3(
      rawFx - dotFU * up.x,
      rawFy - dotFU * up.y,
      rawFz - dotFU * up.z
    ).normalize();

    const right = Vec3.cross(forward, up).normalize();

    return { center, up, forward, right };
  }

  /**
   * Builds the normalized observation vector s_t for the neural network.
   * All directional vectors are expressed in the creature's egocentric frame.
   */
  getObservation() {
    const { center, up, forward, right } = this.getTorsoFrame();
    const comVel = this.body.getCenterOfMassVelocity();

    const toFood = new Vec3(
      this.foodPos.x - center.x,
      0,
      this.foodPos.z - center.z
    );
    const distToFood = Math.hypot(toFood.x, toFood.z);
    const invDist = 1 / Math.max(1e-5, distToFood);
    const dirX = toFood.x * invDist;
    const dirZ = toFood.z * invDist;

    // Egocentric angle to food: forward projection (cos) and right projection (sin)
    const foodForward = dirX * forward.x + dirZ * forward.z;
    const foodRight = dirX * right.x + dirZ * right.z;
    const normDist = Math.tanh(distToFood / 5.0);

    // Egocentric linear velocity of CoM
    const velForward = Math.tanh((comVel.x * forward.x + comVel.y * forward.y + comVel.z * forward.z) / 2.0);
    const velRight = Math.tanh((comVel.x * right.x + comVel.y * right.y + comVel.z * right.z) / 2.0);
    const velUp = Math.tanh(comVel.y / 2.0);

    // Torso posture
    const upY = up.y;
    const pitchTilt = forward.y;
    const rollTilt = right.y;
    const normHeight = Math.tanh((center.y - 0.25) * 4.0);

    // Central Pattern Generator (CPG) phase clock (sin, cos) + descending sensory-gated CPG channels
    const clockSin = Math.sin(this.phase);
    const clockCos = Math.cos(this.phase);

    const obs = [
      foodForward,              // 0: cos(angle to food)
      foodRight,                // 1: sin(angle to food)
      normDist,                 // 2: distance to food
      velForward,               // 3: forward speed
      velRight,                 // 4: lateral speed
      velUp,                    // 5: vertical speed
      upY,                      // 6: uprightness
      pitchTilt,                // 7: pitch
      rollTilt,                 // 8: roll
      normHeight,               // 9: CoM height
      clockSin,                 // 10: sin(wt)
      clockCos,                 // 11: cos(wt)
      clockSin * foodForward,   // 12: forward-gated CPG sin
      clockCos * foodForward,   // 13: forward-gated CPG cos
      clockSin * foodRight,     // 14: turn-gated CPG sin
      clockCos * foodRight      // 15: turn-gated CPG cos
    ];

    const dofList = this.logicalDoFs;
    for (let i = 0; i < dofList.length; i++) {
      const primaryMuscle = dofList[i].primary;
      obs.push(Math.max(-1, Math.min(1, primaryMuscle.currentAngleNorm)));
      obs.push(Math.max(-1, Math.min(1, primaryMuscle.angularVelNorm)));
    }

    for (let i = 0; i < this.footNodes.length; i++) {
      const foot = this.footNodes[i];
      const contactSignal = foot.inContact
        ? Math.min(1.0, foot.normalForce / (this.body.totalMass * 9.81))
        : -0.5;
      obs.push(contactSignal);
    }

    return new Float64Array(obs);
  }

  /**
   * Applies the action vector a_t in [-1, 1]^actionDim to the 2-DoF leg controllers
   * so all 4 internal tendons of each leg converge to a geometrically consistent 3D point.
   */
  applyActionVector(actions) {
    let totalPower = 0;
    for (let i = 0; i < this.legControllers.length; i++) {
      const ctrl = this.legControllers[i];
      const actLift = Math.max(-1, Math.min(1, actions[i * 2] || 0));
      const actStride = Math.max(-1, Math.min(1, actions[i * 2 + 1] || 0));
      totalPower += ctrl.applyLegActions(actLift, actStride);
    }
    this.lastMechanicalPower = totalPower;
  }
}

function lockRigidFrame(body, nodes) {
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      body.addBone(nodes[i], nodes[j], null, 0.0, 'torso');
    }
  }
}

/**
 * Attaches a 2-DoF Geometrically-Consistent 4-Tendon Leg at hipNode.
 * Because all 4 internal XPBD muscles (hip, front, rear, lateral) are given the exact
 * Euclidean distances to the target 3D foot position p_0 + (-a_stride * S_x, -a_lift * S_y, 0),
 * there is ZERO internal geometric frustration while preserving exact linear and angular momentum!
 */
function attachOrthogonalLeg({
  body,
  id,
  legIdx,
  hipNode,
  frontAnchor,
  rearAnchor,
  lateralAnchor,
  footY = 0.06,
  strideStroke = 0.17,
  liftStrokeUp = 0.11,
  liftStrokeDown = 0.06,
  color
}) {
  const restFootPos = new Vec3(hipNode.pos.x, footY, hipNode.pos.z);
  const restHipPos = hipNode.pos.clone();
  const restFrontPos = frontAnchor.pos.clone();
  const restRearPos = rearAnchor.pos.clone();
  const restLatPos = lateralAnchor.pos.clone();

  const foot = body.addNode({
    id: `c${id}_foot_${legIdx}`,
    x: restFootPos.x,
    y: restFootPos.y,
    z: restFootPos.z,
    mass: 0.24,
    radius: 0.06,
    friction: 1.65, // High-traction rubber foot pad
    role: 'foot',
    color
  });

  const kneeLift = body.addMuscle({
    name: `leg${legIdx}_lift`,
    nodeA: hipNode,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.55,
    maxFactor: 1.45,
    maxForce: 135.0,
    compliance: 0.00010
  });

  const strideFwd = body.addMuscle({
    name: `leg${legIdx}_stride_fwd`,
    nodeA: frontAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.55,
    maxFactor: 1.45,
    maxForce: 120.0,
    compliance: 0.00014
  });

  const strideBwd = body.addMuscle({
    name: `leg${legIdx}_stride_bwd`,
    nodeA: rearAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.55,
    maxFactor: 1.45,
    maxForce: 120.0,
    compliance: 0.00014
  });

  const latMuscle = body.addMuscle({
    name: `leg${legIdx}_lat`,
    nodeA: lateralAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.70,
    maxFactor: 1.30,
    maxForce: 110.0,
    compliance: 0.00014
  });

  const controller = {
    applyLegActions(actLift, actStride) {
      // actStride in [-1, 1]: +1 = forward (+X), -1 = backward (-X)
      const dx = actStride * strideStroke;
      // actLift in [-1, 1]: -1 = lift UP (+Y), +1 = press DOWN (-Y)
      const dy = actLift < 0 ? -actLift * liftStrokeUp : -actLift * liftStrokeDown;

      const targetPos = new Vec3(
        restFootPos.x + dx,
        restFootPos.y + dy,
        restFootPos.z
      );

      kneeLift.targetAction = actLift;
      strideFwd.targetAction = actStride;

      kneeLift.setTargetLengthDirect(Vec3.dist(restHipPos, targetPos));
      strideFwd.setTargetLengthDirect(Vec3.dist(restFrontPos, targetPos));
      strideBwd.setTargetLengthDirect(Vec3.dist(restRearPos, targetPos));
      latMuscle.setTargetLengthDirect(Vec3.dist(restLatPos, targetPos));

      return (
        kneeLift.instantPower +
        strideFwd.instantPower +
        strideBwd.instantPower +
        latMuscle.instantPower
      );
    }
  };

  return {
    foot,
    controller,
    liftDoF: {
      name: `Лапа ${legIdx + 1}: Сгиб колена (Подъём/Опора)`,
      primary: kneeLift,
      antagonist: null
    },
    strideDoF: {
      name: `Лапа ${legIdx + 1}: Мах бедра (Шаг вперёд/назад)`,
      primary: strideFwd,
      antagonist: strideBwd
    }
  };
}

/**
 * Morphology #1 (Default): 3-Legged Tripod Walker ("Трёхлучевой шагоход «Трипод»")
 * - Rigid 3D torso frame with 3 legs arranged in a balanced triangle:
 *   Leg 0: Front-Left  (x = +0.16, z = +0.26)
 *   Leg 1: Front-Right (x = +0.16, z = -0.26)
 *   Leg 2: Rear-Center (x = -0.22, z =  0.00)
 * - 6 logical DoFs (2 per leg: Vertical Knee Lift + Horizontal Sagittal Stride)
 */
function buildTripodWalker({ id, origin = new Vec3(0, 0, 0), color = '#38bdf8' }) {
  const body = new ArticulatedBody();

  const torsoTop = body.addNode({
    id: `c${id}_top`,
    x: origin.x,
    y: origin.y + 0.44,
    z: origin.z,
    mass: 0.35,
    radius: 0.08,
    friction: 0.04,
    role: 'torso',
    color
  });

  const torsoBottom = body.addNode({
    id: `c${id}_bottom`,
    x: origin.x,
    y: origin.y + 0.20,
    z: origin.z,
    mass: 0.30,
    radius: 0.068,
    friction: 0.04,
    role: 'torso',
    color
  });

  const noseNode = body.addNode({
    id: `c${id}_nose`,
    x: origin.x + 0.40,
    y: origin.y + 0.22,
    z: origin.z,
    mass: 0.20,
    radius: 0.065,
    friction: 0.04,
    role: 'head',
    color
  });

  const tailNode = body.addNode({
    id: `c${id}_tail`,
    x: origin.x - 0.46,
    y: origin.y + 0.22,
    z: origin.z,
    mass: 0.20,
    radius: 0.065,
    friction: 0.04,
    role: 'hip',
    color
  });

  // 3 Hip sockets directly above the 3 feet
  const hipCoords = [
    { x: +0.16, y: 0.32, z: +0.26 }, // Leg 0: Front-Left
    { x: +0.16, y: 0.32, z: -0.26 }, // Leg 1: Front-Right
    { x: -0.22, y: 0.32, z: 0.0 }    // Leg 2: Rear-Center
  ];

  const hips = hipCoords.map((hc, idx) =>
    body.addNode({
      id: `c${id}_hip_${idx}`,
      x: origin.x + hc.x,
      y: origin.y + hc.y,
      z: origin.z + hc.z,
      mass: 0.24,
      radius: 0.065,
      friction: 0.04,
      role: 'hip',
      color
    })
  );

  // Lock all 7 torso nodes into one solid 3D rigid body
  lockRigidFrame(body, [torsoTop, torsoBottom, noseNode, tailNode, hips[0], hips[1], hips[2]]);

  const feet = [];
  const legControllers = [];
  const logicalDoFs = [];

  for (let i = 0; i < 3; i++) {
    const lateralAnchor = i === 0 ? hips[1] : i === 1 ? hips[0] : hips[0];
    const leg = attachOrthogonalLeg({
      body,
      id,
      legIdx: i,
      hipNode: hips[i],
      frontAnchor: noseNode,
      rearAnchor: tailNode,
      lateralAnchor,
      footY: origin.y + 0.06,
      color
    });
    feet.push(leg.foot);
    legControllers.push(leg.controller);
    logicalDoFs.push(leg.liftDoF, leg.strideDoF);
  }

  const creature = new CreatureInstance({
    id,
    morphologyKey: 'tripod',
    body,
    frameRefs: {
      top: torsoTop,
      bottom: torsoBottom,
      front: noseNode,
      tail: tailNode,
      backLeft: hips[0],
      backRight: hips[1]
    },
    footNodes: feet,
    hipNodes: hips,
    legControllers,
    color,
    spawnOrigin: origin
  });
  creature.logicalDoFs = logicalDoFs;
  return creature;
}

/**
 * Morphology #2: 4-Legged Quadruped Walker ("Четырёхногий шагоход «Квадрупод»")
 * - 8 logical DoFs (4 legs x 2 DoFs)
 */
function buildQuadrupedWalker({ id, origin = new Vec3(0, 0, 0), color = '#a855f7' }) {
  const body = new ArticulatedBody();

  const torsoTop = body.addNode({
    id: `c${id}_top`,
    x: origin.x,
    y: origin.y + 0.44,
    z: origin.z,
    mass: 0.36,
    radius: 0.08,
    friction: 0.04,
    role: 'torso',
    color
  });

  const torsoBottom = body.addNode({
    id: `c${id}_bottom`,
    x: origin.x,
    y: origin.y + 0.20,
    z: origin.z,
    mass: 0.30,
    radius: 0.068,
    friction: 0.04,
    role: 'torso',
    color
  });

  const noseNode = body.addNode({
    id: `c${id}_nose`,
    x: origin.x + 0.42,
    y: origin.y + 0.22,
    z: origin.z,
    mass: 0.20,
    radius: 0.065,
    friction: 0.04,
    role: 'head',
    color
  });

  const tailNode = body.addNode({
    id: `c${id}_tail`,
    x: origin.x - 0.42,
    y: origin.y + 0.22,
    z: origin.z,
    mass: 0.20,
    radius: 0.065,
    friction: 0.04,
    role: 'hip',
    color
  });

  const hipCoords = [
    { x: +0.18, y: 0.32, z: +0.24 }, // Front-Left
    { x: +0.18, y: 0.32, z: -0.24 }, // Front-Right
    { x: -0.18, y: 0.32, z: +0.24 }, // Rear-Left
    { x: -0.18, y: 0.32, z: -0.24 }  // Rear-Right
  ];

  const hips = hipCoords.map((hc, idx) =>
    body.addNode({
      id: `c${id}_hip_${idx}`,
      x: origin.x + hc.x,
      y: origin.y + hc.y,
      z: origin.z + hc.z,
      mass: 0.22,
      radius: 0.065,
      friction: 0.04,
      role: 'hip',
      color
    })
  );

  lockRigidFrame(body, [torsoTop, torsoBottom, noseNode, tailNode, ...hips]);

  const feet = [];
  const legControllers = [];
  const logicalDoFs = [];
  for (let i = 0; i < 4; i++) {
    const oppositeLateral = i % 2 === 0 ? hips[i + 1] : hips[i - 1];
    const leg = attachOrthogonalLeg({
      body,
      id,
      legIdx: i,
      hipNode: hips[i],
      frontAnchor: noseNode,
      rearAnchor: tailNode,
      lateralAnchor: oppositeLateral,
      footY: origin.y + 0.06,
      color
    });
    feet.push(leg.foot);
    legControllers.push(leg.controller);
    logicalDoFs.push(leg.liftDoF, leg.strideDoF);
  }

  const creature = new CreatureInstance({
    id,
    morphologyKey: 'quadruped',
    body,
    frameRefs: {
      top: torsoTop,
      bottom: torsoBottom,
      front: noseNode,
      tail: tailNode,
      backLeft: hips[2],
      backRight: hips[3]
    },
    footNodes: feet,
    hipNodes: hips,
    legControllers,
    color,
    spawnOrigin: origin
  });
  creature.logicalDoFs = logicalDoFs;
  return creature;
}

export const MORPHOLOGY_REGISTRY = {
  tripod: {
    key: 'tripod',
    name: 'Конструкция №1: 3-лучевой шагоход («Трипод»)',
    shortName: 'Трипод (3 лапы, 6 DoF)',
    description: 'Жёсткий 3D-каркас торса + 3 двухстепенные лапы (передняя левая, передняя правая и задняя толчковая). Каждая лапа управляется сгибом колена (подъём/опора) и продольным махом бедра (тяга вперёд/назад).',
    obsDim: 31,
    actDim: 6,
    build: buildTripodWalker
  },
  quadruped: {
    key: 'quadruped',
    name: 'Конструкция №2: 4-лучевой шагоход («Квадрупод»)',
    shortName: 'Квадрупод (4 лапы, 8 DoF)',
    description: 'Четырёхногий шагоход с 8 активными степенями свободы (вертикальный сгиб колена + продольный мах бедра на каждую из 4 лап). Высокая продольная и боковая устойчивость.',
    obsDim: 36,
    actDim: 8,
    build: buildQuadrupedWalker
  }
};
