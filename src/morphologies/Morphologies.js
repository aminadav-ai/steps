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

    // Task state (Food foraging & 3-phase animated food lifecycle)
    this.foodPos = new Vec3(0, 0.16, 0);
    this.dissolvePos = new Vec3(0, 0.16, 0);
    this.foodState = 'ACTIVE'; // 'ACTIVE' | 'DISSOLVING' (1s) | 'WAITING' (1s) | 'DROPPING' (1s)
    this.foodStateTimer = 0;
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
    this.smoothedVelocityTowardFood = 0;
    this.lastMechanicalPower = 0;
    this.lastState = null;
    this.lastAction = null;
    this.isGrabbed = false;
    this.grabSavedOffsets = null;
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

    // When overturned on its back (up.y < 0.35) OR during the 2s pause between eaten food and new sky-drop,
    // the creature ignores food and walking gates!
    const isOverturned = up.y < 0.35;
    const noActiveFood = isOverturned || this.foodState === 'DISSOLVING' || this.foodState === 'WAITING';
    const foodForward = noActiveFood ? 0.0 : (dirX * forward.x + dirZ * forward.z);
    const foodRight = noActiveFood ? 0.0 : (dirX * right.x + dirZ * right.z);
    const normDist = noActiveFood ? 0.0 : Math.tanh(distToFood / 5.0);

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

    // Non-negative forward gate (0 when food is behind, when overturned, or while waiting for next sky-drop)
    const angleToFood = noActiveFood ? 0.0 : Math.atan2(foodRight, foodForward);
    const forwardGate = noActiveFood ? 0.0 : Math.max(0.0, foodForward);
    const turnDrive = noActiveFood ? 0.0 : Math.tanh(2.2 * angleToFood);

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
      clockSin * forwardGate,   // 12: forward-gated CPG sin (always forward)
      clockCos * forwardGate,   // 13: forward-gated CPG cos (always forward)
      clockSin * turnDrive,     // 14: yaw-turn-gated CPG sin
      clockCos * turnDrive      // 15: yaw-turn-gated CPG cos
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
  applyActionVector(actions, isRightingMode = false) {
    let totalPower = 0;
    const { up } = this.getTorsoFrame();
    // Differential between front-left (leg 0) and front-right (leg 1) stride commands
    const stride0 = Math.max(-1, Math.min(1, actions[1] || 0));
    const stride1 = Math.max(-1, Math.min(1, actions[3] || 0));
    const diffTurn = 0.5 * (stride1 - stride0);
    for (let i = 0; i < this.legControllers.length; i++) {
      const ctrl = this.legControllers[i];
      const actLift = Math.max(-1, Math.min(1, actions[i * 2] || 0));
      const actStride = Math.max(-1, Math.min(1, actions[i * 2 + 1] || 0));
      totalPower += ctrl.applyLegActions(actLift, actStride, diffTurn, isRightingMode, up);
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
  oppositeLateralAnchor = null,
  dorsalAnchor = null,
  isRearCenter = false,
  footY = 0.06,
  strideStroke = 0.185,
  liftStrokeUp = 0.11,
  liftStrokeDown = 0.125,
  color
}) {
  const restFootPos = new Vec3(hipNode.pos.x, footY, hipNode.pos.z);
  const restHipPos = hipNode.pos.clone();
  const restFrontPos = frontAnchor.pos.clone();
  const restRearPos = rearAnchor.pos.clone();
  const restLatPos = lateralAnchor.pos.clone();
  const restOppLatPos = oppositeLateralAnchor ? oppositeLateralAnchor.pos.clone() : null;
  const restDorsalPos = dorsalAnchor ? dorsalAnchor.pos.clone() : null;

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
    minFactor: 0.35,
    maxFactor: 2.15,
    maxForce: 135.0,
    compliance: 0.00010
  });

  const strideFwd = body.addMuscle({
    name: `leg${legIdx}_stride_fwd`,
    nodeA: frontAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.35,
    maxFactor: 1.95,
    maxForce: 120.0,
    compliance: 0.00014
  });

  const strideBwd = body.addMuscle({
    name: `leg${legIdx}_stride_bwd`,
    nodeA: rearAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.35,
    maxFactor: 1.95,
    maxForce: 120.0,
    compliance: 0.00014
  });

  const latMuscle = body.addMuscle({
    name: `leg${legIdx}_lat`,
    nodeA: lateralAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.35,
    maxFactor: 1.95,
    maxForce: 110.0,
    compliance: 0.00014
  });

  const oppLatMuscle = oppositeLateralAnchor
    ? body.addMuscle({
        name: `leg${legIdx}_lat_opp`,
        nodeA: oppositeLateralAnchor,
        nodeB: foot,
        pivotNode: hipNode,
        minFactor: 0.35,
        maxFactor: 1.95,
        maxForce: 110.0,
        compliance: 0.00014
      })
    : null;

  // Non-coplanar 3D dorsal tendon anchored to torsoTop: eliminates reflection ambiguity across the hip plane
  // and enables beetle-style dorsal hyper-extension to push against the floor when overturned!
  const dorsalMuscle = dorsalAnchor
    ? body.addMuscle({
        name: `leg${legIdx}_dorsal`,
        nodeA: dorsalAnchor,
        nodeB: foot,
        pivotNode: hipNode,
        minFactor: 0.12,
        maxFactor: 1.90,
        maxForce: 165.0,
        compliance: 0.00009
      })
    : null;

  const controller = {
    applyLegActions(actLift, actStride, diffTurn = 0, isRightingMode = false, upVec = null) {
      const footLocalUp = upVec
        ? (foot.pos.x - hipNode.pos.x) * upVec.x +
          (foot.pos.y - hipNode.pos.y) * upVec.y +
          (foot.pos.z - hipNode.pos.z) * upVec.z
        : (foot.pos.y - hipNode.pos.y);

      let dx, dy, dz;
      const needDorsalReach = isRightingMode && actLift < 0 && (!upVec || upVec.y < 0.42);
      if (!needDorsalReach) {
        // Ventral mode (normal walking OR stand-up phase once rolled past up.y >= 0.42):
        dx = actStride * strideStroke;
        dy = actLift < 0 ? -actLift * liftStrokeUp : -actLift * liftStrokeDown;
        dz = 0;
      } else {
        // Beetle Self-Righting Mode ("Замах и упор лапой за спину" when overturned, up.y < 0.52, actLift < 0):
        // Reaches into the dorsal hemisphere (y = 0.38..0.52m, past restHipPos.y = 0.25m) to press against the floor!
        dx = actStride * 0.26;
        dy = 0.24 + (-actLift) * 0.28;
        const outSign = restFootPos.z >= 0 ? 1 : -1;
        dz = isRearCenter ? diffTurn * 0.24 : -outSign * (-actLift) * 0.10;
      }

      const targetPos = new Vec3(
        restFootPos.x + dx,
        restFootPos.y + dy,
        restFootPos.z + dz
      );

      const targetLocalUp = targetPos.y - restHipPos.y;
      const crossingHipPlane =
        (targetLocalUp < -0.03 && footLocalUp > -0.05) ||
        (targetLocalUp > 0.03 && footLocalUp < 0.05);

      // Relax coplanar hip muscles while foot crosses the hip plane (footLocalUp ≈ 0)
      // so the 485 N coplanar snap-through barrier vanishes and dorsalMuscle smoothly pushes/pulls the foot across!
      kneeLift.maxForce = crossingHipPlane ? 14.0 : 135.0;
      strideFwd.maxForce = crossingHipPlane ? 18.0 : 120.0;
      strideBwd.maxForce = crossingHipPlane ? 18.0 : 120.0;
      latMuscle.maxForce = crossingHipPlane ? 18.0 : 110.0;
      if (oppLatMuscle) {
        oppLatMuscle.maxForce = crossingHipPlane ? 18.0 : 110.0;
      }

      kneeLift.targetAction = actLift;
      strideFwd.targetAction = actStride;

      kneeLift.setTargetLengthDirect(Vec3.dist(restHipPos, targetPos));
      strideFwd.setTargetLengthDirect(Vec3.dist(restFrontPos, targetPos));
      strideBwd.setTargetLengthDirect(Vec3.dist(restRearPos, targetPos));
      latMuscle.setTargetLengthDirect(Vec3.dist(restLatPos, targetPos));
      if (oppLatMuscle && restOppLatPos) {
        oppLatMuscle.setTargetLengthDirect(Vec3.dist(restOppLatPos, targetPos));
      }
      if (dorsalMuscle && restDorsalPos) {
        const footInverted = footLocalUp > -0.06;
        dorsalMuscle.maxForce = (isRightingMode || footInverted || crossingHipPlane) ? 195.0 : 0.0;
        dorsalMuscle.setTargetLengthDirect(Vec3.dist(restDorsalPos, targetPos));
      }

      return (
        kneeLift.instantPower +
        strideFwd.instantPower +
        strideBwd.instantPower +
        latMuscle.instantPower +
        (oppLatMuscle ? oppLatMuscle.instantPower : 0) +
        (dorsalMuscle ? dorsalMuscle.instantPower : 0)
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

  // Self-righting Gömböc / keel mass distribution:
  // Light high dorsal spire (torsoTop) + heavy low ballast keel (torsoBottom)
  // keeps the center of mass below the hip plane so the walker naturally rights itself if tilted!
  const torsoTop = body.addNode({
    id: `c${id}_top`,
    x: origin.x,
    y: origin.y + 0.48,
    z: origin.z,
    mass: 0.10,
    radius: 0.065,
    friction: 0.02,
    role: 'torso',
    color
  });

  const torsoBottom = body.addNode({
    id: `c${id}_bottom`,
    x: origin.x,
    y: origin.y + 0.16,
    z: origin.z,
    mass: 0.72,
    radius: 0.065,
    friction: 0.04,
    role: 'torso',
    color
  });

  const noseNode = body.addNode({
    id: `c${id}_nose`,
    x: origin.x + 0.34,
    y: origin.y + 0.24,
    z: origin.z,
    mass: 0.18,
    radius: 0.060,
    friction: 0.04,
    role: 'head',
    color
  });

  const tailNode = body.addNode({
    id: `c${id}_tail`,
    x: origin.x - 0.36,
    y: origin.y + 0.24,
    z: origin.z,
    mass: 0.18,
    radius: 0.060,
    friction: 0.04,
    role: 'hip',
    color
  });

  // 3 Hip sockets in a wide, stable equilateral triangle so the torso never tips onto its tail
  const hipCoords = [
    { x: +0.24, y: 0.25, z: +0.30 }, // Leg 0: Front-Left
    { x: +0.24, y: 0.25, z: -0.30 }, // Leg 1: Front-Right
    { x: -0.34, y: 0.25, z: 0.0 }    // Leg 2: Rear-Center
  ];

  const hips = hipCoords.map((hc, idx) =>
    body.addNode({
      id: `c${id}_hip_${idx}`,
      x: origin.x + hc.x,
      y: origin.y + hc.y,
      z: origin.z + hc.z,
      mass: 0.18,
      radius: 0.060,
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
    const oppositeLateralAnchor = i === 2 ? hips[1] : null;
    const leg = attachOrthogonalLeg({
      body,
      id,
      legIdx: i,
      hipNode: hips[i],
      frontAnchor: noseNode,
      rearAnchor: tailNode,
      lateralAnchor,
      oppositeLateralAnchor,
      dorsalAnchor: torsoTop,
      isRearCenter: i === 2,
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
    y: origin.y + 0.46,
    z: origin.z,
    mass: 0.12,
    radius: 0.068,
    friction: 0.04,
    role: 'torso',
    color
  });

  const torsoBottom = body.addNode({
    id: `c${id}_bottom`,
    x: origin.x,
    y: origin.y + 0.16,
    z: origin.z,
    mass: 0.68,
    radius: 0.065,
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
      dorsalAnchor: torsoTop,
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
