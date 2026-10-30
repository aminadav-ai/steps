import { ArticulatedBody, Vec3 } from '../physics/Physics3D.js';

/**
 * Computes the local 3D coordinate frame { center, forward, up, right }
 * and sensor vector for a creature.
 */
export class CreatureInstance {
  constructor({
    id,
    morphologyKey,
    actuatorMode = 'servo',
    body,
    frameRefs,
    footNodes,
    hipNodes,
    kneeNodes = [],
    legControllers,
    color,
    spawnOrigin
  }) {
    this.id = id;
    this.morphologyKey = morphologyKey;
    this.actuatorMode = actuatorMode; // 'servo' (Rigid Revolute PD Servos) | 'muscle' (Soft XPBD Bio-Muscles)
    this.body = body;
    this.frameRefs = frameRefs; // { top, bottom, front, tail, backLeft, backRight }
    this.torsoTop = frameRefs ? frameRefs.top : null;
    this.torsoBottom = frameRefs ? frameRefs.bottom : null;
    this.footNodes = footNodes;
    this.hipNodes = hipNodes || [];
    this.kneeNodes = kneeNodes || [];
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
   * Returns true if any leg's foot or knee is currently inverted above the hip plane in torso coordinates.
   */
  hasInvertedLeg() {
    const { up } = this.getTorsoFrame();
    for (let i = 0; i < this.footNodes.length; i++) {
      const hip = this.hipNodes[i];
      const foot = this.footNodes[i];
      if (!hip || !foot) continue;
      const footLocalUp =
        (foot.pos.x - hip.pos.x) * up.x +
        (foot.pos.y - hip.pos.y) * up.y +
        (foot.pos.z - hip.pos.z) * up.z;
      if (footLocalUp > -0.04) return true;
    }
    return false;
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

    // When overturned on its back/side (up.y < 0.52) OR during the 2s pause between eaten food and new sky-drop,
    // the creature ignores food and walking gates!
    const isOverturned = up.y < 0.52;
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
   * Applies the action vector a_t in [-1, 1]^actionDim to the creature's actuators
   * so all internal XPBD tendons/servos converge to geometrically consistent 3D targets.
   */
  applyActionVector(actions, isRightingMode = false) {
    if (typeof this.customApplyActions === 'function') {
      this.lastMechanicalPower = this.customApplyActions(actions, isRightingMode);
      return;
    }
    let totalPower = 0;
    const { up, forward, right } = this.getTorsoFrame();
    // Differential between front-left (leg 0) and front-right (leg 1) stride commands
    const stride0 = Math.max(-1, Math.min(1, actions[1] || 0));
    const stride1 = Math.max(-1, Math.min(1, actions[3] || 0));
    const diffTurn = 0.5 * (stride1 - stride0);
    for (let i = 0; i < this.legControllers.length; i++) {
      const ctrl = this.legControllers[i];
      const actLift = Math.max(-1, Math.min(1, actions[i * 2] || 0));
      const actStride = Math.max(-1, Math.min(1, actions[i * 2 + 1] || 0));
      totalPower += ctrl.applyLegActions(actLift, actStride, diffTurn, isRightingMode, up, forward, right);
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
  actuatorMode = 'servo',
  hipNode,
  frontAnchor,
  rearAnchor,
  lateralAnchor,
  oppositeLateralAnchor = null,
  dorsalAnchor = null,
  isRearCenter = false,
  kneeBendSign = null,
  buildWebbedFoot = false,
  enableHipRoll = false,
  restFootOffsetX = 0.0,
  dofLabelPrefix = null,
  footY = 0.06,
  strideStroke = 0.185,
  liftStrokeUp = 0.11,
  liftStrokeDown = 0.125,
  color
}) {
  const isServoMode = actuatorMode !== 'muscle';
  const L_femur = 0.195;
  const L_tibia = 0.195;

  const restFootPos = new Vec3(hipNode.pos.x + restFootOffsetX, footY, hipNode.pos.z);
  const restHipPos = hipNode.pos.clone();
  const restFrontPos = frontAnchor.pos.clone();
  const restRearPos = rearAnchor.pos.clone();
  const restLatPos = lateralAnchor.pos.clone();
  const restOppLatPos = oppositeLateralAnchor ? oppositeLateralAnchor.pos.clone() : null;
  const restDorsalPos = dorsalAnchor ? dorsalAnchor.pos.clone() : null;
  const restComX = 0.5 * (restFrontPos.x + restRearPos.x);
  const restComZ = 0.5 * (restFrontPos.z + restRearPos.z);

  // Exact 3D Revolute 2-Link Knee Kinematics in canonical torso frame:
  // In 'servo' mode, enforces strict 2D sagittal-plane revolute joint geometry (or 3D hip-roll plane when enableHipRoll=true)
  // with outward X-configuration or avian knee orientation
  // while guaranteeing ||kneePos - restHipPos|| === L_femur and ||kneePos - targetFootPos|| === L_tibia.
  const computeKneeTarget3D = targetFootPos => {
    const ux = targetFootPos.x - restHipPos.x;
    const uy = targetFootPos.y - restHipPos.y;
    const uz = (isServoMode && !enableHipRoll) ? 0.0 : (targetFootPos.z - restHipPos.z);
    const rawD = Math.hypot(ux, uy, uz);
    const maxReach = (L_femur + L_tibia) * 0.992;
    const d = Math.max(0.05, Math.min(maxReach, rawD));
    const scale = rawD > 1e-6 ? d / rawD : 1.0;
    const cx = ux * scale;
    const cy = uy * scale;
    const cz = uz * scale;

    const midX = restHipPos.x + 0.5 * cx;
    const midY = restHipPos.y + 0.5 * cy;
    const midZ = restHipPos.z + 0.5 * cz;

    // Outward X-configuration or Avian digitigrade sagittal knee bend vector
    const isFrontLeg = kneeBendSign !== null ? (kneeBendSign > 0) : (restHipPos.x >= restComX);
    const bx0 = isServoMode
      ? (isFrontLeg ? 0.92 : -0.92)
      : (0.65 + (restHipPos.x - restComX) * 1.4);
    const by0 = 0.25;
    const bz0 = isServoMode ? 0.0 : (restHipPos.z - restComZ) * 1.4;
    const dotBU = (bx0 * cx + by0 * cy + bz0 * cz) / Math.max(1e-6, d * d);
    let px = bx0 - dotBU * cx;
    let py = by0 - dotBU * cy;
    let pz = bz0 - dotBU * cz;
    const pLen = Math.hypot(px, py, pz);
    if (pLen > 1e-6) {
      px /= pLen;
      py /= pLen;
      pz /= pLen;
    } else {
      px = isFrontLeg ? 1.0 : -1.0;
      py = 0.0;
      pz = 0.0;
    }

    const hKnee = Math.sqrt(Math.max(1e-5, L_femur * L_femur - 0.25 * d * d));
    return new Vec3(
      midX + px * hKnee,
      midY + py * hKnee,
      midZ + pz * hKnee
    );
  };

  let knee = null;
  let kneeFrontServo = null;
  let kneeRearServo = null;
  let kneeLatServo = null;
  let kneeOppLatServo = null;
  let kneeDorsalServo = null;

  if (isServoMode) {
    const restKneePos = computeKneeTarget3D(restFootPos);
    knee = body.addNode({
      id: `c${id}_knee_${legIdx}`,
      x: restKneePos.x,
      y: restKneePos.y,
      z: restKneePos.z,
      mass: 0.04,
      radius: 0.038,
      friction: 0.25,
      role: 'knee',
      color
    });
  }

  const foot = body.addNode({
    id: `c${id}_foot_${legIdx}`,
    x: restFootPos.x,
    y: restFootPos.y,
    z: restFootPos.z,
    mass: buildWebbedFoot ? 0.16 : (isServoMode ? 0.22 : 0.24),
    radius: 0.06,
    friction: 1.65, // High-traction rubber foot pad
    role: 'foot',
    color
  });

  if (isServoMode && knee) {
    // True Rigid Femur and Tibia links (0.0 compliance = rigid robotic segments)
    body.addBone(hipNode, knee, L_femur, 0.0, 'femur');
    body.addBone(knee, foot, L_tibia, 0.0, 'tibia');

    kneeFrontServo = body.addMuscle({
      name: `leg${legIdx}_knee_front_servo`,
      nodeA: frontAnchor,
      nodeB: knee,
      pivotNode: hipNode,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 95.0,
      compliance: 0.00012
    });
    kneeRearServo = body.addMuscle({
      name: `leg${legIdx}_knee_rear_servo`,
      nodeA: rearAnchor,
      nodeB: knee,
      pivotNode: hipNode,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 95.0,
      compliance: 0.00012
    });
    kneeLatServo = body.addMuscle({
      name: `leg${legIdx}_knee_lat_servo`,
      nodeA: lateralAnchor,
      nodeB: knee,
      pivotNode: hipNode,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 165.0,
      compliance: 0.00004
    });
    if (oppositeLateralAnchor) {
      kneeOppLatServo = body.addMuscle({
        name: `leg${legIdx}_knee_lat_opp_servo`,
        nodeA: oppositeLateralAnchor,
        nodeB: knee,
        pivotNode: hipNode,
        minFactor: 0.25,
        maxFactor: 2.20,
        maxForce: 165.0,
        compliance: 0.00004
      });
    }
    if (dorsalAnchor) {
      kneeDorsalServo = body.addMuscle({
        name: `leg${legIdx}_knee_dorsal_servo`,
        nodeA: dorsalAnchor,
        nodeB: knee,
        pivotNode: hipNode,
        minFactor: 0.15,
        maxFactor: 2.20,
        maxForce: 95.0,
        compliance: 0.00012
      });
    }
  }

  const kneeLift = body.addMuscle({
    name: `leg${legIdx}_lift`,
    nodeA: hipNode,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.35,
    maxFactor: 2.15,
    maxForce: isServoMode ? 175.0 : 135.0,
    compliance: isServoMode ? 0.00006 : 0.00010
  });

  const strideFwd = body.addMuscle({
    name: `leg${legIdx}_stride_fwd`,
    nodeA: frontAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.35,
    maxFactor: 1.95,
    maxForce: isServoMode ? 155.0 : 120.0,
    compliance: isServoMode ? 0.00008 : 0.00014
  });

  const strideBwd = body.addMuscle({
    name: `leg${legIdx}_stride_bwd`,
    nodeA: rearAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.35,
    maxFactor: 1.95,
    maxForce: isServoMode ? 155.0 : 120.0,
    compliance: isServoMode ? 0.00008 : 0.00014
  });

  const latMuscle = body.addMuscle({
    name: `leg${legIdx}_lat`,
    nodeA: lateralAnchor,
    nodeB: foot,
    pivotNode: hipNode,
    minFactor: 0.35,
    maxFactor: 1.95,
    maxForce: isServoMode ? 185.0 : 110.0,
    compliance: isServoMode ? 0.00004 : 0.00014
  });

  const oppLatMuscle = oppositeLateralAnchor
    ? body.addMuscle({
        name: `leg${legIdx}_lat_opp`,
        nodeA: oppositeLateralAnchor,
        nodeB: foot,
        pivotNode: hipNode,
        minFactor: 0.35,
        maxFactor: 1.95,
        maxForce: isServoMode ? 185.0 : 110.0,
        compliance: isServoMode ? 0.00004 : 0.00014
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

  // Optional 3-point flat Webbed Duck Foot / Open_Duck_Mini Wedge Shoe ("стопа-ласта / башмак")
  // Ankle joint (foot) sits in the longitudinal CENTER of the shoe:
  // - toeInner = Front Toe Pad (+0.14 m ahead of ankle, stops forward pitch!)
  // - toeOuter = Rear Heel Pad (-0.12 m behind ankle, stops backward pitch!)
  let webbedFoot = null;
  let toeInnerFront = null;
  let toeInnerRear = null;
  let toeInnerHip = null;
  let toeInnerLat = null;
  let toeInnerDorsal = null;
  let toeOuterFront = null;
  let toeOuterRear = null;
  let toeOuterHip = null;
  let toeOuterLat = null;
  let toeOuterDorsal = null;
  const sideSignZ = restFootPos.z >= 0 ? 1 : -1;
  const restToeInnerOffset = new Vec3(+0.16, 0.0, -sideSignZ * 0.035);
  const restToeOuterOffset = new Vec3(-0.15, 0.0, +sideSignZ * 0.055);

  if (buildWebbedFoot && isServoMode) {
    const toeInnerPos = new Vec3(
      restFootPos.x + restToeInnerOffset.x,
      restFootPos.y + restToeInnerOffset.y,
      restFootPos.z + restToeInnerOffset.z
    );
    const toeOuterPos = new Vec3(
      restFootPos.x + restToeOuterOffset.x,
      restFootPos.y + restToeOuterOffset.y,
      restFootPos.z + restToeOuterOffset.z
    );
    const toeInner = body.addNode({
      id: `c${id}_toe_in_${legIdx}`,
      x: toeInnerPos.x,
      y: toeInnerPos.y,
      z: toeInnerPos.z,
      mass: 0.085,
      radius: 0.056,
      friction: 0.06, // Beveled PLA front shoe tip: full vertical pitch support without lateral scrub tripping during turns!
      role: 'foot_pad',
      color: '#eab308'
    });
    const toeOuter = body.addNode({
      id: `c${id}_toe_out_${legIdx}`,
      x: toeOuterPos.x,
      y: toeOuterPos.y,
      z: toeOuterPos.z,
      mass: 0.085,
      radius: 0.056,
      friction: 0.06, // Beveled PLA rear heel tip: full vertical pitch support without lateral scrub tripping during turns!
      role: 'foot_pad',
      color: '#eab308'
    });

    // Rigid triangular webbed foot plate (heel <-> inner toe <-> outer toe)
    body.addBone(foot, toeInner, null, 0.0, 'foot_pad');
    body.addBone(foot, toeOuter, null, 0.0, 'foot_pad');
    body.addBone(toeInner, toeOuter, null, 0.0, 'foot_pad');

    // True Open_Duck_Mini_v2 Ankle & Shoe Servo Linkage:
    // Holds the 26 cm flat wedge shoe (toeInner at +0.14 m front, toeOuter at -0.12 m rear heel)
    // firmly at the target ankle angle relative to the shin (knee), hip, and vertical torso spine!
    toeInnerFront = body.addMuscle({
      name: `leg${legIdx}_ankle_in_front`,
      nodeA: frontAnchor,
      nodeB: toeInner,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 145.0,
      compliance: 0.00006
    });
    toeInnerRear = body.addMuscle({
      name: `leg${legIdx}_ankle_in_rear`,
      nodeA: rearAnchor,
      nodeB: toeInner,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 145.0,
      compliance: 0.00006
    });
    toeInnerHip = body.addMuscle({
      name: `leg${legIdx}_ankle_in_hip`,
      nodeA: hipNode,
      nodeB: toeInner,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 155.0,
      compliance: 0.00005
    });
    toeInnerLat = body.addMuscle({
      name: `leg${legIdx}_ankle_in_lat`,
      nodeA: lateralAnchor,
      nodeB: toeInner,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 145.0,
      compliance: 0.00006
    });
    if (dorsalAnchor) {
      toeInnerDorsal = body.addMuscle({
        name: `leg${legIdx}_ankle_in_dorsal`,
        nodeA: dorsalAnchor,
        nodeB: toeInner,
        pivotNode: foot,
        minFactor: 0.20,
        maxFactor: 2.20,
        maxForce: 135.0,
        compliance: 0.00007
      });
    }

    toeOuterFront = body.addMuscle({
      name: `leg${legIdx}_ankle_out_front`,
      nodeA: frontAnchor,
      nodeB: toeOuter,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 145.0,
      compliance: 0.00006
    });
    toeOuterRear = body.addMuscle({
      name: `leg${legIdx}_ankle_out_rear`,
      nodeA: rearAnchor,
      nodeB: toeOuter,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 145.0,
      compliance: 0.00006
    });
    toeOuterHip = body.addMuscle({
      name: `leg${legIdx}_ankle_out_hip`,
      nodeA: hipNode,
      nodeB: toeOuter,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 155.0,
      compliance: 0.00005
    });
    toeOuterLat = body.addMuscle({
      name: `leg${legIdx}_ankle_out_lat`,
      nodeA: lateralAnchor,
      nodeB: toeOuter,
      pivotNode: foot,
      minFactor: 0.25,
      maxFactor: 2.20,
      maxForce: 145.0,
      compliance: 0.00006
    });
    if (dorsalAnchor) {
      toeOuterDorsal = body.addMuscle({
        name: `leg${legIdx}_ankle_out_dorsal`,
        nodeA: dorsalAnchor,
        nodeB: toeOuter,
        pivotNode: foot,
        minFactor: 0.20,
        maxFactor: 2.20,
        maxForce: 135.0,
        compliance: 0.00007
      });
    }

    webbedFoot = { heel: foot, toeInner, toeOuter };
  }

  kneeLift.isServo = isServoMode;
  kneeLift.jointAngleRad = isServoMode ? Math.PI * 0.58 : 0;
  kneeLift.maxTorqueNm = 22.0;
  kneeLift.jointTorqueNm = 0.0;
  kneeLift.torqueLoadRatio = 0.0;

  strideFwd.isServo = isServoMode;
  strideFwd.jointAngleRad = 0.0;
  strideFwd.maxTorqueNm = 18.0;
  strideFwd.jointTorqueNm = 0.0;
  strideFwd.torqueLoadRatio = 0.0;

  latMuscle.isServo = isServoMode;
  latMuscle.jointAngleRad = 0.0;
  latMuscle.maxTorqueNm = 18.0;
  latMuscle.jointTorqueNm = 0.0;
  latMuscle.torqueLoadRatio = 0.0;

  if (toeInnerHip) {
    toeInnerHip.isServo = true;
    toeInnerHip.jointAngleRad = 0.0;
    toeInnerHip.maxTorqueNm = 15.0;
    toeInnerHip.jointTorqueNm = 0.0;
    toeInnerHip.torqueLoadRatio = 0.0;
  }
  if (toeOuterFront) {
    toeOuterFront.isServo = true;
    toeOuterFront.jointAngleRad = 0.0;
    toeOuterFront.maxTorqueNm = 14.0;
    toeOuterFront.jointTorqueNm = 0.0;
    toeOuterFront.torqueLoadRatio = 0.0;
  }

  const controller = {
    isServo: isServoMode,
    jointAngleRad: kneeLift.jointAngleRad,
    jointTorqueNm: 0.0,
    maxTorqueNm: 22.0,
    torqueLoadRatio: 0.0,
    applyLegActions(
      actLift,
      actStride,
      diffTurn = 0,
      isRightingMode = false,
      upVec = null,
      forwardVec = null,
      rightVec = null,
      actHipRoll = 0,
      actHipYaw = 0,
      actAnklePitch = null
    ) {
      let footLocalUp = upVec
        ? (foot.pos.x - hipNode.pos.x) * upVec.x +
          (foot.pos.y - hipNode.pos.y) * upVec.y +
          (foot.pos.z - hipNode.pos.z) * upVec.z
        : (foot.pos.y - hipNode.pos.y);

      const needDorsalReach = isRightingMode && actLift < 0 && (!upVec || upVec.y < 0.42);
      const isUprightOrStanding = !isRightingMode || (upVec && upVec.y >= 0.42);

      // Physical Mechanical Joint Hard-Stops in 'servo' mode (conserving exact center-of-mass momentum sum m_i dx_i === 0):
      // 1. Hip Ventral Stop: when the chassis is upright, the leg cannot passively fold/invert above the hip plane.
      // 2. Knee Anti-Hyperextension Stop ("Ʌ" stop): prevents the knee from ever snapping into a backward "V" bend,
      //    while allowing 100% free sagittal stride when the knee is in its normal "Ʌ" working range!
      if (isServoMode && knee && upVec && forwardVec && rightVec) {
        if (isUprightOrStanding && footLocalUp > -0.055) {
          const corrUp = -0.068 - footLocalUp;
          const mF = foot.mass;
          const mH = hipNode.mass;
          const wF = mH / (mF + mH);
          const wH = mF / (mF + mH);
          const dxStop = upVec.x * corrUp;
          const dyStop = upVec.y * corrUp;
          const dzStop = upVec.z * corrUp;

          foot.pos.x += dxStop * wF;
          foot.pos.y += dyStop * wF;
          foot.pos.z += dzStop * wF;
          foot.prevPos.x += dxStop * wF;
          foot.prevPos.y += dyStop * wF;
          foot.prevPos.z += dzStop * wF;

          hipNode.pos.x -= dxStop * wH;
          hipNode.pos.y -= dyStop * wH;
          hipNode.pos.z -= dzStop * wH;
          hipNode.prevPos.x -= dxStop * wH;
          hipNode.prevPos.y -= dyStop * wH;
          hipNode.prevPos.z -= dzStop * wH;

          footLocalUp = -0.068;
        }

        // Check current knee position relative to the hip->foot chord in the torso sagittal plane
        const uxW = foot.pos.x - hipNode.pos.x;
        const uyW = foot.pos.y - hipNode.pos.y;
        const uzW = foot.pos.z - hipNode.pos.z;
        const uf = uxW * forwardVec.x + uyW * forwardVec.y + uzW * forwardVec.z;
        const uu = uxW * upVec.x + uyW * upVec.y + uzW * upVec.z;
        const rawDW = Math.hypot(uf, uu);
        const maxReachW = (L_femur + L_tibia) * 0.992;
        const dW = Math.max(0.05, Math.min(maxReachW, rawDW));

        const isFrontLeg = kneeBendSign !== null ? (kneeBendSign > 0) : (restHipPos.x >= restComX);
        const bf0 = isFrontLeg ? 0.92 : -0.92;
        const bu0 = 0.25;
        const dotBW = (bf0 * uf + bu0 * uu) / Math.max(1e-6, rawDW * rawDW);
        let pf = bf0 - dotBW * uf;
        let pu = bu0 - dotBW * uu;
        const pLenW = Math.hypot(pf, pu);
        if (pLenW > 1e-6) {
          pf /= pLenW;
          pu /= pLenW;
        } else {
          pf = isFrontLeg ? 1.0 : -1.0;
          pu = 0.0;
        }

        // Current knee coordinates relative to hip->foot midpoint in (forwardVec, upVec, rightVec)
        const midX = hipNode.pos.x + 0.5 * uxW;
        const midY = hipNode.pos.y + 0.5 * uyW;
        const midZ = hipNode.pos.z + 0.5 * uzW;
        const kxRel = knee.pos.x - midX;
        const kyRel = knee.pos.y - midY;
        const kzRel = knee.pos.z - midZ;
        const kF = kxRel * forwardVec.x + kyRel * forwardVec.y + kzRel * forwardVec.z;
        const kU = kxRel * upVec.x + kyRel * upVec.y + kzRel * upVec.z;
        const kR = (knee.pos.x - hipNode.pos.x) * rightVec.x + (knee.pos.y - hipNode.pos.y) * rightVec.y + (knee.pos.z - hipNode.pos.z) * rightVec.z;

        const hActual = kF * pf + kU * pu;
        const hTarget = Math.sqrt(Math.max(1e-5, L_femur * L_femur - 0.25 * dW * dW));
        const minHealthyH = Math.min(0.055, hTarget * 0.65);

        // Only engage sagittal knee hard-stop if knee tries to hyperextend/invert below minHealthyH,
        // plus enforce planar bearing alignment (allowing hip_roll offset when enableHipRoll=true)
        const targetKR = enableHipRoll ? (actHipRoll * 0.025) : 0.0;
        const corrH = hActual < minHealthyH ? (hTarget - hActual) * 0.85 : 0.0;
        const corrR = (targetKR - kR) * 0.75;
        if (corrH > 1e-5 || Math.abs(corrR) > 1e-5) {
          const errKX = corrH * (pf * forwardVec.x + pu * upVec.x) + corrR * rightVec.x;
          const errKY = corrH * (pf * forwardVec.y + pu * upVec.y) + corrR * rightVec.y;
          const errKZ = corrH * (pf * forwardVec.z + pu * upVec.z) + corrR * rightVec.z;

          const mK = knee.mass;
          const mHF = hipNode.mass + foot.mass;
          const wK = mHF / (mK + mHF);
          const wHF = mK / (mK + mHF);

          knee.pos.x += errKX * wK;
          knee.pos.y += errKY * wK;
          knee.pos.z += errKZ * wK;
          knee.prevPos.x += errKX * wK;
          knee.prevPos.y += errKY * wK;
          knee.prevPos.z += errKZ * wK;

          hipNode.pos.x -= errKX * wHF;
          hipNode.pos.y -= errKY * wHF;
          hipNode.pos.z -= errKZ * wHF;
          hipNode.prevPos.x -= errKX * wHF;
          hipNode.prevPos.y -= errKY * wHF;
          hipNode.prevPos.z -= errKZ * wHF;

          foot.pos.x -= errKX * wHF;
          foot.pos.y -= errKY * wHF;
          foot.pos.z -= errKZ * wHF;
          foot.prevPos.x -= errKX * wHF;
          foot.prevPos.y -= errKY * wHF;
          foot.prevPos.z -= errKZ * wHF;
        }
      }

      let dx, dy, dz;
      const yawDiffStride = enableHipRoll ? ((legIdx === 0 ? -actHipYaw : +actHipYaw) * 0.065) : 0.0;
      if (!needDorsalReach) {
        // Ventral mode (normal walking, standing tall, squatting onto bottom, OR stand-up phase once rolled past up.y >= 0.42):
        // For 5-DoF bipedal duck (buildWebbedFoot), onboard BNO055 IMU pitch/roll feedback (forwardVec.y, rightVec.y)
        // drives the physical hip & ankle servo targets to keep the CoM over the flat wedge shoes!
        dx = actStride * strideStroke + yawDiffStride;
        dy = actLift < 0 ? -actLift * liftStrokeUp : -actLift * liftStrokeDown;
        dz = enableHipRoll ? (actHipRoll * 0.048) : 0;
      } else {
        // Self-Righting Mode ("Замах и упор лапой за спину" when overturned, up.y < 0.42, actLift < 0):
        dx = actStride * 0.26;
        dy = 0.24 + (-actLift) * 0.28;
        const outSign = restFootPos.z >= 0 ? 1 : -1;
        dz = isServoMode
          ? (enableHipRoll ? actHipRoll * 0.065 : 0.0)
          : (isRearCenter ? diffTurn * 0.24 : -outSign * (-actLift) * 0.10);
      }

      let rawTargetPos = new Vec3(
        restFootPos.x + dx,
        restFootPos.y + dy,
        restFootPos.z + dz
      );

      // Physical STS3215 Servo Rotor Angular Speed / Slew-Rate Filter for 5-DoF Biped Legs:
      // Prevents instantaneous 20 cm step commands from launching the 1.88 kg biped 1 m into the air!
      if (buildWebbedFoot) {
        if (needDorsalReach) {
          this._smoothTargetPos = restFootPos.clone();
        } else if (!this._smoothTargetPos) {
          this._smoothTargetPos = rawTargetPos.clone();
        } else {
          const alpha = 0.15;
          this._smoothTargetPos.x += (rawTargetPos.x - this._smoothTargetPos.x) * alpha;
          this._smoothTargetPos.y += (rawTargetPos.y - this._smoothTargetPos.y) * alpha;
          this._smoothTargetPos.z += (rawTargetPos.z - this._smoothTargetPos.z) * alpha;
          rawTargetPos = this._smoothTargetPos.clone();
        }
      }

      // In servo mode, clamp targetPos strictly inside the 2-link rigid femur+tibia reach sphere (L_femur + L_tibia)
      let targetPos = rawTargetPos;
      if (isServoMode) {
        const ux = rawTargetPos.x - restHipPos.x;
        const uy = rawTargetPos.y - restHipPos.y;
        const uz = rawTargetPos.z - restHipPos.z;
        const rawDist = Math.hypot(ux, uy, uz);
        const maxReach = (L_femur + L_tibia) * 0.992;
        if (rawDist > maxReach) {
          const s = maxReach / rawDist;
          targetPos = new Vec3(
            restHipPos.x + ux * s,
            restHipPos.y + uy * s,
            restHipPos.z + uz * s
          );
        }
      }

      const targetLocalUp = targetPos.y - restHipPos.y;
      const crossingHipPlane =
        (targetLocalUp < -0.03 && footLocalUp > -0.05) ||
        (targetLocalUp > 0.03 && footLocalUp < 0.05);

      // Relax coplanar hip muscles while foot crosses the hip plane (footLocalUp ≈ 0)
      // so the coplanar snap-through barrier vanishes and dorsalMuscle smoothly pushes/pulls the foot across!
      kneeLift.maxForce = crossingHipPlane ? 14.0 : (isServoMode ? 175.0 : 135.0);
      strideFwd.maxForce = crossingHipPlane ? 18.0 : (isServoMode ? 155.0 : 120.0);
      strideBwd.maxForce = crossingHipPlane ? 18.0 : (isServoMode ? 155.0 : 120.0);
      latMuscle.maxForce = crossingHipPlane ? (isServoMode ? 45.0 : 18.0) : (isServoMode ? 185.0 : 110.0);
      if (oppLatMuscle) {
        oppLatMuscle.maxForce = crossingHipPlane ? (isServoMode ? 45.0 : 18.0) : (isServoMode ? 185.0 : 110.0);
      }

      kneeLift.targetAction = actLift;
      strideFwd.targetAction = actStride;
      latMuscle.targetAction = actHipRoll;

      kneeLift.setTargetLengthDirect(Vec3.dist(restHipPos, targetPos));
      strideFwd.setTargetLengthDirect(Vec3.dist(restFrontPos, targetPos));
      strideBwd.setTargetLengthDirect(Vec3.dist(restRearPos, targetPos));
      latMuscle.setTargetLengthDirect(Vec3.dist(restLatPos, targetPos));
      if (oppLatMuscle && restOppLatPos) {
        oppLatMuscle.setTargetLengthDirect(Vec3.dist(restOppLatPos, targetPos));
      }
      if (dorsalMuscle && restDorsalPos) {
        const footInverted = footLocalUp > -0.06;
        dorsalMuscle.maxForce = (isRightingMode || footInverted || crossingHipPlane) ? 195.0 : (isServoMode ? 115.0 : 0.0);
        dorsalMuscle.setTargetLengthDirect(Vec3.dist(restDorsalPos, targetPos));
      }

      if (isServoMode && knee) {
        const targetKneePos = computeKneeTarget3D(targetPos);
        kneeFrontServo.setTargetLengthDirect(Vec3.dist(restFrontPos, targetKneePos));
        kneeRearServo.setTargetLengthDirect(Vec3.dist(restRearPos, targetKneePos));
        kneeLatServo.setTargetLengthDirect(Vec3.dist(restLatPos, targetKneePos));
        if (kneeOppLatServo && restOppLatPos) {
          kneeOppLatServo.setTargetLengthDirect(Vec3.dist(restOppLatPos, targetKneePos));
        }
        if (kneeDorsalServo && restDorsalPos) {
          kneeDorsalServo.setTargetLengthDirect(Vec3.dist(restDorsalPos, targetKneePos));
        }

        if (webbedFoot) {
          // 5-DoF Ankle Pitch + Hip Yaw rotation of the Open_Duck_Mini flat wedge shoe:
          // - foot is the center ankle joint
          // - toeInner (+0.14 m front pad) and toeOuter (-0.12 m rear heel pad) rotate around the ankle!
          const yawRad = actHipYaw * 0.36;
          const cosY = Math.cos(yawRad);
          const sinY = Math.sin(yawRad);
          const inRx = restToeInnerOffset.x * cosY - restToeInnerOffset.z * sinY;
          const inRz = restToeInnerOffset.x * sinY + restToeInnerOffset.z * cosY;
          const outRx = restToeOuterOffset.x * cosY - restToeOuterOffset.z * sinY;
          const outRz = restToeOuterOffset.x * sinY + restToeOuterOffset.z * cosY;

          const ankleCmd = actAnklePitch !== null ? actAnklePitch : 0.0;
          const anklePitchY = -ankleCmd * 0.015;
          const targetToeIn = new Vec3(
            targetPos.x + inRx,
            targetPos.y + anklePitchY,
            targetPos.z + inRz
          );
          const targetToeOut = new Vec3(
            targetPos.x + outRx,
            targetPos.y - anklePitchY,
            targetPos.z + outRz
          );
          toeInnerFront.setTargetLengthDirect(Vec3.dist(restFrontPos, targetToeIn));
          toeInnerRear.setTargetLengthDirect(Vec3.dist(restRearPos, targetToeIn));
          toeInnerHip.setTargetLengthDirect(Vec3.dist(restHipPos, targetToeIn));
          toeInnerLat.setTargetLengthDirect(Vec3.dist(restLatPos, targetToeIn));
          if (toeInnerDorsal && restDorsalPos) {
            toeInnerDorsal.setTargetLengthDirect(Vec3.dist(restDorsalPos, targetToeIn));
          }

          toeOuterFront.setTargetLengthDirect(Vec3.dist(restFrontPos, targetToeOut));
          toeOuterRear.setTargetLengthDirect(Vec3.dist(restRearPos, targetToeOut));
          toeOuterHip.setTargetLengthDirect(Vec3.dist(restHipPos, targetToeOut));
          toeOuterLat.setTargetLengthDirect(Vec3.dist(restLatPos, targetToeOut));
          if (toeOuterDorsal && restDorsalPos) {
            toeOuterDorsal.setTargetLengthDirect(Vec3.dist(restDorsalPos, targetToeOut));
          }

          toeInnerHip.targetAction = ankleCmd;
          toeInnerHip.jointAngleRad = ankleCmd * 0.55;
          toeInnerHip.jointTorqueNm = Math.max(-15.0, Math.min(15.0, toeInnerHip.appliedForce * 0.16));
          toeInnerHip.torqueLoadRatio = Math.min(1.0, Math.abs(toeInnerHip.jointTorqueNm) / 15.0);

          toeOuterFront.targetAction = actHipYaw;
          toeOuterFront.jointAngleRad = yawRad;
          toeOuterFront.jointTorqueNm = Math.max(-14.0, Math.min(14.0, (toeOuterFront.appliedForce - toeInnerFront.appliedForce) * 0.14));
          toeOuterFront.torqueLoadRatio = Math.min(1.0, Math.abs(toeOuterFront.jointTorqueNm) / 14.0);

          // Aggregate ground contact across all 3 nodes of the flat duck sole (heel + inner toe + outer toe)
          if (webbedFoot.toeInner.inContact || webbedFoot.toeOuter.inContact) {
            foot.inContact = true;
            foot.normalForce = Math.max(
              foot.normalForce || 0,
              (webbedFoot.toeInner.normalForce || 0) + (webbedFoot.toeOuter.normalForce || 0)
            );
          }
        }

        // Compute Revolute PD Servo joint angle & torque telemetry (N*m) for robotics HUD & 3D drum rendering
        const curChord = Math.max(0.05, Math.min(L_femur + L_tibia, Vec3.dist(hipNode.pos, foot.pos)));
        const cosKnee = Math.max(-1, Math.min(1, (L_femur * L_femur + L_tibia * L_tibia - curChord * curChord) / (2 * L_femur * L_tibia)));
        kneeLift.isServo = true;
        kneeLift.jointAngleRad = Math.acos(cosKnee);
        kneeLift.maxTorqueNm = 22.0;
        kneeLift.jointTorqueNm = Math.max(-22.0, Math.min(22.0, kneeLift.appliedForce * L_tibia));
        kneeLift.torqueLoadRatio = Math.min(1.0, Math.abs(kneeLift.jointTorqueNm) / 22.0);

        strideFwd.isServo = true;
        strideFwd.jointAngleRad = actStride * 0.65;
        strideFwd.maxTorqueNm = 18.0;
        strideFwd.jointTorqueNm = Math.max(-18.0, Math.min(18.0, (strideFwd.appliedForce - strideBwd.appliedForce) * L_femur));
        strideFwd.torqueLoadRatio = Math.min(1.0, Math.abs(strideFwd.jointTorqueNm) / 18.0);

        latMuscle.isServo = true;
        latMuscle.jointAngleRad = actHipRoll * 0.45;
        latMuscle.maxTorqueNm = 18.0;
        latMuscle.jointTorqueNm = Math.max(-18.0, Math.min(18.0, latMuscle.appliedForce * L_femur));
        latMuscle.torqueLoadRatio = Math.min(1.0, Math.abs(latMuscle.jointTorqueNm) / 18.0);

        this.jointAngleRad = kneeLift.jointAngleRad;
        this.jointTorqueNm = kneeLift.jointTorqueNm;
        this.torqueLoadRatio = Math.max(kneeLift.torqueLoadRatio, strideFwd.torqueLoadRatio);
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

  const prefix = dofLabelPrefix || `Лапа ${legIdx + 1}`;
  const sideCode = legIdx === 0 ? 'L' : 'R';
  return {
    foot,
    knee,
    webbedFoot,
    controller,
    liftDoF: {
      name: isServoMode
        ? `${prefix}: Серво колена θ_knee (Н·м)`
        : `${prefix}: Сгиб колена (Подъём/Опора)`,
      primary: kneeLift,
      antagonist: null
    },
    strideDoF: {
      name: isServoMode
        ? `${prefix}: Серво бедра θ_hip (Н·м)`
        : `${prefix}: Мах бедра (Шаг вперёд/назад)`,
      primary: strideFwd,
      antagonist: strideBwd
    },
    duck5DoFs: buildWebbedFoot ? [
      {
        name: `${prefix}: hip_yaw (Поворот стопы)`,
        shortLabel: `${sideCode} hip_yaw`,
        primary: toeOuterFront,
        antagonist: toeInnerFront
      },
      {
        name: `${prefix}: hip_roll (Боковой перенос веса)`,
        shortLabel: `${sideCode} hip_roll`,
        primary: latMuscle,
        antagonist: null
      },
      {
        name: `${prefix}: hip_pitch (Мах бедра)`,
        shortLabel: `${sideCode} hip_pitch`,
        primary: strideFwd,
        antagonist: strideBwd
      },
      {
        name: `${prefix}: knee_pitch (Колено / Присед)`,
        shortLabel: `${sideCode} knee`,
        primary: kneeLift,
        antagonist: null
      },
      {
        name: `${prefix}: ankle_pitch (Стопа-ласта)`,
        shortLabel: `${sideCode} ankle`,
        primary: toeInnerHip,
        antagonist: toeOuterHip
      }
    ] : null
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
function buildTripodWalker({ id, origin = new Vec3(0, 0, 0), color = '#38bdf8', actuatorMode = 'servo' }) {
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

  const isServo = actuatorMode !== 'muscle';
  const noseNode = body.addNode({
    id: `c${id}_nose`,
    x: origin.x + (isServo ? 0.42 : 0.34),
    y: origin.y + (isServo ? 0.27 : 0.24),
    z: origin.z,
    mass: 0.18,
    radius: 0.060,
    friction: 0.04,
    role: 'head',
    color
  });

  const tailNode = body.addNode({
    id: `c${id}_tail`,
    x: origin.x - (isServo ? 0.46 : 0.36),
    y: origin.y + (isServo ? 0.27 : 0.24),
    z: origin.z,
    mass: 0.18,
    radius: 0.060,
    friction: 0.04,
    role: 'hip',
    color
  });

  // 3 Hip sockets in a wide, stable equilateral triangle so the torso never tips onto its tail.
  // In 'servo' mode, hip height is raised to 0.30 m (matching quadruped ground clearance) and rear hip is at x = -0.24 m
  // so the rear knee has a full 22 cm lever arm ahead of tailNode (-0.46 m)!
  const hipCoords = isServo
    ? [
        { x: +0.20, y: 0.30, z: +0.28 }, // Leg 0: Front-Left
        { x: +0.20, y: 0.30, z: -0.28 }, // Leg 1: Front-Right
        { x: -0.24, y: 0.30, z: 0.0 }    // Leg 2: Rear-Center
      ]
    : [
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
  const knees = [];
  const legControllers = [];
  const logicalDoFs = [];

  for (let i = 0; i < 3; i++) {
    const lateralAnchor = i === 0 ? hips[1] : i === 1 ? hips[0] : hips[0];
    const oppositeLateralAnchor = i === 2 ? hips[1] : null;
    const leg = attachOrthogonalLeg({
      body,
      id,
      legIdx: i,
      actuatorMode,
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
    if (leg.knee) knees.push(leg.knee);
    legControllers.push(leg.controller);
    logicalDoFs.push(leg.liftDoF, leg.strideDoF);
  }

  const creature = new CreatureInstance({
    id,
    morphologyKey: 'tripod',
    actuatorMode,
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
    kneeNodes: knees,
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
function buildQuadrupedWalker({ id, origin = new Vec3(0, 0, 0), color = '#a855f7', actuatorMode = 'servo' }) {
  const body = new ArticulatedBody();

  const torsoTop = body.addNode({
    id: `c${id}_top`,
    x: origin.x,
    y: origin.y + 0.46,
    z: origin.z,
    mass: 0.10,
    radius: 0.068,
    friction: 0.04,
    role: 'torso',
    color
  });

  const torsoBottom = body.addNode({
    id: `c${id}_bottom`,
    x: origin.x,
    y: origin.y + 0.14,
    z: origin.z,
    mass: 0.88,
    radius: 0.065,
    friction: 0.04,
    role: 'torso',
    color
  });

  const noseNode = body.addNode({
    id: `c${id}_nose`,
    x: origin.x + 0.44,
    y: origin.y + 0.24,
    z: origin.z,
    mass: 0.20,
    radius: 0.065,
    friction: 0.04,
    role: 'head',
    color
  });

  const tailNode = body.addNode({
    id: `c${id}_tail`,
    x: origin.x - 0.44,
    y: origin.y + 0.24,
    z: origin.z,
    mass: 0.20,
    radius: 0.065,
    friction: 0.04,
    role: 'hip',
    color
  });

  const hipCoords = [
    { x: +0.23, y: 0.28, z: +0.28 }, // Front-Left
    { x: +0.23, y: 0.28, z: -0.28 }, // Front-Right
    { x: -0.23, y: 0.28, z: +0.28 }, // Rear-Left
    { x: -0.23, y: 0.28, z: -0.28 }  // Rear-Right
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
  const knees = [];
  const legControllers = [];
  const logicalDoFs = [];
  for (let i = 0; i < 4; i++) {
    const oppositeLateral = i % 2 === 0 ? hips[i + 1] : hips[i - 1];
    const leg = attachOrthogonalLeg({
      body,
      id,
      legIdx: i,
      actuatorMode,
      hipNode: hips[i],
      frontAnchor: noseNode,
      rearAnchor: tailNode,
      lateralAnchor: oppositeLateral,
      dorsalAnchor: torsoTop,
      footY: origin.y + 0.06,
      color
    });
    feet.push(leg.foot);
    if (leg.knee) knees.push(leg.knee);
    legControllers.push(leg.controller);
    logicalDoFs.push(leg.liftDoF, leg.strideDoF);
  }

  const creature = new CreatureInstance({
    id,
    morphologyKey: 'quadruped',
    actuatorMode,
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
    kneeNodes: knees,
    legControllers,
    color,
    spawnOrigin: origin
  });
  creature.logicalDoFs = logicalDoFs;
  return creature;
}

/**
 * Morphology #3: Bipedal Servo Duck Walker ("Двуногий робот «Open Duck Mini / Микро-Утка»", 14 DoF, Revolute PD Servos)
 * Full 14-Servo kinematic compatibility with Open_Duck_Mini (apirrone/Open_Duck_Mini, Apache 2.0):
 * - Left Leg  (5 DoF: a[0..4]):  left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee_pitch, left_ankle_pitch
 * - Right Leg (5 DoF: a[5..9]):  right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee_pitch, right_ankle_pitch
 * - Neck & Head Counterbalance (4 DoF: a[10..13]): neck_pitch, head_pitch, head_yaw, head_roll
 * - Pure 2-legged bipedal stance on two wide 3-point flat duck feet (heel + inner toe + outer toe)
 * - Capable of rising tall on its legs (knee_pitch > 0) and squatting all the way down to sit on its bottom (torsoBottom)
 */
function buildMicroDuckWalker({ id, origin = new Vec3(0, 0, 0), color = '#fbbf24' }) {
  const duckActuatorMode = 'servo';
  const body = new ArticulatedBody();

  // Upper torso / neck base mount & low heavy pelvic seat ("попа" / battery keel)
  const torsoTop = body.addNode({
    id: `c${id}_top`,
    x: origin.x,
    y: origin.y + 0.46,
    z: origin.z,
    mass: 0.12,
    radius: 0.055,
    friction: 0.02,
    role: 'torso',
    color
  });

  const torsoBottom = body.addNode({
    id: `c${id}_bottom`,
    x: origin.x,
    y: origin.y + 0.145,
    z: origin.z,
    mass: 0.82,
    radius: 0.062,
    friction: 0.35, // Allows stable sitting on its bottom when knees squat!
    role: 'torso',
    color
  });

  const noseNode = body.addNode({
    id: `c${id}_nose`,
    x: origin.x + 0.36,
    y: origin.y + 0.28,
    z: origin.z,
    mass: 0.18,
    radius: 0.052,
    friction: 0.04,
    role: 'hip',
    color
  });

  const tailNode = body.addNode({
    id: `c${id}_tail`,
    x: origin.x - 0.36,
    y: origin.y + 0.28,
    z: origin.z,
    mass: 0.16,
    radius: 0.052,
    friction: 0.04,
    role: 'hip',
    color
  });

  // Left & Right 3-axis Hip Servo assemblies (hip_yaw + hip_roll + hip_pitch) — strictly 2 legs!
  const hipCoords = [
    { x: 0.00, y: 0.30, z: +0.22 }, // Leg 0: Left 5-DoF Duck Leg + Flat 3-Point Duck Foot
    { x: 0.00, y: 0.30, z: -0.22 }  // Leg 1: Right 5-DoF Duck Leg + Flat 3-Point Duck Foot
  ];

  const hips = hipCoords.map((hc, idx) =>
    body.addNode({
      id: `c${id}_hip_${idx}`,
      x: origin.x + hc.x,
      y: origin.y + hc.y,
      z: origin.z + hc.z,
      mass: 0.18,
      radius: 0.058,
      friction: 0.04,
      role: 'hip',
      color
    })
  );

  // Lock the 6-node pelvic torso cage into one rigid 3D body
  lockRigidFrame(body, [torsoTop, torsoBottom, noseNode, tailNode, hips[0], hips[1]]);

  // Physical Articulated Neck & Heavy BDX Head-Block Counterweight (4 DoF: neck_pitch, head_pitch, head_yaw, head_roll)
  const restTorsoTop = torsoTop.pos.clone();
  const restTorsoBottom = torsoBottom.pos.clone();
  const restNosePos = noseNode.pos.clone();
  const restTailPos = tailNode.pos.clone();
  const restHipLPos = hips[0].pos.clone();
  const restHipRPos = hips[1].pos.clone();

  const L_neck = 0.175;
  const baseNeckPitchRad = 0.30;
  const restHeadPos = new Vec3(
    restTorsoTop.x + L_neck * Math.sin(baseNeckPitchRad),
    restTorsoTop.y + L_neck * Math.cos(baseNeckPitchRad),
    restTorsoTop.z
  );

  const headNode = body.addNode({
    id: `c${id}_duck_head`,
    x: restHeadPos.x,
    y: restHeadPos.y,
    z: restHeadPos.z,
    mass: 0.28,
    radius: 0.062,
    friction: 0.12,
    role: 'duck_head',
    color: '#f8fafc'
  });

  // Rigid neck column link between torsoTop (neck_pitch base) and headNode (head_pitch / head_yaw / head_roll hub)
  body.addBone(torsoTop, headNode, L_neck, 0.0, 'neck');

  const neckPitchServo = body.addMuscle({
    name: 'duck_neck_pitch_servo',
    nodeA: noseNode,
    nodeB: headNode,
    pivotNode: torsoTop,
    minFactor: 0.25,
    maxFactor: 2.20,
    maxForce: 115.0,
    compliance: 0.00008
  });
  const headPitchServo = body.addMuscle({
    name: 'duck_head_pitch_servo',
    nodeA: tailNode,
    nodeB: headNode,
    pivotNode: torsoTop,
    minFactor: 0.25,
    maxFactor: 2.20,
    maxForce: 115.0,
    compliance: 0.00008
  });
  const headYawServo = body.addMuscle({
    name: 'duck_head_yaw_servo',
    nodeA: hips[0],
    nodeB: headNode,
    pivotNode: torsoTop,
    minFactor: 0.25,
    maxFactor: 2.20,
    maxForce: 115.0,
    compliance: 0.00008
  });
  const headRollServo = body.addMuscle({
    name: 'duck_head_roll_servo',
    nodeA: hips[1],
    nodeB: headNode,
    pivotNode: torsoTop,
    minFactor: 0.25,
    maxFactor: 2.20,
    maxForce: 115.0,
    compliance: 0.00008
  });
  const headVerticalServo = body.addMuscle({
    name: 'duck_head_vertical_servo',
    nodeA: torsoBottom,
    nodeB: headNode,
    pivotNode: torsoTop,
    minFactor: 0.35,
    maxFactor: 2.20,
    maxForce: 125.0,
    compliance: 0.00007
  });

  for (const s of [neckPitchServo, headPitchServo, headYawServo, headRollServo]) {
    s.isServo = true;
    s.jointAngleRad = 0.0;
    s.maxTorqueNm = 12.0;
    s.jointTorqueNm = 0.0;
    s.torqueLoadRatio = 0.0;
  }

  const feet = [];
  const knees = [];
  const webbedFeet = [];
  const legControllers = [];
  const logicalDoFs = [];
  const dofPrefixes = ['Лапа L', 'Лапа R'];

  for (let i = 0; i < 2; i++) {
    const lateralAnchor = i === 0 ? hips[1] : hips[0];
    const leg = attachOrthogonalLeg({
      body,
      id,
      legIdx: i,
      actuatorMode: duckActuatorMode,
      hipNode: hips[i],
      frontAnchor: noseNode,
      rearAnchor: tailNode,
      lateralAnchor,
      oppositeLateralAnchor: null,
      dorsalAnchor: torsoTop,
      isRearCenter: false,
      kneeBendSign: +1,          // Open_Duck_Mini_v2 forward-pointing knee (">"): thigh goes forward-down, shin goes backward-down to ankle!
      buildWebbedFoot: true,     // 3-point flat wedge shoe sole (center ankle + front toe pad + rear heel pad)
      enableHipRoll: true,       // Active hip_roll & hip_yaw 3D kinematics
      restFootOffsetX: -0.02,    // Ankle sits at x = -0.02 under hip, with front toe at +0.14 m and rear heel at -0.17 m!
      strideStroke: 0.165,       // Balanced Open_Duck_Mini waddle stride so feet stay underneath the CoM!
      liftStrokeUp: 0.095,       // Allows deep knee flexion to sit squarely on torsoBottom while keeping both flat shoes planted!
      liftStrokeDown: 0.115,     // Allows tall knee extension to rise high on both legs!
      dofLabelPrefix: dofPrefixes[i],
      footY: origin.y + 0.06,
      color
    });
    feet.push(leg.foot);
    if (leg.knee) knees.push(leg.knee);
    if (leg.webbedFoot) webbedFeet.push(leg.webbedFoot);
    legControllers.push(leg.controller);
    // Push 5 DoFs for this leg: [hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch]
    logicalDoFs.push(...leg.duck5DoFs);
  }

  // Append the 4 Neck & Head DoFs (indices 10..13) to complete the 14-DoF Open_Duck_Mini specification
  logicalDoFs.push(
    {
      name: 'Шея: neck_pitch (Продольный баланс)',
      shortLabel: 'neck_pitch',
      primary: neckPitchServo,
      antagonist: headPitchServo
    },
    {
      name: 'Голова: head_pitch (Кивок головы)',
      shortLabel: 'head_pitch',
      primary: headPitchServo,
      antagonist: neckPitchServo
    },
    {
      name: 'Голова: head_yaw (Поворот к цели)',
      shortLabel: 'head_yaw',
      primary: headYawServo,
      antagonist: headRollServo
    },
    {
      name: 'Голова: head_roll (Боковой контр-баланс)',
      shortLabel: 'head_roll',
      primary: headRollServo,
      antagonist: headYawServo
    }
  );

  const creature = new CreatureInstance({
    id,
    morphologyKey: 'duck',
    actuatorMode: duckActuatorMode,
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
    kneeNodes: knees,
    legControllers,
    color,
    spawnOrigin: origin
  });
  creature.logicalDoFs = logicalDoFs;
  creature.webbedFeet = webbedFeet;
  creature.headNode = headNode;
  creature.isDuck = true;

  // Custom 14-DoF Action Dispatcher for Open_Duck_Mini (100% pure XPBD physics — zero artificial coordinate rotations!):
  // a[0..4]  = Left Leg  (hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch)
  // a[5..9]  = Right Leg (hip_yaw, hip_roll, right_hip_pitch, right_knee_pitch, right_ankle_pitch)
  // a[10..13]= Neck/Head (neck_pitch, head_pitch, head_yaw, head_roll)
  creature.customApplyActions = (actions, isRightingMode = false) => {
    const clamp1 = v => Math.max(-1, Math.min(1, v || 0));
    const { up, forward, right } = creature.getTorsoFrame();
    let totalPower = 0;

    // 1. Left Leg (5 DoF: actions[0..4])
    const lHipYaw = clamp1(actions[0]);
    const lHipRoll = clamp1(actions[1]);
    const lHipPitch = clamp1(actions[2]);
    const lKneePitch = clamp1(actions[3]);
    const lAnklePitch = clamp1(actions[4]);

    // 2. Right Leg (5 DoF: actions[5..9])
    const rHipYaw = clamp1(actions[5]);
    const rHipRoll = clamp1(actions[6]);
    const rHipPitch = clamp1(actions[7]);
    const rKneePitch = clamp1(actions[8]);
    const rAnklePitch = clamp1(actions[9]);

    const diffTurn = 0.5 * (rHipPitch - lHipPitch);

    totalPower += legControllers[0].applyLegActions(
      lKneePitch,
      lHipPitch,
      diffTurn,
      isRightingMode,
      up,
      forward,
      right,
      lHipRoll,
      lHipYaw,
      lAnklePitch
    );
    totalPower += legControllers[1].applyLegActions(
      rKneePitch,
      rHipPitch,
      diffTurn,
      isRightingMode,
      up,
      forward,
      right,
      rHipRoll,
      rHipYaw,
      rAnklePitch
    );

    // 3. Articulated Neck & Head Counterbalance (4 DoF: actions[10..13])
    const actNeckPitch = clamp1(actions[10]);
    const actHeadPitch = clamp1(actions[11]);
    const actHeadYaw = clamp1(actions[12]);
    const actHeadRoll = clamp1(actions[13]);

    const thetaPitch = baseNeckPitchRad + actNeckPitch * 0.45 + actHeadPitch * 0.25;
    const thetaRoll = actHeadRoll * 0.35 + actHeadYaw * 0.18;
    const targetHeadPos = new Vec3(
      restTorsoTop.x + L_neck * Math.sin(thetaPitch) * Math.cos(thetaRoll),
      restTorsoTop.y + L_neck * Math.cos(thetaPitch) * Math.cos(thetaRoll),
      restTorsoTop.z + L_neck * Math.sin(thetaRoll)
    );

    neckPitchServo.targetAction = actNeckPitch;
    headPitchServo.targetAction = actHeadPitch;
    headYawServo.targetAction = actHeadYaw;
    headRollServo.targetAction = actHeadRoll;

    neckPitchServo.setTargetLengthDirect(Vec3.dist(restNosePos, targetHeadPos));
    headPitchServo.setTargetLengthDirect(Vec3.dist(restTailPos, targetHeadPos));
    headYawServo.setTargetLengthDirect(Vec3.dist(restHipLPos, targetHeadPos));
    headRollServo.setTargetLengthDirect(Vec3.dist(restHipRPos, targetHeadPos));
    headVerticalServo.setTargetLengthDirect(Vec3.dist(restTorsoBottom, targetHeadPos));

    neckPitchServo.jointAngleRad = actNeckPitch * 0.55;
    neckPitchServo.jointTorqueNm = Math.max(-12.0, Math.min(12.0, (neckPitchServo.appliedForce - headPitchServo.appliedForce) * L_neck));
    neckPitchServo.torqueLoadRatio = Math.min(1.0, Math.abs(neckPitchServo.jointTorqueNm) / 12.0);

    headPitchServo.jointAngleRad = actHeadPitch * 0.45;
    headPitchServo.jointTorqueNm = Math.max(-12.0, Math.min(12.0, headPitchServo.appliedForce * L_neck));
    headPitchServo.torqueLoadRatio = Math.min(1.0, Math.abs(headPitchServo.jointTorqueNm) / 12.0);

    headYawServo.jointAngleRad = actHeadYaw * 0.65;
    headYawServo.jointTorqueNm = Math.max(-12.0, Math.min(12.0, (headYawServo.appliedForce - headRollServo.appliedForce) * L_neck));
    headYawServo.torqueLoadRatio = Math.min(1.0, Math.abs(headYawServo.jointTorqueNm) / 12.0);

    headRollServo.jointAngleRad = actHeadRoll * 0.50;
    headRollServo.jointTorqueNm = Math.max(-12.0, Math.min(12.0, headRollServo.appliedForce * L_neck));
    headRollServo.torqueLoadRatio = Math.min(1.0, Math.abs(headRollServo.jointTorqueNm) / 12.0);

    totalPower +=
      neckPitchServo.instantPower +
      headPitchServo.instantPower +
      headYawServo.instantPower +
      headRollServo.instantPower +
      headVerticalServo.instantPower;

    return totalPower;
  };

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
  },
  duck: {
    key: 'duck',
    name: 'Конструкция №3: Двуногий робот «Микро-Утка / Open Duck Mini» (Biped, 14 DoF)',
    shortName: 'Микро-Утка (Biped, 14 DoF)',
    description: 'Двуногий робот на 14 жёстких сервоприводах (Revolute PD, Н·м) с полной по-суставной совместимостью со стандартом Open Duck Mini: по 5 сервоприводов в каждой лапе (hip_yaw, hip_roll, hip_pitch, knee, ankle) + 4 сервопривода в шее и голове-балансире (neck_pitch, head_pitch, head_yaw, head_roll). Способен ходить вразвалочку, высоко приподниматься на лапах и садиться на попу.',
    obsDim: 46,
    actDim: 14,
    build: buildMicroDuckWalker
  }
};
