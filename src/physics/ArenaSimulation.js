import { PhysicsWorld3D, Vec3 } from './Physics3D.js';
import { MORPHOLOGY_REGISTRY } from '../morphologies/Morphologies.js';
import { ActorCriticBrain, PPOTrainer, NESTrainer } from '../brains/NeuralRL.js';

const WORKER_PALETTE = [
  '#fbbf24', // Worker 0: Gold Leader (Deterministic / Low-noise Policy)
  '#38bdf8',
  '#34d399',
  '#a78bfa',
  '#f472b6',
  '#60a5fa',
  '#4ade80',
  '#c084fc',
  '#fb7185',
  '#2dd4bf',
  '#818cf8',
  '#facc15',
  '#fb923c',
  '#22d3ee',
  '#a3e635',
  '#e879f9'
];

export class ArenaSimulation {
  constructor({
    morphologyKey = 'tripod',
    actuatorMode = 'servo',
    algorithm = 'ppo',
    populationSize = 1,
    arenaHalfSize = 11.0,
    floorFriction = 1.0
  } = {}) {
    this.morphologyKey = morphologyKey;
    this.actuatorMode = actuatorMode; // 'servo' (default in v2) | 'muscle' (v1 bio-muscles)
    this.algorithm = algorithm;
    this.populationSize = populationSize;
    this.arenaHalfSize = arenaHalfSize;
    this.floorFriction = floorFriction;

    this.controlDt = 0.05; // 20 Hz neural control tick
    this.substepsPerControl = 10;
    this.subDt = this.controlDt / this.substepsPerControl; // 0.005s (200 Hz physics)
    this.substepCursor = 0;
    this.cpgOmega = 2 * Math.PI * 1.65; // 1.65 Hz lively bounding stride frequency
    this.simTime = 0;
    this.totalSteps = 0;
    this.totalFoodCollected = 0;
    this.bestCreatureIdx = 0;

    // Telemetry history for live charts and scientific logs
    this.history = [];
    this.recentStepRewards = [];
    this.recentVelocities = [];
    this.c0Trajectory = [];
    this.c0FoodEvents = [];
    this.c0OverturnEvents = [];
    this.c0LastFoodSpawnTime = 0;
    this.c0CurrentlyOverturned = false;

    this._initSimulation();
  }

  _initSimulation() {
    const morph = MORPHOLOGY_REGISTRY[this.morphologyKey] || MORPHOLOGY_REGISTRY.tripod;
    this.morphSpec = morph;

    this.world = new PhysicsWorld3D({
      gravity: -9.81,
      floorFriction: this.floorFriction,
      arenaHalfSize: this.arenaHalfSize,
      substeps: 10
    });

    this.brain = new ActorCriticBrain(morph.obsDim, morph.actDim, 32, 24);
    if (this.algorithm === 'nes') {
      this.trainer = new NESTrainer(this.brain, this.populationSize);
    } else {
      this.trainer = new PPOTrainer(this.brain, this.populationSize);
    }

    this.creatures = [];
    for (let i = 0; i < this.populationSize; i++) {
      const origin = this._computeSpawnOrigin(i);
      const color = WORKER_PALETTE[i % WORKER_PALETTE.length];
      const creature = morph.build({ id: i, origin, color, actuatorMode: this.actuatorMode });
      this.world.addBody(creature.body);
      this.creatures.push(creature);
      this._spawnFoodForCreature(creature, true);
    }
  }

  _computeSpawnOrigin(idx) {
    if (idx === 0) {
      return new Vec3(0, 0, 0);
    }
    // Arrange other workers in two concentric rings on the same arena
    const ring = idx <= 6 ? 1 : 2;
    const countInRing = ring === 1 ? Math.min(6, this.populationSize - 1) : this.populationSize - 7;
    const posInRing = ring === 1 ? idx - 1 : idx - 7;
    const radius = ring === 1 ? 2.4 : 4.6;
    const angle = (2 * Math.PI * posInRing) / Math.max(1, countInRing);
    return new Vec3(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
  }

  _spawnFoodForCreature(creature, initial = false) {
    const com = creature.body.getCenterOfMass();
    // Adaptive curriculum distance: starts at 1.15..1.85 m, grows with foodCollected up to 5.0 m
    const minDist = 1.15 + Math.min(2.2, creature.foodCollected * 0.25);
    const maxDist = 1.85 + Math.min(3.5, creature.foodCollected * 0.45);
    const dist = minDist + Math.random() * (maxDist - minDist);

    // On first spawn, place food generally in front (+X with +/- 35 deg spread) so initial gradient is rich
    const angle = initial
      ? (Math.random() - 0.5) * 1.1
      : Math.random() * Math.PI * 2;

    const margin = 1.4;
    const bound = this.arenaHalfSize - margin;
    let fx = com.x + Math.cos(angle) * dist;
    let fz = com.z + Math.sin(angle) * dist;

    // Keep food inside arena walls
    if (Math.abs(fx) > bound || Math.abs(fz) > bound) {
      fx = Math.max(-bound, Math.min(bound, com.x * 0.4 + (Math.random() - 0.5) * 4.0));
      fz = Math.max(-bound, Math.min(bound, com.z * 0.4 + (Math.random() - 0.5) * 4.0));
    }

    creature.foodPos.set(fx, 0.16, fz);
    creature.prevFoodDist = Math.hypot(fx - com.x, fz - com.z);
  }

  _resetCreatureBody(idx) {
    const oldCreature = this.creatures[idx];
    const origin = this._computeSpawnOrigin(idx);
    const fresh = this.morphSpec.build({
      id: idx,
      origin,
      color: oldCreature.color,
      actuatorMode: this.actuatorMode
    });
    fresh.foodCollected = oldCreature.foodCollected;
    fresh.cumulativeReward = oldCreature.cumulativeReward;
    this.world.bodies[idx] = fresh.body;
    this.creatures[idx] = fresh;
    this._spawnFoodForCreature(fresh, false);
  }

  /**
   * Gently flips an overturned or stuck creature right back onto its feet at its exact current
   * (x, z) floor location and horizontal heading, preserving 100% of its learned neural network,
   * statistics, and current food target!
   */
  uprightCreature(idx = 0) {
    const c = this.creatures[idx];
    if (!c) return;

    const com = c.body.getCenterOfMass();
    // Compute horizontal heading (preserve current horizontal nose->tail direction, or face toward food)
    const nose = c.frameRefs.front.pos;
    const tail = c.frameRefs.tail ? c.frameRefs.tail.pos : com;
    let hx = nose.x - tail.x;
    let hz = nose.z - tail.z;
    if (Math.hypot(hx, hz) < 0.05) {
      hx = c.foodPos.x - com.x;
      hz = c.foodPos.z - com.z;
    }
    const yaw = Math.atan2(hz, hx);
    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);

    // Build a temporary canonical reference at (0, 0, 0) to get exact upright node offsets
    const template = this.morphSpec.build({
      id: idx,
      origin: new Vec3(0, 0, 0),
      color: c.color,
      actuatorMode: this.actuatorMode
    });

    // Apply Euclidean rigid rotation (around Y) + translation to (com.x, 0.015, com.z)
    const margin = 0.6;
    const bound = this.arenaHalfSize - margin;
    const cx = Math.max(-bound, Math.min(bound, com.x));
    const cz = Math.max(-bound, Math.min(bound, com.z));

    for (let i = 0; i < c.body.nodes.length; i++) {
      const refPos = template.body.nodes[i].pos;
      const rx = refPos.x * cosY - refPos.z * sinY;
      const rz = refPos.x * sinY + refPos.z * cosY;
      const ry = refPos.y + 0.01;

      const node = c.body.nodes[i];
      node.pos.set(cx + rx, ry, cz + rz);
      node.prevPos.copy(node.pos);
      node.vel.set(0, 0, 0);
      node.inContact = false;
      node.normalForce = 0;
      node.frictionForceVec.set(0, 0, 0);
    }

    for (let m = 0; m < c.body.muscles.length; m++) {
      const mus = c.body.muscles[m];
      mus.targetLength = mus.baseLength;
      mus.directTargetLength = mus.baseLength;
      mus.targetAction = 0;
      mus.smoothedAction = 0;
      mus.lambda = 0;
    }

    c.postThrowTimer = 0;
    if (c.lastAction) {
      c.lastAction.fill(0);
      c.applyActionVector(c.lastAction, false);
    }
    if (this.trainer) {
      this.trainer.overturnConfirmSteps = 0;
      this.trainer.rightingPauseStepsRemaining = 0;
      if (this.trainer.wasOverturned) {
        this.trainer.wasOverturned = false;
        this.trainer.metaState.mode = this.trainer.modeBeforeOverturn || 'LEARNING_WALK';
        this.trainer.metaState.isLearningActive = this.trainer.metaState.mode === 'LEARNING_WALK';
      }
    }

    c.prevFoodDist = Math.hypot(c.foodPos.x - cx, c.foodPos.z - cz);
  }

  /**
   * Moves the food orb to a new floor coordinate (fx, fz), clamped inside the arena walls.
   */
  moveFoodTo(idx = 0, fx, fz, isDragging = false) {
    const c = this.creatures[idx];
    if (!c) return;
    const bound = this.arenaHalfSize - 0.65;
    const clampedX = Math.max(-bound, Math.min(bound, fx));
    const clampedZ = Math.max(-bound, Math.min(bound, fz));
    c.foodPos.set(clampedX, isDragging ? 0.24 : 0.16, clampedZ);
    c.foodState = 'ACTIVE';
    c.foodStateTimer = 0.0;
    c.postFoodTurnGraceTimer = 8.0;
    const com = c.body.getCenterOfMass();
    c.prevFoodDist = Math.hypot(clampedX - com.x, clampedZ - com.z);
  }

  /**
   * Picks up the walker with the user's mouse cursor, lifting it gently above the floor.
   */
  grabCreature(idx = 0) {
    const c = this.creatures[idx];
    if (!c) return;
    c.isGrabbed = true;
    const com = c.body.getCenterOfMass();
    c.grabSavedOffsets = c.body.nodes.map(n => new Vec3(
      n.pos.x - com.x,
      n.pos.y - com.y,
      n.pos.z - com.z
    ));
    c.grabTargetPos = new Vec3(com.x, Math.max(0.46, com.y + 0.28), com.z);
    for (let i = 0; i < c.body.nodes.length; i++) {
      const n = c.body.nodes[i];
      const off = c.grabSavedOffsets[i];
      n.pos.set(c.grabTargetPos.x + off.x, Math.max(0.06, c.grabTargetPos.y + off.y), c.grabTargetPos.z + off.z);
      n.prevPos.copy(n.pos);
      n.vel.set(0, 0, 0);
      n.inContact = false;
      n.normalForce = 0;
    }
    c.prevFoodDist = Math.hypot(c.foodPos.x - c.grabTargetPos.x, c.foodPos.z - c.grabTargetPos.z);
  }

  /**
   * Drags the grabbed walker to (targetX, targetZ) in mid-air with dynamic inertial tilt from pointer velocity (vx, vz).
   */
  dragCreatureTo(idx = 0, targetX, targetZ, vx = 0, vz = 0) {
    const c = this.creatures[idx];
    if (!c || !c.isGrabbed) return;
    const bound = this.arenaHalfSize - 0.65;
    const cx = Math.max(-bound, Math.min(bound, targetX));
    const cz = Math.max(-bound, Math.min(bound, targetZ));
    const cy = 0.48;
    if (!c.grabTargetPos) c.grabTargetPos = new Vec3(cx, cy, cz);
    c.grabTargetPos.set(cx, cy, cz);
    c.grabDragVel = { vx, vz };

    if (c.grabSavedOffsets) {
      // Subtle pendulum swing tilt when dragging fast with the mouse
      const tiltX = Math.max(-0.55, Math.min(0.55, vx * 0.09));
      const tiltZ = Math.max(-0.55, Math.min(0.55, vz * 0.09));
      for (let i = 0; i < c.body.nodes.length; i++) {
        const n = c.body.nodes[i];
        const off = c.grabSavedOffsets[i];
        const rx = off.x - off.y * tiltX;
        const rz = off.z - off.y * tiltZ;
        const ry = off.y + (off.x * tiltX + off.z * tiltZ) * 0.5;
        n.pos.set(cx + rx, Math.max(0.06, cy + ry), cz + rz);
        n.prevPos.copy(n.pos);
        n.vel.set(0, 0, 0);
        n.inContact = false;
        n.normalForce = 0;
      }
    }
    c.prevFoodDist = Math.hypot(c.foodPos.x - cx, c.foodPos.z - cz);
  }

  /**
   * Releases the grabbed walker at its current mid-air position, imparting the mouse throw velocity
   * and (if tossed briskly) an angular tumble impulse so it can roll or land on its back/side.
   */
  releaseCreature(idx = 0, throwVx = 0, throwVz = 0) {
    const c = this.creatures[idx];
    if (!c) return;
    c.isGrabbed = false;
    const subDt = this.subDt;
    const rawSpeed = Math.hypot(throwVx, throwVz);
    const maxThrowSpeed = 4.2;
    const scale = rawSpeed > maxThrowSpeed ? maxThrowSpeed / rawSpeed : 1.0;
    const vx = throwVx * scale;
    const vz = throwVz * scale;
    const speed = Math.hypot(vx, vz);
    const vy = speed > 0.6 ? Math.min(1.35, speed * 0.26) : -0.15;

    const com = c.body.getCenterOfMass();
    // If tossed briskly (speed > 0.65 m/s), impart angular tumble so the walker flips in the air
    const tumbleOmega = speed > 0.65 ? Math.min(10.5, (speed - 0.45) * 4.2) : 0.0;
    const dirX = speed > 1e-4 ? vx / speed : 1.0;
    const dirZ = speed > 1e-4 ? vz / speed : 0.0;

    for (let i = 0; i < c.body.nodes.length; i++) {
      const n = c.body.nodes[i];
      const rx = n.pos.x - com.x;
      const ry = n.pos.y - com.y;
      const rz = n.pos.z - com.z;
      // Rotational velocity v_rot = omega_vec x r, where omega_vec rolls the body along the throw direction
      const rotVx = dirX * (ry * tumbleOmega);
      const rotVz = dirZ * (ry * tumbleOmega);
      const rotVy = -(rx * dirX + rz * dirZ) * tumbleOmega;

      const nodeVx = vx + rotVx;
      const nodeVy = vy + rotVy;
      const nodeVz = vz + rotVz;
      n.prevPos.set(
        n.pos.x - nodeVx * subDt,
        n.pos.y - nodeVy * subDt,
        n.pos.z - nodeVz * subDt
      );
      n.vel.set(nodeVx, nodeVy, nodeVz);
    }
    c.grabSavedOffsets = null;
    c.grabTargetPos = null;
    c.postThrowTimer = 0.55;
    c.prevFoodDist = Math.hypot(c.foodPos.x - com.x, c.foodPos.z - com.z);
  }

  setFloorFriction(mu) {
    const nextMu = Math.max(0, Math.min(2.0, mu));
    const prevNotified = this._lastNotifiedFriction !== undefined ? this._lastNotifiedFriction : this.floorFriction;
    const isConsolidated = Boolean(this.trainer && this.trainer.metaState && this.trainer.metaState.mode === 'CONSOLIDATED');
    if (this.creatures) {
      for (const c of this.creatures) {
        c.postFoodTurnGraceTimer = 0;
        c.isPausedOrTurningToFood = false;
      }
    }
    if (
      this.trainer &&
      this.trainer.notifyEnvironmentChange &&
      (Math.abs(nextMu - prevNotified) >= 0.04 || (nextMu < 0.78 && isConsolidated))
    ) {
      this._lastNotifiedFriction = nextMu;
      this.trainer.notifyEnvironmentChange(
        `Изменилось трение пола (μ = ${nextMu.toFixed(2)}) — адаптация походки к новому сцеплению`,
        nextMu
      );
    }
    this.floorFriction = nextMu;
    this.world.floorFriction = this.floorFriction;
  }

  /**
   * Advances the simulation by a single 200Hz physics substep (subDt = 0.005s).
   * At substep 0, queries the neural network for target actions;
   * at substep 10, evaluates rewards and triggers PPO/NES learning when ready.
   * This allows ultra-smooth slow-motion inspection (0.05x, 0.1x, 0.25x, 0.5x, 1.0x)
   * with 100% identical deterministic physics!
   */
  stepPhysicsSubstep() {
    const dt = this.controlDt;
    const subDt = this.subDt;

    if (this.substepCursor === 0) {
      this.totalSteps += 1;
      for (let i = 0; i < this.populationSize; i++) {
        this.creatures[i].phase = (this.creatures[i].phase + this.cpgOmega * dt) % (Math.PI * 2);
        this.creatures[i].ageTime += dt;
        this.creatures[i].episodeSteps += 1;
      }
      this.trainer.selectActions(this.creatures);
      this.world.resetContactTelemetry((body, idx) => {
        const c = this.creatures[idx];
        if (c && c.lastAction) {
          const isActivelyRighting = Boolean(
            this.trainer &&
            this.trainer.wasOverturned &&
            !(this.trainer.rightingPauseStepsRemaining > 0)
          );
          c.applyActionVector(c.lastAction, isActivelyRighting);
        }
      });
    }

    this.world.stepSingleSubstep(subDt);
    // Keep any mouse-grabbed creature suspended at the user's cursor in mid-air
    for (let i = 0; i < this.populationSize; i++) {
      const c = this.creatures[i];
      if (c.isGrabbed && c.grabTargetPos && c.grabSavedOffsets) {
        const vx = c.grabDragVel ? c.grabDragVel.vx : 0;
        const vz = c.grabDragVel ? c.grabDragVel.vz : 0;
        this.dragCreatureTo(i, c.grabTargetPos.x, c.grabTargetPos.z, vx, vz);
      } else if (c.postThrowTimer > 0) {
        c.postThrowTimer = Math.max(0, c.postThrowTimer - subDt);
      }
    }
    this.simTime += subDt;
    this.substepCursor += 1;

    // Advance 3-phase animated food lifecycle (1s DISSOLVING -> 1s WAITING -> 1s SKY-DROPPING -> ACTIVE)
    if (!this._isHeadlessFastTrain) {
      for (let i = 0; i < this.populationSize; i++) {
        const c = this.creatures[i];
        if (c.foodState === 'DISSOLVING') {
          c.foodStateTimer += subDt;
          if (c.foodStateTimer >= 1.0) {
            c.foodState = 'WAITING';
            c.foodStateTimer = 0.0;
          }
        } else if (c.foodState === 'WAITING') {
          c.foodStateTimer += subDt;
          if (c.foodStateTimer >= 1.0) {
            this._spawnFoodForCreature(c, false);
            c.foodState = 'DROPPING';
            c.foodStateTimer = 0.0;
            c.foodPos.y = 3.20;
          }
        } else if (c.foodState === 'DROPPING') {
          c.foodStateTimer += subDt;
          const p = Math.min(1.0, c.foodStateTimer / 1.0);
          if (p <= 0.72) {
            const u = p / 0.72;
            c.foodPos.y = 0.16 + 3.04 * (1.0 - u * u);
          } else {
            const v = (p - 0.72) / 0.28;
            c.foodPos.y = 0.16 + 0.42 * 4.0 * v * (1.0 - v);
          }
          if (c.foodStateTimer >= 1.0) {
            c.foodPos.y = 0.16;
            c.foodState = 'ACTIVE';
            c.foodStateTimer = 0.0;
          }
        }
      }
    }

    if (this.substepCursor < this.substepsPerControl) {
      return null;
    }

    this.substepCursor = 0;
    return this._completeControlStep(dt);
  }

  /**
   * Executes one full neural control step (dt = 0.05s = 10 physics substeps).
   */
  stepControl() {
    let result = null;
    do {
      result = this.stepPhysicsSubstep();
    } while (this.substepCursor !== 0);
    return result;
  }

  _completeControlStep(dt) {
    // 3. Evaluate physical rewards, food collection, and posture termination
    const rewards = new Float64Array(this.populationSize);
    const dones = new Uint8Array(this.populationSize);

    let stepVelSum = 0;
    let stepRewSum = 0;

    for (let i = 0; i < this.populationSize; i++) {
      const c = this.creatures[i];
      const { center, up, forward } = c.getTorsoFrame();

      const dx = c.foodPos.x - center.x;
      const dz = c.foodPos.z - center.z;
      const newDist = Math.hypot(dx, dz);

      // Heading alignment toward food
      const invD = 1 / Math.max(1e-5, newDist);
      const alignCos = (dx * invD) * forward.x + (dz * invD) * forward.z;
      c.alignToFood = alignCos;

      // Velocity toward food (m/s)
      const isInThrowFlight = (c.postThrowTimer || 0) > 0;
      const isOverturnedNow = !c.isGrabbed && !isInThrowFlight && Boolean(this.trainer && this.trainer.wasOverturned);
      const isWaitingForFood = c.foodState === 'DISSOLVING' || c.foodState === 'WAITING';
      const isDroppingFood = c.foodState === 'DROPPING';

      const prevAlign = c.prevAlignToFood !== undefined ? c.prevAlignToFood : alignCos;
      const alignRate = (alignCos - prevAlign) / dt;
      c.prevAlignToFood = alignCos;

      if (isWaitingForFood || isDroppingFood) {
        // Give the walker a generous grace window after the new orb lands to pivot/turn toward it and accelerate
        c.postFoodTurnGraceTimer = 8.0;
      } else if (c.postFoodTurnGraceTimer > 0) {
        if (alignCos < 0.82 && alignRate > 0.03) {
          // Walker is actively pivoting toward the newly spawned target — decay timer slowly so it has all the time it needs
          c.postFoodTurnGraceTimer = Math.max(0, c.postFoodTurnGraceTimer - dt * 0.25);
        } else if (alignCos < 0.82) {
          c.postFoodTurnGraceTimer = Math.max(0, c.postFoodTurnGraceTimer - dt);
        } else {
          c.postFoodTurnGraceTimer = Math.max(0, c.postFoodTurnGraceTimer - dt * 2.5);
        }
      }

      const rawVelToward = (c.isGrabbed || isInThrowFlight || isWaitingForFood || isOverturnedNow)
        ? 0
        : (c.prevFoodDist - newDist) / dt;
      c.prevFoodDist = newDist;
      c.lastVelocityTowardFood = rawVelToward;

      // Smooth low-pass filtered velocity toward target (for HUD display)
      if (c.isGrabbed || isInThrowFlight || isWaitingForFood || isOverturnedNow) {
        c.smoothedVelocityTowardFood = 0.96 * c.smoothedVelocityTowardFood;
      } else {
        c.smoothedVelocityTowardFood = 0.965 * c.smoothedVelocityTowardFood + 0.035 * rawVelToward;
      }

      // Whether the walker is currently standing without a target or turning in place toward a newly spawned orb:
      // Meta-plasticity ("Чутьё") must NOT mistake waiting for food or turning in place for a loss of walking skill!
      c.isPausedOrTurningToFood =
        isWaitingForFood ||
        isDroppingFood ||
        (c.postFoodTurnGraceTimer > 0 && (alignCos < 0.82 || rawVelToward < 0.14));

      const velToward = rawVelToward;
      stepVelSum += velToward;

      const prevUpY = c.prevUpY !== undefined ? c.prevUpY : up.y;
      c.prevUpY = up.y;

      let r = 0;
      if (c.isGrabbed || isInThrowFlight) {
        c.lastVelocityTowardFood = 0;
        r = 0;
      } else if (isOverturnedNow) {
        // When overturned on its back: ZERO thoughts about food or walking!
        // Reward is 100% dedicated to the Beetle Self-Righting task: increasing up.y toward +1
        // and leveraging feet against the floor to roll back onto its feet.
        const upVel = (up.y - prevUpY) / dt;
        const contactBonus = c.footNodes.some(f => f.inContact) ? 0.18 : 0.0;
        r = 3.5 * upVel + 0.35 * up.y + contactBonus;
        c.lastVelocityTowardFood = 0;
        dones[i] = 1;
      } else if (isWaitingForFood) {
        // During the 1s dissolve + 1s pause between food orbs, hold distance reference steady
        c.prevFoodDist = newDist;
        c.lastVelocityTowardFood = 0;
        r = 0.08 * Math.max(0, up.y - 0.55);
      } else {
        // Normal upright walking reward components:
        // 1) Approach velocity reward (favors moving nose-first toward food)
        const forwardFactor = Math.max(0.25, 0.65 + 0.35 * alignCos);
        r = velToward * forwardFactor * 2.5;
        // 2) Orientation alignment reward
        r += 0.22 * alignCos;
        // 3) Upright posture maintenance + self-inflicted overturn penalty (teaches walker to keep balance!)
        r += up.y >= 0.55 ? 0.08 * (up.y - 0.55) : 0.45 * (up.y - 0.55);
        if (up.y < 0.22) {
          r -= 1.65;
          dones[i] = 1;
        }
        // 4) Mechanical power penalty (smooth, efficient locomotion)
        r -= 0.00035 * c.lastMechanicalPower;

        // Check if food reached! (center or front hip within reach radius, and orb is near ground level)
        const frontDist = Math.hypot(c.foodPos.x - c.frameRefs.front.pos.x, c.foodPos.z - c.frameRefs.front.pos.z);
        if ((newDist < 0.46 || frontDist < 0.38) && c.foodPos.y <= 0.65) {
          r += 8.5; // Major foraging reward
          c.foodCollected += 1;
          this.totalFoodCollected += 1;
          if (i === 0) {
            this.c0FoodEvents.push({
              foodNumber: c.foodCollected,
              simTime: Number(this.simTime.toFixed(2)),
              update: this.trainer.updateCount,
              foodPos: [Number(c.foodPos.x.toFixed(2)), Number(c.foodPos.z.toFixed(2))],
              timeToReachSec: Number((this.simTime - this.c0LastFoodSpawnTime).toFixed(2)),
              metaMode: this.trainer.metaState ? this.trainer.metaState.mode : 'LEARNING_WALK'
            });
            this.c0LastFoodSpawnTime = this.simTime;
          }
          if (this._isHeadlessFastTrain) {
            this._spawnFoodForCreature(c, false);
            c.foodState = 'ACTIVE';
            c.foodStateTimer = 0;
          } else {
            // Start 1.0s animated dissolve at current spot, followed by 1.0s pause, then 1.0s sky drop!
            c.dissolvePos.copy(c.foodPos);
            c.foodState = 'DISSOLVING';
            c.foodStateTimer = 0.0;
          }
        }
      }

      if (i === 0) {
        if (isOverturnedNow && !this.c0CurrentlyOverturned) {
          this.c0CurrentlyOverturned = true;
          this.c0OverturnEvents.push({
            event: 'overturned',
            simTime: Number(this.simTime.toFixed(2)),
            update: this.trainer.updateCount,
            pos: [Number(center.x.toFixed(2)), Number(center.z.toFixed(2))],
            upY: Number(up.y.toFixed(3)),
            rightingSkillPct: this.trainer.metaState ? this.trainer.metaState.rightingSkillPct : 20
          });
        } else if (!isOverturnedNow && up.y >= 0.65 && this.c0CurrentlyOverturned) {
          this.c0CurrentlyOverturned = false;
          this.c0OverturnEvents.push({
            event: 'recovered_by_beetle_reflex',
            simTime: Number(this.simTime.toFixed(2)),
            update: this.trainer.updateCount,
            pos: [Number(center.x.toFixed(2)), Number(center.z.toFixed(2))],
            upY: Number(up.y.toFixed(3)),
            attemptsUsed: this.trainer.metaState ? this.trainer.metaState.rightingAttempts : 1,
            rightingSkillPct: this.trainer.metaState ? this.trainer.metaState.rightingSkillPct : 100
          });
        }
      }

      c.lastStepReward = r;
      c.episodeReward += r;
      c.cumulativeReward += r;
      rewards[i] = r;
      stepRewSum += r;
    }

    // Track leader with strong hysteresis so camera never flickers between close workers
    const currentBest = this.creatures[this.bestCreatureIdx] || this.creatures[0];
    const currentBestScore = currentBest.foodCollected * 35 + currentBest.cumulativeReward + (this.bestCreatureIdx === 0 ? 15 : 0);
    for (let i = 0; i < this.populationSize; i++) {
      const score = this.creatures[i].foodCollected * 35 + this.creatures[i].cumulativeReward + (i === 0 ? 15 : 0);
      // Require a substantial margin (+35 points = +1 full food + reward lead) to switch camera target
      if (score > currentBestScore + 35) {
        this.bestCreatureIdx = i;
      }
    }

    // Record 1 Hz trajectory sample for the main observed creature (c0)
    if (this.totalSteps % 20 === 0 && this.creatures[0]) {
      const c0 = this.creatures[0];
      const com = c0.body.getCenterOfMass();
      const vel = c0.body.getCenterOfMassVelocity();
      const { up, forward } = c0.getTorsoFrame();
      const dx = c0.foodPos.x - com.x;
      const dz = c0.foodPos.z - com.z;
      const dist = Math.hypot(dx, dz);
      const alignCos = (dx * forward.x + dz * forward.z) / Math.max(1e-5, dist);
      this.c0Trajectory.push({
        t: Number(this.simTime.toFixed(1)),
        upd: this.trainer.updateCount,
        pos: [Number(com.x.toFixed(2)), Number(com.y.toFixed(2)), Number(com.z.toFixed(2))],
        vel: [Number(vel.x.toFixed(2)), Number(vel.z.toFixed(2))],
        speed: Number(Math.hypot(vel.x, vel.z).toFixed(3)),
        velToFood: Number(c0.lastVelocityTowardFood.toFixed(3)),
        foodPos: [Number(c0.foodPos.x.toFixed(2)), Number(c0.foodPos.z.toFixed(2))],
        distToFood: Number(dist.toFixed(2)),
        alignCos: Number(alignCos.toFixed(2)),
        yawDeg: Number(((Math.atan2(forward.z, forward.x) * 180) / Math.PI).toFixed(1)),
        upY: Number(up.y.toFixed(3)),
        contacts: c0.footNodes.map(f => (f.inContact ? 1 : 0)),
        normalForces: c0.footNodes.map(f => Number(f.normalForce.toFixed(1))),
        powerW: Number(c0.lastMechanicalPower.toFixed(1))
      });
      if (this.c0Trajectory.length > 360) {
        this.c0Trajectory.shift();
      }
    }

    const isLearningNow = !this.trainer.metaState || this.trainer.metaState.isLearningActive;
    if (isLearningNow) {
      this.recentStepRewards.push(stepRewSum / this.populationSize);
      this.recentVelocities.push(stepVelSum / this.populationSize);
      if (this.recentStepRewards.length > 64) {
        this.recentStepRewards.shift();
        this.recentVelocities.shift();
      }
    }

    // 4. Pass transition outcome to PPO / NES trainer
    const updateInfo = this.trainer.recordStepOutcome(this.creatures, rewards, dones);
    if (updateInfo) {
      const meanVel = this.recentVelocities.reduce((a, b) => a + b, 0) / Math.max(1, this.recentVelocities.length);
      const meanRew = this.recentStepRewards.reduce((a, b) => a + b, 0) / Math.max(1, this.recentStepRewards.length);
      const c0 = this.creatures[0];
      const com0 = c0.body.getCenterOfMass();
      const { up: up0, forward: fwd0 } = c0.getTorsoFrame();
      const dx0 = c0.foodPos.x - com0.x;
      const dz0 = c0.foodPos.z - com0.z;
      const dist0 = Math.hypot(dx0, dz0);
      const entry = {
        update: updateInfo.updateCount,
        didUpdateWeights: updateInfo.didUpdateWeights !== false,
        simTime: Number(this.simTime.toFixed(1)),
        meanReward: Number(meanRew.toFixed(4)),
        meanVelocity: Number(meanVel.toFixed(4)),
        totalFood: this.totalFoodCollected,
        c0Food: c0.foodCollected,
        leaderFood: this.creatures[this.bestCreatureIdx].foodCollected,
        c0Pos: [Number(com0.x.toFixed(2)), Number(com0.z.toFixed(2))],
        c0FoodPos: [Number(c0.foodPos.x.toFixed(2)), Number(c0.foodPos.z.toFixed(2))],
        c0DistToFood: Number(dist0.toFixed(2)),
        c0AlignCos: Number(((dx0 * fwd0.x + dz0 * fwd0.z) / Math.max(1e-5, dist0)).toFixed(2)),
        c0UpY: Number(up0.y.toFixed(3)),
        c0VelToFood: Number(c0.lastVelocityTowardFood.toFixed(3)),
        metaMode: this.trainer.metaState ? this.trainer.metaState.mode : 'LEARNING_WALK',
        isLearningActive: this.trainer.metaState ? this.trainer.metaState.isLearningActive : true,
        needToLearnPct: this.trainer.metaState ? this.trainer.metaState.needToLearnPct : 80,
        policyLoss: Number((updateInfo.policyLoss || 0).toFixed(4)),
        valueLoss: Number((updateInfo.valueLoss || 0).toFixed(4)),
        entropy: Number((updateInfo.entropy || 0).toFixed(4))
      };
      // Only append to learning curve history when an actual learning epoch updated weights!
      if (entry.didUpdateWeights) {
        this.history.push(entry);
      }
      return entry;
    }
    return null;
  }

  getDetailedTelemetry() {
    const c0 = this.creatures[0];
    const com = c0.body.getCenterOfMass();
    const vel = c0.body.getCenterOfMassVelocity();
    const { up, forward } = c0.getTorsoFrame();
    const dx = c0.foodPos.x - com.x;
    const dz = c0.foodPos.z - com.z;
    const dist = Math.hypot(dx, dz);

    // Compute CPG & sensory-gated weight norms in actorSkip.W to diagnose neural learning progress
    const skipW = this.brain.actorSkip.W;
    let cpgDriveNorm = 0;
    let turnDriveNorm = 0;
    for (let a = 0; a < this.brain.actDim; a++) {
      const row = a * this.brain.obsDim;
      cpgDriveNorm += Math.hypot(skipW[row + 10], skipW[row + 11], skipW[row + 12], skipW[row + 13]);
      turnDriveNorm += Math.hypot(skipW[row + 14], skipW[row + 15]);
    }

    return {
      simTime: Number(this.simTime.toFixed(2)),
      updateCount: this.trainer.updateCount,
      floorFriction: this.floorFriction,
      totalPopulationFood: this.totalFoodCollected,
      metaState: this.trainer.metaState ? { ...this.trainer.metaState } : null,
      mainWalker: {
        foodCollected: c0.foodCollected,
        cumulativeReward: Number(c0.cumulativeReward.toFixed(2)),
        pos: [Number(com.x.toFixed(2)), Number(com.y.toFixed(2)), Number(com.z.toFixed(2))],
        vel: [Number(vel.x.toFixed(2)), Number(vel.y.toFixed(2)), Number(vel.z.toFixed(2))],
        speed: Number(Math.hypot(vel.x, vel.z).toFixed(3)),
        velTowardFood: Number(c0.lastVelocityTowardFood.toFixed(3)),
        foodPos: [Number(c0.foodPos.x.toFixed(2)), Number(c0.foodPos.z.toFixed(2))],
        distToFood: Number(dist.toFixed(2)),
        headingAlignCos: Number(((dx * forward.x + dz * forward.z) / Math.max(1e-5, dist)).toFixed(3)),
        yawDeg: Number(((Math.atan2(forward.z, forward.x) * 180) / Math.PI).toFixed(1)),
        upY: Number(up.y.toFixed(3)),
        isOverturned: up.y < 0.35,
        footContacts: c0.footNodes.map(f => f.inContact),
        footNormalForcesN: c0.footNodes.map(f => Number(f.normalForce.toFixed(1)))
      },
      neuralDiagnostics: {
        cpgForwardWeightNorm: Number(cpgDriveNorm.toFixed(3)),
        cpgTurnWeightNorm: Number(turnDriveNorm.toFixed(3)),
        lastPolicyLoss: Number((this.trainer.lastPolicyLoss || 0).toFixed(4)),
        lastValueLoss: Number((this.trainer.lastValueLoss || 0).toFixed(4)),
        lastEntropy: Number((this.trainer.lastEntropy || 0).toFixed(4))
      },
      c0FoodEvents: this.c0FoodEvents,
      c0OverturnEvents: this.c0OverturnEvents,
      recentTrajectory: this.c0Trajectory.slice(-120)
    };
  }

  /**
   * Serializes the entire individual creature, its physical body state, trained neural network weights,
   * trainer/meta-plasticity state, and learning curve history for instant persistence across page reloads.
   */
  exportCheckpoint(uiState = null, cameraState = null) {
    const c0 = this.creatures[0];
    return {
      version: 2,
      savedAt: new Date().toISOString(),
      morphologyKey: this.morphologyKey,
      actuatorMode: this.actuatorMode,
      algorithm: this.algorithm,
      populationSize: this.populationSize,
      arenaHalfSize: this.arenaHalfSize,
      floorFriction: this.floorFriction,
      simTime: this.simTime,
      totalSteps: this.totalSteps,
      totalFoodCollected: this.totalFoodCollected,
      bestCreatureIdx: this.bestCreatureIdx,
      history: this.history.slice(-250),
      recentStepRewards: this.recentStepRewards.slice(-64),
      recentVelocities: this.recentVelocities.slice(-64),
      c0FoodEvents: this.c0FoodEvents.slice(-100),
      c0OverturnEvents: this.c0OverturnEvents.slice(-50),
      creature0: c0 ? {
        phase: c0.phase,
        ageTime: c0.ageTime,
        episodeSteps: c0.episodeSteps,
        episodeReward: c0.episodeReward,
        cumulativeReward: c0.cumulativeReward,
        lastStepReward: c0.lastStepReward,
        lastVelocityTowardFood: c0.lastVelocityTowardFood,
        smoothedVelocityTowardFood: c0.smoothedVelocityTowardFood,
        lastMechanicalPower: c0.lastMechanicalPower,
        foodCollected: c0.foodCollected,
        prevFoodDist: c0.prevFoodDist,
        foodPos: [c0.foodPos.x, c0.foodPos.y, c0.foodPos.z],
        dissolvePos: [c0.dissolvePos.x, c0.dissolvePos.y, c0.dissolvePos.z],
        foodState: c0.foodState,
        foodStateTimer: c0.foodStateTimer,
        nodes: c0.body.nodes.map(n => ({
          pos: [n.pos.x, n.pos.y, n.pos.z],
          prevPos: [n.prevPos.x, n.prevPos.y, n.prevPos.z],
          vel: [n.vel.x, n.vel.y, n.vel.z],
          inContact: n.inContact,
          normalForce: n.normalForce
        })),
        muscles: c0.body.muscles.map(m => ({
          targetLength: m.targetLength,
          directTargetLength: m.directTargetLength,
          targetAction: m.targetAction,
          smoothedAction: m.smoothedAction
        }))
      } : null,
      brain: this.brain && typeof this.brain.toJSON === 'function' ? this.brain.toJSON() : null,
      trainer: this.trainer && typeof this.trainer.toJSON === 'function' ? this.trainer.toJSON() : null,
      uiState,
      cameraState
    };
  }

  /**
   * Restores a previously saved individual creature, its physical posture, food state,
   * neural network weights, and trainer state into this simulation instance.
   */
  importCheckpoint(ckpt) {
    if (!ckpt || ckpt.morphologyKey !== this.morphologyKey || ckpt.algorithm !== this.algorithm) {
      return false;
    }
    if (typeof ckpt.floorFriction === 'number') {
      this.floorFriction = ckpt.floorFriction;
      this.world.floorFriction = ckpt.floorFriction;
    }
    if (typeof ckpt.simTime === 'number') this.simTime = ckpt.simTime;
    if (typeof ckpt.totalSteps === 'number') this.totalSteps = ckpt.totalSteps;
    if (typeof ckpt.totalFoodCollected === 'number') this.totalFoodCollected = ckpt.totalFoodCollected;
    if (typeof ckpt.bestCreatureIdx === 'number') this.bestCreatureIdx = ckpt.bestCreatureIdx;
    if (Array.isArray(ckpt.history)) this.history = ckpt.history.slice();
    if (Array.isArray(ckpt.recentStepRewards)) this.recentStepRewards = ckpt.recentStepRewards.slice();
    if (Array.isArray(ckpt.recentVelocities)) this.recentVelocities = ckpt.recentVelocities.slice();
    if (Array.isArray(ckpt.c0FoodEvents)) this.c0FoodEvents = ckpt.c0FoodEvents.slice();
    if (Array.isArray(ckpt.c0OverturnEvents)) this.c0OverturnEvents = ckpt.c0OverturnEvents.slice();

    const c0 = this.creatures[0];
    if (c0 && ckpt.creature0) {
      const sc = ckpt.creature0;
      if (typeof sc.phase === 'number') c0.phase = sc.phase;
      if (typeof sc.ageTime === 'number') c0.ageTime = sc.ageTime;
      if (typeof sc.episodeSteps === 'number') c0.episodeSteps = sc.episodeSteps;
      if (typeof sc.episodeReward === 'number') c0.episodeReward = sc.episodeReward;
      if (typeof sc.cumulativeReward === 'number') c0.cumulativeReward = sc.cumulativeReward;
      if (typeof sc.lastStepReward === 'number') c0.lastStepReward = sc.lastStepReward;
      if (typeof sc.lastVelocityTowardFood === 'number') c0.lastVelocityTowardFood = sc.lastVelocityTowardFood;
      if (typeof sc.smoothedVelocityTowardFood === 'number') c0.smoothedVelocityTowardFood = sc.smoothedVelocityTowardFood;
      if (typeof sc.lastMechanicalPower === 'number') c0.lastMechanicalPower = sc.lastMechanicalPower;
      if (typeof sc.foodCollected === 'number') c0.foodCollected = sc.foodCollected;
      if (typeof sc.prevFoodDist === 'number') c0.prevFoodDist = sc.prevFoodDist;
      if (Array.isArray(sc.foodPos) && sc.foodPos.length === 3) {
        c0.foodPos.set(sc.foodPos[0], sc.foodPos[1], sc.foodPos[2]);
      }
      if (Array.isArray(sc.dissolvePos) && sc.dissolvePos.length === 3) {
        c0.dissolvePos.set(sc.dissolvePos[0], sc.dissolvePos[1], sc.dissolvePos[2]);
      }
      if (typeof sc.foodState === 'string') c0.foodState = sc.foodState;
      if (typeof sc.foodStateTimer === 'number') c0.foodStateTimer = sc.foodStateTimer;

      if (Array.isArray(sc.nodes) && sc.nodes.length === c0.body.nodes.length) {
        for (let i = 0; i < sc.nodes.length; i++) {
          const sn = sc.nodes[i];
          const n = c0.body.nodes[i];
          if (Array.isArray(sn.pos)) n.pos.set(sn.pos[0], sn.pos[1], sn.pos[2]);
          if (Array.isArray(sn.prevPos)) n.prevPos.set(sn.prevPos[0], sn.prevPos[1], sn.prevPos[2]);
          if (Array.isArray(sn.vel)) n.vel.set(sn.vel[0], sn.vel[1], sn.vel[2]);
          n.inContact = Boolean(sn.inContact);
          n.normalForce = Number(sn.normalForce || 0);
        }
      }
      if (Array.isArray(sc.muscles) && sc.muscles.length === c0.body.muscles.length) {
        for (let i = 0; i < sc.muscles.length; i++) {
          const sm = sc.muscles[i];
          const m = c0.body.muscles[i];
          if (typeof sm.targetLength === 'number') m.targetLength = sm.targetLength;
          if (typeof sm.directTargetLength === 'number') m.directTargetLength = sm.directTargetLength;
          if (typeof sm.targetAction === 'number') m.targetAction = sm.targetAction;
          if (typeof sm.smoothedAction === 'number') m.smoothedAction = sm.smoothedAction;
        }
      }
    }

    if (ckpt.brain && this.brain && typeof this.brain.fromJSON === 'function') {
      this.brain.fromJSON(ckpt.brain);
    }
    if (ckpt.trainer && this.trainer && typeof this.trainer.fromJSON === 'function') {
      this.trainer.fromJSON(ckpt.trainer);
    }

    if (c0 && this.brain) {
      const obs = c0.getObservation();
      this.brain.forwardActor(obs, true, null);
      this.brain.forwardCritic(obs, true);
    }
    this.substepCursor = 0;
    return true;
  }
}
