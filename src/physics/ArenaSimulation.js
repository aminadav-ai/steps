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
    algorithm = 'ppo',
    populationSize = 16,
    arenaHalfSize = 11.0,
    floorFriction = 1.0
  } = {}) {
    this.morphologyKey = morphologyKey;
    this.algorithm = algorithm;
    this.populationSize = populationSize;
    this.arenaHalfSize = arenaHalfSize;
    this.floorFriction = floorFriction;

    this.controlDt = 0.05; // 20 Hz neural control tick
    this.substepsPerControl = 10;
    this.subDt = this.controlDt / this.substepsPerControl; // 0.005s (200 Hz physics)
    this.substepCursor = 0;
    this.cpgOmega = 2 * Math.PI * 1.45; // 1.45 Hz natural stride frequency
    this.simTime = 0;
    this.totalSteps = 0;
    this.totalFoodCollected = 0;
    this.bestCreatureIdx = 0;

    // Telemetry history for live charts
    this.history = [];
    this.recentStepRewards = [];
    this.recentVelocities = [];

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
      const creature = morph.build({ id: i, origin, color });
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
      color: oldCreature.color
    });
    fresh.foodCollected = oldCreature.foodCollected;
    fresh.cumulativeReward = oldCreature.cumulativeReward;
    this.world.bodies[idx] = fresh.body;
    this.creatures[idx] = fresh;
    this._spawnFoodForCreature(fresh, false);
  }

  setFloorFriction(mu) {
    this.floorFriction = Math.max(0, Math.min(2.0, mu));
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
          c.applyActionVector(c.lastAction);
        }
      });
    }

    this.world.stepSingleSubstep(subDt);
    this.simTime += subDt;
    this.substepCursor += 1;

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

      // Velocity toward food (m/s)
      const velToward = (c.prevFoodDist - newDist) / dt;
      c.prevFoodDist = newDist;
      c.lastVelocityTowardFood = velToward;
      stepVelSum += velToward;

      // Heading alignment toward food
      const invD = 1 / Math.max(1e-5, newDist);
      const alignCos = (dx * invD) * forward.x + (dz * invD) * forward.z;

      // Reward components:
      // 1) Approach velocity reward (favors moving nose-first toward food)
      const forwardFactor = Math.max(0.25, 0.65 + 0.35 * alignCos);
      let r = velToward * forwardFactor * 2.5;
      // 2) Orientation alignment reward
      r += 0.22 * alignCos;
      // 3) Upright posture maintenance
      r += 0.08 * Math.max(-1, up.y - 0.55);
      // 4) Mechanical power penalty (smooth, efficient locomotion)
      r -= 0.00035 * c.lastMechanicalPower;

      // Check if food reached! (center or front hip within reach radius)
      const frontDist = Math.hypot(c.foodPos.x - c.frameRefs.front.pos.x, c.foodPos.z - c.frameRefs.front.pos.z);
      if (newDist < 0.46 || frontDist < 0.38) {
        r += 8.5; // Major foraging reward
        c.foodCollected += 1;
        this.totalFoodCollected += 1;
        this._spawnFoodForCreature(c, false);
      }

      // Only reset a creature if it physically flips upside down; never interrupt continuous walking!
      const flipped = up.y < 0.18 || center.y < 0.10;
      if (flipped) {
        r -= 2.2;
        dones[i] = 1;
      }

      c.lastStepReward = r;
      c.episodeReward += r;
      c.cumulativeReward += r;
      rewards[i] = r;
      stepRewSum += r;

      if (dones[i]) {
        this._resetCreatureBody(i);
      }
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

    this.recentStepRewards.push(stepRewSum / this.populationSize);
    this.recentVelocities.push(stepVelSum / this.populationSize);
    if (this.recentStepRewards.length > 64) {
      this.recentStepRewards.shift();
      this.recentVelocities.shift();
    }

    // 4. Pass transition outcome to PPO / NES trainer
    const updateInfo = this.trainer.recordStepOutcome(this.creatures, rewards, dones);
    if (updateInfo) {
      const meanVel = this.recentVelocities.reduce((a, b) => a + b, 0) / Math.max(1, this.recentVelocities.length);
      const meanRew = this.recentStepRewards.reduce((a, b) => a + b, 0) / Math.max(1, this.recentStepRewards.length);
      const entry = {
        update: updateInfo.updateCount,
        simTime: Number(this.simTime.toFixed(1)),
        meanReward: Number(meanRew.toFixed(4)),
        meanVelocity: Number(meanVel.toFixed(4)),
        totalFood: this.totalFoodCollected,
        leaderFood: this.creatures[this.bestCreatureIdx].foodCollected,
        policyLoss: Number((updateInfo.policyLoss || 0).toFixed(4)),
        valueLoss: Number((updateInfo.valueLoss || 0).toFixed(4)),
        entropy: Number((updateInfo.entropy || 0).toFixed(4))
      };
      this.history.push(entry);
      return entry;
    }
    return null;
  }
}
