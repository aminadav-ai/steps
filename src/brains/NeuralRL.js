/**
 * Neural Network & Reinforcement Learning Engine
 *
 * Implements:
 * 1. ActorCriticBrain: Two-layer Tanh MLP + Residual Sensor-Motor Skip Connection (Actor)
 *    and Two-layer Tanh Value Network (Critic), with exact analytical Backpropagation
 *    and Adam Optimizer.
 * 2. PPOTrainer: Proximal Policy Optimization with Generalized Advantage Estimation (GAE),
 *    Parameter-Space Exploration (Plappert et al., ICLR 2018), Clipped Surrogate Objective,
 *    and Self-Imitation (SIL) elite anchor.
 * 3. NESTrainer: Separable Natural Evolution Strategies (OpenAI-ES / SEP-CMA-ES) for comparison.
 */

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Dense Linear Layer with exact analytical forward/backward pass and Adam optimizer state.
 */
export class DenseLayer {
  constructor(inDim, outDim, initScale = null) {
    this.inDim = inDim;
    this.outDim = outDim;
    const scale = initScale !== null ? initScale : Math.sqrt(2.0 / (inDim + outDim));

    this.W = new Float64Array(inDim * outDim);
    this.b = new Float64Array(outDim);
    this.dW = new Float64Array(inDim * outDim);
    this.db = new Float64Array(outDim);

    // Adam first and second moments
    this.mW = new Float64Array(inDim * outDim);
    this.vW = new Float64Array(inDim * outDim);
    this.mb = new Float64Array(outDim);
    this.vb = new Float64Array(outDim);

    for (let i = 0; i < this.W.length; i++) {
      this.W[i] = randn() * scale;
    }
  }

  zeroGrad() {
    this.dW.fill(0);
    this.db.fill(0);
  }

  forward(x, out) {
    const { inDim, outDim, W, b } = this;
    for (let o = 0; o < outDim; o++) {
      let sum = b[o];
      const rowOffset = o * inDim;
      for (let i = 0; i < inDim; i++) {
        sum += W[rowOffset + i] * x[i];
      }
      out[o] = sum;
    }
    return out;
  }

  backward(x, dOut, dIn) {
    const { inDim, outDim, W, dW, db } = this;
    if (dIn) dIn.fill(0);
    for (let o = 0; o < outDim; o++) {
      const gradO = dOut[o];
      db[o] += gradO;
      const rowOffset = o * inDim;
      for (let i = 0; i < inDim; i++) {
        dW[rowOffset + i] += gradO * x[i];
        if (dIn) {
          dIn[i] += W[rowOffset + i] * gradO;
        }
      }
    }
  }

  adamStep(lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, stepNum = 1, gradClip = 1.0) {
    const biasCorr1 = 1 - Math.pow(beta1, stepNum);
    const biasCorr2 = 1 - Math.pow(beta2, stepNum);

    for (let i = 0; i < this.W.length; i++) {
      const g = Math.max(-gradClip, Math.min(gradClip, this.dW[i]));
      this.mW[i] = beta1 * this.mW[i] + (1 - beta1) * g;
      this.vW[i] = beta2 * this.vW[i] + (1 - beta2) * g * g;
      const mHat = this.mW[i] / biasCorr1;
      const vHat = this.vW[i] / biasCorr2;
      this.W[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }

    for (let o = 0; o < this.b.length; o++) {
      const g = Math.max(-gradClip, Math.min(gradClip, this.db[o]));
      this.mb[o] = beta1 * this.mb[o] + (1 - beta1) * g;
      this.vb[o] = beta2 * this.vb[o] + (1 - beta2) * g * g;
      const mHat = this.mb[o] / biasCorr1;
      const vHat = this.vb[o] / biasCorr2;
      this.b[o] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  toJSON() {
    return {
      W: Array.from(this.W),
      b: Array.from(this.b),
      mW: Array.from(this.mW),
      vW: Array.from(this.vW),
      mb: Array.from(this.mb),
      vb: Array.from(this.vb)
    };
  }

  fromJSON(data) {
    if (!data) return;
    if (Array.isArray(data.W) && data.W.length === this.W.length) this.W.set(data.W);
    if (Array.isArray(data.b) && data.b.length === this.b.length) this.b.set(data.b);
    if (Array.isArray(data.mW) && data.mW.length === this.mW.length) this.mW.set(data.mW);
    if (Array.isArray(data.vW) && data.vW.length === this.vW.length) this.vW.set(data.vW);
    if (Array.isArray(data.mb) && data.mb.length === this.mb.length) this.mb.set(data.mb);
    if (Array.isArray(data.vb) && data.vb.length === this.vb.length) this.vb.set(data.vb);
  }
}

/**
 * Actor-Critic Neural Network with Residual CPG/Sensor-Motor Skip Connection
 * Architecture:
 *   Actor:  s -> Dense(32) -> tanh -> Dense(24) -> tanh -> Dense(actDim)
 *           + SkipLinear(s -> actDim)  ==>  tanh(z) = mean action \mu(s) in (-1, 1)
 *   Critic: s -> Dense(32) -> tanh -> Dense(24) -> tanh -> Dense(1) = V(s)
 */
export class ActorCriticBrain {
  constructor(obsDim, actDim, hidden1 = 32, hidden2 = 24, morphologyKey = 'quadruped') {
    this.obsDim = obsDim;
    this.actDim = actDim;
    this.hidden1 = hidden1;
    this.hidden2 = hidden2;
    this.morphologyKey = morphologyKey;

    // Actor layers
    this.actorL1 = new DenseLayer(obsDim, hidden1, Math.sqrt(0.5 / (obsDim + hidden1)));
    this.actorL2 = new DenseLayer(hidden1, hidden2, Math.sqrt(0.5 / (hidden1 + hidden2)));
    this.actorOut = new DenseLayer(hidden2, actDim, 0.02);
    // Residual reflex skip connection (direct sensor+CPG clock -> motor logits)
    this.actorSkip = new DenseLayer(obsDim, actDim, 0.02);

    this._seedInitialMotorPrior();

    // Learnable log-standard-deviation per action channel
    this.logStd = new Float64Array(actDim).fill(-1.05);

    // Critic layers
    this.criticL1 = new DenseLayer(obsDim, hidden1, Math.sqrt(2 / (obsDim + hidden1)));
    this.criticL2 = new DenseLayer(hidden1, hidden2, Math.sqrt(2 / (hidden1 + hidden2)));
    this.criticOut = new DenseLayer(hidden2, 1, 0.1);

    // Dedicated Self-Righting Sub-Network ("Рефлекс жука / Нейросеть вставания"):
    // Completely isolated from walking weights; active and trained ONLY when overturned (up.y < 0.35)
    this.rightingLayer = new DenseLayer(obsDim, actDim, 0.04);
    this.rightingGain = 0.32;       // Starts weak (32%) so first overturn attempt only rocks the shell
    this.rightingAsym = 0.28;       // Learns asymmetric dorsal extension (one leg levers floor, opposite tucks)
    this.rightingSweep = 0.32;      // Learns longitudinal sweep leverage
    this.rightingSideSign = 1.0;    // Preferred roll-over edge (+1 or -1), adapts if blocked
    this.rightingAttempts = 0;
    this.rightingSuccesses = 0;
    this.rightingPhase = 0;
    this.duckRightingBestProgress = 0;
    this.rightingHiddenDim = 16;
    this.rightingH1 = null;
    this.rightingOut = null;

    // Two-Level Genetic Algorithm ("Language of Poses": GA #1 Poses + GA #2 Servo Commands + Quantized Memory) for the Duck:
    // 6D Quantized Pose Vector P = [comY_norm, upY, pitchY, rollY, legExtL, legExtR], quantized with step 0.25
    this.poseGoal = Float64Array.from([1.0, 1.0, 0.0, 0.0, 0.75, 0.75]); // Target pose: "Standing Duck"
    this.poseCurrent = null;
    this.poseGen = 1;
    this.posePopSize = 4;
    this.poseCursor = 0;
    this.posePop = [];

    this.gaPopSize = 4;
    this.gaGeneration = 1;
    this.gaCursor = 0;
    this.gaPopulation = [];
    this.gaBestVector = null;
    this.gaBestFitness = -Infinity;
    this.gaBestOrigin = 'случайный';
    this.gaArchive = [];
    // Deduplicated cell-to-cell transition memory: Map("fromKey->toKey", { fromKey, toKey, pFrom, pTo, cmdVec, matchPct, goalPct, maxComY })
    this.poseTransitionMemory = new Map();
    // Unreachable pose-mutant memory: Set("fromKey->mutantKey")
    this.poseUnreachableMemory = new Set();
    this._winStartPose6D = null;

    this._waveBaseCrestUpY = null;
    this._waveBaseMeanUpY = null;
    this._waveBaseSpanUpY = 0;
    this._waveBaseCrestTopY = null;
    this._waveBaseMeanTopY = null;
    this._waveBaseSpanTopY = 0;
    this._waveStagnantCount = 0;
    this._resetRightingWindow(0, 0.18);
    if (this.morphologyKey === 'duck') {
      this._initDuckGAPopulation();
    }

    this.adamStepCount = 0;
    this.maxSkipRowNorm = 1.80;

    // Live activation buffers for UI visualization
    this.lastObs = new Float64Array(obsDim);
    this.lastH1 = new Float64Array(hidden1);
    this.lastH2 = new Float64Array(hidden2);
    this.lastMu = new Float64Array(actDim);
    this.lastValue = 0;
  }

  toJSON() {
    return {
      obsDim: this.obsDim,
      actDim: this.actDim,
      hidden1: this.hidden1,
      hidden2: this.hidden2,
      actorL1: this.actorL1.toJSON(),
      actorL2: this.actorL2.toJSON(),
      actorOut: this.actorOut.toJSON(),
      actorSkip: this.actorSkip.toJSON(),
      criticL1: this.criticL1.toJSON(),
      criticL2: this.criticL2.toJSON(),
      criticOut: this.criticOut.toJSON(),
      rightingLayer: this.rightingLayer.toJSON(),
      rightingH1: this.rightingH1 ? this.rightingH1.toJSON() : null,
      rightingOut: this.rightingOut ? this.rightingOut.toJSON() : null,
      cpgTemplate: this.cpgTemplate ? Array.from(this.cpgTemplate) : null,
      logStd: Array.from(this.logStd),
      rightingGain: this.rightingGain,
      rightingAsym: this.rightingAsym,
      rightingSweep: this.rightingSweep,
      rightingSideSign: this.rightingSideSign,
      rightingAttempts: this.rightingAttempts,
      rightingSuccesses: this.rightingSuccesses,
      rightingPhase: this.rightingPhase,
      gaGeneration: this.gaGeneration,
      gaCursor: this.gaCursor,
      poseGen: this.poseGen,
      poseCursor: this.poseCursor,
      poseCurrent: this.poseCurrent ? Array.from(this.poseCurrent) : null,
      gaBestVector: this.gaBestVector ? Array.from(this.gaBestVector) : null,
      gaBestFitness: Number.isFinite(this.gaBestFitness) ? this.gaBestFitness : null,
      gaBestOrigin: this.gaBestOrigin,
      gaArchive: Array.isArray(this.gaArchive)
        ? this.gaArchive.map(item => ({
            vec: Array.from(item.vec),
            fitness: item.fitness,
            maxComY: item.maxComY
          }))
        : [],
      poseTransitionMemory: this.poseTransitionMemory
        ? Array.from(this.poseTransitionMemory.values()).map(item => ({
            fromKey: item.fromKey,
            toKey: item.toKey,
            pFrom: Array.from(item.pFrom),
            pTo: Array.from(item.pTo),
            cmdVec: Array.from(item.cmdVec),
            matchPct: item.matchPct,
            goalPct: item.goalPct,
            maxComY: item.maxComY
          }))
        : [],
      poseUnreachableMemory: this.poseUnreachableMemory ? Array.from(this.poseUnreachableMemory) : [],
      adamStepCount: this.adamStepCount,
      maxSkipRowNorm: this.maxSkipRowNorm
    };
  }

  fromJSON(data) {
    if (!data || data.obsDim !== this.obsDim || data.actDim !== this.actDim) return false;
    this.actorL1.fromJSON(data.actorL1);
    this.actorL2.fromJSON(data.actorL2);
    this.actorOut.fromJSON(data.actorOut);
    this.actorSkip.fromJSON(data.actorSkip);
    this.criticL1.fromJSON(data.criticL1);
    this.criticL2.fromJSON(data.criticL2);
    this.criticOut.fromJSON(data.criticOut);
    this.rightingLayer.fromJSON(data.rightingLayer);
    if (Array.isArray(data.cpgTemplate) && this.cpgTemplate && data.cpgTemplate.length === this.cpgTemplate.length) {
      this.cpgTemplate.set(data.cpgTemplate);
    }
    if (Array.isArray(data.logStd) && data.logStd.length === this.logStd.length) {
      this.logStd.set(data.logStd);
    }
    if (typeof data.rightingGain === 'number') this.rightingGain = data.rightingGain;
    if (typeof data.rightingAsym === 'number') this.rightingAsym = data.rightingAsym;
    if (typeof data.rightingSweep === 'number') this.rightingSweep = data.rightingSweep;
    if (typeof data.rightingSideSign === 'number') this.rightingSideSign = data.rightingSideSign;
    if (typeof data.rightingAttempts === 'number') this.rightingAttempts = data.rightingAttempts;
    if (typeof data.rightingSuccesses === 'number') this.rightingSuccesses = data.rightingSuccesses;
    if (typeof data.rightingPhase === 'number') this.rightingPhase = data.rightingPhase;
    if (typeof data.gaGeneration === 'number') this.gaGeneration = data.gaGeneration;
    if (typeof data.poseGen === 'number') this.poseGen = data.poseGen;
    if (Array.isArray(data.poseCurrent) && data.poseCurrent.length === 6) {
      this.poseCurrent = Float64Array.from(data.poseCurrent);
    }
    if (Array.isArray(data.gaBestVector) && data.gaBestVector.length === this.actDim) {
      this.gaBestVector = Float64Array.from(data.gaBestVector);
    }
    if (typeof data.gaBestFitness === 'number' && Number.isFinite(data.gaBestFitness)) {
      this.gaBestFitness = data.gaBestFitness;
    }
    if (typeof data.gaBestOrigin === 'string') {
      this.gaBestOrigin = data.gaBestOrigin;
    }
    if (Array.isArray(data.gaArchive)) {
      this.gaArchive = data.gaArchive
        .filter(item => item && Array.isArray(item.vec) && item.vec.length === this.actDim)
        .map(item => ({
          vec: Float64Array.from(item.vec),
          fitness: Number(item.fitness) || 0,
          maxComY: Number(item.maxComY) || 0
        }));
    }
    if (Array.isArray(data.poseTransitionMemory)) {
      this.poseTransitionMemory = new Map();
      for (const item of data.poseTransitionMemory) {
        if (item && item.fromKey && item.toKey && Array.isArray(item.cmdVec) && item.cmdVec.length === this.actDim) {
          this.poseTransitionMemory.set(`${item.fromKey}->${item.toKey}`, {
            fromKey: item.fromKey,
            toKey: item.toKey,
            pFrom: Float64Array.from(item.pFrom || [0, 0, 0, 0, 0, 0]),
            pTo: Float64Array.from(item.pTo || [0, 0, 0, 0, 0, 0]),
            cmdVec: Float64Array.from(item.cmdVec),
            matchPct: Number(item.matchPct) || 100,
            goalPct: Number(item.goalPct) || 50,
            maxComY: Number(item.maxComY) || 0.18
          });
        }
      }
    }
    if (Array.isArray(data.poseUnreachableMemory)) {
      this.poseUnreachableMemory = new Set(data.poseUnreachableMemory);
    }
    if (this.morphologyKey === 'duck') {
      this._initDuckGAPopulation();
    }
    if (typeof data.adamStepCount === 'number') this.adamStepCount = data.adamStepCount;
    if (typeof data.maxSkipRowNorm === 'number') this.maxSkipRowNorm = data.maxSkipRowNorm;
    return true;
  }

  /**
   * Seeds a weak embryonic CPG reflex prior (amplitude 0.16) so the newborn creature
   * makes small unrefined stepping motions at Update 0, and RL rapidly amplifies
   * both stride power and steering toward food.
   */
  _seedInitialMotorPrior() {
    // Obs indices:
    // 10 = clockSin, 11 = clockCos
    // 12 = clockSin * forwardGate, 13 = clockCos * forwardGate
    // 14 = clockSin * turnDrive,   15 = clockCos * turnDrive
    const psi = (210 * Math.PI) / 180;

    if (this.morphologyKey === 'duck' && this.actDim === 14) {
      // 14-DoF Open_Duck_Mini Center-of-Mass Height Maximization Prior:
      // Leg 0 (Left):  [0]=hip_yaw, [1]=hip_roll, [2]=hip_pitch, [3]=knee_pitch, [4]=ankle_pitch
      // Leg 1 (Right): [5]=hip_yaw, [6]=hip_roll, [7]=hip_pitch, [8]=knee_pitch, [9]=ankle_pitch
      // Neck & Head:   [10]=neck_pitch, [11]=head_pitch, [12]=head_yaw, [13]=head_roll
      for (let leg = 0; leg < 2; leg++) {
        const base = leg * 5;
        const rollRow = (base + 1) * this.obsDim;
        const pitchRow = (base + 2) * this.obsDim;
        const kneeRow = (base + 3) * this.obsDim;
        const ankleRow = (base + 4) * this.obsDim;

        // Active knee extension bias to maximize Center of Mass height + feedback on CoM height (obs[9])
        this.actorSkip.b[base + 3] = 0.78;
        this.actorSkip.W[kneeRow + 9] = -0.35; // When CoM height (obs[9]) is below target, push legs harder!
        this.actorSkip.W[kneeRow + 5] = -0.18; // Damp vertical oscillation (obs[5] = velUp)

        // Sagittal & lateral balance feedback to keep CoM supported high above the feet
        this.actorSkip.W[ankleRow + 7] = 0.24;  // pitchTilt -> ankle stabilization
        this.actorSkip.W[pitchRow + 7] = -0.22; // pitchTilt -> hip balance
        this.actorSkip.W[pitchRow + 3] = -0.18; // velForward damping
        this.actorSkip.W[rollRow + 8] = -0.25;  // rollTilt -> hip_roll lateral stabilization
        this.actorSkip.W[rollRow + 4] = -0.16;  // velRight damping
      }

      // Neck & Head Counterbalance (DoFs 10..13):
      const neckPitchRow = 10 * this.obsDim;
      const headPitchRow = 11 * this.obsDim;
      const headRollRow = 13 * this.obsDim;

      this.actorSkip.b[10] = 0.15;
      this.actorSkip.b[11] = 0.10;
      this.actorSkip.W[neckPitchRow + 7] = 0.32;
      this.actorSkip.W[headPitchRow + 7] = 0.25;
      this.actorSkip.W[headRollRow + 8] = -0.28;

      this.cpgTemplate = Float64Array.from(this.actorSkip.W);
      return;
    }

    const numLegs = Math.floor(this.actDim / 2);

    for (let leg = 0; leg < numLegs; leg++) {
      // Bilaterally symmetric phase schedule:
      // Tripod (3 legs): Front-Left (0) & Front-Right (0) alternate with Rear-Center (pi)
      // Quadruped (4 legs): Diagonal trot pairs FL+BR (0) and FR+BL (pi)
      const delta = numLegs === 3
        ? (leg === 2 ? Math.PI : 0.0)
        : (leg === 0 || leg === 3 ? 0.0 : Math.PI);

      const cosLift = Math.cos(delta);
      const sinLift = Math.sin(delta);

      const thetaStride = delta + psi;
      const cosStride = Math.cos(thetaStride);
      const sinStride = Math.sin(thetaStride);

      const liftRow = (leg * 2) * this.obsDim;
      const strideRow = (leg * 2 + 1) * this.obsDim;

      // Embryonic stepping reflex (starts modest at 0.22 so RL rapidly amplifies speed & steering):
      // lift_k   ~ cos(phi + delta)       = cos(delta)*cos(phi) - sin(delta)*sin(phi)
      // stride_k ~ cos(phi + delta + psi) = cos(theta)*cos(phi) - sin(theta)*sin(phi)
      this.actorSkip.W[liftRow + 11] = 0.22 * cosLift;
      this.actorSkip.W[liftRow + 10] = -0.22 * sinLift;

      // Forward-gated propulsion (engages when facing food, pauses forward travel while pivoting)
      this.actorSkip.W[strideRow + 13] = 0.22 * cosStride;
      this.actorSkip.W[strideRow + 12] = -0.22 * sinStride;

      // Differential yaw steering toward food:
      // When turnDrive > 0 (food on +Z / left-front leg 0 side):
      // leg 0 (+Z) reverses stride (-1) and leg 1 (-Z) advances stride (+1), pivoting nose to +Z!
      if (leg === 0 || (numLegs === 4 && leg === 2)) {
        this.actorSkip.W[strideRow + 15] = -0.22 * cosStride;
        this.actorSkip.W[strideRow + 14] = +0.22 * sinStride;
      } else if (leg === 1 || (numLegs === 4 && leg === 3)) {
        this.actorSkip.W[strideRow + 15] = +0.22 * cosStride;
        this.actorSkip.W[strideRow + 14] = -0.22 * sinStride;
      }
    }

    // Store normalized CPG resonance template for coherent single-walker exploration
    this.cpgTemplate = Float64Array.from(this.actorSkip.W);
  }

  /**
   * Quantizes a single normalized coordinate v in [-1, +1] to a coarse grid with step 0.25
   * ({-1.0, -0.75, -0.5, -0.25, 0.0, +0.25, +0.5, +0.75, +1.0}) so memory never bloats with micro-jitter noise.
   */
  _quantizePoseCoord(v) {
    const clamped = Math.max(-1.0, Math.min(1.0, Number(v) || 0));
    const q = Math.round(clamped * 4) / 4;
    return Object.is(q, -0) ? 0 : q;
  }

  /**
   * Quantizes a 6D pose vector to the coarse 0.25 grid.
   */
  quantizePose6D(poseVec) {
    const out = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      out[i] = this._quantizePoseCoord(poseVec ? poseVec[i] : 0);
    }
    return out;
  }

  /**
   * Extracts the compact 6-dimensional quantized Pose Vector P in [-1, +1]^6 from the Duck's sensors:
   *   [0] comY_norm : normalized Center of Mass height (0.12 m -> -1.0, 0.20 m -> 0.0, 0.28 m -> +1.0)
   *   [1] upY       : torso vertical projection (+1 = upright, -1 = inverted)
   *   [2] pitchY    : sagittal tilt (+1 = on back/nose up, -1 = on belly/nose down)
   *   [3] rollY     : lateral tilt (+1 = right side down, -1 = left side down)
   *   [4] legExtL   : left leg extension (-1 = tucked, +1 = extended)
   *   [5] legExtR   : right leg extension (-1 = tucked, +1 = extended)
   */
  extractDuckPose6D(obs, comY = 0.15) {
    const comNorm = (comY - 0.20) / 0.08;
    const upY = obs && obs.length > 6 ? obs[6] : 0.0;
    const pitchY = obs && obs.length > 7 ? obs[7] : 0.0;
    const rollY = obs && obs.length > 8 ? obs[8] : 0.0;
    const legExtL = obs && obs.length > 22 ? obs[22] * 2.4 : 0.0;
    const legExtR = obs && obs.length > 32 ? obs[32] * 2.4 : 0.0;
    return this.quantizePose6D([comNorm, upY, pitchY, rollY, legExtL, legExtR]);
  }

  /**
   * Returns the canonical cell key string for a quantized 6D pose vector (for O(1) memory deduplication).
   */
  getPoseKey6D(poseVec) {
    const q = this.quantizePose6D(poseVec);
    return `${q[0].toFixed(2)}|${q[1].toFixed(2)}|${q[2].toFixed(2)}|${q[3].toFixed(2)}|${q[4].toFixed(2)}|${q[5].toFixed(2)}`;
  }

  /**
   * Weighted Euclidean distance D(P_A, P_B) in [0, 2] between two 6D pose vectors.
   * Weights prioritize Center of Mass height (comY) and torso uprightness (upY).
   */
  computePoseDistance6D(pA, pB) {
    if (!pA || !pB) return 2.0;
    const weights = [2.2, 2.0, 1.2, 1.2, 0.7, 0.7]; // sum = 8.0
    let sumSq = 0;
    for (let i = 0; i < 6; i++) {
      const diff = (pA[i] || 0) - (pB[i] || 0);
      sumSq += weights[i] * diff * diff;
    }
    return Math.sqrt(sumSq / 8.0);
  }

  /**
   * Computes percentage similarity (0..100%) between an achieved pose and a target pose.
   */
  computePoseMatchPct6D(pActual, pTarget) {
    const dist = this.computePoseDistance6D(pActual, pTarget);
    return Math.max(0, Math.min(100, (1.0 - dist / 1.35) * 100));
  }

  /**
   * Mutates a 6D quantized pose vector toward the goal pose P_goal by discrete 0.25 grid steps.
   */
  _mutatePose6D(basePose, goalPose, stepBias = 0.5) {
    const child = Float64Array.from(basePose);
    let changed = 0;
    for (let i = 0; i < 6; i++) {
      if (Math.random() < 0.65) {
        const diff = goalPose[i] - basePose[i];
        let delta = 0;
        if (Math.abs(diff) >= 0.25 && Math.random() < stepBias) {
          delta = Math.sign(diff) * (Math.random() < 0.5 ? 0.25 : 0.50);
        } else {
          delta = (Math.floor(Math.random() * 5) - 2) * 0.25;
        }
        const nextVal = this._quantizePoseCoord(child[i] + delta);
        if (nextVal !== child[i]) changed += 1;
        child[i] = nextVal;
      }
    }
    if (changed === 0) {
      const idx = Math.floor(Math.random() * 6);
      const dir = goalPose[idx] >= child[idx] ? 0.25 : -0.25;
      child[idx] = this._quantizePoseCoord(child[idx] + (child[idx] + dir <= 1.0 && child[idx] + dir >= -1.0 ? dir : -dir));
    }
    return child;
  }

  /**
   * Crosses over two 6D quantized poses (e.g., current pose and goal pose).
   */
  _crossoverPose6D(poseA, poseB) {
    const child = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      const blend = 0.5 * (poseA[i] + poseB[i]);
      child[i] = this._quantizePoseCoord(Math.random() < 0.5 ? blend : (Math.random() < 0.5 ? poseA[i] : poseB[i]));
    }
    return child;
  }

  /**
   * Generates a random 14-dimensional servo command vector v in [-1.0, +1.0]^14 for the Duck.
   */
  _randomDuckVector() {
    const v = new Float64Array(this.actDim);
    for (let a = 0; a < this.actDim; a++) {
      v[a] = Math.random() * 2.0 - 1.0;
    }
    return v;
  }

  /**
   * Mutates a 14D servo vector by perturbing genes with Gaussian noise N(0, sigma) or occasional full gene replacement.
   */
  _mutateDuckVector(parentVec, mutationRate = 0.45, sigma = 0.45) {
    const child = Float64Array.from(parentVec);
    let mutatedCount = 0;
    for (let a = 0; a < this.actDim; a++) {
      if (Math.random() < mutationRate) {
        mutatedCount += 1;
        if (Math.random() < 0.25) {
          child[a] = Math.random() * 2.0 - 1.0;
        } else {
          child[a] = Math.max(-1.0, Math.min(1.0, child[a] + randn() * sigma));
        }
      }
    }
    while (mutatedCount < 2) {
      const idx = Math.floor(Math.random() * this.actDim);
      child[idx] = Math.max(-1.0, Math.min(1.0, child[idx] + (Math.random() * 1.4 - 0.7)));
      mutatedCount += 1;
    }
    return child;
  }

  /**
   * Crosses over two 14D parent vectors (either by functional limb blocks: Left Leg [0..4], Right Leg [5..9],
   * Neck/Head [10..13], or by uniform gene-wise crossover) with a small mutation tweak.
   */
  _crossoverDuckVectors(vecA, vecB) {
    const child = new Float64Array(this.actDim);
    if (Math.random() < 0.55) {
      const useALeft = Math.random() < 0.5;
      const useARight = Math.random() < 0.5;
      const useANeck = (useALeft === useARight) ? !useALeft : (Math.random() < 0.5);
      for (let a = 0; a < 5; a++) child[a] = useALeft ? vecA[a] : vecB[a];
      for (let a = 5; a < 10; a++) child[a] = useARight ? vecA[a] : vecB[a];
      for (let a = 10; a < this.actDim; a++) child[a] = useANeck ? vecA[a] : vecB[a];
    } else {
      for (let a = 0; a < this.actDim; a++) {
        child[a] = Math.random() < 0.5 ? vecA[a] : vecB[a];
      }
    }
    const tweakIdx = Math.floor(Math.random() * this.actDim);
    child[tweakIdx] = Math.max(-1.0, Math.min(1.0, child[tweakIdx] + randn() * 0.22));
    return child;
  }

  _makeGACandidate(vec, origin = 'случайный') {
    const v = Float64Array.from(vec);
    return {
      vec: v,
      steps: [v],
      origin,
      fitness: null,
      maxComY: 0,
      endComY: 0,
      holdSec: 0,
      matchPct: 0
    };
  }

  /**
   * Records a verified transition (pFrom -> pTo via cmdVec) in the deduplicated cell memory.
   * Only transitions between distinct quantized pose cells (D >= 0.22) are stored, preventing memory bloat.
   */
  rememberPoseTransition(pFrom, pTo, cmdVec, maxComY = 0.20) {
    if (!pFrom || !pTo || !cmdVec || cmdVec.length !== this.actDim) return false;
    const qFrom = this.quantizePose6D(pFrom);
    const qTo = this.quantizePose6D(pTo);
    const fromKey = this.getPoseKey6D(qFrom);
    const toKey = this.getPoseKey6D(qTo);
    const shiftDist = this.computePoseDistance6D(qFrom, qTo);
    if (fromKey === toKey || shiftDist < 0.22) {
      return false; // Ignore micro-jitter within the same quantized pose cell
    }

    if (!this.poseTransitionMemory) this.poseTransitionMemory = new Map();
    const transKey = `${fromKey}->${toKey}`;
    const goalPct = this.computePoseMatchPct6D(qTo, this.poseGoal);
    const existing = this.poseTransitionMemory.get(transKey);

    if (!existing || maxComY >= existing.maxComY - 0.005 || goalPct >= existing.goalPct) {
      this.poseTransitionMemory.set(transKey, {
        fromKey,
        toKey,
        pFrom: qFrom,
        pTo: qTo,
        cmdVec: Float64Array.from(cmdVec),
        matchPct: 100,
        goalPct: Number(goalPct.toFixed(1)),
        maxComY: Number(maxComY.toFixed(3))
      });
    }

    // Cap transition memory at 80 unique cell-to-cell transitions (evict lowest goalPct if full)
    if (this.poseTransitionMemory.size > 80) {
      let worstKey = null;
      let worstScore = Infinity;
      for (const [k, val] of this.poseTransitionMemory.entries()) {
        if (val.goalPct < worstScore) {
          worstScore = val.goalPct;
          worstKey = k;
        }
      }
      if (worstKey) this.poseTransitionMemory.delete(worstKey);
    }
    return true;
  }

  /**
   * Initializes GA #1 (Pose Population) anchored at the Duck's current quantized pose P_current.
   * Applies Filter #1 (Memory screening) to immediately discard pose mutants known to be unreachable from P_current.
   */
  _initPoseGAPopulation(basePose = null) {
    const startPose = this.quantizePose6D(basePose || this.poseCurrent || [-0.75, -0.25, 0.50, 0.0, 0.0, 0.0]);
    this.poseCurrent = startPose;
    const fromKey = this.getPoseKey6D(startPose);

    this.posePopSize = 4;
    this.poseCursor = 0;
    this.posePop = [];

    const origins = ['шаг к цели', 'мутация позы', 'кроссовер к цели', 'прямой шаг к цели'];
    for (let m = 0; m < this.posePopSize; m++) {
      let candidatePose = null;
      // Filter #1 (Memory-based instant screening): avoid generating a mutant already marked unreachable from fromKey
      for (let attempt = 0; attempt < 8; attempt++) {
        if (m === 0) {
          candidatePose = this._mutatePose6D(startPose, this.poseGoal, 0.70);
        } else if (m === 1) {
          candidatePose = this._mutatePose6D(startPose, this.poseGoal, 0.45);
        } else if (m === 2) {
          candidatePose = this._crossoverPose6D(startPose, this.poseGoal);
        } else {
          candidatePose = this._mutatePose6D(this._crossoverPose6D(startPose, this.poseGoal), this.poseGoal, 0.85);
        }
        const candKey = this.getPoseKey6D(candidatePose);
        const isDistinct = candKey !== fromKey && this.computePoseDistance6D(startPose, candidatePose) >= 0.22;
        const isKnownUnreachable = this.poseUnreachableMemory && this.poseUnreachableMemory.has(`${fromKey}->${candKey}`);
        if (isDistinct && !isKnownUnreachable) {
          break;
        }
      }
      this.posePop.push({
        pose: candidatePose,
        poseKey: this.getPoseKey6D(candidatePose),
        origin: origins[m],
        fitness: null,
        reachable: null,
        matchPct: 0,
        bestServoVec: null
      });
    }

    this._initServoGAForCurrentPoseMutant();
  }

  /**
   * Initializes GA #2 (Personal Servo Command Vector GA) for the active Pose Mutant in GA #1.
   * Queries poseTransitionMemory for similar starting poses (P_from ~ P_current) and target poses (P_to ~ P_mutant).
   */
  _initServoGAForCurrentPoseMutant() {
    if (!this.posePop || this.posePop.length === 0) {
      this._initPoseGAPopulation();
      return;
    }
    const pIdx = Math.max(0, Math.min(this.posePop.length - 1, this.poseCursor || 0));
    const targetMutant = this.posePop[pIdx];
    const curPose = this.poseCurrent || Float64Array.from([-0.75, -0.25, 0.50, 0.0, 0.0, 0.0]);

    this.gaPopSize = 4;
    this.gaCursor = 0;
    this.gaPopulation = [];

    // Search transition memory for known transitions starting near curPose
    const matchingMemories = [];
    if (this.poseTransitionMemory && this.poseTransitionMemory.size > 0) {
      for (const item of this.poseTransitionMemory.values()) {
        const fromDist = this.computePoseDistance6D(item.pFrom, curPose);
        if (fromDist <= 0.50) {
          const toMutantMatch = this.computePoseMatchPct6D(item.pTo, targetMutant.pose);
          const toGoalMatch = item.goalPct || this.computePoseMatchPct6D(item.pTo, this.poseGoal);
          const score = 0.65 * toMutantMatch + 0.35 * toGoalMatch - 20.0 * fromDist;
          matchingMemories.push({ item, score });
        }
      }
      matchingMemories.sort((a, b) => b.score - a.score);
    }

    if (matchingMemories.length > 0) {
      const bestMemVec = matchingMemories[0].item.cmdVec;
      this.gaPopulation.push(this._makeGACandidate(bestMemVec, 'память'));
      this.gaPopulation.push(this._makeGACandidate(this._mutateDuckVector(bestMemVec, 0.40, 0.38), 'мутация памяти'));
      if (matchingMemories.length > 1) {
        const secondMemVec = matchingMemories[1].item.cmdVec;
        this.gaPopulation.push(this._makeGACandidate(this._crossoverDuckVectors(bestMemVec, secondMemVec), 'кроссовер'));
      } else {
        this.gaPopulation.push(this._makeGACandidate(this._mutateDuckVector(bestMemVec, 0.60, 0.60), 'мутация'));
      }
    } else if (this.gaArchive && this.gaArchive.length > 0) {
      const archVec = this.gaArchive[0].vec;
      this.gaPopulation.push(this._makeGACandidate(archVec, 'память'));
      this.gaPopulation.push(this._makeGACandidate(this._mutateDuckVector(archVec, 0.45, 0.45), 'мутация'));
      if (this.gaBestVector && this.gaBestVector.length === this.actDim) {
        this.gaPopulation.push(this._makeGACandidate(this._crossoverDuckVectors(archVec, this.gaBestVector), 'кроссовер'));
      }
    } else if (this.gaBestVector && this.gaBestVector.length === this.actDim) {
      this.gaPopulation.push(this._makeGACandidate(this._mutateDuckVector(this.gaBestVector, 0.40, 0.40), 'мутация'));
      const randPartner = this._randomDuckVector();
      this.gaPopulation.push(this._makeGACandidate(this._crossoverDuckVectors(this.gaBestVector, randPartner), 'кроссовер'));
    }

    while (this.gaPopulation.length < this.gaPopSize) {
      this.gaPopulation.push(this._makeGACandidate(this._randomDuckVector(), 'случайный'));
    }
  }

  _initDuckGAPopulation() {
    this.poseGen = this.poseGen || 1;
    this.gaGeneration = this.gaGeneration || 1;
    this._initPoseGAPopulation(this.poseCurrent);
  }

  /**
   * Saves a successful 14D vector into the long-term GA archive so future tasks/falls can reuse it.
   */
  rememberDuckGAVector(vec, fitness = 1.0, maxComY = 0.25) {
    if (!vec || vec.length !== this.actDim) return;
    if (!Array.isArray(this.gaArchive)) this.gaArchive = [];
    this.gaArchive.push({
      vec: Float64Array.from(vec),
      fitness: Number(fitness) || 0,
      maxComY: Number(maxComY) || 0
    });
    this.gaArchive.sort((a, b) => b.fitness - a.fitness);
    if (this.gaArchive.length > 8) {
      this.gaArchive.length = 8;
    }
  }

  /**
   * Returns the currently active GA #2 servo vector candidate and GA #1 pose mutant metadata.
   */
  getCurrentDuckGACandidate() {
    if (!this.posePop || this.posePop.length === 0 || !this.gaPopulation || this.gaPopulation.length === 0) {
      this._initDuckGAPopulation();
    }
    const pIdx = Math.max(0, Math.min(this.posePop.length - 1, this.poseCursor || 0));
    const poseMutant = this.posePop[pIdx];
    const sIdx = Math.max(0, Math.min(this.gaPopulation.length - 1, this.gaCursor || 0));
    const servoCand = this.gaPopulation[sIdx];
    const curGoalPct = this.poseCurrent ? Math.round(this.computePoseMatchPct6D(this.poseCurrent, this.poseGoal)) : 35;

    return {
      vec: servoCand.vec,
      steps: servoCand.steps,
      origin: servoCand.origin || 'случайный',
      gen: this.gaGeneration || 1,
      idx: sIdx + 1,
      popSize: this.gaPopulation.length,
      poseGen: this.poseGen || 1,
      poseIdx: pIdx + 1,
      posePopSize: this.posePop.length,
      poseOrigin: poseMutant ? poseMutant.origin : 'шаг к цели',
      curGoalPct,
      memSize: this.poseTransitionMemory ? this.poseTransitionMemory.size : 0
    };
  }

  _resetRightingWindow(upY = 0, topY = 0.18) {
    this._winSteps = 0;
    this._winHoldSteps = 0;
    this._winStartUpY = upY;
    this._winMinUpY = upY;
    this._winMaxUpY = upY;
    this._winSumUpY = 0;
    this._winPrevUpY = upY;
    this._winStartTopY = topY;
    this._winMinTopY = topY;
    this._winMaxTopY = topY;
    this._winSumTopY = 0;
    this._winPrevTopY = topY;
    this._winStartPose6D = null;
  }

  resetRightingWaveState(upY = 0, topY = 0.18) {
    this._waveBaseCrestUpY = upY;
    this._waveBaseMeanUpY = upY;
    this._waveBaseSpanUpY = 0;
    this._waveBaseCrestTopY = topY;
    this._waveBaseMeanTopY = topY;
    this._waveBaseSpanTopY = 0;
    this._waveStagnantCount = 0;
    this._resetRightingWindow(upY, topY);
  }

  /**
   * Accumulates evaluation window statistics for the current vector:
   * records the starting 6D quantized pose on step 1, and tracks Center of Mass height (comY) & hold duration.
   */
  recordRightingStepTrace(obs, _act, metricY) {
    if (this._waveBaseCrestUpY === null) {
      this.resetRightingWaveState(metricY, 0);
    }
    if (this._winSteps === 0 && this.morphologyKey === 'duck' && obs) {
      this._winStartPose6D = this.extractDuckPose6D(obs, metricY);
      if (!this.poseCurrent) {
        this._initPoseGAPopulation(this._winStartPose6D);
      }
    }
    this._winSteps += 1;
    if (metricY < this._winMinUpY) this._winMinUpY = metricY;
    if (metricY > this._winMaxUpY) this._winMaxUpY = metricY;
    this._winSumUpY += metricY;
    this._winPrevUpY = metricY;

    if (metricY >= this._winStartUpY + 0.008 || metricY >= 0.20) {
      this._winHoldSteps = (this._winHoldSteps || 0) + 1;
    }
  }

  /**
   * Applies the active 14-dimensional servo command vector from GA #2 directly to the Duck's 14 servos.
   */
  forwardRightingReflex(obs, dt = 0.05, recordVis = true, currentComY = 0.15) {
    this.rightingPhase = (this.rightingPhase + dt * 8.4) % (Math.PI * 2);
    const mu = new Float64Array(this.actDim);

    if (this.morphologyKey === 'duck' && this.actDim === 14) {
      if (!this.poseCurrent && obs) {
        this._initPoseGAPopulation(this.extractDuckPose6D(obs, currentComY));
      }
      const cand = this.getCurrentDuckGACandidate();
      mu.set(cand.vec);
    } else {
      const zRight = new Float64Array(this.actDim);
      this.rightingLayer.forward(obs, zRight);
      const upY = obs[6];
      const rollTilt = obs[8];
      if (Math.abs(rollTilt) > 0.18) {
        this.rightingSideSign = rollTilt >= 0 ? -1.0 : 1.0;
      }
      const side = this.rightingSideSign;
      const s = Math.sin(this.rightingPhase);
      const c = Math.cos(this.rightingPhase);

      const numLegs = Math.floor(this.actDim / 2);
      for (let leg = 0; leg < numLegs; leg++) {
        const liftIdx = leg * 2;
        const strideIdx = leg * 2 + 1;
        let liftCmd = 0;
        let strideCmd = 0;

        if (upY >= 0.42) {
          liftCmd = +0.12;
          strideCmd = 0.15 * this.rightingSweep * (leg === 0 ? s : leg === 1 ? -s : c);
        } else if (leg === 0) {
          liftCmd = side >= 0
            ? -this.rightingGain * (0.85 + 0.15 * Math.max(0, s))
            : +0.15 * this.rightingAsym;
          strideCmd = this.rightingSweep * s;
        } else if (leg === 1) {
          liftCmd = side >= 0
            ? +0.15 * this.rightingAsym
            : -this.rightingGain * (0.85 + 0.15 * Math.max(0, s));
          strideCmd = -this.rightingSweep * s;
        } else {
          liftCmd = -this.rightingGain * (0.70 + 0.30 * Math.max(0, c));
          strideCmd = this.rightingSweep * c;
        }

        mu[liftIdx] = Math.tanh(liftCmd + 0.15 * zRight[liftIdx]);
        mu[strideIdx] = Math.tanh(strideCmd + 0.15 * zRight[strideIdx]);
      }
    }

    if (recordVis) {
      this.lastObs.set(obs);
      const z1 = new Float64Array(this.hidden1);
      this.actorL1.forward(obs, z1);
      for (let i = 0; i < this.hidden1; i++) {
        this.lastH1[i] = Math.tanh(z1[i]);
      }
      const z2 = new Float64Array(this.hidden2);
      this.actorL2.forward(this.lastH1, z2);
      for (let i = 0; i < this.hidden2; i++) {
        this.lastH2[i] = Math.tanh(z2[i]);
      }
      this.lastMu.set(mu);
    }

    return mu;
  }

  /**
   * Evaluates the outcome of GA #2's 14D servo vector on the Duck's 6D quantized pose:
   * - Measures start pose P_start and resulting pose P_after.
   * - Checks how close P_after is to the target Pose Mutant P_mutant (matchPct) and to P_goal (goalPct).
   * - If the Duck's physical pose genuinely changed (D(P_start, P_after) >= 0.22):
   *     1. Stores the verified transition (P_start -> P_after, U) in deduplicated poseTransitionMemory.
   *     2. Immediately updates GA #1's base pose P_current = P_after, generates fresh pose mutants around P_after,
   *        and restarts the climb toward P_goal from the new physical posture!
   * - If the pose did NOT change and all GA #2 candidates failed to reach P_mutant (bestMatchPct < 68%):
   *     marks P_mutant as unreachable (fitness = 0), stores it in poseUnreachableMemory, and moves to the next pose mutant.
   */
  updateRightingReflex({
    startUpY,
    maxUpY,
    endUpY,
    startComY = 0.15,
    maxComY = 0.15,
    endComY = 0.15,
    contactSteps,
    lastObs,
    lastAct
  }) {
    this.rightingAttempts += 1;
    const progress = Math.max(0, maxUpY - startUpY) + Math.max(0, endUpY - startUpY);

    if (this.morphologyKey === 'duck' && this.actDim === 14) {
      if (!this.posePop || this.posePop.length === 0 || !this.gaPopulation || this.gaPopulation.length === 0) {
        this._initDuckGAPopulation();
      }
      const pIdx = Math.max(0, Math.min(this.posePop.length - 1, this.poseCursor || 0));
      const targetMutant = this.posePop[pIdx];
      const sIdx = Math.max(0, Math.min(this.gaPopulation.length - 1, this.gaCursor || 0));
      const cand = this.gaPopulation[sIdx];

      const pStart = this._winStartPose6D || this.poseCurrent || this.extractDuckPose6D(lastObs, startComY);
      const pAfter = this.extractDuckPose6D(lastObs, endComY);

      const poseShiftDist = this.computePoseDistance6D(pStart, pAfter);
      const poseChanged = (this.getPoseKey6D(pStart) !== this.getPoseKey6D(pAfter)) && (poseShiftDist >= 0.22);

      const matchPct = this.computePoseMatchPct6D(pAfter, targetMutant.pose);
      const goalPct = this.computePoseMatchPct6D(pAfter, this.poseGoal);

      const steps = Math.max(1, this._winSteps || 1);
      const minComY = Number.isFinite(this._winMinUpY) ? Math.min(this._winMinUpY, startComY, endComY) : Math.min(startComY, endComY);
      const meanComY = this._winSteps > 0 ? (this._winSumUpY / steps) : 0.5 * (startComY + endComY);
      const holdSteps = this._winHoldSteps || 0;
      const holdRatio = holdSteps / steps;
      const holdSec = holdSteps * 0.05;
      const risePeak = Math.max(0, maxComY - startComY);
      const riseNet = Math.max(0, endComY - startComY);

      // GA #2 fitness: rewards closeness to target pose mutant + closeness to goal pose + physical CoM lift
      const fitness =
        0.012 * matchPct +
        0.015 * goalPct +
        3.0 * risePeak +
        2.0 * riseNet +
        1.2 * holdRatio * maxComY;

      cand.fitness = fitness;
      cand.maxComY = maxComY;
      cand.endComY = endComY;
      cand.holdSec = holdSec;
      cand.matchPct = matchPct;

      if (matchPct > (targetMutant.matchPct || 0)) {
        targetMutant.matchPct = matchPct;
        targetMutant.bestServoVec = Float64Array.from(cand.vec);
      }

      if (fitness > this.gaBestFitness) {
        this.gaBestFitness = fitness;
        this.gaBestVector = Float64Array.from(cand.vec);
        this.gaBestOrigin = cand.origin;
      }
      if (maxComY >= 0.235) {
        this.rememberDuckGAVector(cand.vec, fitness, maxComY);
      }

      const normProgress = Math.max(0, (maxComY - 0.145) * 6.0) + risePeak * 4.0;
      if (normProgress > this.duckRightingBestProgress) {
        this.duckRightingBestProgress = normProgress;
      }

      // If the Duck's physical pose actually changed to a new quantized cell:
      // 1. Save transition (pStart -> pAfter, cand.vec) into deduplicated memory
      if (poseChanged) {
        this.rememberPoseTransition(pStart, pAfter, cand.vec, maxComY);
      }

      this.lastRightingWaveMetrics = {
        gaGen: this.gaGeneration,
        gaIdx: sIdx + 1,
        gaPopSize: this.gaPopulation.length,
        gaOrigin: cand.origin,
        poseGen: this.poseGen,
        poseIdx: pIdx + 1,
        poseOrigin: targetMutant.origin,
        poseChanged,
        poseShiftDist: Number(poseShiftDist.toFixed(3)),
        poseMatchPct: Number(matchPct.toFixed(1)),
        goalMatchPct: Number(goalPct.toFixed(1)),
        memTransitions: this.poseTransitionMemory ? this.poseTransitionMemory.size : 0,
        memUnreachable: this.poseUnreachableMemory ? this.poseUnreachableMemory.size : 0,
        pStart: Array.from(pStart),
        pAfter: Array.from(pAfter),
        pMutant: Array.from(targetMutant.pose),
        fitness: Number(fitness.toFixed(3)),
        bestFitness: Number(this.gaBestFitness.toFixed(3)),
        riseCm: Number((risePeak * 100).toFixed(1)),
        holdSec: Number(holdSec.toFixed(2)),
        startComY: Number(startComY.toFixed(3)),
        minComY: Number(minComY.toFixed(3)),
        maxComY: Number(maxComY.toFixed(3)),
        meanComY: Number(meanComY.toFixed(3)),
        endComY: Number(endComY.toFixed(3)),
        vec: Array.from(cand.vec, v => Number(v.toFixed(2)))
      };

      // Live System Rule: if the Duck's pose genuinely changed during GA #2,
      // immediately update P_current = pAfter, rebuild GA #1 pose mutants around pAfter, and strive toward P_goal from the new pose!
      if (poseChanged) {
        this.poseGen += 1;
        this.gaGeneration += 1;
        this._initPoseGAPopulation(pAfter);
        this._resetRightingWindow(endComY, 0);
        return this.getRightingSkillPct();
      }

      // Otherwise (Duck remained in the same quantized pose cell), advance GA #2 cursor
      this.gaCursor = sIdx + 1;
      if (this.gaCursor >= this.gaPopulation.length) {
        this.gaGeneration += 1;
        const bestMatch = targetMutant.matchPct || 0;
        const fromKey = this.getPoseKey6D(this.poseCurrent || pStart);
        // Filter #2 (Physical trial outcome): if GA #2 could not reach targetMutant (bestMatch < 68%),
        // reject this pose mutant as unreachable and record in poseUnreachableMemory!
        if (bestMatch < 68.0) {
          targetMutant.reachable = false;
          targetMutant.fitness = 0;
          if (!this.poseUnreachableMemory) this.poseUnreachableMemory = new Set();
          this.poseUnreachableMemory.add(`${fromKey}->${targetMutant.poseKey}`);
          if (this.poseUnreachableMemory.size > 120) {
            const firstKey = this.poseUnreachableMemory.values().next().value;
            this.poseUnreachableMemory.delete(firstKey);
          }
        } else {
          targetMutant.reachable = true;
          const mGoalPct = this.computePoseMatchPct6D(targetMutant.pose, this.poseGoal);
          targetMutant.fitness = mGoalPct * Math.pow(bestMatch / 100.0, 2);
        }

        // Advance to next Pose Mutant in GA #1
        this.poseCursor = pIdx + 1;
        if (this.poseCursor >= this.posePop.length) {
          this.poseGen += 1;
          this._initPoseGAPopulation(pAfter);
        } else {
          this._initServoGAForCurrentPoseMutant();
        }
      }

      this._resetRightingWindow(endComY, 0);
      return this.getRightingSkillPct();
    }

    // Policy gradient / biomechanical reinforcement step on dorsal leverage gains for radial walkers:
    const gainStep = 0.28 + 0.15 * Math.min(1.0, progress + contactSteps * 0.05);
    this.rightingGain = Math.min(1.0, this.rightingGain + gainStep * (1.0 - this.rightingGain * 0.65));
    this.rightingAsym = Math.min(1.0, this.rightingAsym + gainStep * (1.0 - this.rightingAsym * 0.65));
    this.rightingSweep = Math.min(0.95, this.rightingSweep + 0.22 * (1.0 - this.rightingSweep * 0.7));

    if (endUpY < -0.35 && (maxUpY - startUpY) < 0.08 && this.rightingAttempts >= 3) {
      this.rightingSideSign *= -1.0;
    }

    if (lastObs && lastAct) {
      const lr = 0.015;
      for (let a = 0; a < this.actDim; a++) {
        const row = a * this.obsDim;
        for (let c = 6; c <= 11 && c < this.obsDim; c++) {
          this.rightingLayer.W[row + c] = Math.max(
            -0.6,
            Math.min(0.6, this.rightingLayer.W[row + c] + lr * progress * lastAct[a] * lastObs[c])
          );
        }
      }
    }

    return this.getRightingSkillPct();
  }

  getRightingSkillPct() {
    if (this.morphologyKey === 'duck') {
      const successScore = Math.min(60, this.rightingSuccesses * 25);
      const progressScore = Math.min(35, Math.round((this.duckRightingBestProgress || 0) * 45));
      return Math.round(Math.max(5, Math.min(100, 5 + successScore + progressScore)));
    }
    const raw = ((this.rightingGain - 0.32) / (1.0 - 0.32)) * 80 + 20;
    return Math.round(Math.max(20, Math.min(100, raw)));
  }

  /**
   * Forward pass of the Actor network, optionally with parameter-space exploration perturbation
   * on the skip matrix for coherent gait & steering exploration.
   */
  forwardActor(obs, recordVis = false, skipPerturb = null) {
    const z1 = new Float64Array(this.hidden1);
    const h1 = new Float64Array(this.hidden1);
    this.actorL1.forward(obs, z1);
    for (let i = 0; i < this.hidden1; i++) h1[i] = Math.tanh(z1[i]);

    const z2 = new Float64Array(this.hidden2);
    const h2 = new Float64Array(this.hidden2);
    this.actorL2.forward(h1, z2);
    for (let i = 0; i < this.hidden2; i++) h2[i] = Math.tanh(z2[i]);

    const zDeep = new Float64Array(this.actDim);
    const zSkip = new Float64Array(this.actDim);
    this.actorOut.forward(h2, zDeep);
    this.actorSkip.forward(obs, zSkip);

    if (skipPerturb) {
      for (let a = 0; a < this.actDim; a++) {
        const row = a * this.obsDim;
        let pSum = 0;
        for (let i = 0; i < this.obsDim; i++) {
          pSum += skipPerturb[row + i] * obs[i];
        }
        zSkip[a] += pSum;
      }
    }

    const preTanh = new Float64Array(this.actDim);
    const mu = new Float64Array(this.actDim);
    for (let a = 0; a < this.actDim; a++) {
      preTanh[a] = zDeep[a] + zSkip[a];
      mu[a] = Math.tanh(preTanh[a]);
    }

    if (recordVis) {
      this.lastObs.set(obs);
      this.lastH1.set(h1);
      this.lastH2.set(h2);
      this.lastMu.set(mu);
    }

    return { h1, h2, preTanh, mu };
  }

  /**
   * Forward pass of the Critic network V(s).
   */
  forwardCritic(obs, recordVis = false) {
    const z1 = new Float64Array(this.hidden1);
    const h1 = new Float64Array(this.hidden1);
    this.criticL1.forward(obs, z1);
    for (let i = 0; i < this.hidden1; i++) h1[i] = Math.tanh(z1[i]);

    const z2 = new Float64Array(this.hidden2);
    const h2 = new Float64Array(this.hidden2);
    this.criticL2.forward(h1, z2);
    for (let i = 0; i < this.hidden2; i++) h2[i] = Math.tanh(z2[i]);

    const out = new Float64Array(1);
    this.criticOut.forward(h2, out);
    if (recordVis) {
      this.lastValue = out[0];
    }
    return { h1, h2, value: out[0] };
  }

  /**
   * Backpropagates gradient dL/d(mu) through the Actor network (accumulating into dW, db).
   */
  backwardActorFromMu(obs, forwardCache, dLdMu) {
    const { h1, h2, mu } = forwardCache;
    const dPreTanh = new Float64Array(this.actDim);
    for (let a = 0; a < this.actDim; a++) {
      const dtanh = Math.max(0.08, 1.0 - mu[a] * mu[a]);
      dPreTanh[a] = dLdMu[a] * dtanh;
    }

    // Backprop through skip layer
    this.actorSkip.backward(obs, dPreTanh, null);

    // Backprop through deep MLP path
    const dH2 = new Float64Array(this.hidden2);
    this.actorOut.backward(h2, dPreTanh, dH2);

    const dZ2 = new Float64Array(this.hidden2);
    for (let i = 0; i < this.hidden2; i++) {
      dZ2[i] = dH2[i] * (1.0 - h2[i] * h2[i]);
    }

    const dH1 = new Float64Array(this.hidden1);
    this.actorL2.backward(h1, dZ2, dH1);

    const dZ1 = new Float64Array(this.hidden1);
    for (let i = 0; i < this.hidden1; i++) {
      dZ1[i] = dH1[i] * (1.0 - h1[i] * h1[i]);
    }

    this.actorL1.backward(obs, dZ1, null);
  }

  /**
   * Backpropagates scalar gradient dL/dV through the Critic network.
   */
  backwardCritic(obs, criticCache, dLdV) {
    const { h1, h2 } = criticCache;
    const dOut = new Float64Array([dLdV]);
    const dH2 = new Float64Array(this.hidden2);
    this.criticOut.backward(h2, dOut, dH2);

    const dZ2 = new Float64Array(this.hidden2);
    for (let i = 0; i < this.hidden2; i++) {
      dZ2[i] = dH2[i] * (1.0 - h2[i] * h2[i]);
    }

    const dH1 = new Float64Array(this.hidden1);
    this.criticL2.backward(h1, dZ2, dH1);

    const dZ1 = new Float64Array(this.hidden1);
    for (let i = 0; i < this.hidden1; i++) {
      dZ1[i] = dH1[i] * (1.0 - h1[i] * h1[i]);
    }

    this.criticL1.backward(obs, dZ1, null);
  }

  zeroActorGrad() {
    this.actorL1.zeroGrad();
    this.actorL2.zeroGrad();
    this.actorOut.zeroGrad();
    this.actorSkip.zeroGrad();
  }

  zeroCriticGrad() {
    this.criticL1.zeroGrad();
    this.criticL2.zeroGrad();
    this.criticOut.zeroGrad();
  }

  stepAdam(actorLr = 0.003, criticLr = 0.0045, deepScale = 0.25) {
    this.adamStepCount += 1;
    const s = this.adamStepCount;
    this.actorL1.adamStep(actorLr * deepScale, 0.9, 0.999, 1e-8, s, 0.5);
    this.actorL2.adamStep(actorLr * deepScale, 0.9, 0.999, 1e-8, s, 0.5);
    this.actorOut.adamStep(actorLr * deepScale, 0.9, 0.999, 1e-8, s, 0.5);
    this.actorSkip.adamStep(actorLr * 0.85, 0.9, 0.999, 1e-8, s, 0.6);

    // Gentle L2 weight decay on deep residual MLP layers + bias centering + row norm bounding on actorSkip.W
    // so single-walker trajectories never accumulate directional bias or post-convergence saturation
    for (const layer of [this.actorL1, this.actorL2, this.actorOut]) {
      const W = layer.W;
      for (let i = 0; i < W.length; i++) {
        W[i] *= 0.992;
      }
      for (let o = 0; o < layer.b.length; o++) {
        layer.b[o] *= 0.90;
      }
    }
    for (let o = 0; o < this.actorSkip.b.length; o++) {
      this.actorSkip.b[o] = 0;
    }
    const skipW = this.actorSkip.W;
    for (let a = 0; a < this.actDim; a++) {
      const row = a * this.obsDim;
      let normSq = 0;
      for (let c = 0; c < this.obsDim; c++) {
        // Keep actorSkip strictly on oscillatory CPG & sensory-gated CPG channels (10..15)
        // so static sensors never create a DC leg lock!
        if (c < 10 || c > 15) {
          skipW[row + c] = 0;
          continue;
        }
        const w = Math.max(-1.25, Math.min(1.25, skipW[row + c]));
        skipW[row + c] = w;
        normSq += w * w;
      }
      const maxRowNorm = this.maxSkipRowNorm || 1.80;
      if (normSq > maxRowNorm * maxRowNorm) {
        const scale = maxRowNorm / Math.sqrt(normSq);
        for (let c = 10; c <= 15 && c < this.obsDim; c++) {
          skipW[row + c] *= scale;
        }
      }
    }

    this.criticL1.adamStep(criticLr, 0.9, 0.999, 1e-8, s, 1.0);
    this.criticL2.adamStep(criticLr, 0.9, 0.999, 1e-8, s, 1.0);
    this.criticOut.adamStep(criticLr, 0.9, 0.999, 1e-8, s, 1.0);
  }

  getActorFlatParams() {
    const layers = [this.actorL1, this.actorL2, this.actorOut, this.actorSkip];
    let total = 0;
    for (const l of layers) total += l.W.length + l.b.length;
    const flat = new Float64Array(total);
    let offset = 0;
    for (const l of layers) {
      flat.set(l.W, offset);
      offset += l.W.length;
      flat.set(l.b, offset);
      offset += l.b.length;
    }
    return flat;
  }

  setActorFlatParams(flat) {
    const layers = [this.actorL1, this.actorL2, this.actorOut, this.actorSkip];
    let offset = 0;
    for (const l of layers) {
      l.W.set(flat.subarray(offset, offset + l.W.length));
      offset += l.W.length;
      l.b.set(flat.subarray(offset, offset + l.b.length));
      offset += l.b.length;
    }
  }
}

/**
 * PPO (Proximal Policy Optimization) + GAE + Meta-Plasticity Homeostasis + Beetle Self-Righting Reflex
 * Operates directly on the single visible walker (numWorkers = 1) or a population.
 */
export class PPOTrainer {
  constructor(brain, numWorkers = 1, {
    rolloutSteps = 32,
    gamma = 0.98,
    gaeLambda = 0.94,
    clipEps = 0.22,
    actorLr = 0.0032,
    criticLr = 0.005,
    ppoEpochs = 4
  } = {}) {
    this.brain = brain;
    this.numWorkers = numWorkers;
    this.rolloutSteps = rolloutSteps;
    this.gamma = gamma;
    this.gaeLambda = gaeLambda;
    this.clipEps = clipEps;
    this.actorLr = actorLr;
    this.criticLr = criticLr;
    this.ppoEpochs = ppoEpochs;

    this.stepCursor = 0;
    this.updateCount = 0;

    // Rollout buffers: [step][worker]
    this.bufObs = [];
    this.bufAct = [];
    this.bufLogProb = [];
    this.bufRew = [];
    this.bufVal = [];
    this.bufDone = [];
    this._resetBuffers();

    // Coherent parameter-space exploration perturbation for single-walker temporal windows
    this.paramSigma = 0.14;
    this.workerSkipPerturbs = [];
    this.bestLocomotionScore = -Infinity;
    this.baselineLocomotionScore = 0.12;
    this._sampleWorkerParamNoise();

    // Temporally correlated Ornstein-Uhlenbeck action exploration
    this.ouNoise = Array.from({ length: numWorkers }, () => new Float64Array(brain.actDim));

    // Elite Self-Imitation Buffer
    this.eliteReplay = [];
    this.maxEliteSize = 256;

    // Beetle Self-Righting ("Рефлекс жука") learning window state when overturned (up.y < 0.35)
    this.rightingStepCursor = 0;
    this.rightingHorizon = 14; // 0.70s micro-epoch per self-righting attempt
    this.rightingStartUpY = -1.0;
    this.rightingMaxUpY = -1.0;
    this.rightingContactSteps = 0;
    this.wasOverturned = false;
    this.modeBeforeOverturn = 'LEARNING_WALK';

    // Homeostatic Meta-Plasticity Sense ("Чутьё нейросети: нужно ли дообучаться или нет")
    const isDuckBrain = brain.morphologyKey === 'duck';
    this.maxCpgGain = 1.95;
    this.minFoodsToConsolidate = isDuckBrain ? 0 : 2;
    this.alwaysLearn = false;
    this.manualFreeze = false;

    this.metaState = {
      mode: 'LEARNING_WALK', // 'LEARNING_WALK' | 'CONSOLIDATED' | 'LEARNING_RIGHTING'
      ambitionPreset: 'min',
      isLearningActive: true,
      modeTitle: isDuckBrain
        ? '🟢 АКТИВНОЕ ОБУЧЕНИЕ: МАКСИМИЗАЦИЯ ВЫСОТЫ ЦМ'
        : '🟢 АКТИВНОЕ ОБУЧЕНИЕ: БАЗОВЫЙ ШАГ К ЦЕЛИ',
      reasonText: isDuckBrain
        ? 'Единая задача: максимизация высоты центра масс (ЦМ)'
        : 'Формирование стабильного базового навыка ходьбы и наведения на цель',
      competenceEMA: isDuckBrain ? 0.265 : 0.04,
      targetCompetence: isDuckBrain ? 0.235 : 0.18,
      relearnThreshold: isDuckBrain ? 0.205 : 0.125,
      surpriseEMA: 0.25,
      needToLearnPct: isDuckBrain ? 35 : 88,
      stableEpochsCount: 0,
      lowVelEpochsCount: 0,
      stallStepsCount: 0,
      rightingAttempts: 0,
      rightingSuccesses: 0,
      rightingSkillPct: brain.getRightingSkillPct()
    };

    // Telemetry metrics
    this.lastPolicyLoss = 0;
    this.lastValueLoss = 0;
    this.lastEntropy = 0;
    this.lastMeanStepReward = 0;
  }

  _resetBuffers() {
    this.bufObs = [];
    this.bufAct = [];
    this.bufLogProb = [];
    this.bufRew = [];
    this.bufVal = [];
    this.bufDone = [];
    this.stepCursor = 0;
    this.epochVelSum = 0;
    this.epochStraightSteps = 0;
  }

  /**
   * External notification when environment changes (e.g. floor friction slider changed)
   */
  notifyEnvironmentChange(reason = 'Изменились физические параметры среды — проверка адаптации', nextMu = null) {
    this.metaState.surpriseEMA = Math.max(this.metaState.surpriseEMA, 0.9);
    if (this.manualFreeze || this.metaState.mode === 'LEARNING_RIGHTING') {
      return;
    }
    const isSlipperyOrChanged = typeof nextMu === 'number' ? nextMu < 0.78 : true;
    if (this.metaState.mode === 'CONSOLIDATED' || isSlipperyOrChanged) {
      this.metaState.mode = 'LEARNING_WALK';
      this.metaState.isLearningActive = true;
      this.metaState.stableEpochsCount = 0;
      this.metaState.lowVelEpochsCount = 0;
      this.metaState.stallStepsCount = 0;
      this.metaState.competenceEMA = Number(
        Math.min(this.metaState.competenceEMA, (this.metaState.targetCompetence || 0.18) * 0.58).toFixed(3)
      );
      this.minTrainEpochsRemaining = Math.max(this.minTrainEpochsRemaining || 0, 2);
      this.paramSigma = Math.max(this.paramSigma, 0.085);
      this._sampleWorkerParamNoise();
      this.metaState.needToLearnPct = Math.max(this.metaState.needToLearnPct || 0, 78);
      this.metaState.modeTitle = '🟢 АКТИВНОЕ ДООБУЧЕНИЕ: АДАПТАЦИЯ К СРЕДЕ';
      this.metaState.reasonText = reason;
    }
  }

  /**
   * Adjusts the balance between learning and consolidation ("Амбиции обучения / Порог мастерства").
   * Supports presets: 'min' (calm crawl), 'bound' (energetic hop, default), 'turbo' (max gallop), 'infinite' (always learn),
   * or a numeric target speed in [0.14, 0.60] m/s.
   */
  setLearningAmbition(presetOrTarget) {
    this.manualFreeze = false;
    const isDuckBrain = this.brain.morphologyKey === 'duck';
    const prevTarget = this.metaState.targetCompetence || (isDuckBrain ? 0.215 : 0.18);
    const prevPreset = this.metaState.ambitionPreset || 'min';

    if (isDuckBrain) {
      this.minFoodsToConsolidate = 0;
      if (presetOrTarget === 'min') {
        this.metaState.ambitionPreset = 'min';
        this.metaState.targetCompetence = 0.235;
        this.metaState.relearnThreshold = 0.205;
        this.alwaysLearn = false;
      } else if (presetOrTarget === 'bound') {
        this.metaState.ambitionPreset = 'bound';
        this.metaState.targetCompetence = 0.268;
        this.metaState.relearnThreshold = 0.245;
        this.alwaysLearn = false;
      } else if (presetOrTarget === 'turbo') {
        this.metaState.ambitionPreset = 'turbo';
        this.metaState.targetCompetence = 0.280;
        this.metaState.relearnThreshold = 0.252;
        this.alwaysLearn = false;
      } else if (presetOrTarget === 'infinite') {
        this.metaState.ambitionPreset = 'infinite';
        this.metaState.targetCompetence = 0.360;
        this.metaState.relearnThreshold = 0.280;
        this.alwaysLearn = true;
      } else if (typeof presetOrTarget === 'number') {
        const t = Math.max(0.22, Math.min(0.36, presetOrTarget));
        this.metaState.ambitionPreset = 'custom';
        this.metaState.targetCompetence = Number(t.toFixed(3));
        this.metaState.relearnThreshold = Number((t - 0.030).toFixed(3));
        this.alwaysLearn = t >= 0.355;
      }
      if (this.metaState.mode !== 'LEARNING_RIGHTING') {
        if (this.alwaysLearn || this.metaState.competenceEMA < this.metaState.targetCompetence) {
          this.metaState.mode = 'LEARNING_WALK';
          this.metaState.isLearningActive = true;
          this.metaState.stableEpochsCount = 0;
          this.metaState.needToLearnPct = this.alwaysLearn ? 100 : 65;
          this.metaState.modeTitle = this.alwaysLearn
            ? '♾️ ВЕЧНОЕ ОБУЧЕНИЕ: МАКСИМИЗАЦИЯ ВЫСОТЫ ЦМ'
            : '🟢 АКТИВНОЕ ОБУЧЕНИЕ: ПОДЪЁМ ЦЕНТРА МАСС';
          this.metaState.reasonText = `Утка стремится поднять и удерживать ЦМ ≥ ${this.metaState.targetCompetence.toFixed(2)} м`;
        } else {
          this.metaState.mode = 'CONSOLIDATED';
          this.metaState.isLearningActive = false;
          this.metaState.needToLearnPct = 0;
          this.metaState.modeTitle = '🛡️ ЗАДАЧА ВЫПОЛНЕНА: ВЫСОТА ЦМ УДЕРЖИВАЕТСЯ';
          this.metaState.reasonText = `Высота ЦМ (${this.metaState.competenceEMA.toFixed(2)} м) ≥ порога ${this.metaState.targetCompetence.toFixed(2)} м — обучение остановлено`;
        }
      }
      return;
    }

    if (presetOrTarget === 'min') {
      this.metaState.ambitionPreset = 'min';
      this.metaState.targetCompetence = 0.18;
      this.metaState.relearnThreshold = 0.125;
      this.minFoodsToConsolidate = 2;
      this.maxCpgGain = 1.95;
      this.brain.maxSkipRowNorm = 1.30;
      this.alwaysLearn = false;
    } else if (presetOrTarget === 'bound') {
      this.metaState.ambitionPreset = 'bound';
      this.metaState.targetCompetence = 0.34;
      this.metaState.relearnThreshold = 0.23;
      this.minFoodsToConsolidate = 4;
      this.maxCpgGain = 2.45;
      this.brain.maxSkipRowNorm = 1.52;
      this.alwaysLearn = false;
    } else if (presetOrTarget === 'turbo') {
      this.metaState.ambitionPreset = 'turbo';
      this.metaState.targetCompetence = 0.48;
      this.metaState.relearnThreshold = 0.32;
      this.minFoodsToConsolidate = 6;
      this.maxCpgGain = 2.95;
      this.brain.maxSkipRowNorm = 1.75;
      this.alwaysLearn = false;
    } else if (presetOrTarget === 'infinite') {
      this.metaState.ambitionPreset = 'infinite';
      this.metaState.targetCompetence = 0.60;
      this.metaState.relearnThreshold = 0.45;
      this.minFoodsToConsolidate = 99999;
      this.maxCpgGain = 3.10;
      this.brain.maxSkipRowNorm = 1.85;
      this.alwaysLearn = true;
    } else if (typeof presetOrTarget === 'number') {
      const t = Math.max(0.14, Math.min(0.60, presetOrTarget));
      this.metaState.ambitionPreset = 'custom';
      this.metaState.targetCompetence = Number(t.toFixed(2));
      this.metaState.relearnThreshold = Number((t * 0.68).toFixed(2));
      this.minFoodsToConsolidate = t < 0.22 ? 2 : t < 0.40 ? 4 : 6;
      this.maxCpgGain = Number((1.4 + (t - 0.14) * 3.8).toFixed(2));
      this.brain.maxSkipRowNorm = Number((1.10 + (t - 0.14) * 1.65).toFixed(2));
      this.alwaysLearn = t >= 0.59;
    }

    const raisedAmbition =
      this.metaState.targetCompetence > prevTarget + 0.005 ||
      (presetOrTarget !== 'min' && presetOrTarget !== prevPreset);

    if (raisedAmbition) {
      // Scale competenceEMA relative to the newly unlocked CPG regime so raising ambition ALWAYS starts active training!
      this.metaState.competenceEMA = Number(
        Math.min(this.metaState.competenceEMA, this.metaState.targetCompetence * 0.65).toFixed(3)
      );
      this.minTrainEpochsRemaining = this.metaState.targetCompetence >= 0.44 ? 6 : 4;
    }

    // Immediately re-evaluate if the walker should wake up to train harder or consolidate now!
    if (this.metaState.mode !== 'LEARNING_RIGHTING') {
      if (this.alwaysLearn || raisedAmbition || this.metaState.competenceEMA < this.metaState.targetCompetence) {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.stableEpochsCount = 0;
        this.paramSigma = Math.max(this.paramSigma, 0.105);
        this._sampleWorkerParamNoise();
        this.metaState.needToLearnPct = Math.round(
          Math.max(45, Math.min(98, ((this.metaState.targetCompetence - this.metaState.competenceEMA) / this.metaState.targetCompetence) * 90 + 30))
        );
        this.metaState.modeTitle = this.alwaysLearn
          ? '♾️ ВЕЧНОЕ ОБУЧЕНИЕ: МАКСИМАЛЬНЫЙ РАЗГОН'
          : this.metaState.targetCompetence >= 0.44
            ? '🟢 АКТИВНОЕ ОБУЧЕНИЕ: РАЗГОН ГАЛОПА К ЦЕЛИ'
            : '🟢 АКТИВНОЕ ОБУЧЕНИЕ: РАЗГОН СКАЧКА К ЦЕЛИ';
        this.metaState.reasonText = this.alwaysLearn
          ? 'Включён режим непрерывного обучения (100% пластичность без остановки)'
          : `Планка мастерства повышена до ${this.metaState.targetCompetence.toFixed(2)} м/с — шагоход дообучает ${this.metaState.targetCompetence >= 0.44 ? 'мощный галоп' : 'прыгучий скачок'}!`;
      } else {
        this.metaState.mode = 'CONSOLIDATED';
        this.metaState.isLearningActive = false;
        this.metaState.needToLearnPct = 10;
        this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
        this.metaState.reasonText = `Текущая скорость (${this.metaState.competenceEMA.toFixed(2)} м/с) уже выше порога ${this.metaState.targetCompetence.toFixed(2)} м/с — веса зафиксированы`;
      }
    }
  }

  /**
   * Manual toggle to either freeze weights immediately or force active learning
   */
  toggleManualFreeze() {
    if (this.metaState.mode === 'LEARNING_RIGHTING') return;
    if (this.metaState.mode === 'CONSOLIDATED') {
      this.manualFreeze = false;
      this.metaState.mode = 'LEARNING_WALK';
      this.metaState.isLearningActive = true;
      this.metaState.stableEpochsCount = 0;
      this.metaState.lowVelEpochsCount = 0;
      this.metaState.stallStepsCount = 0;
      this.minTrainEpochsRemaining = Math.max(this.minTrainEpochsRemaining || 0, 2);
      this.metaState.needToLearnPct = 80;
      this.paramSigma = Math.max(this.paramSigma, 0.085);
      this._sampleWorkerParamNoise();
      this.metaState.modeTitle = '🟢 АКТИВНОЕ ДООБУЧЕНИЕ (ВКЛЮЧЕНО ВРУЧНУЮ)';
      this.metaState.reasonText = this.brain.morphologyKey === 'duck'
        ? 'Пользователь вручную запустил дообучение максимизации высоты центра масс'
        : 'Пользователь вручную запустил процесс дообучения походки';
    } else {
      this.manualFreeze = true;
      this.alwaysLearn = false;
      this.metaState.mode = 'CONSOLIDATED';
      this.metaState.isLearningActive = false;
      this.metaState.needToLearnPct = 0;
      this.metaState.modeTitle = '🔒 ОБУЧЕНИЕ ЗАМОРОЖЕНО ВРУЧНУЮ';
      this.metaState.reasonText = 'Веса нейросети зафиксированы пользователем — шагоход действует на текущем навыке';
    }
  }

  _sampleWorkerParamNoise() {
    const dim = this.brain.actorSkip.W.length;
    this.workerSkipPerturbs = [];
    const tpl = this.brain.cpgTemplate;
    const isDuckBrain = this.brain.morphologyKey === 'duck';

    for (let w = 0; w < this.numWorkers; w++) {
      // If more than 1 worker, worker 0 is clean leader; if single walker (numWorkers === 1),
      // worker 0 explores only when metaState.mode === 'LEARNING_WALK'!
      if (this.numWorkers > 1 && w === 0) {
        this.workerSkipPerturbs.push(null);
        continue;
      }
      const p = new Float64Array(dim);
      if (isDuckBrain) {
        // For the Duck, only perturb postural balance & leg extension channels (no forward-walking CPG drive)
        for (let a = 0; a < this.brain.actDim; a++) {
          const row = a * this.brain.obsDim;
          for (let c = 3; c < this.brain.obsDim; c++) {
            if (c >= 10 && c <= 15) continue;
            p[row + c] = randn() * this.paramSigma * 0.25;
          }
        }
        this.workerSkipPerturbs.push(p);
        continue;
      }
      // Coherent low-dimensional CPG resonance gain exploration + synaptic fine-tuning:
      // Biases exploration toward constructive stride/lift/turn amplification so a single walker learns rapidly
      const dGainLift = (Math.random() * 1.15 - 0.35) * this.paramSigma;
      const dGainFwd = (Math.random() * 1.25 - 0.30) * this.paramSigma;
      const dGainTurn = (Math.random() * 1.25 - 0.30) * this.paramSigma;

      for (let a = 0; a < this.brain.actDim; a++) {
        const row = a * this.brain.obsDim;
        for (let c = 10; c <= 15 && c < this.brain.obsDim; c++) {
          const baseTpl = tpl ? (tpl[row + c] / 0.22) : 0;
          const coh = (c === 10 || c === 11)
            ? dGainLift * baseTpl
            : (c === 12 || c === 13)
              ? dGainFwd * baseTpl
              : dGainTurn * baseTpl;
          p[row + c] = coh + randn() * this.paramSigma * 0.45;
        }
      }
      this.workerSkipPerturbs.push(p);
    }
  }

  computeLogProb(mu, act) {
    let lp = 0;
    for (let a = 0; a < this.brain.actDim; a++) {
      const std = Math.exp(this.brain.logStd[a]);
      const varInv = 1.0 / (std * std);
      const diff = act[a] - mu[a];
      lp += -0.5 * diff * diff * varInv - this.brain.logStd[a] - 0.5 * Math.log(2 * Math.PI);
    }
    return lp;
  }

  selectActions(creatures) {
    const stepObs = new Array(this.numWorkers);
    const stepAct = new Array(this.numWorkers);
    const stepLogProb = new Float64Array(this.numWorkers);
    const stepVal = new Float64Array(this.numWorkers);

    const c0 = creatures[0];
    const isDuck0 = c0.morphologyKey === 'duck';
    const upY0 = c0.getTorsoFrame().up.y;
    const comY0 = c0.body.getCenterOfMass().y;
    const topY0 = c0.torsoTop ? c0.torsoTop.pos.y : comY0;

    // For the Duck, only Center of Mass height (comY0) determines whether it has fallen (< 0.195 m)
    // or is coping with the task (>= targetCompetence ~0.215 m). Torso tilt and upper-torso height are ignored!
    const isPhysicallyTiltedOnBack =
      !c0.isGrabbed &&
      !(c0.postThrowTimer > 0) &&
      (isDuck0
        ? (((comY0 < (this.alwaysLearn ? (this.metaState.relearnThreshold || 0.238) : 0.205)) || upY0 < 0.45) && !this.manualFreeze)
        : (upY0 < 0.22));
    const confirmDelay = isDuck0 ? 1 : (this.overturnConfirmDelay || 20);

    if (!this.wasOverturned) {
      if (isPhysicallyTiltedOnBack) {
        this.overturnConfirmSteps = (this.overturnConfirmSteps || 0) + 1;
        // For radial spiders (Tripod/Quadruped), wait 1.0s on back before starting;
        // for the Duck, never force calmAct=0 ("dead pause") — let the neural network act continuously!
        if (!isDuck0 && this.overturnConfirmSteps < confirmDelay) {
          if (this.overturnConfirmSteps >= 6) {
            const obs = c0.getObservation();
            const calmAct = new Float64Array(this.brain.actDim);
            c0.lastState = obs;
            c0.lastAction = calmAct;
            c0.applyActionVector(calmAct, false);
            this._pendingStep = { isRightingStep: false, isOverturnPause: true };
            return;
          }
        }
      } else {
        this.overturnConfirmSteps = 0;
      }
    }

    this.uprightHoldSteps = 0;
    const targetCom0 = this.metaState.targetCompetence || 0.255;
    if (isDuck0) {
      const isTrueHighCom = comY0 >= targetCom0 && (!c0.torsoTop || !c0.torsoBottom || c0.torsoTop.pos.y > c0.torsoBottom.pos.y + 0.15);
      this.duckComHoldSteps = isTrueHighCom ? (this.duckComHoldSteps || 0) + 1 : 0;
    }
    const isRecoveredUpright = isDuck0
      ? (this.duckComHoldSteps >= 8)
      : (upY0 >= 0.65 && !c0.hasInvertedLeg());

    const isOverturned0 =
      !c0.isGrabbed &&
      !(c0.postThrowTimer > 0) &&
      ((isPhysicallyTiltedOnBack && this.overturnConfirmSteps >= confirmDelay) ||
        (this.wasOverturned && !isRecoveredUpright));

    // 1. Check if main walker is confirmed overturned / low-CoM -> engage wave-envelope CoM maximization / righting!
    if (isOverturned0) {
      if (!this.wasOverturned) {
        this.wasOverturned = true;
        this.uprightHoldSteps = 0;
        this.rightingPauseStepsRemaining = 0;
        this.modeBeforeOverturn = this.metaState.mode === 'LEARNING_RIGHTING' ? 'LEARNING_WALK' : this.metaState.mode;
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingStartTopY = topY0;
        this.rightingMaxTopY = topY0;
        this.rightingStartComY = comY0;
        this.rightingMaxComY = comY0;
        this.rightingContactSteps = 0;
        this.rightingBottomContactSteps = 0;
        this.rightingHeadContactSteps = 0;
        this.brain.rightingPhase = 0;
        this.brain._duckInHighHeadstand = false;
        this.brain.resetRightingWaveState(isDuck0 ? comY0 : upY0, topY0);
        if (isDuck0) {
          this.brain._initPoseGAPopulation(this.brain.extractDuckPose6D(c0.getObservation(), comY0));
        }
      }

      this.metaState.mode = 'LEARNING_RIGHTING';
      this.metaState.isLearningActive = true;
      this.metaState.needToLearnPct = 100;
      this.metaState.rightingAttempts = this.brain.rightingAttempts + 1;
      this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();
      this.metaState.modeTitle = isDuck0
        ? '🧬 ДВУХУРОВНЕВЫЙ ГА: ЯЗЫК ПОЗ (ГА №1 Поз + ГА №2 Серво)'
        : '🔄 ПОПЫТКА ПОДНЯТЬСЯ: УЧИТСЯ ВСТАВАТЬ НА НОГИ';

      const obs = c0.getObservation();
      // Rest pause between righting attempts (only for radial spiders; disabled for the Duck so we never interrupt its GA vectors!)
      if (!isDuck0 && this.rightingPauseStepsRemaining > 0) {
        const calmAct = new Float64Array(this.brain.actDim);
        c0.lastState = obs;
        c0.lastAction = calmAct;
        c0.applyActionVector(calmAct, false);
        this.metaState.reasonText = `Передышка на спине перед попыткой #${this.metaState.rightingAttempts} (гашение колебаний, навык ${this.metaState.rightingSkillPct}%)`;
        this._pendingStep = { isRightingStep: true, isRestPause: true, obs, act: calmAct, upY: upY0, comY: comY0 };
        return;
      }

      if (isDuck0) {
        const cand = this.brain.getCurrentDuckGACandidate();
        this.metaState.reasonText = `ГА1 Поза #${cand.poseGen}.${cand.poseIdx} (${cand.poseOrigin}, к цели ${cand.curGoalPct}%) | ГА2 Вектор #${cand.idx}/${cand.popSize} (${cand.origin}) | память: ${cand.memSize} перех. | ЦМ=${comY0.toFixed(3)} м`;
      } else {
        this.metaState.reasonText = `Корпус перевёрнут/наклонён (up.y=${upY0.toFixed(2)}): память ходьбы защищена, о еде не думает — попытка подняться #${this.metaState.rightingAttempts} (навык ${this.metaState.rightingSkillPct}%)`;
      }

      const muRight = this.brain.forwardRightingReflex(obs, 0.05, true, comY0);
      c0.lastState = obs;
      c0.lastAction = muRight;
      c0.applyActionVector(muRight, isDuck0 ? (comY0 < 0.245 || upY0 < 0.58) : (upY0 < 0.42));

      this._pendingStep = { isRightingStep: true, isRestPause: false, obs, act: muRight, upY: upY0, comY: comY0 };
      return;
    }

    // 2. If walker just recovered its target Center of Mass / upright posture!
    if (this.wasOverturned && isRecoveredUpright) {
      this._completeRightingRecovery(comY0);
    }

    const isTrainingWalk = this.metaState.mode === 'LEARNING_WALK';

    for (let w = 0; w < this.numWorkers; w++) {
      const c = creatures[w];
      const obs = c.getObservation();
      const isLeader = w === 0;

      // Evaluate base policy mean for PPO logProb
      const { mu: baseMu } = this.brain.forwardActor(obs, isLeader, null);
      // In LEARNING_WALK mode, apply coherent parameter exploration; in CONSOLIDATED mode, use pure mastered policy
      const activePerturb = isTrainingWalk ? this.workerSkipPerturbs[w] : null;
      const { mu: workerMu } = activePerturb
        ? this.brain.forwardActor(obs, false, activePerturb)
        : { mu: baseMu };

      const { value } = this.brain.forwardCritic(obs, isLeader);

      const act = new Float64Array(this.brain.actDim);
      const ou = this.ouNoise[w];
      for (let a = 0; a < this.brain.actDim; a++) {
        const std = Math.exp(this.brain.logStd[a]);
        ou[a] = 0.78 * ou[a] + Math.sqrt(1 - 0.78 * 0.78) * randn();
        const noiseScale = !isTrainingWalk ? 0.015 : (isDuck0 ? 0.05 : (this.numWorkers === 1 ? 0.14 : (w === 0 ? 0.02 : 0.32)));
        const raw = workerMu[a] + ou[a] * std * noiseScale;
        act[a] = Math.max(-1, Math.min(1, raw));
      }

      stepObs[w] = obs;
      stepAct[w] = act;
      stepLogProb[w] = this.computeLogProb(baseMu, act);
      stepVal[w] = value;

      c.lastState = obs;
      c.lastAction = act;
      c.applyActionVector(act, false);
    }

    this._pendingStep = { isRightingStep: false, stepObs, stepAct, stepLogProb, stepVal };
  }

  _completeRightingRecovery(recoveredComY = null) {
    this.wasOverturned = false;
    this.uprightHoldSteps = 0;
    this.overturnConfirmSteps = 0;
    this.rightingPauseStepsRemaining = 0;
    this.brain.rightingSuccesses += 1;
    this.metaState.rightingSuccesses = this.brain.rightingSuccesses;
    // Lock in the successful righting reflex!
    this.brain.rightingGain = Math.min(1.0, Math.max(0.88, this.brain.rightingGain));
    this.brain.rightingAsym = Math.min(1.0, Math.max(0.85, this.brain.rightingAsym));
    this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();

    if (this.brain.morphologyKey === 'duck') {
      const cand = this.brain.getCurrentDuckGACandidate();
      if (cand && cand.vec) {
        this.brain.rememberDuckGAVector(cand.vec, 2.5, recoveredComY || 0.26);
      }
      const target = this.metaState.targetCompetence || 0.215;
      if (typeof recoveredComY === 'number') {
        this.metaState.competenceEMA = Number(Math.max(this.metaState.competenceEMA, recoveredComY).toFixed(3));
      }
      if (!this.alwaysLearn) {
        this.metaState.mode = 'CONSOLIDATED';
        this.metaState.isLearningActive = false;
        this.metaState.needToLearnPct = 0;
        this.metaState.modeTitle = '🛡️ ЗАДАЧА ВЫПОЛНЕНА: ВЫСОТА ЦМ УДЕРЖИВАЕТСЯ';
        this.metaState.reasonText = `Утка подняла центр масс выше ${target.toFixed(2)} м (подъёмов: ${this.metaState.rightingSuccesses}) — вектор сохранён в память ГА`;
      } else {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.modeTitle = '♾️ ВЕЧНОЕ ОБУЧЕНИЕ: МАКСИМИЗАЦИЯ ВЫСОТЫ ЦМ';
        this.metaState.reasonText = `Утка подняла центр масс (подъёмов: ${this.metaState.rightingSuccesses}) и продолжает максимизировать ЦМ`;
      }
      return;
    }

    // Restore previous walking mode (CONSOLIDATED or LEARNING_WALK)
    this.metaState.mode = this.modeBeforeOverturn || 'LEARNING_WALK';
    this.metaState.isLearningActive = this.metaState.mode === 'LEARNING_WALK';
    if (this.metaState.mode === 'CONSOLIDATED') {
      this.metaState.needToLearnPct = 0;
      this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
      this.metaState.reasonText = `Успешно поднялся на ноги (встал сам: ${this.metaState.rightingSuccesses})! Продолжает ходьбу с защищёнными весами`;
    } else {
      this.metaState.modeTitle = '🟢 АКТИВНОЕ ОБУЧЕНИЕ: ХОДЬБА К ЦЕЛИ';
      this.metaState.reasonText = `Успешно поднялся на ноги (встал сам: ${this.metaState.rightingSuccesses})! Продолжает обучение ходьбе к цели`;
    }
  }

  recordStepOutcome(creatures, rewards, dones) {
    if (!this._pendingStep) return null;

    const c0 = creatures[0];
    const isDuck0 = c0.morphologyKey === 'duck';
    const upY0 = c0.getTorsoFrame().up.y;
    const comY0 = c0.body.getCenterOfMass().y;
    const topY0 = c0.torsoTop ? c0.torsoTop.pos.y : comY0;

    if (this._pendingStep.isOverturnPause) {
      return null;
    }

    // Handle Self-Righting / Low-CoM Learning step:
    if (this._pendingStep.isRightingStep) {
      const targetCom0 = this.metaState.targetCompetence || 0.255;
      const canCompleteNow = isDuck0
        ? ((this.duckComHoldSteps || 0) >= 8 && comY0 >= targetCom0)
        : (upY0 >= 0.65 && !c0.hasInvertedLeg());
      if (this.wasOverturned && canCompleteNow) {
        this._completeRightingRecovery(comY0);
        return null;
      }

      if (!isDuck0 && this._pendingStep.isRestPause) {
        this.rightingPauseStepsRemaining = Math.max(0, (this.rightingPauseStepsRemaining || 0) - 1);
        if (this.rightingPauseStepsRemaining === 0) {
          this.rightingStartUpY = upY0;
          this.rightingMaxUpY = upY0;
          this.rightingStartTopY = topY0;
          this.rightingMaxTopY = topY0;
          this.rightingContactSteps = 0;
        }
        return null;
      }

      this.rightingStepCursor += 1;
      if (upY0 > this.rightingMaxUpY) this.rightingMaxUpY = upY0;
      if (topY0 > (this.rightingMaxTopY ?? topY0)) this.rightingMaxTopY = topY0;
      if (comY0 > (this.rightingMaxComY ?? comY0)) this.rightingMaxComY = comY0;
      this.brain.recordRightingStepTrace(
        this._pendingStep.obs,
        this._pendingStep.act,
        isDuck0 ? comY0 : upY0
      );
      const anyContact = c0.footNodes.some(f => f.inContact) ||
        (c0.webbedFeet ? c0.webbedFeet.some(wf => wf.toeInner.inContact || wf.toeOuter.inContact) : false);
      if (anyContact) this.rightingContactSteps += 1;
      if (c0.frameRefs && c0.frameRefs.bottom && c0.frameRefs.bottom.inContact) {
        this.rightingBottomContactSteps = (this.rightingBottomContactSteps || 0) + 1;
      }
      if (c0.headNode && c0.headNode.inContact) {
        this.rightingHeadContactSteps = (this.rightingHeadContactSteps || 0) + 1;
      }
      let stepMinFootClearance = Infinity;
      for (const f of c0.footNodes) {
        stepMinFootClearance = Math.min(stepMinFootClearance, Math.max(0, f.pos.y - f.radius));
      }
      if (c0.webbedFeet) {
        for (const wf of c0.webbedFeet) {
          stepMinFootClearance = Math.min(
            stepMinFootClearance,
            Math.max(0, wf.toeInner.pos.y - wf.toeInner.radius),
            Math.max(0, wf.toeOuter.pos.y - wf.toeOuter.radius)
          );
        }
      }
      if (stepMinFootClearance < (this.rightingMinFootClearanceY ?? Infinity)) {
        this.rightingMinFootClearanceY = stepMinFootClearance;
      }

      const horizon = isDuck0 ? 16 : this.rightingHorizon;
      if (this.rightingStepCursor >= horizon) {
        const startTopY = this.rightingStartTopY ?? topY0;
        const maxTopY = this.rightingMaxTopY ?? topY0;
        const startComY = this.rightingStartComY ?? comY0;
        const maxComY = this.rightingMaxComY ?? comY0;
        const skillPct = this.brain.updateRightingReflex({
          startComY,
          maxComY,
          endComY: comY0,
          startUpY: this.rightingStartUpY,
          maxUpY: this.rightingMaxUpY,
          endUpY: upY0,
          startTopY,
          maxTopY,
          endTopY: topY0,
          contactSteps: this.rightingContactSteps,
          lastObs: this._pendingStep.obs,
          lastAct: this._pendingStep.act
        });
        if (!this.rightingHistory) this.rightingHistory = [];
        this.rightingHistory.push({
          attempt: this.brain.rightingAttempts,
          startComY: Number(startComY.toFixed(3)),
          maxComY: Number(maxComY.toFixed(3)),
          endComY: Number(comY0.toFixed(3)),
          deltaComY: Number((comY0 - startComY).toFixed(3)),
          startUpY: Number(this.rightingStartUpY.toFixed(3)),
          maxUpY: Number(this.rightingMaxUpY.toFixed(3)),
          endUpY: Number(upY0.toFixed(3)),
          deltaUpY: Number((upY0 - this.rightingStartUpY).toFixed(3)),
          ...(this.brain.lastRightingWaveMetrics || {}),
          startTopY: Number(startTopY.toFixed(3)),
          maxTopY: Number(maxTopY.toFixed(3)),
          endTopY: Number(topY0.toFixed(3)),
          footContactSteps: this.rightingContactSteps,
          bottomContactSteps: this.rightingBottomContactSteps || 0,
          headContactSteps: this.rightingHeadContactSteps || 0,
          minFootClearanceY: Number(
            (Number.isFinite(this.rightingMinFootClearanceY) ? this.rightingMinFootClearanceY : 0).toFixed(3)
          ),
          skillPct
        });
        if (this.rightingHistory.length > 80) this.rightingHistory.shift();

        this.metaState.rightingAttempts = this.brain.rightingAttempts;
        this.metaState.rightingSkillPct = skillPct;
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingStartTopY = topY0;
        this.rightingMaxTopY = topY0;
        this.rightingStartComY = comY0;
        this.rightingMaxComY = comY0;
        this.rightingContactSteps = 0;
        this.rightingBottomContactSteps = 0;
        this.rightingHeadContactSteps = 0;
        this.rightingMinFootClearanceY = Infinity;
        // Only radial beetles pause; the Duck's neural network runs continuously without interruption!
        if (!isDuck0 && upY0 < 0.35) {
          this.rightingPauseStepsRemaining = this.rightingPauseDuration || 8;
        }
        this.updateCount += 1;
        return {
          updateCount: this.updateCount,
          didUpdateWeights: true,
          isRightingUpdate: true,
          policyLoss: this.lastPolicyLoss || 0,
          valueLoss: this.lastValueLoss || 0,
          entropy: this.lastEntropy || 0
        };
      }
      return null;
    }

    // If walker is held by user's mouse, in post-throw air flight, in the pause between food orbs,
    // or lying on its back after the terminal crash step (overturnConfirmSteps > 1),
    // do not push non-locomotion steps into the walking rollout buffer!
    if (
      c0.isGrabbed ||
      c0.postThrowTimer > 0 ||
      (!isDuck0 && upY0 < 0.22 && (this.overturnConfirmSteps || 0) > 1) ||
      (!isDuck0 && (c0.foodState === 'DISSOLVING' || c0.foodState === 'WAITING' || c0.foodState === 'DROPPING'))
    ) {
      this.metaState.stallStepsCount = 0;
      return null;
    }

    const { stepObs, stepAct, stepLogProb, stepVal } = this._pendingStep;

    this.epochVelSum = (this.epochVelSum || 0) + (c0.lastVelocityTowardFood || 0);
    this.epochStraightSteps = (this.epochStraightSteps || 0) + 1;
    this.metaState.stallStepsCount = 0;

    this.bufObs.push(stepObs);
    this.bufAct.push(stepAct);
    this.bufLogProb.push(stepLogProb);
    this.bufRew.push(Float64Array.from(rewards));
    this.bufVal.push(stepVal);
    this.bufDone.push(Uint8Array.from(dones));

    for (let w = 0; w < this.numWorkers; w++) {
      // Only add smooth high-velocity locomotion steps to eliteReplay (exclude 8.5 food-capture spike steps)
      if (rewards[w] > 0.35 && rewards[w] < 4.0) {
        this.eliteReplay.push({
          obs: stepObs[w],
          act: stepAct[w],
          rew: rewards[w]
        });
        if (this.eliteReplay.length > this.maxEliteSize) {
          this.eliteReplay.shift();
        }
      }
    }

    this.stepCursor += 1;
    if (this.stepCursor >= this.rolloutSteps) {
      return this.performPPOUpdate(creatures);
    }
    return null;
  }

  /**
   * Evaluates the brain's Homeostatic Meta-Plasticity Sense ("Чутьё нейросети")
   * at each rollout boundary to decide whether to continue training or consolidate/freeze weights.
   */
  _evaluateMetaPlasticity(creatures, epochLocomotionScore, meanTdSurprise) {
    const c0 = creatures[0];
    const isDuck0 = c0.morphologyKey === 'duck';
    const rolloutLen = Math.max(1, this.bufObs.length || this.rolloutSteps);

    if (isDuck0) {
      const meanRolloutComY = (this.epochVelSum || 0) / rolloutLen;
      const currentComY = c0.smoothedComY || c0.body.getCenterOfMass().y;
      const approxComY = 0.6 * currentComY + 0.4 * (meanRolloutComY > 0 ? meanRolloutComY : currentComY);
      this.metaState.competenceEMA = Number(
        (0.65 * this.metaState.competenceEMA + 0.35 * approxComY).toFixed(3)
      );
      this.metaState.surpriseEMA = Number(
        (0.75 * this.metaState.surpriseEMA + 0.25 * meanTdSurprise).toFixed(3)
      );
      const comp = this.metaState.competenceEMA;
      const target = this.metaState.targetCompetence || 0.215;

      if (this.minTrainEpochsRemaining > 0) {
        this.minTrainEpochsRemaining -= 1;
      }
      if (this.manualFreeze) {
        this.metaState.mode = 'CONSOLIDATED';
        this.metaState.isLearningActive = false;
        this.metaState.needToLearnPct = 0;
        return;
      }
      if (this.alwaysLearn) {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.needToLearnPct = 100;
        this.metaState.modeTitle = '♾️ ВЕЧНОЕ ОБУЧЕНИЕ: МАКСИМИЗАЦИЯ ВЫСОТЫ ЦМ';
        this.metaState.reasonText = `Непрерывная максимизация высоты центра масс (ЦМ=${comp.toFixed(2)} м)`;
        return;
      }
      if (comp >= target && !(this.minTrainEpochsRemaining > 0)) {
        this.metaState.mode = 'CONSOLIDATED';
        this.metaState.isLearningActive = false;
        this.metaState.stableEpochsCount += 1;
        this.metaState.needToLearnPct = 0;
        this.metaState.modeTitle = '🛡️ ЗАДАЧА ВЫПОЛНЕНА: ВЫСОТА ЦМ УДЕРЖИВАЕТСЯ';
        this.metaState.reasonText = `Утка справляется с задачей (высота ЦМ ${comp.toFixed(2)} м ≥ ${target.toFixed(2)} м) — обучение прекращено`;
      } else {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.needToLearnPct = Math.round(Math.max(25, Math.min(95, ((target - comp) / target) * 200 + 35)));
        this.metaState.modeTitle = '🟢 АКТИВНОЕ ОБУЧЕНИЕ: МАКСИМИЗАЦИЯ ВЫСОТЫ ЦМ';
        this.metaState.reasonText = `Утка стремится максимизировать высоту центра масс (ЦМ=${comp.toFixed(2)} м → ${target.toFixed(2)} м)`;
      }
      return;
    }

    // Compute effective goal-directed competence grounded in physical velocity toward food (m/s)
    const meanEpochRew = epochLocomotionScore / rolloutLen;
    const meanRolloutVel = (this.epochVelSum || 0) / rolloutLen;
    const recentVel = Math.max(0, c0.smoothedVelocityTowardFood || c0.lastVelocityTowardFood || 0);
    const physicalVel = Math.max(0, 0.55 * recentVel + 0.45 * Math.max(0, meanRolloutVel));
    const rewVelComponent = Math.max(0, (meanEpochRew - 0.18) / 2.5);
    // Cap reward-based proxy so posture reward cannot mask low physical speed on slippery ice
    const cappedRewVel = Math.min(rewVelComponent, physicalVel * 1.15 + 0.02);
    const approxEpochVel = Math.max(0.0, Math.min(0.65, 0.65 * physicalVel + 0.35 * cappedRewVel));

    const emaAlpha = approxEpochVel < this.metaState.competenceEMA ? 0.30 : 0.32;
    this.metaState.competenceEMA = Number(
      ((1 - emaAlpha) * this.metaState.competenceEMA + emaAlpha * approxEpochVel).toFixed(3)
    );
    this.metaState.surpriseEMA = Number(
      (0.75 * this.metaState.surpriseEMA + 0.25 * meanTdSurprise).toFixed(3)
    );

    const comp = this.metaState.competenceEMA;
    const target = this.metaState.targetCompetence;
    const foodCount = c0.foodCollected;
    const minFoods = this.minFoodsToConsolidate || 4;

    if (this.minTrainEpochsRemaining > 0) {
      this.minTrainEpochsRemaining -= 1;
    }

    if (this.manualFreeze) {
      this.metaState.mode = 'CONSOLIDATED';
      this.metaState.isLearningActive = false;
      this.metaState.needToLearnPct = 0;
      return;
    }

    if (this.alwaysLearn) {
      this.metaState.mode = 'LEARNING_WALK';
      this.metaState.isLearningActive = true;
      this.metaState.needToLearnPct = 100;
      this.metaState.modeTitle = '♾️ ВЕЧНОЕ ОБУЧЕНИЕ: МАКСИМАЛЬНЫЙ РАЗГОН';
      this.metaState.reasonText = `Непрерывное обучение включено (тек. эффективность ~${comp.toFixed(2)} м/с, целей: ${foodCount})`;
      return;
    }

    if (this.metaState.mode === 'LEARNING_WALK') {
      if (comp >= target && foodCount >= minFoods && !(this.minTrainEpochsRemaining > 0)) {
        this.metaState.stableEpochsCount += 1;
      } else if (!(this.minTrainEpochsRemaining > 0)) {
        this.metaState.stableEpochsCount = Math.max(0, this.metaState.stableEpochsCount - 1);
      }

      const rawNeed = Math.round(
        Math.max(
          this.minTrainEpochsRemaining > 0 ? 45 : 15,
          Math.min(98, ((target * 1.20 - comp) / (target * 1.20)) * 85 + (foodCount < minFoods ? 20 : 0))
        )
      );
      this.metaState.needToLearnPct = rawNeed;

      // Transition to CONSOLIDATED once the walker has collected >= minFoods, finished mandatory wake epochs, and maintained target velocity for 2 epochs
      if (this.metaState.stableEpochsCount >= 2 && foodCount >= minFoods && !(this.minTrainEpochsRemaining > 0)) {
        this.metaState.mode = 'CONSOLIDATED';
        this.metaState.isLearningActive = false;
        this.metaState.lowVelEpochsCount = 0;
        this.metaState.stallStepsCount = 0;
        this.metaState.needToLearnPct = 0;
        this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
        this.metaState.reasonText = `Режим освоен (${comp.toFixed(2)} м/с ≥ ${target.toFixed(2)} м/с, целей: ${foodCount}) — веса строго зафиксированы`;
      } else {
        this.metaState.isLearningActive = true;
        this.metaState.modeTitle = target >= 0.44
          ? '🟢 АКТИВНОЕ ОБУЧЕНИЕ: РАЗГОН ГАЛОПА К ЦЕЛИ'
          : '🟢 АКТИВНОЕ ОБУЧЕНИЕ: РАЗГОН СКАЧКА К ЦЕЛИ';
        this.metaState.reasonText = this.minTrainEpochsRemaining > 0
          ? `Активный разгон новой амбиции (${target.toFixed(2)} м/с): осталось мин. ${this.minTrainEpochsRemaining} эп. дообучения (тек. ~${comp.toFixed(2)} м/с)`
          : foodCount < minFoods
            ? `Разгон походки: собрано ${foodCount}/${minFoods} целей (эффективность ~${comp.toFixed(2)} / ${target.toFixed(2)} м/с)`
            : `Доводка скорости: ${comp.toFixed(2)} м/с → цель ${target.toFixed(2)} м/с`;
      }
    } else if (this.metaState.mode === 'CONSOLIDATED') {
      // Strict consolidation: once mastered, learning stays OFF permanently until the user explicitly clicks
      // the unfreeze/re-learn button or changes floor friction / learning ambition.
      this.metaState.lowVelEpochsCount = 0;
      this.metaState.stallStepsCount = 0;
      this.metaState.needToLearnPct = 0;
      this.metaState.isLearningActive = false;
      this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
      this.metaState.reasonText = `Навык строго зафиксирован (целей: ${foodCount}) — шагоход действует самостоятельно без изменения весов`;
    }
  }

  performPPOUpdate(creatures) {
    const T = this.bufObs.length;
    const W = this.numWorkers;
    const totalSamples = T * W;

    const nextValues = new Float64Array(W);
    for (let w = 0; w < W; w++) {
      const obsT = creatures[w].getObservation();
      nextValues[w] = this.brain.forwardCritic(obsT, false).value;
    }

    const advantages = Array.from({ length: T }, () => new Float64Array(W));
    const returns = Array.from({ length: T }, () => new Float64Array(W));

    let rewardSum = 0;
    let absDeltaSum = 0;
    for (let w = 0; w < W; w++) {
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const r = this.bufRew[t][w];
        rewardSum += r;
        const notDone = this.bufDone[t][w] ? 0.0 : 1.0;
        const vNext = t === T - 1 ? nextValues[w] : this.bufVal[t + 1][w];
        const delta = r + this.gamma * vNext * notDone - this.bufVal[t][w];
        absDeltaSum += Math.abs(delta);
        gae = delta + this.gamma * this.gaeLambda * notDone * gae;
        advantages[t][w] = gae;
        returns[t][w] = gae + this.bufVal[t][w];
      }
    }

    const workerLocomotionScore = new Float64Array(W);
    for (let w = 0; w < W; w++) {
      let score = 0;
      for (let t = 0; t < T; t++) {
        const r = this.bufRew[t][w];
        score += r > 5.0 ? r - 8.5 : r;
      }
      workerLocomotionScore[w] = score;
    }

    const meanTdSurprise = absDeltaSum / Math.max(1, totalSamples);
    const wasLearningActive = this.metaState.mode === 'LEARNING_WALK';

    // Update Homeostatic Meta-Plasticity Sense
    this._evaluateMetaPlasticity(creatures, workerLocomotionScore[0], meanTdSurprise);

    // If in CONSOLIDATED mode (learning paused by meta-sense), skip weight updates and do NOT increment epoch counter!
    if (!wasLearningActive && this.metaState.mode === 'CONSOLIDATED') {
      this.lastMeanStepReward = rewardSum / totalSamples;
      this._resetBuffers();
      return {
        updateCount: this.updateCount,
        didUpdateWeights: false,
        meanStepReward: this.lastMeanStepReward,
        policyLoss: 0,
        valueLoss: 0,
        entropy: this.lastEntropy,
        metaMode: this.metaState.mode,
        isLearningActive: false
      };
    }

    // Single-Walker (or Multi-Worker) Parameter-Space Natural Step on actorSkip.W:
    const skipW = this.brain.actorSkip.W;
    const tpl = this.brain.cpgTemplate;
    if (!this.bestSkipW) {
      this.bestSkipW = Float64Array.from(skipW);
      this.bestScore = 0.14;
    }
    const score0 = workerLocomotionScore[0] / T;
    if (W === 1) {
      const pert0 = this.workerSkipPerturbs[0];
      if (pert0) {
        if (score0 >= this.baselineLocomotionScore - 0.02) {
          const alpha = score0 > this.bestScore ? 0.42 : 0.20;
          for (let i = 0; i < skipW.length; i++) {
            skipW[i] += alpha * pert0[i];
          }
          if (score0 > this.bestScore) {
            this.bestScore = 0.65 * this.bestScore + 0.35 * score0;
            this.bestSkipW = Float64Array.from(skipW);
          }
        } else if (score0 < this.bestScore * 0.70) {
          // Restore toward best known CPG gait matrix + resonance template so a poor exploration rollout never degrades locomotion
          for (let i = 0; i < skipW.length; i++) {
            skipW[i] = 0.85 * this.bestSkipW[i] + 0.15 * skipW[i];
          }
        }
        // Ensure resonance CPG drive & steering gain ramp up energetically to maxCpgGain for bounding gallop!
        if (tpl && this.brain.morphologyKey !== 'duck') {
          const maxGain = this.maxCpgGain || 3.25;
          const targetGain = Math.min(maxGain, 1.35 + 0.16 * this.updateCount);
          for (let i = 0; i < skipW.length; i++) {
            if (Math.abs(tpl[i]) > 1e-4) {
              skipW[i] = 0.72 * skipW[i] + 0.28 * (tpl[i] * targetGain);
            }
          }
        }
        this.baselineLocomotionScore = 0.75 * this.baselineLocomotionScore + 0.25 * Math.max(0.08, score0);
      }
    } else {
      const rankedWorkers = Array.from({ length: W - 1 }, (_, i) => i + 1).sort(
        (a, b) => workerLocomotionScore[b] - workerLocomotionScore[a]
      );
      const bestWorker = rankedWorkers[0];
      if (bestWorker !== undefined && workerLocomotionScore[bestWorker] > workerLocomotionScore[0] + 0.6) {
        const topWeights = [0.22, 0.10, 0.05];
        for (let rank = 0; rank < Math.min(3, rankedWorkers.length); rank++) {
          const wIdx = rankedWorkers[rank];
          if (workerLocomotionScore[wIdx] > workerLocomotionScore[0]) {
            const pert = this.workerSkipPerturbs[wIdx];
            const alpha = topWeights[rank];
            for (let i = 0; i < skipW.length; i++) {
              skipW[i] += alpha * pert[i];
            }
          }
        }
      }
    }

    // Flatten dataset and normalize GAE advantages
    const flatObs = [];
    const flatAct = [];
    const flatOldLp = new Float64Array(totalSamples);
    const flatAdv = new Float64Array(totalSamples);
    const flatRet = new Float64Array(totalSamples);

    let idx = 0;
    let advMean = 0;
    for (let t = 0; t < T; t++) {
      for (let w = 0; w < W; w++) {
        flatObs.push(this.bufObs[t][w]);
        flatAct.push(this.bufAct[t][w]);
        flatOldLp[idx] = this.bufLogProb[t][w];
        flatAdv[idx] = advantages[t][w];
        flatRet[idx] = returns[t][w];
        advMean += advantages[t][w];
        idx++;
      }
    }
    advMean /= totalSamples;
    let advVar = 0;
    for (let i = 0; i < totalSamples; i++) {
      const d = flatAdv[i] - advMean;
      advVar += d * d;
    }
    const advStd = Math.sqrt(advVar / totalSamples) + 1e-6;
    const epochBonus = W === 1 ? Math.max(-0.2, Math.min(0.75, (score0 - 0.12) * 2.5)) : 0.0;
    for (let i = 0; i < totalSamples; i++) {
      flatAdv[i] = (flatAdv[i] - advMean) / advStd + epochBonus;
    }

    // Mini-batch PPO + Adam Backpropagation updates
    const batchSize = Math.min(32, totalSamples);
    const indices = Array.from({ length: totalSamples }, (_, i) => i);
    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    let numBatches = 0;

    for (let ep = 0; ep < this.ppoEpochs; ep++) {
      for (let i = totalSamples - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }

      for (let start = 0; start < totalSamples; start += batchSize) {
        const end = Math.min(totalSamples, start + batchSize);
        const count = end - start;
        const invCount = 1.0 / count;

        this.brain.zeroActorGrad();
        this.brain.zeroCriticGrad();

        for (let b = start; b < end; b++) {
          const k = indices[b];
          const obs = flatObs[k];
          const act = flatAct[k];
          const oldLp = flatOldLp[k];
          const adv = flatAdv[k];
          const targetRet = flatRet[k];

          const actorCache = this.brain.forwardActor(obs, false, null);
          const { mu } = actorCache;
          const newLp = this.computeLogProb(mu, act);
          const ratio = Math.exp(Math.max(-4, Math.min(4, newLp - oldLp)));

          const clippedRatio = Math.max(1 - this.clipEps, Math.min(1 + this.clipEps, ratio));
          const surr1 = ratio * adv;
          const surr2 = clippedRatio * adv;
          totalPolicyLoss += -Math.min(surr1, surr2) * invCount;

          const isUnclipped = (adv >= 0 && ratio < 1 + this.clipEps) || (adv < 0 && ratio > 1 - this.clipEps);
          if (isUnclipped) {
            const dLdMu = new Float64Array(this.brain.actDim);
            for (let a = 0; a < this.brain.actDim; a++) {
              const std = Math.exp(this.brain.logStd[a]);
              const varInv = 1.0 / (std * std);
              const diff = act[a] - mu[a];
              dLdMu[a] = -(ratio * adv) * diff * varInv * invCount;
            }
            this.brain.backwardActorFromMu(obs, actorCache, dLdMu);
          }

          const criticCache = this.brain.forwardCritic(obs, false);
          const vErr = criticCache.value - targetRet;
          totalValueLoss += 0.5 * vErr * vErr * invCount;
          const dLdV = Math.max(-2.0, Math.min(2.0, vErr)) * invCount;
          this.brain.backwardCritic(obs, criticCache, dLdV);
        }

        // Self-Imitation Learning (SIL) on top elite transitions
        const lrDecay = 1.0 / (1.0 + 0.04 * this.updateCount);
        if (this.eliteReplay.length >= 8) {
          const silSamples = 8;
          const silWeight = (0.14 * lrDecay) / silSamples;
          for (let s = 0; s < silSamples; s++) {
            const sample = this.eliteReplay[Math.floor(Math.random() * this.eliteReplay.length)];
            const cache = this.brain.forwardActor(sample.obs, false, null);
            const dLdMu = new Float64Array(this.brain.actDim);
            for (let a = 0; a < this.brain.actDim; a++) {
              dLdMu[a] = (cache.mu[a] - sample.act[a]) * silWeight;
            }
            this.brain.backwardActorFromMu(sample.obs, cache, dLdMu);
          }
        }

        this.brain.stepAdam(this.actorLr * lrDecay, this.criticLr);
        numBatches++;
      }
    }

    this.updateCount += 1;
    this.paramSigma = Math.max(0.035, this.paramSigma * 0.968);
    this._sampleWorkerParamNoise();

    this.lastPolicyLoss = totalPolicyLoss / Math.max(1, numBatches);
    this.lastValueLoss = totalValueLoss / Math.max(1, numBatches);
    this.lastMeanStepReward = rewardSum / totalSamples;

    let ent = 0;
    for (let a = 0; a < this.brain.actDim; a++) {
      ent += this.brain.logStd[a] + 0.5 * Math.log(2 * Math.PI * Math.E);
    }
    this.lastEntropy = ent / this.brain.actDim;

    this._resetBuffers();

    return {
      updateCount: this.updateCount,
      didUpdateWeights: true,
      meanStepReward: this.lastMeanStepReward,
      policyLoss: this.lastPolicyLoss,
      valueLoss: this.lastValueLoss,
      entropy: this.lastEntropy,
      metaMode: this.metaState.mode,
      isLearningActive: this.metaState.isLearningActive
    };
  }

  toJSON() {
    return {
      updateCount: this.updateCount,
      paramSigma: this.paramSigma,
      bestScore: this.bestScore,
      bestSkipW: this.bestSkipW ? Array.from(this.bestSkipW) : null,
      baselineLocomotionScore: this.baselineLocomotionScore,
      maxCpgGain: this.maxCpgGain,
      minFoodsToConsolidate: this.minFoodsToConsolidate,
      alwaysLearn: this.alwaysLearn,
      manualFreeze: this.manualFreeze,
      minTrainEpochsRemaining: this.minTrainEpochsRemaining || 0,
      wasOverturned: this.wasOverturned,
      modeBeforeOverturn: this.modeBeforeOverturn,
      metaState: { ...this.metaState },
      lastPolicyLoss: this.lastPolicyLoss,
      lastValueLoss: this.lastValueLoss,
      lastEntropy: this.lastEntropy,
      lastMeanStepReward: this.lastMeanStepReward,
      eliteReplay: this.eliteReplay.slice(-64).map(item => ({
        obs: Array.from(item.obs),
        act: Array.from(item.act),
        ret: item.ret
      }))
    };
  }

  fromJSON(data) {
    if (!data) return false;
    if (typeof data.updateCount === 'number') this.updateCount = data.updateCount;
    if (typeof data.paramSigma === 'number') this.paramSigma = data.paramSigma;
    if (typeof data.bestScore === 'number') this.bestScore = data.bestScore;
    if (Array.isArray(data.bestSkipW)) this.bestSkipW = Float64Array.from(data.bestSkipW);
    if (typeof data.baselineLocomotionScore === 'number') this.baselineLocomotionScore = data.baselineLocomotionScore;
    if (typeof data.maxCpgGain === 'number') this.maxCpgGain = data.maxCpgGain;
    if (typeof data.minFoodsToConsolidate === 'number') this.minFoodsToConsolidate = data.minFoodsToConsolidate;
    if (typeof data.alwaysLearn === 'boolean') this.alwaysLearn = data.alwaysLearn;
    if (typeof data.manualFreeze === 'boolean') this.manualFreeze = data.manualFreeze;
    if (typeof data.minTrainEpochsRemaining === 'number') this.minTrainEpochsRemaining = data.minTrainEpochsRemaining;
    if (typeof data.wasOverturned === 'boolean') this.wasOverturned = data.wasOverturned;
    if (typeof data.modeBeforeOverturn === 'string') this.modeBeforeOverturn = data.modeBeforeOverturn;
    if (data.metaState && typeof data.metaState === 'object') {
      this.metaState = { ...this.metaState, ...data.metaState };
      const minRelearn = Number(((this.metaState.targetCompetence || 0.18) * 0.68).toFixed(3));
      if (!this.metaState.relearnThreshold || this.metaState.relearnThreshold < minRelearn) {
        this.metaState.relearnThreshold = minRelearn;
      }
    }
    if (typeof data.lastPolicyLoss === 'number') this.lastPolicyLoss = data.lastPolicyLoss;
    if (typeof data.lastValueLoss === 'number') this.lastValueLoss = data.lastValueLoss;
    if (typeof data.lastEntropy === 'number') this.lastEntropy = data.lastEntropy;
    if (typeof data.lastMeanStepReward === 'number') this.lastMeanStepReward = data.lastMeanStepReward;
    if (Array.isArray(data.eliteReplay)) {
      this.eliteReplay = data.eliteReplay.map(item => ({
        obs: Float64Array.from(item.obs),
        act: Float64Array.from(item.act),
        ret: item.ret
      }));
    }
    this._resetBuffers();
    this._sampleWorkerParamNoise();
    return true;
  }
}

/**
 * Natural Evolution Strategies / SEP-CMA-ES Trainer (Algorithm #2 for comparison, supports numWorkers = 1)
 */
export class NESTrainer {
  constructor(brain, numWorkers = 1, {
    horizonSteps = 32,
    sigma = 0.08,
    lr = 0.025
  } = {}) {
    this.brain = brain;
    this.numWorkers = numWorkers;
    this.horizonSteps = horizonSteps;
    this.sigma = sigma;
    this.lr = lr;

    this.centerParams = brain.getActorFlatParams();
    this.paramDim = this.centerParams.length;
    this.adamM = new Float64Array(this.paramDim);
    this.adamV = new Float64Array(this.paramDim);

    this.workerPerturbations = [];
    this.workerReturns = new Float64Array(numWorkers);
    this.baselineReturn = 0.10;
    this.stepCursor = 0;
    this.epochVelSum = 0;
    this.updateCount = 0;
    this.lastMeanStepReward = 0;

    this.rightingStepCursor = 0;
    this.rightingHorizon = 14;
    this.rightingStartUpY = -1.0;
    this.rightingMaxUpY = -1.0;
    this.rightingContactSteps = 0;
    this.wasOverturned = false;
    this.modeBeforeOverturn = 'LEARNING_WALK';

    const isDuckBrain = brain.morphologyKey === 'duck';
    this.metaState = {
      mode: 'LEARNING_WALK',
      isLearningActive: true,
      modeTitle: isDuckBrain
        ? '🟢 АКТИВНОЕ ОБУЧЕНИЕ (NES): МАКСИМИЗАЦИЯ ВЫСОТЫ ЦМ'
        : '🟢 АКТИВНОЕ ОБУЧЕНИЕ (NES): ХОДЬБА К ЦЕЛИ',
      reasonText: isDuckBrain
        ? 'Единая задача: максимизация высоты центра масс (ЦМ)'
        : 'Эволюционный поиск моторного навыка шага и руления',
      competenceEMA: isDuckBrain ? 0.265 : 0.03,
      targetCompetence: isDuckBrain ? 0.255 : 0.18,
      relearnThreshold: isDuckBrain ? 0.238 : 0.125,
      ambitionPreset: 'min',
      surpriseEMA: 0.20,
      needToLearnPct: isDuckBrain ? 35 : 85,
      stableEpochsCount: 0,
      lowVelEpochsCount: 0,
      stallStepsCount: 0,
      rightingAttempts: 0,
      rightingSuccesses: 0,
      rightingSkillPct: brain.getRightingSkillPct()
    };
    this.alwaysLearn = false;
    this.manualFreeze = false;
    this.minFoodsToConsolidate = isDuckBrain ? 0 : 2;

    this._sampleWorkerPopulation();
  }

  setLearningAmbition(presetOrTarget) {
    this.manualFreeze = false;
    const isDuckBrain = this.brain.morphologyKey === 'duck';
    if (isDuckBrain) {
      this.minFoodsToConsolidate = 0;
      if (presetOrTarget === 'min') {
        this.alwaysLearn = false;
        this.metaState.ambitionPreset = 'min';
        this.metaState.targetCompetence = 0.255;
        this.metaState.relearnThreshold = 0.238;
      } else if (presetOrTarget === 'bound') {
        this.alwaysLearn = false;
        this.metaState.ambitionPreset = 'bound';
        this.metaState.targetCompetence = 0.268;
        this.metaState.relearnThreshold = 0.245;
      } else if (presetOrTarget === 'turbo') {
        this.alwaysLearn = false;
        this.metaState.ambitionPreset = 'turbo';
        this.metaState.targetCompetence = 0.280;
        this.metaState.relearnThreshold = 0.252;
      } else if (presetOrTarget === 'infinite') {
        this.alwaysLearn = true;
        this.metaState.ambitionPreset = 'infinite';
        this.metaState.targetCompetence = 0.360;
        this.metaState.relearnThreshold = 0.280;
      } else if (typeof presetOrTarget === 'number') {
        const t = Math.max(0.22, Math.min(0.36, presetOrTarget));
        this.alwaysLearn = t >= 0.355;
        this.metaState.ambitionPreset = 'custom';
        this.metaState.targetCompetence = Number(t.toFixed(3));
        this.metaState.relearnThreshold = Number((t - 0.030).toFixed(3));
      }
      if (this.metaState.mode !== 'LEARNING_RIGHTING') {
        if (this.alwaysLearn || this.metaState.competenceEMA < this.metaState.targetCompetence) {
          this.metaState.mode = 'LEARNING_WALK';
          this.metaState.isLearningActive = true;
          this.metaState.stableEpochsCount = 0;
          this.metaState.needToLearnPct = this.alwaysLearn ? 100 : 65;
          this.metaState.modeTitle = '🟢 АКТИВНОЕ ОБУЧЕНИЕ (NES): ПОДЪЁМ ЦМ';
          this.metaState.reasonText = `Утка стремится поднять и удерживать ЦМ ≥ ${this.metaState.targetCompetence.toFixed(2)} м`;
        } else {
          this.metaState.mode = 'CONSOLIDATED';
          this.metaState.isLearningActive = false;
          this.metaState.needToLearnPct = 0;
          this.metaState.modeTitle = '🛡️ ЗАДАЧА ВЫПОЛНЕНА: ВЫСОТА ЦМ УДЕРЖИВАЕТСЯ';
          this.metaState.reasonText = `Высота ЦМ (${this.metaState.competenceEMA.toFixed(2)} м) ≥ ${this.metaState.targetCompetence.toFixed(2)} м — обучение остановлено`;
        }
      }
      return;
    }

    const prevTarget = this.metaState.targetCompetence || 0.18;
    if (presetOrTarget === 'min') {
      this.alwaysLearn = false;
      this.metaState.ambitionPreset = 'min';
      this.metaState.targetCompetence = 0.18;
      this.metaState.relearnThreshold = 0.125;
      this.minFoodsToConsolidate = 2;
    } else if (presetOrTarget === 'bound') {
      this.alwaysLearn = false;
      this.metaState.ambitionPreset = 'bound';
      this.metaState.targetCompetence = 0.34;
      this.metaState.relearnThreshold = 0.23;
      this.minFoodsToConsolidate = 4;
    } else if (presetOrTarget === 'turbo') {
      this.alwaysLearn = false;
      this.metaState.ambitionPreset = 'turbo';
      this.metaState.targetCompetence = 0.46;
      this.metaState.relearnThreshold = 0.31;
      this.minFoodsToConsolidate = 6;
    } else if (presetOrTarget === 'infinite') {
      this.alwaysLearn = true;
      this.metaState.ambitionPreset = 'infinite';
      this.metaState.targetCompetence = 0.60;
      this.metaState.relearnThreshold = 0.45;
      this.minFoodsToConsolidate = 9999;
    } else if (typeof presetOrTarget === 'number') {
      this.alwaysLearn = presetOrTarget >= 0.54;
      this.metaState.ambitionPreset = 'custom';
      this.metaState.targetCompetence = Number(presetOrTarget.toFixed(2));
      this.metaState.relearnThreshold = Number(Math.max(0.10, presetOrTarget * 0.68).toFixed(2));
      this.minFoodsToConsolidate = presetOrTarget > 0.38 ? 5 : presetOrTarget > 0.24 ? 3 : 2;
    }
    const raisedAmbition = this.metaState.targetCompetence > prevTarget + 0.005 || presetOrTarget !== 'min';
    if (raisedAmbition) {
      this.metaState.competenceEMA = Number(
        Math.min(this.metaState.competenceEMA, this.metaState.targetCompetence * 0.65).toFixed(3)
      );
    }
    if (this.alwaysLearn || raisedAmbition || this.metaState.competenceEMA < this.metaState.targetCompetence) {
      if (this.metaState.mode !== 'LEARNING_RIGHTING') {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.stableEpochsCount = 0;
        this.metaState.needToLearnPct = 82;
        this.metaState.modeTitle = '🟢 АКТИВНОЕ ДООБУЧЕНИЕ: РАЗГОН ПОХОДКИ';
        this.metaState.reasonText = this.alwaysLearn
          ? 'Включён режим непрерывного обучения (100% пластичность без остановки)'
          : `Планка мастерства повышена до ${this.metaState.targetCompetence.toFixed(2)} м/с — шагоход доучивает мощный прыжок!`;
      }
    }
  }

  toggleManualFreeze() {
    if (this.metaState.mode === 'CONSOLIDATED') {
      this.manualFreeze = false;
      this.metaState.mode = 'LEARNING_WALK';
      this.metaState.isLearningActive = true;
      this.metaState.stableEpochsCount = 0;
      this.metaState.needToLearnPct = 85;
      this.metaState.modeTitle = '🟢 РУЧНОЕ ПРОБУЖДЕНИЕ: ДООБУЧЕНИЕ ВКЛЮЧЕНО';
      this.metaState.reasonText = 'Пользователь вручную запустил дообучение';
    } else {
      this.manualFreeze = true;
      this.alwaysLearn = false;
      this.metaState.mode = 'CONSOLIDATED';
      this.metaState.isLearningActive = false;
      this.metaState.needToLearnPct = 0;
      this.metaState.modeTitle = '🛡️ РУЧНАЯ КОНСОЛИДАЦИЯ: ОБУЧЕНИЕ ОСТАНОВЛЕНО';
      this.metaState.reasonText = 'Пользователь зафиксировал текущие веса нейросети вручную';
    }
  }

  notifyEnvironmentChange(reason = 'Изменились параметры среды — проверка адаптации', nextMu = null) {
    this.metaState.surpriseEMA = Math.max(this.metaState.surpriseEMA || 0.2, 0.9);
    if (this.manualFreeze || this.metaState.mode === 'LEARNING_RIGHTING') {
      return;
    }
    const isSlipperyOrChanged = typeof nextMu === 'number' ? nextMu < 0.78 : true;
    if (this.metaState.mode === 'CONSOLIDATED' || isSlipperyOrChanged) {
      this.metaState.mode = 'LEARNING_WALK';
      this.metaState.isLearningActive = true;
      this.metaState.stableEpochsCount = 0;
      this.metaState.lowVelEpochsCount = 0;
      this.metaState.stallStepsCount = 0;
      this.metaState.competenceEMA = Number(
        Math.min(this.metaState.competenceEMA, (this.metaState.targetCompetence || 0.18) * 0.58).toFixed(3)
      );
      this.metaState.needToLearnPct = Math.max(this.metaState.needToLearnPct || 0, 78);
      this.metaState.modeTitle = '🟢 АКТИВНОЕ ДООБУЧЕНИЕ (NES): АДАПТАЦИЯ К СРЕДЕ';
      this.metaState.reasonText = reason;
    }
  }

  _sampleWorkerPopulation() {
    this.workerPerturbations = [];
    this.workerReturns.fill(0);
    for (let w = 0; w < this.numWorkers; w++) {
      const eps = new Float64Array(this.paramDim);
      for (let i = 0; i < this.paramDim; i++) eps[i] = randn();
      this.workerPerturbations.push(eps);
    }
  }

  selectActions(creatures) {
    const c0 = creatures[0];
    const isDuck0 = c0.morphologyKey === 'duck';
    const upY0 = c0.getTorsoFrame().up.y;
    const comY0 = c0.body.getCenterOfMass().y;
    const topY0 = c0.torsoTop ? c0.torsoTop.pos.y : comY0;
    const isPhysicallyTiltedOnBack =
      !c0.isGrabbed &&
      !(c0.postThrowTimer > 0) &&
      (isDuck0
        ? (((comY0 < (this.alwaysLearn ? (this.metaState.relearnThreshold || 0.238) : 0.205)) || upY0 < 0.45) && !this.manualFreeze)
        : (upY0 < 0.22));
    const confirmDelay = isDuck0 ? 1 : 24;

    if (!this.wasOverturned) {
      if (isPhysicallyTiltedOnBack) {
        this.overturnConfirmSteps = (this.overturnConfirmSteps || 0) + 1;
        if (!isDuck0 && this.overturnConfirmSteps < confirmDelay && this.overturnConfirmSteps >= 6) {
          const obs = c0.getObservation();
          const calmAct = new Float64Array(this.brain.actDim);
          c0.lastState = obs;
          c0.lastAction = calmAct;
          c0.applyActionVector(calmAct, false);
          this._lastRightingStep = { isOverturnPause: true };
          return;
        }
      } else {
        this.overturnConfirmSteps = 0;
      }
    }

    this.uprightHoldSteps = 0;
    const targetCom0 = this.metaState.targetCompetence || 0.255;
    if (isDuck0) {
      const isTrueHighCom = comY0 >= targetCom0 && (!c0.torsoTop || !c0.torsoBottom || c0.torsoTop.pos.y > c0.torsoBottom.pos.y + 0.15);
      this.duckComHoldSteps = isTrueHighCom ? (this.duckComHoldSteps || 0) + 1 : 0;
    }
    const isRecoveredUpright = isDuck0
      ? (this.duckComHoldSteps >= 8)
      : (upY0 >= 0.65 && !c0.hasInvertedLeg());

    const isOverturned0 =
      !c0.isGrabbed &&
      !(c0.postThrowTimer > 0) &&
      ((isPhysicallyTiltedOnBack && this.overturnConfirmSteps >= confirmDelay) || (this.wasOverturned && !isRecoveredUpright));
    if (isOverturned0) {
      if (!this.wasOverturned) {
        this.wasOverturned = true;
        this.uprightHoldSteps = 0;
        this.rightingPauseStepsRemaining = 0;
        this.modeBeforeOverturn = this.metaState.mode === 'LEARNING_RIGHTING' ? 'LEARNING_WALK' : this.metaState.mode;
        this.stepCursor = 0;
        this.workerReturns.fill(0);
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingStartTopY = topY0;
        this.rightingMaxTopY = topY0;
        this.rightingStartComY = comY0;
        this.rightingMaxComY = comY0;
        this.rightingContactSteps = 0;
        this.brain.rightingPhase = 0;
        this.brain._duckInHighHeadstand = false;
        this.brain.resetRightingWaveState(isDuck0 ? comY0 : upY0, topY0);
        if (isDuck0) {
          this.brain._initPoseGAPopulation(this.brain.extractDuckPose6D(c0.getObservation(), comY0));
        }
      }
      this.metaState.mode = 'LEARNING_RIGHTING';
      this.metaState.isLearningActive = true;
      this.metaState.needToLearnPct = 100;
      this.metaState.rightingAttempts = this.brain.rightingAttempts + 1;
      this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();
      this.metaState.modeTitle = isDuck0
        ? '🧬 ДВУХУРОВНЕВЫЙ ГА: ЯЗЫК ПОЗ (ГА №1 Поз + ГА №2 Серво)'
        : '🔄 ПОПЫТКА ПОДНЯТЬСЯ: УЧИТСЯ ВСТАВАТЬ НА НОГИ';

      const obs = c0.getObservation();
      if (!isDuck0 && this.rightingPauseStepsRemaining > 0) {
        const calmAct = new Float64Array(this.brain.actDim);
        c0.lastState = obs;
        c0.lastAction = calmAct;
        c0.applyActionVector(calmAct, false);
        this.metaState.reasonText = `Передышка на спине перед попыткой #${this.metaState.rightingAttempts}`;
        this._lastRightingStep = { isRestPause: true, obs, act: calmAct };
        return;
      }

      if (isDuck0) {
        const cand = this.brain.getCurrentDuckGACandidate();
        this.metaState.reasonText = `ГА1 Поза #${cand.poseGen}.${cand.poseIdx} (${cand.poseOrigin}, к цели ${cand.curGoalPct}%) | ГА2 Вектор #${cand.idx}/${cand.popSize} (${cand.origin}) | память: ${cand.memSize} перех. | ЦМ=${comY0.toFixed(3)} м`;
      } else {
        this.metaState.reasonText = `Корпус перевёрнут/наклонён (up.y=${upY0.toFixed(2)}): память ходьбы защищена — попытка подняться #${this.metaState.rightingAttempts}`;
      }

      const muRight = this.brain.forwardRightingReflex(obs, 0.05, true, comY0);
      c0.lastState = obs;
      c0.lastAction = muRight;
      c0.applyActionVector(muRight, isDuck0 ? (comY0 < 0.245 || upY0 < 0.58) : true);
      this._lastRightingStep = { obs, act: muRight, comY: comY0 };
      return;
    }

    if (this.wasOverturned && isRecoveredUpright) {
      this.wasOverturned = false;
      this.uprightHoldSteps = 0;
      this.overturnConfirmSteps = 0;
      this.rightingPauseStepsRemaining = 0;
      this.brain.rightingSuccesses += 1;
      this.metaState.rightingSuccesses = this.brain.rightingSuccesses;
      this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();
      if (isDuck0 && !this.alwaysLearn) {
        this.metaState.mode = 'CONSOLIDATED';
        this.metaState.isLearningActive = false;
        this.metaState.needToLearnPct = 0;
        this.metaState.competenceEMA = Number(Math.max(this.metaState.competenceEMA, comY0).toFixed(3));
        this.metaState.modeTitle = '🛡️ ЗАДАЧА ВЫПОЛНЕНА: ВЫСОТА ЦМ УДЕРЖИВАЕТСЯ';
        this.metaState.reasonText = `Утка подняла центр масс выше ${targetCom0.toFixed(2)} м — обучение остановлено`;
      } else {
        this.metaState.mode = this.modeBeforeOverturn || 'LEARNING_WALK';
        this.metaState.isLearningActive = this.metaState.mode === 'LEARNING_WALK';
      }
    }
    this._lastRightingStep = null;

    const isTrainingWalk = this.metaState.mode === 'LEARNING_WALK';
    const tempParams = new Float64Array(this.paramDim);
    for (let w = 0; w < this.numWorkers; w++) {
      const eps = this.workerPerturbations[w];
      const sig = isTrainingWalk ? this.sigma : 0.0;
      for (let i = 0; i < this.paramDim; i++) {
        tempParams[i] = this.centerParams[i] + sig * eps[i];
      }
      this.brain.setActorFlatParams(tempParams);
      const obs = creatures[w].getObservation();
      const { mu } = this.brain.forwardActor(obs, w === 0, null);
      creatures[w].lastState = obs;
      creatures[w].lastAction = mu;
      creatures[w].applyActionVector(mu, false);
    }
    this.brain.setActorFlatParams(this.centerParams);
  }

  recordStepOutcome(creatures, rewards, _dones) {
    const c0 = creatures[0];
    const isDuck0 = c0.morphologyKey === 'duck';
    const upY0 = c0.getTorsoFrame().up.y;
    const comY0 = c0.body.getCenterOfMass().y;
    const topY0 = c0.torsoTop ? c0.torsoTop.pos.y : comY0;
    if (this._lastRightingStep) {
      if (this._lastRightingStep.isOverturnPause) {
        return null;
      }
      if (this._lastRightingStep.isRestPause) {
        this.rightingPauseStepsRemaining = Math.max(0, (this.rightingPauseStepsRemaining || 0) - 1);
        if (this.rightingPauseStepsRemaining === 0) {
          this.rightingStartUpY = upY0;
          this.rightingMaxUpY = upY0;
          this.rightingStartTopY = topY0;
          this.rightingMaxTopY = topY0;
          this.rightingContactSteps = 0;
        }
        return null;
      }
      this.rightingStepCursor += 1;
      if (upY0 > this.rightingMaxUpY) this.rightingMaxUpY = upY0;
      if (topY0 > (this.rightingMaxTopY ?? topY0)) this.rightingMaxTopY = topY0;
      if (comY0 > (this.rightingMaxComY ?? comY0)) this.rightingMaxComY = comY0;
      this.brain.recordRightingStepTrace(
        this._lastRightingStep.obs,
        this._lastRightingStep.act,
        isDuck0 ? comY0 : upY0
      );
      const anyContact = c0.footNodes.some(f => f.inContact) ||
        (c0.webbedFeet ? c0.webbedFeet.some(wf => wf.toeInner.inContact || wf.toeOuter.inContact) : false);
      if (anyContact) this.rightingContactSteps += 1;
      if (c0.frameRefs && c0.frameRefs.bottom && c0.frameRefs.bottom.inContact) {
        this.rightingBottomContactSteps = (this.rightingBottomContactSteps || 0) + 1;
      }
      if (c0.headNode && c0.headNode.inContact) {
        this.rightingHeadContactSteps = (this.rightingHeadContactSteps || 0) + 1;
      }
      let stepMinFootClearance = Infinity;
      for (const f of c0.footNodes) {
        stepMinFootClearance = Math.min(stepMinFootClearance, Math.max(0, f.pos.y - f.radius));
      }
      if (c0.webbedFeet) {
        for (const wf of c0.webbedFeet) {
          stepMinFootClearance = Math.min(
            stepMinFootClearance,
            Math.max(0, wf.toeInner.pos.y - wf.toeInner.radius),
            Math.max(0, wf.toeOuter.pos.y - wf.toeOuter.radius)
          );
        }
      }
      if (stepMinFootClearance < (this.rightingMinFootClearanceY ?? Infinity)) {
        this.rightingMinFootClearanceY = stepMinFootClearance;
      }
      const horizon = isDuck0 ? 16 : this.rightingHorizon;
      if (this.rightingStepCursor >= horizon) {
        const startTopY = this.rightingStartTopY ?? topY0;
        const maxTopY = this.rightingMaxTopY ?? topY0;
        const startComY = this.rightingStartComY ?? comY0;
        const maxComY = this.rightingMaxComY ?? comY0;
        const skillPct = this.brain.updateRightingReflex({
          startComY,
          maxComY,
          endComY: comY0,
          startUpY: this.rightingStartUpY,
          maxUpY: this.rightingMaxUpY,
          endUpY: upY0,
          startTopY,
          maxTopY,
          endTopY: topY0,
          contactSteps: this.rightingContactSteps,
          lastObs: this._lastRightingStep.obs,
          lastAct: this._lastRightingStep.act
        });
        this.metaState.rightingSkillPct = skillPct;
        if (!this.rightingHistory) this.rightingHistory = [];
        this.rightingHistory.push({
          attempt: this.brain.rightingAttempts,
          startComY: Number(startComY.toFixed(3)),
          maxComY: Number(maxComY.toFixed(3)),
          endComY: Number(comY0.toFixed(3)),
          deltaComY: Number((comY0 - startComY).toFixed(3)),
          startUpY: Number(this.rightingStartUpY.toFixed(3)),
          maxUpY: Number(this.rightingMaxUpY.toFixed(3)),
          endUpY: Number(upY0.toFixed(3)),
          deltaUpY: Number((upY0 - this.rightingStartUpY).toFixed(3)),
          ...(this.brain.lastRightingWaveMetrics || {}),
          startTopY: Number(startTopY.toFixed(3)),
          maxTopY: Number(maxTopY.toFixed(3)),
          endTopY: Number(topY0.toFixed(3)),
          footContactSteps: this.rightingContactSteps,
          bottomContactSteps: this.rightingBottomContactSteps || 0,
          headContactSteps: this.rightingHeadContactSteps || 0,
          minFootClearanceY: Number(
            (Number.isFinite(this.rightingMinFootClearanceY) ? this.rightingMinFootClearanceY : 0).toFixed(3)
          ),
          skillPct
        });
        if (this.rightingHistory.length > 80) this.rightingHistory.shift();

        this.metaState.rightingAttempts = this.brain.rightingAttempts;
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingStartTopY = topY0;
        this.rightingMaxTopY = topY0;
        this.rightingStartComY = comY0;
        this.rightingMaxComY = comY0;
        this.rightingContactSteps = 0;
        this.rightingBottomContactSteps = 0;
        this.rightingHeadContactSteps = 0;
        this.rightingMinFootClearanceY = Infinity;
        if (!isDuck0 && upY0 < 0.35) {
          this.rightingPauseStepsRemaining = 14;
        }
        this.updateCount += 1;
        return {
          updateCount: this.updateCount,
          didUpdateWeights: true,
          isRightingUpdate: true,
          policyLoss: 0,
          valueLoss: 0,
          entropy: 0
        };
      }
      return null;
    }

    if (
      (!isDuck0 && upY0 < 0.35) ||
      (!isDuck0 && (c0.foodState === 'DISSOLVING' || c0.foodState === 'WAITING' || c0.foodState === 'DROPPING'))
    ) {
      this.metaState.stallStepsCount = 0;
      return null;
    }

    for (let w = 0; w < this.numWorkers; w++) {
      const r = rewards[w];
      this.workerReturns[w] += r > 5.0 ? r - 8.5 : r;
    }
    const isPausedOrTurning = Boolean(c0.isPausedOrTurningToFood);
    const smoothVel = c0.smoothedVelocityTowardFood !== undefined
      ? c0.smoothedVelocityTowardFood
      : (c0.lastVelocityTowardFood || 0);
    if (!isPausedOrTurning) {
      this.epochVelSum = (this.epochVelSum || 0) + (c0.lastVelocityTowardFood || 0);
    }
    this.metaState.stallStepsCount = 0;

    this.stepCursor += 1;
    if (this.stepCursor < this.horizonSteps) return null;

    const meanStepRew = this.workerReturns[0] / this.horizonSteps;
    const meanRolloutVel = (this.epochVelSum || 0) / Math.max(1, this.horizonSteps);
    this.epochVelSum = 0;
    this.lastMeanStepReward = meanStepRew;
    const approxVel = isDuck0
      ? (c0.smoothedComY || comY0)
      : Math.max(0, 0.55 * smoothVel + 0.45 * Math.max(0, meanRolloutVel));
    if (this.metaState.mode !== 'CONSOLIDATED' || approxVel > this.metaState.competenceEMA) {
      const emaAlpha = approxVel < this.metaState.competenceEMA ? 0.45 : 0.25;
      this.metaState.competenceEMA = Number(
        ((1 - emaAlpha) * this.metaState.competenceEMA + emaAlpha * approxVel).toFixed(3)
      );
    }

    if (this.manualFreeze) {
      this.metaState.mode = 'CONSOLIDATED';
      this.metaState.isLearningActive = false;
      this.metaState.needToLearnPct = 0;
    } else if (this.alwaysLearn) {
      this.metaState.mode = 'LEARNING_WALK';
      this.metaState.isLearningActive = true;
      this.metaState.needToLearnPct = 100;
    } else if (this.metaState.mode === 'LEARNING_WALK') {
      if (isDuck0) {
        if (this.metaState.competenceEMA >= (this.metaState.targetCompetence || 0.215)) {
          this.metaState.mode = 'CONSOLIDATED';
          this.metaState.isLearningActive = false;
          this.metaState.needToLearnPct = 0;
          this.metaState.modeTitle = '🛡️ ЗАДАЧА ВЫПОЛНЕНА: ВЫСОТА ЦМ УДЕРЖИВАЕТСЯ';
          this.metaState.reasonText = `Высота ЦМ (${this.metaState.competenceEMA.toFixed(2)} м) стабильна — обучение прекращено`;
        }
      } else if (this.metaState.competenceEMA >= this.metaState.targetCompetence && c0.foodCollected >= (this.minFoodsToConsolidate || 4)) {
        this.metaState.stableEpochsCount += 1;
        if (this.metaState.stableEpochsCount >= 2) {
          this.metaState.mode = 'CONSOLIDATED';
          this.metaState.isLearningActive = false;
          this.metaState.needToLearnPct = 0;
          this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
          this.metaState.reasonText = `Скорость (${this.metaState.competenceEMA.toFixed(2)} м/с) стабильна — веса строго зафиксированы`;
        }
      }
    } else if (this.metaState.mode === 'CONSOLIDATED') {
      this.metaState.isLearningActive = false;
      this.metaState.needToLearnPct = 0;
    }

    const didUpdateWeights = Boolean(this.metaState.isLearningActive);
    if (didUpdateWeights) {
      this.updateCount += 1;
      const adv = meanStepRew - this.baselineReturn;
      this.baselineReturn = 0.8 * this.baselineReturn + 0.2 * meanStepRew;
      const stepScale = Math.max(-1.0, Math.min(1.0, adv * 2.5)) * this.lr;
      const eps = this.workerPerturbations[0];
      for (let i = 0; i < this.paramDim; i++) {
        this.centerParams[i] = Math.max(-1.1, Math.min(1.1, this.centerParams[i] + stepScale * eps[i]));
      }
      this.brain.setActorFlatParams(this.centerParams);
    }

    this.stepCursor = 0;
    this._sampleWorkerPopulation();

    return {
      updateCount: this.updateCount,
      didUpdateWeights,
      meanStepReward: this.lastMeanStepReward,
      policyLoss: 0,
      valueLoss: 0,
      entropy: Math.log(this.sigma),
      metaMode: this.metaState.mode,
      isLearningActive: this.metaState.isLearningActive
    };
  }

  toJSON() {
    return {
      updateCount: this.updateCount,
      sigma: this.sigma,
      centerParams: Array.from(this.centerParams),
      baselineReturn: this.baselineReturn,
      alwaysLearn: this.alwaysLearn,
      manualFreeze: this.manualFreeze,
      minFoodsToConsolidate: this.minFoodsToConsolidate,
      metaState: { ...this.metaState },
      lastMeanStepReward: this.lastMeanStepReward
    };
  }

  fromJSON(data) {
    if (!data) return false;
    if (typeof data.updateCount === 'number') this.updateCount = data.updateCount;
    if (typeof data.sigma === 'number') this.sigma = data.sigma;
    if (Array.isArray(data.centerParams) && data.centerParams.length === this.centerParams.length) {
      this.centerParams.set(data.centerParams);
      this.brain.setActorFlatParams(this.centerParams);
    }
    if (typeof data.baselineReturn === 'number') this.baselineReturn = data.baselineReturn;
    if (typeof data.alwaysLearn === 'boolean') this.alwaysLearn = data.alwaysLearn;
    if (typeof data.manualFreeze === 'boolean') this.manualFreeze = data.manualFreeze;
    if (typeof data.minFoodsToConsolidate === 'number') this.minFoodsToConsolidate = data.minFoodsToConsolidate;
    if (data.metaState && typeof data.metaState === 'object') {
      this.metaState = { ...this.metaState, ...data.metaState };
      const minRelearn = Number(((this.metaState.targetCompetence || 0.18) * 0.68).toFixed(3));
      if (!this.metaState.relearnThreshold || this.metaState.relearnThreshold < minRelearn) {
        this.metaState.relearnThreshold = minRelearn;
      }
    }
    if (typeof data.lastMeanStepReward === 'number') this.lastMeanStepReward = data.lastMeanStepReward;
    this.stepCursor = 0;
    this.epochVelSum = 0;
    this._sampleWorkerPopulation();
    return true;
  }
}
