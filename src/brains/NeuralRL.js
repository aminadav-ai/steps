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
}

/**
 * Actor-Critic Neural Network with Residual CPG/Sensor-Motor Skip Connection
 * Architecture:
 *   Actor:  s -> Dense(32) -> tanh -> Dense(24) -> tanh -> Dense(actDim)
 *           + SkipLinear(s -> actDim)  ==>  tanh(z) = mean action \mu(s) in (-1, 1)
 *   Critic: s -> Dense(32) -> tanh -> Dense(24) -> tanh -> Dense(1) = V(s)
 */
export class ActorCriticBrain {
  constructor(obsDim, actDim, hidden1 = 32, hidden2 = 24) {
    this.obsDim = obsDim;
    this.actDim = actDim;
    this.hidden1 = hidden1;
    this.hidden2 = hidden2;

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

    // Dedicated Beetle Self-Righting Reflex Sub-Network ("Рефлекс жука"):
    // Completely isolated from walking weights; active and trained ONLY when overturned (up.y < 0.35)
    this.rightingLayer = new DenseLayer(obsDim, actDim, 0.04);
    this.rightingGain = 0.32;       // Starts weak (32%) so first overturn attempt only rocks the shell
    this.rightingAsym = 0.28;       // Learns asymmetric dorsal extension (one leg levers floor, opposite tucks)
    this.rightingSweep = 0.32;      // Learns longitudinal sweep leverage
    this.rightingSideSign = 1.0;    // Preferred roll-over edge (+1 or -1), adapts if blocked
    this.rightingAttempts = 0;
    this.rightingSuccesses = 0;
    this.rightingPhase = 0;

    this.adamStepCount = 0;
    this.maxSkipRowNorm = 1.80;

    // Live activation buffers for UI visualization
    this.lastObs = new Float64Array(obsDim);
    this.lastH1 = new Float64Array(hidden1);
    this.lastH2 = new Float64Array(hidden2);
    this.lastMu = new Float64Array(actDim);
    this.lastValue = 0;
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
    const numLegs = Math.floor(this.actDim / 2);
    // Resonance phase lag between vertical knee lift and horizontal hip stride (210 deg = 7*pi/6)
    const psi = (210 * Math.PI) / 180;

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
   * Forward pass of the dedicated Beetle Self-Righting Reflex ("Рефлекс жука")
   * when the walker is overturned on its back (up.y < 0.35).
   * Ignores food completely and coordinates dorsal leg extension + lateral tuck to roll onto feet.
   */
  forwardRightingReflex(obs, dt = 0.05, recordVis = true) {
    this.rightingPhase = (this.rightingPhase + dt * 6.8) % (Math.PI * 2);
    const numLegs = Math.floor(this.actDim / 2);
    const zRight = new Float64Array(this.actDim);
    this.rightingLayer.forward(obs, zRight);

    // Keep roll direction consistent throughout the attempt so it rolls all the way over the edge!
    const side = this.rightingSideSign;
    const s = Math.sin(this.rightingPhase);
    const c = Math.cos(this.rightingPhase);

    const mu = new Float64Array(this.actDim);
    for (let leg = 0; leg < numLegs; leg++) {
      const liftIdx = leg * 2;
      const strideIdx = leg * 2 + 1;
      let liftCmd = 0;
      let strideCmd = 0;

      if (leg === 0) {
        // Front-Left leg: if side >= 0, hyper-extend dorsally (-1) into floor to lever shell up; else tuck (+1)
        liftCmd = side >= 0
          ? -this.rightingGain * (0.82 + 0.18 * Math.max(0, s))
          : +this.rightingAsym;
        strideCmd = this.rightingSweep * s;
      } else if (leg === 1) {
        // Front-Right leg: opposite of Front-Left to create maximum roll torque across torso edge
        liftCmd = side >= 0
          ? +this.rightingAsym
          : -this.rightingGain * (0.82 + 0.18 * Math.max(0, s));
        strideCmd = -this.rightingSweep * s;
      } else {
        // Rear leg(s): rhythmic dorsal rocking lever
        liftCmd = -this.rightingGain * (0.75 * c + 0.25);
        strideCmd = this.rightingSweep * c;
      }

      mu[liftIdx] = Math.tanh(liftCmd + 0.15 * zRight[liftIdx]);
      mu[strideIdx] = Math.tanh(strideCmd + 0.15 * zRight[strideIdx]);
    }

    if (recordVis) {
      this.lastObs.set(obs);
      for (let i = 0; i < this.hidden1; i++) {
        this.lastH1[i] = Math.tanh(obs[6] * Math.cos(i) + obs[8] * Math.sin(i));
      }
      for (let i = 0; i < this.hidden2; i++) {
        this.lastH2[i] = Math.tanh(this.rightingGain * Math.sin(this.rightingPhase + i * 0.5));
      }
      this.lastMu.set(mu);
    }

    return mu;
  }

  /**
   * Updates the Beetle Self-Righting Reflex parameters at the end of a 12-step (0.6s) attempt on the back.
   * Reinforces dorsal extension gain, lateral tuck asymmetry, and posture-to-motor weights based on
   * achieved upward rotation (deltaUpY, maxUpY) and floor contact leverage.
   */
  updateRightingReflex({ startUpY, maxUpY, endUpY, contactSteps, lastObs, lastAct }) {
    this.rightingAttempts += 1;
    const progress = Math.max(0, maxUpY - startUpY) + Math.max(0, endUpY - startUpY);

    // Policy gradient / biomechanical reinforcement step on dorsal leverage gains:
    // Each attempt on its back teaches the beetle reflex that stronger dorsal push + opposite leg tuck
    // increases torso roll angular momentum and lifts up.y toward +1!
    const gainStep = 0.28 + 0.15 * Math.min(1.0, progress + contactSteps * 0.05);
    this.rightingGain = Math.min(1.0, this.rightingGain + gainStep * (1.0 - this.rightingGain * 0.65));
    this.rightingAsym = Math.min(1.0, this.rightingAsym + gainStep * (1.0 - this.rightingAsym * 0.65));
    this.rightingSweep = Math.min(0.95, this.rightingSweep + 0.22 * (1.0 - this.rightingSweep * 0.7));

    // If an attempt with high gain still didn't complete the roll over the edge, try rolling to the other side!
    if (endUpY < 0.25 && this.rightingAttempts >= 2) {
      this.rightingSideSign *= -1.0;
    }

    // Hebbian / policy-gradient update on rightingLayer from posture sensors (6,7,8)
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
    // Default ambition = 'bound' ("Бодрый скачок": trains energetic bounding gallop up to ~0.34+ m/s)
    this.maxCpgGain = 1.95;
    this.minFoodsToConsolidate = 2;
    this.alwaysLearn = false;
    this.manualFreeze = false;

    this.metaState = {
      mode: 'LEARNING_WALK', // 'LEARNING_WALK' | 'CONSOLIDATED' | 'LEARNING_RIGHTING'
      ambitionPreset: 'min',
      isLearningActive: true,
      modeTitle: '🟢 АКТИВНОЕ ОБУЧЕНИЕ: БАЗОВЫЙ ШАГ К ЦЕЛИ',
      reasonText: 'Формирование стабильного базового навыка ходьбы и наведения на цель',
      competenceEMA: 0.04,
      targetCompetence: 0.18,
      relearnThreshold: 0.085,
      surpriseEMA: 0.25,
      needToLearnPct: 88,
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
  }

  /**
   * External notification when environment changes (e.g. floor friction slider changed)
   */
  notifyEnvironmentChange(reason = 'Изменились физические параметры среды — проверка адаптации') {
    this.metaState.surpriseEMA = Math.max(this.metaState.surpriseEMA, 0.9);
    if (this.metaState.mode === 'CONSOLIDATED') {
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
    if (presetOrTarget === 'min') {
      this.metaState.ambitionPreset = 'min';
      this.metaState.targetCompetence = 0.18;
      this.metaState.relearnThreshold = 0.085;
      this.minFoodsToConsolidate = 2;
      this.maxCpgGain = 1.95;
      this.brain.maxSkipRowNorm = 1.30;
      this.alwaysLearn = false;
    } else if (presetOrTarget === 'bound') {
      this.metaState.ambitionPreset = 'bound';
      this.metaState.targetCompetence = 0.34;
      this.metaState.relearnThreshold = 0.16;
      this.minFoodsToConsolidate = 4;
      this.maxCpgGain = 3.25;
      this.brain.maxSkipRowNorm = 1.80;
      this.alwaysLearn = false;
    } else if (presetOrTarget === 'turbo') {
      this.metaState.ambitionPreset = 'turbo';
      this.metaState.targetCompetence = 0.48;
      this.metaState.relearnThreshold = 0.24;
      this.minFoodsToConsolidate = 6;
      this.maxCpgGain = 3.90;
      this.brain.maxSkipRowNorm = 2.15;
      this.alwaysLearn = false;
    } else if (presetOrTarget === 'infinite') {
      this.metaState.ambitionPreset = 'infinite';
      this.metaState.targetCompetence = 0.60;
      this.metaState.relearnThreshold = 0.45;
      this.minFoodsToConsolidate = 99999;
      this.maxCpgGain = 3.90;
      this.brain.maxSkipRowNorm = 2.15;
      this.alwaysLearn = true;
    } else if (typeof presetOrTarget === 'number') {
      const t = Math.max(0.14, Math.min(0.60, presetOrTarget));
      this.metaState.ambitionPreset = 'custom';
      this.metaState.targetCompetence = Number(t.toFixed(2));
      this.metaState.relearnThreshold = Number((t * 0.48).toFixed(2));
      this.minFoodsToConsolidate = t < 0.22 ? 2 : t < 0.40 ? 4 : 6;
      this.maxCpgGain = Number((1.4 + (t - 0.14) * 5.6).toFixed(2));
      this.brain.maxSkipRowNorm = Number((1.10 + (t - 0.14) * 2.3).toFixed(2));
      this.alwaysLearn = t >= 0.59;
    }

    // Immediately re-evaluate if the walker should wake up to train harder or consolidate now!
    if (this.metaState.mode !== 'LEARNING_RIGHTING') {
      if (this.alwaysLearn || this.metaState.competenceEMA < this.metaState.targetCompetence) {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.stableEpochsCount = 0;
        this.paramSigma = Math.max(this.paramSigma, 0.09);
        this.metaState.needToLearnPct = Math.round(
          Math.max(35, Math.min(98, ((this.metaState.targetCompetence - this.metaState.competenceEMA) / this.metaState.targetCompetence) * 90 + 25))
        );
        this.metaState.modeTitle = '🟢 АКТИВНОЕ ОБУЧЕНИЕ: РАЗГОН СКАЧКА К ЦЕЛИ';
        this.metaState.reasonText = this.alwaysLearn
          ? 'Включён режим непрерывного обучения (100% пластичность без остановки)'
          : `Планка мастерства повышена до ${this.metaState.targetCompetence.toFixed(2)} м/с — шагоход разгоняет прыгучую походку!`;
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
      this.metaState.needToLearnPct = 80;
      this.metaState.modeTitle = '🟢 АКТИВНОЕ ДООБУЧЕНИЕ (ВКЛЮЧЕНО ВРУЧНУЮ)';
      this.metaState.reasonText = 'Пользователь вручную запустил процесс дообучения походки';
    } else {
      this.manualFreeze = true;
      this.alwaysLearn = false;
      this.metaState.mode = 'CONSOLIDATED';
      this.metaState.isLearningActive = false;
      this.metaState.needToLearnPct = 0;
      this.metaState.modeTitle = '🔒 ОБУЧЕНИЕ ЗАМОРОЖЕНО ВРУЧНУЮ';
      this.metaState.reasonText = 'Веса нейросети зафиксированы пользователем — шагоход идёт на текущем навыке';
    }
  }

  _sampleWorkerParamNoise() {
    const dim = this.brain.actorSkip.W.length;
    this.workerSkipPerturbs = [];
    const tpl = this.brain.cpgTemplate;

    for (let w = 0; w < this.numWorkers; w++) {
      // If more than 1 worker, worker 0 is clean leader; if single walker (numWorkers === 1),
      // worker 0 explores only when metaState.mode === 'LEARNING_WALK'!
      if (this.numWorkers > 1 && w === 0) {
        this.workerSkipPerturbs.push(null);
        continue;
      }
      const p = new Float64Array(dim);
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
    const upY0 = c0.getTorsoFrame().up.y;
    const isOverturned0 = !c0.isGrabbed && !(c0.postThrowTimer > 0) && upY0 < 0.35;

    // 1. Check if main walker is overturned on its back -> engage Beetle Self-Righting Reflex!
    if (isOverturned0) {
      if (!this.wasOverturned) {
        this.wasOverturned = true;
        this.modeBeforeOverturn = this.metaState.mode === 'LEARNING_RIGHTING' ? 'LEARNING_WALK' : this.metaState.mode;
        // Clear any partial walking rollout so falling/overturned steps NEVER corrupt walking weights!
        this._resetBuffers();
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingContactSteps = 0;
      }

      this.metaState.mode = 'LEARNING_RIGHTING';
      this.metaState.isLearningActive = true;
      this.metaState.needToLearnPct = 100;
      this.metaState.rightingAttempts = this.brain.rightingAttempts + 1;
      this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();
      this.metaState.modeTitle = '🪲 РЕФЛЕКС ЖУКА: УЧИТСЯ ПЕРЕВОРАЧИВАТЬСЯ НА НОГИ';
      this.metaState.reasonText = `Корпус на спине (up.y=${upY0.toFixed(2)}): память ходьбы защищена, о еде не думает — попытка переворота #${this.metaState.rightingAttempts} (навык ${this.metaState.rightingSkillPct}%)`;

      const obs = c0.getObservation();
      const muRight = this.brain.forwardRightingReflex(obs, 0.05, true);
      c0.lastState = obs;
      c0.lastAction = muRight;
      c0.applyActionVector(muRight, true); // true = unlock dorsal beetle joint stroke!

      this._pendingStep = { isRightingStep: true, obs, act: muRight, upY: upY0 };
      return;
    }

    // 2. If walker just recovered from overturned back onto its feet!
    if (this.wasOverturned && upY0 >= 0.45) {
      this.wasOverturned = false;
      this.brain.rightingSuccesses += 1;
      this.metaState.rightingSuccesses = this.brain.rightingSuccesses;
      // Lock in the successful righting reflex!
      this.brain.rightingGain = Math.min(1.0, Math.max(0.88, this.brain.rightingGain));
      this.brain.rightingAsym = Math.min(1.0, Math.max(0.85, this.brain.rightingAsym));
      this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();

      // Restore previous walking mode (CONSOLIDATED or LEARNING_WALK)
      this.metaState.mode = this.modeBeforeOverturn || 'LEARNING_WALK';
      this.metaState.isLearningActive = this.metaState.mode === 'LEARNING_WALK';
      if (this.metaState.mode === 'CONSOLIDATED') {
        this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
        this.metaState.reasonText = `Успешно встал на ноги (переворотов: ${this.metaState.rightingSuccesses})! Продолжает ходьбу с защищёнными весами`;
      } else {
        this.metaState.modeTitle = '🟢 АКТИВНОЕ ОБУЧЕНИЕ: ХОДЬБА К ЦЕЛИ';
        this.metaState.reasonText = `Успешно встал на ноги (переворотов: ${this.metaState.rightingSuccesses})! Продолжает обучение ходьбе к цели`;
      }
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
        const noiseScale = !isTrainingWalk ? 0.015 : (this.numWorkers === 1 ? 0.14 : (w === 0 ? 0.02 : 0.32));
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

  recordStepOutcome(creatures, rewards, dones) {
    if (!this._pendingStep) return null;

    const c0 = creatures[0];
    const upY0 = c0.getTorsoFrame().up.y;

    // Handle Beetle Self-Righting Learning step when overturned on back:
    if (this._pendingStep.isRightingStep) {
      this.rightingStepCursor += 1;
      if (upY0 > this.rightingMaxUpY) this.rightingMaxUpY = upY0;
      const anyContact = c0.footNodes.some(f => f.inContact);
      if (anyContact) this.rightingContactSteps += 1;

      if (this.rightingStepCursor >= this.rightingHorizon) {
        const skillPct = this.brain.updateRightingReflex({
          startUpY: this.rightingStartUpY,
          maxUpY: this.rightingMaxUpY,
          endUpY: upY0,
          contactSteps: this.rightingContactSteps,
          lastObs: this._pendingStep.obs,
          lastAct: this._pendingStep.act
        });
        this.metaState.rightingAttempts = this.brain.rightingAttempts;
        this.metaState.rightingSkillPct = skillPct;
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingContactSteps = 0;
      }
      return null;
    }

    // If walker suddenly tipped over, is held by user's mouse, OR is in the 2-second pause between food orbs (DISSOLVING / WAITING),
    // do not push non-locomotion steps into the walking rollout buffer!
    if (c0.isGrabbed || c0.postThrowTimer > 0 || upY0 < 0.32 || c0.foodState === 'DISSOLVING' || c0.foodState === 'WAITING') {
      return null;
    }

    const { stepObs, stepAct, stepLogProb, stepVal } = this._pendingStep;

    // Track stall steps for meta-plasticity wake-up detector
    if (c0.lastVelocityTowardFood < 0.04) {
      this.metaState.stallStepsCount += 1;
    } else {
      this.metaState.stallStepsCount = Math.max(0, this.metaState.stallStepsCount - 2);
    }

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
    // Compute effective goal-directed competence (combining forward velocity + active yaw alignment when turning to new target)
    const meanEpochRew = epochLocomotionScore / Math.max(1, this.rolloutSteps);
    const recentVel = c0.lastVelocityTowardFood || 0;
    const approxEpochVel = Math.max(0.0, Math.min(0.8, meanEpochRew * 0.48 + 0.30 * Math.max(0, recentVel)));

    this.metaState.competenceEMA = Number(
      (0.76 * this.metaState.competenceEMA + 0.24 * approxEpochVel).toFixed(3)
    );
    this.metaState.surpriseEMA = Number(
      (0.75 * this.metaState.surpriseEMA + 0.25 * meanTdSurprise).toFixed(3)
    );

    const comp = this.metaState.competenceEMA;
    const target = this.metaState.targetCompetence;
    const foodCount = c0.foodCollected;
    const minFoods = this.minFoodsToConsolidate || 4;

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
      if (comp >= target && foodCount >= minFoods) {
        this.metaState.stableEpochsCount += 1;
      } else {
        this.metaState.stableEpochsCount = Math.max(0, this.metaState.stableEpochsCount - 1);
      }

      const rawNeed = Math.round(
        Math.max(15, Math.min(98, ((target * 1.20 - comp) / (target * 1.20)) * 85 + (foodCount < minFoods ? 20 : 0)))
      );
      this.metaState.needToLearnPct = rawNeed;

      // Transition to CONSOLIDATED once the walker has collected >= minFoods and maintained high velocity for 2 epochs
      if (this.metaState.stableEpochsCount >= 2 && foodCount >= minFoods) {
        this.metaState.mode = 'CONSOLIDATED';
        this.metaState.isLearningActive = false;
        this.metaState.lowVelEpochsCount = 0;
        this.metaState.stallStepsCount = 0;
        this.metaState.needToLearnPct = 10;
        this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
        this.metaState.reasonText = `Бодрый скачок освоен (${comp.toFixed(2)} м/с ≥ ${target.toFixed(2)} м/с, целей: ${foodCount}) — веса защищены от переобучения`;
      } else {
        this.metaState.isLearningActive = true;
        this.metaState.modeTitle = '🟢 АКТИВНОЕ ОБУЧЕНИЕ: РАЗГОН СКАЧКА К ЦЕЛИ';
        this.metaState.reasonText = foodCount < minFoods
          ? `Разгон прыгучей походки: собрано ${foodCount}/${minFoods} целей (эффективность ~${comp.toFixed(2)} / ${target.toFixed(2)} м/с)`
          : `Доводка скорости скачка: ${comp.toFixed(2)} м/с → цель ${target.toFixed(2)} м/с`;
      }
    } else if (this.metaState.mode === 'CONSOLIDATED') {
      // Monitor if environment changed, obstacle blocked path, or velocity dropped persistently
      if (comp < this.metaState.relearnThreshold && this.metaState.stallStepsCount > 90) {
        this.metaState.lowVelEpochsCount += 1;
      } else {
        this.metaState.lowVelEpochsCount = Math.max(0, this.metaState.lowVelEpochsCount - 1);
      }

      const deficit = Math.max(0, target - comp);
      this.metaState.needToLearnPct = Math.round(
        Math.min(95, 10 + (deficit / target) * 55 + (this.metaState.stallStepsCount > 60 ? 25 : 0))
      );

      if (this.metaState.lowVelEpochsCount >= 2 || this.metaState.stallStepsCount > 140) {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.stableEpochsCount = 0;
        this.metaState.stallStepsCount = 0;
        this.metaState.needToLearnPct = 78;
        this.paramSigma = Math.max(this.paramSigma, 0.085);
        this.metaState.modeTitle = '🟢 АКТИВНОЕ ДООБУЧЕНИЕ: АДАПТАЦИЯ К СРЕДЕ';
        this.metaState.reasonText = `Мета-чутьё обнаружило просадку скорости (${comp.toFixed(2)} м/с) — дообучение включено!`;
      } else {
        this.metaState.isLearningActive = false;
        this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
        this.metaState.reasonText = `Навык стабилен (эффективность ~${comp.toFixed(2)} м/с, целей: ${foodCount}) — нейросеть не меняет веса, защищая походку`;
      }
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

    // If in CONSOLIDATED mode (learning paused by meta-sense), skip weight updates completely!
    if (!wasLearningActive && this.metaState.mode === 'CONSOLIDATED') {
      this.updateCount += 1;
      this.lastMeanStepReward = rewardSum / totalSamples;
      this._resetBuffers();
      return {
        updateCount: this.updateCount,
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
        if (tpl) {
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
      meanStepReward: this.lastMeanStepReward,
      policyLoss: this.lastPolicyLoss,
      valueLoss: this.lastValueLoss,
      entropy: this.lastEntropy,
      metaMode: this.metaState.mode,
      isLearningActive: this.metaState.isLearningActive
    };
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
    this.updateCount = 0;
    this.lastMeanStepReward = 0;

    this.rightingStepCursor = 0;
    this.rightingHorizon = 14;
    this.rightingStartUpY = -1.0;
    this.rightingMaxUpY = -1.0;
    this.rightingContactSteps = 0;
    this.wasOverturned = false;
    this.modeBeforeOverturn = 'LEARNING_WALK';

    this.metaState = {
      mode: 'LEARNING_WALK',
      isLearningActive: true,
      modeTitle: '🟢 АКТИВНОЕ ОБУЧЕНИЕ (NES): ХОДЬБА К ЦЕЛИ',
      reasonText: 'Эволюционный поиск моторного навыка шага и руления',
      competenceEMA: 0.03,
      targetCompetence: 0.18,
      relearnThreshold: 0.085,
      ambitionPreset: 'min',
      surpriseEMA: 0.20,
      needToLearnPct: 85,
      stableEpochsCount: 0,
      lowVelEpochsCount: 0,
      stallStepsCount: 0,
      rightingAttempts: 0,
      rightingSuccesses: 0,
      rightingSkillPct: brain.getRightingSkillPct()
    };
    this.alwaysLearn = false;
    this.manualFreeze = false;
    this.minFoodsToConsolidate = 2;

    this._sampleWorkerPopulation();
  }

  setLearningAmbition(presetOrTarget) {
    this.manualFreeze = false;
    if (presetOrTarget === 'min') {
      this.alwaysLearn = false;
      this.metaState.ambitionPreset = 'min';
      this.metaState.targetCompetence = 0.18;
      this.metaState.relearnThreshold = 0.085;
      this.minFoodsToConsolidate = 2;
    } else if (presetOrTarget === 'bound') {
      this.alwaysLearn = false;
      this.metaState.ambitionPreset = 'bound';
      this.metaState.targetCompetence = 0.34;
      this.metaState.relearnThreshold = 0.15;
      this.minFoodsToConsolidate = 4;
    } else if (presetOrTarget === 'turbo') {
      this.alwaysLearn = false;
      this.metaState.ambitionPreset = 'turbo';
      this.metaState.targetCompetence = 0.46;
      this.metaState.relearnThreshold = 0.22;
      this.minFoodsToConsolidate = 6;
    } else if (presetOrTarget === 'infinite') {
      this.alwaysLearn = true;
      this.metaState.ambitionPreset = 'infinite';
      this.metaState.targetCompetence = 0.60;
      this.metaState.relearnThreshold = 0.35;
      this.minFoodsToConsolidate = 9999;
    } else if (typeof presetOrTarget === 'number') {
      this.alwaysLearn = presetOrTarget >= 0.54;
      this.metaState.ambitionPreset = 'custom';
      this.metaState.targetCompetence = Number(presetOrTarget.toFixed(2));
      this.metaState.relearnThreshold = Number(Math.max(0.08, presetOrTarget * 0.46).toFixed(2));
      this.minFoodsToConsolidate = presetOrTarget > 0.38 ? 5 : presetOrTarget > 0.24 ? 3 : 2;
    }
    if (this.alwaysLearn || this.metaState.competenceEMA < this.metaState.targetCompetence) {
      if (this.metaState.mode === 'CONSOLIDATED') {
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
      this.metaState.reasonText = 'Пользователь вручную запустил дообучение для улучшения походки';
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

  notifyEnvironmentChange(reason = 'Изменились параметры среды — проверка адаптации') {
    if (this.metaState.mode === 'CONSOLIDATED') {
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
    const upY0 = c0.getTorsoFrame().up.y;
    if (upY0 < 0.35) {
      if (!this.wasOverturned) {
        this.wasOverturned = true;
        this.modeBeforeOverturn = this.metaState.mode === 'LEARNING_RIGHTING' ? 'LEARNING_WALK' : this.metaState.mode;
        this.stepCursor = 0;
        this.workerReturns.fill(0);
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingContactSteps = 0;
      }
      this.metaState.mode = 'LEARNING_RIGHTING';
      this.metaState.isLearningActive = true;
      this.metaState.needToLearnPct = 100;
      this.metaState.rightingAttempts = this.brain.rightingAttempts + 1;
      this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();
      this.metaState.modeTitle = '🪲 РЕФЛЕКС ЖУКА: УЧИТСЯ ПЕРЕВОРАЧИВАТЬСЯ НА НОГИ';
      this.metaState.reasonText = `Корпус на спине (up.y=${upY0.toFixed(2)}): память ходьбы защищена — попытка переворота #${this.metaState.rightingAttempts}`;

      const obs = c0.getObservation();
      const muRight = this.brain.forwardRightingReflex(obs, 0.05, true);
      c0.lastState = obs;
      c0.lastAction = muRight;
      c0.applyActionVector(muRight, true);
      this._lastRightingStep = { obs, act: muRight };
      return;
    }

    if (this.wasOverturned && upY0 >= 0.45) {
      this.wasOverturned = false;
      this.brain.rightingSuccesses += 1;
      this.metaState.rightingSuccesses = this.brain.rightingSuccesses;
      this.metaState.rightingSkillPct = this.brain.getRightingSkillPct();
      this.metaState.mode = this.modeBeforeOverturn || 'LEARNING_WALK';
      this.metaState.isLearningActive = this.metaState.mode === 'LEARNING_WALK';
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
    const upY0 = c0.getTorsoFrame().up.y;
    if (this._lastRightingStep) {
      this.rightingStepCursor += 1;
      if (upY0 > this.rightingMaxUpY) this.rightingMaxUpY = upY0;
      if (c0.footNodes.some(f => f.inContact)) this.rightingContactSteps += 1;
      if (this.rightingStepCursor >= this.rightingHorizon) {
        this.metaState.rightingSkillPct = this.brain.updateRightingReflex({
          startUpY: this.rightingStartUpY,
          maxUpY: this.rightingMaxUpY,
          endUpY: upY0,
          contactSteps: this.rightingContactSteps,
          lastObs: this._lastRightingStep.obs,
          lastAct: this._lastRightingStep.act
        });
        this.metaState.rightingAttempts = this.brain.rightingAttempts;
        this.rightingStepCursor = 0;
        this.rightingStartUpY = upY0;
        this.rightingMaxUpY = upY0;
        this.rightingContactSteps = 0;
      }
      return null;
    }

    if (upY0 < 0.32 || c0.foodState === 'DISSOLVING' || c0.foodState === 'WAITING') {
      return null;
    }

    for (let w = 0; w < this.numWorkers; w++) {
      const r = rewards[w];
      this.workerReturns[w] += r > 5.0 ? r - 8.5 : r;
    }
    this.stepCursor += 1;
    if (this.stepCursor < this.horizonSteps) return null;

    const meanStepRew = this.workerReturns[0] / this.horizonSteps;
    this.lastMeanStepReward = meanStepRew;
    this.metaState.competenceEMA = Number(
      (0.75 * this.metaState.competenceEMA + 0.25 * Math.max(0, c0.lastVelocityTowardFood)).toFixed(3)
    );

    if (this.manualFreeze) {
      this.metaState.mode = 'CONSOLIDATED';
      this.metaState.isLearningActive = false;
      this.metaState.needToLearnPct = 0;
    } else if (this.alwaysLearn) {
      this.metaState.mode = 'LEARNING_WALK';
      this.metaState.isLearningActive = true;
      this.metaState.needToLearnPct = 100;
    } else if (this.metaState.mode === 'LEARNING_WALK') {
      if (this.metaState.competenceEMA >= this.metaState.targetCompetence && c0.foodCollected >= (this.minFoodsToConsolidate || 4)) {
        this.metaState.stableEpochsCount += 1;
        if (this.metaState.stableEpochsCount >= 2) {
          this.metaState.mode = 'CONSOLIDATED';
          this.metaState.isLearningActive = false;
          this.metaState.needToLearnPct = 14;
          this.metaState.modeTitle = '🛡️ НАВЫК ОСВОЕН: ОБУЧЕНИЕ ОТКЛЮЧЕНО';
          this.metaState.reasonText = `Скорость (${this.metaState.competenceEMA.toFixed(2)} м/с) стабильна — веса заморожены`;
        }
      }
    } else if (this.metaState.mode === 'CONSOLIDATED') {
      if (this.metaState.competenceEMA < this.metaState.relearnThreshold) {
        this.metaState.mode = 'LEARNING_WALK';
        this.metaState.isLearningActive = true;
        this.metaState.needToLearnPct = 75;
        this.metaState.modeTitle = '🟢 АКТИВНОЕ ДООБУЧЕНИЕ (NES): АДАПТАЦИЯ';
        this.metaState.reasonText = 'Скорость просела — возобновлён поиск параметров';
      }
    }

    this.updateCount += 1;
    if (this.metaState.isLearningActive) {
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
      meanStepReward: this.lastMeanStepReward,
      policyLoss: 0,
      valueLoss: 0,
      entropy: Math.log(this.sigma),
      metaMode: this.metaState.mode,
      isLearningActive: this.metaState.isLearningActive
    };
  }
}
