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

    this.adamStepCount = 0;

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
    // 12 = clockSin * foodForward, 13 = clockCos * foodForward
    // 14 = clockSin * foodRight,   15 = clockCos * foodRight
    const numLegs = Math.floor(this.actDim / 2);
    for (let leg = 0; leg < numLegs; leg++) {
      const phaseShift = (2 * Math.PI * leg) / numLegs;
      const cosDelta = Math.cos(phaseShift);
      const sinDelta = Math.sin(phaseShift);

      const liftRow = (leg * 2) * this.obsDim;
      const strideRow = (leg * 2 + 1) * this.obsDim;

      // Embryonic 3-phase wave (starts weak at 0.22 so learning is clearly observable):
      // lift_k   ~ cos(phi + delta_k) = cos(delta_k)*cos(phi) - sin(delta_k)*sin(phi)
      // stride_k ~ sin(phi + delta_k) = cos(delta_k)*sin(phi) + sin(delta_k)*cos(phi)
      this.actorSkip.W[liftRow + 11] = 0.16 * cosDelta;
      this.actorSkip.W[liftRow + 10] = -0.16 * sinDelta;

      this.actorSkip.W[strideRow + 10] = -0.07 * cosDelta;
      this.actorSkip.W[strideRow + 11] = -0.07 * sinDelta;
      this.actorSkip.W[strideRow + 12] = -0.07 * cosDelta;
      this.actorSkip.W[strideRow + 13] = -0.07 * sinDelta;

      // Embryonic differential steering on front lateral legs (leg 0 at +right, leg 1 at -right)
      if (leg === 0) {
        this.actorSkip.W[strideRow + 14] = +0.06 * cosDelta;
        this.actorSkip.W[strideRow + 15] = +0.06 * sinDelta;
      } else if (leg === 1) {
        this.actorSkip.W[strideRow + 14] = -0.06 * cosDelta;
        this.actorSkip.W[strideRow + 15] = -0.06 * sinDelta;
      }
    }
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

  stepAdam(actorLr = 0.003, criticLr = 0.0045) {
    this.adamStepCount += 1;
    const s = this.adamStepCount;
    this.actorL1.adamStep(actorLr, 0.9, 0.999, 1e-8, s, 0.8);
    this.actorL2.adamStep(actorLr, 0.9, 0.999, 1e-8, s, 0.8);
    this.actorOut.adamStep(actorLr, 0.9, 0.999, 1e-8, s, 0.8);
    this.actorSkip.adamStep(actorLr * 1.5, 0.9, 0.999, 1e-8, s, 0.8);

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
 * PPO (Proximal Policy Optimization) + GAE + Parameter-Space Exploration + Adam Trainer
 * Operates on N parallel workers on the same arena in real time.
 */
export class PPOTrainer {
  constructor(brain, numWorkers = 16, {
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

    // Parameter-Space Exploration perturbations per worker (Plappert et al., ICLR 2018)
    this.paramSigma = 0.16;
    this.workerSkipPerturbs = [];
    this._sampleWorkerParamNoise();

    // Temporally correlated Ornstein-Uhlenbeck action exploration per worker
    this.ouNoise = Array.from({ length: numWorkers }, () => new Float64Array(brain.actDim));

    // Elite Self-Imitation Buffer
    this.eliteReplay = [];
    this.maxEliteSize = 256;

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

  _sampleWorkerParamNoise() {
    const dim = this.brain.actorSkip.W.length;
    this.workerSkipPerturbs = [];
    // Worker 0 is the clean unperturbed policy leader
    this.workerSkipPerturbs.push(null);

    for (let w = 1; w < this.numWorkers; w++) {
      const p = new Float64Array(dim);
      // Focus exploration on CPG (10,11) and sensory-gated CPG steering/drive (12,13,14,15)
      for (let a = 0; a < this.brain.actDim; a++) {
        const row = a * this.brain.obsDim;
        for (let c = 10; c <= 15 && c < this.brain.obsDim; c++) {
          p[row + c] = randn() * this.paramSigma * 1.35;
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

    for (let w = 0; w < this.numWorkers; w++) {
      const c = creatures[w];
      const obs = c.getObservation();
      const isLeader = w === 0;

      // Evaluate base policy mean for PPO logProb
      const { mu: baseMu } = this.brain.forwardActor(obs, isLeader, null);
      // Evaluate worker's perturbed policy for coherent rollout action
      const { mu: workerMu } = w === 0
        ? { mu: baseMu }
        : this.brain.forwardActor(obs, false, this.workerSkipPerturbs[w]);

      const { value } = this.brain.forwardCritic(obs, isLeader);

      const act = new Float64Array(this.brain.actDim);
      const ou = this.ouNoise[w];
      for (let a = 0; a < this.brain.actDim; a++) {
        const std = Math.exp(this.brain.logStd[a]);
        ou[a] = 0.78 * ou[a] + Math.sqrt(1 - 0.78 * 0.78) * randn();
        const noiseScale = w === 0 ? 0.02 : 0.35;
        const raw = workerMu[a] + ou[a] * std * noiseScale;
        act[a] = Math.max(-1, Math.min(1, raw));
      }

      stepObs[w] = obs;
      stepAct[w] = act;
      stepLogProb[w] = this.computeLogProb(baseMu, act);
      stepVal[w] = value;

      c.lastState = obs;
      c.lastAction = act;
      c.applyActionVector(act);
    }

    this._pendingStep = { stepObs, stepAct, stepLogProb, stepVal };
  }

  recordStepOutcome(creatures, rewards, dones) {
    if (!this._pendingStep) return null;
    const { stepObs, stepAct, stepLogProb, stepVal } = this._pendingStep;

    this.bufObs.push(stepObs);
    this.bufAct.push(stepAct);
    this.bufLogProb.push(stepLogProb);
    this.bufRew.push(Float64Array.from(rewards));
    this.bufVal.push(stepVal);
    this.bufDone.push(Uint8Array.from(dones));

    for (let w = 0; w < this.numWorkers; w++) {
      if (rewards[w] > 0.35) {
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
    const workerTotalReward = new Float64Array(W);

    let rewardSum = 0;
    for (let w = 0; w < W; w++) {
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const r = this.bufRew[t][w];
        rewardSum += r;
        workerTotalReward[w] += r;
        const notDone = this.bufDone[t][w] ? 0.0 : 1.0;
        const vNext = t === T - 1 ? nextValues[w] : this.bufVal[t + 1][w];
        const delta = r + this.gamma * vNext * notDone - this.bufVal[t][w];
        gae = delta + this.gamma * this.gaeLambda * notDone * gae;
        advantages[t][w] = gae;
        returns[t][w] = gae + this.bufVal[t][w];
      }
    }

    // Parameter-Space Elite Natural Step: if exploring workers outperformed the mean policy,
    // blend the top 3 worker parameter perturbations into actorSkip.W before PPO fine-tuning!
    const rankedWorkers = Array.from({ length: W - 1 }, (_, i) => i + 1).sort(
      (a, b) => workerTotalReward[b] - workerTotalReward[a]
    );
    const bestWorker = rankedWorkers[0];
    if (bestWorker !== undefined && workerTotalReward[bestWorker] > workerTotalReward[0] + 0.25) {
      const skipW = this.brain.actorSkip.W;
      const topWeights = [0.52, 0.22, 0.10];
      for (let rank = 0; rank < Math.min(3, rankedWorkers.length); rank++) {
        const wIdx = rankedWorkers[rank];
        if (workerTotalReward[wIdx] > workerTotalReward[0]) {
          const pert = this.workerSkipPerturbs[wIdx];
          const alpha = topWeights[rank];
          for (let i = 0; i < skipW.length; i++) {
            skipW[i] += alpha * pert[i];
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
    for (let i = 0; i < totalSamples; i++) {
      flatAdv[i] = (flatAdv[i] - advMean) / advStd;
    }

    // Mini-batch PPO + Adam Backpropagation updates
    const batchSize = 64;
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
        if (this.eliteReplay.length >= 8) {
          const silSamples = 8;
          const silWeight = 0.18 / silSamples;
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

        this.brain.stepAdam(this.actorLr, this.criticLr);
        numBatches++;
      }
    }

    this.updateCount += 1;
    this.paramSigma = Math.max(0.06, this.paramSigma * 0.985);
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
      entropy: this.lastEntropy
    };
  }
}

/**
 * Natural Evolution Strategies / SEP-CMA-ES Trainer (Algorithm #2 for comparison)
 */
export class NESTrainer {
  constructor(brain, numWorkers = 16, {
    horizonSteps = 64,
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
    this.stepCursor = 0;
    this.updateCount = 0;
    this.lastMeanStepReward = 0;

    this._sampleWorkerPopulation();
  }

  _sampleWorkerPopulation() {
    this.workerPerturbations = [];
    this.workerReturns.fill(0);
    this.workerPerturbations.push(new Float64Array(this.paramDim));

    for (let w = 1; w < this.numWorkers; w++) {
      if (w % 2 === 1 || w === this.numWorkers - 1) {
        const eps = new Float64Array(this.paramDim);
        for (let i = 0; i < this.paramDim; i++) eps[i] = randn();
        this.workerPerturbations.push(eps);
      } else {
        const prev = this.workerPerturbations[w - 1];
        const neg = new Float64Array(this.paramDim);
        for (let i = 0; i < this.paramDim; i++) neg[i] = -prev[i];
        this.workerPerturbations.push(neg);
      }
    }
  }

  selectActions(creatures) {
    const tempParams = new Float64Array(this.paramDim);
    for (let w = 0; w < this.numWorkers; w++) {
      const eps = this.workerPerturbations[w];
      for (let i = 0; i < this.paramDim; i++) {
        tempParams[i] = this.centerParams[i] + this.sigma * eps[i];
      }
      this.brain.setActorFlatParams(tempParams);
      const obs = creatures[w].getObservation();
      const { mu } = this.brain.forwardActor(obs, w === 0, null);
      creatures[w].lastState = obs;
      creatures[w].lastAction = mu;
      creatures[w].applyActionVector(mu);
    }
    this.brain.setActorFlatParams(this.centerParams);
  }

  recordStepOutcome(_creatures, rewards, _dones) {
    for (let w = 0; w < this.numWorkers; w++) {
      this.workerReturns[w] += rewards[w];
    }
    this.stepCursor += 1;
    if (this.stepCursor < this.horizonSteps) return null;

    const W = this.numWorkers;
    let sumRaw = 0;
    for (let w = 0; w < W; w++) sumRaw += this.workerReturns[w];
    this.lastMeanStepReward = sumRaw / (W * this.horizonSteps);

    const order = Array.from({ length: W }, (_, i) => i).sort(
      (a, b) => this.workerReturns[b] - this.workerReturns[a]
    );
    const utilities = new Float64Array(W);
    for (let rank = 0; rank < W; rank++) {
      const wIdx = order[rank];
      utilities[wIdx] = Math.max(0, Math.log(W / 2 + 1) - Math.log(rank + 1));
    }
    let uSum = 0;
    for (let w = 0; w < W; w++) uSum += utilities[w];
    for (let w = 0; w < W; w++) utilities[w] = utilities[w] / Math.max(1e-8, uSum) - 1 / W;

    this.updateCount += 1;
    const beta1 = 0.9, beta2 = 0.999;
    const bc1 = 1 - Math.pow(beta1, this.updateCount);
    const bc2 = 1 - Math.pow(beta2, this.updateCount);

    for (let i = 0; i < this.paramDim; i++) {
      let grad = 0;
      for (let w = 1; w < W; w++) {
        grad += utilities[w] * this.workerPerturbations[w][i];
      }
      grad /= this.sigma;
      this.adamM[i] = beta1 * this.adamM[i] + (1 - beta1) * grad;
      this.adamV[i] = beta2 * this.adamV[i] + (1 - beta2) * grad * grad;
      const mHat = this.adamM[i] / bc1;
      const vHat = this.adamV[i] / bc2;
      this.centerParams[i] += this.lr * mHat / (Math.sqrt(vHat) + 1e-8);
    }

    this.brain.setActorFlatParams(this.centerParams);
    this.stepCursor = 0;
    this._sampleWorkerPopulation();

    return {
      updateCount: this.updateCount,
      meanStepReward: this.lastMeanStepReward,
      policyLoss: 0,
      valueLoss: 0,
      entropy: Math.log(this.sigma)
    };
  }
}
