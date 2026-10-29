import { Vec3 } from '../physics/Physics3D.js';

/**
 * Pure 3D Perspective HTML5 Canvas Renderer for the Physics & RL Laboratory.
 * Renders:
 * - 3D Arena Floor Grid & Containment Walls
 * - All parallel creatures (3D rigid torso, 2-segment femur/tibia limbs, muscle tension lines, shadows)
 * - Physical Force Telemetry Vectors: Normal Reaction N_i and Floor Friction Vector F_friction
 * - Food targets and egocentric target sensor rays
 */
export class Renderer3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // 3D Orbit & Stage-Framing Camera state
    this.camYaw = -0.75;
    this.camPitch = 0.44;
    this.camDist = 3.6;
    this.camTarget = new Vec3(0, 0.18, 0);

    // Goal camera state for smooth stage transitions when a new food target spawns
    this.goalCamYaw = -0.75;
    this.goalCamDist = 3.6;
    this.goalCamTarget = new Vec3(0, 0.18, 0);
    this.lockedFoodPos = new Vec3(NaN, NaN, NaN);
    this.filteredLeaderPos = new Vec3(0, 0.18, 0);

    this.followLeader = true;
    this.showForceVectors = true;
    this.showAllPopulation = false;
    this.onTogglePause = null;
    this.onCameraAction = null;

    this._initPointerControls();
  }

  getCameraTelemetry() {
    return {
      yawDeg: Number(((this.camYaw * 180) / Math.PI).toFixed(1)),
      pitchDeg: Number(((this.camPitch * 180) / Math.PI).toFixed(1)),
      dist: Number(this.camDist.toFixed(2)),
      target: [
        Number(this.camTarget.x.toFixed(2)),
        Number(this.camTarget.y.toFixed(2)),
        Number(this.camTarget.z.toFixed(2))
      ],
      followLeader: this.followLeader
    };
  }

  _initPointerControls() {
    let dragging = false;
    let downX = 0;
    let downY = 0;
    let lastX = 0;
    let lastY = 0;
    let maxDragDist = 0;
    let usedShiftStrafe = false;
    let startYaw = 0;
    let startPitch = 0;
    let startTargetX = 0;
    let startTargetZ = 0;
    let wheelTimer = null;
    let wheelStartDist = null;

    this.canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      dragging = true;
      downX = e.clientX;
      downY = e.clientY;
      lastX = e.clientX;
      lastY = e.clientY;
      maxDragDist = 0;
      usedShiftStrafe = e.shiftKey;
      startYaw = this.camYaw;
      startPitch = this.camPitch;
      startTargetX = this.camTarget.x;
      startTargetZ = this.camTarget.z;
      this.canvas.setPointerCapture(e.pointerId);
    });

    this.canvas.addEventListener('pointermove', e => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      maxDragDist = Math.max(maxDragDist, Math.hypot(e.clientX - downX, e.clientY - downY));

      if (e.shiftKey) {
        usedShiftStrafe = true;
        // Quake-style strafe pan in the camera's horizontal / screen plane
        const cosP = Math.cos(this.camPitch);
        const sinP = Math.sin(this.camPitch);
        const cosY = Math.cos(this.camYaw);
        const sinY = Math.sin(this.camYaw);
        const camForward = new Vec3(-cosP * cosY, -sinP, -cosP * sinY).normalize();
        const worldUp = new Vec3(0, 1, 0);
        const camRight = Vec3.cross(camForward, worldUp).normalize();
        const horizFwd = new Vec3(camForward.x, 0, camForward.z).normalize();

        const panScale = this.camDist * 0.0018;
        const shiftX = (-dx * camRight.x + dy * horizFwd.x) * panScale;
        const shiftZ = (-dx * camRight.z + dy * horizFwd.z) * panScale;

        this.camTarget.x += shiftX;
        this.camTarget.z += shiftZ;
        this.goalCamTarget.x = this.camTarget.x;
        this.goalCamTarget.z = this.camTarget.z;
      } else {
        this.camYaw -= dx * 0.007;
        this.goalCamYaw = this.camYaw;
        this.camPitch = Math.max(0.12, Math.min(1.42, this.camPitch + dy * 0.006));
      }
    });

    this.canvas.addEventListener('pointerup', e => {
      if (!dragging) return;
      dragging = false;
      try {
        this.canvas.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      // Single click (without dragging) toggles simulation pause/resume
      if (maxDragDist < 5) {
        if (typeof this.onTogglePause === 'function') {
          this.onTogglePause('canvas_click');
        }
      } else if (typeof this.onCameraAction === 'function') {
        if (usedShiftStrafe) {
          this.onCameraAction({
            action: 'camera_strafe',
            dragPx: Math.round(maxDragDist),
            fromTarget: [Number(startTargetX.toFixed(2)), Number(startTargetZ.toFixed(2))],
            toTarget: [Number(this.camTarget.x.toFixed(2)), Number(this.camTarget.z.toFixed(2))],
            strafeMeters: Number(Math.hypot(this.camTarget.x - startTargetX, this.camTarget.z - startTargetZ).toFixed(2)),
            camera: this.getCameraTelemetry()
          });
        } else {
          this.onCameraAction({
            action: 'camera_orbit',
            dragPx: Math.round(maxDragDist),
            fromYawDeg: Number(((startYaw * 180) / Math.PI).toFixed(1)),
            toYawDeg: Number(((this.camYaw * 180) / Math.PI).toFixed(1)),
            fromPitchDeg: Number(((startPitch * 180) / Math.PI).toFixed(1)),
            toPitchDeg: Number(((this.camPitch * 180) / Math.PI).toFixed(1)),
            camera: this.getCameraTelemetry()
          });
        }
      }
    });

    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      if (wheelStartDist === null) {
        wheelStartDist = this.camDist;
      }
      this.camDist = Math.max(1.3, Math.min(22.0, this.camDist * (1 + e.deltaY * 0.001)));
      this.goalCamDist = this.camDist;

      if (wheelTimer) clearTimeout(wheelTimer);
      wheelTimer = setTimeout(() => {
        if (typeof this.onCameraAction === 'function' && wheelStartDist !== null) {
          this.onCameraAction({
            action: 'camera_zoom',
            fromDist: Number(wheelStartDist.toFixed(2)),
            toDist: Number(this.camDist.toFixed(2)),
            camera: this.getCameraTelemetry()
          });
        }
        wheelStartDist = null;
      }, 280);
    }, { passive: false });
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(300, Math.floor(rect.width * dpr));
    const h = Math.max(240, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  /**
   * Projects a 3D world point (x, y, z) into 2D canvas coordinates { sx, sy, depth, scale }.
   */
  project(pt, camPos, camForward, camRight, camUp, width, height, fovScale) {
    const dx = pt.x - camPos.x;
    const dy = pt.y - camPos.y;
    const dz = pt.z - camPos.z;

    const cz = dx * camForward.x + dy * camForward.y + dz * camForward.z;
    if (cz <= 0.15) return null;

    const cx = dx * camRight.x + dy * camRight.y + dz * camRight.z;
    const cy = dx * camUp.x + dy * camUp.y + dz * camUp.z;

    const invZ = fovScale / cz;
    return {
      sx: width * 0.5 + cx * invZ,
      sy: height * 0.5 - cy * invZ,
      depth: cz,
      scale: invZ
    };
  }

  render(sim, interpAlpha = 1.0) {
    this.resize();
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    const fovScale = Math.min(width, height) * 0.92;

    const focusIdx = this.showAllPopulation ? (sim.bestCreatureIdx || 0) : 0;
    const leader = sim.creatures[focusIdx] || sim.creatures[0];

    if (this.followLeader && leader) {
      const food = leader.foodPos;
      const foodChanged =
        Number.isNaN(this.lockedFoodPos.x) ||
        Math.hypot(food.x - this.lockedFoodPos.x, food.z - this.lockedFoodPos.z) > 0.01;

      if (foodChanged) {
        this.lockedFoodPos.copy(food);
        const com = leader.body.getCenterOfMass();

        // Center the stage between the walker's current position and the new food target
        const midX = 0.5 * (com.x + food.x);
        const midZ = 0.5 * (com.z + food.z);
        this.goalCamTarget.set(midX, 0.18, midZ);

        // Choose distance so both walker and food target fit comfortably inside the frame
        const span = Math.hypot(food.x - com.x, food.z - com.z);
        this.goalCamDist = Math.max(3.1, Math.min(8.0, span * 1.15 + 1.9));

        // Choose a clean 3/4 side-angle viewpoint relative to the walker->food line
        // picking the side closest to current camYaw for a gentle, minimal rotation
        const pathAngle = Math.atan2(food.z - com.z, food.x - com.x);
        const candA = pathAngle + Math.PI * 0.62;
        const candB = pathAngle - Math.PI * 0.62;
        const diffA = Math.atan2(Math.sin(candA - this.camYaw), Math.cos(candA - this.camYaw));
        const diffB = Math.atan2(Math.sin(candB - this.camYaw), Math.cos(candB - this.camYaw));
        this.goalCamYaw = Math.abs(diffA) <= Math.abs(diffB) ? this.camYaw + diffA : this.camYaw + diffB;
      }

      // Smoothly glide camera to the locked stage framing, and remain 100% stationary once settled!
      const dxT = this.goalCamTarget.x - this.camTarget.x;
      const dzT = this.goalCamTarget.z - this.camTarget.z;
      if (Math.hypot(dxT, dzT) > 0.0005) {
        this.camTarget.x += dxT * 0.065;
        this.camTarget.z += dzT * 0.065;
      } else {
        this.camTarget.x = this.goalCamTarget.x;
        this.camTarget.z = this.goalCamTarget.z;
      }
      this.camTarget.y = 0.18;

      const dDist = this.goalCamDist - this.camDist;
      if (Math.abs(dDist) > 0.0005) {
        this.camDist += dDist * 0.065;
      } else {
        this.camDist = this.goalCamDist;
      }

      const dYaw = Math.atan2(Math.sin(this.goalCamYaw - this.camYaw), Math.cos(this.goalCamYaw - this.camYaw));
      if (Math.abs(dYaw) > 0.0005) {
        this.camYaw += dYaw * 0.055;
      } else {
        this.camYaw = this.goalCamYaw;
      }
    }

    const cosP = Math.cos(this.camPitch);
    const sinP = Math.sin(this.camPitch);
    const cosY = Math.cos(this.camYaw);
    const sinY = Math.sin(this.camYaw);

    const camPos = new Vec3(
      this.camTarget.x + this.camDist * cosP * cosY,
      this.camTarget.y + this.camDist * sinP,
      this.camTarget.z + this.camDist * cosP * sinY
    );

    const camForward = Vec3.sub(this.camTarget, camPos).normalize();
    const worldUp = new Vec3(0, 1, 0);
    const camRight = Vec3.cross(camForward, worldUp).normalize();
    const camUp = Vec3.cross(camRight, camForward).normalize();

    const proj = pt => this.project(pt, camPos, camForward, camRight, camUp, width, height, fovScale);

    // 1. Background sky/lab gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
    bgGrad.addColorStop(0, '#090d16');
    bgGrad.addColorStop(1, '#111827');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // 2. Render 3D Floor Grid & Arena Containment Walls
    this._renderArenaFloorAndWalls(ctx, proj, sim.arenaHalfSize, sim.floorFriction);

    // 3. Render strictly the single main creature (idx 0) unless population view is enabled
    if (!this.showAllPopulation) {
      if (sim.creatures[0]) {
        this._renderCreatureAndFood(ctx, proj, sim.creatures[0], true, true, 1.0, interpAlpha);
      }
    } else {
      const indices = [];
      for (let i = 0; i < sim.creatures.length; i++) {
        indices.push(i);
      }
      indices.sort((a, b) => {
        if (a === focusIdx) return 1;
        if (b === focusIdx) return -1;
        const dA = Vec3.dist(sim.creatures[a].body.getCenterOfMass(), camPos);
        const dB = Vec3.dist(sim.creatures[b].body.getCenterOfMass(), camPos);
        return dB - dA;
      });

      for (const idx of indices) {
        const isFocused = idx === focusIdx || idx === 0;
        const alpha = isFocused ? 1.0 : 0.42;
        this._renderCreatureAndFood(ctx, proj, sim.creatures[idx], isFocused, idx === focusIdx, alpha, interpAlpha);
      }
    }
  }

  _renderArenaFloorAndWalls(ctx, proj, halfSize, floorFriction) {
    const step = 1.0;
    const iceTint = Math.max(0, 1 - floorFriction);

    ctx.lineWidth = 1;
    for (let x = -halfSize; x <= halfSize; x += step) {
      const isMajor = Math.abs(x % 5) < 0.01 || Math.abs(x) === halfSize;
      ctx.strokeStyle = isMajor
        ? `rgba(${Math.round(56 + iceTint * 100)}, 189, 248, 0.26)`
        : 'rgba(148, 163, 184, 0.11)';
      ctx.beginPath();
      let started = false;
      for (let z = -halfSize; z <= halfSize; z += 1.0) {
        const p = proj(new Vec3(x, 0, z));
        if (!p) continue;
        if (!started) {
          ctx.moveTo(p.sx, p.sy);
          started = true;
        } else {
          ctx.lineTo(p.sx, p.sy);
        }
      }
      ctx.stroke();
    }

    for (let z = -halfSize; z <= halfSize; z += step) {
      const isMajor = Math.abs(z % 5) < 0.01 || Math.abs(z) === halfSize;
      ctx.strokeStyle = isMajor
        ? `rgba(${Math.round(56 + iceTint * 100)}, 189, 248, 0.26)`
        : 'rgba(148, 163, 184, 0.11)';
      ctx.beginPath();
      let started = false;
      for (let x = -halfSize; x <= halfSize; x += 1.0) {
        const p = proj(new Vec3(x, 0, z));
        if (!p) continue;
        if (!started) {
          ctx.moveTo(p.sx, p.sy);
          started = true;
        } else {
          ctx.lineTo(p.sx, p.sy);
        }
      }
      ctx.stroke();
    }

    // 3D Containment Wall Fence (height = 0.45 m) around perimeter
    const wallH = 0.45;
    const corners = [
      new Vec3(-halfSize, 0, -halfSize),
      new Vec3(+halfSize, 0, -halfSize),
      new Vec3(+halfSize, 0, +halfSize),
      new Vec3(-halfSize, 0, +halfSize)
    ];

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
    ctx.lineWidth = 2;
    for (let c = 0; c < 4; c++) {
      const cA = corners[c];
      const cB = corners[(c + 1) % 4];
      const topA = new Vec3(cA.x, wallH, cA.z);
      const topB = new Vec3(cB.x, wallH, cB.z);

      this._draw3DSegment(ctx, proj, cA, cB, 16);
      this._draw3DSegment(ctx, proj, topA, topB, 16);

      const pBot = proj(cA);
      const pTop = proj(topA);
      if (pBot && pTop) {
        ctx.beginPath();
        ctx.moveTo(pBot.sx, pBot.sy);
        ctx.lineTo(pTop.sx, pTop.sy);
        ctx.stroke();
      }
    }
  }

  _draw3DSegment(ctx, proj, a, b, subdivisions = 8) {
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= subdivisions; i++) {
      const t = i / subdivisions;
      const pt = new Vec3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
      );
      const p = proj(pt);
      if (!p) continue;
      if (!started) {
        ctx.moveTo(p.sx, p.sy);
        started = true;
      } else {
        ctx.lineTo(p.sx, p.sy);
      }
    }
    ctx.stroke();
  }

  _renderCreatureAndFood(ctx, proj, creature, isFocused, isLeader, alpha, interpAlpha = 1.0) {
    ctx.save();
    ctx.globalAlpha = alpha;

    const a = Math.max(0.0, Math.min(1.0, interpAlpha));
    const posOf = node => {
      if (!node.prevPos || a >= 0.999) return node.pos;
      return new Vec3(
        node.prevPos.x + (node.pos.x - node.prevPos.x) * a,
        node.prevPos.y + (node.pos.y - node.prevPos.y) * a,
        node.prevPos.z + (node.pos.z - node.prevPos.z) * a
      );
    };

    const com = creature.body.getCenterOfMass();
    const { forward, right } = creature.getTorsoFrame();

    // 1. Target line from creature nose to its food sphere
    const pNose = proj(posOf(creature.frameRefs.front));
    const pFood = proj(creature.foodPos);
    const pFoodShadow = proj(new Vec3(creature.foodPos.x, 0.005, creature.foodPos.z));

    if (pFoodShadow) {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
      ctx.beginPath();
      ctx.ellipse(
        pFoodShadow.sx,
        pFoodShadow.sy,
        Math.max(3, creature.foodRadius * pFoodShadow.scale * 0.9),
        Math.max(2, creature.foodRadius * pFoodShadow.scale * 0.45),
        0,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }

    if (pNose && pFood) {
      ctx.strokeStyle = isLeader ? 'rgba(251, 191, 36, 0.65)' : 'rgba(52, 211, 153, 0.35)';
      ctx.lineWidth = isLeader ? 2 : 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(pNose.sx, pNose.sy);
      ctx.lineTo(pFood.sx, pFood.sy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw 3D Food Sphere (Nutrient Orb)
    if (pFood) {
      const rPx = Math.max(4, creature.foodRadius * 0.72 * pFood.scale);
      const grad = ctx.createRadialGradient(
        pFood.sx - rPx * 0.3,
        pFood.sy - rPx * 0.3,
        rPx * 0.1,
        pFood.sx,
        pFood.sy,
        rPx
      );
      grad.addColorStop(0, '#a7f3d0');
      grad.addColorStop(0.5, '#10b981');
      grad.addColorStop(1, '#065f46');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pFood.sx, pFood.sy, rPx, 0, Math.PI * 2);
      ctx.fill();
      if (isLeader) {
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 1.8;
        ctx.stroke();
      }
    }

    // 2. Ground shadow of torso CoM and feet
    const pComShadow = proj(new Vec3(com.x, 0.004, com.z));
    if (pComShadow) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.38)';
      ctx.beginPath();
      ctx.ellipse(
        pComShadow.sx,
        pComShadow.sy,
        Math.max(6, 0.30 * pComShadow.scale),
        Math.max(3, 0.16 * pComShadow.scale),
        0,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }

    // 3. Render Rigid Torso Structural Frame
    ctx.strokeStyle = isLeader ? '#fde047' : creature.color;
    ctx.lineWidth = isLeader ? 2.6 : 1.6;
    for (const bone of creature.body.bones) {
      if (bone.role !== 'torso') continue;
      const pA = proj(posOf(bone.nodeA));
      const pB = proj(posOf(bone.nodeB));
      if (pA && pB) {
        ctx.beginPath();
        ctx.moveTo(pA.sx, pA.sy);
        ctx.lineTo(pB.sx, pB.sy);
        ctx.stroke();
      }
    }

    // 4. Render 2-Segment Articulated Legs (Hip -> Knee -> Foot) + Muscle Tension Lines
    const L_femur = 0.16;
    const L_tibia = 0.16;
    for (let i = 0; i < creature.footNodes.length; i++) {
      const hip = creature.hipNodes[i];
      const foot = creature.footNodes[i];
      if (!hip || !foot) continue;

      const hipPos = posOf(hip);
      const footPos = posOf(foot);

      // Compute exact 3D knee joint position using 2-link inverse kinematics
      // Knee bends outward/forward along the leg's sagittal plane
      const mid = new Vec3(
        0.5 * (hipPos.x + footPos.x),
        0.5 * (hipPos.y + footPos.y),
        0.5 * (hipPos.z + footPos.z)
      );
      const legVec = Vec3.sub(footPos, hipPos);
      const legLen = Math.max(0.05, legVec.length());
      const halfSpan = Math.min(L_femur * 0.98, legLen * 0.5);
      const kneeOut = Math.sqrt(Math.max(0.0004, L_femur * L_femur - halfSpan * halfSpan));

      // Outward radial + anterior knee direction
      const bendDir = new Vec3(
        forward.x * 0.65 + (hipPos.x - com.x) * 1.4,
        0.25,
        forward.z * 0.65 + (hipPos.z - com.z) * 1.4
      ).normalize();

      const kneePos = new Vec3(
        mid.x + bendDir.x * kneeOut,
        mid.y + bendDir.y * kneeOut,
        mid.z + bendDir.z * kneeOut
      );

      const pHip = proj(hipPos);
      const pKnee = proj(kneePos);
      const pFoot = proj(footPos);

      // Draw active muscle lines colored by instantaneous force (cyan = pull, orange/red = push)
      const liftDof = creature.logicalDoFs[i * 2];
      const strideDof = creature.logicalDoFs[i * 2 + 1];
      if (isFocused && liftDof && strideDof) {
        for (const mus of [liftDof.primary, strideDof.primary]) {
          const pA = proj(posOf(mus.nodeA));
          const pB = proj(posOf(mus.nodeB));
          if (pA && pB) {
            const fNorm = Math.max(-1, Math.min(1, mus.appliedForce / 80.0));
            ctx.strokeStyle = fNorm >= 0
              ? `rgba(251, 113, 133, ${0.28 + 0.55 * Math.abs(fNorm)})`
              : `rgba(56, 189, 248, ${0.28 + 0.55 * Math.abs(fNorm)})`;
            ctx.lineWidth = 1.4 + Math.abs(fNorm) * 1.8;
            ctx.beginPath();
            ctx.moveTo(pA.sx, pA.sy);
            ctx.lineTo(pB.sx, pB.sy);
            ctx.stroke();
          }
        }
      }

      // Draw Femur (Hip -> Knee) and Tibia (Knee -> Foot)
      if (pHip && pKnee && pFoot) {
        ctx.strokeStyle = isLeader ? '#f8fafc' : '#cbd5e1';
        ctx.lineWidth = Math.max(2.0, 0.026 * pKnee.scale);
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(pHip.sx, pHip.sy);
        ctx.lineTo(pKnee.sx, pKnee.sy);
        ctx.lineTo(pFoot.sx, pFoot.sy);
        ctx.stroke();

        // Knee joint cap
        ctx.fillStyle = '#fb923c';
        ctx.beginPath();
        ctx.arc(pKnee.sx, pKnee.sy, Math.max(2.5, 0.028 * pKnee.scale), 0, Math.PI * 2);
        ctx.fill();
      }

      // Foot contact ring & Physical Reaction Force vectors (Normal N_i and Friction F_tr)
      const pFootGround = proj(new Vec3(footPos.x, 0.005, footPos.z));
      if (foot.inContact && pFootGround) {
        const ringRad = Math.max(4, Math.min(22, (0.07 + foot.normalForce * 0.003) * pFootGround.scale));
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.ellipse(pFootGround.sx, pFootGround.sy, ringRad, ringRad * 0.48, 0, 0, Math.PI * 2);
        ctx.stroke();

        if (this.showForceVectors && isFocused) {
          // Green vector: Normal Reaction Force N_i
          const nTop = proj(new Vec3(
            footPos.x,
            footPos.y + Math.min(0.35, foot.normalForce * 0.008),
            footPos.z
          ));
          if (pFoot && nTop) {
            ctx.strokeStyle = '#4ade80';
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            ctx.moveTo(pFoot.sx, pFoot.sy);
            ctx.lineTo(nTop.sx, nTop.sy);
            ctx.stroke();
          }

          // Yellow/Red vector: Horizontal Floor Friction Force F_friction propelling the creature!
          const fVec = foot.frictionForceVec;
          const fEnd = proj(new Vec3(
            footPos.x + fVec.x * 0.015,
            footPos.y,
            footPos.z + fVec.z * 0.015
          ));
          if (pFoot && fEnd) {
            ctx.strokeStyle = '#facc15';
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            ctx.moveTo(pFoot.sx, pFoot.sy);
            ctx.lineTo(fEnd.sx, fEnd.sy);
            ctx.stroke();
          }
        }
      }
    }

    // 5. Render 3D Nodes (Torso, Hips, Head Sensor, Feet)
    for (const node of creature.body.nodes) {
      const p = proj(posOf(node));
      if (!p) continue;
      const rPx = Math.max(3, node.radius * p.scale * 0.85);

      if (node.role === 'foot') {
        ctx.fillStyle = node.inContact ? '#4ade80' : '#94a3b8';
      } else if (node.role === 'head') {
        ctx.fillStyle = '#f43f5e';
      } else if (node.role === 'torso') {
        ctx.fillStyle = isLeader ? '#fbbf24' : creature.color;
      } else {
        ctx.fillStyle = creature.color;
      }

      ctx.beginPath();
      ctx.arc(p.sx, p.sy, rPx, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    ctx.restore();
  }
}
