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
    this.showAllPopulation = false;
    this.onTogglePause = null;
    this.onCameraAction = null;
    this.onSceneInteract = null;

    this._sim = null;
    this._lastCam = null;
    this.hoveredObject = null; // 'food' | 'walker' | null
    this.activeDragMode = null; // 'food' | 'walker' | 'camera' | null

    this._initPointerControls();
  }

  _clientToCanvas(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = this.canvas.width / Math.max(1, rect.width);
    return {
      px: (clientX - rect.left) * dpr,
      py: (clientY - rect.top) * dpr,
      dpr
    };
  }

  _intersectHorizontalPlane(px, py, planeY = 0.16) {
    if (!this._lastCam) return null;
    const { camPos, camForward, camRight, camUp, width, height, fovScale } = this._lastCam;
    const cx = (px - width * 0.5) / fovScale;
    const cy = (height * 0.5 - py) / fovScale;
    const rayDir = new Vec3(
      camForward.x + cx * camRight.x + cy * camUp.x,
      camForward.y + cx * camRight.y + cy * camUp.y,
      camForward.z + cx * camRight.z + cy * camUp.z
    ).normalize();

    if (Math.abs(rayDir.y) < 1e-4) return null;
    const t = (planeY - camPos.y) / rayDir.y;
    if (t <= 0.1 || t > 65.0) return null;

    const half = this._sim ? this._sim.arenaHalfSize - 0.65 : 10.35;
    return new Vec3(
      Math.max(-half, Math.min(half, camPos.x + rayDir.x * t)),
      planeY,
      Math.max(-half, Math.min(half, camPos.z + rayDir.z * t))
    );
  }

  _hitTestScene(clientX, clientY) {
    if (!this._sim || !this._lastCam) return null;
    const c0 = this._sim.creatures[0];
    if (!c0) return null;
    const { px, py, dpr } = this._clientToCanvas(clientX, clientY);
    const { camPos, camForward, camRight, camUp, width, height, fovScale } = this._lastCam;
    const proj = pt => this.project(pt, camPos, camForward, camRight, camUp, width, height, fovScale);

    // 1. Check distance to Food Orb
    let foodDistPx = Infinity;
    const foodTargetPt = c0.foodState === 'DISSOLVING' ? c0.dissolvePos : c0.foodPos;
    const pFood = proj(foodTargetPt);
    if (pFood && c0.foodState !== 'WAITING') {
      const foodHitRad = Math.max(24 * dpr, 0.24 * pFood.scale);
      const d = Math.hypot(px - pFood.sx, py - pFood.sy);
      if (d <= foodHitRad) {
        foodDistPx = d;
      }
    }

    // 2. Check distance to Walker (Torso center + body nodes)
    let walkerDistPx = Infinity;
    const com = c0.body.getCenterOfMass();
    const pCom = proj(com);
    if (pCom) {
      const comHitRad = Math.max(30 * dpr, 0.34 * pCom.scale);
      const dCom = Math.hypot(px - pCom.sx, py - pCom.sy);
      if (dCom <= comHitRad) {
        walkerDistPx = dCom;
      }
    }
    for (const n of c0.body.nodes) {
      const pn = proj(n.pos);
      if (!pn) continue;
      const nodeHitRad = Math.max(18 * dpr, (n.radius + 0.08) * pn.scale);
      const dn = Math.hypot(px - pn.sx, py - pn.sy);
      if (dn <= nodeHitRad && dn < walkerDistPx) {
        walkerDistPx = dn;
      }
    }

    if (foodDistPx < Infinity && foodDistPx <= walkerDistPx) return 'food';
    if (walkerDistPx < Infinity) return 'walker';
    return null;
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

    // State for 3D object dragging (food orb or walker)
    let dragOffsetXZ = { x: 0, z: 0 };
    let dragStartWorldPos = null;
    let lastWorldX = 0;
    let lastWorldZ = 0;
    let lastMoveTime = 0;
    let throwVx = 0;
    let throwVz = 0;

    const finishPointerInteraction = (pointerId = null, wasCancelled = false) => {
      if (!dragging) return;
      const mode = this.activeDragMode;
      dragging = false;
      this.activeDragMode = null;
      this.canvas.classList.remove('dragging');
      if (pointerId !== null) {
        try {
          if (this.canvas.hasPointerCapture(pointerId)) {
            this.canvas.releasePointerCapture(pointerId);
          }
        } catch {
          // ignore
        }
      }

      const c0 = this._sim && this._sim.creatures[0];
      if (mode === 'food' && this._sim && c0) {
        this._sim.moveFoodTo(0, c0.foodPos.x, c0.foodPos.z, false);
        this.lockedFoodPos.set(NaN, NaN, NaN);
        if (typeof this.onSceneInteract === 'function' && dragStartWorldPos) {
          this.onSceneInteract({
            action: 'drag_food',
            fromPos: [Number(dragStartWorldPos.x.toFixed(2)), Number(dragStartWorldPos.z.toFixed(2))],
            toPos: [Number(c0.foodPos.x.toFixed(2)), Number(c0.foodPos.z.toFixed(2))]
          });
        }
        return;
      }

      if (mode === 'walker' && this._sim && c0) {
        // Decay velocity if mouse stood still for > 90 ms before release (gentle drop vs brisk throw)
        const idleMs = performance.now() - lastMoveTime;
        const decay = idleMs > 110 ? 0.0 : Math.max(0.0, 1.0 - idleMs / 140);
        const finalVx = wasCancelled ? 0 : throwVx * decay;
        const finalVz = wasCancelled ? 0 : throwVz * decay;
        this._sim.releaseCreature(0, finalVx, finalVz);
        // Track the thrown walker directly without rotating camera yaw so the user sees where it lands and self-rights
        this._postThrowTrackUntil = performance.now() + 3200;
        this._wasTrackingThrowOrOverturn = true;
        this.goalCamYaw = this.camYaw;
        const com = c0.body.getCenterOfMass();
        if (typeof this.onSceneInteract === 'function' && dragStartWorldPos) {
          this.onSceneInteract({
            action: 'drag_throw_walker',
            fromPos: [Number(dragStartWorldPos.x.toFixed(2)), Number(dragStartWorldPos.z.toFixed(2))],
            toPos: [Number(com.x.toFixed(2)), Number(com.z.toFixed(2))],
            throwVel: [Number(finalVx.toFixed(2)), Number(finalVz.toFixed(2))],
            throwSpeed: Number(Math.hypot(finalVx, finalVz).toFixed(2))
          });
        }
        return;
      }

      if (wasCancelled) return;

      // Single click on background (without dragging) toggles simulation pause/resume
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
    };

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

      const hit = !e.shiftKey ? this._hitTestScene(e.clientX, e.clientY) : null;
      const c0 = this._sim && this._sim.creatures[0];
      const { px, py } = this._clientToCanvas(e.clientX, e.clientY);

      if (hit === 'food' && c0) {
        this.activeDragMode = 'food';
        this.canvas.classList.add('dragging');
        const planeHit = this._intersectHorizontalPlane(px, py, 0.16);
        const startFood = c0.foodState === 'DISSOLVING' ? c0.dissolvePos : c0.foodPos;
        dragStartWorldPos = startFood.clone();
        dragOffsetXZ = planeHit
          ? { x: startFood.x - planeHit.x, z: startFood.z - planeHit.z }
          : { x: 0, z: 0 };
        this._sim.moveFoodTo(0, startFood.x, startFood.z, true);
      } else if (hit === 'walker' && c0) {
        this.activeDragMode = 'walker';
        this.canvas.classList.add('dragging');
        const com = c0.body.getCenterOfMass();
        dragStartWorldPos = com.clone();
        const planeHit = this._intersectHorizontalPlane(px, py, 0.25);
        dragOffsetXZ = planeHit
          ? { x: com.x - planeHit.x, z: com.z - planeHit.z }
          : { x: 0, z: 0 };
        lastWorldX = com.x;
        lastWorldZ = com.z;
        lastMoveTime = performance.now();
        throwVx = 0;
        throwVz = 0;
        this._sim.grabCreature(0);
      } else {
        this.activeDragMode = 'camera';
      }

      this.canvas.setPointerCapture(e.pointerId);
    });

    this.canvas.addEventListener('pointermove', e => {
      if (!dragging) {
        const hover = this._hitTestScene(e.clientX, e.clientY);
        this.hoveredObject = hover;
        this.canvas.classList.toggle('grab-hover', Boolean(hover));
        return;
      }

      if (!this.canvas.hasPointerCapture(e.pointerId)) return;

      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      maxDragDist = Math.max(maxDragDist, Math.hypot(e.clientX - downX, e.clientY - downY));

      if (this.activeDragMode === 'food' && this._sim) {
        const { px, py } = this._clientToCanvas(e.clientX, e.clientY);
        const planeHit = this._intersectHorizontalPlane(px, py, 0.16);
        if (planeHit) {
          this._sim.moveFoodTo(0, planeHit.x + dragOffsetXZ.x, planeHit.z + dragOffsetXZ.z, true);
        }
        return;
      }

      if (this.activeDragMode === 'walker' && this._sim) {
        const { px, py } = this._clientToCanvas(e.clientX, e.clientY);
        const planeHit = this._intersectHorizontalPlane(px, py, 0.25);
        if (planeHit) {
          const now = performance.now();
          const dtMove = Math.max(0.008, Math.min(0.12, (now - lastMoveTime) / 1000));
          const nextX = planeHit.x + dragOffsetXZ.x;
          const nextZ = planeHit.z + dragOffsetXZ.z;
          const instVx = (nextX - lastWorldX) / dtMove;
          const instVz = (nextZ - lastWorldZ) / dtMove;
          throwVx = 0.55 * throwVx + 0.45 * instVx;
          throwVz = 0.55 * throwVz + 0.45 * instVz;
          lastWorldX = nextX;
          lastWorldZ = nextZ;
          lastMoveTime = now;
          this._sim.dragCreatureTo(0, nextX, nextZ, throwVx, throwVz);
        }
        return;
      }

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
      finishPointerInteraction(e.pointerId, false);
    });

    this.canvas.addEventListener('pointercancel', e => {
      finishPointerInteraction(e.pointerId, true);
    });

    this.canvas.addEventListener('lostpointercapture', () => {
      finishPointerInteraction(null, true);
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
    this._sim = sim;
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    const fovScale = Math.min(width, height) * 0.92;

    const focusIdx = this.showAllPopulation ? (sim.bestCreatureIdx || 0) : 0;
    const leader = sim.creatures[focusIdx] || sim.creatures[0];

    if (this.followLeader && leader && !this.activeDragMode) {
      const food = leader.foodPos;
      const com = leader.body.getCenterOfMass();
      const isOverturned = leader.getTorsoFrame().up.y < 0.48;
      const inPostThrow = performance.now() < (this._postThrowTrackUntil || 0);

      if (inPostThrow || isOverturned) {
        // Keep camera angle (yaw) fixed and follow the walker directly so the user clearly sees where it landed and how it self-rights
        this._wasTrackingThrowOrOverturn = true;
        this.goalCamYaw = this.camYaw;
        this.goalCamTarget.set(com.x, 0.18, com.z);
        this.goalCamDist = Math.max(3.4, Math.min(4.8, this.camDist));
      } else {
        // Only re-frame camera when a new food target is ACTIVE or begins DROPPING from the sky, or right after recovery
        const hasVisibleTarget = leader.foodState === 'ACTIVE' || leader.foodState === 'DROPPING';
        const foodChanged =
          hasVisibleTarget && (
            Number.isNaN(this.lockedFoodPos.x) ||
            Math.hypot(food.x - this.lockedFoodPos.x, food.z - this.lockedFoodPos.z) > 0.01
          );

        if (this._wasTrackingThrowOrOverturn || foodChanged) {
          const justRecovered = Boolean(this._wasTrackingThrowOrOverturn);
          this._wasTrackingThrowOrOverturn = false;
          this.lockedFoodPos.copy(food);

          // Center the stage between the walker's current position and the food target
          const midX = 0.5 * (com.x + food.x);
          const midZ = 0.5 * (com.z + food.z);
          this.goalCamTarget.set(midX, 0.18, midZ);

          // Choose distance so both walker and food target fit comfortably inside the frame
          const span = Math.hypot(food.x - com.x, food.z - com.z);
          this.goalCamDist = Math.max(3.1, Math.min(8.0, span * 1.15 + 1.9));

          if (justRecovered) {
            // Preserve user's viewing angle after recovery from a throw
            this.goalCamYaw = this.camYaw;
          } else {
            // Choose a clean 3/4 side-angle viewpoint relative to the walker->food line
            const pathAngle = Math.atan2(food.z - com.z, food.x - com.x);
            const candA = pathAngle + Math.PI * 0.62;
            const candB = pathAngle - Math.PI * 0.62;
            const diffA = Math.atan2(Math.sin(candA - this.camYaw), Math.cos(candA - this.camYaw));
            const diffB = Math.atan2(Math.sin(candB - this.camYaw), Math.cos(candB - this.camYaw));
            this.goalCamYaw = Math.abs(diffA) <= Math.abs(diffB) ? this.camYaw + diffA : this.camYaw + diffB;
          }
        }
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

    this._lastCam = { camPos, camForward, camRight, camUp, width, height, fovScale };
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

    // 1. Render 3-Phase Animated Food Lifecycle (DISSOLVING 1s -> WAITING 1s -> DROPPING 1s -> ACTIVE)
    const pNose = proj(posOf(creature.frameRefs.front));
    const fState = creature.foodState || 'ACTIVE';
    const fTimer = Math.max(0.0, Math.min(1.0, creature.foodStateTimer || 0.0));

    if (fState === 'DISSOLVING') {
      // Phase 1 (1.0s): Eaten food orb shrinks, rises slightly, and dissolves with golden-emerald absorption rings
      const dPos = new Vec3(
        creature.dissolvePos.x,
        0.16 + 0.24 * fTimer,
        creature.dissolvePos.z
      );
      const pDissolve = proj(dPos);
      const pGround = proj(new Vec3(creature.dissolvePos.x, 0.005, creature.dissolvePos.z));
      const fadeAlpha = Math.max(0.0, 1.0 - fTimer);

      if (pGround) {
        const ringR = Math.max(6, (0.18 + 0.55 * fTimer) * pGround.scale);
        ctx.strokeStyle = `rgba(251, 191, 36, ${(0.75 * fadeAlpha).toFixed(3)})`;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.ellipse(pGround.sx, pGround.sy, ringR, ringR * 0.48, 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (pDissolve) {
        const shrink = Math.max(0.08, 1.0 - 0.88 * fTimer * fTimer);
        const rPx = Math.max(2, creature.foodRadius * 0.72 * pDissolve.scale * shrink);

        ctx.save();
        ctx.globalAlpha = alpha * fadeAlpha;

        // Expanding glowing halo around dissolving orb
        const haloR = rPx * (1.4 + 2.2 * fTimer);
        const haloGrad = ctx.createRadialGradient(pDissolve.sx, pDissolve.sy, rPx * 0.2, pDissolve.sx, pDissolve.sy, haloR);
        haloGrad.addColorStop(0, 'rgba(253, 224, 71, 0.85)');
        haloGrad.addColorStop(0.5, 'rgba(52, 211, 153, 0.45)');
        haloGrad.addColorStop(1, 'rgba(16, 185, 129, 0.0)');
        ctx.fillStyle = haloGrad;
        ctx.beginPath();
        ctx.arc(pDissolve.sx, pDissolve.sy, haloR, 0, Math.PI * 2);
        ctx.fill();

        // Shrinking core orb
        ctx.fillStyle = '#fde047';
        ctx.beginPath();
        ctx.arc(pDissolve.sx, pDissolve.sy, rPx, 0, Math.PI * 2);
        ctx.fill();

        // Energy particles streaming toward walker's nose
        if (pNose) {
          for (let k = 0; k < 5; k++) {
            const u = (fTimer + k * 0.18) % 1.0;
            const px = pDissolve.sx + (pNose.sx - pDissolve.sx) * u + Math.sin(u * 10 + k) * 6 * (1 - u);
            const py = pDissolve.sy + (pNose.sy - pDissolve.sy) * u - Math.sin(u * Math.PI) * 10;
            ctx.fillStyle = k % 2 === 0 ? '#fde047' : '#34d399';
            ctx.beginPath();
            ctx.arc(px, py, Math.max(1.8, 3.5 * (1 - u)), 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.restore();
      }
    } else if (fState === 'WAITING') {
      // Phase 2 (1.0s): Complete pause between eaten orb and new orb — arena is clear!
    } else {
      // Phase 3 ('DROPPING' from sky for 1.0s) or Phase 4 ('ACTIVE' on floor)
      const pFood = proj(creature.foodPos);
      const pFoodShadow = proj(new Vec3(creature.foodPos.x, 0.005, creature.foodPos.z));
      const heightAboveFloor = Math.max(0, creature.foodPos.y - 0.16);

      if (fState === 'DROPPING' && pFood && pFoodShadow) {
        // Draw vertical sky-drop light beam and landing ripple ring
        const pSkyTop = proj(new Vec3(creature.foodPos.x, 3.4, creature.foodPos.z));
        if (pSkyTop) {
          const beamAlpha = Math.max(0.0, 0.55 * (1.0 - fTimer));
          const beamGrad = ctx.createLinearGradient(pSkyTop.sx, pSkyTop.sy, pFoodShadow.sx, pFoodShadow.sy);
          beamGrad.addColorStop(0, 'rgba(56, 189, 248, 0.0)');
          beamGrad.addColorStop(0.5, `rgba(52, 211, 153, ${beamAlpha.toFixed(3)})`);
          beamGrad.addColorStop(1, `rgba(251, 191, 36, ${(beamAlpha * 1.3).toFixed(3)})`);
          ctx.strokeStyle = beamGrad;
          ctx.lineWidth = Math.max(2, 0.12 * pFoodShadow.scale * (1.0 - 0.5 * fTimer));
          ctx.beginPath();
          ctx.moveTo(pSkyTop.sx, pSkyTop.sy);
          ctx.lineTo(pFoodShadow.sx, pFoodShadow.sy);
          ctx.stroke();
        }

        if (fTimer > 0.68) {
          const bounceProg = (fTimer - 0.68) / 0.32;
          const rippleR = Math.max(5, (0.18 + 0.55 * bounceProg) * pFoodShadow.scale);
          ctx.strokeStyle = `rgba(52, 211, 153, ${(0.7 * (1 - bounceProg)).toFixed(3)})`;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.ellipse(pFoodShadow.sx, pFoodShadow.sy, rippleR, rippleR * 0.48, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      if (pFoodShadow) {
        const shadowScale = Math.max(0.25, 1.0 - heightAboveFloor * 0.24);
        ctx.fillStyle = `rgba(16, 185, 129, ${(0.28 * shadowScale).toFixed(3)})`;
        ctx.beginPath();
        ctx.ellipse(
          pFoodShadow.sx,
          pFoodShadow.sy,
          Math.max(3, creature.foodRadius * pFoodShadow.scale * 0.9 * shadowScale),
          Math.max(2, creature.foodRadius * pFoodShadow.scale * 0.45 * shadowScale),
          0,
          0,
          Math.PI * 2
        );
        ctx.fill();
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
          ctx.strokeStyle = (this.hoveredObject === 'food' || this.activeDragMode === 'food') ? '#38bdf8' : '#fbbf24';
          ctx.lineWidth = (this.hoveredObject === 'food' || this.activeDragMode === 'food') ? 2.8 : 1.8;
          ctx.stroke();
        }
        if ((this.hoveredObject === 'food' || this.activeDragMode === 'food') && pFoodShadow) {
          const grabR = Math.max(8, 0.30 * pFoodShadow.scale);
          ctx.strokeStyle = this.activeDragMode === 'food' ? '#38bdf8' : 'rgba(56, 189, 248, 0.78)';
          ctx.lineWidth = 2.0;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.ellipse(pFoodShadow.sx, pFoodShadow.sy, grabR, grabR * 0.48, 0, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // 2. Ground shadow of torso CoM and feet (+ interactive hover/grab drop-target ring)
    const pComShadow = proj(new Vec3(com.x, 0.004, com.z));
    if (pComShadow) {
      const liftFactor = Math.max(0.45, 1.0 - Math.max(0, com.y - 0.18) * 0.85);
      ctx.fillStyle = `rgba(0, 0, 0, ${(0.38 * liftFactor).toFixed(2)})`;
      ctx.beginPath();
      ctx.ellipse(
        pComShadow.sx,
        pComShadow.sy,
        Math.max(6, 0.30 * pComShadow.scale * liftFactor),
        Math.max(3, 0.16 * pComShadow.scale * liftFactor),
        0,
        0,
        Math.PI * 2
      );
      ctx.fill();

      if (this.hoveredObject === 'walker' || this.activeDragMode === 'walker') {
        const ringRad = Math.max(12, 0.38 * pComShadow.scale);
        ctx.strokeStyle = this.activeDragMode === 'walker' ? '#38bdf8' : 'rgba(56, 189, 248, 0.75)';
        ctx.lineWidth = 2.2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.ellipse(pComShadow.sx, pComShadow.sy, ringRad, ringRad * 0.50, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (creature.isGrabbed) {
          const pComAir = proj(com);
          if (pComAir) {
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
            ctx.lineWidth = 1.6;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(pComAir.sx, pComAir.sy);
            ctx.lineTo(pComShadow.sx, pComShadow.sy);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
    }

    // 3. Render Rigid Torso Structural Frame
    // In 'servo' mode (default in v2): renders a clean Central Robotic Spine Box (Z=0) + Cantilever Shoulder Axles
    // so that outboard 2-DoF legs have 100% unobstructed 360-degree sagittal rotation without intersecting frame tubes!
    // In 'muscle' mode (v1): renders all-to-all tensegrity/truss bones.
    const isServoMode = creature.actuatorMode !== 'muscle';
    ctx.strokeStyle = isLeader ? '#fde047' : creature.color;
    ctx.lineWidth = isLeader ? 2.6 : 1.6;
    if (isServoMode && creature.frameRefs) {
      const nTop = posOf(creature.frameRefs.top);
      const nBot = posOf(creature.frameRefs.bottom);
      const rawFront = posOf(creature.frameRefs.front);
      const rawTail = posOf(creature.frameRefs.tail);

      if (creature.isDuck) {
        // Open-Frame 3D Box / Octagonal Skeleton of Open_Duck_Mini_v2 (trunk_assembly without opaque armor)
        const { up } = creature.getTorsoFrame();
        const boxCenter = new Vec3(
          0.52 * nTop.x + 0.48 * nBot.x,
          0.52 * nTop.y + 0.48 * nBot.y,
          0.52 * nTop.z + 0.48 * nBot.z
        );
        const trunkPt = (fx, uy, rz) => new Vec3(
          boxCenter.x + forward.x * fx + up.x * uy + right.x * rz,
          boxCenter.y + forward.y * fx + up.y * uy + right.y * rz,
          boxCenter.z + forward.z * fx + up.z * uy + right.z * rz
        );
        const tfl = trunkPt(+0.056, +0.058, +0.058);
        const tfr = trunkPt(+0.056, +0.058, -0.058);
        const tbl = trunkPt(-0.056, +0.058, +0.058);
        const tbr = trunkPt(-0.056, +0.058, -0.058);
        const bfl = trunkPt(+0.050, -0.062, +0.052);
        const bfr = trunkPt(+0.050, -0.062, -0.052);
        const bbl = trunkPt(-0.050, -0.062, +0.052);
        const bbr = trunkPt(-0.050, -0.062, -0.052);

        // Subtle translucent glass/polycarbonate side panels so all internal servos & battery remain visible
        const drawQuad = (q1, q2, q3, q4, fillCol) => {
          const p1 = proj(q1), p2 = proj(q2), p3 = proj(q3), p4 = proj(q4);
          if (p1 && p2 && p3 && p4) {
            ctx.fillStyle = fillCol;
            ctx.beginPath();
            ctx.moveTo(p1.sx, p1.sy);
            ctx.lineTo(p2.sx, p2.sy);
            ctx.lineTo(p3.sx, p3.sy);
            ctx.lineTo(p4.sx, p4.sy);
            ctx.closePath();
            ctx.fill();
          }
        };
        drawQuad(bfl, bfr, bbr, bbl, 'rgba(15, 23, 42, 0.55)'); // Heavy battery pack keel floor
        drawQuad(tfl, tfr, bfr, bfl, 'rgba(56, 189, 248, 0.08)');

        // Structural open-frame cage bars
        ctx.strokeStyle = isLeader ? '#fde047' : creature.color;
        ctx.lineWidth = isLeader ? 2.4 : 1.7;
        const trunkEdges = [
          [tfl, tfr], [tfr, tbr], [tbr, tbl], [tbl, tfl],
          [bfl, bfr], [bfr, bbr], [bbr, bbl], [bbl, bfl],
          [tfl, bfl], [tfr, bfr], [tbl, bbl], [tbr, bbr],
          [nTop, nBot]
        ];
        for (const [a, b] of trunkEdges) {
          const pA = proj(a);
          const pB = proj(b);
          if (pA && pB) {
            ctx.beginPath();
            ctx.moveTo(pA.sx, pA.sy);
            ctx.lineTo(pB.sx, pB.sy);
            ctx.stroke();
          }
        }

        // Wide lateral hip brackets from trunk_assembly out to left/right hip_roll -> hip_pitch housings
        ctx.strokeStyle = isLeader ? '#38bdf8' : '#94a3b8';
        ctx.lineWidth = isLeader ? 3.2 : 2.2;
        for (const hip of creature.hipNodes.slice(0, 2)) {
          const hPos = posOf(hip);
          const mountPt = new Vec3(boxCenter.x, hPos.y, boxCenter.z);
          const pM = proj(mountPt);
          const pH = proj(hPos);
          if (pM && pH) {
            ctx.beginPath();
            ctx.moveTo(pM.sx, pM.sy);
            ctx.lineTo(pH.sx, pH.sy);
            ctx.stroke();
          }
        }
      } else {
        const nFront = rawFront;
        const nTail = rawTail;
        const spineCenter = new Vec3(
          0.5 * (nFront.x + nTail.x),
          0.5 * (nFront.y + nTail.y),
          0.5 * (nFront.z + nTail.z)
        );
        const spineAxis = Vec3.sub(nFront, nTail).normalize();

        // Central spine diamond truss (strictly in mid-plane, never crossing leg sweep planes)
        const spinePairs = [
          [nTop, nFront],
          [nTop, nTail],
          [nBot, nFront],
          [nBot, nTail],
          [nTop, nBot],
          [nFront, nTail]
        ];
        for (const [a, b] of spinePairs) {
          const pA = proj(a);
          const pB = proj(b);
          if (pA && pB) {
            ctx.beginPath();
            ctx.moveTo(pA.sx, pA.sy);
            ctx.lineTo(pB.sx, pB.sy);
            ctx.stroke();
          }
        }

        // Cantilever shoulder standoff axles from central spine out to each Hip Servo housing
        ctx.strokeStyle = isLeader ? '#38bdf8' : '#94a3b8';
        ctx.lineWidth = isLeader ? 3.2 : 2.0;
        for (const hip of creature.hipNodes) {
          const hPos = posOf(hip);
          const rel = Vec3.sub(hPos, spineCenter);
          const along = rel.dot(spineAxis);
          const mountPt = new Vec3(
            spineCenter.x + spineAxis.x * along,
            hPos.y,
            spineCenter.z + spineAxis.z * along
          );
          const pM = proj(mountPt);
          const pH = proj(hPos);
          if (pM && pH) {
            ctx.beginPath();
            ctx.moveTo(pM.sx, pM.sy);
            ctx.lineTo(pH.sx, pH.sy);
            ctx.stroke();
          }
        }
      }
    } else {
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
    }

    // 4. Render 2-Segment Articulated Legs (Hip -> Knee -> Foot)
    // In 'servo' mode (default in v2): uses physical kneeNodes + outboard cantilever offset + cylindrical PD servo motor drums
    // In 'muscle' mode (v1): draws active muscle tension lines + IK knee
    const L_femur = 0.16;
    const drawServoDrum = (pJoint, radiusPx, loadRatio, rotorAngleRad) => {
      if (!pJoint) return;
      const rOuter = Math.max(4.5, radiusPx);
      // Outer metallic servo housing
      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.arc(pJoint.sx, pJoint.sy, rOuter, 0, Math.PI * 2);
      ctx.fill();

      // Torque load ring (cyan/emerald at low torque -> amber -> crimson at peak torque N*m)
      const load = Math.max(0, Math.min(1, loadRatio || 0));
      ctx.strokeStyle = load > 0.65
        ? '#fb7185'
        : load > 0.32
          ? '#fbbf24'
          : '#38bdf8';
      ctx.lineWidth = Math.max(1.8, rOuter * 0.34);
      ctx.beginPath();
      ctx.arc(pJoint.sx, pJoint.sy, rOuter * 0.78, 0, Math.PI * 2);
      ctx.stroke();

      // Rotor angle indicator tick
      const tickAngle = rotorAngleRad || 0;
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = Math.max(1.3, rOuter * 0.22);
      ctx.beginPath();
      ctx.moveTo(pJoint.sx, pJoint.sy);
      ctx.lineTo(
        pJoint.sx + Math.cos(tickAngle) * (rOuter * 0.72),
        pJoint.sy + Math.sin(tickAngle) * (rOuter * 0.72)
      );
      ctx.stroke();

      // Center shaft bolt
      ctx.fillStyle = '#e2e8f0';
      ctx.beginPath();
      ctx.arc(pJoint.sx, pJoint.sy, Math.max(1.6, rOuter * 0.25), 0, Math.PI * 2);
      ctx.fill();
    };

    const visibleLegCount = creature.isDuck ? 2 : creature.footNodes.length;
    for (let i = 0; i < visibleLegCount; i++) {
      const hip = creature.hipNodes[i];
      const foot = creature.footNodes[i];
      if (!hip || !foot) continue;

      const hipPos = posOf(hip);
      const footPos = posOf(foot);

      let kneePos;
      if (isServoMode && creature.kneeNodes && creature.kneeNodes[i]) {
        // Real physical 3D knee node from XPBD rigid-link simulation
        kneePos = posOf(creature.kneeNodes[i]);
      } else {
        // Visual 2-link IK fallback for v1 bio-muscle mode
        const mid = new Vec3(
          0.5 * (hipPos.x + footPos.x),
          0.5 * (hipPos.y + footPos.y),
          0.5 * (hipPos.z + footPos.z)
        );
        const legVec = Vec3.sub(footPos, hipPos);
        const legLen = Math.max(0.05, legVec.length());
        const halfSpan = Math.min(L_femur * 0.98, legLen * 0.5);
        const kneeOut = Math.sqrt(Math.max(0.0004, L_femur * L_femur - halfSpan * halfSpan));
        const bendDir = new Vec3(
          forward.x * 0.65 + (hipPos.x - com.x) * 1.4,
          0.25,
          forward.z * 0.65 + (hipPos.z - com.z) * 1.4
        ).normalize();
        kneePos = new Vec3(
          mid.x + bendDir.x * kneeOut,
          mid.y + bendDir.y * kneeOut,
          mid.z + bendDir.z * kneeOut
        );
      }

      const pHip = proj(hipPos);
      const pKnee = proj(kneePos);
      const pFoot = proj(footPos);

      const isDuck5DoF = Boolean(creature.isDuck && creature.logicalDoFs.length >= 14);
      const yawDof = isDuck5DoF ? creature.logicalDoFs[i * 5 + 0] : null;
      const rollDof = isDuck5DoF ? creature.logicalDoFs[i * 5 + 1] : null;
      const strideDof = isDuck5DoF ? creature.logicalDoFs[i * 5 + 2] : creature.logicalDoFs[i * 2 + 1];
      const liftDof = isDuck5DoF ? creature.logicalDoFs[i * 5 + 3] : creature.logicalDoFs[i * 2];
      const ankleDof = isDuck5DoF ? creature.logicalDoFs[i * 5 + 4] : null;

      // In 'muscle' (v1) mode only: draw active muscle lines colored by instantaneous force
      if (!isServoMode && isFocused && liftDof && strideDof) {
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
        if (isServoMode) {
          // Outboard cantilever servo shaft offset (like Open_Duck_Mini / ODRI Solo-8 / Unitree)
          // so the femur & tibia rotate on the outer face of the Hip Servo Drum without colliding with the chassis!
          const latRel = Vec3.sub(hipPos, com).dot(right);
          const outSign = Math.abs(latRel) > 0.04 ? Math.sign(latRel) : 1.0;
          const outOffset = 0.038 * outSign;
          const hipOutPos = new Vec3(
            hipPos.x + right.x * outOffset,
            hipPos.y + right.y * outOffset,
            hipPos.z + right.z * outOffset
          );
          const kneeOutPos = new Vec3(
            kneePos.x + right.x * outOffset,
            kneePos.y + right.y * outOffset,
            kneePos.z + right.z * outOffset
          );
          const pHipOut = proj(hipOutPos) || pHip;
          const pKneeOut = proj(kneeOutPos) || pKnee;

          // Cantilever rotor shaft from Hip Roll housing to outboard Hip Pitch hub
          ctx.strokeStyle = '#94a3b8';
          ctx.lineWidth = Math.max(3.0, 0.032 * pHip.scale);
          ctx.beginPath();
          ctx.moveTo(pHip.sx, pHip.sy);
          ctx.lineTo(pHipOut.sx, pHipOut.sy);
          ctx.stroke();

          // Rigid Robotic Link Casing (outer dark armor + bright structural core)
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = Math.max(4.5, 0.048 * pKneeOut.scale);
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.beginPath();
          ctx.moveTo(pHipOut.sx, pHipOut.sy);
          ctx.lineTo(pKneeOut.sx, pKneeOut.sy);
          ctx.lineTo(pFoot.sx, pFoot.sy);
          ctx.stroke();

          ctx.strokeStyle = isLeader ? '#e2e8f0' : '#cbd5e1';
          ctx.lineWidth = Math.max(2.2, 0.024 * pKneeOut.scale);
          ctx.beginPath();
          ctx.moveTo(pHipOut.sx, pHipOut.sy);
          ctx.lineTo(pKneeOut.sx, pKneeOut.sy);
          ctx.lineTo(pFoot.sx, pFoot.sy);
          ctx.stroke();

          // For 14-DoF Open_Duck_Mini: render the 3 distinct Hip Servo Drums (hip_yaw, hip_roll, hip_pitch)
          if (isDuck5DoF) {
            const { up } = creature.getTorsoFrame();
            const hipYawPos = new Vec3(
              hipPos.x - right.x * (0.048 * outSign) + up.x * 0.038,
              hipPos.y - right.y * (0.048 * outSign) + up.y * 0.038,
              hipPos.z - right.z * (0.048 * outSign) + up.z * 0.038
            );
            const pHipYaw = proj(hipYawPos);
            if (pHipYaw) {
              ctx.strokeStyle = '#38bdf8';
              ctx.lineWidth = Math.max(2.0, 0.020 * pHip.scale);
              ctx.beginPath();
              ctx.moveTo(pHipYaw.sx, pHipYaw.sy);
              ctx.lineTo(pHip.sx, pHip.sy);
              ctx.stroke();

              const yawServo = yawDof ? yawDof.primary : null;
              const yawLoad = yawServo ? (yawServo.torqueLoadRatio ?? 0.15) : 0.15;
              const yawAng = yawServo ? (yawServo.jointAngleRad || 0) : 0;
              drawServoDrum(pHipYaw, 0.032 * pHipYaw.scale, yawLoad, yawAng);
            }

            // Outboard hip_pitch Servo Drum
            const pitchServo = strideDof ? strideDof.primary : null;
            const pitchLoad = pitchServo ? (pitchServo.torqueLoadRatio ?? 0.2) : 0.2;
            const pitchAng = Math.atan2(pKneeOut.sy - pHipOut.sy, pKneeOut.sx - pHipOut.sx);
            drawServoDrum(pHipOut, 0.038 * pHipOut.scale, pitchLoad, pitchAng);
          }

          // Knee Revolute PD Servo Motor / Coaxial Pulley Drum (knee_pitch)
          const kneeServo = liftDof ? liftDof.primary : null;
          const kneeLoad = kneeServo ? (kneeServo.torqueLoadRatio ?? Math.min(1, Math.abs(kneeServo.appliedForce || 0) / 90.0)) : 0;
          const kneeAng = Math.atan2(pFoot.sy - pKneeOut.sy, pFoot.sx - pKneeOut.sx);
          drawServoDrum(pKneeOut, 0.042 * pKneeOut.scale, kneeLoad, kneeAng);
        } else {
          ctx.strokeStyle = isLeader ? '#f8fafc' : '#cbd5e1';
          ctx.lineWidth = Math.max(2.0, 0.026 * pKnee.scale);
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(pHip.sx, pHip.sy);
          ctx.lineTo(pKnee.sx, pKnee.sy);
          ctx.lineTo(pFoot.sx, pFoot.sy);
          ctx.stroke();

          // Knee joint cap (v1 bio-muscle mode)
          ctx.fillStyle = '#fb923c';
          ctx.beginPath();
          ctx.arc(pKnee.sx, pKnee.sy, Math.max(2.5, 0.028 * pKnee.scale), 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Foot contact ring on floor
      const pFootGround = proj(new Vec3(footPos.x, 0.005, footPos.z));
      const nForce = Math.max(foot.renderNormalForce || 0, foot.normalForce || 0);
      if ((foot.inContact || nForce > 0.15) && pFootGround) {
        const ringRad = Math.max(4, Math.min(22, (0.07 + nForce * 0.004) * pFootGround.scale));
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.ellipse(pFootGround.sx, pFootGround.sy, ringRad, ringRad * 0.48, 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      // 4b. Render 3D Yellow Trapezoidal Wedge Boot ("объёмный жёлтый башмак-ласта / утюжок", foot_assembly) + Ankle Pitch Servo Drum for Open_Duck_Mini_v2
      if (creature.webbedFeet && creature.webbedFeet[i] && pFoot) {
        const wf = creature.webbedFeet[i];
        const posIn = posOf(wf.toeInner);  // Front toe pad (+0.16 m ahead of ankle)
        const posOut = posOf(wf.toeOuter); // Rear heel pad (-0.15 m behind ankle)
        const { up: tUp } = creature.getTorsoFrame();

        // Foot longitudinal and lateral unit axes derived from rear heel (posOut) -> front toe (posIn)
        const footLong = Vec3.sub(posIn, posOut);
        footLong.y *= 0.25;
        const fDir = footLong.length() > 1e-4 ? footLong.normalize() : forward;
        const rDir = Vec3.cross(tUp, fDir).normalize();
        const uDir = Vec3.cross(fDir, rDir).normalize();

        // Bottom flat trapezoidal sole corners (wider at front toe, narrower at rear heel)
        const fl = new Vec3(posIn.x + rDir.x * 0.064, posIn.y, posIn.z + rDir.z * 0.064);
        const fr = new Vec3(posIn.x - rDir.x * 0.064, posIn.y, posIn.z - rDir.z * 0.064);
        const bl = new Vec3(posOut.x + rDir.x * 0.048, posOut.y, posOut.z + rDir.z * 0.048);
        const br = new Vec3(posOut.x - rDir.x * 0.048, posOut.y, posOut.z - rDir.z * 0.048);

        // Raised top ankle collar around center ankle joint (footPos)
        const collarC = new Vec3(
          footPos.x + uDir.x * 0.036,
          footPos.y + uDir.y * 0.036,
          footPos.z + uDir.z * 0.036
        );
        const tfl = new Vec3(collarC.x + fDir.x * 0.034 + rDir.x * 0.030, collarC.y, collarC.z + fDir.z * 0.034 + rDir.z * 0.030);
        const tfr = new Vec3(collarC.x + fDir.x * 0.034 - rDir.x * 0.030, collarC.y, collarC.z + fDir.z * 0.034 - rDir.z * 0.030);
        const tbl = new Vec3(collarC.x - fDir.x * 0.032 + rDir.x * 0.028, collarC.y, collarC.z - fDir.z * 0.032 + rDir.z * 0.028);
        const tbr = new Vec3(collarC.x - fDir.x * 0.032 - rDir.x * 0.028, collarC.y, collarC.z - fDir.z * 0.032 - rDir.z * 0.028);

        const pFl = proj(fl), pFr = proj(fr), pBl = proj(bl), pBr = proj(br);
        const pTfl = proj(tfl), pTfr = proj(tfr), pTbl = proj(tbl), pTbr = proj(tbr);
        const pIn = proj(posIn);
        const pOut = proj(posOut);
        const anyWebContact = foot.inContact || wf.toeInner.inContact || wf.toeOuter.inContact;

        if (pFl && pFr && pBl && pBr && pTfl && pTfr && pTbl && pTbr) {
          // 1. Dark TPU non-slip sole plate on bottom
          ctx.fillStyle = anyWebContact
            ? 'rgba(30, 41, 59, 0.82)'
            : 'rgba(51, 65, 85, 0.65)';
          ctx.beginPath();
          ctx.moveTo(pFl.sx, pFl.sy);
          ctx.lineTo(pFr.sx, pFr.sy);
          ctx.lineTo(pBr.sx, pBr.sy);
          ctx.lineTo(pBl.sx, pBl.sy);
          ctx.closePath();
          ctx.fill();

          // 2. Semi-transparent yellow 3D wedge upper slopes (Open_Duck_Mini_v2 signature yellow shoe)
          const wedgeFill = anyWebContact
            ? 'rgba(250, 204, 21, 0.50)'
            : 'rgba(234, 179, 8, 0.36)';
          const drawShoeFace = (a, b, c, d, col) => {
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.moveTo(a.sx, a.sy);
            ctx.lineTo(b.sx, b.sy);
            ctx.lineTo(c.sx, c.sy);
            ctx.lineTo(d.sx, d.sy);
            ctx.closePath();
            ctx.fill();
          };
          drawShoeFace(pFl, pFr, pTfr, pTfl, wedgeFill); // Front sloped vamp
          drawShoeFace(pBl, pBr, pTbr, pTbl, wedgeFill); // Rear heel slope
          drawShoeFace(pFl, pBl, pTbl, pTfl, 'rgba(250, 204, 21, 0.38)'); // Left side panel
          drawShoeFace(pFr, pBr, pTbr, pTfr, 'rgba(250, 204, 21, 0.38)'); // Right side panel

          // 3. Structural wireframe edges of the 3D wedge shoe (so internal ankle servo stays clearly visible!)
          ctx.strokeStyle = anyWebContact ? '#fde047' : '#eab308';
          ctx.lineWidth = Math.max(1.6, 0.015 * pFoot.scale);
          const shoeEdges = [
            [pFl, pFr], [pFr, pBr], [pBr, pBl], [pBl, pFl],
            [pTfl, pTfr], [pTfr, pTbr], [pTbr, pTbl], [pTbl, pTfl],
            [pFl, pTfl], [pFr, pTfr], [pBl, pTbl], [pBr, pTbr]
          ];
          for (const [eA, eB] of shoeEdges) {
            ctx.beginPath();
            ctx.moveTo(eA.sx, eA.sy);
            ctx.lineTo(eB.sx, eB.sy);
            ctx.stroke();
          }

          // Contact indicator LEDs at front toe, center ankle, and rear heel
          if (pIn && pOut) {
            for (const [pPad, inCont] of [[pIn, wf.toeInner.inContact], [pFoot, foot.inContact], [pOut, wf.toeOuter.inContact]]) {
              ctx.fillStyle = inCont ? '#4ade80' : '#f59e0b';
              ctx.beginPath();
              ctx.arc(pPad.sx, pPad.sy, Math.max(2.4, 0.020 * pFoot.scale), 0, Math.PI * 2);
              ctx.fill();
            }
          }

          // 4. Ankle Pitch Servo Drum right inside the center ankle apex (ankle_pitch)
          const ankleServo = (ankleDof && ankleDof.primary) ? ankleDof.primary : (strideDof ? strideDof.primary : null);
          const ankleLoad = ankleServo ? (ankleServo.torqueLoadRatio ?? 0.2) : 0.2;
          const ankleAng = pIn ? Math.atan2(pIn.sy - pFoot.sy, pIn.sx - pFoot.sx) : 0;
          drawServoDrum(pFoot, 0.036 * pFoot.scale, ankleLoad, ankleAng);
        }
      }
    }

    // 4c. Render Open-Frame 4-DoF Neck & Massive BDX / Open_Duck_Mini_v2 Head Assembly (neck_pitch, head_pitch, head_yaw, head_roll, 2 headlight-eyes, yellow counterweight cap, 2 antennas)
    if (creature.isDuck && creature.frameRefs) {
      const { up } = creature.getTorsoFrame();
      const nTop = posOf(creature.frameRefs.top);
      const nBot = posOf(creature.frameRefs.bottom);

      // Draw seating bumper ring at torsoBottom ("попа" / pelvic seat)
      const pBot = proj(nBot);
      if (pBot) {
        const seatContact = creature.frameRefs.bottom.inContact || nBot.y <= 0.075;
        ctx.fillStyle = seatContact ? '#4ade80' : '#fbbf24';
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(pBot.sx, pBot.sy, Math.max(4.0, 0.044 * pBot.scale), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }

      const neckPitchServo = creature.logicalDoFs[10] ? creature.logicalDoFs[10].primary : null;
      const headPitchServo = creature.logicalDoFs[11] ? creature.logicalDoFs[11].primary : null;
      const headYawServo = creature.logicalDoFs[12] ? creature.logicalDoFs[12].primary : null;
      const headRollServo = creature.logicalDoFs[13] ? creature.logicalDoFs[13].primary : null;

      const neckLoad = neckPitchServo ? (neckPitchServo.torqueLoadRatio ?? 0.2) : 0.2;
      const headPitchLoad = headPitchServo ? (headPitchServo.torqueLoadRatio ?? 0.2) : 0.2;
      const headYawLoad = headYawServo ? (headYawServo.torqueLoadRatio ?? 0.2) : 0.2;
      const headRollLoad = headRollServo ? (headRollServo.torqueLoadRatio ?? 0.2) : 0.2;

      const headPitchCmd = headPitchServo ? (headPitchServo.targetAction || 0) : 0;
      const headYawCmd = headYawServo ? (headYawServo.targetAction || 0) : 0;
      const headRollCmd = headRollServo ? (headRollServo.targetAction || 0) : 0;

      const headPos = creature.headNode
        ? posOf(creature.headNode)
        : new Vec3(
            nTop.x + forward.x * 0.09 + up.x * 0.16,
            nTop.y + forward.y * 0.09 + up.y * 0.16,
            nTop.z + forward.z * 0.09 + up.z * 0.16
          );

      // Local orthonormal head frame rotated by head_yaw, head_pitch, head_roll
      const yawA = headYawCmd * 0.45;
      const pitchA = -headPitchCmd * 0.32;
      const rollA = headRollCmd * 0.30;
      const cosY = Math.cos(yawA);
      const sinY = Math.sin(yawA);
      const hFwd0 = new Vec3(
        forward.x * cosY + right.x * sinY,
        forward.y * cosY + right.y * sinY,
        forward.z * cosY + right.z * sinY
      ).normalize();
      const hRight0 = new Vec3(
        right.x * cosY - forward.x * sinY,
        right.y * cosY - forward.y * sinY,
        right.z * cosY - forward.z * sinY
      ).normalize();
      const hFwd = new Vec3(
        hFwd0.x * Math.cos(pitchA) + up.x * Math.sin(pitchA),
        hFwd0.y * Math.cos(pitchA) + up.y * Math.sin(pitchA),
        hFwd0.z * Math.cos(pitchA) + up.z * Math.sin(pitchA)
      ).normalize();
      const hRight = new Vec3(
        hRight0.x * Math.cos(rollA) + up.x * Math.sin(rollA),
        hRight0.y * Math.cos(rollA) + up.y * Math.sin(rollA),
        hRight0.z * Math.cos(rollA) + up.z * Math.sin(rollA)
      ).normalize();
      const hUp = Vec3.cross(hRight, hFwd).normalize();

      const pTop = proj(nTop);
      const pHead = proj(headPos);

      if (pTop && pHead) {
        // Metallic curved/inclined neck column (nTop -> headPos)
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = Math.max(5.5, 0.058 * pTop.scale);
        ctx.beginPath();
        ctx.moveTo(pTop.sx, pTop.sy);
        ctx.lineTo(pHead.sx, pHead.sy);
        ctx.stroke();

        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = Math.max(2.8, 0.028 * pTop.scale);
        ctx.beginPath();
        ctx.moveTo(pTop.sx, pTop.sy);
        ctx.lineTo(pHead.sx, pHead.sy);
        ctx.stroke();

        // Servo #11: neck_pitch drum at base of neck (nTop)
        const neckAng = Math.atan2(pHead.sy - pTop.sy, pHead.sx - pTop.sx);
        drawServoDrum(pTop, 0.042 * pTop.scale, neckLoad, neckAng);

        // Massive Open-Frame BDX / Open_Duck_Mini_v2 Head Truss (proportionally wider & longer than torso, like the real robot!)
        const headPt = (fx, uy, rz) => new Vec3(
          headPos.x + hFwd.x * fx + hUp.x * uy + hRight.x * rz,
          headPos.y + hFwd.y * fx + hUp.y * uy + hRight.y * rz,
          headPos.z + hFwd.z * fx + hUp.z * uy + hRight.z * rz
        );

        const flt = headPt(+0.130, +0.054, +0.088);
        const frt = headPt(+0.130, +0.054, -0.088);
        const flb = headPt(+0.145, -0.040, +0.088);
        const frb = headPt(+0.145, -0.040, -0.088);
        const blt = headPt(-0.105, +0.056, +0.078);
        const brt = headPt(-0.105, +0.056, -0.078);
        const blb = headPt(-0.105, -0.036, +0.078);
        const brb = headPt(-0.105, -0.036, -0.078);

        // Rear Rounded Yellow Counterweight Cap (iconic yellow back-of-head mass on Open_Duck_Mini_v2!)
        const cwl = headPt(-0.138, +0.012, +0.056);
        const cwr = headPt(-0.138, +0.012, -0.056);
        const pBlt = proj(blt), pBrt = proj(brt), pBlb = proj(blb), pBrb = proj(brb);
        const pCwl = proj(cwl), pCwr = proj(cwr);
        if (pBlt && pBrt && pBlb && pBrb && pCwl && pCwr) {
          ctx.fillStyle = 'rgba(250, 204, 21, 0.72)';
          ctx.strokeStyle = '#fde047';
          ctx.lineWidth = Math.max(1.5, 0.014 * pHead.scale);
          ctx.beginPath();
          ctx.moveTo(pBlt.sx, pBlt.sy);
          ctx.lineTo(pBrt.sx, pBrt.sy);
          ctx.lineTo(pCwr.sx, pCwr.sy);
          ctx.lineTo(pBrb.sx, pBrb.sy);
          ctx.lineTo(pBlb.sx, pBlb.sy);
          ctx.lineTo(pCwl.sx, pCwl.sy);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }

        // Subtle translucent glass/polycarbonate visor panels on the massive head block
        const pFlt = proj(flt), pFrt = proj(frt), pFlb = proj(flb), pFrb = proj(frb);
        if (pFlt && pFrt && pFlb && pFrb && pBlt && pBrt) {
          ctx.fillStyle = 'rgba(248, 250, 252, 0.14)';
          ctx.beginPath();
          ctx.moveTo(pFlt.sx, pFlt.sy);
          ctx.lineTo(pFrt.sx, pFrt.sy);
          ctx.lineTo(pBrt.sx, pBrt.sy);
          ctx.lineTo(pBlt.sx, pBlt.sy);
          ctx.closePath();
          ctx.fill();

          ctx.fillStyle = 'rgba(15, 23, 42, 0.42)';
          ctx.beginPath();
          ctx.moveTo(pFlt.sx, pFlt.sy);
          ctx.lineTo(pFrt.sx, pFrt.sy);
          ctx.lineTo(pFrb.sx, pFrb.sy);
          ctx.lineTo(pFlb.sx, pFlb.sy);
          ctx.closePath();
          ctx.fill();
        }

        // Yellow-Orange Duck Bill / Lower Visor Lip protruding from lower front of head truss
        const beakL = headPt(+0.205, -0.032, +0.052);
        const beakR = headPt(+0.205, -0.032, -0.052);
        const pBeakL = proj(beakL);
        const pBeakR = proj(beakR);
        if (pBeakL && pBeakR && pFlb && pFrb) {
          ctx.fillStyle = 'rgba(250, 204, 21, 0.82)';
          ctx.strokeStyle = '#fde047';
          ctx.lineWidth = Math.max(1.5, 0.015 * pHead.scale);
          ctx.beginPath();
          ctx.moveTo(pFlb.sx, pFlb.sy);
          ctx.lineTo(pBeakL.sx, pBeakL.sy);
          ctx.lineTo(pBeakR.sx, pBeakR.sy);
          ctx.lineTo(pFrb.sx, pFrb.sy);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }

        // Draw open-frame head structural truss tubes
        ctx.strokeStyle = isLeader ? '#fde047' : '#e2e8f0';
        ctx.lineWidth = Math.max(2.1, 0.021 * pHead.scale);
        const headEdges = [
          [flt, frt], [frt, frb], [frb, flb], [flb, flt],
          [blt, brt], [brt, brb], [brb, blb], [blb, blt],
          [flt, blt], [frt, brt], [flb, blb], [frb, brb]
        ];
        for (const [eA, eB] of headEdges) {
          const pA = proj(eA);
          const pB = proj(eB);
          if (pA && pB) {
            ctx.beginPath();
            ctx.moveTo(pA.sx, pA.sy);
            ctx.lineTo(pB.sx, pB.sy);
            ctx.stroke();
          }
        }

        // Servos #12, #13, #14 clearly visible inside the open-frame head gimbal (head_pitch, head_yaw, head_roll)
        const pGimbalPitch = proj(headPt(-0.035, 0.0, 0.0));
        const pGimbalRoll = proj(headPt(+0.045, +0.012, 0.0));
        drawServoDrum(pHead, 0.038 * pHead.scale, headYawLoad, yawA);
        if (pGimbalPitch) drawServoDrum(pGimbalPitch, 0.030 * pGimbalPitch.scale, headPitchLoad, pitchA);
        if (pGimbalRoll) drawServoDrum(pGimbalRoll, 0.028 * pGimbalRoll.scale, headRollLoad, rollA);

        // Two Rear Vertical Antennas (iconic Open_Duck_Mini_v2 / BDX silhouette!)
        for (const sideS of [+1, -1]) {
          const antBase = headPt(-0.082, +0.056, 0.056 * sideS);
          const antTip = headPt(-0.095, +0.205, 0.064 * sideS);
          const pAB = proj(antBase);
          const pAT = proj(antTip);
          if (pAB && pAT) {
            ctx.strokeStyle = '#cbd5e1';
            ctx.lineWidth = Math.max(1.8, 0.016 * pHead.scale);
            ctx.beginPath();
            ctx.moveTo(pAB.sx, pAB.sy);
            ctx.lineTo(pAT.sx, pAT.sy);
            ctx.stroke();

            ctx.fillStyle = '#38bdf8';
            ctx.beginPath();
            ctx.arc(pAT.sx, pAT.sy, Math.max(2.4, 0.019 * pAT.scale), 0, Math.PI * 2);
            ctx.fill();
          }
        }

        // Two Large Round Front Headlight-Eyes on the sloped front visor of the head block
        for (const sideS of [+1, -1]) {
          const eyePos = headPt(+0.140, +0.008, 0.054 * sideS);
          const pEye = proj(eyePos);
          if (pEye) {
            const eyeR = Math.max(4.4, 0.036 * pEye.scale);
            // Dark bezel ring
            ctx.fillStyle = '#0f172a';
            ctx.beginPath();
            ctx.arc(pEye.sx, pEye.sy, eyeR * 1.18, 0, Math.PI * 2);
            ctx.fill();
            // Glowing white/cyan LED headlight ring
            ctx.fillStyle = '#f8fafc';
            ctx.beginPath();
            ctx.arc(pEye.sx, pEye.sy, eyeR, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = Math.max(1.6, eyeR * 0.28);
            ctx.stroke();
            // Pupil highlight
            ctx.fillStyle = '#0284c7';
            ctx.beginPath();
            ctx.arc(pEye.sx, pEye.sy, eyeR * 0.42, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    // 5. Render 3D Nodes (Torso, Hips / Hip Servo Drums, Head Sensor, Feet)
    for (const node of creature.body.nodes) {
      if (node.role === 'knee' || node.role === 'foot_pad' || node.role === 'duck_head') continue;
      if (creature.isDuck) {
        if (
          node === creature.frameRefs.top ||
          node === creature.frameRefs.bottom ||
          node === creature.frameRefs.front ||
          node === creature.frameRefs.tail
        ) {
          continue;
        }
      }
      const p = proj(posOf(node));
      if (!p) continue;
      const rPx = Math.max(3, node.radius * p.scale * 0.85);

      if (isServoMode && node.role === 'hip') {
        const hipIdx = creature.hipNodes.indexOf(node);
        if (hipIdx >= 0) {
          const isDuck5DoF = Boolean(creature.isDuck && creature.logicalDoFs.length >= 14);
          const hipDof = isDuck5DoF
            ? creature.logicalDoFs[hipIdx * 5 + 1] // hip_roll drum at hipPos
            : creature.logicalDoFs[hipIdx * 2 + 1];
          const hipServo = hipDof ? hipDof.primary : null;
          const hipLoad = hipServo ? (hipServo.torqueLoadRatio ?? Math.min(1, Math.abs(hipServo.appliedForce || 0) / 85.0)) : 0;
          const kneeNode = creature.kneeNodes && creature.kneeNodes[hipIdx];
          const pKnee = kneeNode ? proj(posOf(kneeNode)) : null;
          const hipAng = pKnee ? Math.atan2(pKnee.sy - p.sy, pKnee.sx - p.sx) : 0;
          drawServoDrum(p, rPx * 1.08, hipLoad, hipAng);
          continue;
        }
      }

      if (node.role === 'foot') {
        if (creature.isDuck) continue; // Drawn as Ankle Servo Drum + Webbed Duck Foot in 4b
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
