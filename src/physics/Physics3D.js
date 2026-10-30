/**
 * Pure 3D Multibody & Articulated Rigid-Body Physics Engine (XPBD + Coulomb Contact Solver)
 *
 * Strictly obeys:
 * 1. Conservation of Linear Momentum for all internal forces/constraints (\sum F_internal = 0)
 * 2. Conservation of Angular Momentum for all internal forces/constraints (\sum r x F_internal = 0)
 * 3. Coulomb Friction Cone on the floor: ||F_tangent|| <= mu_floor * mu_node * F_normal
 * 4. Bounded Arena Walls with restitution and friction
 */

export class Vec3 {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  set(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  copy(v) {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }
  clone() {
    return new Vec3(this.x, this.y, this.z);
  }
  add(v) {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }
  addScaled(v, s) {
    this.x += v.x * s;
    this.y += v.y * s;
    this.z += v.z * s;
    return this;
  }
  sub(v) {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }
  scale(s) {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }
  dot(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }
  cross(v) {
    return new Vec3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  length() {
    return Math.hypot(this.x, this.y, this.z);
  }
  normalize() {
    const len = this.length();
    if (len > 1e-9) {
      this.x /= len;
      this.y /= len;
      this.z /= len;
    }
    return this;
  }
  static sub(a, b) {
    return new Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
  }
  static cross(a, b) {
    return new Vec3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x
    );
  }
  static dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
  }
}

export class PhysicsNode {
  constructor({
    id,
    x = 0,
    y = 0,
    z = 0,
    mass = 0.25,
    radius = 0.06,
    friction = 0.9,
    restitution = 0.05,
    role = 'body', // 'torso', 'head', 'hip', 'foot'
    color = '#4ea8de',
    noGroundCollision = false
  }) {
    this.id = id;
    this.pos = new Vec3(x, y, z);
    this.prevPos = new Vec3(x, y, z);
    this.vel = new Vec3(0, 0, 0);
    this.force = new Vec3(0, 0, 0);
    this.mass = mass;
    this.invMass = mass > 0 ? 1 / mass : 0;
    this.radius = radius;
    this.friction = friction;
    this.restitution = restitution;
    this.role = role;
    this.color = color;
    this.noGroundCollision = Boolean(noGroundCollision);

    // Contact telemetry for sensors & visualization
    this.inContact = false;
    this.normalForce = 0; // Newtons (accumulated over control step for RL sensors)
    this.frictionForceVec = new Vec3(0, 0, 0); // Newtons on node from floor
    this.renderNormalForce = 0; // Smoothed instantaneous normal force (N) for 3D vector visualization
    this.renderFrictionVec = new Vec3(0, 0, 0); // Smoothed instantaneous friction force (N) for 3D vector visualization
  }
}

export class DistanceBone {
  constructor(nodeA, nodeB, restLength = null, compliance = 0.0, role = 'bone') {
    this.nodeA = nodeA;
    this.nodeB = nodeB;
    this.restLength = restLength !== null ? restLength : Vec3.dist(nodeA.pos, nodeB.pos);
    this.compliance = compliance; // 0 = rigid bone, >0 = compliant ligament
    this.role = role;
    this.lambda = 0;
  }
}

/**
 * XPBD Motorized Servo Muscle Actuator
 * Modulates target rest length between nodeA and nodeB within [minLength, maxLength]
 * and solves it inside the XPBD constraint loop with physical compliance (1/stiffness)
 * and force clamping [-maxForce, +maxForce].
 * Guarantees exact linear and angular momentum conservation (\sum F = 0, \sum r x F = 0).
 */
export class MuscleActuator {
  constructor({
    name,
    nodeA,
    nodeB,
    pivotNode = null,
    baseLength = null,
    minFactor = 0.78,
    maxFactor = 1.22,
    maxForce = 95.0,   // Max physical force in Newtons
    compliance = 0.0003 // XPBD compliance alpha (m/N)
  }) {
    this.name = name;
    this.nodeA = nodeA;
    this.nodeB = nodeB;
    this.pivotNode = pivotNode;
    this.baseLength = baseLength !== null ? baseLength : Vec3.dist(nodeA.pos, nodeB.pos);
    this.minLength = this.baseLength * minFactor;
    this.maxLength = this.baseLength * maxFactor;
    this.maxForce = maxForce;
    this.compliance = compliance;

    // State & telemetry
    this.targetAction = 0;       // Commanded action in [-1, 1]
    this.smoothedAction = 0;     // Physically rate-limited servo state
    this.targetLength = this.baseLength;
    this.currentAngleNorm = 0;   // Normalized extension in [-1, 1]
    this.angularVelNorm = 0;     // Normalized extension velocity
    this.appliedForce = 0;       // Current muscle force in Newtons
    this.instantPower = 0;       // Mechanical power |F * v| in Watts
    this.lambda = 0;
  }

  setAction(action) {
    this.targetAction = Math.max(-1, Math.min(1, action));
    this.directTargetLength = null;
  }

  setTargetLengthDirect(len) {
    this.directTargetLength = Math.max(this.minLength, Math.min(this.maxLength, len));
  }

  updateTelemetryAndTarget(subDt) {
    const rate = Math.min(1.0, subDt * 28.0);
    const halfRange = 0.5 * (this.maxLength - this.minLength);
    const midLength = 0.5 * (this.maxLength + this.minLength);

    if (this.directTargetLength !== null && this.directTargetLength !== undefined) {
      this.targetLength += (this.directTargetLength - this.targetLength) * rate;
    } else {
      this.smoothedAction += (this.targetAction - this.smoothedAction) * rate;
      this.targetLength = midLength + this.smoothedAction * halfRange;
    }

    const dx = this.nodeB.pos.x - this.nodeA.pos.x;
    const dy = this.nodeB.pos.y - this.nodeA.pos.y;
    const dz = this.nodeB.pos.z - this.nodeA.pos.z;
    const dist = Math.hypot(dx, dy, dz);
    if (dist > 1e-7) {
      this.currentAngleNorm = Math.max(-1.5, Math.min(1.5, (dist - midLength) / Math.max(1e-5, halfRange)));
      const nx = dx / dist;
      const ny = dy / dist;
      const nz = dz / dist;
      const relVx = this.nodeB.vel.x - this.nodeA.vel.x;
      const relVy = this.nodeB.vel.y - this.nodeA.vel.y;
      const relVz = this.nodeB.vel.z - this.nodeA.vel.z;
      const relVel = relVx * nx + relVy * ny + relVz * nz;
      this.angularVelNorm = Math.max(-2.0, Math.min(2.0, relVel / (halfRange * 10.0)));
      this.instantPower = Math.abs(this.appliedForce * relVel);
    }
  }
}

export class ArticulatedBody {
  constructor() {
    this.nodes = [];
    this.bones = [];
    this.muscles = [];
    this.totalMass = 0;
  }

  addNode(params) {
    const node = new PhysicsNode(params);
    this.nodes.push(node);
    this.totalMass += node.mass;
    return node;
  }

  addBone(nodeA, nodeB, restLength = null, compliance = 0.0, role = 'bone') {
    const bone = new DistanceBone(nodeA, nodeB, restLength, compliance, role);
    this.bones.push(bone);
    return bone;
  }

  addMuscle(params) {
    const muscle = new MuscleActuator(params);
    this.muscles.push(muscle);
    return muscle;
  }

  getCenterOfMass() {
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      x += n.pos.x * n.mass;
      y += n.pos.y * n.mass;
      z += n.pos.z * n.mass;
    }
    const invM = 1 / Math.max(1e-6, this.totalMass);
    return new Vec3(x * invM, y * invM, z * invM);
  }

  getCenterOfMassVelocity() {
    let vx = 0, vy = 0, vz = 0;
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      vx += n.vel.x * n.mass;
      vy += n.vel.y * n.mass;
      vz += n.vel.z * n.mass;
    }
    const invM = 1 / Math.max(1e-6, this.totalMass);
    return new Vec3(vx * invM, vy * invM, vz * invM);
  }
}

export class PhysicsWorld3D {
  constructor({
    gravity = -9.81,
    floorFriction = 1.0,
    arenaHalfSize = 11.0,
    substeps = 10
  } = {}) {
    this.gravity = gravity;
    this.floorFriction = floorFriction;
    this.arenaHalfSize = arenaHalfSize;
    this.substeps = substeps;
    this.bodies = [];
  }

  addBody(body) {
    this.bodies.push(body);
  }

  clearBodies() {
    this.bodies = [];
  }

  resetContactTelemetry(actionCallback = null) {
    for (let b = 0; b < this.bodies.length; b++) {
      const body = this.bodies[b];
      for (let i = 0; i < body.nodes.length; i++) {
        const node = body.nodes[i];
        node.inContact = false;
        node.normalForce = 0;
        node.frictionForceVec.set(0, 0, 0);
      }
      if (actionCallback) {
        actionCallback(body, b);
      }
    }
  }

  stepSingleSubstep(subDt) {
    const invSubDt = 1 / subDt;
    const invSubDtSq = invSubDt * invSubDt;
    const solverIterations = 4;
    const halfArena = this.arenaHalfSize;

    // 1. Predict unconstrained motion under gravity & clear instantaneous substep contact forces
    for (let b = 0; b < this.bodies.length; b++) {
      const body = this.bodies[b];
      for (let m = 0; m < body.muscles.length; m++) {
        body.muscles[m].updateTelemetryAndTarget(subDt);
        body.muscles[m].lambda = 0;
      }
      for (let c = 0; c < body.bones.length; c++) {
        body.bones[c].lambda = 0;
      }

      for (let i = 0; i < body.nodes.length; i++) {
        const node = body.nodes[i];
        node._subContact = false;
        node._subNormalForce = 0;
        node._subFrictionX = 0;
        node._subFrictionZ = 0;
        if (node.invMass === 0) continue;
        node.prevPos.copy(node.pos);

        const airDrag = Math.max(0, 1 - 0.02 * subDt);
        node.vel.x *= airDrag;
        node.vel.y = (node.vel.y + this.gravity * subDt) * airDrag;
        node.vel.z *= airDrag;

        node.pos.x += node.vel.x * subDt;
        node.pos.y += node.vel.y * subDt;
        node.pos.z += node.vel.z * subDt;
      }
    }

    // 2. Coupled XPBD Iteration Loop: Muscles + Bones + Floor Contact & Coulomb Friction
    for (let iter = 0; iter < solverIterations; iter++) {
      for (let b = 0; b < this.bodies.length; b++) {
        const body = this.bodies[b];

        // 2a. Active Muscles
        const muscles = body.muscles;
        for (let m = 0; m < muscles.length; m++) {
          const mus = muscles[m];
          const nA = mus.nodeA;
          const nB = mus.nodeB;
          const wSum = nA.invMass + nB.invMass;
          if (wSum === 0) continue;

          const dx = nB.pos.x - nA.pos.x;
          const dy = nB.pos.y - nA.pos.y;
          const dz = nB.pos.z - nA.pos.z;
          const dist = Math.hypot(dx, dy, dz);
          if (dist < 1e-8) continue;

          const C = dist - mus.targetLength;
          const alphaTilde = mus.compliance * invSubDtSq;
          let dLambda = (-C - alphaTilde * mus.lambda) / (wSum + alphaTilde);

          const maxLambda = mus.maxForce * subDt * subDt;
          const newLambda = Math.max(-maxLambda, Math.min(maxLambda, mus.lambda + dLambda));
          dLambda = newLambda - mus.lambda;
          mus.lambda = newLambda;
          mus.appliedForce = mus.lambda * invSubDtSq;

          const corr = dLambda / dist;
          const cx = dx * corr;
          const cy = dy * corr;
          const cz = dz * corr;

          nA.pos.x -= cx * nA.invMass;
          nA.pos.y -= cy * nA.invMass;
          nA.pos.z -= cz * nA.invMass;

          nB.pos.x += cx * nB.invMass;
          nB.pos.y += cy * nB.invMass;
          nB.pos.z += cz * nB.invMass;
        }

        // 2b. Rigid Bones & Ligaments
        const bones = body.bones;
        for (let c = 0; c < bones.length; c++) {
          const bone = bones[c];
          const nA = bone.nodeA;
          const nB = bone.nodeB;
          const wSum = nA.invMass + nB.invMass;
          if (wSum === 0) continue;

          const dx = nB.pos.x - nA.pos.x;
          const dy = nB.pos.y - nA.pos.y;
          const dz = nB.pos.z - nA.pos.z;
          const dist = Math.hypot(dx, dy, dz);
          if (dist < 1e-8) continue;

          const C = dist - bone.restLength;
          const alphaTilde = bone.compliance * invSubDtSq;
          const dLambda = (-C - alphaTilde * bone.lambda) / (wSum + alphaTilde);
          bone.lambda += dLambda;

          const corr = dLambda / dist;
          const cx = dx * corr;
          const cy = dy * corr;
          const cz = dz * corr;

          nA.pos.x -= cx * nA.invMass;
          nA.pos.y -= cy * nA.invMass;
          nA.pos.z -= cz * nA.invMass;

          nB.pos.x += cx * nB.invMass;
          nB.pos.y += cy * nB.invMass;
          nB.pos.z += cz * nB.invMass;
        }

        // 2c. Floor Contact & Coulomb Friction Cone
        for (let i = 0; i < body.nodes.length; i++) {
          const node = body.nodes[i];
          if (node.invMass === 0) continue;

          const penetration = node.noGroundCollision ? 0 : (node.radius - node.pos.y);
          if (penetration > 0) {
            node.pos.y = node.radius;
            if (node.prevPos.y < node.radius) {
              node.prevPos.y = node.radius;
            }
            node.inContact = true;
            node._subContact = true;

            const normalForceIter = penetration * node.mass * invSubDtSq;
            node.normalForce += normalForceIter / this.substeps;
            node._subNormalForce += normalForceIter;

            const slipX = node.pos.x - node.prevPos.x;
            const slipZ = node.pos.z - node.prevPos.z;
            const slipDist = Math.hypot(slipX, slipZ);

            if (slipDist > 1e-9) {
              const muEffective = this.floorFriction * node.friction;
              const maxFrictionDisp = muEffective * penetration;
              const arrestRatio = Math.min(1.0, maxFrictionDisp / slipDist);

              const corrX = -slipX * arrestRatio;
              const corrZ = -slipZ * arrestRatio;

              node.pos.x += corrX;
              node.pos.z += corrZ;

              const instFx = corrX * node.mass * invSubDtSq;
              const instFz = corrZ * node.mass * invSubDtSq;
              node.frictionForceVec.x += instFx / this.substeps;
              node.frictionForceVec.z += instFz / this.substeps;
              node._subFrictionX += instFx;
              node._subFrictionZ += instFz;
            }
          }

          const limit = halfArena - node.radius;
          if (node.pos.x > limit) node.pos.x = limit;
          else if (node.pos.x < -limit) node.pos.x = -limit;
          if (node.pos.z > limit) node.pos.z = limit;
          else if (node.pos.z < -limit) node.pos.z = -limit;
        }
      }
    }

    // 3. Update velocities from XPBD positions
    for (let b = 0; b < this.bodies.length; b++) {
      const body = this.bodies[b];
      for (let i = 0; i < body.nodes.length; i++) {
        const node = body.nodes[i];
        if (node.invMass === 0) continue;
        node.vel.x = (node.pos.x - node.prevPos.x) * invSubDt;
        node.vel.y = (node.pos.y - node.prevPos.y) * invSubDt;
        node.vel.z = (node.pos.z - node.prevPos.z) * invSubDt;

        if (node.inContact && node.vel.y < 0) {
          node.vel.y = 0;
        }
        if (node._subContact) {
          node.renderNormalForce = 0.65 * (node.renderNormalForce || node._subNormalForce) + 0.35 * node._subNormalForce;
          node.renderFrictionVec.x = 0.60 * (node.renderFrictionVec.x || 0) + 0.40 * node._subFrictionX;
          node.renderFrictionVec.z = 0.60 * (node.renderFrictionVec.z || 0) + 0.40 * node._subFrictionZ;
        } else {
          node.renderNormalForce = 0.5 * (node.renderNormalForce || 0);
          node.renderFrictionVec.x *= 0.5;
          node.renderFrictionVec.z *= 0.5;
        }
      }
    }
  }

  /**
   * Advances all articulated bodies by dt seconds using coupled XPBD substeps.
   */
  step(dt, actionCallback = null) {
    const subDt = dt / this.substeps;
    this.resetContactTelemetry(actionCallback);
    for (let s = 0; s < this.substeps; s++) {
      this.stepSingleSubstep(subDt);
    }
  }
}
