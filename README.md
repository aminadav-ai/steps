# STEPS // 3D Physics & Neural Locomotion Laboratory

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-10b981?style=for-the-badge&logo=github)](https://aminadav-ai.github.io/steps/)
[![Release](https://img.shields.io/badge/Release-v1.0.0-38bdf8?style=for-the-badge)](https://github.com/aminadav-ai/steps/releases/tag/v1.0.0)
[![Physics](https://img.shields.io/badge/Physics-200_Hz_3D_XPBD_%2B_Coulomb-f59e0b?style=for-the-badge)](#1-first-principles-3d-physics-engine)
[![RL](https://img.shields.io/badge/RL-PPO_Actor--Critic_%26_NES-a855f7?style=for-the-badge)](#2-reinforcement-learning--neural-architecture)
[![Languages](https://img.shields.io/badge/i18n-8_Languages_(EN%2FRU%2FHE%2FES%2FDE%2FFR%2FZH%2FJA)-ec4899?style=for-the-badge)](#6-multilingual-interface--per-user-persistence)
[![License: MIT](https://img.shields.io/badge/License-MIT-94a3b8?style=for-the-badge)](./LICENSE)

> **🌐 Live Interactive Laboratory:** **[https://aminadav-ai.github.io/steps/](https://aminadav-ai.github.io/steps/)**

![STEPS — 3D Physics & Neural Locomotion Laboratory](./assets/screenshot.png)

**STEPS** is an interactive, zero-dependency 3D physics and reinforcement learning laboratory running directly in the browser. A single multi-legged creature learns from scratch in real time how to coordinate its muscles, push against the floor using true Coulomb friction, steer toward food targets, and self-right when overturned—with **zero keyframed animations or hidden external forces**.

---

## ✨ Key Scientific & Engineering Highlights

### 1. First-Principles 3D Physics Engine (`200 Hz` Symplectic Euler + XPBD)
- **Strict Conservation of Linear & Angular Momentum:** Every muscle is an internal pairwise force-producing actuator ($\vec{F}_{ij} = -\vec{F}_{ji}$). On a frictionless floor ($\mu = 0.00$), vigorous leg flailing produces **zero center-of-mass drift** down to machine precision (`~2.7e-13 m`).
- **True Coulomb FrictionCone & Normal Impulse:** Locomotion emerges exclusively from ground normal reaction forces ($\vec{N}$) and tangential static/dynamic friction ($\|\vec{F}_{\text{fr}}\| \le \mu \|\vec{N}\|$), visualized live on every foot contact.
- **Anti-Inversion Joint Constraints:** Custom dorsal/ventral structural tendons prevent non-physical leg snap-through across the hip plane while allowing full natural articulation during flips and recovery.

### 2. Reinforcement Learning & Neural Architecture
- **31-Dimensional Proprioceptive & Exteroceptive Observation Space:**
  - Target heading ($\cos\alpha$, $\sin\alpha$) and distance in the creature's local torso frame;
  - Local 3D torso velocity, orientation vectors, angular rates, and foot contact sensors;
  - Central Pattern Generator (CPG) phase oscillators ($\sin\omega t$, $\cos\omega t$) and sensory-gated steering channels.
- **Actor-Critic Network (`31 → 32 → 24 → 6/8` + Direct CPG Skip-Synapses):**
  - Trained live via **PPO (Proximal Policy Optimization + Generalized Advantage Estimation + Adam Backpropagation)** or **SEP-CMA-ES / Natural Evolution Strategies**.
  - Live interactive neural network visualizer displaying firing activations, skip-connection weights, and Critic value estimates $V(s)$.

### 3. Homeostatic Meta-Plasticity Sense ("AI Meta-Sense")
- The creature continuously monitors its own locomotion competence and prediction surprise to decide **when to learn and when to consolidate**:
  - **Active Learning (`LEARNING_WALK`):** Adapts synaptic weights until the target speed threshold and target collection count are achieved.
  - **Skill Consolidation (`CONSOLIDATED`):** Automatically freezes neural weights and stops the AI epoch counter once mastery is reached, protecting the learned gait from catastrophic forgetting.
  - **4 Ambition Regimes:** `🐢 Min Walk` (`0.18 m/s`), `🐇 Bound` (`0.34 m/s`), `🚀 Gallop` (`0.48 m/s`), and `♾️ Infinite` (`100%` continuous plasticity). Raising the ambition threshold immediately wakes up training to master a faster gait.

### 4. Two-Phase Beetle Self-Righting Reflex ("Рефлекс жука")
- If you grab and flip the creature upside down onto its back (`up.y < 0.35`), it immediately pauses walking updates—protecting its walking memory—and enters **`🪲 LEARNING_RIGHTING`**.
- Using a dedicated two-phase righting reflex, it extends its floor-side leg as a lever against the ground, rolls its torso past vertical, and transitions smoothly to a ventral stance without leg inversion.

### 5. Direct 3D Mouse Manipulation & Time Control
- **Grab, Move, or Throw the Walker:** Drag the creature directly in 3D space or toss it onto its back to test its self-righting reflex.
- **Move the Target Orb:** Drag the emerald target orb anywhere on the arena floor in real time.
- **3-Phase Target Lifecycle:** Reached targets dissolve over `1.0 s`, pause for `1.0 s`, and drop from the sky with gravity and bounce over `1.0 s`.
- **Time Scale & Headless Fast-Train:** Observe mechanics in `0.1×` slow motion, `1×` real time, up to `15×` fast-forward, or run `+25 Epochs` instantaneously without rendering.

### 6. Multilingual Interface & Per-User Persistence
- **8 Built-in Languages (<28 KB static dictionary, 0% physics overhead):**
  - Automatically detects your browser language (`navigator.languages`) with **English (`en`)** as the universal fallback.
  - Instant switching between **🇬🇧 English**, **🇷🇺 Русский**, **🇮🇱 עברית (with native RTL layout)**, **🇪🇸 Español**, **🇩🇪 Deutsch**, **🇫🇷 Français**, **🇨🇳 中文**, and **🇯🇵 日本語**.
- **Per-User Browser Persistence (`localStorage`):**
  - Each visitor's creature, neural network weights, morphology, ambition setting, and language preference are automatically saved in their own browser's `localStorage` (`steps_saved_creature_v1`, `steps_lang_v1`) and restored upon page reload.

---

## 🚀 Quick Start (Local Development)

**Prerequisites:** [Node.js](https://nodejs.org/) v18+ (no external `npm install` packages required—100% pure ES modules).

```bash
# 1. Clone the repository
git clone https://github.com/aminadav-ai/steps.git
cd steps

# 2. Start the local server (serves at http://localhost:3000)
npm start

# 3. Run the automated physics & RL verification suite
npm test
```

---

## 🗂️ Project Structure

```text
steps/
├── index.html                        # Main application layout & 8-language selector
├── styles.css                        # Responsive dark/light scientific UI (CSS Logical Properties + RTL)
├── server.js                         # Zero-dependency Node HTTP server & local experiment archiver
├── src/
│   ├── main.js                       # Application loop, HUD telemetry, localStorage & i18n controller
│   ├── i18n/
│   │   └── translations.js           # Complete dictionary for EN, RU, HE (RTL), ES, DE, FR, ZH, JA
│   ├── physics/
│   │   ├── Vec3.js                   # Allocation-free 3D vector math
│   │   ├── PhysicsEngine3D.js        # 200 Hz Symplectic Euler + XPBD constraints + Coulomb friction
│   │   └── ArenaSimulation.js        # Arena environment, target lifecycle, checkpoints & rewards
│   ├── morphologies/
│   │   └── Morphologies.js           # Tripod (6 DoF) & Quadruped (8 DoF) 3D articulated walkers
│   ├── brains/
│   │   └── NeuralRL.js               # Actor-Critic MLP, PPO+GAE+Adam, SEP-CMA-ES, Meta-Sense & Beetle Reflex
│   ├── render/
│   │   ├── Renderer3D.js             # Custom 3D perspective renderer, raycasting & force vectors
│   │   └── TelemetryRenderers.js     # Real-time Neural Network graph & Learning Dynamics chart
│   └── logging/
│       └── ExperimentLogger.js       # Telemetry logger (auto-switches to static mode on GitHub Pages)
└── tests/
    └── verify_physics_and_rl.js      # Automated verification of momentum conservation, PPO & self-righting
```

---

## 📄 License

Released under the [MIT License](./LICENSE).
