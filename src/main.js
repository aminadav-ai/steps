import { ArenaSimulation } from './physics/ArenaSimulation.js';
import { MORPHOLOGY_REGISTRY } from './morphologies/Morphologies.js';
import { Renderer3D } from './render/Renderer3D.js';
import { NeuralGraphRenderer, LearningChartRenderer } from './render/TelemetryRenderers.js';
import { ExperimentLogger } from './logging/ExperimentLogger.js';

const arenaCanvas = document.getElementById('arenaCanvas');
const nnCanvas = document.getElementById('nnCanvas');
const chartCanvas = document.getElementById('chartCanvas');

const renderer3D = new Renderer3D(arenaCanvas);
const nnRenderer = new NeuralGraphRenderer(nnCanvas);
const chartRenderer = new LearningChartRenderer(chartCanvas);
const logger = new ExperimentLogger();

let sim = null;
let paused = false;
let simSpeed = 1;

// DOM Elements
const selectMorphology = document.getElementById('selectMorphology');
const selectAlgorithm = document.getElementById('selectAlgorithm');
const morphDescription = document.getElementById('morphDescription');
const nnDimLabel = document.getElementById('nnDimLabel');
const btnNewExperiment = document.getElementById('btnNewExperiment');
const btnPause = document.getElementById('btnPause');
const btnStepOnce = document.getElementById('btnStepOnce');
const btnUprightWalker = document.getElementById('btnUprightWalker');
const btnFastTrain = document.getElementById('btnFastTrain');
const speedButtons = document.getElementById('speedButtons');
const rangeFriction = document.getElementById('rangeFriction');
const labelFrictionVal = document.getElementById('labelFrictionVal');
const chkFollowLeader = document.getElementById('chkFollowLeader');
const chkShowForces = document.getElementById('chkShowForces');
const chkShowAllPop = document.getElementById('chkShowAllPop');

const hudUpdate = document.getElementById('hudUpdate');
const hudSimTime = document.getElementById('hudSimTime');
const hudTotalFood = document.getElementById('hudTotalFood');
const hudLeaderId = document.getElementById('hudLeaderId');
const hudLeaderFood = document.getElementById('hudLeaderFood');
const hudLeaderVel = document.getElementById('hudLeaderVel');
const hudFriction = document.getElementById('hudFriction');
const expStatusBadge = document.getElementById('expStatusBadge');

const metricMeanVel = document.getElementById('metricMeanVel');
const metricMeanRew = document.getElementById('metricMeanRew');
const metricPower = document.getElementById('metricPower');
const logsContainer = document.getElementById('logsContainer');
const logsCountBadge = document.getElementById('logsCountBadge');

function formatFrictionLabel(mu) {
  if (mu < 0.05) return `${mu.toFixed(2)} (Абсолютный лёд — 0 тяги!)`;
  if (mu < 0.35) return `${mu.toFixed(2)} (Скользкий лёд)`;
  if (mu < 0.75) return `${mu.toFixed(2)} (Гладкий пластик)`;
  if (mu <= 1.15) return `${mu.toFixed(2)} (Резина / Бетон)`;
  return `${mu.toFixed(2)} (Высокий зацеп)`;
}

function renderLogsList(logs) {
  if (!logs || logs.length === 0) {
    logsCountBadge.textContent = '0 файлов';
    logsContainer.innerHTML = '<div class="log-entry"><span>Нет сохранённых логов</span></div>';
    return;
  }
  logsCountBadge.textContent = `${logs.length} файл(ов)`;
  logsContainer.innerHTML = logs.slice(0, 12).map(l => {
    const tag = l.filename === 'current_experiment.json' ? '🟢 Текущий' : '📦 Архив';
    const food = l.totalFoodCollected !== undefined ? `Еда: ${l.totalFoodCollected}` : '';
    const upd = l.updatesCount !== undefined ? `Upd: ${l.updatesCount}` : '';
    return `<div class="log-entry" title="${l.filename}">
      <span>${tag} <strong>${l.morphology || 'tripod'}/${(l.algorithm || 'ppo').toUpperCase()}</strong></span>
      <span>${upd} ${food}</span>
    </div>`;
  }).join('');
}

async function startNewExperiment() {
  const morphologyKey = selectMorphology.value;
  const algorithm = selectAlgorithm.value;
  const floorFriction = parseFloat(rangeFriction.value);

  const spec = MORPHOLOGY_REGISTRY[morphologyKey];
  morphDescription.textContent = spec.description;
  nnDimLabel.textContent = `${spec.obsDim} → 32 → 24 → ${spec.actDim}`;

  sim = new ArenaSimulation({
    morphologyKey,
    algorithm,
    populationSize: 16,
    arenaHalfSize: 11.0,
    floorFriction
  });
  renderer3D.lockedFoodPos.set(NaN, NaN, NaN);

  chartRenderer.render(sim.history);

  await logger.startNewExperiment({
    morphology: morphologyKey,
    algorithm,
    populationSize: 16,
    config: { floorFriction, arenaHalfSize: 11.0 }
  });
  renderLogsList(logger.savedLogsList);
}

function getUIState() {
  return {
    paused,
    simSpeed,
    floorFriction: parseFloat(rangeFriction.value),
    followLeader: chkFollowLeader.checked,
    showForceVectors: chkShowForces.checked,
    showAllPopulation: chkShowAllPop.checked,
    morphology: selectMorphology.value,
    algorithm: selectAlgorithm.value
  };
}

// Event Listeners
selectMorphology.addEventListener('change', () => {
  logger.recordUserEvent('select_morphology', { morphology: selectMorphology.value }, sim, renderer3D, getUIState());
  startNewExperiment();
});
selectAlgorithm.addEventListener('change', () => {
  logger.recordUserEvent('select_algorithm', { algorithm: selectAlgorithm.value }, sim, renderer3D, getUIState());
  startNewExperiment();
});
btnNewExperiment.addEventListener('click', () => {
  logger.recordUserEvent('btn_new_experiment', { reason: 'user_reset_ai' }, sim, renderer3D, getUIState());
  startNewExperiment();
});

function setPausedState(nextPaused, source = 'ui') {
  paused = nextPaused;
  btnPause.textContent = paused ? 'Продолжить' : 'Пауза';
  expStatusBadge.textContent = paused ? '⏸ ПАУЗА' : '● АКТИВЕН';
  expStatusBadge.style.color = paused ? '#fbbf24' : '#4ade80';
  logger.recordUserEvent(paused ? 'pause' : 'resume', { source }, sim, renderer3D, getUIState());
}

function handleUpdateEntry(updateEntry) {
  if (!updateEntry) return;
  chartRenderer.render(sim.history);
  const fullSnapshot = logger.buildTelemetrySnapshot(sim, renderer3D, getUIState());
  logger.logUpdate(
    updateEntry,
    sim.morphologyKey,
    sim.algorithm,
    fullSnapshot
  ).then(() => {
    if (updateEntry.update % 3 === 0) {
      logger.refreshLogsList().then(list => renderLogsList(list));
    }
  });
}

btnPause.addEventListener('click', () => {
  setPausedState(!paused, 'btn_pause');
});

// Single click on the 3D arena canvas toggles pause/resume
renderer3D.onTogglePause = (source = 'canvas_click') => {
  setPausedState(!paused, source);
};

// Camera orbit, Shift+drag strafe, and wheel zoom events
renderer3D.onCameraAction = camEvent => {
  logger.recordCameraEvent(camEvent, sim, renderer3D, getUIState());
};

btnUprightWalker.addEventListener('click', () => {
  if (!sim) return;
  const focusIdx = renderer3D.showAllPopulation ? (sim.bestCreatureIdx || 0) : 0;
  const c = sim.creatures[focusIdx] || sim.creatures[0];
  const wasOverturned = c ? c.getTorsoFrame().up.y < 0.35 : false;
  sim.uprightCreature(focusIdx);
  renderer3D.lockedFoodPos.set(NaN, NaN, NaN);
  logger.recordUserEvent('btn_upright_walker', { focusIdx, wasOverturned }, sim, renderer3D, getUIState());
  updateHUD();
});

btnStepOnce.addEventListener('click', () => {
  if (!sim) return;
  if (!paused) setPausedState(true, 'btn_step_once_auto_pause');
  simTimeAccum = 0;
  const updateEntry = sim.stepControl();
  handleUpdateEntry(updateEntry);
  logger.recordUserEvent('btn_step_once', {}, sim, renderer3D, getUIState());
  const leader = sim.creatures[sim.bestCreatureIdx || 0];
  nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null);
  updateHUD();
});

let isFastTraining = false;

async function runFastHeadlessTraining(targetUpdates = 25) {
  if (!sim || isFastTraining) return;
  isFastTraining = true;
  btnFastTrain.disabled = true;
  expStatusBadge.textContent = '⚡ ТУРБО-ОБУЧЕНИЕ...';
  expStatusBadge.style.color = '#38bdf8';

  // Preserve the main creature's current physical position and food target so it doesn't jump
  // while the neural network trains in the background!
  const c0 = sim.creatures[0];
  if (c0.getTorsoFrame().up.y < 0.45) {
    sim.uprightCreature(0);
  }
  const savedNodes = c0.body.nodes.map(n => ({
    pos: n.pos.clone(),
    prevPos: n.prevPos.clone(),
    vel: n.vel.clone()
  }));
  const savedFoodPos = c0.foodPos.clone();
  const savedPrevFoodDist = c0.prevFoodDist;
  const savedFoodCollected = c0.foodCollected;
  const savedPhase = c0.phase;

  sim._isHeadlessFastTrain = true;
  for (let u = 1; u <= targetUpdates; u++) {
    let updateEntry = null;
    while (!updateEntry) {
      updateEntry = sim.stepControl();
    }
    handleUpdateEntry(updateEntry);
    btnFastTrain.textContent = `⚡ Обучение в фоне: ${u} / ${targetUpdates} эпох...`;

    if (u % 3 === 0) {
      updateHUD();
      await new Promise(r => setTimeout(r, 0));
    }
  }
  sim._isHeadlessFastTrain = false;

  // Restore c0 to its exact pre-training physical state so it continues seamlessly from the exact same spot
  // with its newly upgraded neural network brain!
  for (let i = 0; i < savedNodes.length; i++) {
    c0.body.nodes[i].pos.copy(savedNodes[i].pos);
    c0.body.nodes[i].prevPos.copy(savedNodes[i].prevPos);
    c0.body.nodes[i].vel.set(0, 0, 0);
  }
  c0.foodPos.copy(savedFoodPos);
  c0.prevFoodDist = savedPrevFoodDist;
  c0.foodCollected = savedFoodCollected;
  c0.phase = savedPhase;
  c0.overturnSteps = 0;

  simTimeAccum = 0;
  lastFrameTime = performance.now();
  setPausedState(false);

  // Set speed selector back to 1x so user observes the trained mechanics in real time
  simSpeed = 1;
  for (const b of speedButtons.querySelectorAll('button')) {
    b.classList.toggle('active', b.dataset.speed === '1');
  }

  btnFastTrain.disabled = false;
  btnFastTrain.textContent = `⚡ Быстро обучить без рендера (+25 эпох)`;
  isFastTraining = false;
  updateHUD();
}

btnFastTrain.addEventListener('click', () => {
  logger.recordUserEvent('btn_fast_train', { epochs: 25 }, sim, renderer3D, getUIState());
  runFastHeadlessTraining(25);
});

speedButtons.addEventListener('click', e => {
  const btn = e.target.closest('button[data-speed]');
  if (!btn) return;
  const prevSpeed = simSpeed;
  simSpeed = parseFloat(btn.dataset.speed) || 1;
  for (const b of speedButtons.querySelectorAll('button')) {
    b.classList.toggle('active', b === btn);
  }
  logger.recordUserEvent('speed_change', { fromSpeed: prevSpeed, toSpeed: simSpeed }, sim, renderer3D, getUIState());
});

rangeFriction.addEventListener('input', () => {
  const mu = parseFloat(rangeFriction.value);
  labelFrictionVal.textContent = formatFrictionLabel(mu);
  hudFriction.textContent = mu.toFixed(2);
  if (sim) {
    sim.setFloorFriction(mu);
  }
});

rangeFriction.addEventListener('change', () => {
  const mu = parseFloat(rangeFriction.value);
  logger.recordUserEvent('friction_change', { floorFriction: mu }, sim, renderer3D, getUIState());
});

chkFollowLeader.addEventListener('change', () => {
  renderer3D.followLeader = chkFollowLeader.checked;
  if (renderer3D.followLeader) {
    renderer3D.lockedFoodPos.set(NaN, NaN, NaN);
  }
  logger.recordUserEvent('toggle_follow_camera', { followLeader: chkFollowLeader.checked }, sim, renderer3D, getUIState());
});

chkShowForces.addEventListener('change', () => {
  renderer3D.showForceVectors = chkShowForces.checked;
  logger.recordUserEvent('toggle_force_vectors', { showForceVectors: chkShowForces.checked }, sim, renderer3D, getUIState());
});

chkShowAllPop.addEventListener('change', () => {
  renderer3D.showAllPopulation = chkShowAllPop.checked;
  logger.recordUserEvent('toggle_population_view', { showAllPopulation: chkShowAllPop.checked }, sim, renderer3D, getUIState());
});

function updateHUD() {
  if (!sim) return;
  const focusIdx = renderer3D.showAllPopulation ? (sim.bestCreatureIdx || 0) : 0;
  const leader = sim.creatures[focusIdx] || sim.creatures[0];

  hudUpdate.textContent = sim.trainer.updateCount;
  hudSimTime.textContent = `${sim.simTime.toFixed(1)} с`;
  hudTotalFood.textContent = sim.totalFoodCollected;
  hudLeaderId.textContent = focusIdx;
  hudLeaderFood.textContent = leader.foodCollected;
  hudLeaderVel.textContent = `${leader.lastVelocityTowardFood.toFixed(2)} м/с`;

  const meanVel = sim.recentVelocities.length > 0
    ? sim.recentVelocities.reduce((a, b) => a + b, 0) / sim.recentVelocities.length
    : 0;
  const meanRew = sim.recentStepRewards.length > 0
    ? sim.recentStepRewards.reduce((a, b) => a + b, 0) / sim.recentStepRewards.length
    : 0;

  metricMeanVel.textContent = `${meanVel.toFixed(3)} м/с`;
  metricMeanRew.textContent = meanRew.toFixed(3);
  metricPower.textContent = `${leader.lastMechanicalPower.toFixed(1)} Вт`;

  // Highlight the "Help Stand Up" button if the walker is tilted onto its side or back
  const { up } = leader.getTorsoFrame();
  const isOverturned = up.y < 0.35;
  btnUprightWalker.classList.toggle('amber-alert', isOverturned);
  btnUprightWalker.textContent = isOverturned
    ? '⚠️ Перевернулся — Помочь встать!'
    : '🤚 Помочь встать на ноги';
}

let frameCounter = 0;
let lastFrameTime = performance.now();
let simTimeAccum = 0;

function animate(now = performance.now()) {
  requestAnimationFrame(animate);
  const realDt = Math.min(0.05, Math.max(0.0, (now - lastFrameTime) / 1000));
  lastFrameTime = now;

  if (!sim) return;

  if (!paused && !isFastTraining) {
    simTimeAccum += realDt * simSpeed;
    // Cap accumulator to avoid spiral of death if tab was backgrounded
    const maxSubsteps = Math.max(20, Math.ceil(simSpeed * 16));
    let substepsRun = 0;
    while (simTimeAccum >= sim.subDt && substepsRun < maxSubsteps) {
      simTimeAccum -= sim.subDt;
      substepsRun++;
      const updateEntry = sim.stepPhysicsSubstep();
      handleUpdateEntry(updateEntry);
    }
    if (substepsRun >= maxSubsteps) {
      simTimeAccum = 0;
    }
  }

  const interpAlpha = paused ? 1.0 : Math.min(1.0, simTimeAccum / sim.subDt);
  renderer3D.render(sim, interpAlpha);

  frameCounter++;
  if (frameCounter % 2 === 0) {
    const focusIdx = renderer3D.showAllPopulation ? (sim.bestCreatureIdx || 0) : 0;
    const leader = sim.creatures[focusIdx] || sim.creatures[0];
    nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null);
    updateHUD();
  }
  // Periodic telemetry flush every ~2.5 seconds (150 frames) so trajectory & camera state stay up-to-date on disk
  if (frameCounter % 150 === 0 && !paused) {
    logger.scheduleSnapshotFlush(sim, renderer3D, getUIState());
  }
}

startNewExperiment().then(() => {
  requestAnimationFrame(animate);
});
