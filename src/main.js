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

const hudUpdate = document.getElementById('hudUpdate');
const hudSimTime = document.getElementById('hudSimTime');
const hudTotalFood = document.getElementById('hudTotalFood');
const hudLeaderVel = document.getElementById('hudLeaderVel');
const hudFriction = document.getElementById('hudFriction');
const expStatusBadge = document.getElementById('expStatusBadge');

const metaActiveBadge = document.getElementById('metaActiveBadge');
const metaModeBanner = document.getElementById('metaModeBanner');
const metaNeedVal = document.getElementById('metaNeedVal');
const metaNeedFill = document.getElementById('metaNeedFill');
const metaRightingVal = document.getElementById('metaRightingVal');
const metaRightingFill = document.getElementById('metaRightingFill');
const metaReasonBox = document.getElementById('metaReasonBox');
const ambitionButtons = document.getElementById('ambitionButtons');
const rangeAmbition = document.getElementById('rangeAmbition');
const labelAmbitionVal = document.getElementById('labelAmbitionVal');
const btnToggleFreeze = document.getElementById('btnToggleFreeze');

let currentAmbitionPreset = 'min';

function formatAmbitionLabel(val, preset) {
  if (preset === 'infinite' || val >= 0.54) return '∞ (Вечное обучение — 100%)';
  if (val <= 0.21) return `${val.toFixed(2)} м/с (Минимум / Шаг)`;
  if (val <= 0.39) return `${val.toFixed(2)} м/с (Бодрый скачок)`;
  return `${val.toFixed(2)} м/с (Турбо-галоп)`;
}

const metricMeanVel = document.getElementById('metricMeanVel');
const metricMeanRew = document.getElementById('metricMeanRew');
const metricPower = document.getElementById('metricPower');
const learningChartBadge = document.getElementById('learningChartBadge');
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

const CREATURE_STORAGE_KEY = 'steps_saved_creature_v1';

function saveCreatureCheckpoint() {
  if (!sim || typeof sim.exportCheckpoint !== 'function') return null;
  try {
    const ckpt = sim.exportCheckpoint(getUIState(), renderer3D ? renderer3D.getCameraTelemetry() : null);
    localStorage.setItem(CREATURE_STORAGE_KEY, JSON.stringify(ckpt));
    return ckpt;
  } catch {
    return null;
  }
}

function applyRestoredUIState(ckpt) {
  if (!ckpt) return;
  if (ckpt.morphologyKey && MORPHOLOGY_REGISTRY[ckpt.morphologyKey]) {
    selectMorphology.value = ckpt.morphologyKey;
  }
  if (ckpt.algorithm) {
    selectAlgorithm.value = ckpt.algorithm;
  }
  if (typeof ckpt.floorFriction === 'number') {
    rangeFriction.value = ckpt.floorFriction.toFixed(2);
    labelFrictionVal.textContent = formatFrictionLabel(ckpt.floorFriction);
    hudFriction.textContent = ckpt.floorFriction.toFixed(2);
  }
  const m = ckpt.trainer && ckpt.trainer.metaState ? ckpt.trainer.metaState : null;
  if (m) {
    currentAmbitionPreset = m.ambitionPreset || 'min';
    const tVal = Math.min(0.55, m.targetCompetence || 0.18);
    if (rangeAmbition) rangeAmbition.value = tVal.toFixed(2);
    if (labelAmbitionVal) labelAmbitionVal.textContent = formatAmbitionLabel(tVal, currentAmbitionPreset);
    if (ambitionButtons) {
      for (const b of ambitionButtons.querySelectorAll('button')) {
        b.classList.toggle('active', b.dataset.ambition === currentAmbitionPreset);
      }
    }
  }
}

async function startNewExperiment({ clearSaved = false } = {}) {
  if (clearSaved) {
    try {
      localStorage.removeItem(CREATURE_STORAGE_KEY);
    } catch {
      // ignore
    }
  }
  const morphologyKey = selectMorphology.value;
  const algorithm = selectAlgorithm.value;
  const floorFriction = parseFloat(rangeFriction.value);

  const spec = MORPHOLOGY_REGISTRY[morphologyKey];
  morphDescription.textContent = spec.description;
  nnDimLabel.textContent = `${spec.obsDim} → 32 → 24 → ${spec.actDim}`;

  sim = new ArenaSimulation({
    morphologyKey,
    algorithm,
    populationSize: 1,
    arenaHalfSize: 11.0,
    floorFriction
  });
  if (sim.trainer && typeof sim.trainer.setLearningAmbition === 'function') {
    if (currentAmbitionPreset === 'custom') {
      sim.trainer.setLearningAmbition(parseFloat(rangeAmbition.value));
    } else {
      sim.trainer.setLearningAmbition(currentAmbitionPreset);
    }
  }
  renderer3D.lockedFoodPos.set(NaN, NaN, NaN);

  chartRenderer.render(sim.history, true);

  await logger.startNewExperiment({
    morphology: morphologyKey,
    algorithm,
    populationSize: 1,
    config: { floorFriction, arenaHalfSize: 11.0 }
  });
  saveCreatureCheckpoint();
  renderLogsList(logger.savedLogsList);
  updateHUD(true);
}

async function restoreOrStartExperiment() {
  let ckpt = null;
  try {
    const raw = localStorage.getItem(CREATURE_STORAGE_KEY);
    if (raw) ckpt = JSON.parse(raw);
  } catch {
    ckpt = null;
  }

  if (ckpt && ckpt.morphologyKey && MORPHOLOGY_REGISTRY[ckpt.morphologyKey]) {
    applyRestoredUIState(ckpt);
    const spec = MORPHOLOGY_REGISTRY[ckpt.morphologyKey];
    morphDescription.textContent = spec.description;
    nnDimLabel.textContent = `${spec.obsDim} → 32 → 24 → ${spec.actDim}`;

    sim = new ArenaSimulation({
      morphologyKey: ckpt.morphologyKey,
      algorithm: ckpt.algorithm || 'ppo',
      populationSize: 1,
      arenaHalfSize: 11.0,
      floorFriction: typeof ckpt.floorFriction === 'number' ? ckpt.floorFriction : 1.0
    });
    sim.importCheckpoint(ckpt);
    renderer3D.lockedFoodPos.set(NaN, NaN, NaN);

    const isLearningNow = !sim.trainer.metaState || sim.trainer.metaState.isLearningActive;
    chartRenderer.render(sim.history, isLearningNow);
    const leader = sim.creatures[0];
    nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null);
    await logger.refreshLogsList();
    renderLogsList(logger.savedLogsList);
    updateHUD(true);
    saveCreatureCheckpoint();
    return;
  }

  await startNewExperiment({ clearSaved: false });
}

function getUIState() {
  return {
    paused,
    simSpeed,
    floorFriction: parseFloat(rangeFriction.value),
    followLeader: chkFollowLeader.checked,
    showForceVectors: chkShowForces.checked,
    morphology: selectMorphology.value,
    algorithm: selectAlgorithm.value
  };
}

// Event Listeners
selectMorphology.addEventListener('change', () => {
  logger.recordUserEvent('select_morphology', { morphology: selectMorphology.value }, sim, renderer3D, getUIState());
  startNewExperiment({ clearSaved: true });
});
selectAlgorithm.addEventListener('change', () => {
  logger.recordUserEvent('select_algorithm', { algorithm: selectAlgorithm.value }, sim, renderer3D, getUIState());
  startNewExperiment({ clearSaved: true });
});
btnNewExperiment.addEventListener('click', () => {
  logger.recordUserEvent('btn_new_experiment', { reason: 'user_reset_ai' }, sim, renderer3D, getUIState());
  startNewExperiment({ clearSaved: true });
});

function setPausedState(nextPaused, source = 'ui') {
  paused = nextPaused;
  btnPause.textContent = paused ? 'Продолжить' : 'Пауза';
  expStatusBadge.textContent = paused ? '⏸ ПАУЗА' : '● АКТИВЕН';
  expStatusBadge.style.color = paused ? '#fbbf24' : '#4ade80';
  saveCreatureCheckpoint();
  logger.recordUserEvent(paused ? 'pause' : 'resume', { source }, sim, renderer3D, getUIState());
}

function handleUpdateEntry(updateEntry) {
  if (!updateEntry) return;
  const isLearningNow = !sim.trainer.metaState || sim.trainer.metaState.isLearningActive;
  chartRenderer.render(sim.history, isLearningNow);
  saveCreatureCheckpoint();
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

// Direct 3D mouse manipulation of food orb (drag & drop) and walker (grab, move, throw)
renderer3D.onSceneInteract = interactEvent => {
  saveCreatureCheckpoint();
  logger.recordUserEvent(interactEvent.action, interactEvent, sim, renderer3D, getUIState());
  updateHUD();
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

  const c0 = sim.creatures[0];
  sim._isHeadlessFastTrain = true;
  for (let u = 1; u <= targetUpdates; u++) {
    let updateEntry = null;
    while (!updateEntry) {
      updateEntry = sim.stepControl();
    }
    handleUpdateEntry(updateEntry);
    btnFastTrain.textContent = `⚡ Обучение без рендера: ${u} / ${targetUpdates} эпох...`;

    if (u % 3 === 0) {
      updateHUD();
      await new Promise(r => setTimeout(r, 0));
    }
  }
  sim._isHeadlessFastTrain = false;

  if (c0.getTorsoFrame().up.y < 0.45) {
    sim.uprightCreature(0);
  }
  renderer3D.lockedFoodPos.set(NaN, NaN, NaN);

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

if (ambitionButtons) {
  ambitionButtons.addEventListener('click', e => {
    const btn = e.target.closest('button[data-ambition]');
    if (!btn) return;
    const preset = btn.dataset.ambition;
    currentAmbitionPreset = preset;
    for (const b of ambitionButtons.querySelectorAll('button')) {
      b.classList.toggle('active', b === btn);
    }
    if (sim && sim.trainer && typeof sim.trainer.setLearningAmbition === 'function') {
      sim.trainer.setLearningAmbition(preset);
      const tVal = Math.min(0.55, sim.trainer.metaState.targetCompetence);
      rangeAmbition.value = tVal.toFixed(2);
      labelAmbitionVal.textContent = formatAmbitionLabel(tVal, preset);
    }
    if (paused && preset !== 'min') {
      setPausedState(false, 'ambition_change_auto_resume');
    }
    logger.recordUserEvent('set_learning_ambition', { preset, targetCompetence: sim?.trainer?.metaState?.targetCompetence }, sim, renderer3D, getUIState());
    updateHUD();
  });
}

if (rangeAmbition) {
  rangeAmbition.addEventListener('input', () => {
    const val = parseFloat(rangeAmbition.value);
    currentAmbitionPreset = val >= 0.54 ? 'infinite' : 'custom';
    if (ambitionButtons) {
      for (const b of ambitionButtons.querySelectorAll('button')) {
        b.classList.toggle('active', currentAmbitionPreset === 'infinite' && b.dataset.ambition === 'infinite');
      }
    }
    labelAmbitionVal.textContent = formatAmbitionLabel(val, currentAmbitionPreset);
    if (sim && sim.trainer && typeof sim.trainer.setLearningAmbition === 'function') {
      sim.trainer.setLearningAmbition(val);
    }
    updateHUD();
  });

  rangeAmbition.addEventListener('change', () => {
    const val = parseFloat(rangeAmbition.value);
    logger.recordUserEvent('slider_learning_ambition', { targetCompetence: val }, sim, renderer3D, getUIState());
  });
}

if (btnToggleFreeze) {
  btnToggleFreeze.addEventListener('click', () => {
    if (!sim || !sim.trainer || typeof sim.trainer.toggleManualFreeze !== 'function') return;
    sim.trainer.toggleManualFreeze();
    logger.recordUserEvent('toggle_manual_freeze', {
      mode: sim.trainer.metaState.mode,
      isLearningActive: sim.trainer.metaState.isLearningActive
    }, sim, renderer3D, getUIState());
    updateHUD();
  });
}

let lastVelHudUpdateNow = 0;

function updateHUD(forceVelUpdate = false) {
  if (!sim) return;
  const leader = sim.creatures[0];

  hudUpdate.textContent = sim.trainer.updateCount;
  hudSimTime.textContent = `${Math.floor(sim.simTime)} с`;
  hudTotalFood.textContent = leader.foodCollected;

  const now = performance.now();
  if (forceVelUpdate || now - lastVelHudUpdateNow >= 280) {
    lastVelHudUpdateNow = now;
    hudLeaderVel.textContent = `${(leader.smoothedVelocityTowardFood || 0).toFixed(2)} м/с`;
  }

  const m = sim.trainer.metaState;
  const isLearningNow = !m || m.isLearningActive;

  if (learningChartBadge) {
    learningChartBadge.textContent = isLearningNow ? '● ИДЁТ ЗАПИСЬ' : '⏸ ЗАФИКСИРОВАНО';
    learningChartBadge.style.color = isLearningNow ? '#34d399' : '#94a3b8';
  }

  // Only update "Динамика обучения" metrics while learning is active so the card freezes when CONSOLIDATED
  if (isLearningNow || forceVelUpdate) {
    const meanVel = sim.recentVelocities.length > 0
      ? sim.recentVelocities.reduce((a, b) => a + b, 0) / sim.recentVelocities.length
      : 0;
    const meanRew = sim.recentStepRewards.length > 0
      ? sim.recentStepRewards.reduce((a, b) => a + b, 0) / sim.recentStepRewards.length
      : 0;

    metricMeanVel.textContent = `${meanVel.toFixed(3)} м/с`;
    metricMeanRew.textContent = meanRew.toFixed(3);
    metricPower.textContent = `${leader.lastMechanicalPower.toFixed(1)} Вт`;
  }

  // Update Meta-Plasticity & Beetle Self-Righting Status Card on right panel
  if (m && metaActiveBadge) {
    const modeClass = m.mode === 'CONSOLIDATED'
      ? 'consolidated'
      : m.mode === 'LEARNING_RIGHTING'
        ? 'righting'
        : 'learning';

    metaActiveBadge.className = `meta-badge ${modeClass}`;
    metaActiveBadge.textContent = m.mode === 'CONSOLIDATED'
      ? '⏸ ДООБУЧЕНИЕ ОТКЛЮЧЕНО'
      : m.mode === 'LEARNING_RIGHTING'
        ? '🪲 ОБУЧЕНИЕ ПЕРЕВОРОТУ'
        : '● ОБУЧЕНИЕ ИДЁТ';

    metaModeBanner.className = `meta-mode-banner ${modeClass}`;
    metaModeBanner.textContent = m.modeTitle;

    metaNeedVal.textContent = `${m.needToLearnPct}%`;
    metaNeedFill.style.inlineSize = `${Math.max(4, Math.min(100, m.needToLearnPct))}%`;

    metaRightingVal.textContent = `${m.rightingSkillPct}% (встал сам: ${m.rightingSuccesses})`;
    metaRightingFill.style.inlineSize = `${Math.max(8, Math.min(100, m.rightingSkillPct))}%`;

    metaReasonBox.textContent = m.reasonText;
    metaReasonBox.style.borderInlineStartColor = m.mode === 'CONSOLIDATED'
      ? '#38bdf8'
      : m.mode === 'LEARNING_RIGHTING'
        ? '#f59e0b'
        : '#34d399';

    if (btnToggleFreeze) {
      const isCons = m.mode === 'CONSOLIDATED';
      btnToggleFreeze.textContent = isCons
        ? '🔓 Включить дообучение сейчас'
        : '🔒 Заморозить веса сейчас';
      btnToggleFreeze.classList.toggle('emerald', isCons);
    }
  }

  // Highlight the "Help Stand Up" button if the walker is tilted onto its side or back
  const { up } = leader.getTorsoFrame();
  const isOverturned = up.y < 0.35;
  btnUprightWalker.classList.toggle('amber-alert', isOverturned);
  btnUprightWalker.textContent = isOverturned
    ? '⚠️ На спине (учится сам или нажми помочь)'
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
  // Periodic telemetry & checkpoint flush every ~2.0 seconds (120 frames) so creature state & trajectory stay up-to-date
  if (frameCounter % 120 === 0 && !paused) {
    saveCreatureCheckpoint();
    logger.scheduleSnapshotFlush(sim, renderer3D, getUIState());
  }
}

window.addEventListener('beforeunload', () => {
  saveCreatureCheckpoint();
});

restoreOrStartExperiment().then(() => {
  requestAnimationFrame(animate);
});
