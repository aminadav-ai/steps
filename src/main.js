import { ArenaSimulation } from './physics/ArenaSimulation.js';
import { MORPHOLOGY_REGISTRY } from './morphologies/Morphologies.js';
import { Renderer3D } from './render/Renderer3D.js';
import { NeuralGraphRenderer, LearningChartRenderer } from './render/TelemetryRenderers.js';
import { ExperimentLogger } from './logging/ExperimentLogger.js';
import { TRANSLATIONS } from './i18n/translations.js';

const LANG_STORAGE_KEY = 'steps_lang_v1';

function detectInitialLanguage() {
  try {
    const savedLang = localStorage.getItem(LANG_STORAGE_KEY);
    if (savedLang && TRANSLATIONS[savedLang]) {
      return savedLang;
    }
  } catch {
    // ignore storage errors
  }

  const browserLangs = (typeof navigator !== 'undefined')
    ? (Array.isArray(navigator.languages) && navigator.languages.length > 0
      ? navigator.languages
      : [navigator.language || ''])
    : [];

  for (const raw of browserLangs) {
    if (!raw || typeof raw !== 'string') continue;
    let base = raw.toLowerCase().split(/[-_]/)[0];
    // Handle legacy ISO code 'iw' for Hebrew
    if (base === 'iw') base = 'he';
    if (TRANSLATIONS[base]) {
      return base;
    }
  }

  return 'en';
}

let currentLang = detectInitialLanguage();

function getT() {
  return TRANSLATIONS[currentLang] || TRANSLATIONS.en;
}

const arenaCanvas = document.getElementById('arenaCanvas');
const nnCanvas = document.getElementById('nnCanvas');
const chartCanvas = document.getElementById('chartCanvas');
const sidebarPane = document.getElementById('sidebarPane');
const selectLanguage = document.getElementById('selectLanguage');

const renderer3D = new Renderer3D(arenaCanvas);
const nnRenderer = new NeuralGraphRenderer(nnCanvas);
const chartRenderer = new LearningChartRenderer(chartCanvas);
const logger = new ExperimentLogger();

let sim = null;
let paused = false;
let simSpeed = 1;

// DOM Elements
const selectMorphology = document.getElementById('selectMorphology');
const selectActuator = document.getElementById('selectActuator');
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

const hudUpdate = document.getElementById('hudUpdate');
const hudSimTime = document.getElementById('hudSimTime');
const hudTotalFood = document.getElementById('hudTotalFood');
const hudLeaderVel = document.getElementById('hudLeaderVel');
const hudFriction = document.getElementById('hudFriction');
const expStatusBadge = document.getElementById('expStatusBadge');

const ytPausePlayOverlay = document.getElementById('ytPausePlayOverlay');
const ytOverlayBtn = document.getElementById('ytOverlayBtn');
const ytIconPause = document.getElementById('ytIconPause');
const ytIconPlay = document.getElementById('ytIconPlay');
let ytOverlayTimer = null;
let isResumingTransition = false;

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
  const t = getT();
  if (preset === 'infinite' || val >= 0.54) return t.ambitionInfinite;
  if (val <= 0.21) return t.ambitionMin(val.toFixed(2));
  if (val <= 0.39) return t.ambitionBound(val.toFixed(2));
  return t.ambitionTurbo(val.toFixed(2));
}

const metricMeanVel = document.getElementById('metricMeanVel');
const metricMeanRew = document.getElementById('metricMeanRew');
const metricPower = document.getElementById('metricPower');
const learningChartBadge = document.getElementById('learningChartBadge');
const logsArchiveCard = document.getElementById('logsArchiveCard');
const logsContainer = document.getElementById('logsContainer');
const logsCountBadge = document.getElementById('logsCountBadge');

function formatFrictionLabel(mu) {
  const t = getT();
  if (mu < 0.05) return t.frictionZero(mu.toFixed(2));
  if (mu < 0.35) return t.frictionLow(mu.toFixed(2));
  if (mu < 0.75) return t.frictionMid(mu.toFixed(2));
  if (mu <= 1.15) return t.frictionNormal(mu.toFixed(2));
  return t.frictionHigh(mu.toFixed(2));
}

function getMorphDescriptionText(morphKey) {
  const t = getT();
  return morphKey === 'quadruped' ? t.morphDescQuadruped : t.morphDescTripod;
}

function renderLogsList(logs) {
  if (logsArchiveCard) {
    logsArchiveCard.hidden = !logger.enabled;
  }
  if (!logger.enabled) return;
  const t = getT();
  if (!logs || logs.length === 0) {
    logsCountBadge.textContent = t.logsCount(0);
    logsContainer.innerHTML = `<div class="log-entry"><span>${t.logsEmpty}</span></div>`;
    return;
  }
  logsCountBadge.textContent = t.logsCount(logs.length);
  logsContainer.innerHTML = logs.slice(0, 12).map(l => {
    const tag = l.filename === 'current_experiment.json' ? t.logTagCurrent : t.logTagArchive;
    const food = l.totalFoodCollected !== undefined ? `${t.logFoodShort} ${l.totalFoodCollected}` : '';
    const upd = l.updatesCount !== undefined ? `Upd: ${l.updatesCount}` : '';
    return `<div class="log-entry" title="${l.filename}">
      <span>${tag} <strong>${l.morphology || 'tripod'}/${(l.algorithm || 'ppo').toUpperCase()}</strong></span>
      <span>${upd} ${food}</span>
    </div>`;
  }).join('');
}

function translateMetaState(m, t) {
  if (!m) return { modeTitle: t.metaBannerLearningWalk, reasonText: '' };
  if (currentLang === 'ru' && m.modeTitle && m.reasonText) {
    return { modeTitle: m.modeTitle, reasonText: m.reasonText };
  }
  const leader = sim && sim.creatures ? sim.creatures[0] : null;
  const food = leader ? leader.foodCollected : 0;
  const minF = (sim && sim.trainer && sim.trainer.minFoodsToConsolidate) ? sim.trainer.minFoodsToConsolidate : 2;
  const vStr = (m.competenceEMA || 0).toFixed(2);
  const tStr = (m.targetCompetence || 0.18).toFixed(2);

  if (m.mode === 'LEARNING_RIGHTING') {
    const upY = leader ? leader.getTorsoFrame().up.y.toFixed(2) : '-0.50';
    return {
      modeTitle: t.metaBannerRighting,
      reasonText: t.metaReasonRighting(upY, m.rightingAttempts || 1, m.rightingSkillPct || 20)
    };
  }
  if (m.mode === 'CONSOLIDATED') {
    if (sim && sim.trainer && sim.trainer.manualFreeze) {
      return {
        modeTitle: t.metaBannerManualFrozen,
        reasonText: t.metaReasonManualFrozen
      };
    }
    return {
      modeTitle: t.metaBannerConsolidated,
      reasonText: t.metaReasonConsolidated(vStr, tStr, food)
    };
  }
  // LEARNING_WALK
  if (m.ambitionPreset === 'infinite' || (sim && sim.trainer && sim.trainer.alwaysLearn)) {
    return {
      modeTitle: t.metaBannerInfinite,
      reasonText: t.metaReasonInfinite(vStr, food)
    };
  }
  if ((m.targetCompetence || 0.18) >= 0.44) {
    return {
      modeTitle: t.metaBannerLearningTurbo,
      reasonText: t.metaReasonLearning(food, minF, vStr, tStr)
    };
  }
  if ((m.targetCompetence || 0.18) >= 0.25) {
    return {
      modeTitle: t.metaBannerLearningBound,
      reasonText: t.metaReasonLearning(food, minF, vStr, tStr)
    };
  }
  return {
    modeTitle: t.metaBannerLearningWalk,
    reasonText: t.metaReasonLearning(food, minF, vStr, tStr)
  };
}

function applyLanguage(lang, { persist = false } = {}) {
  if (!TRANSLATIONS[lang]) lang = 'en';
  currentLang = lang;
  if (persist) {
    try {
      localStorage.setItem(LANG_STORAGE_KEY, lang);
    } catch {
      // ignore
    }
  }
  const t = getT();
  document.documentElement.lang = lang;
  if (t.labTitle) {
    document.title = t.labTitle;
  }
  if (sidebarPane) {
    sidebarPane.dir = lang === 'he' ? 'rtl' : 'ltr';
  }
  if (selectLanguage && selectLanguage.value !== lang) {
    selectLanguage.value = lang;
  }

  for (const el of document.querySelectorAll('[data-i18n]')) {
    const key = el.getAttribute('data-i18n');
    if (typeof t[key] === 'string') {
      el.textContent = t[key];
    }
  }

  if (selectMorphology && morphDescription) {
    morphDescription.textContent = getMorphDescriptionText(selectMorphology.value);
  }
  if (rangeFriction && labelFrictionVal) {
    labelFrictionVal.textContent = formatFrictionLabel(parseFloat(rangeFriction.value));
  }
  if (rangeAmbition && labelAmbitionVal) {
    labelAmbitionVal.textContent = formatAmbitionLabel(parseFloat(rangeAmbition.value), currentAmbitionPreset);
  }
  if (btnPause) {
    btnPause.textContent = paused ? t.btnResume : t.btnPause;
  }
  if (expStatusBadge && !isFastTraining) {
    expStatusBadge.textContent = paused ? t.statusPaused : t.statusActive;
  }
  if (btnFastTrain && !isFastTraining) {
    btnFastTrain.textContent = t.btnFastTrain;
  }

  renderLogsList(logger.savedLogsList);
  if (sim) {
    const isLearningNow = !sim.trainer.metaState || sim.trainer.metaState.isLearningActive;
    chartRenderer.render(sim.history, isLearningNow, t);
    const leader = sim.creatures[sim.bestCreatureIdx || 0] || sim.creatures[0];
    nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null, t);
    updateHUD(true);
  }
}

if (selectLanguage) {
  selectLanguage.value = currentLang;
  selectLanguage.addEventListener('change', () => {
    applyLanguage(selectLanguage.value, { persist: true });
  });
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
  if (selectActuator && (ckpt.actuatorMode === 'servo' || ckpt.actuatorMode === 'muscle')) {
    selectActuator.value = ckpt.actuatorMode;
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

async function startNewExperiment({ clearSaved = false, resetDefaults = clearSaved } = {}) {
  if (clearSaved) {
    try {
      localStorage.removeItem(CREATURE_STORAGE_KEY);
    } catch {
      // ignore
    }
  }
  if (resetDefaults) {
    currentAmbitionPreset = 'min';
    if (rangeAmbition) {
      rangeAmbition.value = '0.18';
    }
    if (labelAmbitionVal) {
      labelAmbitionVal.textContent = formatAmbitionLabel(0.18, 'min');
    }
    if (ambitionButtons) {
      for (const b of ambitionButtons.querySelectorAll('button')) {
        b.classList.toggle('active', b.dataset.ambition === 'min');
      }
    }
    if (rangeFriction) {
      rangeFriction.value = '1.00';
    }
    if (labelFrictionVal) {
      labelFrictionVal.textContent = formatFrictionLabel(1.0);
    }
    if (hudFriction) {
      hudFriction.textContent = '1.00';
    }
  }
  const morphologyKey = selectMorphology.value;
  const actuatorMode = selectActuator ? selectActuator.value : 'servo';
  const algorithm = selectAlgorithm.value;
  const floorFriction = parseFloat(rangeFriction.value);

  const spec = MORPHOLOGY_REGISTRY[morphologyKey];
  morphDescription.textContent = getMorphDescriptionText(morphologyKey);
  nnDimLabel.textContent = `${spec.obsDim} → 32 → 24 → ${spec.actDim}`;

  sim = new ArenaSimulation({
    morphologyKey,
    actuatorMode,
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

  chartRenderer.render(sim.history, true, getT());

  await logger.startNewExperiment({
    morphology: morphologyKey,
    actuatorMode,
    algorithm,
    populationSize: 1,
    config: { floorFriction, arenaHalfSize: 11.0, actuatorMode }
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
    morphDescription.textContent = getMorphDescriptionText(ckpt.morphologyKey);
    nnDimLabel.textContent = `${spec.obsDim} → 32 → 24 → ${spec.actDim}`;

    sim = new ArenaSimulation({
      morphologyKey: ckpt.morphologyKey,
      actuatorMode: ckpt.actuatorMode || (selectActuator ? selectActuator.value : 'servo'),
      algorithm: ckpt.algorithm || 'ppo',
      populationSize: 1,
      arenaHalfSize: 11.0,
      floorFriction: typeof ckpt.floorFriction === 'number' ? ckpt.floorFriction : 1.0
    });
    sim.importCheckpoint(ckpt);
    renderer3D.lockedFoodPos.set(NaN, NaN, NaN);

    const t = getT();
    const isLearningNow = !sim.trainer.metaState || sim.trainer.metaState.isLearningActive;
    chartRenderer.render(sim.history, isLearningNow, t);
    const leader = sim.creatures[0];
    nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null, t);
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
    morphology: selectMorphology.value,
    actuatorMode: selectActuator ? selectActuator.value : 'servo',
    algorithm: selectAlgorithm.value
  };
}

// Event Listeners
selectMorphology.addEventListener('change', () => {
  logger.recordUserEvent('select_morphology', { morphology: selectMorphology.value }, sim, renderer3D, getUIState());
  startNewExperiment({ clearSaved: true });
});
if (selectActuator) {
  selectActuator.addEventListener('change', () => {
    logger.recordUserEvent('select_actuator', { actuatorMode: selectActuator.value }, sim, renderer3D, getUIState());
    startNewExperiment({ clearSaved: true });
  });
}
selectAlgorithm.addEventListener('change', () => {
  logger.recordUserEvent('select_algorithm', { algorithm: selectAlgorithm.value }, sim, renderer3D, getUIState());
  startNewExperiment({ clearSaved: true });
});
btnNewExperiment.addEventListener('click', () => {
  logger.recordUserEvent('btn_new_experiment', { reason: 'user_reset_ai' }, sim, renderer3D, getUIState());
  startNewExperiment({ clearSaved: true, resetDefaults: true });
});

function setPausedState(nextPaused, source = 'ui') {
  if (ytOverlayTimer) {
    clearTimeout(ytOverlayTimer);
    ytOverlayTimer = null;
  }
  isResumingTransition = false;
  paused = nextPaused;
  const t = getT();
  btnPause.textContent = paused ? t.btnResume : t.btnPause;
  expStatusBadge.textContent = paused ? t.statusPaused : t.statusActive;
  expStatusBadge.style.color = paused ? '#fbbf24' : '#4ade80';

  if (ytPausePlayOverlay && ytOverlayBtn) {
    ytOverlayBtn.classList.remove('resuming');
    if (paused) {
      // 1. Show transparent circle with large Pause icon (||) for 1 second
      ytPausePlayOverlay.dataset.icon = 'pause';
      if (ytIconPause) ytIconPause.style.display = 'block';
      if (ytIconPlay) ytIconPlay.style.display = 'none';
      ytPausePlayOverlay.hidden = false;
      ytOverlayTimer = setTimeout(() => {
        ytOverlayTimer = null;
        if (paused && !isResumingTransition) {
          // 2. Switch to circle with Play triangle (▶) after 1 second
          ytPausePlayOverlay.dataset.icon = 'play';
          if (ytIconPause) ytIconPause.style.display = 'none';
          if (ytIconPlay) ytIconPlay.style.display = 'block';
        }
      }, 1000);
    } else {
      ytPausePlayOverlay.hidden = true;
    }
  }

  saveCreatureCheckpoint();
  logger.recordUserEvent(paused ? 'pause' : 'resume', { source }, sim, renderer3D, getUIState());
}

function triggerResumeFromOverlay(source = 'yt_play_circle') {
  if (!paused || isResumingTransition) return;
  if (ytOverlayTimer) {
    clearTimeout(ytOverlayTimer);
    ytOverlayTimer = null;
  }
  isResumingTransition = true;
  if (ytPausePlayOverlay && ytOverlayBtn) {
    ytPausePlayOverlay.dataset.icon = 'play';
    if (ytIconPause) ytIconPause.style.display = 'none';
    if (ytIconPlay) ytIconPlay.style.display = 'block';
    ytPausePlayOverlay.hidden = false;
    ytOverlayBtn.classList.add('resuming');
  }
  ytOverlayTimer = setTimeout(() => {
    ytOverlayTimer = null;
    simTimeAccum = 0;
    lastFrameTime = performance.now();
    setPausedState(false, source);
  }, 1000);
}

function handleUpdateEntry(updateEntry) {
  if (!updateEntry) return;
  const isLearningNow = !sim.trainer.metaState || sim.trainer.metaState.isLearningActive;
  chartRenderer.render(sim.history, isLearningNow, getT());
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
  if (paused) {
    triggerResumeFromOverlay('btn_pause');
  } else {
    setPausedState(true, 'btn_pause');
  }
});

if (ytOverlayBtn) {
  ytOverlayBtn.addEventListener('click', e => {
    e.stopPropagation();
    if (paused) {
      triggerResumeFromOverlay('yt_play_circle');
    } else {
      setPausedState(true, 'yt_pause_circle');
    }
  });
}

// Single click on the 3D arena canvas toggles pause/resume
renderer3D.onTogglePause = (source = 'canvas_click') => {
  if (paused) {
    triggerResumeFromOverlay(source);
  } else {
    setPausedState(true, source);
  }
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
  nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null, getT());
  updateHUD();
});

let isFastTraining = false;

async function runFastHeadlessTraining(targetUpdates = 25) {
  if (!sim || isFastTraining) return;
  isFastTraining = true;
  btnFastTrain.disabled = true;
  expStatusBadge.textContent = getT().statusFastTrain;
  expStatusBadge.style.color = '#38bdf8';

  const c0 = sim.creatures[0];
  sim._isHeadlessFastTrain = true;
  for (let u = 1; u <= targetUpdates; u++) {
    let updateEntry = null;
    while (!updateEntry) {
      updateEntry = sim.stepControl();
    }
    handleUpdateEntry(updateEntry);
    btnFastTrain.textContent = getT().btnFastTrainProgress(u, targetUpdates);

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
  btnFastTrain.textContent = getT().btnFastTrain;
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
    updateHUD();
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
  const t = getT();
  const leader = sim.creatures[0];

  hudUpdate.textContent = sim.trainer.updateCount;
  hudSimTime.textContent = `${Math.floor(sim.simTime)} ${t.unitSec}`;
  hudTotalFood.textContent = leader.foodCollected;

  const now = performance.now();
  if (forceVelUpdate || now - lastVelHudUpdateNow >= 280) {
    lastVelHudUpdateNow = now;
    hudLeaderVel.textContent = `${(leader.smoothedVelocityTowardFood || 0).toFixed(2)} ${t.unitMps}`;
  }

  const m = sim.trainer.metaState;
  const isLearningNow = !m || m.isLearningActive;

  if (learningChartBadge) {
    learningChartBadge.textContent = isLearningNow ? t.chartBadgeRecording : t.chartBadgeFrozen;
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

    metricMeanVel.textContent = `${meanVel.toFixed(3)} ${t.unitMps}`;
    metricMeanRew.textContent = meanRew.toFixed(3);
    metricPower.textContent = `${leader.lastMechanicalPower.toFixed(1)} ${t.unitWatt}`;
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
      ? t.metaBadgeConsolidated
      : m.mode === 'LEARNING_RIGHTING'
        ? t.metaBadgeRighting
        : t.metaBadgeLearning;

    const { modeTitle, reasonText } = translateMetaState(m, t);
    metaModeBanner.className = `meta-mode-banner ${modeClass}`;
    metaModeBanner.textContent = modeTitle;

    metaNeedVal.textContent = `${m.needToLearnPct}%`;
    metaNeedFill.style.inlineSize = `${Math.max(4, Math.min(100, m.needToLearnPct))}%`;

    metaRightingVal.textContent = t.metaRightingStat(m.rightingSkillPct, m.rightingSuccesses);
    metaRightingFill.style.inlineSize = `${Math.max(8, Math.min(100, m.rightingSkillPct))}%`;

    metaReasonBox.textContent = reasonText;
    metaReasonBox.style.borderInlineStartColor = m.mode === 'CONSOLIDATED'
      ? '#38bdf8'
      : m.mode === 'LEARNING_RIGHTING'
        ? '#f59e0b'
        : '#34d399';

    if (btnToggleFreeze) {
      const isCons = m.mode === 'CONSOLIDATED';
      btnToggleFreeze.textContent = isCons
        ? t.btnUnfreezeNow
        : t.btnFreezeNow;
      btnToggleFreeze.classList.toggle('emerald', isCons);
    }
  }

  // Highlight the "Help Stand Up" button if the walker is tilted onto its side or back
  const { up } = leader.getTorsoFrame();
  const isOverturned = up.y < 0.35;
  btnUprightWalker.classList.toggle('amber-alert', isOverturned);
  btnUprightWalker.textContent = isOverturned
    ? t.btnUprightAlert
    : t.btnUpright;
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
    nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null, getT());
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

applyLanguage(currentLang);
restoreOrStartExperiment().then(() => {
  applyLanguage(currentLang);
  requestAnimationFrame(animate);
});
