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

  chartRenderer.render(sim.history);

  await logger.startNewExperiment({
    morphology: morphologyKey,
    algorithm,
    populationSize: 16,
    config: { floorFriction, arenaHalfSize: 11.0 }
  });
  renderLogsList(logger.savedLogsList);
}

// Event Listeners
selectMorphology.addEventListener('change', () => startNewExperiment());
selectAlgorithm.addEventListener('change', () => startNewExperiment());
btnNewExperiment.addEventListener('click', () => startNewExperiment());

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.textContent = paused ? 'Продолжить' : 'Пауза';
  expStatusBadge.textContent = paused ? '⏸ ПАУЗА' : '● АКТИВЕН';
  expStatusBadge.style.color = paused ? '#fbbf24' : '#4ade80';
});

speedButtons.addEventListener('click', e => {
  const btn = e.target.closest('button[data-speed]');
  if (!btn) return;
  simSpeed = parseInt(btn.dataset.speed, 10) || 1;
  for (const b of speedButtons.querySelectorAll('button')) {
    b.classList.toggle('active', b === btn);
  }
});

rangeFriction.addEventListener('input', () => {
  const mu = parseFloat(rangeFriction.value);
  labelFrictionVal.textContent = formatFrictionLabel(mu);
  hudFriction.textContent = mu.toFixed(2);
  if (sim) {
    sim.setFloorFriction(mu);
  }
});

chkFollowLeader.addEventListener('change', () => {
  renderer3D.followLeader = chkFollowLeader.checked;
});

chkShowForces.addEventListener('change', () => {
  renderer3D.showForceVectors = chkShowForces.checked;
});

chkShowAllPop.addEventListener('change', () => {
  renderer3D.showAllPopulation = chkShowAllPop.checked;
});

function updateHUD() {
  if (!sim) return;
  const leaderIdx = sim.bestCreatureIdx || 0;
  const leader = sim.creatures[leaderIdx] || sim.creatures[0];

  hudUpdate.textContent = sim.trainer.updateCount;
  hudSimTime.textContent = `${sim.simTime.toFixed(1)} с`;
  hudTotalFood.textContent = sim.totalFoodCollected;
  hudLeaderId.textContent = leaderIdx;
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
}

let frameCounter = 0;
function animate() {
  requestAnimationFrame(animate);
  if (!sim) return;

  if (!paused) {
    for (let i = 0; i < simSpeed; i++) {
      const updateEntry = sim.stepControl();
      if (updateEntry) {
        chartRenderer.render(sim.history);
        logger.logUpdate(
          updateEntry,
          sim.morphologyKey,
          sim.algorithm,
          {
            update: updateEntry.update,
            totalFood: updateEntry.totalFood,
            meanVelocity: updateEntry.meanVelocity
          }
        ).then(() => {
          if (updateEntry.update % 3 === 0) {
            logger.refreshLogsList().then(list => renderLogsList(list));
          }
        });
      }
    }
  }

  renderer3D.render(sim);
  frameCounter++;
  if (frameCounter % 2 === 0) {
    const leader = sim.creatures[sim.bestCreatureIdx || 0];
    nnRenderer.render(sim.brain, leader ? leader.logicalDoFs : null);
    updateHUD();
  }
}

startNewExperiment().then(() => {
  requestAnimationFrame(animate);
});
