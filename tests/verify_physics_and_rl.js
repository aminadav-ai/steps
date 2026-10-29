import { ArenaSimulation } from '../src/physics/ArenaSimulation.js';

console.log('=== 1. Проверка законов сохранения импульса (чистая физика без анимаций) ===');

// Тест А: На абсолютно скользком льду (floorFriction = 0) внутренние мышцы НЕ МОГУТ сдвинуть центр масс!
const iceSim = new ArenaSimulation({
  morphologyKey: 'tripod',
  algorithm: 'ppo',
  populationSize: 1,
  floorFriction: 0.0
});

const c0 = iceSim.creatures[0];
const initialCom = c0.body.getCenterOfMass();

for (let step = 0; step < 60; step++) {
  // Подаём сильные хаотические сигналы на все 6 мышц
  const fakeActions = [
    Math.sin(step * 0.5),
    Math.cos(step * 0.5),
    -Math.sin(step * 0.5),
    Math.sin(step * 0.8),
    Math.cos(step * 0.3),
    -Math.cos(step * 0.7)
  ];
  c0.applyActionVector(fakeActions);
  iceSim.world.step(0.05, body => {
    c0.applyActionVector(fakeActions);
  });
}

const afterIceCom = c0.body.getCenterOfMass();
const horizontalDriftOnIce = Math.hypot(afterIceCom.x - initialCom.x, afterIceCom.z - initialCom.z);
console.log(`Дрейф центра масс при трении mu = 0.0 (60 шагов активного дёрганья лап): ${horizontalDriftOnIce.toExponential(4)} м`);
if (horizontalDriftOnIce > 1e-6) {
  console.error('ОШИБКА: Нарушен закон сохранения импульса!');
  process.exit(1);
} else {
  console.log('ОК: Закон сохранения импульса выполняется с машинной точностью (движение невозможно без трения опоры о пол).');
}

console.log('\n=== 2. Проверка обучения PPO (Actor-Critic + GAE + Adam) на арене с трением mu = 1.0 ===');
const trainSim = new ArenaSimulation({
  morphologyKey: 'tripod',
  algorithm: 'ppo',
  populationSize: 16,
  floorFriction: 1.0
});

let firstMetrics = null;
let lastMetrics = null;

for (let s = 0; s < 32 * 12; s++) {
  const upd = trainSim.stepControl();
  if (upd) {
    if (!firstMetrics) firstMetrics = upd;
    lastMetrics = upd;
    if (upd.update % 3 === 0 || upd.update === 1) {
      console.log(
        `Update #${upd.update} | t=${upd.simTime}s | MeanRew=${upd.meanReward.toFixed(3)} | VelTowardFood=${upd.meanVelocity.toFixed(3)} m/s | TotalFood=${upd.totalFood}`
      );
    }
  }
}

console.log('\nИтог проверки обучения:');
console.log(`Старт: Vel=${firstMetrics.meanVelocity.toFixed(3)} m/s, Food=${firstMetrics.totalFood}`);
console.log(`После 12 обновлений PPO: Vel=${lastMetrics.meanVelocity.toFixed(3)} m/s, Food=${lastMetrics.totalFood}`);
