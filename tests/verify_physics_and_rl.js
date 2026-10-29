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

console.log('\n=== 2. Проверка обучения одиночного шагохода (populationSize = 1, Мета-Чутьё + PPO) ===');
const trainSim = new ArenaSimulation({
  morphologyKey: 'tripod',
  algorithm: 'ppo',
  populationSize: 1,
  floorFriction: 1.0
});

let firstMetrics = null;
let lastMetrics = null;

for (let s = 0; s < 32 * 25; s++) {
  const upd = trainSim.stepControl();
  if (upd) {
    if (!firstMetrics) firstMetrics = upd;
    lastMetrics = upd;
    if (upd.update % 5 === 0 || upd.update === 1) {
      console.log(
        `Update #${upd.update} | t=${upd.simTime}s | Mode=${upd.metaMode} (active=${upd.isLearningActive}, need=${upd.needToLearnPct}%) | Vel=${upd.meanVelocity.toFixed(3)} m/s | Food=${upd.c0Food}`
      );
    }
  }
}

console.log('\n=== 3. Проверка «Рефлекса жука» (обучение перевороту со спины на ноги без участия еды) ===');
const walker = trainSim.creatures[0];
const com = walker.body.getCenterOfMass();
const flipAngle = Math.PI * 0.94;
for (const n of walker.body.nodes) {
  const dy = n.pos.y - com.y;
  const dz = n.pos.z - com.z;
  n.pos.y = 0.35 + dy * Math.cos(flipAngle) - dz * Math.sin(flipAngle);
  n.pos.z = com.z + dy * Math.sin(flipAngle) + dz * Math.cos(flipAngle);
  n.prevPos.copy(n.pos);
}

for (let step = 1; step <= 60; step++) {
  trainSim.stepControl();
  const upY = walker.getTorsoFrame().up.y;
  const m = trainSim.trainer.metaState;
  if (step === 1 || step % 14 === 0 || upY > 0.65) {
    console.log(`Шаг на спине #${step} | up.y=${upY.toFixed(3)} | режим=${m.mode} | попытка=${m.rightingAttempts} | навык=${m.rightingSkillPct}%`);
    if (upY > 0.65 && step > 5) {
      console.log('ОК: Шагоход самостоятельно обучился рычагу и перевернулся со спины обратно на лапы!');
      break;
    }
  }
}

console.log('\nИтог проверки обучения:');
console.log(`Старт: Vel=${firstMetrics.meanVelocity.toFixed(3)} m/s, Food=${firstMetrics.c0Food}`);
console.log(`После 25 эпох (1 шагоход): Vel=${lastMetrics.meanVelocity.toFixed(3)} m/s, Food=${lastMetrics.c0Food}, Mode=${lastMetrics.metaMode}`);
