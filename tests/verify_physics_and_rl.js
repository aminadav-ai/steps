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
if (iceSim.actuatorMode !== 'servo' || c0.actuatorMode !== 'servo' || c0.kneeNodes.length !== 3) {
  console.error('ОШИБКА: По умолчанию во 2-й версии должны быть включены жёсткие сервоприводы (servo) с физическими узлами коленей!');
  process.exit(1);
}
console.log(`Режим приводов по умолчанию: ${c0.actuatorMode} (узлов коленей: ${c0.kneeNodes.length}, всего физических узлов: ${c0.body.nodes.length})`);
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
  trainSim._isHeadlessFastTrain = true;
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
trainSim._isHeadlessFastTrain = false;

console.log('\n=== 2б. Проверка 3-секундной анимации шарика еды (1с растворение -> 1с пауза -> 1с падение с неба) ===');
const wFood = trainSim.creatures[0];
// Поместим шарик еды прямо перед носом шагохода для проверки фаз анимации
const nosePos = wFood.getTorsoFrame().center;
wFood.foodPos.set(nosePos.x, 0.16, nosePos.z);
wFood.foodState = 'ACTIVE';
wFood.foodStateTimer = 0;
trainSim.stepControl();
console.log(`Сразу после поедания: foodState=${wFood.foodState}, dissolvePos=(${wFood.dissolvePos.x.toFixed(2)}, ${wFood.dissolvePos.z.toFixed(2)})`);
if (wFood.foodState !== 'DISSOLVING') {
  console.error('ОШИБКА: После поедания должен включаться режим DISSOLVING');
  process.exit(1);
}
for (let i = 0; i < 20; i++) trainSim.stepControl(); // +1.0s
console.log(`Через 1.0 с (конец растворения): foodState=${wFood.foodState}`);
if (wFood.foodState !== 'WAITING') {
  console.error('ОШИБКА: После 1.0 с растворения должна включаться 1-секундная пауза WAITING');
  process.exit(1);
}
for (let i = 0; i < 10; i++) trainSim.stepControl(); // +0.5s into WAITING
for (let i = 0; i < 12; i++) trainSim.stepControl(); // into DROPPING
console.log(`Через 2.1 с (начало падения с неба): foodState=${wFood.foodState}, высота шарика y=${wFood.foodPos.y.toFixed(2)} м`);
if (wFood.foodState !== 'DROPPING' || wFood.foodPos.y <= 1.0) {
  console.error('ОШИБКА: Новый шарик должен падать сверху с неба в фазе DROPPING');
  process.exit(1);
}
for (let i = 0; i < 20; i++) trainSim.stepControl(); // finish drop
console.log(`Через 3.1 с (приземление шарика): foodState=${wFood.foodState}, высота шарика y=${wFood.foodPos.y.toFixed(2)} м`);
if (wFood.foodState !== 'ACTIVE' || Math.abs(wFood.foodPos.y - 0.16) > 1e-3) {
  console.error('ОШИБКА: После падения шарик должен перейти в состояние ACTIVE на высоте 0.16 м');
  process.exit(1);
}
console.log('ОК: Трёхфазный цикл шарика еды (1с растворение -> 1с пауза -> 1с падение с неба) работает штатно.');

console.log('\n=== 3. Проверка «Рефлекса жука» (обучение перевороту со спины на ноги без участия еды) ===');
const walker = trainSim.creatures[0];
const com = walker.body.getCenterOfMass();
const { right, up } = walker.getTorsoFrame();
const flipAngle = Math.PI * 0.94;
const cosA = Math.cos(flipAngle);
const sinA = Math.sin(flipAngle);
for (const n of walker.body.nodes) {
  const rx = n.pos.x - com.x;
  const ry = n.pos.y - com.y;
  const rz = n.pos.z - com.z;
  // Project onto local (right, up) plane and rotate by flipAngle around local forward axis
  const uCoord = rx * right.x + ry * right.y + rz * right.z;
  const vCoord = rx * up.x + ry * up.y + rz * up.z;
  const fCoordX = rx - uCoord * right.x - vCoord * up.x;
  const fCoordY = ry - uCoord * right.y - vCoord * up.y;
  const fCoordZ = rz - uCoord * right.z - vCoord * up.z;
  const newU = uCoord * cosA - vCoord * sinA;
  const newV = uCoord * sinA + vCoord * cosA;
  n.pos.set(
    com.x + fCoordX + newU * right.x + newV * up.x,
    0.35 + fCoordY + newU * right.y + newV * up.y,
    com.z + fCoordZ + newU * right.z + newV * up.z
  );
  n.prevPos.copy(n.pos);
}

let recoveredFromBack = false;
for (let step = 1; step <= 70; step++) {
  trainSim.stepControl();
  const upY = walker.getTorsoFrame().up.y;
  const invLegs = walker.hasInvertedLeg();
  const m = trainSim.trainer.metaState;
  if (step === 1 || step % 14 === 0 || (upY >= 0.68 && !invLegs)) {
    console.log(`Шаг на спине #${step} | up.y=${upY.toFixed(3)} | invLegs=${invLegs} | режим=${m.mode} | попытка=${m.rightingAttempts} | навык=${m.rightingSkillPct}%`);
    if (upY >= 0.68 && !invLegs && step > 5) {
      console.log('ОК: Шагоход самостоятельно перевернулся со спины и расправил все лапы под корпус!');
      recoveredFromBack = true;
      break;
    }
  }
}
if (!recoveredFromBack) {
  console.error('ОШИБКА: Шагоход не завершил переворот или оставил лапы запрокинутыми наверх!');
  process.exit(1);
}

// Тест 3б: Искусственно выворачиваем колени вниз в "V"-прогиб и запрокидываем стопы выше бедра при вертикальном корпусе (как на скриншоте пользователя)
walker.footNodes[0].pos.y = walker.hipNodes[0].pos.y + 0.16;
walker.footNodes[0].prevPos.copy(walker.footNodes[0].pos);
walker.kneeNodes[0].pos.y = walker.hipNodes[0].pos.y - 0.18;
walker.kneeNodes[0].prevPos.copy(walker.kneeNodes[0].pos);
walker.footNodes[2].pos.y = walker.hipNodes[2].pos.y + 0.16;
walker.footNodes[2].prevPos.copy(walker.footNodes[2].pos);
walker.kneeNodes[2].pos.y = walker.hipNodes[2].pos.y - 0.18;
walker.kneeNodes[2].prevPos.copy(walker.kneeNodes[2].pos);

trainSim.stepControl();
console.log(`После выворачивания коленей/стоп при вертикальном корпусе: hasInvertedLeg=${walker.hasInvertedLeg()}, mode=${trainSim.trainer.metaState.mode}`);
if (walker.hasInvertedLeg() || trainSim.trainer.metaState.mode === 'LEARNING_RIGHTING') {
  console.error('ОШИБКА: Механические упоры суставов должны мгновенно блокировать выворачивание лап/коленей без ложного включения LEARNING_RIGHTING!');
  process.exit(1);
}
console.log('ОК: Механические упоры суставов (Hip & Knee Hard-Stops) мгновенно блокируют выворачивание лап/коленей.');

// Тест 3в: Проверка, что ожидание нового шарика еды и разворот к нему НЕ включают ложное дообучение из CONSOLIDATED
trainSim.uprightCreature(0);
trainSim.trainer.wasOverturned = false;
trainSim.trainer.metaState.mode = 'CONSOLIDATED';
trainSim.trainer.metaState.isLearningActive = false;
trainSim.trainer.metaState.competenceEMA = 0.22;
trainSim.trainer.metaState.stallStepsCount = 0;
trainSim.trainer.metaState.lowVelEpochsCount = 0;
// Ставим шарик еды позади шагохода (разворот на ~160 градусов после поедания предыдущего шарика)
const { forward: wFwd } = walker.getTorsoFrame();
const wCom = walker.body.getCenterOfMass();
walker.foodPos.set(wCom.x - wFwd.x * 1.8, walker.foodRadius, wCom.z - wFwd.z * 1.8);
walker.postFoodTurnGraceTimer = 6.0;
walker.smoothedVelocityTowardFood = 0.01;
for (let s = 0; s < 30; s++) {
  walker.foodPos.set(
    walker.body.getCenterOfMass().x - walker.getTorsoFrame().forward.x * 1.8,
    walker.foodRadius,
    walker.body.getCenterOfMass().z - walker.getTorsoFrame().forward.z * 1.8
  );
  walker.smoothedVelocityTowardFood = 0.01;
  trainSim.stepControl();
}
console.log(`Во время разворота к новой цели позади шагохода (30 шагов): mode=${trainSim.trainer.metaState.mode}, isLearningActive=${trainSim.trainer.metaState.isLearningActive}`);
if (trainSim.trainer.metaState.mode !== 'CONSOLIDATED' || trainSim.trainer.metaState.isLearningActive) {
  console.error('ОШИБКА: Во время паузы между целями или разворота к новой цели дообучение НЕ должно ложно включаться!');
  process.exit(1);
}
console.log('ОК: Пауза между шариками и разворот к новой цели не вызывают ложного пробуждения дообучения.');

console.log('\n=== 4. Проверка перетаскивания шарика еды и броска шагохода мышью ===');
if (trainSim.trainer.metaState.ambitionPreset !== 'min' || trainSim.trainer.metaState.targetCompetence !== 0.18) {
  console.error('ОШИБКА: По умолчанию амбиция обучения должна быть min (0.18 м/с)!');
  process.exit(1);
}
console.log(`ОК: Амбиция обучения по умолчанию = ${trainSim.trainer.metaState.ambitionPreset} (${trainSim.trainer.metaState.targetCompetence} м/с)`);

trainSim.moveFoodTo(0, -2.5, 3.2, false);
console.log(`Шарик еды переставлен мышью в: (${walker.foodPos.x.toFixed(2)}, ${walker.foodPos.z.toFixed(2)})`);
trainSim.grabCreature(0);
trainSim.dragCreatureTo(0, 1.5, -1.2, 2.2, -1.8);
trainSim.releaseCreature(0, 2.2, -1.8);
for (let i = 0; i < 6; i++) trainSim.stepControl();
const afterThrowCom = walker.body.getCenterOfMass();
console.log(`Шагоход схвачен и брошен мышью в новую точку: (${afterThrowCom.x.toFixed(2)}, ${afterThrowCom.z.toFixed(2)}), сглаженная скорость HUD=${walker.smoothedVelocityTowardFood.toFixed(2)} м/с`);

console.log('\n=== 5. Проверка пробуждения дообучения при включении скользкого льда и падении скорости ===');
trainSim.uprightCreature(0);
trainSim.trainer.metaState.mode = 'CONSOLIDATED';
trainSim.trainer.metaState.isLearningActive = false;
trainSim.setFloorFriction(0.15);
console.log(`После перевода ползунка трения на скользкий лёд (mu=0.15): mode=${trainSim.trainer.metaState.mode}, isLearningActive=${trainSim.trainer.metaState.isLearningActive}`);
if (trainSim.trainer.metaState.mode !== 'LEARNING_WALK' || !trainSim.trainer.metaState.isLearningActive) {
  console.error('ОШИБКА: При включении скользкого льда дообучение должно сразу пробуждаться!');
  process.exit(1);
}

// Проверим также автоматическое пробуждение мета-чутья чисто по падению скорости из CONSOLIDATED при движении прямо к цели
trainSim.trainer.metaState.mode = 'CONSOLIDATED';
trainSim.trainer.metaState.isLearningActive = false;
trainSim.trainer.metaState.stallStepsCount = 0;
trainSim.trainer.metaState.lowVelEpochsCount = 0;
walker.postFoodTurnGraceTimer = 0;
walker.isPausedOrTurningToFood = false;
walker.smoothedVelocityTowardFood = 0.05;
for (let s = 0; s < 32; s++) {
  walker.postFoodTurnGraceTimer = 0;
  trainSim.stepControl();
}
console.log(`После просадки скорости на льду: mode=${trainSim.trainer.metaState.mode}, isLearningActive=${trainSim.trainer.metaState.isLearningActive}, comp=${trainSim.trainer.metaState.competenceEMA} м/с`);
if (trainSim.trainer.metaState.mode !== 'LEARNING_WALK' || !trainSim.trainer.metaState.isLearningActive) {
  console.error('ОШИБКА: При падении скорости на скользком льду мета-чутьё должно автоматически включать дообучение!');
  process.exit(1);
}
console.log('ОК: Дообучение надёжно пробуждается при переходе на скользкий лёд и при просадке скорости.');

console.log('\n=== 6. Проверка переключателя режима приводов: servo (по умолчанию) <-> muscle (v1) ===');
const leg0 = walker.legControllers[0];
const { right: tRight } = walker.getTorsoFrame();
const maxBearingDeflection = Math.max(
  ...walker.hipNodes.map((h, i) => {
    const kErr = Math.abs((walker.kneeNodes[i].pos.x - h.pos.x) * tRight.x + (walker.kneeNodes[i].pos.y - h.pos.y) * tRight.y + (walker.kneeNodes[i].pos.z - h.pos.z) * tRight.z);
    const fErr = Math.abs((walker.footNodes[i].pos.x - h.pos.x) * tRight.x + (walker.footNodes[i].pos.y - h.pos.y) * tRight.y + (walker.footNodes[i].pos.z - h.pos.z) * tRight.z);
    return Math.max(kErr, fErr);
  })
);
console.log(`Сервопривод #0: isServo=${leg0.isServo}, угол=${(leg0.jointAngleRad * 180 / Math.PI).toFixed(1)}°, момент=${leg0.jointTorqueNm.toFixed(2)} Н·м (нагрузка ${(leg0.torqueLoadRatio * 100).toFixed(1)}%), боковой люфт подшипника=${(maxBearingDeflection * 1000).toFixed(2)} мм`);
if (!leg0.isServo || typeof leg0.jointTorqueNm !== 'number' || maxBearingDeflection > 0.02) {
  console.error('ОШИБКА: В режиме servo должна рассчитываться телеметрия крутящего момента (Н·м) и соблюдаться жёсткость плоскости подшипника!');
  process.exit(1);
}

const muscleSim = new ArenaSimulation({
  morphologyKey: 'tripod',
  actuatorMode: 'muscle',
  algorithm: 'ppo',
  populationSize: 1,
  floorFriction: 0.0
});
const m0 = muscleSim.creatures[0];
if (m0.actuatorMode !== 'muscle' || m0.kneeNodes.length !== 0 || m0.legControllers[0].isServo !== false) {
  console.error('ОШИБКА: При выборе режима muscle должна создаваться классическая био-мышечная конструкция v1!');
  process.exit(1);
}
console.log(`Классический режим приводов: ${m0.actuatorMode} (узлов коленей: ${m0.kneeNodes.length}, всего физических узлов: ${m0.body.nodes.length})`);

console.log('\nИтог проверки обучения:');
console.log(`Старт: Vel=${firstMetrics.meanVelocity.toFixed(3)} m/s, Food=${firstMetrics.c0Food}`);
console.log(`После 25 эпох (1 шагоход): Vel=${lastMetrics.meanVelocity.toFixed(3)} m/s, Food=${lastMetrics.c0Food}, Mode=${lastMetrics.metaMode}`);
