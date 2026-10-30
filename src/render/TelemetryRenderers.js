/**
 * Real-time Neural Network Graph & Learning Curve Chart Renderers
 */

const INPUT_OBS_INDICES = [0, 1, 2, 3, 6, 10, 11, 12, 13, 14, 15];
const DEFAULT_INPUT_LABELS = [
  'cos(α) Еда',
  'sin(α) Еда',
  'Дист. Еда',
  'Скор. V_fwd',
  'Осанка Up_y',
  'CPG sin(ωt)',
  'CPG cos(ωt)',
  'Тяга×sin',
  'Тяга×cos',
  'Руль×sin',
  'Руль×cos'
];

export class NeuralGraphRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  render(brain, logicalDoFs, t = null) {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(260, Math.floor(rect.width * dpr));
    const h = Math.max(180, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    const obs = brain.lastObs;
    const h1 = brain.lastH1;
    const h2 = brain.lastH2;
    const mu = brain.lastMu;
    const val = brain.lastValue;

    const colX = [
      w * 0.19, // Inputs
      w * 0.43, // Hidden 1 (subset of 10 nodes shown)
      w * 0.64, // Hidden 2 (subset of 8 nodes shown)
      w * 0.85  // Motor Outputs + Critic V(s)
    ];

    const labels = (t && t.nnInLabels) ? t.nnInLabels : DEFAULT_INPUT_LABELS;
    const inNodes = INPUT_OBS_INDICES.map((obsIdx, i) => ({
      x: colX[0],
      y: ((i + 0.8) / (INPUT_OBS_INDICES.length + 0.6)) * h,
      val: obs[obsIdx] || 0,
      label: labels[i] || DEFAULT_INPUT_LABELS[i],
      obsIdx
    }));

    const h1Count = 10;
    const h1Nodes = Array.from({ length: h1Count }, (_, i) => ({
      x: colX[1],
      y: ((i + 0.8) / (h1Count + 0.6)) * h,
      val: h1[i] || 0
    }));

    const h2Count = 8;
    const h2Nodes = Array.from({ length: h2Count }, (_, i) => ({
      x: colX[2],
      y: ((i + 0.8) / (h2Count + 0.6)) * h,
      val: h2[i] || 0
    }));

    const outCount = brain.actDim;
    const outNodes = Array.from({ length: outCount }, (_, a) => {
      const dofObj = logicalDoFs && logicalDoFs[a] ? logicalDoFs[a] : null;
      let outLabel;
      if (dofObj && dofObj.shortLabel) {
        outLabel = dofObj.shortLabel;
      } else {
        const legNum = Math.floor(a / 2) + 1;
        const isLift = a % 2 === 0;
        outLabel = t
          ? (isLift && typeof t.nnOutLift === 'function' ? t.nnOutLift(legNum) : (!isLift && typeof t.nnOutStride === 'function' ? t.nnOutStride(legNum) : `Out ${a + 1}`))
          : (dofObj ? dofObj.name.replace('Лапа ', 'Л').replace(' (Подъём/Опора)', '').replace(' (Шаг вперёд/назад)', '') : `Out ${a + 1}`);
      }
      return {
        x: colX[3],
        y: ((a + 0.65) / (outCount + 1.35)) * h,
        val: mu[a] || 0,
        label: outLabel
      };
    });

    // 1. Draw Skip-Connection Synapses (actorSkip.W) from Inputs directly to Motor Outputs
    for (let a = 0; a < outCount; a++) {
      const row = a * brain.obsDim;
      for (let i = 0; i < inNodes.length; i++) {
        const weight = brain.actorSkip.W[row + inNodes[i].obsIdx] || 0;
        if (Math.abs(weight) < 0.04) continue;
        const alpha = Math.min(0.85, Math.abs(weight) * 1.8);
        ctx.strokeStyle = weight >= 0
          ? `rgba(52, 211, 153, ${alpha})`
          : `rgba(251, 113, 133, ${alpha})`;
        ctx.lineWidth = Math.min(2.8, 0.8 + Math.abs(weight) * 3.2);
        ctx.beginPath();
        ctx.moveTo(inNodes[i].x, inNodes[i].y);
        ctx.bezierCurveTo(
          colX[1],
          inNodes[i].y,
          colX[2],
          outNodes[a].y,
          outNodes[a].x,
          outNodes[a].y
        );
        ctx.stroke();
      }
    }

    // 2. Draw MLP Layer Connections (subtle background web)
    ctx.lineWidth = 0.75;
    for (let i = 0; i < inNodes.length; i++) {
      for (let j = 0; j < h1Nodes.length; j += 2) {
        const wVal = brain.actorL1.W[j * brain.obsDim + inNodes[i].obsIdx] || 0;
        if (Math.abs(wVal) < 0.12) continue;
        ctx.strokeStyle = wVal > 0 ? 'rgba(56, 189, 248, 0.16)' : 'rgba(244, 63, 94, 0.16)';
        ctx.beginPath();
        ctx.moveTo(inNodes[i].x, inNodes[i].y);
        ctx.lineTo(h1Nodes[j].x, h1Nodes[j].y);
        ctx.stroke();
      }
    }

    for (let j = 0; j < h1Nodes.length; j++) {
      for (let k = 0; k < h2Nodes.length; k++) {
        const wVal = brain.actorL2.W[k * brain.hidden1 + j] || 0;
        if (Math.abs(wVal) < 0.14) continue;
        ctx.strokeStyle = wVal > 0 ? 'rgba(56, 189, 248, 0.18)' : 'rgba(244, 63, 94, 0.18)';
        ctx.beginPath();
        ctx.moveTo(h1Nodes[j].x, h1Nodes[j].y);
        ctx.lineTo(h2Nodes[k].x, h2Nodes[k].y);
        ctx.stroke();
      }
    }

    for (let k = 0; k < h2Nodes.length; k++) {
      for (let a = 0; a < outNodes.length; a++) {
        const wVal = brain.actorOut.W[a * brain.hidden2 + k] || 0;
        if (Math.abs(wVal) < 0.05) continue;
        ctx.strokeStyle = wVal > 0 ? 'rgba(52, 211, 153, 0.28)' : 'rgba(251, 113, 133, 0.28)';
        ctx.beginPath();
        ctx.moveTo(h2Nodes[k].x, h2Nodes[k].y);
        ctx.lineTo(outNodes[a].x, outNodes[a].y);
        ctx.stroke();
      }
    }

    // 3. Draw Neurons and Labels
    const fontSize = Math.max(9, Math.floor(10 * dpr));
    ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
    ctx.textBaseline = 'middle';

    const drawNode = (node, r) => {
      const v = Math.max(-1, Math.min(1, node.val));
      ctx.fillStyle = v >= 0
        ? `rgb(${Math.round(30 + 26 * v)}, ${Math.round(60 + 165 * v)}, ${Math.round(90 + 140 * v)})`
        : `rgb(${Math.round(60 + 190 * (-v))}, ${Math.round(40 + 40 * (-v))}, ${Math.round(70 + 50 * (-v))})`;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    const rIn = Math.max(3.5, 4.2 * dpr);
    for (const n of inNodes) {
      drawNode(n, rIn);
      ctx.fillStyle = '#cbd5e1';
      ctx.textAlign = 'right';
      ctx.fillText(n.label, n.x - rIn - 4 * dpr, n.y);
    }

    const rHid = Math.max(3, 3.6 * dpr);
    for (const n of h1Nodes) drawNode(n, rHid);
    for (const n of h2Nodes) drawNode(n, rHid);

    const rOut = outCount > 8 ? Math.max(2.9, 3.4 * dpr) : Math.max(4.5, 5.2 * dpr);
    if (outCount > 8) {
      const outFontSize = Math.max(7.5, Math.floor(8.2 * dpr));
      ctx.font = `${outFontSize}px system-ui, -apple-system, sans-serif`;
    }
    for (const n of outNodes) {
      drawNode(n, rOut);
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'left';
      ctx.fillText(n.label, n.x + rOut + 3.5 * dpr, n.y);
    }
    if (outCount > 8) {
      ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
    }

    // Critic V(s) badge at bottom right
    const criticY = h - 14 * dpr;
    ctx.fillStyle = '#fbbf24';
    ctx.textAlign = 'right';
    const criticText = (t && typeof t.nnCriticLabel === 'function')
      ? t.nnCriticLabel(val.toFixed(2))
      : `Критик V(s) = ${val.toFixed(2)}`;
    ctx.fillText(criticText, w - 8 * dpr, criticY);
  }
}

export class LearningChartRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  render(history, isLearningActive = true, t = null) {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(260, Math.floor(rect.width * dpr));
    const h = Math.max(140, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    const padL = 36 * dpr;
    const padR = 14 * dpr;
    const padT = 18 * dpr;
    const padB = 22 * dpr;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
    }

    if (!history || history.length === 0) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = `${Math.floor(11 * dpr)}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(t && t.chartWaitingNote ? t.chartWaitingNote : 'Ожидание первого обновления весов (1.6 сек)...', w * 0.5, h * 0.5);
      return;
    }

    const vels = history.map(d => d.meanVelocity);
    const rews = history.map(d => d.meanReward);
    const minVal = Math.min(-0.05, ...vels, ...rews);
    const maxVal = Math.max(0.25, ...vels, ...rews);
    const span = Math.max(1e-4, maxVal - minVal);

    const toX = idx => padL + (history.length === 1 ? plotW * 0.5 : (idx / (history.length - 1)) * plotW);
    const toY = v => padT + plotH - ((v - minVal) / span) * plotH;

    // Zero baseline
    if (minVal < 0 && maxVal > 0) {
      const y0 = toY(0);
      ctx.strokeStyle = 'rgba(248, 250, 252, 0.25)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padL, y0);
      ctx.lineTo(padL + plotW, y0);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Plot Mean Reward curve (Emerald)
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = toX(i);
      const y = toY(history[i].meanReward);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Plot Velocity Toward Food curve (Cyan)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.2 * dpr;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = toX(i);
      const y = toY(history[i].meanVelocity);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${Math.floor(9.5 * dpr)}px system-ui, sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(maxVal.toFixed(2), padL - 4 * dpr, padT + 4 * dpr);
    ctx.fillText(minVal.toFixed(2), padL - 4 * dpr, padT + plotH);

    // Legend
    ctx.textAlign = 'left';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(t && t.chartLegendVel ? t.chartLegendVel : '● Скорость к еде (м/с)', padL + 4 * dpr, padT - 5 * dpr);
    ctx.fillStyle = '#34d399';
    ctx.fillText(t && t.chartLegendRew ? t.chartLegendRew : '● Средняя награда', padL + 135 * dpr, padT - 5 * dpr);

    if (!isLearningActive) {
      ctx.textAlign = 'right';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(t && t.chartPausedNote ? t.chartPausedNote : '⏸ Обучение выкл.', padL + plotW, padT + plotH + 14 * dpr);
    }
  }
}
