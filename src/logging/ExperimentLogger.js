/**
 * Client-side Experiment Logger that synchronizes with server.js (/api/experiment/*)
 * ONLY when running locally in Antigravity (localhost / 127.0.0.1).
 * Automatically disabled on GitHub Pages and public hosts so logs are recorded
 * and displayed exclusively for the project author.
 */
export class ExperimentLogger {
  constructor() {
    this.currentExperimentId = null;
    this.lastArchivedFile = null;
    this.savedLogsList = [];
    this.userEvents = [];
    this.cameraEvents = [];
    this._flushTimer = null;

    const host = (typeof window !== 'undefined' && window.location) ? window.location.hostname : '';
    const isLocalHost = host === 'localhost' || host === '127.0.0.1' || host === '[::1]';
    this.enabled = isLocalHost;
    this.staticMode = !isLocalHost;
  }

  async startNewExperiment({ morphology, algorithm, populationSize, config }) {
    if (!this.enabled || this.staticMode) return null;
    this.currentExperimentId = `exp_${Date.now()}`;
    this.userEvents = [{
      tWall: new Date().toISOString(),
      simTime: 0,
      update: 0,
      event: 'experiment_start',
      details: { morphology, algorithm, populationSize, config }
    }];
    this.cameraEvents = [];
    try {
      const res = await fetch('./api/experiment/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experimentId: this.currentExperimentId,
          morphology,
          algorithm,
          populationSize,
          config
        })
      });
      if (!res.ok) {
        this.staticMode = true;
        this.enabled = false;
        return null;
      }
      const data = await res.json();
      if (data.archivedPreviousLog) {
        this.lastArchivedFile = data.archivedPreviousLog;
      }
      await this.refreshLogsList();
      return data;
    } catch {
      this.staticMode = true;
      this.enabled = false;
      return null;
    }
  }

  buildTelemetrySnapshot(sim, renderer3D, uiState = {}) {
    if (!this.enabled || !sim) return null;
    return {
      capturedAt: new Date().toISOString(),
      uiState,
      cameraState: renderer3D ? renderer3D.getCameraTelemetry() : null,
      walkerAndTraining: sim.getDetailedTelemetry(),
      userEventsCount: this.userEvents.length,
      cameraEventsCount: this.cameraEvents.length,
      userEvents: this.userEvents.slice(-150),
      cameraEvents: this.cameraEvents.slice(-150)
    };
  }

  recordUserEvent(eventName, details = {}, sim = null, renderer3D = null, uiState = {}) {
    if (!this.enabled || this.staticMode) return;
    const c0 = sim && sim.creatures ? sim.creatures[0] : null;
    const com = c0 ? c0.body.getCenterOfMass() : null;
    const upY = c0 ? c0.getTorsoFrame().up.y : null;

    this.userEvents.push({
      tWall: new Date().toISOString(),
      simTime: sim ? Number(sim.simTime.toFixed(2)) : 0,
      update: sim && sim.trainer ? sim.trainer.updateCount : 0,
      event: eventName,
      details,
      walkerContext: c0 ? {
        pos: [Number(com.x.toFixed(2)), Number(com.z.toFixed(2))],
        foodCollected: c0.foodCollected,
        upY: Number(upY.toFixed(3)),
        isOverturned: upY < 0.35
      } : null
    });
    if (this.userEvents.length > 200) {
      this.userEvents.shift();
    }
    this.scheduleSnapshotFlush(sim, renderer3D, uiState);
  }

  recordCameraEvent(camEvent, sim = null, renderer3D = null, uiState = {}) {
    if (!this.enabled || this.staticMode) return;
    this.cameraEvents.push({
      tWall: new Date().toISOString(),
      simTime: sim ? Number(sim.simTime.toFixed(2)) : 0,
      update: sim && sim.trainer ? sim.trainer.updateCount : 0,
      ...camEvent
    });
    if (this.cameraEvents.length > 200) {
      this.cameraEvents.shift();
    }
    this.scheduleSnapshotFlush(sim, renderer3D, uiState);
  }

  scheduleSnapshotFlush(sim, renderer3D, uiState = {}) {
    if (!this.enabled || this.staticMode || !sim) return;
    if (this._flushTimer) clearTimeout(this._flushTimer);
    this._flushTimer = setTimeout(() => {
      this._flushTimer = null;
      const snapshot = this.buildTelemetrySnapshot(sim, renderer3D, uiState);
      this.logUpdate(null, sim.morphologyKey, sim.algorithm, snapshot);
    }, 220);
  }

  async logUpdate(entry, morphology, algorithm, bestSnapshot = null) {
    if (!this.enabled || this.staticMode) return;
    try {
      const res = await fetch('./api/experiment/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          morphology,
          algorithm,
          entry,
          bestSnapshot
        })
      });
      if (!res.ok) {
        this.staticMode = true;
        this.enabled = false;
      }
    } catch {
      this.staticMode = true;
      this.enabled = false;
    }
  }

  async refreshLogsList() {
    if (!this.enabled || this.staticMode) return this.savedLogsList;
    try {
      const res = await fetch('./api/experiment/list');
      if (!res.ok) {
        this.staticMode = true;
        this.enabled = false;
        return this.savedLogsList;
      }
      const data = await res.json();
      if (data.ok && Array.isArray(data.logs)) {
        this.savedLogsList = data.logs;
      }
    } catch {
      this.staticMode = true;
      this.enabled = false;
    }
    return this.savedLogsList;
  }
}
