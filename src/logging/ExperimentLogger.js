/**
 * Client-side Experiment Logger that synchronizes with server.js (/api/experiment/*)
 * to ensure all previous experiment logs are preserved on disk (renamed with timestamps)
 * before starting any new experiment.
 */
export class ExperimentLogger {
  constructor() {
    this.currentExperimentId = null;
    this.lastArchivedFile = null;
    this.savedLogsList = [];
    this.userEvents = [];
    this.cameraEvents = [];
    this._flushTimer = null;
    this.staticMode = typeof window !== 'undefined' && window.location.hostname.endsWith('.github.io');
  }

  async startNewExperiment({ morphology, algorithm, populationSize, config }) {
    this.currentExperimentId = `exp_${Date.now()}`;
    this.userEvents = [{
      tWall: new Date().toISOString(),
      simTime: 0,
      update: 0,
      event: 'experiment_start',
      details: { morphology, algorithm, populationSize, config }
    }];
    this.cameraEvents = [];
    if (this.staticMode) return null;
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
      return null;
    }
  }

  buildTelemetrySnapshot(sim, renderer3D, uiState = {}) {
    if (!sim) return null;
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
    if (!sim) return;
    if (this._flushTimer) clearTimeout(this._flushTimer);
    this._flushTimer = setTimeout(() => {
      this._flushTimer = null;
      const snapshot = this.buildTelemetrySnapshot(sim, renderer3D, uiState);
      this.logUpdate(null, sim.morphologyKey, sim.algorithm, snapshot);
    }, 220);
  }

  async logUpdate(entry, morphology, algorithm, bestSnapshot = null) {
    if (this.staticMode) return;
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
      if (!res.ok) this.staticMode = true;
    } catch {
      this.staticMode = true;
    }
  }

  async refreshLogsList() {
    if (this.staticMode) return this.savedLogsList;
    try {
      const res = await fetch('./api/experiment/list');
      if (!res.ok) {
        this.staticMode = true;
        return this.savedLogsList;
      }
      const data = await res.json();
      if (data.ok && Array.isArray(data.logs)) {
        this.savedLogsList = data.logs;
      }
    } catch {
      this.staticMode = true;
    }
    return this.savedLogsList;
  }
}
