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
    try {
      const res = await fetch('/api/experiment/start', {
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
      const data = await res.json();
      if (data.archivedPreviousLog) {
        this.lastArchivedFile = data.archivedPreviousLog;
      }
      await this.refreshLogsList();
      return data;
    } catch (err) {
      console.warn('[ExperimentLogger] Running in standalone/offline mode:', err.message);
      return null;
    }
  }

  async resumeExistingExperiment() {
    try {
      const res = await fetch('/logs/current_experiment.json', { cache: 'no-store' });
      if (!res.ok) return null;
      const data = await res.json();
      if (data && data.experimentId) {
        this.currentExperimentId = data.experimentId;
      }
      if (data && data.bestSnapshot) {
        if (Array.isArray(data.bestSnapshot.userEvents)) {
          this.userEvents = data.bestSnapshot.userEvents.slice();
        }
        if (Array.isArray(data.bestSnapshot.cameraEvents)) {
          this.cameraEvents = data.bestSnapshot.cameraEvents.slice();
        }
      }
      await this.refreshLogsList();
      return data;
    } catch {
      return null;
    }
  }

  buildTelemetrySnapshot(sim, renderer3D, uiState = {}) {
    if (!sim) return null;
    const camState = renderer3D ? renderer3D.getCameraTelemetry() : null;
    return {
      capturedAt: new Date().toISOString(),
      uiState,
      cameraState: camState,
      walkerAndTraining: sim.getDetailedTelemetry(),
      savedCheckpoint: typeof sim.exportCheckpoint === 'function' ? sim.exportCheckpoint(uiState, camState) : null,
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
    try {
      await fetch('/api/experiment/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          morphology,
          algorithm,
          entry,
          bestSnapshot
        })
      });
    } catch {
      // Ignore network error if static file opened directly
    }
  }

  async refreshLogsList() {
    try {
      const res = await fetch('/api/experiment/list');
      const data = await res.json();
      if (data.ok && Array.isArray(data.logs)) {
        this.savedLogsList = data.logs;
      }
    } catch {
      // Ignore
    }
    return this.savedLogsList;
  }
}
