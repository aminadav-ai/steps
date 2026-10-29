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
  }

  async startNewExperiment({ morphology, algorithm, populationSize, config }) {
    this.currentExperimentId = `exp_${Date.now()}`;
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
