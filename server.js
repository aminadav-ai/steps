import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 3000;
const LOGS_DIR = path.join(__dirname, 'logs');
const CURRENT_LOG_FILE = path.join(LOGS_DIR, 'current_experiment.json');

if (!fs.existsSync(LOGS_DIR)) {
  fs.mkdirSync(LOGS_DIR, { recursive: true });
}

/**
 * Archives the existing current_experiment.json by renaming it with a timestamp
 * so that previous experiment logs are ALWAYS preserved and never overwritten.
 */
function archiveExistingLog() {
  if (fs.existsSync(CURRENT_LOG_FILE)) {
    try {
      const stat = fs.statSync(CURRENT_LOG_FILE);
      if (stat.size > 0) {
        let metaSuffix = '';
        try {
          const prev = JSON.parse(fs.readFileSync(CURRENT_LOG_FILE, 'utf8'));
          if (prev.morphology && prev.algorithm) {
            metaSuffix = `_${prev.morphology}_${prev.algorithm}`;
          }
        } catch {
          // Ignore parse error, still preserve file
        }
        const iso = new Date().toISOString().replace(/[:.]/g, '-');
        const archivedName = `experiment_${iso}${metaSuffix}.json`;
        const archivedPath = path.join(LOGS_DIR, archivedName);
        fs.renameSync(CURRENT_LOG_FILE, archivedPath);
        console.log(`[ExperimentLogger] Previous log preserved as: logs/${archivedName}`);
        return archivedName;
      }
    } catch (err) {
      console.error('[ExperimentLogger] Error archiving log:', err);
    }
  }
  return null;
}

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon'
};

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => {
      data += chunk;
      if (data.length > 10 * 1024 * 1024) {
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  // API: Start a new experiment (archives any previous log first)
  if (req.method === 'POST' && url.pathname === '/api/experiment/start') {
    try {
      const archivedFile = archiveExistingLog();
      const raw = await readBody(req);
      const payload = raw ? JSON.parse(raw) : {};
      const initialLog = {
        experimentId: payload.experimentId || `exp_${Date.now()}`,
        startedAt: new Date().toISOString(),
        morphology: payload.morphology || 'tripod',
        algorithm: payload.algorithm || 'ppo',
        populationSize: payload.populationSize || 16,
        config: payload.config || {},
        archivedPreviousLog: archivedFile,
        history: [],
        bestSnapshot: null
      };
      fs.writeFileSync(CURRENT_LOG_FILE, JSON.stringify(initialLog, null, 2), 'utf8');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, archivedPreviousLog: archivedFile, experimentId: initialLog.experimentId }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: err.message }));
    }
    return;
  }

  // API: Append update / checkpoint to current experiment log
  if (req.method === 'POST' && url.pathname === '/api/experiment/log') {
    try {
      const raw = await readBody(req);
      const payload = JSON.parse(raw);
      let current = {
        experimentId: `exp_${Date.now()}`,
        startedAt: new Date().toISOString(),
        morphology: payload.morphology || 'tripod',
        algorithm: payload.algorithm || 'ppo',
        history: []
      };
      if (fs.existsSync(CURRENT_LOG_FILE)) {
        try {
          current = JSON.parse(fs.readFileSync(CURRENT_LOG_FILE, 'utf8'));
        } catch {
          archiveExistingLog();
        }
      }
      if (payload.entry) {
        current.history.push(payload.entry);
      }
      if (payload.bestSnapshot) {
        current.bestSnapshot = payload.bestSnapshot;
      }
      current.updatedAt = new Date().toISOString();
      fs.writeFileSync(CURRENT_LOG_FILE, JSON.stringify(current, null, 2), 'utf8');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, entries: current.history.length }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: err.message }));
    }
    return;
  }

  // API: List all saved experiment logs
  if (req.method === 'GET' && url.pathname === '/api/experiment/list') {
    try {
      const files = fs.readdirSync(LOGS_DIR)
        .filter(f => f.endsWith('.json'))
        .map(f => {
          const full = path.join(LOGS_DIR, f);
          const stat = fs.statSync(full);
          let summary = {};
          try {
            const parsed = JSON.parse(fs.readFileSync(full, 'utf8'));
            const last = parsed.history && parsed.history.length > 0
              ? parsed.history[parsed.history.length - 1]
              : null;
            summary = {
              experimentId: parsed.experimentId,
              startedAt: parsed.startedAt,
              morphology: parsed.morphology,
              algorithm: parsed.algorithm,
              updatesCount: parsed.history ? parsed.history.length : 0,
              lastMeanReward: last ? last.meanReward : null,
              totalFoodCollected: last ? last.totalFood : 0
            };
          } catch {
            // ignore
          }
          return {
            filename: f,
            sizeBytes: stat.size,
            modifiedAt: stat.mtime.toISOString(),
            ...summary
          };
        })
        .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, logs: files }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: err.message }));
    }
    return;
  }

  // Static files
  let safePath = path.normalize(url.pathname).replace(/^(\.\.[/\\])+/, '');
  if (safePath === '/' || safePath === '\\') {
    safePath = '/index.html';
  }
  const filePath = path.join(__dirname, safePath);
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': contentType });
    fs.createReadStream(filePath).pipe(res);
  });
});

server.listen(PORT, () => {
  console.log(`Steps 3D Physics & RL Lab running at http://localhost:${PORT}`);
});
