// Web Worker client for the GitHub Pages Pyodide runtime.

let nextRequestId = 0;
let worker;
const pendingRequests = new Map();
const runtimeListeners = new Set();
let warmupPromise;
let runtimeSnapshot = Object.freeze({
  state: "idle",
  stage: 0,
  total: 5,
  message: "Starting the Python solver in the background"
});

function publishRuntimeStatus(status) {
  runtimeSnapshot = Object.freeze({
    state: status.state,
    stage: status.stage,
    total: status.total,
    message: status.message
  });
  for (const listener of runtimeListeners) listener(runtimeSnapshot);
}

function solverWorker() {
  if (!worker) {
    worker = new Worker(new URL("./browser-worker.mjs", import.meta.url), {
      type: "module"
    });
    worker.addEventListener("message", ({ data }) => {
      if (data.type === "runtime-status") {
        publishRuntimeStatus(data);
        return;
      }
      const pending = pendingRequests.get(data.id);
      if (!pending) return;
      if (data.type === "progress") {
        pending.onProgress(data.message);
        return;
      }
      pendingRequests.delete(data.id);
      if (data.type === "result") pending.resolve(data.payload);
      else pending.reject(new Error(data.message));
    });
    worker.addEventListener("error", (event) => {
      for (const pending of pendingRequests.values()) {
        pending.reject(new Error(event.message || "The browser solver could not start."));
      }
      pendingRequests.clear();
      worker = undefined;
      warmupPromise = undefined;
      publishRuntimeStatus({
        state: "error",
        stage: 0,
        total: runtimeSnapshot.total,
        message: "Python solver worker stopped unexpectedly"
      });
    });
  }
  return worker;
}

export function createSolverClient() {
  function request(message, onProgress = () => {}) {
    const id = nextRequestId++;
    return new Promise((resolve, reject) => {
      pendingRequests.set(id, { resolve, reject, onProgress });
      solverWorker().postMessage({ id, ...message });
    });
  }

  return {
    subscribeRuntimeStatus(listener) {
      runtimeListeners.add(listener);
      listener(runtimeSnapshot);
      return () => runtimeListeners.delete(listener);
    },
    warmup() {
      warmupPromise ??= request({ action: "warmup" }).catch((error) => {
        warmupPromise = undefined;
        throw error;
      });
      return warmupPromise;
    },
    solve(payload, onProgress = () => {}) {
      return request({ action: "solve", mathJson: payload.math_json }, onProgress);
    }
  };
}
