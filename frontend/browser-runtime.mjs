// Web Worker client for the GitHub Pages Pyodide runtime.

let nextRequestId = 0;
let worker;
const pendingRequests = new Map();

function solverWorker() {
  if (!worker) {
    worker = new Worker(new URL("./browser-worker.mjs", import.meta.url), {
      type: "module"
    });
    worker.addEventListener("message", ({ data }) => {
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
    });
  }
  return worker;
}

export function createSolverClient() {
  return {
    solve(payload, onProgress = () => {}) {
      const id = nextRequestId++;
      return new Promise((resolve, reject) => {
        pendingRequests.set(id, { resolve, reject, onProgress });
        solverWorker().postMessage({ id, mathJson: payload.math_json });
      });
    }
  };
}
