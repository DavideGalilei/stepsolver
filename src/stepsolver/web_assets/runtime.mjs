// Server-backed solver client used by the FastAPI application.

import { parseSolveResult } from "./solve-result.mjs";

export function createSolverClient() {
  const readyStatus = Object.freeze({
    state: "ready",
    stage: 1,
    total: 1,
    message: "Solver server ready",
    steps: Object.freeze([])
  });
  return {
    subscribeRuntimeStatus(listener) {
      listener(readyStatus);
      return () => {};
    },
    async warmup() {},
    async solve(payload, onProgress = () => {}) {
      onProgress("Sending the problem to the solver server");
      const response = await fetch("./api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) {
        const message = typeof result?.detail === "string"
          ? result.detail
          : "The server could not solve this query.";
        throw new Error(message);
      }
      return parseSolveResult(result);
    }
  };
}
