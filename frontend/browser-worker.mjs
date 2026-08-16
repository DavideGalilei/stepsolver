// Run the Python StepSolver package off the main browser thread.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const wheelUrl = new URL("../packages/__STEPSOLVER_WHEEL__", import.meta.url).href;
let runtimePromise;

function progress(id, message) {
  self.postMessage({ id, type: "progress", message });
}

async function createRuntime(id) {
  progress(id, "Loading Python…");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });
  progress(id, "Loading symbolic mathematics…");
  await pyodide.loadPackage("micropip");
  pyodide.globals.set("stepsolver_wheel_url", wheelUrl);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("sympy==1.14.0")
await micropip.install(stepsolver_wheel_url, deps=False)
from stepsolver.browser import solve_mathjson_json
`);
  return pyodide;
}

async function runtime(id) {
  runtimePromise ??= createRuntime(id).catch((error) => {
    runtimePromise = undefined;
    throw error;
  });
  return runtimePromise;
}

self.addEventListener("message", async ({ data }) => {
  try {
    const pyodide = await runtime(data.id);
    progress(data.id, "Solving…");
    const solve = pyodide.globals.get("solve_mathjson_json");
    try {
      const result = solve(JSON.stringify(data.mathJson));
      self.postMessage({ id: data.id, type: "result", payload: JSON.parse(result) });
    } finally {
      solve.destroy();
    }
  } catch (error) {
    self.postMessage({
      id: data.id,
      type: "error",
      message: error instanceof Error ? error.message : String(error)
    });
  }
});
