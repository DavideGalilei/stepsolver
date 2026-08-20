// Run the Python StepSolver package off the main browser thread.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const wheelUrl = new URL("../packages/__STEPSOLVER_WHEEL__", import.meta.url);
wheelUrl.searchParams.set("v", "__STEPSOLVER_WHEEL_VERSION__");
const RUNTIME_STEPS = Object.freeze([
  "Download the Python runtime",
  "Load Python's package installer",
  "Install the SymPy mathematics engine",
  "Install StepSolver",
  "Import the StepSolver Python code"
]);
const RUNTIME_STAGE_COUNT = RUNTIME_STEPS.length;
let runtimePromise;

function progress(id, message) {
  self.postMessage({ id, type: "progress", message });
}

function runtimeStatus(state, stage, message) {
  self.postMessage({
    type: "runtime-status",
    state,
    stage,
    total: RUNTIME_STAGE_COUNT,
    message,
    steps: RUNTIME_STEPS
  });
}

async function createRuntime() {
  try {
    runtimeStatus("loading", 1, RUNTIME_STEPS[0]);
    const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });

    runtimeStatus("loading", 2, RUNTIME_STEPS[1]);
    await pyodide.loadPackage("micropip");
    await pyodide.runPythonAsync("import micropip");

    runtimeStatus("loading", 3, RUNTIME_STEPS[2]);
    await pyodide.runPythonAsync('await micropip.install("sympy==1.14.0")');

    runtimeStatus("loading", 4, RUNTIME_STEPS[3]);
    pyodide.globals.set("stepsolver_wheel_url", wheelUrl.href);
    await pyodide.runPythonAsync("await micropip.install(stepsolver_wheel_url, deps=False)");

    runtimeStatus("loading", 5, RUNTIME_STEPS[4]);
    await pyodide.runPythonAsync(`
from stepsolver.browser import solve_mathjson_json
`);
    runtimeStatus("ready", RUNTIME_STAGE_COUNT, "Python solver ready");
    return pyodide;
  } catch (error) {
    runtimeStatus("error", 0, "Python solver could not be loaded");
    throw error;
  }
}

async function runtime() {
  runtimePromise ??= createRuntime().catch((error) => {
    runtimePromise = undefined;
    throw error;
  });
  return runtimePromise;
}

self.addEventListener("message", async ({ data }) => {
  try {
    const pyodide = await runtime();
    if (data.action === "warmup") {
      self.postMessage({ id: data.id, type: "result", payload: null });
      return;
    }
    progress(data.id, "Running StepSolver's symbolic engine");
    const solve = pyodide.globals.get("solve_mathjson_json");
    try {
      const result = solve(JSON.stringify(data.mathJson));
      const payload = JSON.parse(result);
      if (payload.error) {
        self.postMessage({ id: data.id, type: "error", message: payload.error });
      } else {
        self.postMessage({ id: data.id, type: "result", payload });
      }
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
