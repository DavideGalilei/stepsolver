import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

class WorkerStub {
  static instance;

  listeners = new Map();
  messages = [];

  constructor() {
    WorkerStub.instance = this;
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  postMessage(message) {
    this.messages.push(message);
  }

  emit(type, data) {
    this.listeners.get(type)?.({ data });
  }
}

globalThis.Worker = WorkerStub;

const { createSolverClient } = await import("./browser-runtime.mjs");

test("warmup exposes shared Python runtime stages before solve", async () => {
  const client = createSolverClient();
  const statuses = [];
  const steps = [
    "Download the Python runtime",
    "Load Python's package installer",
    "Install Pydantic validation support",
    "Install the SymPy mathematics engine",
    "Install StepSolver",
    "Import the StepSolver Python code"
  ];
  const unsubscribe = client.subscribeRuntimeStatus((status) => statuses.push(status));
  const warming = client.warmup();
  const worker = WorkerStub.instance;

  assert.deepEqual(worker.messages, [{ id: 0, action: "warmup" }]);
  worker.emit("message", {
    type: "runtime-status",
    state: "loading",
    stage: 4,
    total: 6,
    message: "Install the SymPy mathematics engine",
    steps
  });
  worker.emit("message", {
    type: "runtime-status",
    state: "ready",
    stage: 6,
    total: 6,
    message: "Python solver ready",
    steps
  });
  worker.emit("message", { id: 0, type: "result", payload: null });
  await warming;

  assert.equal(statuses.at(-2).message, "Install the SymPy mathematics engine");
  assert.deepEqual(statuses.at(-1), {
    state: "ready",
    stage: 6,
    total: 6,
    message: "Python solver ready",
    steps
  });

  const lateStatuses = [];
  client.subscribeRuntimeStatus((status) => lateStatuses.push(status));
  assert.equal(lateStatuses[0].state, "ready");

  const secondClient = createSolverClient();
  const solving = secondClient.solve({ math_json: ["Add", 1, 1] });
  assert.equal(WorkerStub.instance, worker);
  assert.deepEqual(worker.messages.at(-1), {
    id: 1,
    action: "solve",
    mathJson: ["Add", 1, 1]
  });
  worker.emit("message", { id: 1, type: "result", payload: { status: "exact" } });
  assert.deepEqual(await solving, { status: "exact" });
  assert.equal(statuses.filter((status) => status.state === "loading").length, 1);
  unsubscribe();
});

test("worker reports each concrete Python initialization phase", () => {
  const source = readFileSync(new URL("./browser-worker.mjs", import.meta.url), "utf8");
  const phases = [
    "Download the Python runtime",
    "Load Python's package installer",
    "Install Pydantic validation support",
    "Install the SymPy mathematics engine",
    "Install StepSolver",
    "Import the StepSolver Python code",
    "Python solver ready"
  ];

  let previousIndex = -1;
  for (const phase of phases) {
    const index = source.indexOf(phase);
    assert.ok(index > previousIndex, `${phase} should follow the preceding phase`);
    previousIndex = index;
  }
  assert.match(source, /searchParams\.set\("v", "__STEPSOLVER_WHEEL_VERSION__"\)/);

  const pydanticInstall = source.indexOf('loadPackage("pydantic")');
  const stepSolverInstall = source.indexOf("micropip.install(stepsolver_wheel_url");
  const stepSolverImport = source.indexOf("from stepsolver.browser import solve_mathjson_json");
  assert.ok(pydanticInstall >= 0, "the worker should install Pydantic");
  assert.ok(pydanticInstall < stepSolverInstall, "Pydantic should precede StepSolver installation");
  assert.ok(pydanticInstall < stepSolverImport, "Pydantic should precede StepSolver import");
});
