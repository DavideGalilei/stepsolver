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
    stage: 3,
    total: 5,
    message: "Install the SymPy mathematics engine",
    steps
  });
  worker.emit("message", {
    type: "runtime-status",
    state: "ready",
    stage: 5,
    total: 5,
    message: "Python solver ready",
    steps
  });
  worker.emit("message", { id: 0, type: "result", payload: null });
  await warming;

  assert.equal(statuses.at(-2).message, "Install the SymPy mathematics engine");
  assert.deepEqual(statuses.at(-1), {
    state: "ready",
    stage: 5,
    total: 5,
    message: "Python solver ready",
    steps
  });

  const lateStatuses = [];
  client.subscribeRuntimeStatus((status) => lateStatuses.push(status));
  assert.equal(lateStatuses[0].state, "ready");
  unsubscribe();
});

test("worker reports each concrete Python initialization phase", () => {
  const source = readFileSync(new URL("./browser-worker.mjs", import.meta.url), "utf8");
  const phases = [
    "Download the Python runtime",
    "Load Python's package installer",
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
});
