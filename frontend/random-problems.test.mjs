import assert from "node:assert/strict";
import test from "node:test";

import { ComputeEngine } from "@cortex-js/compute-engine";

import {
  RANDOM_PROBLEMS,
  chooseRandomProblem
} from "../src/stepsolver/web_assets/random-problems.mjs";

test("random problem selection does not repeat the current expression", () => {
  const current = RANDOM_PROBLEMS[0];
  assert.notEqual(chooseRandomProblem(current, () => 0), current);
  assert.notEqual(chooseRandomProblem(current, () => 0.999), current);
});

test("every random problem parses into MathJSON", () => {
  const computeEngine = new ComputeEngine();

  for (const problem of RANDOM_PROBLEMS) {
    const parsed = computeEngine.parse(problem, { form: "raw" });
    assert.notEqual(parsed.json[0], "Error", problem);
  }
});
