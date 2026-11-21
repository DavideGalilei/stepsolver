import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

import { compileSolveValidator } from "./compile-solve-validator.mjs";
import { createSolverClient } from "../src/stepsolver/web_assets/runtime.mjs";

import { parseSolveResult } from "../src/stepsolver/web_assets/solve-result.mjs";

const result = {
  status: "exact",
  source: "1+1",
  formatted_ascii: "2",
  result_latex: "2",
  reason: null,
  steps: [{
    number: 1,
    rule: "Add",
    explanation: "Add the two integers.",
    explanation_parts: [{ text: "Add ", latex: null }, { text: null, latex: "1+1" }],
    before_ascii: "1+1",
    after_ascii: "2",
    before_latex: "1+1",
    after_latex: "2",
    verification_method: "equivalent",
    verification_detail: "Both sides simplify to 2.",
    notes: [],
    introduced_constraints: []
  }]
};

test("accepts one normalized step with prose and inline mathematics", () => {
  assert.deepEqual(parseSolveResult(result), result);
});

test("rejects an invalid nested step before it is rendered", () => {
  const malformed = structuredClone(result);
  malformed.steps[0].explanation_parts[1].latex = 2;
  assert.throws(() => parseSolveResult(malformed), /invalid result/);
});


test("validates a generated artifact built from the checked schema", async () => {
  const asset = new URL("../src/stepsolver/web_assets/solve-result-validator.mjs", import.meta.url);
  assert.equal(await readFile(asset, "utf8"), await compileSolveValidator());
});

test("rejects additional fields, inconsistent answers, and out-of-order steps", () => {
  const [step] = result.steps;
  const invalid = [
    { ...result, answer: "unexpected" },
    { ...result, steps: [{ ...step, notes: [{ label: "Extra", expression_ascii: "2",
      expression_latex: "2", source: "unexpected" }] }] },
    { ...result, status: "unsolved" },
    { ...result, steps: [{ ...step, number: 2 }] },
    { ...result, steps: [{ ...step, explanation_parts: [{ text: "Explain", latex: "2" }] }] }
  ];
  for (const value of invalid) {
    assert.throws(() => parseSolveResult(value), /invalid result/);
  }
});

test("the HTTP boundary rejects malformed results before returning them", async (context) => {
  context.mock.method(globalThis, "fetch", async () => Response.json({ ...result, extra: 1 }));
  await assert.rejects(createSolverClient().solve({ latex: "1+1", math_json: ["Add", 1, 1] }),
    /invalid result/);
});
