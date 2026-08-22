import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { ComputeEngine } from "@cortex-js/compute-engine";

const cases = JSON.parse(
  readFileSync(new URL("./fixtures/editor-cases.json", import.meta.url), "utf8")
);
const computeEngine = new ComputeEngine();

test("MathJSON contract: indexed-root series", () => {
  const parsed = computeEngine.parse(
    String.raw`\sum_{n=1}^{+\infty}{\sqrt[n]{2^n}+2^n}`,
    { form: "raw" },
  );
  assert.deepEqual(parsed.json, [
    "Sum",
    ["Add", ["Root", ["Power", 2, "n"], "n"], ["Power", 2, "n"]],
    ["Tuple", "n", 1, "PositiveInfinity"],
  ]);
});

test("MathJSON contract: indexed root containing an additive radicand", () => {
  const parsed = computeEngine.parse(
    String.raw`\sum_{n=1}^{+\infty}{\sqrt[n]{2^n+2^n}}`,
    { form: "raw" },
  );
  assert.deepEqual(parsed.json, [
    "Sum",
    ["Root", ["Add", ["Power", 2, "n"], ["Power", 2, "n"]], "n"],
    ["Tuple", "n", 1, "PositiveInfinity"],
  ]);
});

test("MathJSON contract: reciprocal-factorial series tail", () => {
  const parsed = computeEngine.parse(
    String.raw`\sum_{n=3}^{+\infty}\frac{1}{n!}`,
    { form: "raw" },
  );
  assert.deepEqual(parsed.json, [
    "Sum",
    ["Divide", 1, ["Factorial", "n"]],
    ["Tuple", "n", 3, "PositiveInfinity"],
  ]);
});

test("MathJSON contract: inverse hyperbolic function", () => {
  const parsed = computeEngine.parse(String.raw`\operatorname{arsinh}(x)`, {
    form: "raw",
  });
  assert.deepEqual(parsed.json, ["Arsinh", "x"]);
});

for (const editorCase of cases) {
  test(`MathJSON contract: ${editorCase.name}`, () => {
    const parsed = computeEngine.parse(editorCase.latex, { form: "raw" });
    assert.deepEqual(parsed.json, editorCase.math_json);
  });
}
