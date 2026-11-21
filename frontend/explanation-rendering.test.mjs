import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const solutionSource = readFileSync(
  new URL("../src/stepsolver/web_assets/solution-view.mjs", import.meta.url),
  "utf8"
);
const styleSource = readFileSync(
  new URL("../src/stepsolver/web_assets/style.css", import.meta.url),
  "utf8"
);

test("step explanations render structured mathematics through MathLive", () => {
  assert.match(solutionSource, /function createStepExplanation\(step\)/);
  assert.match(
    solutionSource,
    /createInlineMath\(part\.latex, "step-explanation-math"\)/
  );
  assert.match(solutionSource, /document\.createElement\("math-span"\)/);
  assert.match(solutionSource, /math\.textContent = latex/);
  assert.match(solutionSource, /document\.createTextNode\(part\.text\)/);
  assert.doesNotMatch(solutionSource, /explanation\.textContent = step\.explanation/);
  assert.doesNotMatch(
    solutionSource,
    /createReadonlyMath\(part\.latex, "step-explanation-math"\)/
  );
  const inlineMathStyle = styleSource.match(/\.step-explanation-math\s*\{[^}]+\}/s);
  assert.ok(inlineMathStyle);
  assert.match(inlineMathStyle[0], /display:\s*inline-flex/);
  assert.match(inlineMathStyle[0], /white-space:\s*nowrap/);
  assert.match(inlineMathStyle[0], /margin:\s*0/);
});
