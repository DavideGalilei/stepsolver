import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const appSource = readFileSync(
  new URL("../src/stepsolver/web_assets/app.js", import.meta.url),
  "utf8"
);
const styleSource = readFileSync(
  new URL("../src/stepsolver/web_assets/style.css", import.meta.url),
  "utf8"
);

test("step explanations render structured mathematics through MathLive", () => {
  assert.match(appSource, /function createStepExplanation\(step\)/);
  assert.match(
    appSource,
    /createInlineMath\(part\.latex, "step-explanation-math"\)/
  );
  assert.match(appSource, /document\.createElement\("math-span"\)/);
  assert.match(appSource, /math\.textContent = latex/);
  assert.match(appSource, /document\.createTextNode\(part\.text\)/);
  assert.doesNotMatch(appSource, /explanation\.textContent = step\.explanation/);
  assert.doesNotMatch(
    appSource,
    /createReadonlyMath\(part\.latex, "step-explanation-math"\)/
  );
  const inlineMathStyle = styleSource.match(/\.step-explanation-math\s*\{[^}]+\}/s);
  assert.ok(inlineMathStyle);
  assert.match(inlineMathStyle[0], /display:\s*inline-flex/);
  assert.match(inlineMathStyle[0], /white-space:\s*nowrap/);
  assert.match(inlineMathStyle[0], /margin:\s*0/);
});
