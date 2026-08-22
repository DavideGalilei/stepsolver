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
    /createReadonlyMath\(part\.latex, "step-explanation-math"\)/
  );
  assert.match(appSource, /document\.createTextNode\(part\.text\)/);
  assert.doesNotMatch(appSource, /explanation\.textContent = step\.explanation/);
  assert.match(styleSource, /\.step-explanation-math\s*\{/);
  assert.match(styleSource, /display:\s*inline-block/);
});
