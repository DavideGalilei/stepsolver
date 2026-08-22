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

test("step transformations use an accessible SVG icon instead of a Unicode arrow", () => {
  assert.doesNotMatch(appSource, /arrow\.textContent\s*=\s*["']→["']/);
  assert.match(appSource, /createElementNS\([^\n]+"svg"\)/);
  assert.match(appSource, /arrow\.setAttribute\("role", "img"\)/);
  assert.match(appSource, /arrow\.setAttribute\("aria-label", "becomes"\)/);
  assert.match(styleSource, /\.step-arrow-icon\s*\{/);
  assert.match(styleSource, /stroke:\s*currentColor/);
});
