import assert from "node:assert/strict";
import test from "node:test";

import { contentVersion } from "./artifact-version.mjs";

test("artifact versions are stable for identical content", () => {
  assert.equal(contentVersion("wheel bytes"), contentVersion("wheel bytes"));
});

test("artifact versions change when Python package content changes", () => {
  assert.notEqual(contentVersion("old wheel"), contentVersion("new wheel"));
});
