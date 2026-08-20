import assert from "node:assert/strict";
import test from "node:test";

import { convertLatexToMarkup, validateLatex } from "mathlive/ssr";

test("canceled factors render with two visible diagonal strokes", () => {
  const latex = String.raw`\color{#e93242}{\xcancel{x - 2}}`;
  assert.deepEqual(validateLatex(latex), []);

  const markup = convertLatexToMarkup(latex);
  assert.equal(markup.match(/<line /g)?.length, 2);
  assert.match(markup, /stroke="currentColor"/);
  assert.match(markup, /z-index:2/);
});
