import assert from "node:assert/strict";
import test from "node:test";

import { convertLatexToMarkup, validateLatex } from "mathlive/ssr";

test("canceled factors render with two visible diagonal strokes", () => {
  const latex = String.raw`\xcancel{x - 2}`;
  assert.deepEqual(validateLatex(latex), []);

  const markup = convertLatexToMarkup(latex);
  assert.equal(markup.match(/<line /g)?.length, 2);
  assert.match(markup, /stroke="currentColor"/);
  assert.match(markup, /z-index:2/);
  assert.doesNotMatch(markup, /#e93242/);
});

test("an introduced multiplier highlights only its own value", () => {
  const latex = String.raw`\textcolor{#4f8cff}{2}\cdot\left(\frac{5x}{2}+2\right)`;
  assert.deepEqual(validateLatex(latex), []);

  const markup = convertLatexToMarkup(latex);
  assert.match(markup, /color:#4f8cff/);
  assert.equal(markup.match(/color:#4f8cff/g)?.length, 1);
});
