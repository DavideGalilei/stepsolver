import assert from "node:assert/strict";
import test from "node:test";

import { convertLatexToMarkup, validateLatex } from "mathlive/ssr";

test("canceled factors are red beneath two white diagonal strokes", () => {
  const latex = String.raw`\textcolor{#ffffff}{\xcancel{\textcolor{#ff5362}{x - 2}}}`;
  assert.deepEqual(validateLatex(latex), []);

  const markup = convertLatexToMarkup(latex);
  assert.equal(markup.match(/<line /g)?.length, 2);
  assert.match(markup, /stroke="currentColor"/);
  assert.match(markup, /z-index:2/);
  assert.match(markup, /color:#ffffff/);
  assert.match(markup, /color:#ff5362/);
});

test("an introduced multiplication highlights its operator and parentheses", () => {
  const latex = String.raw`\textcolor{#4f8cff}{2\cdot\left(\textcolor{#f4f4f5}{\frac{5x}{2}+2}\right)}`;
  assert.deepEqual(validateLatex(latex), []);

  const markup = convertLatexToMarkup(latex);
  assert.match(markup, /color:#4f8cff/);
  assert.match(markup, /color:#f4f4f5/);
});
