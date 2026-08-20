// Curated examples used by the random-problem control.

export const RANDOM_PROBLEMS = Object.freeze([
  "x^2-4=0",
  "\\frac{5x}{2}+2=\\frac{3x}{2}+10",
  "\\frac{\\mathrm{d}}{\\mathrm{d}x}(\\sin(x)e^x)",
  "\\frac{\\mathrm{d}}{\\mathrm{d}x}(x^x)",
  "\\int_0^\\pi\\sin(x)\\,\\mathrm{d}x",
  "\\int x e^x\\,\\mathrm{d}x",
  "\\int \\ln(x)\\,\\mathrm{d}x",
  "\\lim_{x\\to0}\\frac{\\sin(x)}{x}",
  "\\lim_{x\\to0}\\frac{e^x-1}{x}",
  "\\sum_{n=1}^{10}n^2",
  "\\sum_{n=1}^{\\infty}\\frac{1}{n(n+1)}"
]);

export function chooseRandomProblem(currentExpression, random = Math.random) {
  const alternatives = RANDOM_PROBLEMS.filter((problem) => problem !== currentExpression);
  const index = Math.min(Math.floor(random() * alternatives.length), alternatives.length - 1);
  return alternatives[index];
}
