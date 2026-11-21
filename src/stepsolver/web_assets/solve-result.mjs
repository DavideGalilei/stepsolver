import validateResult from "./solve-result-validator.mjs";

export function parseSolveResult(result) {
  if (!validateResult(result)
      || (result.status === "unsolved") !== (result.result_latex === null)
      || !result.steps.every((step, index) =>
        step.number === index + 1 && step.explanation_parts.every(
          (part) => (part.text === null) !== (part.latex === null)
        )
      )) {
    throw new Error("The solver returned an invalid result.");
  }
  return result;
}
