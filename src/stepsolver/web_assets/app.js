"use strict";

import "https://esm.run/mathlive";
import { ComputeEngine } from "https://esm.run/@cortex-js/compute-engine";

const computeEngine = new ComputeEngine();
const form = document.querySelector("#solve-form");
const expressionField = document.querySelector("#expression-field");
const solveButton = document.querySelector("#solve-button");
const resultSection = document.querySelector("#result-section");
const statusText = document.querySelector("#status-text");
const answerBlock = document.querySelector("#answer-block");
const workingBlock = document.querySelector("#working-block");
const resultLatex = document.querySelector("#result-latex");
const stepsContainer = document.querySelector("#steps");
const asciiOutput = document.querySelector("#ascii-output");
const errorBox = document.querySelector("#error-box");

function createReadonlyMath(latex, className) {
  const field = document.createElement("math-field");
  field.className = className;
  field.readOnly = true;
  field.value = latex;
  return field;
}

function createStep(step) {
  const article = document.createElement("article");
  article.className = "step";
  const number = document.createElement("div");
  number.className = "step-number";
  number.textContent = `${step.number}.`;
  const body = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = step.rule;
  const explanation = document.createElement("p");
  explanation.textContent = step.explanation;
  const transformation = document.createElement("div");
  transformation.className = "step-transformation";
  const before = createReadonlyMath(step.before_latex, "step-math");
  const arrow = document.createElement("div");
  arrow.className = "step-arrow";
  arrow.textContent = "→";
  arrow.setAttribute("aria-label", "becomes");
  const after = createReadonlyMath(step.after_latex, "step-math step-math-after");
  transformation.append(before, arrow, after);
  const verification = document.createElement("details");
  verification.className = "step-verification";
  const verificationSummary = document.createElement("summary");
  verificationSummary.textContent = "Why this step is valid";
  const verificationDetail = document.createElement("p");
  verificationDetail.textContent = `${step.verification_method}: ${step.verification_detail}`;
  verification.append(verificationSummary, verificationDetail);
  body.append(heading, explanation, transformation, verification);
  article.append(number, body);
  return article;
}

function buildPayload() {
  const latex = expressionField.value.trim();
  const parsed = computeEngine.parse(latex, { form: "raw" });
  return { latex, math_json: parsed.json };
}

async function solve() {
  solveButton.disabled = true;
  solveButton.firstElementChild.textContent = "Solving…";
  resultSection.classList.remove("hidden");
  errorBox.classList.add("hidden");
  statusText.classList.remove("is-error");
  stepsContainer.replaceChildren();
  try {
    const response = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload())
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail ?? "The server could not solve this query.");
    statusText.textContent = payload.status === "exact" ? "Exact answer" : "No exact answer";
    asciiOutput.textContent = payload.formatted_ascii;
    if (payload.status === "exact") {
      answerBlock.classList.remove("hidden");
      resultLatex.value = payload.result_latex;
    } else {
      answerBlock.classList.add("hidden");
      statusText.classList.add("is-error");
      errorBox.textContent = payload.reason;
      errorBox.classList.remove("hidden");
    }
    for (const step of payload.steps) stepsContainer.append(createStep(step));
    workingBlock.classList.toggle("hidden", payload.steps.length === 0);
    resultSection.scrollIntoView({ block: "start" });
  } catch (error) {
    answerBlock.classList.add("hidden");
    workingBlock.classList.add("hidden");
    statusText.textContent = "Input error";
    statusText.classList.add("is-error");
    errorBox.textContent = error instanceof Error ? error.message : "Unexpected browser error.";
    errorBox.classList.remove("hidden");
  } finally {
    solveButton.disabled = false;
    solveButton.firstElementChild.textContent = "Solve";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (expressionField.value.trim()) solve();
});

for (const key of document.querySelectorAll(".symbol-key")) {
  key.addEventListener("click", () => {
    expressionField.executeCommand(["insert", key.dataset.insert, { selectionMode: "placeholder" }]);
    expressionField.focus();
  });
}

for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    expressionField.value = example.dataset.expression;
    expressionField.focus();
  });
}
