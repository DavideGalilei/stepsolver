"use strict";

import { createSolverClient } from "./runtime.mjs";
import { ComputeEngine, MathfieldElement } from "./vendor.mjs";

MathfieldElement.fontsDirectory = new URL("./fonts/", import.meta.url).href;
MathfieldElement.soundsDirectory = null;

const computeEngine = new ComputeEngine();
const solverClient = createSolverClient();
const form = document.querySelector("#solve-form");
const expressionField = document.querySelector("#expression-field");
const mobileKeyboardProxy = document.querySelector("#mobile-keyboard-proxy");
const solveButton = document.querySelector("#solve-button");
const resultSection = document.querySelector("#result-section");
const statusText = document.querySelector("#status-text");
const answerBlock = document.querySelector("#answer-block");
const workingBlock = document.querySelector("#working-block");
const resultLatex = document.querySelector("#result-latex");
const stepsContainer = document.querySelector("#steps");
const asciiOutput = document.querySelector("#ascii-output");
const errorBox = document.querySelector("#error-box");

function hideMathVirtualKeyboard() {
  window.mathVirtualKeyboard?.hide();
}

const mobileKeyboardQuery = window.matchMedia("(hover: none) and (pointer: coarse)");
let composingNativeText = false;

function usesMobileKeyboard() {
  return mobileKeyboardQuery.matches;
}

function focusMobileKeyboard() {
  if (!usesMobileKeyboard()) return;
  hideMathVirtualKeyboard();
  mobileKeyboardProxy.focus({ preventScroll: true });
  mobileKeyboardProxy.setSelectionRange(0, mobileKeyboardProxy.value.length);
}

function insertNativeText(text) {
  if (!text) return;
  expressionField.insert(text, {
    insertionMode: "replaceSelection",
    selectionMode: "after"
  });
}

function clearMobileKeyboardProxy() {
  mobileKeyboardProxy.value = "";
}

expressionField.addEventListener("pointerup", focusMobileKeyboard);
expressionField.addEventListener("focusin", hideMathVirtualKeyboard);
mobileKeyboardProxy.addEventListener("focus", () => {
  expressionField.classList.add("is-mobile-editing");
  hideMathVirtualKeyboard();
});
mobileKeyboardProxy.addEventListener("blur", () => {
  expressionField.classList.remove("is-mobile-editing");
  clearMobileKeyboardProxy();
});
mobileKeyboardProxy.addEventListener("compositionstart", () => {
  composingNativeText = true;
});
mobileKeyboardProxy.addEventListener("compositionend", (event) => {
  composingNativeText = false;
  insertNativeText(event.data || mobileKeyboardProxy.value);
  clearMobileKeyboardProxy();
});
mobileKeyboardProxy.addEventListener("beforeinput", (event) => {
  if (event.inputType === "insertCompositionText" || composingNativeText) return;
  if (event.inputType === "insertLineBreak" || event.inputType === "insertParagraph") {
    event.preventDefault();
    clearMobileKeyboardProxy();
    form.requestSubmit();
    return;
  }
  if (event.inputType === "deleteContentBackward") {
    event.preventDefault();
    expressionField.executeCommand("deleteBackward");
    return;
  }
  if (event.inputType === "deleteContentForward") {
    event.preventDefault();
    expressionField.executeCommand("deleteForward");
    return;
  }
  if (event.inputType === "historyUndo" || event.inputType === "historyRedo") {
    event.preventDefault();
    expressionField.executeCommand(event.inputType === "historyUndo" ? "undo" : "redo");
    return;
  }
  if (event.inputType.startsWith("insert") && event.data) {
    event.preventDefault();
    insertNativeText(event.data);
    clearMobileKeyboardProxy();
  }
});
mobileKeyboardProxy.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.isComposing) return;
  event.preventDefault();
  clearMobileKeyboardProxy();
  form.requestSubmit();
});
mobileKeyboardProxy.addEventListener("input", () => {
  if (composingNativeText) return;
  insertNativeText(mobileKeyboardProxy.value);
  clearMobileKeyboardProxy();
});
mobileKeyboardProxy.addEventListener("paste", (event) => {
  const text = event.clipboardData?.getData("text/plain");
  if (!text) return;
  event.preventDefault();
  insertNativeText(text);
  clearMobileKeyboardProxy();
});

function createReadonlyMath(latex, className) {
  const field = document.createElement("math-field");
  field.className = className;
  field.readOnly = true;
  field.value = latex;
  return field;
}

function createMathViewport(field, className) {
  const viewport = document.createElement("div");
  viewport.className = `math-viewport ${className}`;
  viewport.tabIndex = 0;
  viewport.append(field);
  return viewport;
}

function createNotes(step) {
  const notes = document.createElement("div");
  notes.className = "step-notes";
  const derivativeNotes = new Map(
    step.notes
      .filter((note) => note.label.endsWith(" factor derivative"))
      .map((note) => [note.label, note])
  );
  if (step.rule === "Apply the product rule" && derivativeNotes.size === 2) {
    const table = document.createElement("table");
    table.className = "derivative-table";
    const caption = document.createElement("caption");
    caption.textContent = "Differentiate each factor once";
    const header = document.createElement("tr");
    for (const label of ["Factor", "Derivative"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      header.append(cell);
    }
    const head = document.createElement("thead");
    head.append(header);
    const body = document.createElement("tbody");
    for (const [label, note] of derivativeNotes) {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = label.startsWith("First") ? "First" : "Second";
      const derivative = document.createElement("td");
      derivative.append(
        createMathViewport(
          createReadonlyMath(note.expression_latex, "step-note-math"),
          "step-note-viewport"
        )
      );
      row.append(name, derivative);
      body.append(row);
    }
    table.append(caption, head, body);
    notes.append(table);
  }
  for (const note of step.notes) {
    if (derivativeNotes.has(note.label)) continue;
    const noteBlock = document.createElement("div");
    noteBlock.className = "step-note";
    const noteLabel = document.createElement("div");
    noteLabel.className = "step-note-label";
    noteLabel.textContent = note.label;
    const noteMath = createReadonlyMath(note.expression_latex, "step-note-math");
    noteBlock.append(noteLabel, createMathViewport(noteMath, "step-note-viewport"));
    notes.append(noteBlock);
  }
  return notes;
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
  const notes = createNotes(step);
  const transformation = document.createElement("div");
  transformation.className = "step-transformation";
  const before = createReadonlyMath(step.before_latex, "step-math");
  const beforeViewport = createMathViewport(before, "step-math-viewport");
  const arrow = document.createElement("div");
  arrow.className = "step-arrow";
  arrow.textContent = "→";
  arrow.setAttribute("aria-label", "becomes");
  const after = createReadonlyMath(step.after_latex, "step-math step-math-after");
  const afterViewport = createMathViewport(after, "step-math-viewport");
  transformation.append(beforeViewport, arrow, afterViewport);
  const verification = document.createElement("details");
  verification.className = "step-verification";
  const verificationSummary = document.createElement("summary");
  verificationSummary.textContent = "Why this step is valid";
  const verificationDetail = document.createElement("p");
  verificationDetail.textContent = `${step.verification_method}: ${step.verification_detail}`;
  verification.append(verificationSummary, verificationDetail);
  body.append(heading, explanation);
  if (step.notes.length > 0) body.append(notes);
  body.append(transformation, verification);
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
  solveButton.textContent = "Solving…";
  resultSection.classList.remove("hidden");
  errorBox.classList.add("hidden");
  statusText.classList.remove("is-error");
  stepsContainer.replaceChildren();
  try {
    const payload = await solverClient.solve(buildPayload(), (message) => {
      solveButton.textContent = message;
      statusText.textContent = message.replace("…", "");
      statusText.classList.add("is-loading");
    });
    statusText.classList.remove("is-loading");
    const completed = payload.status === "exact" || payload.status === "divergent";
    if (payload.status === "divergent") statusText.textContent = "Diverges";
    else if (payload.status === "exact") statusText.textContent = "Exact answer";
    else statusText.textContent = "No exact answer";
    asciiOutput.textContent = payload.formatted_ascii;
    if (completed) {
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
    statusText.classList.remove("is-loading");
    answerBlock.classList.add("hidden");
    workingBlock.classList.add("hidden");
    statusText.textContent = "Input error";
    statusText.classList.add("is-error");
    errorBox.textContent = error instanceof Error ? error.message : "Unexpected browser error.";
    errorBox.classList.remove("hidden");
  } finally {
    solveButton.disabled = false;
    solveButton.textContent = "Solve";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (expressionField.value.trim()) solve();
});

function insertSymbolTemplate(key) {
  const template = key.dataset.insert;
  if (!template) return;
  if (!usesMobileKeyboard()) expressionField.focus({ preventScroll: true });
  expressionField.insert(template, {
    insertionMode: "replaceSelection",
    selectionMode: "placeholder"
  });
  focusMobileKeyboard();
}

for (const key of document.querySelectorAll(".symbol-key")) {
  const label = key.querySelector("math-field");
  if (label) {
    label.tabIndex = -1;
    label.setAttribute("aria-hidden", "true");
  }
  key.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    insertSymbolTemplate(key);
  });
  key.addEventListener("click", (event) => {
    if (event.detail === 0) insertSymbolTemplate(key);
  });
}

for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    expressionField.value = example.dataset.expression;
    if (usesMobileKeyboard()) focusMobileKeyboard();
    else expressionField.focus();
  });
}

function warmSolver() {
  void solverClient.warmup().catch(() => {});
}

expressionField.addEventListener("focus", warmSolver, { once: true });

const connection = navigator.connection;
const constrainedConnection =
  connection?.saveData || connection?.effectiveType === "slow-2g" || connection?.effectiveType === "2g";
if (!constrainedConnection) {
  if ("requestIdleCallback" in window) window.requestIdleCallback(warmSolver, { timeout: 2500 });
  else window.setTimeout(warmSolver, 1200);
}
