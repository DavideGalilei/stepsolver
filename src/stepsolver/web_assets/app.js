"use strict";

import { createSolverClient } from "./runtime.mjs";
import { ComputeEngine, MathfieldElement } from "./vendor.mjs";

MathfieldElement.fontsDirectory = new URL("./fonts/", import.meta.url).href;
MathfieldElement.soundsDirectory = null;

const computeEngine = new ComputeEngine();
const solverClient = createSolverClient();
const form = document.querySelector("#solve-form");
const mathToolbar = document.querySelector(".math-toolbar");
const expressionField = document.querySelector("#expression-field");
const mobileKeyboardProxy = document.querySelector("#mobile-keyboard-proxy");
const mathKeyboardButton = document.querySelector("#math-keyboard-button");
const solveButton = document.querySelector("#solve-button");
const resultSection = document.querySelector("#result-section");
const statusText = document.querySelector("#status-text");
const answerBlock = document.querySelector("#answer-block");
const workingBlock = document.querySelector("#working-block");
const resultLatex = document.querySelector("#result-latex");
const stepsContainer = document.querySelector("#steps");
const asciiOutput = document.querySelector("#ascii-output");
const errorBox = document.querySelector("#error-box");
const themeToggle = document.querySelector("#theme-toggle");
const themePreference = window.matchMedia("(prefers-color-scheme: dark)");
const themeStorageKey = "stepsolver-theme";

function storedTheme() {
  try {
    const theme = window.localStorage.getItem(themeStorageKey);
    return theme === "light" || theme === "dark" ? theme : null;
  } catch (_) {
    return null;
  }
}

function effectiveTheme() {
  return document.documentElement.dataset.theme ?? (themePreference.matches ? "dark" : "light");
}

function updateThemeToggle() {
  const dark = effectiveTheme() === "dark";
  const label = dark ? "Switch to light theme" : "Switch to dark theme";
  themeToggle.setAttribute("aria-pressed", String(dark));
  themeToggle.setAttribute("aria-label", label);
  themeToggle.title = label;
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(themeStorageKey, theme);
  } catch (_) {
    // The selected theme still applies for the current page.
  }
  updateThemeToggle();
}

themeToggle.addEventListener("click", () => {
  setTheme(effectiveTheme() === "dark" ? "light" : "dark");
});

themePreference.addEventListener("change", () => {
  if (storedTheme() === null) updateThemeToggle();
});

window.addEventListener("storage", (event) => {
  if (event.key !== themeStorageKey) return;
  if (event.newValue === "light" || event.newValue === "dark") {
    document.documentElement.dataset.theme = event.newValue;
  } else {
    delete document.documentElement.dataset.theme;
  }
  updateThemeToggle();
});

updateThemeToggle();

function hideMathVirtualKeyboard() {
  window.mathVirtualKeyboard?.hide();
  mathKeyboardButton.setAttribute("aria-pressed", "false");
}

const mobileKeyboardQuery = window.matchMedia("(hover: none) and (pointer: coarse)");
const nativeMathTemplates = new Map([
  ["^", "^{#0}"],
  ["_", "_{#0}"]
]);
const mobileKeyboardSentinel = "\u2060";
let composingNativeText = false;
let lastNativeEnterAt = 0;

function usesMobileKeyboard() {
  return mobileKeyboardQuery.matches;
}

function focusMobileKeyboard() {
  if (!usesMobileKeyboard()) return;
  hideMathVirtualKeyboard();
  mobileKeyboardProxy.focus({ preventScroll: true });
  resetMobileKeyboardProxy();
}

function insertNativeCharacter(character) {
  const template = nativeMathTemplates.get(character);
  expressionField.insert(template ?? character, {
    insertionMode: "replaceSelection",
    selectionMode: template ? "placeholder" : "after"
  });
}

function insertNativeText(text) {
  if (!text) return;
  for (const character of text) insertNativeCharacter(character);
  showNativeCaret(true);
}

function mobileProxyText() {
  return mobileKeyboardProxy.value.startsWith(mobileKeyboardSentinel)
    ? mobileKeyboardProxy.value.slice(mobileKeyboardSentinel.length)
    : mobileKeyboardProxy.value;
}

function resetMobileKeyboardProxy() {
  mobileKeyboardProxy.value = mobileKeyboardSentinel;
  mobileKeyboardProxy.setSelectionRange(
    mobileKeyboardSentinel.length,
    mobileKeyboardSentinel.length
  );
}

function addSystemRow() {
  const added = expressionField.executeCommand("addRowAfter");
  if (added) showNativeCaret(true);
  return added;
}

function currentMathCellIsEmpty() {
  // MathLive 0.110 has no public current-cell API. Keep this adapter isolated.
  const model = expressionField._mathfield?.model;
  let atom = model?.at(model.position);
  while (atom && !Array.isArray(atom.parentBranch)) atom = atom.parent;
  if (!atom || !Array.isArray(atom.parentBranch)) return false;
  const [row, column] = atom.parentBranch;
  const cell = atom.parent?.getCell?.(row, column);
  return (
    Array.isArray(cell) &&
    cell.every((cellAtom) => cellAtom.type === "first" || cellAtom.type === "placeholder")
  );
}

function removeEmptySystemRow() {
  return currentMathCellIsEmpty() && expressionField.executeCommand("removeRow");
}

function handleNativeEnter() {
  const now = window.performance.now();
  if (now - lastNativeEnterAt < 250) return;
  lastNativeEnterAt = now;
  const addedRow = addSystemRow();
  resetMobileKeyboardProxy();
  if (!addedRow) form.requestSubmit();
}

function showNativeCaret(show) {
  const mathfieldContent = expressionField.shadowRoot?.querySelector('[part="content"]');
  mathfieldContent?.classList.toggle("ML__focused", show);
  if (!show) return;
  window.requestAnimationFrame(() => {
    const renderedContent = expressionField.shadowRoot?.querySelector('[part="content"]');
    renderedContent?.classList.add("ML__focused");
  });
}

expressionField.addEventListener("pointerup", focusMobileKeyboard);
expressionField.addEventListener("beforeinput", (event) => {
  if (usesMobileKeyboard()) return;
  if (event.inputType === "insertLineBreak" || event.inputType === "insertParagraph") {
    if (addSystemRow()) event.preventDefault();
    return;
  }
  if (
    (event.inputType === "deleteContentBackward" ||
      event.inputType === "deleteContentForward") &&
    removeEmptySystemRow()
  ) {
    event.preventDefault();
  }
});
mobileKeyboardProxy.addEventListener("pointerdown", (event) => {
  const offset = expressionField.getOffsetFromPoint(event.clientX, event.clientY, { bias: 0 });
  expressionField.position = offset >= 0 ? offset : expressionField.lastOffset;
  showNativeCaret(true);
});
mobileKeyboardProxy.addEventListener("focus", () => {
  expressionField.classList.add("is-mobile-editing");
  resetMobileKeyboardProxy();
  showNativeCaret(true);
  hideMathVirtualKeyboard();
});
mobileKeyboardProxy.addEventListener("blur", () => {
  expressionField.classList.remove("is-mobile-editing");
  showNativeCaret(false);
  resetMobileKeyboardProxy();
});
mobileKeyboardProxy.addEventListener("compositionstart", () => {
  composingNativeText = true;
});
mobileKeyboardProxy.addEventListener("compositionend", (event) => {
  composingNativeText = false;
  insertNativeText(event.data || mobileProxyText());
  resetMobileKeyboardProxy();
});
mobileKeyboardProxy.addEventListener("beforeinput", (event) => {
  if (event.inputType === "insertCompositionText" || composingNativeText) return;
  if (event.inputType === "insertLineBreak" || event.inputType === "insertParagraph") {
    event.preventDefault();
    handleNativeEnter();
    return;
  }
  if (event.inputType === "deleteContentBackward") {
    event.preventDefault();
    if (!removeEmptySystemRow()) expressionField.executeCommand("deleteBackward");
    resetMobileKeyboardProxy();
    showNativeCaret(true);
    return;
  }
  if (event.inputType === "deleteContentForward") {
    event.preventDefault();
    if (!removeEmptySystemRow()) expressionField.executeCommand("deleteForward");
    resetMobileKeyboardProxy();
    showNativeCaret(true);
    return;
  }
  if (event.inputType === "historyUndo" || event.inputType === "historyRedo") {
    event.preventDefault();
    expressionField.executeCommand(event.inputType === "historyUndo" ? "undo" : "redo");
    resetMobileKeyboardProxy();
    showNativeCaret(true);
    return;
  }
  if (event.inputType.startsWith("insert") && event.data) {
    event.preventDefault();
    insertNativeText(event.data);
    resetMobileKeyboardProxy();
  }
});
mobileKeyboardProxy.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.isComposing) return;
  event.preventDefault();
  handleNativeEnter();
});
mobileKeyboardProxy.addEventListener("input", () => {
  if (composingNativeText) return;
  const text = mobileProxyText();
  if (!mobileKeyboardProxy.value) expressionField.executeCommand("deleteBackward");
  else insertNativeText(text);
  resetMobileKeyboardProxy();
  showNativeCaret(true);
});
mobileKeyboardProxy.addEventListener("paste", (event) => {
  const text = event.clipboardData?.getData("text/plain");
  if (!text) return;
  event.preventDefault();
  insertNativeText(text);
  resetMobileKeyboardProxy();
});

mathKeyboardButton.addEventListener("click", () => {
  mobileKeyboardProxy.blur();
  expressionField.focus({ preventScroll: true });
  window.mathVirtualKeyboard?.show();
  mathKeyboardButton.setAttribute("aria-pressed", "true");
});

function createReadonlyMath(latex, className) {
  const field = document.createElement("math-field");
  field.className = className;
  field.readOnly = true;
  field.value = latex;
  return field;
}

function enableTouchMathScrolling(viewport) {
  let gesture = null;
  const finishGesture = (event) => {
    if (gesture?.pointerId !== event.pointerId) return;
    if (viewport.hasPointerCapture(event.pointerId)) {
      viewport.releasePointerCapture(event.pointerId);
    }
    gesture = null;
  };
  viewport.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "touch") return;
    gesture = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startScrollLeft: viewport.scrollLeft,
      horizontal: null
    };
  });
  viewport.addEventListener("pointermove", (event) => {
    if (gesture?.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - gesture.startX;
    const deltaY = event.clientY - gesture.startY;
    if (gesture.horizontal === null) {
      if (Math.max(Math.abs(deltaX), Math.abs(deltaY)) < 6) return;
      gesture.horizontal = Math.abs(deltaX) > Math.abs(deltaY);
      if (!gesture.horizontal) {
        gesture = null;
        return;
      }
      viewport.setPointerCapture(event.pointerId);
    }
    viewport.scrollLeft = gesture.startScrollLeft - deltaX;
  });
  viewport.addEventListener("pointerup", finishGesture);
  viewport.addEventListener("pointercancel", finishGesture);
  viewport.addEventListener("lostpointercapture", finishGesture);
}

function createMathViewport(field, className) {
  const viewport = document.createElement("div");
  viewport.className = `math-viewport ${className}`;
  viewport.tabIndex = 0;
  viewport.append(field);
  enableTouchMathScrolling(viewport);
  return viewport;
}

enableTouchMathScrolling(document.querySelector(".result-viewport"));

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

function createConstraints(step) {
  const constraints = document.createElement("div");
  constraints.className = "step-constraints";
  const heading = document.createElement("h4");
  heading.textContent = "Domain restrictions introduced here";
  constraints.append(heading);
  for (const constraint of step.introduced_constraints) {
    const item = document.createElement("div");
    item.className = "step-constraint";
    const math = createReadonlyMath(constraint.expression_latex, "step-constraint-math");
    const explanation = document.createElement("p");
    explanation.textContent = constraint.explanation;
    item.append(createMathViewport(math, "step-note-viewport"), explanation);
    constraints.append(item);
  }
  return constraints;
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
  if (step.introduced_constraints.length > 0) body.append(createConstraints(step));
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
  if (usesMobileKeyboard()) mobileKeyboardProxy.blur();
  hideMathVirtualKeyboard();
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
    const completed =
      payload.status === "exact" ||
      payload.status === "divergent" ||
      payload.status === "undefined";
    if (payload.status === "divergent") statusText.textContent = "Diverges";
    else if (payload.status === "undefined") statusText.textContent = "Undefined";
    else if (payload.status === "exact") {
      statusText.textContent = payload.result_latex?.includes("\\approx")
        ? "Numerical answer"
        : "Exact answer";
    }
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
  const useNativeMobileKeyboard = usesMobileKeyboard();
  if (!useNativeMobileKeyboard) expressionField.focus({ preventScroll: true });
  const insertionOptions = {
    format: "latex",
    mode: "math",
    selectionMode: "placeholder"
  };
  if (key.dataset.structure === "system") {
    expressionField.setValue(template, {
      ...insertionOptions,
      insertionMode: "replaceAll"
    });
  } else {
    expressionField.insert(template, {
      ...insertionOptions,
      insertionMode: "replaceSelection",
      scrollIntoView: false
    });
  }
  if (useNativeMobileKeyboard) focusMobileKeyboard();
  else {
    expressionField.focus({ preventScroll: true });
    showNativeCaret(true);
  }
}

function symbolKeyFromEvent(event) {
  const target = event.target;
  if (!(target instanceof Element)) return null;
  const key = target.closest(".symbol-key");
  return key instanceof HTMLButtonElement && mathToolbar.contains(key) ? key : null;
}

for (const key of document.querySelectorAll(".symbol-key")) {
  const label = key.querySelector("math-field");
  if (label) {
    label.tabIndex = -1;
    label.setAttribute("aria-hidden", "true");
  }
}

let lastMouseInsertion = null;
mathToolbar.addEventListener(
  "pointerdown",
  (event) => {
    if (event.pointerType !== "mouse" || event.button !== 0) return;
    const key = symbolKeyFromEvent(event);
    if (!key) return;
    event.preventDefault();
    event.stopPropagation();
    lastMouseInsertion = { key, at: window.performance.now() };
    insertSymbolTemplate(key);
  },
  { capture: true }
);

mathToolbar.addEventListener(
  "click",
  (event) => {
    const key = symbolKeyFromEvent(event);
    if (!key) return;
    event.preventDefault();
    event.stopPropagation();
    if (
      lastMouseInsertion?.key === key &&
      window.performance.now() - lastMouseInsertion.at < 1000
    ) {
      lastMouseInsertion = null;
      return;
    }
    lastMouseInsertion = null;
    insertSymbolTemplate(key);
  },
  { capture: true }
);

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
