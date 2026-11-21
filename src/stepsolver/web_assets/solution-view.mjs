function createReadonlyMath(latex, className) {
  const field = document.createElement("math-field");
  field.className = className;
  field.readOnly = true;
  field.value = latex;
  return field;
}

function createInlineMath(latex, className) {
  const math = document.createElement("math-span");
  math.className = className;
  math.textContent = latex;
  return math;
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
    if (note.label === "General power rule") noteBlock.classList.add("is-wide");
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

function createStepArrow() {
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.classList.add("step-arrow-icon");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("aria-hidden", "true");
  icon.setAttribute("focusable", "false");

  const shaft = document.createElementNS("http://www.w3.org/2000/svg", "path");
  shaft.setAttribute("d", "M5 12h14");
  const head = document.createElementNS("http://www.w3.org/2000/svg", "path");
  head.setAttribute("d", "m13 6 6 6-6 6");
  icon.append(shaft, head);

  const arrow = document.createElement("div");
  arrow.className = "step-arrow";
  arrow.setAttribute("role", "img");
  arrow.setAttribute("aria-label", "becomes");
  arrow.append(icon);
  return arrow;
}

function createStepExplanation(step) {
  const explanation = document.createElement("p");
  explanation.className = "step-explanation";
  for (const part of step.explanation_parts) {
    if (part.latex !== null) {
      explanation.append(createInlineMath(part.latex, "step-explanation-math"));
    } else if (part.text !== null) {
      explanation.append(document.createTextNode(part.text));
    }
  }
  return explanation;
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
  const explanation = createStepExplanation(step);
  const notes = createNotes(step);
  const transformation = document.createElement("div");
  transformation.className = "step-transformation";
  const before = createReadonlyMath(step.before_latex, "step-math");
  const beforeViewport = createMathViewport(before, "step-math-viewport");
  const arrow = createStepArrow();
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

export function createSolutionView() {
  const resultSection = document.querySelector("#result-section");
  const statusText = document.querySelector("#status-text");
  const answerBlock = document.querySelector("#answer-block");
  const workingBlock = document.querySelector("#working-block");
  const resultLatex = document.querySelector("#result-latex");
  const stepsContainer = document.querySelector("#steps");
  const asciiPanel = document.querySelector("#ascii-panel");
  const asciiOutput = document.querySelector("#ascii-output");
  const errorBox = document.querySelector("#error-box");
  enableTouchMathScrolling(document.querySelector(".result-viewport"));

  function reset() {
    resultSection.classList.remove("hidden");
    errorBox.classList.add("hidden");
    statusText.classList.remove("is-error");
    statusText.textContent = "";
    answerBlock.classList.add("hidden");
    workingBlock.classList.add("hidden");
    asciiPanel.classList.add("hidden");
    stepsContainer.replaceChildren();
    resultSection.scrollIntoView({ block: "start" });
  }

  function render(payload) {
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
    asciiPanel.classList.remove("hidden");
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
  }

  function renderError(error) {
    answerBlock.classList.add("hidden");
    workingBlock.classList.add("hidden");
    statusText.textContent = "Input error";
    statusText.classList.add("is-error");
    errorBox.textContent = error instanceof Error ? error.message : "Unexpected browser error.";
    errorBox.classList.remove("hidden");
  }

  return { reset, render, renderError };
}
