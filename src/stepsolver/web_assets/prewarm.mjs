// Start the browser solver before the editor bundle finishes loading.

import { createSolverClient } from "./runtime.mjs";

void createSolverClient().warmup().catch(() => {});
