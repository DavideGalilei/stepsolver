import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const frontendDirectory = dirname(fileURLToPath(import.meta.url));
const assetDirectory = resolve(frontendDirectory, "../src/stepsolver/web_assets");
const fontDirectory = resolve(assetDirectory, "fonts");

await mkdir(assetDirectory, { recursive: true });
await rm(fontDirectory, { force: true, recursive: true });
await cp(resolve(frontendDirectory, "node_modules/mathlive/fonts"), fontDirectory, {
  recursive: true
});

await build({
  bundle: true,
  entryPoints: [resolve(frontendDirectory, "vendor-entry.mjs")],
  format: "esm",
  legalComments: "external",
  minify: true,
  outfile: resolve(assetDirectory, "vendor.mjs"),
  target: "es2022"
});

const dependencies = [
  ["MathLive 0.110.0", "mathlive/LICENSE.txt"],
  ["CortexJS Compute Engine 0.113.0", "@cortex-js/compute-engine/LICENSE"]
];
const notices = [
  "Third-party software bundled with the StepSolver web interface",
  ""
];
for (const [name, licensePath] of dependencies) {
  notices.push(name, "=".repeat(name.length), "");
  notices.push(
    await readFile(resolve(frontendDirectory, "node_modules", licensePath), "utf8"),
    ""
  );
}
notices.push(
  await readFile(resolve(assetDirectory, "vendor.mjs.LEGAL.txt"), "utf8")
);
await writeFile(
  resolve(assetDirectory, "THIRD_PARTY_NOTICES.txt"),
  `${notices.join("\n").trim()}\n`,
  "utf8"
);
