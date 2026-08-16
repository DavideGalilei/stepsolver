// Assemble the interactive GitHub Pages artifact.

import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = resolve(frontendDirectory, "..");
const outputDirectory = resolve(repositoryDirectory, "_site");
const sourceAssets = resolve(repositoryDirectory, "src/stepsolver/web_assets");
const wheelPath = process.env.STEPSOLVER_WHEEL;

if (!wheelPath) throw new Error("STEPSOLVER_WHEEL must point to the built wheel");

await rm(outputDirectory, { force: true, recursive: true });
await mkdir(resolve(outputDirectory, "static"), { recursive: true });
await mkdir(resolve(outputDirectory, "packages"), { recursive: true });
await cp(sourceAssets, resolve(outputDirectory, "static"), { recursive: true });
await cp(resolve(sourceAssets, "index.html"), resolve(outputDirectory, "index.html"));
await rm(resolve(outputDirectory, "static/index.html"));
const wheelName = basename(wheelPath);
await cp(
  resolve(repositoryDirectory, wheelPath),
  resolve(outputDirectory, "packages", wheelName)
);
const worker = await readFile(resolve(frontendDirectory, "browser-worker.mjs"), "utf8");
await writeFile(
  resolve(outputDirectory, "static/browser-worker.mjs"),
  worker.replace("__STEPSOLVER_WHEEL__", wheelName),
  "utf8"
);
await cp(
  resolve(frontendDirectory, "browser-runtime.mjs"),
  resolve(outputDirectory, "static/runtime.mjs")
);
await writeFile(resolve(outputDirectory, ".nojekyll"), "", "utf8");

const indexPath = resolve(outputDirectory, "index.html");
const index = await readFile(indexPath, "utf8");
await writeFile(indexPath, index.replaceAll('href="/"', 'href="./"'), "utf8");
