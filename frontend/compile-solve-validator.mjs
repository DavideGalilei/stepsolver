import { readFile } from "node:fs/promises";

import Ajv2020 from "ajv/dist/2020.js";
import standaloneCode from "ajv/dist/standalone/index.js";
import { transform } from "esbuild";

export async function compileSolveValidator() {
  const schema = JSON.parse(await readFile(new URL("./solve-result.schema.json", import.meta.url), "utf8"));
  const ajv = new Ajv2020({ code: { source: true, esm: true } });
  const source = standaloneCode(ajv, ajv.compile(schema));
  const { code } = await transform(source, {
    format: "esm",
    target: "es2022",
    minify: true,
    banner: "// Generated from SolvePayload by frontend/build.mjs. See THIRD_PARTY_NOTICES.txt."
  });
  return code;
}
