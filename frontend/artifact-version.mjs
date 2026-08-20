import { createHash } from "node:crypto";

export function contentVersion(content) {
  return createHash("sha256").update(content).digest("hex").slice(0, 16);
}
