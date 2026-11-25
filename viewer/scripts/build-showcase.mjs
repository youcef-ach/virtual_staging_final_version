import { createHash } from "node:crypto";
import { copyFile, lstat, mkdir, readFile, readdir, realpath, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";

const root = resolve(fileURLToPath(new URL("../", import.meta.url)));
const source = join(root, "public", "demo");
const output = join(root, "dist-showcase");
const sceneFiles = ["manifest.json", "poses.json", "navigation_graph.json", "proxy_shell.glb", "proxy_details.glb", "reference.glb", "asset.glb"];
for (const viewpoint of ["demo-entry", "demo-window"]) {
  for (const face of ["px", "nx", "py", "ny", "pz", "nz"]) sceneFiles.push(`${viewpoint}/${face}.jpg`);
}
const evidenceFiles = ["input.webp", "mask2former.webp", "sam.webp", "difference.webp", "manifest.json"].map(file => `evidence/${file}`);
const fixture = JSON.parse(await readFile(join(source, "asset-manifest.json"), "utf8"));
if (fixture.provenance !== "original_synthetic_blender" || fixture.assets.length !== sceneFiles.length) {
  throw new Error("Expected the verified synthetic room fixture");
}
const entries = new Map(fixture.assets.map(entry => [entry.path, entry]));
const sha256 = buffer => createHash("sha256").update(buffer).digest("hex");
const demoFiles = [...sceneFiles, ...evidenceFiles, "asset-manifest.json"];

for (const file of demoFiles) {
  const path = join(source, file);
  if (!(await lstat(path)).isFile() || (await realpath(path)) !== path) throw new Error(`Invalid demo asset ${file}`);
  if (sceneFiles.includes(file)) {
    const bytes = await readFile(path);
    if (entries.get(file)?.sha256 !== sha256(bytes)) throw new Error(`Fixture checksum mismatch for ${file}`);
  }
}
const run = JSON.parse(await readFile(join(source, "evidence", "manifest.json"), "utf8"));
if (!run.models?.mask2former?.revision || !run.models?.sam?.sha256 || !run.inputSha256) {
  throw new Error("Missing model run provenance");
}
for (const file of evidenceFiles.filter(name => name.endsWith(".webp"))) {
  const asset = Object.values(run.assets ?? {}).find(entry => `evidence/${entry.path}` === file);
  if (asset?.sha256 !== sha256(await readFile(join(source, file)))) throw new Error(`Model output checksum mismatch for ${file}`);
}
const target = await realpath(output).catch(error => {
  if (error.code !== "ENOENT") throw error;
  return output;
});
if (target !== output || relative(root, target) !== "dist-showcase") throw new Error("Invalid showcase output directory");

await build({
  root, mode: "showcase", base: "./", publicDir: false,
  build: { outDir: output, emptyOutDir: true, sourcemap: false },
});
for (const file of demoFiles) {
  const destination = join(output, "demo", file);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(join(source, file), destination);
}
const chunks = await readdir(join(output, "assets"));
if (!chunks.length || !chunks.every(file => /^[\w-][\w.-]*\.(js|css)$/.test(file))) throw new Error("Unexpected viewer build output");
const files = ["index.html", ...chunks.map(file => `assets/${file}`), ...demoFiles.map(file => `demo/${file}`)].sort();
await writeFile(join(output, "demo-files.json"), JSON.stringify({ files }, null, 2) + "\n");
console.log(`Standalone showcase built with ${files.length} allowed files.`);
