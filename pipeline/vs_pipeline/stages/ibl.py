"""Stage 8 (G5) -- IBL pre-computation: diffuse irradiance + GGX specular chain per sweep.

Locked decisions (2026-07-15, PLAN G5):
- PURE-NUMPY convolution; no cmft binary. ``Convolver`` below is the clean seam where a
  cmft-backed implementation can drop in later as a per-sweep speed optimization.
- Outputs ``.npy`` first-class (same Windows-OpenEXR rationale as the G-buffer stage) +
  best-effort ``.exr``; KTX2 UASTC HDR is deferred to P4.
- Diffuse irradiance 64x64/face; GGX specular chain 256^2 base x 5 mips (256 -> 16),
  roughness m / (mips - 1); 512^2 base kept as an upgrade knob (judged at G8).
- Validation slice: interior trio (sweeps 50/60/175) + one garden sweep (indices 0-2
  are confirmed outdoor) before the full-property run.

Interim blind highlight boost -- PLACEHOLDER until segmentation-guided highlight
reconstruction (post-G6), exactly as locked in PLAN:

    C_hdr = C_lin * (1 + 12 * max(0, Y - 0.85) ** 1.5)

Orientation note: the extracted canonical faces (F4.0, Set A transforms) form a
seam-consistent cubemap (proven visually in F4.4). The cosine and GGX kernels are
rotationally symmetric, so the convolution result -- as a cubemap in the SAME face
layout as the panos -- is invariant to any global orthogonal relabeling of texel
directions. We therefore use the standard cubemap face->direction convention below;
the validation slice must confirm orientation visually (the bright window direction
in the irradiance map matches the pano).

CLI:
    python -m vs_pipeline.stages.ibl bake --panos extracted_cubemaps -o ibl_out \
        --sweeps "1,50,60,175"
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

IBL_SCHEMA_VERSION = 1
FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")

# Interim blind highlight boost (locked formula -- see module docstring).
BOOST_KNEE = 0.85
BOOST_STRENGTH = 12.0
BOOST_POWER = 1.5

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_CHUNK = 4096  # output-direction rows per convolution chunk (memory bound)


# --------------------------------------------------------------------------- color


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Standard piecewise sRGB EOTF (float in [0, 1])."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`srgb_to_linear` (values clipped to >= 0)."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    return np.where(
        x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055
    ).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float32) @ _LUMA


def highlight_boost(lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interim blind boost: returns (boosted_rgb, per-pixel factor).

    PLACEHOLDER by design until segmentation-guided highlight reconstruction
    (post-G6). Identity below Y = BOOST_KNEE, monotonically increasing above.
    """
    y = luminance(lin)
    factor = 1.0 + BOOST_STRENGTH * np.maximum(0.0, y - BOOST_KNEE) ** BOOST_POWER
    return (lin * factor[..., None]).astype(np.float32), factor.astype(np.float32)


# ------------------------------------------------------------------- cube geometry


def _face_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) texel-center coords in [-1, 1]; a = right (cols), b = down (rows)."""
    a = ((np.arange(size, dtype=np.float32) + 0.5) / size) * 2.0 - 1.0
    aa, bb = np.meshgrid(a, a)
    return aa, bb


def face_directions(face: str, size: int) -> np.ndarray:
    """Unit world directions (size, size, 3) -- standard cubemap convention.

    Any global orthogonal relabeling would produce the identical convolved
    cubemap (rotationally symmetric kernels); internal consistency (correct
    face adjacency) is what matters, and it holds for the canonical faces.
    """
    aa, bb = _face_grid(size)
    one = np.ones_like(aa)
    if face == "px":
        d = (one, -bb, -aa)
    elif face == "nx":
        d = (-one, -bb, aa)
    elif face == "py":
        d = (aa, one, bb)
    elif face == "ny":
        d = (aa, -one, -bb)
    elif face == "pz":
        d = (aa, -bb, one)
    elif face == "nz":
        d = (-aa, -bb, -one)
    else:
        raise ValueError(f"unknown face: {face}")
    dirs = np.stack(d, axis=-1)
    return (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).astype(np.float32)


def face_solid_angles(size: int) -> np.ndarray:
    """Per-texel solid angle (identical for all six faces); sums to ~4*pi/6."""
    aa, bb = _face_grid(size)
    return ((2.0 / size) ** 2 / (1.0 + aa * aa + bb * bb) ** 1.5).astype(np.float32)


def _resized(img: np.ndarray, size: int) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _flatten_env(
    env_faces: dict[str, np.ndarray], size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All six faces at ``size`` -> (dirs (M, 3), solid_angles (M,), colors (M, 3))."""
    sa_face = face_solid_angles(size).reshape(-1)
    dirs, sas, cols = [], [], []
    for face in FACE_NAMES:
        dirs.append(face_directions(face, size).reshape(-1, 3))
        sas.append(sa_face)
        cols.append(_resized(env_faces[face], size).reshape(-1, 3))
    return np.concatenate(dirs), np.concatenate(sas), np.concatenate(cols)


# ---------------------------------------------------------------------- convolvers


@dataclass
class IblParams:
    diffuse_size: int = 64
    irr_source_size: int = 32
    spec_size: int = 256
    spec_mips: int = 5

    def roughness_values(self) -> list[float]:
        if self.spec_mips <= 1:
            return [0.0]
        return [m / (self.spec_mips - 1) for m in range(self.spec_mips)]

    def mip_sizes(self) -> list[int]:
        return [max(1, self.spec_size >> m) for m in range(self.spec_mips)]

    def to_dict(self) -> dict[str, int]:
        return {
            "diffuse_size": self.diffuse_size,
            "irr_source_size": self.irr_source_size,
            "spec_size": self.spec_size,
            "spec_mips": self.spec_mips,
        }


class Convolver:
    """Prefilter interface -- THE cmft seam.

    A cmft-backed implementation (external binary, GPU-accelerated) can subclass
    this and be swapped in as a pure speed optimization; outputs must keep the
    same shapes, face order, and roughness schedule.
    """

    def __init__(self, params: IblParams) -> None:
        self.params = params

    def irradiance(self, env_faces: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def specular_chain(self, env_faces: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError


class NumpyConvolver(Convolver):
    """Reference pure-numpy prefilter (locked engine for G5)."""

    def irradiance(self, env_faces: dict[str, np.ndarray]) -> np.ndarray:
        """Cosine-weighted irradiance / pi -> (6, d, d, 3); constant env == identity."""
        p = self.params
        dirs_e, sa, col = _flatten_env(env_faces, p.irr_source_size)
        weighted = col * sa[:, None]
        faces = []
        for face in FACE_NAMES:
            dirs_o = face_directions(face, p.diffuse_size).reshape(-1, 3)
            cos = np.maximum(dirs_o @ dirs_e.T, 0.0)
            out = (cos @ weighted) / np.pi
            faces.append(out.reshape(p.diffuse_size, p.diffuse_size, 3))
        return np.stack(faces).astype(np.float32)

    def specular_chain(self, env_faces: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Prefiltered GGX chain (split-sum, n = v = r), normalized weighted sums.

        Mip 0 (roughness 0) is the downsampled environment itself (mirror).
        Weight_i = D_GGX(cos_theta_h, alpha) * max(0, r.l_i) * dOmega_i with
        cos_theta_h^2 = (1 + r.l) / 2; normalization keeps constants exact.
        """
        p = self.params
        sizes = p.mip_sizes()
        rough = p.roughness_values()
        chain: list[np.ndarray] = [
            np.stack([_resized(env_faces[f], sizes[0]) for f in FACE_NAMES]).astype(np.float32)
        ]
        for m in range(1, p.spec_mips):
            size, r = sizes[m], rough[m]
            alpha = max(r * r, 1e-4)
            a2 = alpha * alpha
            src = min(p.spec_size, 64 if r < 0.3 else 32)
            dirs_e, sa, col = _flatten_env(env_faces, src)
            faces = []
            for face in FACE_NAMES:
                dirs_o = face_directions(face, size).reshape(-1, 3)
                res = np.empty((dirs_o.shape[0], 3), dtype=np.float32)
                for s in range(0, dirs_o.shape[0], _CHUNK):
                    mu = np.clip(dirs_o[s : s + _CHUNK] @ dirs_e.T, -1.0, 1.0)
                    c2 = (1.0 + mu) * 0.5
                    den = c2 * (a2 - 1.0) + 1.0
                    w = (a2 / (np.pi * den * den)) * np.maximum(mu, 0.0) * sa
                    num = w @ col
                    norm = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
                    res[s : s + _CHUNK] = num / norm
                faces.append(res.reshape(size, size, 3))
            chain.append(np.stack(faces))
        return chain


# ---------------------------------------------------------------------------- bake


def parse_sweep_selection(spec: str) -> set[int]:
    """'1,50,60,175' or '0-2,50' -> set of sweep indices."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _outputs_current(meta_path: Path, sweep_out: Path, params: IblParams) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("params") != params.to_dict():
        return False
    files = meta.get("files", {})
    names = [files.get("irradiance", "")] + list(files.get("specular_mips", []))
    return all(name and (sweep_out / name).is_file() for name in names)


def _try_write_exr(exr_dir: Path, name: str, cube: np.ndarray) -> bool:
    """Best-effort per-face EXR write; returns False if the cv2 build lacks EXR."""
    exr_dir.mkdir(parents=True, exist_ok=True)
    for i, face in enumerate(FACE_NAMES):
        path = exr_dir / f"{name}_{face}.exr"
        try:
            ok = cv2.imwrite(
                str(path), cv2.cvtColor(np.ascontiguousarray(cube[i]), cv2.COLOR_RGB2BGR)
            )
        except cv2.error:
            ok = False
        if not ok:
            return False
    return True


def _tonemap_u8(lin_rgb: np.ndarray) -> np.ndarray:
    return (linear_to_srgb(np.clip(lin_rgb, 0.0, 1.0)) * 255.0).astype(np.uint8)


def _panel(img_u8: np.ndarray, label: str, size: int = 128) -> np.ndarray:
    interp = cv2.INTER_NEAREST if img_u8.shape[0] < size else cv2.INTER_AREA
    img = cv2.resize(np.ascontiguousarray(img_u8), (size, size), interpolation=interp)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def _review_image(
    orig: dict[str, np.ndarray],
    boosted: dict[str, np.ndarray],
    factors: dict[str, np.ndarray],
    irr: np.ndarray,
    mips: list[np.ndarray],
    boost_max: float,
) -> np.ndarray:
    """Per-face rows: pano | boosted | boost heatmap | irradiance | mip0..N (RGB)."""
    rows = []
    denom = max(boost_max - 1.0, 1e-6)
    for i, face in enumerate(FACE_NAMES):
        panels = [
            _panel((np.clip(orig[face], 0, 1) * 255).astype(np.uint8), f"{face} pano"),
            _panel(_tonemap_u8(boosted[face]), "boosted"),
        ]
        heat_u8 = (np.clip((factors[face] - 1.0) / denom, 0, 1) * 255).astype(np.uint8)
        heat = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        panels.append(_panel(heat, "boost"))
        panels.append(_panel(_tonemap_u8(irr[i]), "irr"))
        panels.extend(_panel(_tonemap_u8(mip[i]), f"mip{m}") for m, mip in enumerate(mips))
        rows.append(np.hstack([np.pad(p, ((0, 2), (0, 2), (0, 0))) for p in panels]))
    return np.vstack(rows)


def bake(
    panos_dir: Path,
    out_dir: Path,
    params: IblParams,
    sweep_indices: set[int] | None = None,
    force: bool = False,
    write_exr: bool = True,
    write_review: bool = True,
) -> dict[str, list[str]]:
    """Bake IBL products for the selected sweeps; idempotent per sweep."""
    manifest = json.loads((panos_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = list(manifest.get("sweeps", []))
    if sweep_indices is not None:
        entries = [e for e in entries if int(e["index"]) in sweep_indices]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[str]] = {"baked": [], "skipped": [], "failed": []}
    exr_enabled = write_exr
    conv = NumpyConvolver(params)

    for entry in entries:
        uuid = str(entry["sweep_uuid"])
        idx = int(entry["index"])
        faces_rel = entry.get("faces", {})
        if sorted(faces_rel) != sorted(FACE_NAMES):
            print(f"[{idx:03d}] {uuid}: incomplete faces -> FAILED")
            summary["failed"].append(uuid)
            continue
        sweep_out = out_dir / uuid
        meta_path = sweep_out / "ibl_meta.json"
        if not force and _outputs_current(meta_path, sweep_out, params):
            print(f"[{idx:03d}] {uuid}: up to date -> skipped")
            summary["skipped"].append(uuid)
            continue

        orig: dict[str, np.ndarray] = {}
        for face in FACE_NAMES:
            img = cv2.imread(str(panos_dir / faces_rel[face]), cv2.IMREAD_COLOR)
            if img is None:
                break
            orig[face] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if len(orig) != 6:
            print(f"[{idx:03d}] {uuid}: unreadable face(s) -> FAILED")
            summary["failed"].append(uuid)
            continue

        boosted: dict[str, np.ndarray] = {}
        factors: dict[str, np.ndarray] = {}
        boost_fracs: list[float] = []
        boost_max = 1.0
        for face, srgb in orig.items():
            b, fac = highlight_boost(srgb_to_linear(srgb))
            boosted[face], factors[face] = b, fac
            boost_fracs.append(float((fac > 1.0).mean()))
            boost_max = max(boost_max, float(fac.max()))
        env = {f: _resized(b, params.spec_size) for f, b in boosted.items()}

        irr = conv.irradiance(env)
        mips = conv.specular_chain(env)

        products = [("irradiance", irr)] + [
            (f"specular_mip{m}", mip) for m, mip in enumerate(mips)
        ]
        finite = all(bool(np.isfinite(arr).all()) for _, arr in products)
        base_luma = float(np.mean([luminance(v).mean() for v in env.values()]))
        mip_lumas = [float(luminance(mip).mean()) for mip in mips]
        irr_luma = float(luminance(irr).mean())

        sweep_out.mkdir(parents=True, exist_ok=True)
        files: dict[str, Any] = {"irradiance": "irradiance.npy", "specular_mips": []}
        np.save(sweep_out / "irradiance.npy", irr.astype(np.float32))
        for m, mip in enumerate(mips):
            name = f"specular_mip{m}.npy"
            np.save(sweep_out / name, mip.astype(np.float32))
            files["specular_mips"].append(name)

        if exr_enabled:
            ok = _try_write_exr(sweep_out / "exr", "irradiance", irr)
            for m, mip in enumerate(mips):
                ok = ok and _try_write_exr(sweep_out / "exr", f"specular_mip{m}", mip)
            if not ok:
                exr_enabled = False
                print("  WARNING: cv2 build lacks EXR write support -> .npy only from here")

        if write_review:
            review_dir = out_dir / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            composite = _review_image(orig, boosted, factors, irr, mips, boost_max)
            cv2.imwrite(
                str(review_dir / f"ibl_{idx:03d}_{uuid}.png"),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            )

        stats = {
            "finite": finite,
            "boost_pixel_fraction": float(np.mean(boost_fracs)),
            "boost_max_factor": boost_max,
            "env_base_mean_luma": base_luma,
            "irradiance_mean_luma": irr_luma,
            "mip_mean_luma": mip_lumas,
        }
        meta = {
            "schema_version": IBL_SCHEMA_VERSION,
            "sweep_uuid": uuid,
            "index": idx,
            "params": params.to_dict(),
            "face_order": list(FACE_NAMES),
            "roughness_per_mip": params.roughness_values(),
            "boost": {
                "knee": BOOST_KNEE,
                "strength": BOOST_STRENGTH,
                "power": BOOST_POWER,
                "note": "interim blind boost; placeholder until segmentation-guided "
                "highlight reconstruction",
            },
            "files": files,
            "stats": stats,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        ratio = mip_lumas[-1] / max(base_luma, 1e-9)
        print(
            f"[{idx:03d}] {uuid}: boost {np.mean(boost_fracs):6.2%} px "
            f"(max x{boost_max:.2f})  luma base {base_luma:.4f} irr {irr_luma:.4f} "
            f"mip4/base {ratio:.3f}  {'OK' if finite else 'NAN/INF!'}"
        )
        summary["baked"].append(uuid)

    seeds_path = assemble_seeds(out_dir)
    print(
        f"baked {len(summary['baked'])}, skipped {len(summary['skipped'])}, "
        f"failed {len(summary['failed'])} -> {seeds_path}"
    )
    return summary


def assemble_seeds(out_dir: Path) -> Path:
    """Rebuild lighting_seeds.json from every per-sweep meta on disk (resumable)."""
    sweeps = []
    for meta_path in sorted(out_dir.glob("*/ibl_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        uuid = meta["sweep_uuid"]
        sweeps.append(
            {
                "sweep_uuid": uuid,
                "index": meta["index"],
                "irradiance": f"{uuid}/{meta['files']['irradiance']}",
                "specular_mips": [f"{uuid}/{n}" for n in meta["files"]["specular_mips"]],
                "roughness_per_mip": meta["roughness_per_mip"],
                "params": meta["params"],
                "stats": meta["stats"],
            }
        )
    sweeps.sort(key=lambda s: int(s["index"]))
    payload = {
        "schema_version": IBL_SCHEMA_VERSION,
        "engine": "numpy-v1 (cmft drop-in seam: vs_pipeline.stages.ibl.Convolver)",
        "face_order": list(FACE_NAMES),
        "boost": {"knee": BOOST_KNEE, "strength": BOOST_STRENGTH, "power": BOOST_POWER},
        "sweep_count": len(sweeps),
        "sweeps": sweeps,
    }
    path = out_dir / "lighting_seeds.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 8 (G5): IBL pre-computation (diffuse irradiance + GGX chain)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_bake = sub.add_parser("bake", help="extracted cubemaps -> IBL products + seeds")
    p_bake.add_argument("--panos", type=Path, required=True, help="extracted_cubemaps dir")
    p_bake.add_argument("--output", "-o", type=Path, default=Path("ibl_out"))
    p_bake.add_argument(
        "--sweeps", type=str, default=None, help="index selection, e.g. '1,50,60,175' or '0-2'"
    )
    p_bake.add_argument("--diffuse-size", type=int, default=64)
    p_bake.add_argument("--irr-source-size", type=int, default=32)
    p_bake.add_argument("--spec-size", type=int, default=256)
    p_bake.add_argument("--spec-mips", type=int, default=5)
    p_bake.add_argument("--force", action="store_true", help="rebake even if up to date")
    p_bake.add_argument("--no-exr", action="store_true", help="write .npy only")
    p_bake.add_argument("--no-review", action="store_true", help="skip review composites")
    args = parser.parse_args(argv)

    if not (args.panos / "manifest.json").is_file():
        parser.error(f"manifest.json not found in panos dir: {args.panos}")
    params = IblParams(
        diffuse_size=args.diffuse_size,
        irr_source_size=args.irr_source_size,
        spec_size=args.spec_size,
        spec_mips=args.spec_mips,
    )
    selection = parse_sweep_selection(args.sweeps) if args.sweeps else None
    bake(
        args.panos,
        args.output,
        params,
        sweep_indices=selection,
        force=args.force,
        write_exr=not args.no_exr,
        write_review=not args.no_review,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
