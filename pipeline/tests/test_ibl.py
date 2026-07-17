"""Tests for Stage 8 (G5): IBL pre-computation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vs_pipeline.stages.ibl import (
    FACE_NAMES,
    IblParams,
    NumpyConvolver,
    bake,
    face_directions,
    face_solid_angles,
    highlight_boost,
    linear_to_srgb,
    parse_sweep_selection,
    srgb_to_linear,
)


def test_solid_angles_cover_the_sphere():
    total = 6.0 * float(face_solid_angles(16).sum())
    assert abs(total - 4.0 * np.pi) / (4.0 * np.pi) < 0.01


def test_face_directions_unit_and_axis_aligned():
    for face, axis, sign in (("px", 0, 1), ("ny", 1, -1), ("nz", 2, -1)):
        dirs = face_directions(face, 8)
        assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-6)
        mean = dirs.reshape(-1, 3).mean(axis=0)
        assert sign * mean[axis] > 0.9 * abs(mean).max()


def test_srgb_linear_roundtrip():
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-4)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-4)


def test_boost_identity_below_knee_and_monotonic_above():
    dim = np.full((4, 4, 3), 0.5, dtype=np.float32)
    boosted, factor = highlight_boost(dim)
    assert np.allclose(boosted, dim)
    assert np.allclose(factor, 1.0)
    bright_a = np.full((1, 1, 3), 0.90, dtype=np.float32)
    bright_b = np.full((1, 1, 3), 0.97, dtype=np.float32)
    _, fa = highlight_boost(bright_a)
    _, fb = highlight_boost(bright_b)
    assert float(fa[0, 0]) > 1.0
    assert float(fb[0, 0]) > float(fa[0, 0])


def _constant_env(value: float, size: int) -> dict[str, np.ndarray]:
    return {f: np.full((size, size, 3), value, dtype=np.float32) for f in FACE_NAMES}


def test_constant_env_irradiance_is_identity():
    params = IblParams(diffuse_size=8, irr_source_size=16, spec_size=16, spec_mips=2)
    irr = NumpyConvolver(params).irradiance(_constant_env(0.5, 16))
    assert irr.shape == (6, 8, 8, 3)
    assert np.allclose(irr, 0.5, rtol=0.02)


def test_constant_env_specular_chain_preserves_value():
    params = IblParams(diffuse_size=8, irr_source_size=8, spec_size=16, spec_mips=3)
    chain = NumpyConvolver(params).specular_chain(_constant_env(0.25, 16))
    assert len(chain) == 3
    assert [c.shape[1] for c in chain] == [16, 8, 4]
    for mip in chain:
        assert mip.shape[0] == 6 and mip.shape[-1] == 3
        assert np.allclose(mip, 0.25, atol=1e-3)


def test_parse_sweep_selection():
    assert parse_sweep_selection("1,50,60,175") == {1, 50, 60, 175}
    assert parse_sweep_selection("0-2,50") == {0, 1, 2, 50}


def _write_synth_panos(root: Path) -> Path:
    panos = root / "panos"
    uuid = "sweep-test"
    (panos / uuid).mkdir(parents=True)
    faces = {}
    for i, face in enumerate(FACE_NAMES):
        img = np.full((32, 32, 3), 90 + 10 * i, dtype=np.uint8)
        img[4:12, 4:12] = 255  # bright patch -> boost must activate
        assert cv2.imwrite(str(panos / uuid / f"{face}.jpg"), img)
        faces[face] = f"{uuid}/{face}.jpg"
    manifest = {
        "schema_version": 1,
        "sweep_count": 1,
        "sweeps": [{"sweep_uuid": uuid, "index": 0, "faces": faces}],
    }
    (panos / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return panos


def test_bake_end_to_end_idempotent_and_finite(tmp_path):
    panos = _write_synth_panos(tmp_path)
    out = tmp_path / "ibl"
    params = IblParams(diffuse_size=8, irr_source_size=8, spec_size=16, spec_mips=3)

    summary = bake(panos, out, params, write_exr=False, write_review=True)
    assert summary["baked"] == ["sweep-test"] and not summary["failed"]

    irr = np.load(out / "sweep-test" / "irradiance.npy")
    assert irr.shape == (6, 8, 8, 3) and np.isfinite(irr).all()
    for m, size in enumerate((16, 8, 4)):
        mip = np.load(out / "sweep-test" / f"specular_mip{m}.npy")
        assert mip.shape == (6, size, size, 3) and np.isfinite(mip).all()

    seeds = json.loads((out / "lighting_seeds.json").read_text(encoding="utf-8"))
    assert seeds["sweep_count"] == 1
    assert seeds["sweeps"][0]["stats"]["boost_pixel_fraction"] > 0.0
    assert list((out / "review").glob("ibl_000_*.png"))

    # Second run: param-matched outputs on disk -> skipped, not rebaked.
    summary2 = bake(panos, out, params, write_exr=False, write_review=False)
    assert summary2["skipped"] == ["sweep-test"] and not summary2["baked"]

    # Forced run rebakes.
    summary3 = bake(panos, out, params, force=True, write_exr=False, write_review=False)
    assert summary3["baked"] == ["sweep-test"]


# ----------------------------- G6 slice 4.3: segmentation-guided highlight boost


def test_guided_boost_confined_to_mask_capped_and_kneed():
    from vs_pipeline.stages.ibl import GUIDED_MAX_FACTOR, guided_highlight_boost

    saturated = np.full((4, 4, 3), 1.0, dtype=np.float32)
    weight = np.zeros((4, 4), dtype=np.float32)
    weight[:2, :] = 1.0  # emissive top half
    _, factor = guided_highlight_boost(saturated, weight)
    assert float(factor[:2].min()) > 1.5  # saturated emissive pixels boosted hard
    assert np.allclose(factor[2:], 1.0)  # outside the mask: identity by construction
    assert float(factor.max()) <= GUIDED_MAX_FACTOR + 1e-6

    # Dark emissive pixels (window at dusk, lamp off) stay identity via the knee.
    dark = np.full((4, 4, 3), 0.3, dtype=np.float32)
    _, f_dark = guided_highlight_boost(dark, np.ones((4, 4), dtype=np.float32))
    assert np.allclose(f_dark, 1.0)


def test_load_emissive_mask_union_and_window_id_contract(tmp_path):
    from vs_pipeline.stages.ibl import (
        EMISSIVE_ADE_IDS,
        SEG_WINDOW_CLASS_ID,
        load_emissive_mask,
    )
    from vs_pipeline.stages.segmentation import CLASS_IDS

    assert SEG_WINDOW_CLASS_ID == CLASS_IDS["window"]

    sweep = tmp_path / "seg" / "u1"
    sweep.mkdir(parents=True)
    cls = np.zeros((8, 8), dtype=np.uint8)
    cls[0, 0] = SEG_WINDOW_CLASS_ID
    ade = np.zeros((8, 8), dtype=np.uint8)
    ade[7, 7] = EMISSIVE_ADE_IDS[0]  # sky
    assert cv2.imwrite(str(sweep / "px_cls.png"), cls)
    assert cv2.imwrite(str(sweep / "px_ade.png"), ade)

    mask = load_emissive_mask(tmp_path / "seg", "u1", "px")
    assert mask is not None
    assert mask[0, 0]  # window from the 7-class map
    assert mask[7, 7]  # sky from the RAW ADE map, unioned in
    assert not mask[3, 3]
    assert load_emissive_mask(tmp_path / "seg", "u1", "py") is None


def test_bake_seg_guided_mode_and_identity_bump(tmp_path):
    from vs_pipeline.stages.ibl import SEG_WINDOW_CLASS_ID

    panos = _write_synth_panos(tmp_path)
    out = tmp_path / "ibl"
    params = IblParams(diffuse_size=8, irr_source_size=8, spec_size=16, spec_mips=3)

    # Blind bake first (no seg masks anywhere).
    s1 = bake(panos, out, params, write_exr=False, write_review=False)
    assert s1["baked"] == ["sweep-test"]
    meta = json.loads((out / "sweep-test" / "ibl_meta.json").read_text(encoding="utf-8"))
    assert meta["boost"]["mode"] == "blind"

    # Seg masks appear (window over the bright patch) -> identity bump: the
    # SAME sweep must re-bake in seg_guided mode without --force.
    seg = tmp_path / "seg" / "sweep-test"
    seg.mkdir(parents=True)
    for face in FACE_NAMES:
        cls = np.zeros((32, 32), dtype=np.uint8)
        cls[4:12, 4:12] = SEG_WINDOW_CLASS_ID
        assert cv2.imwrite(str(seg / f"{face}_cls.png"), cls)
    s2 = bake(
        panos, out, params, seg_dir=tmp_path / "seg", write_exr=False, write_review=False
    )
    assert s2["baked"] == ["sweep-test"]
    meta = json.loads((out / "sweep-test" / "ibl_meta.json").read_text(encoding="utf-8"))
    assert meta["boost"]["mode"] == "seg_guided"
    assert meta["stats"]["boost_pixel_fraction"] > 0.0  # bright patch inside the mask

    # Same config again -> idempotent skip.
    s3 = bake(
        panos, out, params, seg_dir=tmp_path / "seg", write_exr=False, write_review=False
    )
    assert s3["skipped"] == ["sweep-test"] and not s3["baked"]
