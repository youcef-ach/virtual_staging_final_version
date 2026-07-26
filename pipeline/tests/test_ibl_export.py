"""G7.2 .env exporter tests: RGBD, solid angles, polynomial chain, container."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vs_pipeline.stages import ibl_export as ie


def test_rgbd_round_trip() -> None:
    rng = np.random.default_rng(7)
    lin = (rng.random((8, 8, 3), dtype=np.float32) * 32.0).astype(np.float32)
    lin[0, 0] = (0.02, 0.5, 1.0)
    lin[0, 1] = (3.0, 10.0, 120.0)
    dec = ie.decode_rgbd(ie.encode_rgbd(lin))
    assert np.allclose(dec, lin, rtol=0.15, atol=0.02)
    # LDR values (D = 1) survive with only gamma-quantization error.
    ldr = np.clip(lin, 0.0, 1.0)
    dec_ldr = ie.decode_rgbd(ie.encode_rgbd(ldr))
    assert np.allclose(dec_ldr, ldr, rtol=0.05, atol=6e-3)


def test_exact_solid_angles_sum_to_4pi() -> None:
    for size in (4, 8, 32):
        total = 6.0 * float(ie.exact_solid_angles(size).sum())
        assert math.isclose(total, 4.0 * math.pi, rel_tol=1e-9)


def test_constant_env_polynomial_identity() -> None:
    # Babylon chain on a constant environment L must evaluate back to L:
    # xx = yy = zz = L, all other terms 0 (the analytic identity).
    env = np.full((6, 8, 8, 3), 0.5, dtype=np.float32)
    poly = ie.polynomial_from_env(env)
    for key in ("xx", "yy", "zz"):
        assert np.allclose(poly[key], 0.5, atol=5e-3), key
    for key in ("x", "y", "z", "xy", "yz", "zx"):
        assert np.allclose(poly[key], 0.0, atol=1e-4), key
    for n in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)):
        assert np.allclose(ie.evaluate_polynomial(poly, n), 0.5, atol=6e-3)


def test_directional_polynomial() -> None:
    env = np.zeros((6, 8, 8, 3), dtype=np.float32)
    env[0] = 2.0  # +X face bright (face order px,nx,py,ny,pz,nz)
    poly = ie.polynomial_from_env(env)
    e_plus = float(ie.evaluate_polynomial(poly, (1.0, 0.0, 0.0)).mean())
    e_minus = float(ie.evaluate_polynomial(poly, (-1.0, 0.0, 0.0)).mean())
    assert e_plus > 0.0
    assert e_plus > e_minus + 0.05


def test_mip_tail_extension() -> None:
    chain = ie.extend_mip_chain([np.ones((6, 16, 16, 3), dtype=np.float32)])
    assert [m.shape[1] for m in chain] == [16, 8, 4, 2, 1]
    assert all(np.allclose(m, 1.0, atol=1e-6) for m in chain)


def test_env_container_layout() -> None:
    rng = np.random.default_rng(3)
    base = (rng.random((6, 16, 16, 3), dtype=np.float32) * 2.0).astype(np.float32)
    chain = ie.extend_mip_chain([base])
    poly = ie.polynomial_from_env(np.full((6, 8, 8, 3), 0.25, dtype=np.float32))
    data = ie.build_env_bytes(chain, poly)

    manifest, payload = ie.read_env(data)
    assert manifest["version"] == 2
    assert manifest["width"] == 16
    assert manifest["imageType"] == "image/png"
    assert manifest["specular"]["lodGenerationScale"] == 0.8
    for key in ("x", "y", "z", "xx", "yy", "zz", "yz", "zx", "xy"):
        assert len(manifest["irradiance"][key]) == 3

    mipmaps = manifest["specular"]["mipmaps"]
    assert len(mipmaps) == 6 * (int(math.log2(16)) + 1)  # 30 slices
    # Positions contiguous and payload-relative.
    pos = 0
    for entry in mipmaps:
        assert entry["position"] == pos
        pos += entry["length"]
    assert pos == len(payload)

    def decode(entry: dict) -> np.ndarray:
        raw = np.frombuffer(
            payload[entry["position"] : entry["position"] + entry["length"]], np.uint8
        )
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        assert img is not None
        return img

    first = decode(mipmaps[0])
    assert first.shape == (16, 16, 4)
    last = decode(mipmaps[-1])
    assert last.shape == (1, 1, 4)
    # Slice 0 decodes back to the base +X face (BGRA -> RGBA -> RGBD decode).
    lin = ie.decode_rgbd(first[..., [2, 1, 0, 3]])
    assert np.allclose(lin, chain[0][0], rtol=0.15, atol=0.02)
