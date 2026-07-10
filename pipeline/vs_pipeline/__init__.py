"""Virtual Staging offline processing pipeline.

Per-property DAG of idempotent, input-hash-versioned, independently re-runnable
stages (see PLAN.md, section 3).

Coordinate Lock Rule (hard invariant)
-------------------------------------
The E57 origin (0, 0, 0) is preserved unchanged through every stage. All poses,
planes and placements are expressed in this single frame. One fixed conversion
is applied exactly once, at ingestion (Matterport Z-up -> runtime Y-up):

    X_rt = X_mp,   Y_rt = Z_mp,   Z_rt = -Y_mp
"""

__version__ = "0.1.0"
