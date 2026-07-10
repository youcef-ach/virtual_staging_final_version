"""Pipeline stages (one module per DAG stage).

Planned modules (implemented per PLAN.md phases):

- ``ingest``        : Stage 1  - ingest & validate            (F3)
- ``pose``          : Stage 2  - pose extraction & QA, Gate 1 (F2/F5)
- ``preprocess``    : Stage 3  - point cloud preprocessing    (G1)
- ``segmentation``  : Stage 4  - semantic segmentation, Gate 2 (G6)
- ``planes``        : Stage 5  - plane fitting & boundaries   (G2)
- ``proxy``         : Stage 6  - proxy mesh synthesis         (G3)
- ``gbuffer``       : Stage 7  - G-buffer baking              (G4)
- ``ibl``           : Stage 8  - IBL pre-computation          (G5)
- ``navgraph``      : Stage 9  - navigation graph             (T1)
- ``publish``       : Stage 10 - compress & publish           (P4)
"""
