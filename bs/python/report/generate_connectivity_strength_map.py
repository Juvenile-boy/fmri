#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def read_matrix(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mat = np.load(path)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        mat = pd.read_csv(path, sep=sep, header=None).to_numpy()
    else:
        raise ValueError(f"Unsupported matrix file: {path}")
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got shape={mat.shape}")
    return mat


def infer_roi_ids(labels_path: Path | None, n_roi: int) -> np.ndarray:
    if labels_path is None:
        return np.arange(1, n_roi + 1, dtype=int)

    suffix = labels_path.suffix.lower()
    sep = "\t" if suffix == ".tsv" else ","
    df = pd.read_csv(labels_path, sep=sep)
    cols = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]

    idx_col = None
    for i, c in enumerate(cols):
        if c in {"index", "id"} or c.startswith("index") or c.startswith("roi"):
            idx_col = df.columns[i]
            break

    if idx_col is None:
        if len(df) != n_roi:
            raise ValueError(
                f"ROI labels has no index column and row count ({len(df)}) != n_roi ({n_roi})"
            )
        return np.arange(1, n_roi + 1, dtype=int)

    idx = pd.to_numeric(df[idx_col], errors="coerce").to_numpy()
    idx = idx[np.isfinite(idx)].astype(int)
    if len(idx) < n_roi:
        raise ValueError(
            f"Index column valid rows ({len(idx)}) < n_roi ({n_roi}). "
            "Please provide complete index column."
        )
    return idx[:n_roi]


def compute_strength(mat: np.ndarray, metric: str) -> np.ndarray:
    x = mat.copy()
    np.fill_diagonal(x, 0.0)
    if metric == "abs_sum":
        s = np.sum(np.abs(x), axis=1)
    elif metric == "positive_sum":
        s = np.sum(np.clip(x, 0.0, None), axis=1)
    elif metric == "degree":
        s = np.sum(np.abs(x) >= 0.3, axis=1).astype(float)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return s


def write_strength_map(
    atlas_path: Path,
    roi_ids: np.ndarray,
    strength: np.ndarray,
    out_path: Path,
) -> None:
    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32)

    out_data = np.zeros(atlas_data.shape, dtype=np.float32)
    for rid, sval in zip(roi_ids, strength):
        out_data[atlas_data == int(rid)] = float(sval)

    out_img = nib.Nifti1Image(out_data, atlas_img.affine, atlas_img.header)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build node-strength brain map from connectivity matrix and atlas."
    )
    ap.add_argument("--conn", required=True, help="Connectivity matrix path (.csv/.tsv/.npy)")
    ap.add_argument("--atlas", required=True, help="Atlas NIfTI path with integer ROI IDs")
    ap.add_argument(
        "--labels",
        default="",
        help="Optional ROI labels CSV/TSV, supports index column (index/id/roi*)",
    )
    ap.add_argument("--out", required=True, help="Output NIfTI path, e.g., conn_strength_map.nii")
    ap.add_argument(
        "--metric",
        choices=["abs_sum", "positive_sum", "degree"],
        default="abs_sum",
        help="Node strength metric",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Optional output CSV of roi_index + strength (ranked descending)",
    )
    args = ap.parse_args()

    conn_path = Path(args.conn)
    atlas_path = Path(args.atlas)
    labels_path = Path(args.labels) if args.labels else None
    out_path = Path(args.out)
    out_csv_path = Path(args.out_csv) if args.out_csv else None

    mat = read_matrix(conn_path)
    roi_ids = infer_roi_ids(labels_path, mat.shape[0])
    strength = compute_strength(mat, args.metric)
    write_strength_map(atlas_path, roi_ids, strength, out_path)

    if out_csv_path is not None:
        df = pd.DataFrame({"index": roi_ids, "strength": strength})
        df = df.sort_values("strength", ascending=False).reset_index(drop=True)
        df.to_csv(out_csv_path, index=False, encoding="utf-8")

    print(f"[OK] wrote NIfTI: {out_path}")
    print(
        f"[INFO] metric={args.metric}, n_roi={len(roi_ids)}, "
        f"strength_range=({float(np.min(strength)):.6f}, {float(np.max(strength)):.6f})"
    )
    if out_csv_path is not None:
        print(f"[OK] wrote CSV: {out_csv_path}")


if __name__ == "__main__":
    main()
