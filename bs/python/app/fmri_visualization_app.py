from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import resample_to_img, smooth_img


st.set_page_config(
    page_title="fMRI 可视化系统",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
  header[data-testid="stHeader"] {
    height: 2.2rem;
    min-height: 2.2rem;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(4px);
  }
  div[data-testid="stToolbar"] {
    top: 0.15rem;
    right: 0.35rem;
    transform: scale(0.9);
    transform-origin: top right;
  }
  /* Sidebar expand control: support different Streamlit DOM versions */
  div[data-testid="stSidebarCollapsedControl"] {
    position: fixed !important;
    top: 0.35rem !important;
    left: 0.35rem !important;
    right: auto !important;
    z-index: 1002 !important;
  }
  div[data-testid="stSidebarCollapsedControl"] > button,
  button[data-testid="collapsedControl"],
  button[aria-label="Open sidebar"] {
    position: fixed !important;
    top: 0.35rem !important;
    left: 0.35rem !important;
    right: auto !important;
    z-index: 1003 !important;
  }
  #MainMenu { visibility: visible; }
  .stApp { background: linear-gradient(180deg, #f7fbff 0%, #eef5f9 60%, #fefefe 100%); }
  .block-container { padding-top: 1.1rem; }
  .app-title {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: 0.2px;
    color: #0d3b66;
    margin-bottom: .2rem;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  }
  .app-subtitle { color: #3c4a57; margin-bottom: 1rem; }
  .metric-card {
    border: 1px solid #d8e2ea;
    border-radius: 10px;
    padding: 10px 12px;
    background: #ffffff;
  }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown('<div class="app-title">fMRI 数据可视化与展示系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">基于 SPM / DPABI 的分析结果展示：统计图像、激活图谱、功能连接矩阵</div>',
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    return json.loads(text)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_matlab_script(script_path: Path, matlab_exe: str, timeout_minutes: int = 0) -> dict:
    if not script_path.exists():
        return {
            "ok": False,
            "returncode": -1,
            "command": "",
            "log": f"MATLAB script not found: {script_path}",
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "end_time": datetime.now().isoformat(timespec="seconds"),
        }

    script_arg = script_path.as_posix()
    matlab_cmd = f"run('{script_arg}');exit;"
    cmd = [matlab_exe, "-batch", matlab_cmd]
    start_ts = datetime.now().isoformat(timespec="seconds")

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(get_repo_root()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=None if timeout_minutes <= 0 else int(timeout_minutes * 60),
            check=False,
        )
        out = (completed.stdout or "").strip()
        err = (completed.stderr or "").strip()
        merged = out
        if err:
            merged = (merged + "\n\n[stderr]\n" + err).strip()
        end_ts = datetime.now().isoformat(timespec="seconds")
        return {
            "ok": completed.returncode == 0,
            "returncode": int(completed.returncode),
            "command": " ".join(cmd),
            "log": merged,
            "start_time": start_ts,
            "end_time": end_ts,
        }
    except FileNotFoundError:
        end_ts = datetime.now().isoformat(timespec="seconds")
        return {
            "ok": False,
            "returncode": -1,
            "command": " ".join(cmd),
            "log": f"MATLAB executable not found: {matlab_exe}",
            "start_time": start_ts,
            "end_time": end_ts,
        }
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        merged = out
        if err:
            merged = (merged + "\n\n[stderr]\n" + err).strip()
        end_ts = datetime.now().isoformat(timespec="seconds")
        return {
            "ok": False,
            "returncode": -9,
            "command": " ".join(cmd),
            "log": f"Timeout after {timeout_minutes} minute(s).\n\n{merged}".strip(),
            "start_time": start_ts,
            "end_time": end_ts,
        }


def list_subject_dirs(derivatives_root: Path) -> list[Path]:
    if not derivatives_root.exists():
        return []
    dirs = [p for p in derivatives_root.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    return sorted(dirs)


def validate_derivatives_root(derivatives_root: Path) -> tuple[bool, str]:
    if not derivatives_root.exists():
        return False, f"结果根目录无效或不存在，不能识别被试: {derivatives_root}"
    if not derivatives_root.is_dir():
        return False, f"结果根目录不是文件夹，不能识别被试: {derivatives_root}"
    return True, ""


def find_candidates(base: Path, patterns: Iterable[str]) -> list[Path]:
    if not base.exists():
        return []
    found: list[Path] = []
    for pat in patterns:
        found.extend(base.glob(pat))
    dedup = sorted({p.resolve() for p in found if p.is_file()})
    return dedup


@st.cache_data(show_spinner=False)
def load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else None
        df = pd.read_csv(p, header=None, sep=sep, engine="python")
        arr = df.values
    else:
        raise ValueError(f"不支持的矩阵格式: {suffix}")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"矩阵必须为方阵，当前 shape={arr.shape}")
    return arr.astype(float)


@st.cache_data(show_spinner=False)
def load_region_labels(path: str, n: int) -> list[str]:
    _, labels = load_region_index_label_pairs(path, n)
    return labels


def load_matrix_checked(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"连接矩阵文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else None
        arr = pd.read_csv(p, header=None, sep=sep, engine="python").values
    else:
        raise ValueError(f"不支持的矩阵格式: {suffix}")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"连接矩阵必须是方阵，当前 shape={arr.shape}")
    return arr.astype(float)


def describe_region_label_file(path: str, n: int) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return f"ROI 标签文件不存在，已使用默认标签: {p}"
    if p.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        return f"ROI 标签文件格式不支持，已使用默认标签: {p.suffix}"
    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(p, sep=sep)
    except Exception as exc:
        return f"ROI 标签文件读取失败，已使用默认标签: {exc}"
    if len(df) < n:
        return f"ROI 标签数量不足（{len(df)} < {n}），已使用默认标签补足或回退。"
    normalized = {str(c).strip().lower().replace("\ufeff", "") for c in df.columns}
    if not any(c in {"label", "name", "region"} or c.startswith("label") for c in normalized):
        return "ROI 标签文件缺少 label/name/region 列，已使用默认标签。"
    return ""


@st.cache_data(show_spinner=False)
def load_region_index_label_pairs(path: str, n: int) -> tuple[np.ndarray, list[str]]:
    roi_ids = np.arange(1, n + 1, dtype=int)
    labels = [f"ROI_{i+1}" for i in range(n)]
    if not path:
        return roi_ids, labels

    p = Path(path)
    if not p.exists() or p.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        return roi_ids, labels

    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(p, sep=sep)
    except Exception:
        return roi_ids, labels

    normalized = {str(c).strip().lower().replace("\ufeff", ""): c for c in df.columns}
    idx_col = next(
        (orig for key, orig in normalized.items() if key in {"index", "id"} or key.startswith("index") or key.startswith("roi")),
        None,
    )
    label_col = next(
        (orig for key, orig in normalized.items() if key in {"label", "name", "region"} or key.startswith("label")),
        None,
    )

    if idx_col is not None:
        idx = pd.to_numeric(df[idx_col], errors="coerce").to_numpy()
        idx = idx[np.isfinite(idx)].astype(int)
        if len(idx) >= n:
            roi_ids = idx[:n]

    if label_col is not None:
        raw_labels = [str(x) for x in df[label_col].tolist()]
        if len(raw_labels) >= n:
            labels = [
                normalize_region_label(label, int(idx))
                for label, idx in zip(raw_labels[:n], roi_ids[:n])
            ]
    else:
        labels = [
            normalize_region_label(label, int(idx))
            for label, idx in zip(labels[:n], roi_ids[:n])
        ]

    return roi_ids, labels


@st.cache_data(show_spinner=False)
def load_aal_translation_map() -> dict[str, str]:
    cfg_dir = get_repo_root() / "configs"
    en_path = cfg_dir / "atlas_labels_aal_en_backup.csv"
    zh_path = cfg_dir / "atlas_labels_aal.csv"
    if not en_path.exists() or not zh_path.exists():
        return {}
    try:
        en_df = pd.read_csv(en_path)
        zh_df = pd.read_csv(zh_path)
    except Exception:
        return {}
    if "index" not in en_df.columns or "label" not in en_df.columns:
        return {}
    if "index" not in zh_df.columns or "label" not in zh_df.columns:
        return {}

    merged = en_df[["index", "label"]].merge(
        zh_df[["index", "label"]],
        on="index",
        suffixes=("_en", "_zh"),
    )
    return {
        str(row["label_en"]).strip(): str(row["label_zh"]).strip()
        for _, row in merged.iterrows()
        if str(row["label_zh"]).strip()
    }


@st.cache_data(show_spinner=False)
def load_default_index_label_map() -> dict[int, str]:
    zh_path = get_repo_root() / "configs" / "atlas_labels_aal.csv"
    if not zh_path.exists():
        return {}
    try:
        df = pd.read_csv(zh_path)
    except Exception:
        return {}
    if "index" not in df.columns or "label" not in df.columns:
        return {}
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            idx = int(row["index"])
            label = str(row["label"]).strip()
        except Exception:
            continue
        if label:
            out[idx] = label
    return out


def normalize_region_label(label: str, roi_index: int) -> str:
    text = str(label).strip()
    generic_names = {
        "",
        f"ROI_{roi_index}",
        f"roi_{roi_index}",
        f"Region_{roi_index}",
        f"region_{roi_index}",
    }
    if text in generic_names:
        return load_default_index_label_map().get(int(roi_index), f"ROI_{roi_index}")
    return translate_region_label(text)


def translate_region_label(label: str) -> str:
    text = str(label).strip()
    if not text or any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text
    return load_aal_translation_map().get(text, text)


def load_roi_indices(path: str, n: int) -> np.ndarray:
    if not path:
        return np.arange(1, n + 1, dtype=int)
    p = Path(path)
    if not p.exists():
        return np.arange(1, n + 1, dtype=int)
    if p.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        return np.arange(1, n + 1, dtype=int)

    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(p, sep=sep)
    except Exception:
        return np.arange(1, n + 1, dtype=int)

    idx_col = None
    for col in df.columns:
        c = str(col).strip().lower().replace("\ufeff", "")
        if c in {"index", "id"} or c.startswith("index") or c.startswith("roi"):
            idx_col = col
            break

    if idx_col is None:
        return np.arange(1, n + 1, dtype=int)

    idx = pd.to_numeric(df[idx_col], errors="coerce").to_numpy()
    idx = idx[np.isfinite(idx)].astype(int)
    if len(idx) < n:
        return np.arange(1, n + 1, dtype=int)
    return idx[:n]


@st.cache_data(show_spinner=False)
def load_nifti(path: str):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    return data, img.affine


def estimate_focus_world_coord(stat_img: nib.Nifti1Image, threshold: float) -> np.ndarray:
    data = np.asanyarray(stat_img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.nan_to_num(data.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    abs_data = np.abs(data)

    mask = abs_data >= float(threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        voxel = np.array(abs_data.shape[:3]) // 2
    else:
        values = abs_data[mask].ravel()
        k = min(800, values.size)
        if values.size > k:
            idx = np.argpartition(values, -k)[-k:]
            coords = coords[idx]
        voxel = np.round(np.median(coords, axis=0)).astype(int)

    world = nib.affines.apply_affine(stat_img.affine, voxel)
    return np.asarray(world, dtype=float)


def resolve_cut_coords(display_mode: str, cut_n: int, stat_img: nib.Nifti1Image | None, threshold: float):
    # nilearn requires int cut_coords only for single-axis modes.
    if display_mode in {"x", "y", "z"}:
        return int(cut_n)

    if stat_img is None:
        return None

    focus = estimate_focus_world_coord(stat_img, threshold)
    if display_mode == "ortho":
        return (float(focus[0]), float(focus[1]), float(focus[2]))

    axis_index = {"x": 0, "y": 1, "z": 2}
    coords = tuple(float(focus[axis_index[a]]) for a in display_mode if a in axis_index)
    if len(coords) == 0:
        return None
    return coords


def suggest_files(subject_dir: Path) -> dict[str, list[Path]]:
    return {
        "stat_maps": find_candidates(
            subject_dir,
            [
                "**/spmT_*.nii",
                "**/con_*.nii",
                "**/zmap*.nii",
                "**/tmap*.nii",
            ],
        ),
        "bg_maps": find_candidates(
            subject_dir,
            [
                "**/mean*.nii",
                "**/swra*.nii",
                "**/wra*.nii",
                "**/swar*.nii",
            ],
        ),
        "conn_mats": find_candidates(
            subject_dir,
            [
                "**/*connect*.csv",
                "**/*connect*.tsv",
                "**/*fc*.csv",
                "**/*corr*.csv",
                "**/*.npy",
            ],
        ),
    }

def parse_multi_paths(manual_text: str) -> list[str]:
    if not manual_text:
        return []
    parts = [x.strip() for x in manual_text.replace("\n", ";").split(";")]
    return [x for x in parts if x]


def build_overlap_stat_image(stat_paths: list[str], voxel_threshold: float, min_ratio: float):
    imgs = []
    datas = []
    for p in stat_paths:
        img = nib.load(p)
        data = np.asanyarray(img.dataobj)
        if data.ndim == 4:
            data = data[..., 0]
        imgs.append(img)
        datas.append(np.nan_to_num(data.astype(float), nan=0.0, posinf=0.0, neginf=0.0))

    ref = imgs[0]
    ref_shape = datas[0].shape
    for i in range(1, len(datas)):
        if datas[i].shape != ref_shape:
            raise ValueError(f"图像维度不一致，无法直接做交集: {Path(stat_paths[i]).name}")

    masks = [(np.abs(d) >= float(voxel_threshold)).astype(float) for d in datas]
    ratio = np.mean(np.stack(masks, axis=0), axis=0)
    kept = (ratio >= float(min_ratio)).astype(float) * ratio
    out_img = nib.Nifti1Image(kept, ref.affine, ref.header)
    return out_img, int(np.sum(kept > 0))


def mni_table_from_top_voxels(stat_data: np.ndarray, affine: np.ndarray, threshold: float, top_k: int = 20) -> pd.DataFrame:
    mask = np.abs(stat_data) >= threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pd.DataFrame(columns=["x", "y", "z", "value_abs"])
    values = np.abs(stat_data[mask]).ravel()
    k = min(top_k, values.size)
    top_idx = np.argpartition(values, -k)[-k:]
    top_coords = coords[top_idx]
    top_vals = values[top_idx]
    mni_xyz = nib.affines.apply_affine(affine, top_coords)
    df = pd.DataFrame(mni_xyz, columns=["x", "y", "z"])
    df["value_abs"] = top_vals
    return df.sort_values("value_abs", ascending=False).reset_index(drop=True)


def _atlas_label_map(labels_csv: str) -> dict[int, str]:
    if not labels_csv or not Path(labels_csv).exists():
        return {}
    p = Path(labels_csv)
    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(p, sep=sep)
    except Exception:
        return {}

    cols = {c.lower(): c for c in df.columns}
    idx_col = cols.get("index") or cols.get("id") or cols.get("roi")
    lbl_col = cols.get("label") or cols.get("name") or cols.get("region")
    if not idx_col or not lbl_col:
        return {}

    out = {}
    for _, row in df.iterrows():
        try:
            k = int(row[idx_col])
            v = str(row[lbl_col])
            if v:
                out[k] = v
        except Exception:
            continue
    return out


def activation_voxel_table(
    stat_data: np.ndarray,
    affine: np.ndarray,
    threshold: float,
    stat_img: nib.Nifti1Image | None = None,
    atlas_path: str = "",
    labels_csv: str = "",
) -> pd.DataFrame:
    mask = np.abs(stat_data) >= float(threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pd.DataFrame(columns=["x", "y", "z", "value", "value_abs", "region_index", "region_name"])
    values = stat_data[mask].ravel().astype(float)
    mni_xyz = nib.affines.apply_affine(affine, coords)
    df = pd.DataFrame(mni_xyz, columns=["x", "y", "z"])
    df["value"] = values
    df["value_abs"] = np.abs(values)
    df["region_index"] = 0
    df["region_name"] = "Unknown"

    if atlas_path and stat_img is not None and Path(atlas_path).exists():
        try:
            atlas_img = nib.load(atlas_path)
            atlas_resampled = resample_to_img(atlas_img, stat_img, interpolation="nearest", force_resample=True)
            atlas_data = np.asanyarray(atlas_resampled.dataobj).astype(int)
            region_idx = atlas_data[mask].ravel()
            region_map = _atlas_label_map(labels_csv)
            df["region_index"] = region_idx
            df["region_name"] = [
                region_map.get(int(i), f"Region_{int(i)}") if int(i) > 0 else "Unknown"
                for i in region_idx
            ]
        except Exception:
            pass

    return df.sort_values("value_abs", ascending=False).reset_index(drop=True)


def build_activation_3d_figure(df_xyz: pd.DataFrame, point_size: int, opacity: float):
    hover_data = {"x": ":.1f", "y": ":.1f", "z": ":.1f", "value": ":.4f", "value_abs": ":.4f"}
    if "region_name" in df_xyz.columns:
        hover_data["region_name"] = True
    if "region_index" in df_xyz.columns:
        hover_data["region_index"] = True
    fig = px.scatter_3d(
        df_xyz,
        x="x",
        y="y",
        z="z",
        color="value",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0.0,
        hover_data=hover_data,
    )
    fig.update_traces(marker=dict(size=int(point_size), opacity=float(opacity)))
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            dragmode="turntable",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="3D 激活体素图（可旋转）",
    )
    return fig


def build_surface_activation_3d_figure(
    stat_img: nib.Nifti1Image,
    threshold: float,
    max_points: int,
    point_size: int,
    point_opacity: float,
    bg_img: nib.Nifti1Image | None = None,
    point_df: pd.DataFrame | None = None,
):
    stat_data = np.asanyarray(stat_img.dataobj)
    if stat_data.ndim == 4:
        stat_data = stat_data[..., 0]
    stat_data = np.nan_to_num(stat_data.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    if bg_img is None:
        bg_data = np.abs(stat_data)
        bg_affine = stat_img.affine
    else:
        try:
            if (
                bg_img.shape[:3] != stat_img.shape[:3]
                or np.max(np.abs(bg_img.affine - stat_img.affine)) > 1e-3
            ):
                bg_img = resample_to_img(bg_img, stat_img, interpolation="continuous", force_resample=True)
        except Exception:
            pass
        bg_data = np.asanyarray(bg_img.dataobj)
        if bg_data.ndim == 4:
            bg_data = bg_data[..., 0]
        bg_data = np.nan_to_num(bg_data.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        bg_affine = bg_img.affine

    # Downsample background for speed.
    step = 2 if max(bg_data.shape) > 96 else 1
    bg_ds = bg_data[::step, ::step, ::step]
    ix, iy, iz = np.indices(bg_ds.shape)
    vox_bg = np.column_stack((ix.ravel() * step, iy.ravel() * step, iz.ravel() * step))
    xyz_bg = nib.affines.apply_affine(bg_affine, vox_bg)
    val_bg = bg_ds.ravel()

    pos = val_bg[np.isfinite(val_bg) & (val_bg > 0)]
    if pos.size == 0:
        iso_min, iso_max = 0.0, 1.0
    else:
        iso_min = float(np.percentile(pos, 65))
        iso_max = float(np.percentile(pos, 99))
        if iso_max <= iso_min:
            iso_max = iso_min + 1e-3

    # Activation points
    if point_df is not None and not point_df.empty:
        p = point_df.head(int(max_points)).copy()
        xyz_act = p[["x", "y", "z"]].to_numpy(dtype=float)
        values = p["value"].to_numpy(dtype=float)
        region_name = p["region_name"].astype(str).to_numpy() if "region_name" in p.columns else None
        region_index = p["region_index"].to_numpy(dtype=int) if "region_index" in p.columns else None
    else:
        mask = np.abs(stat_data) >= float(threshold)
        coords = np.argwhere(mask)
        values = stat_data[mask].ravel()
        if values.size > int(max_points):
            k = int(max_points)
            idx = np.argpartition(np.abs(values), -k)[-k:]
            coords = coords[idx]
            values = values[idx]
        xyz_act = nib.affines.apply_affine(stat_img.affine, coords)
        region_name = None
        region_index = None

    fig = go.Figure()
    fig.add_trace(
        go.Isosurface(
            x=xyz_bg[:, 0],
            y=xyz_bg[:, 1],
            z=xyz_bg[:, 2],
            value=val_bg,
            isomin=iso_min,
            isomax=iso_max,
            surface_count=1,
            opacity=0.20,
            colorscale=[[0.0, "#9aa0a6"], [1.0, "#d0d4d9"]],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="brain_surface",
        )
    )

    vmax = float(np.max(np.abs(values))) if values.size > 0 else 1.0
    vmax = max(vmax, float(threshold), 1e-3)
    hovertemplate = "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<br>value=%{marker.color:.4f}"
    customdata = None
    if region_name is not None:
        if region_index is None:
            customdata = np.column_stack([region_name])
            hovertemplate += "<br>region=%{customdata[0]}"
        else:
            customdata = np.column_stack([region_name, region_index])
            hovertemplate += "<br>region=%{customdata[0]} (%{customdata[1]})"
    hovertemplate += "<extra></extra>"

    fig.add_trace(
        go.Scatter3d(
            x=xyz_act[:, 0] if xyz_act.size else [],
            y=xyz_act[:, 1] if xyz_act.size else [],
            z=xyz_act[:, 2] if xyz_act.size else [],
            mode="markers",
            marker=dict(
                size=int(point_size),
                opacity=float(point_opacity),
                color=values if values.size else [],
                colorscale="RdBu_r",
                cmin=-vmax,
                cmax=vmax,
                colorbar=dict(title="Stat"),
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="activation",
        )
    )

    fig.update_layout(
        title="3D 脑表面 + 激活体素（可旋转）",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            dragmode="turntable",
        ),
        showlegend=False,
    )
    return fig


def build_cortical_surface_panel(
    stat_img: nib.Nifti1Image,
    threshold: float,
    title: str = "Cortical Surface Projection",
    surf_mesh: str = "fsaverage5",
    smooth_fwhm: float = 6.0,
    cmap: str = "turbo",
):
    # 2x2 cortical panel: left/right x medial/lateral, closer to paper figures.
    if smooth_fwhm and float(smooth_fwhm) > 0:
        stat_img = smooth_img(stat_img, float(smooth_fwhm))
    data = np.asanyarray(stat_img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.nan_to_num(data.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    symmetric = bool(vmin < 0 and vmax > 0)
    out = plotting.plot_img_on_surf(
        stat_map=stat_img,
        surf_mesh=surf_mesh,
        hemispheres=["left", "right"],
        views=["lateral", "medial"],
        cmap=cmap,
        colorbar=True,
        threshold=float(threshold),
        bg_on_data=True,
        inflate=True,
        symmetric_cbar=symmetric,
        title=title,
    )
    # nilearn may return a Figure or a tuple like (Figure, axes).
    if hasattr(out, "savefig"):
        return out
    if isinstance(out, tuple):
        for item in out:
            if hasattr(item, "savefig"):
                return item
            fig_attr = getattr(item, "figure", None)
            if fig_attr is not None and hasattr(fig_attr, "savefig"):
                return fig_attr
    raise TypeError(f"Unsupported surface panel return type: {type(out)}")


def build_connectivity_strength_map(
    conn_mat: np.ndarray,
    atlas_img: nib.Nifti1Image,
    roi_ids: np.ndarray,
    metric: str,
    roi_labels: list[str] | None = None,
) -> tuple[nib.Nifti1Image, pd.DataFrame]:
    mat = np.asarray(conn_mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got shape={mat.shape}")
    if len(roi_ids) != mat.shape[0]:
        raise ValueError(f"roi_ids length ({len(roi_ids)}) != matrix size ({mat.shape[0]})")

    x = mat.copy()
    np.fill_diagonal(x, 0.0)
    if metric == "abs_sum":
        strength = np.sum(np.abs(x), axis=1)
    elif metric == "positive_sum":
        strength = np.sum(np.clip(x, 0.0, None), axis=1)
    elif metric == "degree":
        strength = np.sum(np.abs(x) >= 0.3, axis=1).astype(float)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    atlas_data = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int32)
    out_data = np.zeros(atlas_data.shape, dtype=np.float32)
    for rid, sval in zip(roi_ids, strength):
        out_data[atlas_data == int(rid)] = float(sval)

    out_img = nib.Nifti1Image(out_data, atlas_img.affine, atlas_img.header)
    out_img.header.set_data_dtype(np.float32)
    rank_df = pd.DataFrame({"index": roi_ids.astype(int), "strength": strength.astype(float)})
    if roi_labels is not None and len(roi_labels) == len(roi_ids):
        rank_df.insert(1, "label", [translate_region_label(x) for x in roi_labels])
    rank_df = rank_df.sort_values("strength", ascending=False).reset_index(drop=True)
    return out_img, rank_df


def roi_centers_from_atlas(atlas_img: nib.Nifti1Image, roi_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atlas_data = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int32)
    centers = np.full((len(roi_ids), 3), np.nan, dtype=float)
    valid = np.zeros(len(roi_ids), dtype=bool)
    for i, rid in enumerate(roi_ids):
        vox = np.argwhere(atlas_data == int(rid))
        if vox.size == 0:
            continue
        centers[i] = nib.affines.apply_affine(atlas_img.affine, np.mean(vox, axis=0))
        valid[i] = True
    return centers, valid


def select_connectome_edges(
    conn_mat: np.ndarray,
    labels: list[str],
    threshold: float,
    max_edges: int,
    valid_nodes: np.ndarray,
) -> pd.DataFrame:
    mat = np.asarray(conn_mat, dtype=float)
    rows = []
    for i in range(mat.shape[0]):
        if not valid_nodes[i]:
            continue
        for j in range(i + 1, mat.shape[1]):
            if not valid_nodes[j]:
                continue
            r = float(mat[i, j])
            if not np.isfinite(r) or abs(r) < float(threshold):
                continue
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "roi_a": labels[i],
                    "roi_b": labels[j],
                    "r": r,
                    "abs_r": abs(r),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["i", "j", "roi_a", "roi_b", "r", "abs_r"])
    return df.sort_values("abs_r", ascending=False).head(int(max_edges)).reset_index(drop=True)


def _scaled_node_sizes(strength: np.ndarray, min_size: float = 5.0, max_size: float = 24.0) -> np.ndarray:
    strength = np.asarray(strength, dtype=float)
    finite = strength[np.isfinite(strength)]
    if finite.size == 0 or float(np.max(finite)) <= float(np.min(finite)):
        return np.full(strength.shape, (min_size + max_size) / 2.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    return min_size + (strength - lo) / (hi - lo) * (max_size - min_size)


def build_connectome_3d_figure(
    conn_mat: np.ndarray,
    atlas_img: nib.Nifti1Image,
    roi_ids: np.ndarray,
    labels: list[str],
    edge_threshold: float,
    max_edges: int,
) -> tuple[go.Figure, pd.DataFrame]:
    centers, valid_nodes = roi_centers_from_atlas(atlas_img, roi_ids)
    mat = np.asarray(conn_mat, dtype=float)
    strength = np.sum(np.abs(mat - np.diag(np.diag(mat))), axis=1)
    node_sizes = _scaled_node_sizes(strength)
    edges = select_connectome_edges(mat, labels, edge_threshold, max_edges, valid_nodes)

    atlas_data = (np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int32) > 0).astype(float)
    step = 2 if max(atlas_data.shape) > 96 else 1
    atlas_ds = atlas_data[::step, ::step, ::step]
    ix, iy, iz = np.indices(atlas_ds.shape)
    vox = np.column_stack((ix.ravel() * step, iy.ravel() * step, iz.ravel() * step))
    xyz = nib.affines.apply_affine(atlas_img.affine, vox)

    fig = go.Figure()
    fig.add_trace(
        go.Isosurface(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            value=atlas_ds.ravel(),
            isomin=0.5,
            isomax=1.0,
            surface_count=1,
            opacity=0.12,
            colorscale=[[0.0, "#d9dee5"], [1.0, "#f3f5f8"]],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="brain",
        )
    )

    for sign_name, sign_mask, color in [
        ("positive edges", edges["r"] > 0 if not edges.empty else [], "#d73027"),
        ("negative edges", edges["r"] < 0 if not edges.empty else [], "#2166ac"),
    ]:
        xs, ys, zs = [], [], []
        edge_subset = edges.loc[sign_mask] if not edges.empty else edges
        for _, row in edge_subset.iterrows():
            a = centers[int(row["i"])]
            b = centers[int(row["j"])]
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip",
                name=sign_name,
            )
        )

    valid_idx = np.where(valid_nodes)[0]
    node_hover = [
        f"{labels[i]}<br>index={int(roi_ids[i])}<br>strength={strength[i]:.4f}"
        for i in valid_idx
    ]
    fig.add_trace(
        go.Scatter3d(
            x=centers[valid_idx, 0],
            y=centers[valid_idx, 1],
            z=centers[valid_idx, 2],
            mode="markers",
            marker=dict(
                size=node_sizes[valid_idx],
                color=strength[valid_idx],
                colorscale="Viridis",
                opacity=0.95,
                line=dict(color="#111111", width=1),
                colorbar=dict(title="strength"),
            ),
            text=node_hover,
            hovertemplate="%{text}<extra></extra>",
            name="ROI nodes",
        )
    )
    fig.update_layout(
        title="功能连接网络图：节点=脑区，线=ROI-ROI连接",
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            dragmode="turntable",
        ),
        legend=dict(orientation="h", y=0.02, x=0.02),
    )
    return fig, edges


def build_connectome_circle_figure(
    conn_mat: np.ndarray,
    labels: list[str],
    edges: pd.DataFrame,
) -> go.Figure:
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    strength = np.sum(np.abs(conn_mat - np.diag(np.diag(conn_mat))), axis=1)
    node_sizes = _scaled_node_sizes(strength, 4.0, 16.0)

    fig = go.Figure()
    for _, row in edges.iterrows():
        i, j = int(row["i"]), int(row["j"])
        color = "#d73027" if float(row["r"]) > 0 else "#2166ac"
        fig.add_trace(
            go.Scatter(
                x=[x[i], x[j]],
                y=[y[i], y[j]],
                mode="lines",
                line=dict(color=color, width=1 + 3 * float(row["abs_r"])),
                opacity=0.55,
                hovertemplate=f"{labels[i]} ↔ {labels[j]}<br>r={float(row['r']):.4f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=node_sizes, color=strength, colorscale="Viridis", line=dict(color="#111", width=1)),
            text=[f"{label}<br>strength={strength[i]:.4f}" for i, label in enumerate(labels)],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="环形连接图",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=45, b=0),
        height=640,
        plot_bgcolor="white",
    )
    return fig


def atlas_summary(
    stat_data: np.ndarray,
    stat_img: nib.Nifti1Image,
    threshold: float,
    atlas_path: str,
    labels_csv: str,
) -> pd.DataFrame:
    atlas_img = nib.load(atlas_path)
    atlas_resampled = resample_to_img(atlas_img, stat_img, interpolation="nearest", force_resample=True)
    atlas_data = np.asanyarray(atlas_resampled.dataobj).astype(int)
    act_mask = np.abs(stat_data) >= threshold
    idx = atlas_data[act_mask]
    idx = idx[idx > 0]
    if idx.size == 0:
        return pd.DataFrame(columns=["region_index", "region_name", "voxel_count", "mean_abs_stat"])

    uniq, counts = np.unique(idx, return_counts=True)
    rows: list[dict] = []
    labels_df = None
    if labels_csv and Path(labels_csv).exists():
        p = Path(labels_csv)
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        labels_df = pd.read_csv(p, sep=sep)

    for region_i, count_i in zip(uniq, counts):
        region_mask = (atlas_data == region_i) & act_mask
        mean_abs = float(np.mean(np.abs(stat_data[region_mask])))
        region_name = f"Region_{int(region_i)}"
        if labels_df is not None and "index" in labels_df.columns:
            hit = labels_df.loc[labels_df["index"] == int(region_i)]
            if not hit.empty:
                for key in ["label", "name", "region"]:
                    if key in hit.columns:
                        region_name = str(hit.iloc[0][key])
                        break
        rows.append(
            {
                "region_index": int(region_i),
                "region_name": region_name,
                "voxel_count": int(count_i),
                "mean_abs_stat": mean_abs,
            }
        )
    out = pd.DataFrame(rows).sort_values("voxel_count", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    default_cfg = Path(__file__).resolve().parents[2] / "configs" / "study_config.json"
    cfg_path_text = st.sidebar.text_input("配置文件 JSON", str(default_cfg))
    cfg = read_json(Path(cfg_path_text))

    derivatives_default = cfg.get("derivatives_root", "D:/bishe/derivatives")
    derivatives_root = Path(st.sidebar.text_input("结果根目录", derivatives_default))

    root_ok, root_message = validate_derivatives_root(derivatives_root)
    if not root_ok:
        st.sidebar.error(root_message)

    subject_dirs = list_subject_dirs(derivatives_root)
    subject_names = [p.name for p in subject_dirs]
    if subject_names:
        selected_subject = st.sidebar.selectbox("被试", subject_names, index=0)
        subject_dir = derivatives_root / selected_subject
    else:
        selected_subject = ""
        subject_dir = derivatives_root
        if root_ok:
            st.sidebar.warning("未找到被试目录：结果根目录下没有 sub- 开头的文件夹。")
        st.sidebar.warning("在结果根目录下未找到被试文件夹（应为 sub-xxx）。")

    suggest = suggest_files(subject_dir)
    stat_options = [str(p) for p in suggest["stat_maps"]]
    bg_options = [str(p) for p in suggest["bg_maps"]]
    conn_options = [str(p) for p in suggest["conn_mats"]]

    st.sidebar.markdown("### 可选输入")
    atlas_path = st.sidebar.text_input("脑图谱 NIfTI 路径（可选）", "")
    atlas_labels = st.sidebar.text_input("脑图谱标签 CSV/TSV（可选）", "")
    region_labels = st.sidebar.text_input("连接矩阵 ROI 标签 CSV（可选）", "")

    tabs = st.tabs(["统计结果图像", "脑区激活图谱", "功能连接矩阵", "运行状态", "任务执行"])

    with tabs[0]:
        st.subheader("统计结果图像")
        stat_map_paths = st.multiselect("统计图文件（可多选）", options=stat_options)
        stat_map_manual = st.text_input("或手动输入统计图路径（多个用 ; 分隔）", "")
        manual_stats = parse_multi_paths(stat_map_manual)
        chosen_stats = []
        for p in stat_map_paths + manual_stats:
            if p not in chosen_stats:
                chosen_stats.append(p)

        bg_map_path = st.selectbox("背景图（可选）", options=[""] + bg_options)
        bg_map_manual = st.text_input("或手动输入背景图路径", "")
        chosen_bg = bg_map_manual.strip() or bg_map_path

        threshold = st.slider("阈值", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
        overlap_mode = st.selectbox("多图保留模式", ["单图显示", "严格交集(AND)", "一致性比例"])
        min_ratio = 1.0
        if overlap_mode == "一致性比例":
            min_ratio = st.slider("最小一致性比例", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
        elif overlap_mode == "严格交集(AND)":
            min_ratio = 1.0
        display_mode = st.selectbox("显示模式", ["ortho", "x", "y", "z", "xz", "yx", "yz"])
        cut_n = st.slider("切片数", min_value=3, max_value=16, value=12, step=1)

        if chosen_stats:
            missing = [p for p in chosen_stats if not Path(p).exists()]
            if missing:
                st.error("以下统计图不存在:\n" + "\n".join(missing))
            else:
                bg_img = nib.load(chosen_bg) if chosen_bg and Path(chosen_bg).exists() else None
                if len(chosen_stats) == 1 or overlap_mode == "单图显示":
                    stat_img = nib.load(chosen_stats[0])
                    title = f"{Path(chosen_stats[0]).name}（阈值={threshold:.2f}）"
                    stat_thr = threshold
                else:
                    try:
                        stat_img, kept_vox = build_overlap_stat_image(chosen_stats, threshold, min_ratio)
                        title = f"Overlap: n={len(chosen_stats)}, ratio>={min_ratio:.2f}, voxels={kept_vox}"
                        stat_thr = max(0.01, min_ratio)
                    except ValueError as exc:
                        st.error(f"多图交集计算失败: {exc}")
                        stat_img = None
                        stat_thr = threshold

                if stat_img is None:
                    st.stop()
                stat_data_preview = np.asanyarray(stat_img.dataobj)
                if stat_data_preview.ndim == 4:
                    stat_data_preview = stat_data_preview[..., 0]
                if int(np.sum(np.abs(stat_data_preview) >= stat_thr)) == 0:
                    st.warning("当前阈值下没有满足条件的体素。")

                fig_obj = plotting.plot_stat_map(
                    stat_map_img=stat_img,
                    bg_img=bg_img,
                    threshold=stat_thr,
                    display_mode=display_mode,
                    cut_coords=resolve_cut_coords(display_mode, cut_n, stat_img, stat_thr),
                    colorbar=True,
                    draw_cross=False,
                    title=title,
                    figure=plt.figure(figsize=(14, 5), constrained_layout=True),
                )
                st.pyplot(fig_obj.frame_axes.figure, width='stretch')
                fig_obj.close()

    with tabs[1]:
        st.subheader("脑区激活图谱")
        st.caption("图谱模式：提供 atlas 时汇总激活脑区；未提供时显示峰值 MNI 坐标。")

        stat_map_path = st.selectbox("用于激活分析的统计图", options=[""] + stat_options, key="act_stat")
        stat_map_manual = st.text_input("或手动输入统计图路径", "", key="act_stat_manual")
        chosen_stat = stat_map_manual.strip() or stat_map_path
        act_bg_map = st.selectbox("3D背景图（可选，建议 mean/swar）", options=[""] + bg_options, key="act_bg")
        act_bg_manual = st.text_input("或手动输入3D背景图路径", "", key="act_bg_manual")
        chosen_act_bg = act_bg_manual.strip() or act_bg_map
        threshold = st.slider("激活阈值", min_value=0.0, max_value=10.0, value=3.1, step=0.1, key="act_thr")

        if chosen_stat:
            if not Path(chosen_stat).exists():
                st.error(f"未找到统计图: {chosen_stat}")
            else:
                stat_img = nib.load(chosen_stat)
                stat_data = np.asanyarray(stat_img.dataobj)
                if stat_data.ndim == 4:
                    stat_data = stat_data[..., 0]
                    stat_img = nib.Nifti1Image(stat_data, stat_img.affine, stat_img.header)

                active_count = int(np.sum(np.abs(stat_data) >= threshold))
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="metric-card"><b>激活体素数</b><br>{active_count}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f'<div class="metric-card"><b>最大 |统计值|</b><br>{float(np.nanmax(np.abs(stat_data))):.4f}</div>',
                        unsafe_allow_html=True,
                    )

                xyz_df = activation_voxel_table(
                    stat_data=stat_data,
                    affine=stat_img.affine,
                    threshold=threshold,
                    stat_img=stat_img,
                    atlas_path=atlas_path if (atlas_path and Path(atlas_path).exists()) else "",
                    labels_csv=atlas_labels,
                )
                if xyz_df.empty:
                    st.warning("当前阈值下未检测到激活体素。")
                else:
                    max_points = st.slider("3D显示体素上限", min_value=1000, max_value=50000, value=10000, step=1000, key="act_max_points")
                    point_size = st.slider("点大小", min_value=1, max_value=8, value=2, step=1, key="act_point_size")
                    point_opacity = st.slider("点透明度", min_value=0.10, max_value=1.00, value=0.75, step=0.05, key="act_point_opacity")
                    render_style = st.selectbox("3D显示样式", ["脑表面+激活点", "仅激活点"], key="act_render_style")

                    plot_df = xyz_df.head(int(max_points)).copy()
                    bg_img_3d = None
                    if chosen_act_bg:
                        if Path(chosen_act_bg).exists():
                            try:
                                bg_img_3d = nib.load(chosen_act_bg)
                            except Exception:
                                bg_img_3d = None
                        else:
                            st.warning(f"3D背景图不存在: {chosen_act_bg}")

                    if render_style == "脑表面+激活点":
                        fig_3d = build_surface_activation_3d_figure(
                            stat_img=stat_img,
                            threshold=threshold,
                            max_points=int(max_points),
                            point_size=int(point_size),
                            point_opacity=float(point_opacity),
                            bg_img=bg_img_3d,
                            point_df=plot_df,
                        )
                    else:
                        fig_3d = build_activation_3d_figure(plot_df, point_size=point_size, opacity=point_opacity)
                    st.plotly_chart(fig_3d, width='stretch')
                    st.caption(f"当前阈值激活体素总数: {len(xyz_df)}；3D图中显示: {len(plot_df)}")

                    with st.expander("查看激活体素坐标表（Top 200）", expanded=False):
                        st.dataframe(xyz_df.head(200), width='stretch')

                if atlas_path and Path(atlas_path).exists():
                    try:
                        atlas_df = atlas_summary(stat_data, stat_img, threshold, atlas_path, atlas_labels)
                        if atlas_df.empty:
                            st.warning("当前阈值下，激活体素与图谱区域无重叠。")
                        else:
                            with st.expander("查看图谱脑区汇总（可选）", expanded=False):
                                st.dataframe(atlas_df.head(30), width='stretch')
                                fig_bar = px.bar(
                                    atlas_df.head(20),
                                    x="region_name",
                                    y="voxel_count",
                                    hover_data=["mean_abs_stat"],
                                    title="激活脑区 Top20（按体素数）",
                                )
                                fig_bar.update_layout(xaxis_title="脑区", yaxis_title="体素数")
                                st.plotly_chart(fig_bar, width='stretch')
                    except Exception as exc:
                        st.error(f"图谱分析失败: {exc}")

    with tabs[2]:
        st.subheader("功能连接矩阵")
        conn_path = st.selectbox("连接矩阵文件", options=[""] + conn_options)
        conn_manual = st.text_input("或手动输入连接矩阵路径（.csv/.tsv/.npy）", "")
        chosen_conn = conn_manual.strip() or conn_path

        if chosen_conn:
            try:
                mat = load_matrix_checked(chosen_conn)
                n = mat.shape[0]
                default_region_label_path = region_labels.strip()
                sibling_labels = Path(chosen_conn).with_name("roi_labels.csv")
                if not default_region_label_path and sibling_labels.exists():
                    default_region_label_path = str(sibling_labels)
                label_warning = describe_region_label_file(default_region_label_path, n)
                if label_warning:
                    st.warning(label_warning)
                labels = load_region_labels(default_region_label_path, n)
                vmax = float(np.nanmax(np.abs(mat)))
                vmax = max(vmax, 0.2)
                vis_thr = st.slider("显示阈值（|r| 低于该值记为 0）", 0.0, vmax, 0.2, 0.01)

                shown = mat.copy()
                shown[np.abs(shown) < vis_thr] = 0.0

                fig_heat = px.imshow(
                    shown,
                    x=labels,
                    y=labels,
                    color_continuous_scale="RdBu_r",
                    zmin=-vmax,
                    zmax=vmax,
                    aspect="auto",
                    title=f"功能连接矩阵（{Path(chosen_conn).name}）",
                )
                st.plotly_chart(fig_heat, width='stretch')

                triu = np.triu_indices(n, k=1)
                vals = mat[triu]
                strong_mask = np.abs(vals) >= vis_thr
                density = float(np.mean(strong_mask))
                mean_abs = float(np.mean(np.abs(vals)))
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="metric-card"><b>连边密度</b><br>{density:.3f}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><b>平均 |r|</b><br>{mean_abs:.3f}</div>', unsafe_allow_html=True)

                if vals.size > 0:
                    top_k = min(20, vals.size)
                    idx = np.argpartition(np.abs(vals), -top_k)[-top_k:]
                    pairs = []
                    for i in idx:
                        a = triu[0][i]
                        b = triu[1][i]
                        pairs.append({"roi_a": labels[a], "roi_b": labels[b], "r": float(vals[i]), "abs_r": float(abs(vals[i]))})
                    pairs_df = pd.DataFrame(pairs).sort_values("abs_r", ascending=False).reset_index(drop=True)
                    st.dataframe(pairs_df, width='stretch')

                default_roi_index = default_region_label_path
                default_conn_atlas = atlas_path.strip()
                if not default_conn_atlas:
                    for candidate in ["raal.nii", "aal.nii"]:
                        sibling_atlas = Path(chosen_conn).with_name(candidate)
                        if sibling_atlas.exists():
                            default_conn_atlas = str(sibling_atlas)
                            break

                st.markdown("#### 网络图输入")
                conn_atlas_path = st.text_input(
                    "脑图谱 NIfTI（用于ROI坐标/映射）",
                    value=default_conn_atlas,
                    key="conn_strength_atlas_path",
                )
                conn_roi_index_path = st.text_input(
                    "ROI 索引/标签文件（CSV/TSV，含 index,label）",
                    value=default_roi_index,
                    key="conn_strength_roi_index_path",
                )

                st.markdown("#### 功能连接网络图（节点-连边）")
                c_edge_thr, c_edge_n = st.columns(2)
                with c_edge_thr:
                    network_thr = st.slider(
                        "连边阈值（|r|）",
                        min_value=0.0,
                        max_value=vmax,
                        value=min(0.30, vmax),
                        step=0.01,
                        key="conn_network_edge_thr",
                    )
                with c_edge_n:
                    network_max_edges = int(
                        st.number_input(
                            "最多显示连边数",
                            min_value=10,
                            max_value=500,
                            value=80,
                            step=10,
                            key="conn_network_max_edges",
                        )
                    )
                show_network = st.button("生成连接网络图", width='stretch', key="btn_conn_network")
                if show_network:
                    if not conn_atlas_path or not Path(conn_atlas_path).exists():
                        st.error("请先提供有效的脑图谱 NIfTI 路径。")
                    else:
                        try:
                            roi_ids, roi_names = load_region_index_label_pairs(conn_roi_index_path, n)
                            atlas_img = nib.load(conn_atlas_path)
                            net_fig, net_edges = build_connectome_3d_figure(
                                conn_mat=mat,
                                atlas_img=atlas_img,
                                roi_ids=roi_ids,
                                labels=roi_names,
                                edge_threshold=float(network_thr),
                                max_edges=int(network_max_edges),
                            )
                            st.session_state["conn_network_fig"] = net_fig
                            st.session_state["conn_network_edges"] = net_edges
                            st.session_state["conn_network_labels"] = roi_names
                            st.session_state["conn_network_key"] = (
                                chosen_conn,
                                conn_atlas_path,
                                conn_roi_index_path,
                                float(network_thr),
                                int(network_max_edges),
                                n,
                            )
                        except Exception as exc:
                            st.error(f"连接网络图生成失败: {exc}")

                current_network_key = (
                    chosen_conn,
                    conn_atlas_path,
                    conn_roi_index_path,
                    float(network_thr),
                    int(network_max_edges),
                    n,
                )
                net_fig = st.session_state.get("conn_network_fig")
                net_edges = st.session_state.get("conn_network_edges")
                if net_fig is not None and st.session_state.get("conn_network_key") == current_network_key:
                    st.plotly_chart(net_fig, width='stretch')
                    if isinstance(net_edges, pd.DataFrame) and not net_edges.empty:
                        circle_fig = build_connectome_circle_figure(
                            conn_mat=mat,
                            labels=st.session_state.get("conn_network_labels", labels),
                            edges=net_edges,
                        )
                        with st.expander("查看环形连接图", expanded=False):
                            st.plotly_chart(circle_fig, width='stretch')
                            st.dataframe(net_edges[["roi_a", "roi_b", "r", "abs_r"]], width='stretch')

                st.markdown("#### 连接强度脑图（由功能连接矩阵生成）")
                st.caption("当前默认只在内存中生成预览；只有点击保存按钮时才写入 NIfTI/CSV 文件。")
                c_metric, c_thr, c_smooth = st.columns(3)
                with c_metric:
                    conn_metric = st.selectbox(
                        "强度指标",
                        options=["abs_sum", "positive_sum", "degree"],
                        index=0,
                        key="conn_strength_metric",
                    )
                with c_thr:
                    surf_thr = st.number_input(
                        "表面图阈值（strength）",
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        key="conn_strength_surface_thr",
                    )
                with c_smooth:
                    surf_smooth = st.slider(
                        "表面平滑 FWHM(mm)",
                        min_value=0.0,
                        max_value=14.0,
                        value=6.0,
                        step=1.0,
                        key="conn_strength_surface_smooth",
                    )
                c_cmap, c_mesh = st.columns(2)
                with c_cmap:
                    surf_cmap = st.selectbox(
                        "表面配色",
                        options=["turbo", "Spectral_r", "cold_hot", "RdBu_r"],
                        index=0,
                        key="conn_strength_surface_cmap",
                    )
                with c_mesh:
                    surf_mesh = st.selectbox(
                        "表面模板",
                        options=["fsaverage5"],
                        index=0,
                        key="conn_strength_surface_mesh",
                    )
                run_conn_brainmap = st.button("生成/刷新预览", width='stretch', key="btn_conn_strength_map")

                if run_conn_brainmap:
                    if not conn_atlas_path or not Path(conn_atlas_path).exists():
                        st.error("请先提供有效的脑图谱 NIfTI 路径。")
                    else:
                        try:
                            roi_ids, roi_names = load_region_index_label_pairs(conn_roi_index_path, n)
                            atlas_img = nib.load(conn_atlas_path)
                            strength_img, strength_rank = build_connectivity_strength_map(
                                conn_mat=mat,
                                atlas_img=atlas_img,
                                roi_ids=roi_ids,
                                metric=conn_metric,
                                roi_labels=roi_names,
                            )
                            st.session_state["conn_strength_img"] = strength_img
                            st.session_state["conn_strength_rank"] = strength_rank
                            st.session_state["conn_strength_key"] = (
                                chosen_conn,
                                conn_atlas_path,
                                conn_roi_index_path,
                                conn_metric,
                                n,
                            )
                        except Exception as exc:
                            st.error(f"连接强度脑图生成失败: {exc}")

                current_key = (chosen_conn, conn_atlas_path, conn_roi_index_path, conn_metric, n)
                panel_img = st.session_state.get("conn_strength_img")
                rank_df = st.session_state.get("conn_strength_rank")
                if panel_img is not None and st.session_state.get("conn_strength_key") == current_key:
                    st.success("已生成当前预览（未自动保存文件）。")
                    try:
                        panel_fig = build_cortical_surface_panel(
                            stat_img=panel_img,
                            threshold=float(surf_thr),
                            title=f"{Path(chosen_conn).stem} | Connectivity Strength Surface",
                            surf_mesh=surf_mesh,
                            smooth_fwhm=float(surf_smooth),
                            cmap=surf_cmap,
                        )
                        st.pyplot(panel_fig, width='stretch')
                        plt.close(panel_fig)
                    except Exception as exc:
                        st.warning(
                            "表面图渲染失败（首次可能需要下载 fsaverage 模板）: "
                            f"{exc}"
                        )
                    if isinstance(rank_df, pd.DataFrame):
                        with st.expander("查看 ROI 连接强度排序", expanded=False):
                            st.dataframe(rank_df.head(50), width='stretch')
                    if st.button("保存当前脑图与排序表", width='stretch', key="btn_save_conn_strength_map"):
                        out_nii = Path(chosen_conn).with_name(
                            f"{Path(chosen_conn).stem}_strength_{conn_metric}.nii"
                        )
                        out_csv = Path(chosen_conn).with_name(
                            f"{Path(chosen_conn).stem}_strength_{conn_metric}_rank.csv"
                        )
                        nib.save(panel_img, str(out_nii))
                        if isinstance(rank_df, pd.DataFrame):
                            rank_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
                        st.success(f"已保存: {out_nii}")
                        st.caption(f"排序表: {out_csv}")
                else:
                    st.info("点击“生成/刷新预览”后显示连接强度脑图。")
            except Exception as exc:
                st.error(f"连接矩阵加载/显示失败: {exc}")

    with tabs[3]:
        st.subheader("运行状态")
        qc_csv = derivatives_root / "qc" / "preproc_qc_metrics.csv"
        status_csv = derivatives_root / "logs" / "preproc_status.csv"

        if status_csv.exists():
            status_df = pd.read_csv(status_csv)
            st.markdown("#### 预处理状态")
            st.dataframe(status_df, width='stretch')
        else:
            st.info(f"未找到状态文件: {status_csv}")

        if qc_csv.exists():
            qc_df = pd.read_csv(qc_csv)
            st.markdown("#### 质控指标")
            st.dataframe(qc_df, width='stretch')
            if "mean_fd" in qc_df.columns:
                fig = px.bar(qc_df, x="subject_id", y="mean_fd", title="各被试 Mean FD")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info(f"未找到 QC 文件: {qc_csv}")

    with tabs[4]:
        st.subheader("任务执行")
        st.markdown("#### Backend Task Runner (MATLAB)")
        st.caption("直接在页面中运行后端 MATLAB 脚本。任务可能持续几分钟。")

        repo_root = get_repo_root()
        matlab_exe = st.text_input("MATLAB executable", value=cfg.get("matlab_exe", "matlab"), key="matlab_exe")
        matlab_script_override = st.text_input(
            "MATLAB script path override（可选，用于测试脚本不存在场景）",
            value="",
            key="matlab_script_override",
        )
        timeout_minutes = int(
            st.number_input(
                "Timeout minutes (0 = no timeout)",
                min_value=0,
                max_value=24 * 60,
                value=0,
                step=10,
                key="matlab_timeout_minutes",
            )
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            run_preproc = st.button("Run Preprocess", width='stretch', key="btn_run_preproc")
        with c2:
            run_first = st.button("Run First-Level", width='stretch', key="btn_run_first_level")
        with c3:
            run_conn = st.button("Run Connectivity", width='stretch', key="btn_run_connectivity")

        if run_preproc or run_first or run_conn:
            script_path = None
            task_name = ""
            if run_preproc:
                task_name = "Preprocess"
                script_path = repo_root / "matlab" / "preproc" / "run_preproc_batch.m"
            elif run_first:
                task_name = "FirstLevel"
                script_path = repo_root / "matlab" / "stats" / "run_first_level_batch.m"
            else:
                task_name = "Connectivity"
                script_path = repo_root / "matlab" / "connectivity" / "run_connectivity_batch.m"
            if matlab_script_override.strip():
                script_path = Path(matlab_script_override.strip())

            with st.spinner(f"Running {task_name}..."):
                result = run_matlab_script(script_path, matlab_exe, timeout_minutes)
            result["task_name"] = task_name
            result["script_path"] = str(script_path)
            st.session_state["backend_last_result"] = result

        last = st.session_state.get("backend_last_result")
        if last:
            status_text = "SUCCESS" if last.get("ok") else "FAILED"
            st.markdown(f"**Last Task:** {last.get('task_name', '-')} ({status_text})")
            st.write(f"- Script: `{last.get('script_path', '-')}`")
            st.write(f"- Command: `{last.get('command', '-')}`")
            st.write(f"- Return code: `{last.get('returncode', '-')}`")
            st.write(f"- Start: `{last.get('start_time', '-')}`")
            st.write(f"- End: `{last.get('end_time', '-')}`")
            st.text_area("Task Log", value=last.get("log", ""), height=260, key="backend_task_log")


if __name__ == "__main__":
    main()
