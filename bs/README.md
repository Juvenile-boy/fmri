# fMRI Graduation Project System (SPM12 + DPABI + Interactive UI)

This repo now matches the thesis target:

- SPM/DPABI-based preprocessing pipeline
- Integrated result presentation
- Interactive UI for:
  - statistical result images
  - activation atlas summary
  - functional connectivity matrix

## 1) Project Structure

```text
bs\
  configs\
    study_config.json
    study_config.example.json
    subjects.tsv
    subjects.example.tsv
  matlab\
    preproc\
      run_preproc_batch.m
      preprocess_subject_spm.m
    stats\
      run_first_level_batch.m
      first_level_subject_spm.m
    connectivity\
      run_connectivity_batch.m
      compute_subject_connectivity.m
    qc\
      collect_qc_metrics.m
      compute_fd_from_rp.m
  python\
    app\
      fmri_visualization_app.py
    report\
      generate_qc_report.py
  scripts\
    run_pipeline.ps1
    run_visual_app.ps1
  requirements_mature.txt
```

## 2) Environment

- MATLAB R2022b+ with SPM12 and DPABI
- Python 3.11+

```powershell
py -m pip install -r requirements_mature.txt
```

## 3) Configure Inputs

Edit [configs/study_config.json](configs/study_config.json).

Edit [configs/subjects.tsv](configs/subjects.tsv).

`subjects.tsv` columns:

- `subject_id`
- `func_input` (supports 4D NIfTI / 3D NIfTI / folder of NIfTI / DICOM folder)
- `t1_nii` (supports T1 NIfTI file or structural DICOM folder)
- `events_tsv` (optional, for first-level task design; if missing, fallback to single-condition full-run model)

Additional config fields (for advanced modules):

- `first_level_hpf` (default 128)
- `first_level_cvi` (default `AR(1)`)
- `connectivity_atlas_nii` (required for connectivity step)
- `connectivity_atlas_labels` (optional, CSV/TSV with `index,label`)
- `connectivity_min_voxels` (default 20)

## 4) Run Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
```

Outputs:

- `derivatives/sub-*/preproc/*` preprocess outputs
- `derivatives/logs/preproc_status.csv`
- `derivatives/qc/preproc_qc_metrics.csv`
- `derivatives/qc/qc_report.html`

## 4.1) Run First-Level Statistics (SPM)

```matlab
run('D:/Users/DELL/Desktop/bs/matlab/stats/run_first_level_batch.m')
```

Outputs:

- `derivatives/sub-*/first_level/SPM.mat`
- `derivatives/sub-*/first_level/con_*.nii`
- `derivatives/sub-*/first_level/spmT_*.nii`
- `derivatives/logs/first_level_status.csv`

## 4.2) Run Connectivity Matrix Generation

```matlab
run('D:/Users/DELL/Desktop/bs/matlab/connectivity/run_connectivity_batch.m')
```

Outputs:

- `derivatives/sub-*/connectivity/fc_matrix.csv`
- `derivatives/sub-*/connectivity/fc_matrix_fisher_z.csv`
- `derivatives/sub-*/connectivity/roi_labels.csv`
- `derivatives/logs/connectivity_status.csv`

## 5) Launch Interactive Visualization System

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_visual_app.ps1
```

Default URL:

- `http://localhost:8501`

Main tabs:

- `统计结果图像`: stat-map threshold and display controls
- `脑区激活图谱`: atlas-based region summary or peak coordinates
- `功能连接矩阵`: matrix heatmap, thresholding, strongest connections
- `运行状态`: status/QC table dashboard

Accepted data examples in UI:

- Statistical map: `spmT_*.nii`, `con_*.nii`, `zmap*.nii`
- Background map: `mean*.nii`, `wra*.nii`, `swar*.nii`
- Connectivity matrix: `.csv`, `.tsv`, `.npy` (square matrix)
- Atlas labels (optional): CSV/TSV with columns like `index,label`

## 6) Thesis Reproducibility Tips

- Save `study_config.json`, `subjects.tsv`, status CSV, QC CSV as appendix.
- Keep one screenshot per UI tab for the system design chapter.
