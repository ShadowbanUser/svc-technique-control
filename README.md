# Singing-technique control in a frozen zero-shot singing voice converter

Code, trained adapter, result files and listening-test material for the paper

> X. Zhang, "How Much of a Singing Technique Control Is Equalisation? Evidence from a Frozen Zero-Shot Singing Voice Converter," manuscript submitted for publication, 2026.

The paper trains a technique-gated low-rank adapter (9.8 million parameters) on top of a frozen Seed-VC decoder, adding five continuous technique controls (breathy, belt, vocal fry, forte, piano). The output of the converter is unchanged when all controls are zero. The controls are evaluated against real technique recordings of five held-out VocalSet singers, with a cepstral distance that splits into a part a static equaliser can change and a part it cannot, a technique classifier trained on real recordings, and a preregistered listening test with 20 listeners.

## Contents

| Path | What it is |
|---|---|
| `src/steer/` | Adapter (`adapter.py`), residual-stream hooks for the training-free baselines (`hooks.py`), acoustic descriptors (`metrics.py`), equaliser-invariant distance and matched equaliser (`objective.py`) |
| `scripts/train/` | Construction of DTW-aligned technique pairs from VocalSet, adapter training, and the no-alignment ablation cache |
| `scripts/steer/` | Experiments 02 to 08, the analysis of experiment 06 (`analyse_exp06.py`) and of experiments 07 and 08 (`analyse_exp07_08.py`, and `exp08_phonemes.py` for the comparison by GTSinger's phoneme labels), and a comparison of the bootstrap intervals with those from all resamples (`compare_exp06_bootstrap.py`); `probe_v0.py` holds the shared Seed-VC session used by the other scripts |
| `scripts/listen/` | Listening-test package builder (`build_listening_package.py`, with the page template `listening_main_template.html`) and the analysis script (`analyse_listening.py`) |
| `scripts/figures/` | Scripts for Figures 1 to 4 |
| `scripts/run_seed_baseline.py`, `configs/` | Loading of the pinned Seed-VC checkpoint and the Python package versions used |
| `checkpoints/adapter_final.pt` | The trained adapter reported in the paper (checkpoint after the final training step) |
| `results/` | Summary and per-item result files of every experiment in the paper (JSON), and the outputs of the analysis scripts |
| `listening_test/` | The test page and its 64 audio files, the answer key, a labelled stimulus page and aggregated responses |
| `figures/` | Figures 1 to 4 (PDF; Figure 4 also as PNG) and the values plotted in Figure 4 |
| `preregistration/` | The preregistration text of the listening test |
| `launchers_windows/` | The batch files with which each experiment was run, in order |

## Where each result in the paper comes from

| Paper | Script | Result files |
|---|---|---|
| Section 4.3, training pairs | `scripts/train/prep_technique_pairs.py` | pair lists in `results/exp04/train/summary.json` |
| Section 4.4, adapter training | `scripts/train/train_technique_adapter.py` | `checkpoints/adapter_final.pt`, `results/exp04/train/` |
| Section 6.2, training-free control (Tables 1 and 2) | `scripts/steer/exp02_mechanisms.py`, `scripts/steer/exp03_panel.py` | `results/exp02/`, `results/exp03/` |
| Section 6.3, adapter on the test phrases (Tables 3 to 5) | `scripts/train/train_technique_adapter.py`, `scripts/steer/exp04_adapter_eval.py`, `scripts/steer/exp05_controls.py` | `results/exp04/train/summary.json`, `results/exp04/eval/`, `results/exp05/domain_control.json` |
| Sections 6.4 and 6.5, strength sweep and ablation | `scripts/steer/exp05_controls.py`, `scripts/train/prep_self_content.py` | `results/exp05/sweep/`, `results/exp05/ablation_nodtw/` |
| Sections 6.6 and 6.7, real recordings and classifier (Tables 6 to 8, Figures 2 and 3) | `scripts/steer/exp06_groundtruth.py`, `scripts/steer/exp06_classifier.py`, `scripts/steer/analyse_exp06.py`, `scripts/figures/fig23.py` | `results/exp06/`; the numbers of the tables and figures in `results/exp06/analysis/` |
| Check of the bootstrap intervals (Section 5.6) against all 3125 resamples of the five singers | `scripts/steer/analyse_exp06.py --exact`, `scripts/steer/compare_exp06_bootstrap.py` | `results/exp06/analysis_exact/` |
| Section 6.6, speaker similarity and pitch of real vocal fry; Section 6.8, conversion between singers and lyric phrases (Table 9) | `scripts/steer/exp07_crosssinger.py`, `scripts/steer/exp08_lyrics.py`, `scripts/steer/analyse_exp07_08.py`, `scripts/steer/exp08_phonemes.py` | `results/exp07/`, `results/exp08/`, `results/exp07_08_report.txt` |
| Sections 5.7 and 6.9, listening test (Table 10, Figure 4) | `scripts/listen/`, `scripts/figures/fig4.py` | `listening_test/`, `figures/figure4_values.json` |

## Requirements

- Seed-VC, https://github.com/Plachtaa/seed-vc at commit `51383efd921027683c89e5348211d93ff12ac2a8`, checked out at `external/seed-vc/`. The 44.1 kHz singing checkpoint and its auxiliary models are listed with checksums in `configs/seed_weight_manifest.json` and loaded by `scripts/run_seed_baseline.py`.
- VocalSet 1.2 (https://zenodo.org/records/1442513), extracted with `scripts/data/extract_vocalset.py` to `data/vocalset_full/`.
- For experiment 08 only: the English part of GTSinger (https://huggingface.co/datasets/AaronZ345/GTSinger); the phrases used are listed in `data/exp08_manifest.json` with checksums in `data/gtsinger_exp08_sha256.json`. The comparison by phoneme label (`scripts/steer/exp08_phonemes.py`) also reads the GTSinger annotation file (`.json`) of each breathy phrase and the outputs of experiment 08; the checksums of the annotation files are in `results/exp08/phonemes.json`.
- Python 3.10 with PyTorch and torchaudio (CUDA), plus the packages in `configs/requirements_seed_overlay.txt`; the full environment is in `configs/environment_audio_overlay.freeze.txt`. The listening analysis also needs SciPy, pandas and statsmodels.
- All experiments ran on one laptop GPU with 8 GB of memory (RTX 3070 Laptop). Training takes about 78 min.

Paths in the scripts are relative to the repository root. The batch files in `launchers_windows/` show the exact commands; on other systems, run the same Python commands from the repository root.

The analyses and figures are computed from the result files alone (NumPy and Matplotlib; SciPy is needed only for one p value). From the repository root, run in this order:

```
python scripts/steer/analyse_exp06.py
python scripts/steer/analyse_exp06.py --exact
python scripts/steer/compare_exp06_bootstrap.py
python scripts/steer/analyse_exp07_08.py
python scripts/figures/fig23.py
python scripts/figures/fig4.py
```

The intervals come from 2000 cluster resamples with a fixed seed, as reported in the paper; for experiment 06, `--exact` enumerates all 3125 resamples of the five singers instead and writes to `results/exp06/analysis_exact/`, for comparison only. Panels (a) and (b) of Figure 4 need the individual listening responses, which are not included (see below), so `fig4.py` reads them from `figures/figure4_values.json`; panel (c) is read from `results/exp06/analysis/analysis.json`.

## Listening test

- `listening_test/page/START.html` is the test page that all 20 listeners took (in Chinese), with its 64 audio files in `listening_test/page/audio/`. It runs offline in a browser, randomises the order of the trials within each part and the A/B assignment of the Part II trials separately for each listener, stores progress in the browser so that an interrupted session can be resumed, and produces a result string (`SVCLT1|...`); nothing is sent anywhere. It was built by `scripts/listen/build_listening_package.py` from `scripts/listen/listening_main_template.html`. The builder writes the page to `listening_test/` and the answer key to `results/listening_main/key.json`; in this package they are in `listening_test/page/` and at `listening_test/key.json`.
- `listening_test/stimuli_labelled.html` lists all 53 trials with their conditions, so each stimulus can be heard with its label.
- `listening_test/key.json` maps each trial and file to its condition and source file.
- `listening_test/results/` holds aggregated results only: answer counts per stimulus and question (`per_stimulus_counts.csv`) and the per-condition statistics reported in the paper (`summary_aggregated.json`, `report.md`), computed with `scripts/listen/analyse_listening.py`. Listeners were told that results would be published only in anonymous, aggregated form, so individual responses are not included here.

The preregistration (OSF Registries, osf.io/mky35) is copied in `preregistration/`.

## Licences

- Code: GNU General Public License v3.0 (see `LICENSE`), because it builds on and imports Seed-VC, which is released under GPL-3.0.
- Adapter weights, listening-test audio and result files: Creative Commons Attribution 4.0 (CC BY 4.0). They are derived from VocalSet, which is released under CC BY 4.0 by J. Wilkins, P. Seetharaman, A. Wahl and B. Pardo, "VocalSet: A Singing Voice Dataset," ISMIR 2018. The adapter is used together with the Seed-VC checkpoint, which keeps its own licence.
- The metrics of experiment 08 are computed on GTSinger, which is released under CC BY-NC-SA 4.0; no GTSinger audio is included here.

## Checksums

To verify the [v1.0.0 ZIP archived on Zenodo](https://doi.org/10.5281/zenodo.24556806), download the [SHA-256 checksum file for the archived release](https://github.com/ShadowbanUser/svc-technique-control/releases/download/v1.0.0/SHA256SUMS-zenodo-v1.0.0.txt). It covers the same 158 files listed in the original `SHA256SUMS.txt`.

The original checksums were calculated from a Windows working copy. For 41 text files, the working copy used CRLF line endings and the archive uses LF. This accounts for all 41 mismatches. The trained adapter and listening-test audio match their original checksums.

Extract the Zenodo ZIP and place the downloaded checksum file in the folder containing `README.md`. From that folder, run the following command in a terminal with `sha256sum` (for example, Git Bash on Windows):

```sh
sha256sum --check SHA256SUMS-zenodo-v1.0.0.txt
```

All 158 entries should report `OK`.

## Contact

Xurui Zhang, International School of Information Science & Engineering, Dalian University of Technology, Dalian, China. user_06@foxmail.com
