# ClaudeEEG

A Claude Code skill that turns Claude into a domain expert for EEG/MEG analysis, BCI development, and neuroscience signal processing — from raw recordings to production-grade models.

---

## What This Skill Does

ClaudeEEG loads a comprehensive neuroscience knowledge base into Claude's context, covering the full EEG analysis stack:

| Domain | Coverage |
|---|---|
| **Theory** | Frequency band biology, ERP components, cross-frequency coupling, volume conduction, artifact physics |
| **Preprocessing** | Gold-standard pipeline ordering — filtering, ICA, ASR, AutoReject, re-referencing |
| **Libraries** | MNE-Python, EEGLAB/MATLAB, SciPy, scikit-learn, braindecode, pyRiemann |
| **ML / DL** | CSP+LDA, Riemannian geometry, EEGNet, ShallowConvNet, foundation models (LaBraM, BIOT, SignalJEPA) |
| **Statistics** | Permutation tests, FDR/Bonferroni correction, cluster-based inference, mixed-effects models |
| **BCI Paradigms** | Motor Imagery, P300, SSVEP, neurofeedback |
| **Clinical** | Sleep staging, seizure detection, resting-state connectivity |
| **Data Formats** | EDF, BrainVision, EEGLAB `.set`, FIF, CNT, EGI/MFF, BIDS |

---

## Installation

```bash
npx skills add https://github.com/Krish-mal15/ClaudeEEG
```

Once installed, the skill is available in any Claude Code session. Invoke it explicitly or just use a trigger keyword in your message:

```
/ClaudeEEG help me preprocess this EEG dataset
```

---

## Example Prompts

### Preprocessing & Cleaning

```plaintext
"I have 64-channel EEG from a motor imagery task. Walk me through a full preprocessing pipeline."
"My data has heavy ocular artifacts. How should I set up ICA in MNE?"
"Should I use ASR before or after ICA for a mobile EEG recording?"
"Help me implement the dual-dataset ICA trick to preserve slow ERPs."
```

### ERP Analysis

```plaintext
"Extract and plot P300 components from my oddball task data."
"What baseline correction window should I use for a CNV paradigm?"
"My N200 amplitudes look wrong after filtering — what's going on?"
```

### BCI Development

```plaintext
"Build a real-time motor imagery classifier using CSP + LDA."
"Compare Riemannian MDM vs EEGNet for cross-subject motor imagery — which should I use?"
"Set up a P300 speller pipeline with xDAWN spatial filtering."
"Implement an SSVEP decoder using CCA for a 4-target paradigm."
```

### Machine Learning & Deep Learning

```plaintext
"Train EEGNet on BCI Competition IV 2a using braindecode."
"My classifier has 95% within-session accuracy but drops to 60% cross-session. How do I fix this?"
"Fine-tune LaBraM on my small dataset."
"Set up proper leave-one-subject-out cross-validation to avoid data leakage."
```

### Time-Frequency & Connectivity

```plaintext
"Compute alpha ERD/ERS during motor preparation using Morlet wavelets."
"Measure theta-gamma phase-amplitude coupling in my working memory data."
"Calculate wPLI connectivity between frontal and parietal channels."
```

### Sleep & Clinical

```plaintext
"Stage sleep from polysomnography data using AASM criteria."
"Detect interictal epileptiform discharges in a long EEG recording."
"Compute individual alpha frequency (IAF) per subject."
```

### Debugging & QC

```plaintext
"My ICA components all look like noise — what went wrong?"
"Walk me through interpreting these ICLabel labels and probabilities."
"Why does my source localization look wrong?"
"Critique this preprocessing pipeline and flag any ordering errors."
```

---

## Requirements

```bash
pip install mne mne-icalabel braindecode pyriemann autoreject meegkit scikit-learn torch
```

| Package | Purpose |
|---|---|
| `mne >= 1.6` | Core EEG/MEG processing |
| `mne-icalabel` | Automated IC classification |
| `braindecode` | Deep learning for EEG |
| `pyriemann` | Riemannian geometry classifiers |
| `autoreject` | Automated epoch rejection |
| `meegkit` | Zapline / DSS line-noise removal |
| `scipy`, `scikit-learn` | Signal processing and ML |
| `torch` | PyTorch for DL models |

---

## Repo Structure

```
ClaudeEEG/
├── SKILL.md          # The skill itself — loaded into Claude's context
├── README.md         # This file
├── scripts/          # Runnable Python pipeline examples
└── references/
    ├── MNE Resources/    # MNE tutorials (preprocessing, epochs, inverse, stats)
    └── NumPy Resources/  # NumPy reference docs
```

---

## Built-in Guardrails

These behaviors are baked into the skill and activate automatically:

- **Gamma skepticism.** Scalp gamma (30–80 Hz) fully overlaps the muscle artifact range. The skill flags uncritical gamma claims and requires rigorous EMG rejection before accepting gamma findings.
- **ICA ordering enforcement.** The skill warns if you filter after epoching, run ICA on unfiltered data, or skip ASR for mobile recordings.
- **CV hygiene.** ML pipelines always use subject-stratified splits. Random-shuffle splits on sliding windows are flagged as data leakage.
- **2024 literature.** Includes the Kang et al. finding (ICA may hurt DL decoding) and the Callan et al. ASR-before-ICA result for mobile EEG.

---

## Trigger Keywords

The skill activates on any of: `EEG` · `MEG` · `electrophysiology` · `BCI` · `brain-computer interface` · `ICA` · `ERP` · `epochs` · `artifact removal` · `EEGLAB` · `MNE` · `braindecode` · `NeuroPype` · `neural decoding` · `oscillations` · `frequency bands` · `power spectral density` · `source localization` · `motor imagery` · `P300` · `SSVEP` · `neurofeedback` · `neuro`
