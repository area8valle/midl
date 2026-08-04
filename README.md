# Mechanistically Informed Deep Learning for Osteoarthritis Progression

MI-DL combines knee radiographs, fourteen clinical variables, and a differentiable Standard Linear Solid cartilage model to estimate structural progression and time to total knee replacement. The implementation follows the reported EfficientNet-B4 imaging branch, 128–64–64 clinical encoder, patient-specific viscoelastic parameter head, 16-step loading trajectory, cross-attention fusion, weighted progression objective, and Cox partial-likelihood objective.

## Signal path

```text
radiograph -> EfficientNet-B4 -> imaging state -----------+
                              -> loading surrogate -> SLS | -> fusion -> progression score
clinical vector -> clinical encoder -> clinical state ----+           -> TKR risk
```

The constitutive state contains relaxed modulus, stress-relaxation time, strain-retardation time, and terminal strain. Positive transforms enforce relaxed modulus above zero and strain-retardation time above stress-relaxation time.

## Environment

The recorded environment uses Python 3.10, PyTorch 2.1, torchvision 0.16, CUDA 12.1, gin-config 0.5, NumPy 1.26, and SciPy 1.11.

```bash
conda env create -f environment.yml
conda activate midl
pip install -e . --no-deps
docker build -t midl .
```

## Cohorts

OAI version 12.0 supplies the development cohort of 4,796 participants. Participant-level splits are stratified by baseline Kellgren–Lawrence grade at 70% training, 15% validation, and 15% internal evaluation. MOST supplies the independent external cohort of 3,026 participants. Access is governed by each cohort provider and the source data are not redistributed here. Verified dataset locations are listed only in `DATASETS.txt`.

Prepared inputs contain radiograph paths, fourteen clinical columns, participant identifiers, knee side, baseline KL grade, progression label, TKR event indicator, and follow-up time. Preparation validates participant separation and fits clinical medians on the training partition.

```bash
./scripts/prepare_data.sh raw_oai prepared/oai
```

## Training

The principal configuration records AdamW with learning rate `1e-4`, weight decay `1e-2`, five warmup epochs, cosine decay to `1e-6`, batch size 32, gradient clipping at 1.0, at most 100 epochs, and early stopping patience 15. Loss weights are 0.5 for TKR, 0.1 for the physics residual, and 0.01 for parameter regularization.

```bash
midl-train --gin_file configs/experiment/main.gin
```

The reported study uses ten independent seeds and five-fold development-set cross-validation for hyperparameter selection. A final fit uses one NVIDIA A100 40GB GPU, about 11.4GB peak device memory, and approximately 9.2 hours.

## Experiment controls

```bash
midl-train --gin_file configs/experiment/ablation_no_phys_loss.gin
midl-train --gin_file configs/experiment/ablation_no_clinical.gin
midl-train --gin_file configs/experiment/ablation_no_dbm_fc.gin
midl-train --gin_file configs/experiment/ablation_no_all_physics.gin
midl-train --gin_file configs/experiment/supp_disentangle_polynomial.gin
midl-train --gin_file configs/experiment/supp_disentangle_windkessel.gin
```

Additional configurations cover imaging-only input, physics-loss sensitivity, pseudo-time resolution, sample efficiency, and the fully connected mechanism control.

## Evaluation

```bash
midl-evaluate --gin_file configs/experiment/main.gin --split oai_test
midl-evaluate --gin_file configs/experiment/main.gin --gin_file configs/data/most.gin --split most
```

The primary reported values are progression AUROC `0.781 ± 0.008` on OAI and `0.738 ± 0.012` on MOST. Five-year TKR AUC is `0.893 ± 0.009` on OAI and `0.845 ± 0.014` on MOST. Statistical routines include stratified bootstrap confidence intervals, DeLong comparison, sensitivity and specificity at the Youden threshold, Brier score, calibration slope, Cohen's d, precision-recall AUC, net reclassification, integrated discrimination improvement, concordance index, and decision-curve net benefit.

## Data boundaries

The cohorts contain de-identified secondary research data. Local manifests, images, prepared tensors, run outputs, and fitted weights are ignored by version control. The OAI and MOST populations are predominantly White, so external validity beyond these cohorts must be established separately.
