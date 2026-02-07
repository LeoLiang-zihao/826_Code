# Task

## Assignment
- Course: BIOSTAT 826 - Deep Learning for Health
- Title: Coding Assignment 1 - Mortality prediction in MIMIC-IV

## Learning Objectives
- Manipulate and interpret structured EHR data.
- Define and train models in PyTorch.

## Prediction Task
- Predict risk of death during a hospital admission using diagnosis and procedure codes from the same admission.

## Deliverable
- Submit one Jupyter notebook converted to HTML, including code and figures for all assignment parts.

## Data Source
- Dataset location: `\mimic-iv-3.1`

## Part 1 - Prepare EHR Data
- [ ] Load `admissions`, `diagnoses_icd`, and `procedures_icd` from `hosp`.
- [ ] Keep ICD-10 only (drop ICD-9 records).
- [ ] Create diagnosis/procedure categories using first 3 code characters.
- [ ] Drop categories with frequency < 10.
- [ ] Drop death-proxy diagnosis categories:
  - [ ] `I46, I60, I61, I62, R09, R57, R96, R97, R98, R99, Z66, Z51, Z71, G93, J80, J96`
- [ ] Drop death-proxy procedure categories:
  - [ ] `5A1, 0BH, 0W9`
- [ ] Build admission-level binary features per category (`hadm_id`).
- [ ] Build labels from `hospital_expire_flag` in `admissions` and merge with features.
- [ ] Split into train/val/test = 60/20/20.
- [ ] Use memory-efficient approach for feature/label matrix construction.

## Part 2 - Logistic Regression (PyTorch)
- [ ] Implement logistic regression with `BCEWithLogitsLoss` (preferred) or sigmoid + `BCELoss`.
- [ ] Use `torch.optim.SGD` defaults (`lr=1e-3`).
- [ ] Train with minibatch size 200 until convergence.
- [ ] Repeat with much smaller and much larger learning rates.
- [ ] Plot minibatch loss curves for all learning rates on one figure.
- [ ] For one model, list top 20 diagnosis/procedure code categories increasing log-odds.
- [ ] For one model, list top 20 diagnosis/procedure code categories decreasing log-odds.
- [ ] Add text descriptions for listed codes.
- [ ] Add L1 regularization and retrain.
- [ ] Increase L1 strength until most (not all) coefficients are zero.
- [ ] Repeat top/bottom 20 log-odds code lists for L1 model.

## Part 3 - Neural Network Risk Predictor
- [ ] Build NN with one hidden layer (100 units, ReLU).
- [ ] Use BCE loss variant as in Part 2.
- [ ] Train with minibatch size 200.
- [ ] Evaluate validation loss regularly and stop when no longer improving.
- [ ] Keep checkpoint / rerun with fixed seed to recover best validation model.
- [ ] Repeat with hidden sizes 1k and 5k.
- [ ] Create training/validation plots for each hidden size.
- [ ] Downsample training set by 10x and repeat all above NN experiments.

## Part 4 - Performance Evaluation
- [ ] Select one logistic regression model and one neural network model.
- [ ] Plot ROC curves for both models on one figure.
- [ ] Plot PR curves for both models on one figure.
- [ ] Plot calibration curves for both models on one figure.
- [ ] Ensure publication-quality labels, legend, and readable text size.
- [ ] Compute AUROC and AP for each model.
- [ ] Compute calibration slope and intercept for each model.
- [ ] Choose a fixed sensitivity and report corresponding specificity for each model.
- [ ] Build 95% confidence intervals via 1000 bootstrap resamples on test set.

## Progress Checklist
- [ ] Data pipeline implemented
- [ ] Logistic regression experiments complete
- [ ] Neural network experiments complete
- [ ] Evaluation and bootstrap CIs complete
- [ ] Notebook converted to HTML and ready to submit
