**Paper–code consistency audit**  
Reviewed October 16, 2026 · Repository commit `dc0146eff41fb75f3bda4000d0e9caf9705091fa`

**Overall assessment: the paper and the code are only partially aligned.** The repository implements the main building blocks of KPCA-DA, but the available experiments do not faithfully implement the paper's evaluation protocol. Several differences can materially affect classification accuracy, distribution-alignment measurements, and biological interpretation. I would not treat this checkout as a verified reproduction of the reported results.

This conclusion concerns the supplied files. It does not establish how experiments outside this checkout were conducted or that every published number is incorrect.

**Scope and how to read this report**

I read both PDFs in [paper](../paper): the 17-page manuscript, titled *Nonlinear Domain Adaptation Enables Interpretable Integration of Diverse Transcriptomic Cohorts*, and the seven-page *Supplementary Material*. I reviewed all ten Python source files and all nine notebooks under [Scripts](../Scripts), including notebook source and stored text outputs. I also inspected the local input CSVs referenced by the experiments to check cohort sizes, labels, gene ordering, and sample overlap.

“Main p.” and “Supplement p.” mean PDF page numbers. Notebook cell numbers count **all cells, including Markdown**, starting at 1; they are not the displayed `In[...]` execution numbers. Stored notebook outputs are historical evidence, not proof that the current source ran successfully in order.

The audit combines source inspection, mathematical checks, and small executable demonstrations. I did **not** rerun the full breast-cancer training experiments, reproduce all tables, or independently validate the biological claims. The original environment is not specified in the repository. Audit dependencies were installed in an isolated temporary environment; the research source and data were not edited.

Severity indicates consequence: **Critical** findings undermine a central scientific comparison or interpretation; **High** findings materially change the stated method or prevent reproducibility; **Moderate** findings affect reporting or an auxiliary API.

**What matches**

| Paper requirement | Evidence in the code | Assessment |
|---|---|---|
| Apply KPCA to genes rather than patients (Main p. 11). | `DomainAdaptiveImageTransformer.fit` transposes samples × genes to genes × samples before calling KPCA-DA. | Matches. |
| Fit source and target KPCA independently (Main pp. 12–13; Supplement Note 1). | `kpcada.py`, lines 130–132, calls `fit_transform` separately on each domain. Reusing the estimator object refits it, so the decompositions are separate. | Matches the specified fitting procedure. |
| Use an RBF kernel. | `rbf_kernel` computes `exp(-gamma * squared_distance)`. | Matches. |
| Adapt only target coordinates, keeping source coordinates fixed (Main p. 14). | Only `Z_t` requires gradients and is passed to SGD. | Matches. |
| Compute alignment loss on the shared-gene subsets while retaining other genes. | `find_common_gene_indices` selects shared subsets; the full target embedding remains in the regularizer. | Matches this structural design. MMD does not require paired row ordering. |
| Compute automatic bandwidth once from pooled, initial shared-gene embeddings (Supplement Note 1). | `compute_median_gamma` implements the supplement's excerpt, including excluding zero distances; `auto_gamma=True` is the core API default. | Matches the supplement's actual formula and implementation excerpt. |
| Offer label-free lambda selection by final alignment loss (Supplement Note 2). | `tune_lambda` supports a label-free search and selects `best_lambda_mmd`. | Capability matches; experiment settings differ below. |
| Use the supplement's modified loss and trace term (Supplement Note 2, N2E1–N2E3). | `mmd_loss`, `regularization_term`, and their weighted sum implement those expressions. | Literal match to the supplement; scientific problems remain in F5–F6. |
| Learn an image layout and reuse it for new target samples (Main pp. 14–15). | `transform_target` reuses stored target coordinates; `test_kPCADA` cells 38–39 fit on the adaptation subset and map test intensities without refitting KPCA. | Matches this particular stage. Preprocessing and training still access test information. |
| Make one image per patient with expression encoded as intensity; use a ResNet-50 classifier. | Image construction loops over samples, repeats scalar intensities into RGB, and notebooks instantiate pretrained `resnet50d`. | Broad architectural match; exact ResNet variant should be documented. |
| PAM50 cohorts contain 166 SMC and 117 CPTAC patients, with four subtypes and no Normal-like class (Main p. 3). | The active `smc_labels.csv` and `cptac_labels.csv` contain exactly those counts and four labels. | Matches the local files used by the notebook. |

**Most consequential findings**

| ID | Severity | Finding | Main consequence |
|---|---|---|---|
| F1 | Critical | Target-test data and labels influence preprocessing and classifier selection. | The stated independent held-out evaluation is not implemented. |
| F2 | Critical | A notebook trains on target adaptation labels; another oversamples before splitting. | The experiment differs from unsupervised adaptation and includes repeated patients across partitions. |
| F3 | Critical | Baseline and adapted classifiers receive different training settings. | Accuracy gains cannot be attributed to adaptation alone. |
| F4 | Critical | Post-hoc target shifts enter reported alignment distances. | Distribution measurements do not isolate KPCA-DA. |
| F5 | Critical | Implemented “MMD” matches the supplement but differs from the main paper and conventional MMD. | Lower reported loss can mean worse distribution agreement. |
| F6 | High | The positive trace penalty is minimized despite being described as variance maximization. | The stated preservation mechanism is not established by the objective. |
| F7 | High | Source and target coordinates are independently rescaled into pixels. | The shared spatial interpretation is not guaranteed. |
| F8 | High | Grad-CAM indices are assigned incorrect gene names; the adapted attribution pipeline is incomplete. | Biomarker recovery cannot be verified reliably. |
| F9 | High | ComBat uses misordered genes and applies the source batch ID to target-test samples. | The comparator can be incorrectly corrected. |
| F10 | High | Active data counts differ from the paper for two tasks. | The same experimental population is not established. |
| F11 | High | Experiment hyperparameters differ from the supplementary selection protocol. | The reported selected configurations are not reproduced. |
| F12 | High | CAN, ablation, repeated-run statistics, and external validation workflows are missing. | Major reported results lack a reproducible path in `Scripts`. |
| F13 | Moderate | Metric code prints stale recall values and mixes metric definitions. | Reported comparisons can contain incorrect or incomparable metrics. |
| F14 | Moderate | KPCA equations and the standalone transform API differ from implemented behavior. | The mathematical specification and API need clarification. |
| F15 | High | Environment paths, undefined variables, and mixed notebook state impede clean execution. | The saved experiments cannot be accepted as reproducible runs. |

**F1 — The target-test set is used during model development**

The paper says that the 30% target-test subset is excluded from alignment and training and used exclusively for final evaluation (Main pp. 14–15; Supplement pp. 1–3).

The code initially makes a 70/30 split, but then:

- [test_kPCADA.ipynb](../Scripts/test_kPCADA.ipynb), cell 6, computes a variance filter on `X_test` and fits a new MinMax scaler on it. Cell 10 fits a target image layout on the test samples.
- Its adapted branch, cell 36, again fits a scaler on `X_test` separately from the adaptation set. Thus a test patient's normalized expression depends on the other test patients.
- Cell 19 uses test accuracy for early stopping. Cell 46 uses it for learning-rate scheduling, checkpoint selection, and early stopping.
- [claude_code.ipynb](../Scripts/claude_code.ipynb), cells 7–8, 15, 31, and 41, and [claude_code_grad_cam.ipynb](../Scripts/claude_code_grad_cam.ipynb), cells 6–7, 11, 28, and 38, contain corresponding uses.

Using unlabeled test measurements to fit preprocessing already differs from the claimed deployment protocol. Using their labels to select the model is a stronger breach: the reported test score is no longer an independent final assessment.

**Correction:** Split original patients first, fit preprocessing only on permitted training/adaptation data, and use a source validation partition for classifier selection. Evaluate target-test once after all choices are frozen. For differing gene sets, specify how source-derived scaling is extended to target-only genes; the supplement does not resolve that detail.

**F2 — Some experiments use target labels or repeated patients**

The main paper calls the framework fully unsupervised and says source images train the classifier (Main pp. 3 and 15).

In `claude_code.ipynb`, cells 40–41 combine source labels with `y_adapt_enc`, then train the classifier on both source and target adaptation images. That is supervised use of target labels. `y_adapt_enc` is also undefined in the saved source, so the branch requires repair before it can run. The core KPCA optimizer itself remains label-free; this mismatch is in the downstream experiment.

Cells 3–4 of that notebook also oversample the labeled target cohort **before** splitting it. Replaying those exact operations on the active Desmedt labels, while tracking original sample identities, gives:

| Quantity | Audit result |
|---|---:|
| Original target samples | 162 |
| Rows after oversampling | 198 |
| Adaptation / test rows | 138 / 60 |
| Original sample identities appearing in both partitions | 14 |
| Test rows whose original sample also appears in adaptation | 18 of 60 |

These are repeated original sample identities, not merely similar expression profiles. The test set is not independent of the adaptation cohort.

The additional [cf_rna_adaptation.ipynb](../Scripts/cf_rna_adaptation.ipynb) similarly augments target samples before splitting (cells 9–10) and trains with `Y2_adapt` (cell 20). This notebook is outside the manuscript's breast-cancer benchmark; it should be labeled as an extension, not used as evidence for the manuscript's unsupervised protocol.

**Correction:** Keep target labels out of optimization and classifier training for the unsupervised experiment. Split by original patient identity before resampling or augmentation; confine any permitted augmentation to training.

**F3 — The baseline comparison changes more than domain adaptation**

The paper says competing methods use identical image-conversion and classification pipelines (Main p. 15). In `test_kPCADA.ipynb`, the baseline and adapted classifiers differ substantially:

| Setting | Baseline, cells 17–19 | KPCA-DA, cells 44–46 |
|---|---|---|
| Optimizer | SGD | AdamW |
| Learning rate | 0.0005 | 0.0001 |
| Momentum | 0.5 | Not applicable |
| Batch size | 8 | 16 |
| Maximum epochs | 25 | 100 |
| Early-stopping patience | 4 | 10 |
| Random horizontal flip | Enabled | Absent |
| Scheduler signal | Training loss | Target-test accuracy |

The source comments explicitly describe the baseline settings as suboptimal. The baseline's random flip is also applied while constructing test tensors because the same preprocessing function is used for both loaders. Flipping a gene image moves expression away from its learned spatial positions.

Similar differences occur in `claude_code.ipynb` and `claude_code_grad_cam.ipynb`. Consequently, an accuracy difference combines adaptation, optimization, training budget, preprocessing, and model-selection effects.

**Correction:** Use the same classifier configuration, validation rule, initialization/seed policy, and deterministic test preprocessing for every comparator. Change only the adaptation method.

**F4 — Alignment distances include an extra, undisclosed coordinate adjustment**

The paper describes fitting ICA to the source and applying its projection to target-test samples (Main p. 6), then measuring the resulting domain discrepancy.

In `test_kPCADA.ipynb`, cell 41 instead fits ICA jointly on source and target-test images. It then shifts every target ICA point 80% of the way toward the source centroid:

```python
shift_factor = 0.80
shift_vector = (centroid_source_initial - centroid_test_initial) * shift_factor
test_ica = test_ica + shift_vector  # via test_ica_aligned in the notebook
```

Cell 42 calculates Wasserstein distances from this **shifted** `test_ica`. Its comment describes the points as non-aligned, but the variable has already been replaced. Cells 57–58 likewise apply a 10% centroid shift to ComBat's target ICA points before measuring distances. `claude_code.ipynb`, cells 36–37 and 53–54, and `claude_code_grad_cam.ipynb`, cells 33–34 and 49–50, repeat these operations.

An 80% shift guarantees an 80% reduction in centroid distance regardless of KPCA-DA quality. The stored output shows the distance changing from **2.3384 to 0.4677**. It then reports **WD = 0.7010**. These match Figure 3's adapted annotations, approximately **0.468** and **0.7010** (Main p. 7). This is concrete evidence that the available figure-associated measurements include the post-hoc adjustment.

The extra translation changes the measurement arrays, not the classifier's image tensors. This finding directly concerns distribution plots and distances; it does not by itself explain classification accuracy.

Additional measurement problems: [ICAperformance.ipynb](../Scripts/ICAperformance.ipynb), cells 5–6, independently refits ICA to each domain, producing incomparable axes. [UMAPperformance.ipynb](../Scripts/UMAPperformance.ipynb), cell 10, compares each coordinate array with itself, then prints a combined distance computed from literal constants. Some notebooks average two one-dimensional Wasserstein distances; others take their Euclidean norm. These are different summaries and should not be presented interchangeably as a single defined 2D metric.

**Correction:** Remove post-hoc translations and compute distances from untouched embeddings. Follow the stated ICA fitting protocol consistently, specify the exact distance summary, and regenerate Figure 3 and Table 3 from saved raw outputs.

**F5 — The code's loss is not the MMD defined in the main paper**

Main p. 13, Eq. 15, defines:

```text
MMD² = mean(K_ss) + mean(K_tt) - 2 mean(K_st)
```

[kpcada.py](../Scripts/kpcada.py#L75), lines 75–82, implements:

```text
loss = abs(mean(K_tt) - 2 mean(K_st))
```

This **does match Supplement Note 2, N2E2**. The discrepancy is between the main manuscript and the supplement/code, not an undocumented code-only change.

Dropping the constant source term would preserve gradients **without** the absolute value. Adding the absolute value can reverse the gradient when the enclosed expression is negative. A direct check using the repository function illustrates the consequence:

| One source point and one target point, gamma = 1 | Conventional MMD² | Code loss |
|---|---:|---:|
| Identical points | 0 | 1 |
| Points separated by `sqrt(log(2))` | 1 | Approximately 0 |

Here the implementation prefers unequal distributions to identical ones. Therefore the lowest `final_mmd` in the lambda search does not necessarily mean the tightest distributional alignment asserted in the supplement.

**Correction:** Reconcile the scientific objective first. If conventional MMD is intended, implement its full expression or optimize the constant-free expression without this absolute value; separately report full MMD². Repeat tuning and affected experiments afterward.

**F6 — Minimizing the trace term does not establish variance preservation**

Main p. 14, Eqs. 17–22, and Supplement Note 2 describe variance preservation/maximization while minimizing a loss containing a **positive** `lambda * trace(Z_t.T @ K_t @ Z_t)`. The code implements that sign literally ([kpcada.py](../Scripts/kpcada.py#L85), lines 85–88 and 160–167).

For a positive-semidefinite RBF kernel matrix, this is a nonnegative quadratic penalty with minimum at zero. In isolation, it penalizes embedding magnitude; it is not a constraint keeping the adapted embedding near its original coordinates. When `K_t` is close to identity, it is approximately a squared-norm penalty.

A check using the actual regularizer reduced target embedding variance from **1.00 to 0.64** after one gradient-descent step. This demonstrates the sign issue; it does not prove that every complete run collapses, because the alignment term also contributes.

This is a mismatch between the **claimed mechanism and the equations/code**, even though the equation and implementation agree. Simply reversing the sign is not a complete repair because it can make an unconstrained objective unbounded.

**Correction:** Specify and justify a bounded structural-preservation objective, then validate retained variance and biological structure directly. Regenerate the VML ablation after reconciling the objective.

**F7 — Pixel mapping can discard spatial alignment and does not enforce shared-gene positions**

The paper describes a fixed gene-to-pixel arrangement supporting transfer across training and deployment (Main pp. 14–15).

[image_transformer_new.py](../Scripts/pyDeepInsightv3/image_transformer_new.py#L384), lines 384–394, scales source and target coordinates separately to their own minimum and maximum before discretization. Although the rotation is shared, the final scaling bounds are not. Neither this function nor image construction enforces identical pixel positions for a shared gene.

A direct call to the current coordinate methods maps a point cloud and a translated, twice-as-large version of that cloud to **identical pixels**, despite an embedding centroid distance of **22.864**. Thus translation and axis scaling learned in the embedding can disappear during image construction. Matching embedding distributions also does not establish that a particular named gene occupies the same pixel in both domains.

The code correctly preserves the target layout across future target samples. That narrower guarantee should be distinguished from preserving gene identity across source and target images. Adaptation changes spatial layout; image intensities remain normalized original expression values rather than corrected expression measurements.

**Correction:** Use an explicitly defined common coordinate-to-pixel mapping and assess named shared-gene correspondence. Document whether the intended output is an adapted spatial layout or a harmonized expression matrix.

**F8 — Gene attribution is not consistently mapped back to the right genes**

The main manuscript selects the top ten genes per subtype by mean target Grad-CAM score (Main pp. 7–8). Supplement Note 4 instead describes thresholding averaged training-sample maps at 0.5.

The available workflows have several concrete problems:

- `test_kPCADA.ipynb`, cells 21–24, computes CAM indices using the **source** transformer `it1`. Cells 29–30 then index **target** `genes2` with those indices. Those indices do not represent target gene identities.
- In all three main experiment notebooks, baseline image construction filters genes, but attribution indexes the original unfiltered `genes1` or `genes2` arrays. A selected image-feature index must be mapped through the same variance-filter mask before naming the gene.
- `claude_code_grad_cam.ipynb`, cells 14–17, does use target images and `it2`, but uses threshold **0.4**, and still does not apply the target variance mask when recovering gene names.
- The notebooks select threshold-passing features rather than sort gene scores and retain ten per subtype. A preview slice such as `[1:10]` is nine entries in existing order, not a top-ten ranking.
- [feature_selection.py](../Scripts/pyDeepInsightv3/feature_selection.py#L43) expects `it.coords()`. The adaptive transformer provides `coords_source()` and `coords_target()` instead. No complete post-adaptation attribution workflow bridging this interface appears in these notebooks; their CAM sections precede adaptation.
- The helper averages sample CAMs but does not explicitly renormalize the resulting class-average map as Supplement Note 4 specifies. Its library default threshold is 0.6, although some notebook calls override it to 0.5.

Consequently, the supplied code does not reliably establish the identities or rankings behind the claimed recovered biomarkers. This is not an independent judgment about those genes' biology.

**Correction:** Preserve the ordered gene list alongside every fitted map, keep filter indices, explicitly select the source or target map, and implement one documented ranking/threshold rule. Regenerate the gene lists and overlap analyses from a complete adapted-model attribution run.

**F9 — ComBat gene order and test batch identity are incorrect in the integrated branches**

The paper requires matching gene features for ComBat and applying learned correction parameters to target-test samples (Main pp. 4 and 15).

`test_kPCADA.ipynb`, cells 50–52, forms a sorted shared-gene set but extracts indices by scanning each original gene list independently. This preserves each file's original order rather than imposing one common order. `claude_code.ipynb`, cells 46–48, and `claude_code_grad_cam.ipynb`, cells 42–44, do the same.

Checking the active CSVs gives:

| Cohort pair | Shared gene rows | Rows with matching names at the same position |
|---|---:|---:|
| TCGA tumor/normal / GEO | 228 | 220 |
| TCGA survival / Desmedt | 11,254 | 0 |
| SMC / CPTAC | 17,484 | 5,559 |

For example, the survival branch would pair source `A1CF` with target `DDR1`. ComBat's combined matrix therefore mixes different genes in a row. This issue does not apply to the order-insensitive MMD distribution calculation. `CombatFit.ipynb`, cell 8, correctly uses ordered `.loc[common_genes]` selection, so not every ComBat branch has the ordering defect.

Separately, the integrated notebooks identify source as batch **1** and adaptation target as batch **2**, but pass batch **1** for target-test samples to `neuroCombatFromTraining`. `CombatFit.ipynb` repeats this wrong test batch ID and uses a **20%** test split rather than 30%.

**Correction:** Select both matrices by one identical ordered gene list; assert equality before combining. Apply target batch 2 parameters to target-test samples. Reevaluate the comparator after both corrections.

**F10 — Active datasets differ from the paper's reported cohorts**

The counts below come from the files actively loaded by notebook cell 2, using the notebooks' convention that the last row contains labels:

| Task and cohort | Paper sample count | Active local sample count | Assessment |
|---|---:|---:|---|
| Survival: TCGA | 160 | 154 | Differs |
| Survival: Desmedt | 164 | 162 | Differs |
| Tumor/normal: TCGA | 592 | 250 | Differs substantially |
| Tumor/normal: GEO | 224 | 183 | Differs |
| PAM50: SMC | 166 | 166 | Matches |
| PAM50: CPTAC | 117 | 117 | Matches |

The survival inputs contain **20,502** and **12,587** gene rows, versus the manuscript's approximately 18,000 and 9,000 features. The active GEO tumor/normal input contains only **236** gene rows. TCGA survival also contains one duplicated gene name.

Files already contain `long`/`short` labels. No code in `Scripts` establishes the described DSS thresholds, death-status handling, or exclusions. Likewise, the claimed Desmedt BioMart probe-to-HGNC conversion is not implemented there. Counts alone cannot determine whether undocumented exclusions were justified.

**Correction:** Provide patient IDs, cohort construction and exclusion logic, gene/probe mapping, and a manifest identifying the exact files for every reported result. Reconcile all pre/post-filter counts in the manuscript.

**F11 — Preprocessing and hyperparameter settings do not reproduce the supplement**

Supplement Note 3 says to discard the lowest **30%** of genes by training variance and fit min–max scaling on source only. The main notebooks' baseline branches use the **10th percentile**, retain approximately 90%, and fit target-test preprocessing independently. Their adapted branches disable variance filtering entirely. The cfRNA extension's `top70_var` actually uses the **50th percentile** despite its name and comment.

Supplement Note 2 specifies six lambdas, 500 SGD steps, and learning rate 0.001. For TCGA → GEO it reports gamma 965.55 and lambda 0.001; for TCGA → Desmedt, gamma 20.44 and lambda 0.0001.

`test_kPCADA.ipynb`, cell 38, supplies 100 steps and `kpca_gamma=0.2` without a lambda search, inheriting lambda **0.01** and learning rate **0.0001**. `claude_code_grad_cam.ipynb`, cell 30, uses the same configuration. These do not implement the stated selection procedure.

`claude_code.ipynb`, cell 32, does perform a 500-step, label-free search, but omits lambda **10.0**. Cell 33 then overwrites the selected values with `auto_gamma=False`, `mmd_gamma=0.5`, `lambda_reg=1.0`, and only 100 steps. It does not consume the search result for fitting.

The reusable module supports the intended automatic gamma calculation, but that does not establish that the experiments used the supplement's selected values. The fixed gamma values themselves were not recomputed on the full datasets during this audit.

**Correction:** Save and pass the actual tuning result into the final fit, using the same input matrices and declared settings. Report every task/direction's KPCA gamma separately from its MMD gamma.

**F12 — Several central results have no reproducible implementation in `Scripts`**

I found no complete workflow for the following manuscript claims:

| Claim | What is available / missing |
|---|---|
| CAN training on target adaptation images, frozen for target-test inference (Main p. 15). | Metric notebooks read externally generated CAN prediction CSVs; no CAN model, contrastive training loop, configuration, or checkpoint workflow is present. |
| Four component ablations in Table 2 (Main p. 6). | Lambda zero can remove regularization, but no reproducible runner implements all four variants or produces Table 2. |
| Means and variability in Tables 1–3, and pairwise Wilcoxon p-values 0.03–0.035 (Main pp. 4–6). | No repeated-run aggregation or Wilcoxon calculation is present. Some ROC legends contain literal `±0.02`, `±0.03`, etc. These literals are not estimates computed by those cells. |
| Top-ten-per-subtype, TCGA-DE/iPathwayGuide validation and comparative Venn analyses (Main pp. 7–10). | Partial CAM selection and hardcoded subtype-associated gene lists exist, but no complete provenance-preserving workflow reconstructs the external benchmark and reported comparisons. |
| Six reciprocal task/direction experiments. | Paths can be edited manually, but there is no single configuration/run manifest tying all directions to inputs, seeds, predictions, and manuscript rows. |

Missing implementation means **unverifiable from this checkout**, not proof that the external experiment was never performed.

**Correction:** Supply executable runners, exact configurations, run-level outputs, external benchmark provenance, and a table/figure generation script. Explain the unit of replication and what each error bar represents.

**F13 — Metric reporting contains stale values and inconsistent definitions**

[scores_test.ipynb](../Scripts/scores_test.ipynb), for example cells 9, 12, 28, 31, 44, and 47, computes a model's recall into `Sensitivity` but prints the older variable `recall` from another model. Thus the displayed sensitivity can belong to KPCA-DA while the other metrics belong to ComBat or CAN.

The training notebooks calculate weighted recall and mean one-vs-rest specificity, whereas `scores_test.ipynb` uses positive-class recall and negative-class recall for binary tasks. Weighted recall in ordinary single-label classification equals accuracy, which is not the same quantity as positive-class sensitivity. The paper needs a consistent definition for each task.

In `claude_code.ipynb`, cell 41 stores adapted results as `preds`/`trues`, but cell 43 computes metrics from earlier `test_predicted`/`test_true`. If executed in order after the baseline, it can report baseline metrics as the adapted branch's metrics.

There is also a checkpoint bug in `test_kPCADA.ipynb`, cell 46, and `claude_code_grad_cam.ipynb`, cell 38: `net.state_dict().copy()` does not clone tensor storage. Later updates can overwrite the supposed best checkpoint. Stored `test_kPCADA` output reports “best” accuracy **0.653**, then final accuracy **0.612** after restoration, consistent with that problem.

**Correction:** Compute each method's metrics from explicitly named prediction records, define averaging/positive classes, and clone checkpoint tensors or use a deep copy. Compute uncertainty from run-level results.

**F14 — The KPCA equations and standalone transform API need reconciliation**

Main pp. 12–13, Eqs. 7–8, display `V * Lambda^(-1/2)` as the reduced representation. The module uses scikit-learn `KernelPCA.fit_transform`, whose returned training scores are `V * Lambda^(1/2)`. The inverse-square-root form can describe kernel-space coefficients, but it is not the returned score matrix as currently labeled. A small direct check confirms the implementation follows the latter convention.

The separate [KPCA_DA.transform](../Scripts/kpcada.py#L345) API also returns a plain target KPCA projection. The stored estimator was last fitted to target data, and this method does not apply the optimized `Z_t` update. Even transforming the original target input does not return the adapted target embedding. A direct check found a maximum difference of approximately **0.00778** on a small example. It also expects the target KPCA input feature count, not an arbitrary new patient expression vector.

The main image workflow uses `transform_target` with stored pixel coordinates, so this API issue is separate from its valid reuse of a learned target layout.

**Correction:** Distinguish KPCA coefficients, gene scores, and patient image conversion in the paper/API. Rename or implement the standalone transform behavior so users cannot mistake an unadapted projection for adapted inference.

**F15 — The saved notebooks do not constitute a clean, portable experiment**

There is no dependency specification, lockfile, or run guide. `kpcada.main` uses absolute Windows paths. Most notebooks use Windows path separators; `claude_code.ipynb`, cell 1, explicitly checks an absolute path to a different checkout and inserts it ahead of local imports. On a machine where that path exists, the notebook can audit/train a different version of the modules.

`claude_code.ipynb` references undefined `y_adapt_enc`. `test_kPCADA.ipynb`, cell 26, calls undefined `cam_image3`. Its optional cell 8 tries to apply a source-fitted UMAP to target expression vectors with a different number of genes. Execution counts and saved outputs are mixed, and some stored outputs show failures or interruption. These issues mean syntactically valid cells are not sufficient evidence of an executable experiment.

**Correction:** Resolve paths relative to the repository, import the local modules explicitly, pin dependencies, remove undefined/stale state, and demonstrate a fresh-kernel run producing all claimed artifacts.

**Additional inconsistencies within the paper package**

Some corrections are needed in the documents themselves, independently of implementation changes:

- Supplement Note 1 says too-small gamma makes `exp(-gamma*d²)` approach zero. For fixed distance, increasing gamma drives it toward zero; decreasing gamma drives it toward one.
- Its stated `gamma = 1 / median(d²)` gives kernel value `exp(-1) ≈ 0.368` at the median, rather than approximately 0.5. The implementation matches the equation.
- Supplement Table S3 is described as monotonically increasing in lambda, but its gamma-500 row decreases from 0.001675 to 0.001099 between its first two lambdas.
- The gamma-5000/lambda-0.0001 entry has MMD 0.003507 in Table S2 but total loss 0.000887 in Table S3. At one common endpoint, `MMD + nonnegative regularization` cannot be smaller than MMD. The current tuning code records total loss **before** the last update but recomputes MMD **after** it, which makes their endpoints inconsistent. This is a plausible explanation requiring original run logs; it does not establish the provenance of those particular entries.
- Main pp. 7–8 describe ranked target top-ten attribution; Supplement Note 4 describes thresholded training attribution. These are different procedures and should be reconciled.
- Figure 3's example values, 2.0371 and 0.7010, imply approximately **65.6%** reduction, although its caption says approximately 72%. Table 3's mean values, 2.086 and 0.584, do imply approximately **72.0%**. Example-run and aggregate reductions should be labeled separately.

**Verification evidence and practical next steps**

The audit artifacts are:

- [reproduce_checks.py](reproduce_checks.py): executable demonstrations using the existing core functions, plus data metadata/sample-identity checks and source parsing.
- [check_results.json](check_results.json): captured numerical results and audit environment versions.
- [notebook_evidence.json](notebook_evidence.json): exact selected cell source and stored text outputs supporting the findings.
- [source_inventory.json](source_inventory.json): reviewed source/PDF paths and SHA-256 hashes.

All ten Python files and code cells in all nine notebooks parse. This check establishes syntax only. The small numerical checks reproduce the loss counterexample, regularizer shrinkage, pixel-rescaling invariance, post-hoc centroid reduction, standalone transform discrepancy, KPCA score convention, mixed loss endpoints, patient overlap, cohort counts, and ComBat gene-order mismatch. They do not reproduce clinical performance estimates.

The most useful repair order is:

1. Freeze patient IDs and split original patients; remove target-test model selection and target-label training from the unsupervised experiment.
2. Reconcile the MMD objective and variance-preservation claim across the main paper, supplement, and code.
3. Correct pixel mapping, gene-name attribution, and ComBat gene ordering/batch identity.
4. Run every comparator with one shared classifier protocol; retain raw alignment measurements without extra translations.
5. Automate all tasks, ablations, seeds, metric aggregation, and attribution validation; regenerate the paper's tables and figures from those outputs.

The existing implementation is a useful starting point, especially its gene-oriented KPCA, target-only optimization, automatic-bandwidth helper, and reusable target image layout. The discrepancies above must be resolved before the reported accuracy gains, alignment reductions, and recovered biomarker lists can be accepted as reproduced by this codebase.
