**Paper–code consistency audit**  
Originally reviewed October 16, 2026 · Research-source commit `dc0146eff41fb75f3bda4000d0e9caf9705091fa`

**Reverified October 16, 2026** against checkout `e1f10260cbfde806d0ded89fb4edfc7d8b4d036a`: all 21 inventoried paper/source files have unchanged SHA-256 hashes. A fresh run of the executable audit checks reproduced the saved results, and the affected implementation paths were rechecked. The recommendations below are proposed repairs, not implemented changes or measured performance gains.

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

| ID | Severity | Finding | Main consequence | Recommended fix |
|---|---|---|---|---|
| F1 | Critical | Target-test data and labels influence preprocessing and classifier selection. | The stated independent held-out evaluation is not implemented. | Split patients first; freeze preprocessing from source training and permitted unlabeled adaptation data; select the classifier on source validation. |
| F2 | Critical | One notebook both trains on target adaptation labels and oversamples before splitting. | The experiment differs from unsupervised adaptation and includes repeated patients across partitions. | Train the unsupervised classifier on source labels only; split original patient IDs before any resampling or augmentation. |
| F3 | Critical | Baseline and adapted classifiers receive different training settings. | Accuracy gains cannot be attributed to adaptation alone. | Use a strong, shared classifier recipe and equal selection budgets; remove spatial flips from gene-image evaluation. |
| F4 | Critical | Post-hoc target shifts enter reported alignment distances. | Distribution measurements do not isolate KPCA-DA. | Remove post-hoc shifts and literal metrics; use a fixed, declared projection and distance definition; regenerate figures. |
| F5 | Critical | Implemented “MMD” matches the supplement but differs from the main paper and conventional MMD. | Lower reported loss can mean worse distribution agreement. | Use full empirical MMD², including the source term and without the absolute value; retune after correcting it. |
| F6 | High | The positive trace penalty is minimized despite being described as variance maximization. | The stated preservation mechanism is not established by the objective. | Evaluate a bounded penalty for movement from initial coordinates; rename the mechanism and ablate it before adoption. |
| F7 | High | Source and target coordinates are independently rescaled into pixels. | The shared spatial interpretation is not guaranteed. | Freeze one common pixel coordinate frame; separately test shared-gene orientation/anchoring and measure collisions. |
| F8 | High | Grad-CAM indices are assigned incorrect gene names; the adapted attribution pipeline is incomplete. | Biomarker recovery cannot be verified reliably. | Carry gene IDs and filter masks with each domain map; run attribution on the adapted classifier and implement the declared ranking. |
| F9 | High | ComBat uses misordered genes and applies the source batch ID to target-test samples. | The comparator can be incorrectly corrected. | Enforce one ordered, unique shared-gene list and apply the learned target-batch parameters to target-test samples. |
| F10 | High | Active data counts differ from the paper for two tasks. | The same experimental population is not established. | Recover exact cohort provenance or revise the reported cohort version; document every inclusion and gene-mapping decision. |
| F11 | High | Experiment hyperparameters differ from the supplementary selection protocol. | The reported selected configurations are not reproduced. | Consume the saved tuning result; hold initial embeddings and MMD bandwidth fixed across candidates; use label-free selection with structural checks. |
| F12 | High | CAN, ablation, repeated-run statistics, and external validation workflows are missing. | Major reported results lack a reproducible path in `Scripts`. | Build all six task/direction runs, genuine comparators and ablations; generate tables and uncertainty from saved predictions. |
| F13 | Moderate | Metric code prints stale recall values and mixes metric definitions. | Reported comparisons can contain incorrect or incomparable metrics. | Centralize metrics and label definitions; save actual best weights; derive uncertainty from documented repeated runs. |
| F14 | Moderate | KPCA equations and the standalone transform API differ from implemented behavior. | The mathematical specification and API need clarification. | Correct the score/coefficient notation; distinguish unadapted KPCA projection from frozen-layout patient inference. |
| F15 | High | Environment paths, undefined variables, and mixed notebook state impede clean execution. | The saved experiments cannot be accepted as reproducible runs. | Provide one local implementation, portable configuration and a tested dependency lock; verify execution from a fresh process. |

**How the recommendations protect performance**

The objective is credible performance on unseen patients. Removing leakage or strengthening ComBat may reduce the apparent advantage even when the implementation becomes more useful. A repair should therefore pass two separate gates: scientific correctness, then measured predictive value. Correctness repairs are required regardless of whether the old headline score was higher; proposed changes to the method need controlled ablations before becoming defaults.

Preserve the useful design: gene-oriented KPCA, fixed source coordinates during adaptation, unlabeled target adaptation, and reusable patient-image layouts. Avoid changing the loss, regularizer, pixel geometry, classifier and preprocessing simultaneously. Keep a valid no-adaptation baseline and change one component at a time under a common evaluation protocol.

Source validation is suitable for classifier selection, but it is not automatically a useful selector of adaptation strength. If a candidate changes only the target coordinates and leaves source images unchanged, source validation may be identical for every candidate. Even when a common pixel frame changes source images, their accuracy is still only an indirect signal about target performance. F11 therefore specifies a separate label-free adaptation-selection policy. Neither low MMD nor retained variance guarantees accurate target classification.

In particular, the optimized MMD compares **gene-coordinate distributions**, while the prediction task concerns **patients**. Better agreement between those point clouds does not by itself establish better patient representations or preserved class information.

**F1 — The target-test set is used during model development**

The paper says that the 30% target-test subset is excluded from alignment and training and used exclusively for final evaluation (Main pp. 14–15; Supplement pp. 1–3).

The code initially makes a 70/30 split, but then:

- [test_kPCADA.ipynb](../Scripts/test_kPCADA.ipynb), cell 6, computes a variance filter on `X_test` and fits a new MinMax scaler on it. Cell 10 fits a target image layout on the test samples.
- Its adapted branch, cell 36, again fits a scaler on `X_test` separately from the adaptation set. Thus a test patient's normalized expression depends on the other test patients.
- Cell 19 uses test accuracy for early stopping. Cell 46 uses it for learning-rate scheduling, checkpoint selection, and early stopping.
- [claude_code.ipynb](../Scripts/claude_code.ipynb), cells 7–8, 15, 31, and 41, and [claude_code_grad_cam.ipynb](../Scripts/claude_code_grad_cam.ipynb), cells 6–7, 11, 28, and 38, contain corresponding uses.

Using unlabeled test measurements to fit preprocessing already differs from the claimed deployment protocol. Using their labels to select the model is a stronger breach: the reported test score is no longer an independent final assessment.

**Recommended fix — enforce the data boundary before fitting anything.** Create an immutable split by original patient ID, then split the source into training and validation patients. Fit source gene filtering, expression scaling and source layout using source training only. Fit the target layout using the unlabeled target-adaptation partition only. For shared genes, apply the fitted source scaling parameters by gene ID. For target-only genes, use adaptation-only statistics under an explicitly documented extension, because source-only scaling is undefined for genes absent from source. Specify a corresponding domain-specific filtering rule; do not silently discard all private genes just to make array shapes match.

Use source validation for classifier checkpoints, learning-rate scheduling and early stopping. Keep target-test labels outside those functions. Serialize the filters, ordered genes, scalers, coordinates and label encoding together. A later refit on all source patients is permissible only under a frozen recipe, including the training duration chosen earlier; it must create a new, traceable fitted artifact.

**Performance rationale and verification:** Shared scaling can preserve cross-domain expression differences that separate test scaling hides. Held-out values can fall outside the training range; choose and document any clipping policy using development data, and check image conversion does not silently wrap or truncate values. Clipping can suppress useful signal, so it is not an automatic fix. Assert patient disjointness and verify that a patient's final image and prediction are unchanged when unrelated test patients are added or removed. This follows the training-only fitting principle in [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls); its [MinMaxScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) explains the out-of-range/clipping behavior.

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

**Recommended fix — separate the unsupervised benchmark from labeled-target extensions.** Remove true target-adaptation labels from classifier training and from tuning APIs in the manuscript benchmark. Stratified partition creation can use labels under the declared split protocol, but the training/tuning code should receive only target features and IDs. Split original patient groups before duplicates, noisy copies or other augmentations are generated. Verify all derived samples retain their original patient ID.

For imbalance, start with source class-weighted cross-entropy or a source-balanced sampler, selected on source validation. Do not automatically combine both, which can overemphasize minority classes. Keep true-target-label training in a separately named supervised or semi-supervised experiment. Pseudo-labeling would also be a new method component requiring its own validation; it is not necessary to repair this defect.

**Performance rationale and verification:** Source-only imbalance handling preserves a legitimate way to improve minority-class performance. Require zero shared original patient IDs across adaptation/test and training/validation, and demonstrate that changing target-adaptation labels cannot change the fitted unsupervised model when split IDs are held fixed. Recompute scores on the repaired partitions; old duplicate-contaminated scores are not a valid performance target.

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

**Recommended fix — establish a strong common training protocol.** Route baseline, KPCA-DA and ComBat images through the same classifier implementation, initialization policy, source-validation criterion, training budget and deterministic evaluation transforms. The existing AdamW configuration (learning rate 0.0001, batch size 16, up to 100 epochs, patience 10) is a reasonable starting candidate for all three, not an established optimum. Select the shared recipe on source development data and give comparable tuning resources to every method. Use paired seeds and identical patient splits.

Remove random spatial flips unless a separate experiment demonstrates a valid biological invariance; gene-image positions have assigned meanings. With these small cohorts, test a frozen pretrained backbone or frozen batch-normalization statistics as development candidates rather than assuming full fine-tuning is best. CAN needs its actual adaptation loss and training procedure; preserve those method-specific requirements while matching data access, backbone where appropriate, and selection budgets. Clarify any resulting exception to the paper's identical-pipeline statement.

**Performance rationale and verification:** This allows a good classifier to benefit KPCA-DA without attributing optimizer improvements to adaptation. Record the effective settings for every run and compare the two image representations with everything else held constant. Repeated inference on the same saved model/input must be deterministic within the declared execution setup.

**F4 — Alignment distances include an extra, undisclosed coordinate adjustment**

The paper describes fitting ICA to the source and applying its projection to target-test samples (Main p. 6), then measuring the resulting domain discrepancy.

In `test_kPCADA.ipynb`, cell 41 instead fits ICA jointly on source and target-test images. It then shifts every target ICA point 80% of the way toward the source centroid:

```python
shift_factor = 0.80
shift_vector = (centroid_source_initial - centroid_test_initial) * shift_factor
test_ica = test_ica + shift_vector  # via test_ica_aligned in the notebook
```

Cell 42 calculates Wasserstein distances from this **shifted** `test_ica`. Its comment describes the points as non-aligned, but the variable has already been replaced. Cells 57–58 likewise apply a 10% centroid shift to ComBat's target ICA points before measuring distances. `claude_code.ipynb`, cells 36–37 and 53–54, and `claude_code_grad_cam.ipynb`, cells 33–34 and 49–50, repeat these operations.

An 80% shift guarantees an 80% reduction in centroid distance regardless of KPCA-DA quality. The stored output shows the distance changing from **2.3384 to 0.4677**. It then reports **WD = 0.7010**. These match Figure 3's adapted annotations, approximately **0.468** and **0.7010** (Main p. 7). The notebook measurements demonstrably include the adjustment, and their numerical match to the figure is concerning. Without an original run manifest, that match alone does not prove exactly how the published figure was generated.

The extra translation changes the measurement arrays, not the classifier's image tensors. This finding directly concerns distribution plots and distances; it does not by itself explain classification accuracy.

Additional measurement problems: [ICAperformance.ipynb](../Scripts/ICAperformance.ipynb), cells 5–6, independently refits ICA to each domain, producing incomparable axes. [UMAPperformance.ipynb](../Scripts/UMAPperformance.ipynb), cell 10, compares each coordinate array with itself, then prints a combined distance computed from literal constants. Some notebooks average two one-dimensional Wasserstein distances; others take their Euclidean norm. These are different summaries and should not be presented interchangeably as a single defined 2D metric.

**Recommended fix — regenerate measurements without moving the evaluated points.** Delete the post-hoc centroid shifts, self-comparisons and literal distance values. Implement the manuscript's source-fitted ICA projection and apply it unchanged to target samples. Save unshifted projection arrays, projection parameters and patient IDs, then derive both plots and numbers from those arrays. State whether the distance is the mean or Euclidean norm of marginal one-dimensional Wasserstein distances; neither should be mislabeled as a fully computed multivariate Wasserstein distance.

For numerical comparisons between methods, use a common evaluation representation with compatible feature meanings and a common fitted projection/scaling. Independently fitted or whitened projections can change the units of the distance. If methods require different image layouts, show method-specific source-fitted ICA plots as diagnostics and explain the limitation rather than treating their numbers as automatically comparable. Any joint source/target visualization must be separately labeled and excluded from the claimed source-only evaluation protocol.

**Performance rationale and verification:** Alignment is a diagnostic, not the acceptance criterion for a classifier: real differences in class proportions can produce genuine domain separation. Verify that metric functions leave input arrays unchanged and that identical inputs produce zero distance. Regenerate Figure 3/Table 3, including the distinction between example-run and aggregate percentage reductions.

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

**Recommended fix — implement the main paper's full empirical MMD².** Use `mean(K_ss) + mean(K_tt) - 2 * mean(K_st)`, including diagonal terms consistently, and remove the absolute value. This is the biased empirical squared-MMD estimator associated with the stated equation. Cache the source term when source coordinates and bandwidth are fixed. Although dropping a constant without the absolute value gives the same optimization gradient, retaining it makes training logs, tuning and reported distances use one interpretable definition. Reconcile Supplement N2E2 with that decision. The distinction between kernel two-sample estimators is described by [Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html).

Keep the MMD bandwidth fixed across lambda candidates fitted to the same initial embeddings. Recompute all loss components at the same final optimizer state; do not combine a pre-update total loss with post-update MMD. Retune lambda and step size after the change, because the gradient direction and balance of terms have changed. Do not choose bandwidth merely by whichever gives the smallest MMD across different kernels; a nearly constant kernel can make that comparison uninformative.

**Performance rationale and verification:** This restores a meaningful alignment objective but does not prove better classification. Require zero loss for identical empirical distributions, permutation invariance, the correct two-point counterexample, and gradients agreeing with finite differences on small nontrivial arrays. Investigate tiny floating-point negatives with numerical tolerance rather than adding an absolute value. Compare the corrected loss with the legacy loss and with no adaptation under the repaired evaluation protocol.

**F6 — Minimizing the trace term does not establish variance preservation**

Main p. 14, Eqs. 17–22, and Supplement Note 2 describe variance preservation/maximization while minimizing a loss containing a **positive** `lambda * trace(Z_t.T @ K_t @ Z_t)`. The code implements that sign literally ([kpcada.py](../Scripts/kpcada.py#L85), lines 85–88 and 160–167).

For a positive-semidefinite RBF kernel matrix, this is a nonnegative quadratic penalty with minimum at zero. In isolation, it penalizes embedding magnitude; it is not a constraint keeping the adapted embedding near its original coordinates. When `K_t` is close to identity, it is approximately a squared-norm penalty.

A check using the actual regularizer reduced target embedding variance from **1.00 to 0.64** after one gradient-descent step. This demonstrates the sign issue; it does not prove that every complete run collapses, because the alignment term also contributes.

This is a mismatch between the **claimed mechanism and the equations/code**, even though the equation and implementation agree. Simply reversing the sign is not a complete repair because it can make an unconstrained objective unbounded.

**Recommended fix — replace the unsupported preservation claim with a bounded, testable mechanism.** Do not simply negate the trace term. As the first candidate, freeze the initial target gene coordinates `Z_t0` and evaluate a normalized reference penalty:

```text
R_ref = sum((Z_t - Z_t0)^2)
        / (sum((Z_t0 - mean_rows(Z_t0))^2) + epsilon)
L = full_MMD_squared + lambda * R_ref
```

Apply it to all retained target genes, including private genes. For positive lambda it discourages unlimited movement from the reference rather than rewarding expansion or penalizing magnitude around zero. Reject or explicitly handle degenerate initial embeddings. If a rigid orientation step is used under F7, apply it before freezing `Z_t0` and before the candidate search.

This is a **method revision**, not a literal implementation of the present trace equation. Name it reference-coordinate preservation and update the manuscript, supplement and ablation labels together. It discourages deformation; it does not guarantee preserved variance or biological function. With this separable penalty and shared-gene-only MMD, private genes remain at their reference coordinates rather than receiving a learned alignment gradient. Make that behavior explicit. Pairwise/neighborhood preservation could instead propagate shared-gene motion to private genes, but would introduce additional design choices and should be a later, separate ablation.

**Performance rationale and verification:** Compare no adaptation, corrected MMD alone, corrected MMD plus the old trace penalty, and corrected MMD plus this reference penalty. Track normalized displacement, variance ratio, covariance spectrum, neighborhood distortion and pixel collisions alongside classification. These identify collapse or excessive rigidity. Select the new penalty only after its predictive tradeoff is measured; an overly strong reference penalty can prevent useful adaptation.

**F7 — Pixel mapping can discard spatial alignment and does not enforce shared-gene positions**

The paper describes a fixed gene-to-pixel arrangement supporting transfer across training and deployment (Main pp. 14–15).

[image_transformer_new.py](../Scripts/pyDeepInsightv3/image_transformer_new.py#L384), lines 384–394, scales source and target coordinates separately to their own minimum and maximum before discretization. Although the rotation is shared, the final scaling bounds are not. Neither this function nor image construction enforces identical pixel positions for a shared gene.

A direct call to the current coordinate methods maps a point cloud and a translated, twice-as-large version of that cloud to **identical pixels**, despite an embedding centroid distance of **22.864**. Thus translation and axis scaling learned in the embedding can disappear during image construction. Matching embedding distributions also does not establish that a particular named gene occupies the same pixel in both domains.

The code correctly preserves the target layout across future target samples. That narrower guarantee should be distinguished from preserving gene identity across source and target images. Adaptation changes spatial layout; image intensities remain normalized original expression values rather than corrected expression measurements.

**Recommended fix — separate a common coordinate frame from named-gene correspondence.** Fit one shared coordinate-to-pixel transformation using source-training and target-adaptation embeddings, then freeze it for future samples. Define rotation, translation, scale, grid resolution, rounding and collision aggregation explicitly. A shared isotropic scale with padding is a useful candidate when preserving relative geometry matters. Fix the whole conversion path: `coordinate_binning` calls `scale_coordinates` again, so changing only `_calculate_coords` would leave independent rescaling in place. Assignment-based discretization must obey the same coordinate-frame contract.

A common frame alone does not put each shared gene in the same pixel: MMD is invariant to point permutations. As a separate candidate, match shared genes by ID and use a rigid orthogonal Procrustes orientation before adaptation to remove arbitrary KPCA orientation differences. This uses no target labels and preserves within-domain distances under the rigid transform. Check rank/degeneracy and do not mistake SciPy's returned `scale` for a multiplier; [its documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html) defines that value as a sum of singular values. A soft paired-gene anchor may then be ablated if correspondence remains poor.

If the paper intends *exactly identical* shared-gene pixels, implement an explicit shared anchor map and revise the method accordingly. Hard anchors can bypass the learned movement of shared genes, so they should not be imposed silently. Otherwise narrow the claim to a fixed map within each domain and measured, approximate cross-domain correspondence. Continue to describe the output as an adapted layout, not corrected expression.

**Performance rationale and verification:** Joint bounds can compress the source into a small image region when target coordinates have extreme ranges. Measure occupied pixels, collision counts, shared-gene pixel displacement and classification before selecting the mapping; set any resolution or outlier policy using development data. Test fixed-map inference and verify that discretization does not silently renormalize each domain. Compare shared scaling, rigid orientation and soft/hard anchors as distinct changes.

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

**Recommended fix — make attribution consume the same fitted artifact as inference.** Persist each domain's ordered retained gene IDs, filter mask, pixel coordinates, collision membership and class encoding. Give the attribution helper an explicit domain/map interface instead of relying on the baseline-only `coords()` method. Run Grad-CAM on the final adapted classifier and target images using its actual target map; map selected positions back through the retained gene list, never an unrelated or unfiltered array.

Resolve the paper's two attribution definitions. For the main claim, implement mean target CAM scores per subtype, explicit score normalization, descending ranking and a declared top-ten/tie policy. Document how subtype groups are formed. Using target labels for a frozen post-hoc analysis must not feed back into model, threshold or checkpoint selection. If thresholded source-training attribution at 0.5 is also useful, report it as a separate analysis and update Supplement Note 4. Genes sharing a pixel share attribution evidence; report all collision members and the ambiguity rather than implying CAM uniquely separates them.

**Performance rationale and verification:** Repairing gene identities improves interpretability without requiring a change to the classifier. Verify recovery of known gene IDs on a synthetic filtered map, including a collision case. Assess ranking stability across seeds and regenerate overlap analyses. Do not tune the model to recover PAM50/TCGA-DE benchmark genes and then call the same overlap an independent validation.

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

**Recommended fix — align rows explicitly and reuse the correct learned batch.** Resolve duplicated gene identifiers by one documented policy, construct a single ordered list of shared unique genes, and reindex both matrices to that list. Assert exact gene-name equality row by row before fitting ComBat. Fit its parameters on permitted source-training/adaptation data, apply batch 2 parameters to target-test samples, and avoid refitting on those test samples or requiring unavailable target outcome covariates. Use the same patient split as the other methods, including the declared 30% test fraction.

**Performance rationale and verification:** A corrected ComBat may become substantially stronger; that is necessary for a meaningful comparison. Test that input column/row reordering followed by ID-based alignment produces the same correction, and verify the target batch identifier at inference. Report both the main all-available-genes KPCA-DA comparison and a shared-genes-only diagnostic if feasible, so any benefit of retaining private genes is distinguishable from the adaptation objective itself.

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

**Recommended fix — establish a versioned cohort manifest before benchmarking.** Record input hashes, patient IDs, labels, source accession/version, inclusion/exclusion reasons, survival-time units and censoring rules, probe-to-gene mapping version, duplicate-gene handling and pre/post-filter counts. Reconstruct the paper's cohorts if the required original inputs are available. Otherwise identify the local cohorts as a different dataset version and update the paper's counts and result claims after rerunning them. Do not add or discard patients merely to obtain a desired count or score.

Investigate the 236-gene GEO input's provenance before treating it as the manuscript's intended expression dataset. Resolve the duplicated TCGA survival gene consistently across KPCA-DA, ComBat and attribution. Keep label derivation separate from feature preprocessing so exclusions can be audited.

**Performance rationale and verification:** Cohort or gene-set changes can change task difficulty more than a model change. Freeze the manifest before tuning, check patient-level uniqueness and class counts, and generate every table's sample/feature counts from that manifest. Evaluate any recovered full cohort as a new run; do not mix its scores with the existing subsets.

**F11 — Preprocessing and hyperparameter settings do not reproduce the supplement**

Supplement Note 3 says to discard the lowest **30%** of genes by training variance and fit min–max scaling on source only. The main notebooks' baseline branches use the **10th percentile**, retain approximately 90%, and fit target-test preprocessing independently. Their adapted branches disable variance filtering entirely. The cfRNA extension's `top70_var` actually uses the **50th percentile** despite its name and comment.

Supplement Note 2 specifies six lambdas, 500 SGD steps, and learning rate 0.001. For TCGA → GEO it reports gamma 965.55 and lambda 0.001; for TCGA → Desmedt, gamma 20.44 and lambda 0.0001.

`test_kPCADA.ipynb`, cell 38, supplies 100 steps and `kpca_gamma=0.2` without a lambda search, inheriting lambda **0.01** and learning rate **0.0001**. `claude_code_grad_cam.ipynb`, cell 30, uses the same configuration. These do not implement the stated selection procedure.

`claude_code.ipynb`, cell 32, does perform a 500-step, label-free search, but omits lambda **10.0**. Cell 33 then overwrites the selected values with `auto_gamma=False`, `mmd_gamma=0.5`, `lambda_reg=1.0`, and only 100 steps. It does not consume the search result for fitting.

The reusable module supports the intended automatic gamma calculation, but that does not establish that the experiments used the supplement's selected values. The fixed gamma values themselves were not recomputed on the full datasets during this audit.

**Recommended fix — define one reproducible selection policy and consume its result.** First reconcile filtering across all image-based branches: implement the supplement's source-training variance rule (retain approximately 70%, with explicit handling of ties and constant genes), then document the adaptation-only rule for target-private genes. If keeping all genes performs better in development, record that as a tested protocol revision rather than leaving filtering different between baseline and adaptation.

For faithful reproduction of the stated search, use all six lambdas `[0.0001, 0.001, 0.01, 0.1, 1, 10]`, 500 steps and learning rate 0.001. For the corrected objective/regularizer, treat that grid as a starting point rather than reusing the old optimum as evidence. Cache a single pair of initial KPCA embeddings and one MMD bandwidth per split, so lambda candidates do not also change coordinates or kernels. Record source/target KPCA kernel settings separately from MMD bandwidth. Use the saved selected configuration in the final fit; remove manual overwrites. Persist any new regularizer/anchor weights separately.

For the unsupervised runner, prohibit target-label accuracy selection: the current `tune_lambda` API can switch to it when labels are supplied. Use a predeclared label-free criterion: screen candidates for nonfinite values, embedding collapse, excessive reference distortion and unacceptable image crowding; among eligible candidates compare full MMD² using the fixed kernel. Set these screening thresholds on development data before inspecting final target performance, record rejected candidates, and keep an explicit no-adaptation fallback if none is eligible. Minimum MMD alone remains an imperfect proxy for classification.

**Performance rationale and verification:** Source validation selects the classifier, but can be insensitive to target-only adaptation. Where source metadata permit, nested source-only development experiments that hold out a source site/batch as a proxy domain can assess preservation weights and the selection rule using labels from that held-out source group. Artificial shifts can support exploratory checks but do not establish real target performance. Freeze the rule before final target evaluation. Verify the final run reproduces the selected candidate's embeddings, bandwidth and final-state loss components within numerical tolerance.

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

**Recommended fix — reconstruct the complete evidence chain.** Create configurations for all six task/direction combinations and a runner that stores split IDs, seeds, effective settings, fitted artifacts, probabilities, labels and unshifted alignment outputs. Implement all claimed ablations with names matching their actual objectives, including the revised regularizer if adopted. Supply a runnable CAN implementation with its provenance, configuration and checkpoint; otherwise mark that comparator unavailable instead of presenting unexplained prediction files as a reproduced training result.

Generate Tables 1–3, Figure 3, ROC legends and attribution comparisons from those saved records. Recover external gene-benchmark construction and analysis settings before claiming biological validation. Use paired method runs on identical patients/splits and disclose whether repetition changes initialization, the split, or both. A few seeds can screen candidates; choose a final repetition budget in advance and explain the limits of precision.

**Performance rationale and verification:** Report per-task/direction effects, macro-F1 or balanced accuracy and minority-class outcomes, not only a pooled mean. Seed-to-seed variability is not a confidence interval over new patients. If estimating patient uncertainty, use a declared patient-level procedure and explain what is held fixed. Do not treat correlated metrics or overlapping repeated splits as independent samples for a Wilcoxon test; justify the pairing/unit and any multiple-comparison handling. Because existing target results have already been inspected, disclose their reuse; a fresh untouched cohort or reserved evaluation set provides stronger final confirmation when available.

**F13 — Metric reporting contains stale values and inconsistent definitions**

[scores_test.ipynb](../Scripts/scores_test.ipynb), for example cells 9, 12, 28, 31, 44, and 47, computes a model's recall into `Sensitivity` but prints the older variable `recall` from another model. Thus the displayed sensitivity can belong to KPCA-DA while the other metrics belong to ComBat or CAN.

The training notebooks calculate weighted recall and mean one-vs-rest specificity, whereas `scores_test.ipynb` uses positive-class recall and negative-class recall for binary tasks. Weighted recall in ordinary single-label classification equals accuracy, which is not the same quantity as positive-class sensitivity. The paper needs a consistent definition for each task.

In `claude_code.ipynb`, cell 41 stores adapted results as `preds`/`trues`, but cell 43 computes metrics from earlier `test_predicted`/`test_true`. If executed in order after the baseline, it can report baseline metrics as the adapted branch's metrics.

There is also a checkpoint bug in `test_kPCADA.ipynb`, cell 46, and `claude_code_grad_cam.ipynb`, cell 38: `net.state_dict().copy()` does not clone tensor storage. Later updates can overwrite the supposed best checkpoint. Stored `test_kPCADA` output reports “best” accuracy **0.653**, then final accuracy **0.612** after restoration, consistent with that problem.

**Recommended fix — calculate metrics once from explicit prediction records.** Replace notebook-global metric variables with one evaluator receiving patient ID, method/run ID, true label, predicted label, class probabilities and ordered class names. Declare binary positive classes explicitly and use consistent sensitivity/specificity definitions; state the averaging rule for multiclass results. Compute ROC-AUC from scores/probabilities with the same class order. Choose any decision threshold on source validation under a frozen policy.

Save the best classifier immediately or clone its tensors using `deepcopy(model.state_dict())`; a shallow dictionary copy is insufficient, as the [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) explains. Restore that checkpoint before producing predictions, and derive uncertainty only from stored run-level results.

**Performance rationale and verification:** Use balanced accuracy or macro-F1 as a declared primary development metric where imbalance matters, alongside the paper's accuracy and per-class outcomes. Verify the evaluator on a known confusion matrix and check that a saved/restored checkpoint reproduces its predictions. Assert prediction IDs and class encodings match the evaluation manifest. This prevents apparent gains caused by stale variables or majority-class dominance.

**F14 — The KPCA equations and standalone transform API need reconciliation**

Main pp. 12–13, Eqs. 7–8, display `V * Lambda^(-1/2)` as the reduced representation. The module uses scikit-learn `KernelPCA.fit_transform`, whose returned training scores are `V * Lambda^(1/2)`. The inverse-square-root form can describe kernel-space coefficients, but it is not the returned score matrix as currently labeled. A small direct check confirms the implementation follows the latter convention.

The separate [KPCA_DA.transform](../Scripts/kpcada.py#L345) API also returns a plain target KPCA projection. The stored estimator was last fitted to target data, and this method does not apply the optimized `Z_t` update. Even transforming the original target input does not return the adapted target embedding. A direct check found a maximum difference of approximately **0.00778** on a small example. It also expects the target KPCA input feature count, not an arbitrary new patient expression vector.

The main image workflow uses `transform_target` with stored pixel coordinates, so this API issue is separate from its valid reuse of a learned target layout.

**Recommended fix — preserve standard KPCA scores and make inference semantics explicit.** Keep the implemented training scores `V * sqrt(Lambda)` unless a separately justified method change is intended. Correct the manuscript to distinguish them from kernel-expansion coefficients `V / sqrt(Lambda)`. If projections are exposed, retain distinct source and target fitted KPCA estimators instead of one overwritten estimator, and store optimized target gene coordinates separately.

Rename the current standalone method to indicate an unadapted target KPCA projection, or deprecate it in favor of explicit APIs. Expose patient inference as conversion of expression values through the frozen target gene-to-pixel map, with gene-ID reordering and a declared missing/unknown-gene policy. A new patient is not a new gene row in the original genes-by-patients KPCA problem; do not append patients and refit KPCA during deployment. There is no learned out-of-sample extension of the optimized gene coordinates in the current algorithm, so do not promise one implicitly.

**Performance rationale and verification:** This avoids replacing a valid coordinate convention merely to match an incorrect equation. Check score/coefficient identities numerically, round-trip the saved mapping artifact, and verify patient-order/batch invariance at inference. Any future out-of-sample gene embedding requires a separate method and evaluation.

**F15 — The saved notebooks do not constitute a clean, portable experiment**

There is no dependency specification, lockfile, or run guide. `kpcada.main` uses absolute Windows paths. Most notebooks use Windows path separators; `claude_code.ipynb`, cell 1, explicitly checks an absolute path to a different checkout and inserts it ahead of local imports. On a machine where that path exists, the notebook can audit/train a different version of the modules.

`claude_code.ipynb` references undefined `y_adapt_enc`. `test_kPCADA.ipynb`, cell 26, calls undefined `cam_image3`. Its optional cell 8 tries to apply a source-fitted UMAP to target expression vectors with a different number of genes. Execution counts and saved outputs are mixed, and some stored outputs show failures or interruption. These issues mean syntactically valid cells are not sufficient evidence of an executable experiment.

**Recommended fix — make one configured implementation executable from a clean process.** Move experiment orchestration into a small runner using the local `kpcada` and image-transformer implementations, with notebooks calling that runner for analysis. Resolve data/output paths from configuration and the repository root; remove imports from unrelated checkouts. Remove undefined branches and stale globals, and give unsupported optional methods explicit input checks.

Create a dependency specification and lock a tested environment, including PyTorch/timm/scikit-learn versions and the pretrained-weight identifier. Set and record Python, NumPy, PyTorch and estimator seeds, hardware and determinism settings. Record the source commit, input hashes and configuration with each run. Dependency versions used for this audit are evidence of the audit environment, not proof of the original experiment environment.

**Performance rationale and verification:** Start with a CPU/small-data end-to-end smoke run covering fit, save, reload, inference, metrics and attribution, then a real-cohort development pilot before launching all directions. Check imports resolve to this checkout and a fresh process can regenerate artifacts without manually executed cells. Pin the successful environment; make any dependency upgrades separately so they cannot silently change benchmark behavior.

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

The recommended implementation and performance-check sequence is:

| Stage | Changes to make together | Evidence required before proceeding |
|---|---|---|
| 1. Establish trustworthy runs | Freeze cohorts/splits; remove leakage and target-label training; repair imports, checkpoints, metrics and ComBat; adopt the common classifier protocol. | Patient-disjoint manifests, fresh-process execution and reliable saved predictions. Historical contaminated scores remain historical, not a target to match. |
| 2. Establish valid controls | Run no adaptation and the legacy adaptation objective under the repaired protocol, with the same gene policy and classifier budget. | A fair reference for the effect of adaptation. Label the legacy objective explicitly; this is a diagnostic control, not endorsement of its scientific interpretation. |
| 3. Repair and ablate the objective | Compare legacy loss versus full MMD²; then compare no regularizer, old trace regularization and reference preservation. Hold pixel conversion fixed while isolating these effects. | Correct loss/gradient checks, stable optimization, geometry diagnostics and development classification results. Retune each changed objective under the same permitted selection budget. |
| 4. Evaluate pixel geometry | Compare existing versus shared coordinate scaling, then separately add rigid orientation or gene anchoring if justified. | Stable inference, measured correspondence/collisions and development performance. Check the selected objective and map together because their effects can interact. |
| 5. Freeze and evaluate | Freeze the method, selection rule and classifier recipe; run all directions and comparators with paired splits/seeds. | Per-direction predictive metrics, minority-class outcomes, documented uncertainty and raw alignment measurements. Do not select a new variant from final-test outcomes and still call that test untouched. |
| 6. Rebuild the scientific claims | Run adapted attribution and external validation; generate every table/figure and synchronize the main paper and supplement. | Traceable patient predictions, gene identities, configuration versions and claims supported by the new results. |

For selecting a method revision, predeclare the primary predictive metric and a practically meaningful tolerance for deterioration on development tasks, alongside minority-class checks. Prefer the simpler candidate when differences are inconclusive. A lower MMD or more attractive plot is insufficient grounds to adopt a revision. If final evaluation does not show improvement, report that result and distinguish reliable reproduction from a successful method improvement; do not retain leakage or a broken comparator to preserve a performance claim.

The existing implementation is a useful starting point, especially its gene-oriented KPCA, target-only optimization, automatic-bandwidth helper, and reusable target image layout. The discrepancies above must be resolved before the reported accuracy gains, alignment reductions, and recovered biomarker lists can be accepted as reproduced by this codebase.
