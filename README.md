# VGG16 — Patient-balanced batches and conservative MixStyle

This is an untrained experimental candidate. One final standard VGG16 is used at inference; no ensemble, TTA, extra neural network, or weight averaging. The assignment explicitly forbids ensembles.

## Run

The notebook has one code cell. Select Colab GPU, change DATA_DIR, and run. The same 21 development patients and five folds from the previous group-CV experiment are preserved. The reserved six local Test patients remain excluded; this cell never decodes their images or evaluates their labels against model predictions.

## Candidates

Both candidates share the same frozen ImageNet feature cache and the same per-fold FC preparation weights.

- patient_balanced: each full-CNN training batch contains two different patients, two absent and two present images from each patient. The loss is unweighted smoothed CE because the batch is already class-balanced. Class weights are not applied twice. Images from rare positive groups can repeat across batches; CV must verify whether the benefit outweighs this overfitting risk.
- patient_mixstyle: the same training setup, plus parameter-free feature-statistics augmentation after the first two VGG pooling blocks. Statistics are mixed only with a different patient in the training batch. Applied with p=.25; Beta alpha=.3; original statistics retain at least 50% weight. These are conservative project-specific settings, not known optima for sheath ultrasound.

MixStyle is inspired by [Zhou et al., ICLR 2021](https://arxiv.org/abs/2104.02008), with [author implementation](https://github.com/KaiyangZhou/mixstyle-release). It is a domain-generalization hypothesis, not a demonstrated improvement on this dataset. The hooks are inactive during validation and removed after training, including error exits. The saved state dict and exported inference script use standard VGG16 only and require no patient identity.

## Selection and time

Each candidate uses five-fold development CV for at most eight fine-tuning epochs per fold. The cosine schedule retains T_max=11 so these are the first eight epochs of the previous schedule. Three FC warmup epochs are shared by the two candidates within each fold. A common epoch is selected by mean balanced validation CE; one global threshold comes from the selected-epoch OOF predictions. Candidates are ranked by mean fold BA, then worst-fold BA, then mean AUROC. CV selection estimates may be optimistic and are not independent Test estimates.

There are 80 candidate/fold fine-tuning epochs versus 55 in the previous experiment, plus preparation and any final refit. Actual runtime depends on the GPU and Drive. Completed recipe/fold results are reused for identical code/config/data. Interrupted recipe/folds restart; optimizer-state recovery is not provided.

By default, final refitting is skipped unless the selected candidate exceeds the prior mean CV BA 0.8500322788. This gate is a convenience to avoid spending GPU on a clearly weaker CV result; it is not a significance test. Existing model weights are preserved. No new private/independent Test score is claimed. The previous local Test is not re-evaluated.

Results and, when justified, the single final vgg.pt are saved on Drive under task2_vgg_patientstyle. The diagnostic ZIP is vgg_patientstyle_report.zip and excludes weights.

## Verification

On the actual development folds: verified exactly 2 patients x 2 classes x 2 distinct images in every sampled batch, deterministic sampler restart, different-patient statistics partners, eval-hook identity, no double class weighting, no Test decoding/evaluation path, development-only threshold scope, hook cleanup, and training/inference syntax. The local environment has no torch; CUDA training and tensor-gradient execution remain untested. New performance is unknown.
