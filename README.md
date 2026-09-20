BEYOND GEOMETRIC NAMING
Annotation-Grounded Validation of SAE Concepts in DN-CBM
======================================================================

This repository extends Discover-then-Name (DN-CBM) with an
annotation-grounded evaluation of sparse-autoencoder (SAE) concepts on the
SUN Attribute Database.

Extended evaluation: Md. Ashrafur Rahman, Brac University

The central question is whether a readable name assigned to an SAE dictionary
atom from decoder-text cosine similarity is supported by that atom's behavior
on independently annotated images. The evaluation separates three claims:

1. Geometric naming: a decoder direction is close to a text embedding.
2. Attribute detection: the atom activation ranks annotated positive images
   above negative images.
3. Distributed recoverability: an attribute can be predicted from a learned
   combination of atoms assigned the same name.

The original DN-CBM training and visualization code remains available. The
SUN evaluation extension is located in scripts/project/, and its compact,
version-controlled outputs are stored in experiment-results/.

IMPORTANT
---------

The SUN images, SUN annotation package, pretrained model checkpoint, and
generated .pth caches are intentionally excluded from Git. They must be
downloaded or regenerated before the full pipeline can be reproduced. The
committed CSV, JSON, and figure outputs can be inspected without these large
artifacts.


RESEARCH QUESTIONS AND HYPOTHESES
======================================================================

RQ1 - Geometry and detection
Does greater cosine similarity between an SAE atom and its assigned concept
predict stronger held-out detection of that concept?

RQ2 - Distributed information
Do sparse supervised combinations of same-name atoms outperform a
validation-selected single atom and fixed maximum aggregation?

H1 - Positive geometry-detection association
Higher naming cosine should be associated with higher individual-atom AUROC.

H2 - Distributed recoverability
Sparse same-name group readouts should improve held-out detection, including
relative to matched random groups of the same size and similar activation
frequency.


MAIN FINDINGS
======================================================================

Metric                                      Protocol A     Protocol B
---------------------------------------------------------------------
Eligible attributes / atoms                 22 / 58        93 / 7,596
Mean individual-atom AUROC                  0.559          0.515
Atom-level cosine-AUROC Spearman rho        -0.050         0.021
Groups in multi-atom analysis               14             93
Selected-single macro AUROC                 0.609          0.682
Maximum-aggregation macro AUROC             0.626          0.700
Sparse-readout macro AUROC                  0.626          0.801
Sparse minus matched-random macro AUROC     0.102          0.091

The primary atom-level evidence does not support H1: naming cosine is a weak
guide to individual-atom detection in both protocols. H2 receives qualified
support under Protocol B: sparse readouts recover substantially more attribute
information and outperform matched random groups on average. This demonstrates
recoverability, not that every grouped atom is monosemantic or causally
controls the named concept. Protocol A produces much weaker group gains.

A separate compound-node comparison favors direct CLIP. Mean image-level
Jaccard is 0.169 for direct CLIP and 0.110 for the filtered SAE maximum
construction over the matched 100-attribute label space.


EVALUATION DESIGN
======================================================================

Protocol A - Original-name validation
Original DN-CBM names are recovered from the 20,000-word vocabulary and
mapped to SUN attributes through conservative exact string matches. An
unmatched name is unevaluable under this policy; it is not necessarily
semantically incorrect.

Protocol B - SUN-restricted naming
Every decoder atom is assigned to its nearest text embedding among the 102 SUN
attributes with the prompt "a photo of {concept}". This provides broad
coverage but forces every atom into the SUN vocabulary.

Data split
The 14,340 annotated SUN images are deterministically split with seed 49:

  Training:    8,604 images - fitting, standardization, matched controls
  Validation:  2,868 images - model, atom, regularization, threshold selection
  Test:        2,868 images - final metrics and paired uncertainty estimates

Primary attributes require at least 30 positive and 30 negative test examples.
Binary labels use agreement from at least two of three SUN annotators.

Model and evaluation configuration

  Image encoder:        frozen CLIP RN50, 1,024-dimensional features
  Sparse autoencoder:   8,192 atoms, expansion factor 8
  SAE training source:  CC3M, using the published DN-CBM configuration
  Atom metrics:         AUROC and average precision (AP)
  Geometry statistic:   Spearman rank correlation
  Bootstrap:            2,000 hierarchical replicates for geometry
  Group baselines:      validation-selected single atom and raw maximum
  Sparse readout:       class-balanced L1 logistic regression
  C search grid:        0.001, 0.01, 0.1, 1, 10
  Matched controls:     20 random groups per real group
  Random seed:          49


REPOSITORY LAYOUT
======================================================================

  clip/                       Bundled OpenAI CLIP source
  dncbm/                      DN-CBM models, configuration, and data loaders
  sparse_autoencoder/         Bundled SAE package
  scripts/project/            SUN evaluation pipeline, EXP-00 through EXP-10
  scripts/                    Original DN-CBM scripts
  experiment-results/         Committed CSV, JSON, and figure outputs
  data/                       Local datasets and generated caches; ignored
  SAE/                        Local pretrained SAE artifacts; ignored
  requirements.txt            Reference Python environment
  pyproject.toml              Editable DN-CBM package configuration
  LICENSE                     Repository license


SYSTEM REQUIREMENTS
======================================================================

Recommended environment:

  - Linux or WSL2
  - Python 3.10 (Python 3.11 is also accepted by the bundled SAE package)
  - A CUDA-capable GPU for CLIP feature extraction and text embedding
  - Sufficient storage for SUN images and generated activation caches

The reference requirements file pins PyTorch 2.1.2 and CUDA-related packages.
Users on CPU-only systems, macOS, or newer CUDA stacks should install a
compatible PyTorch build for their platform and may need to adapt the pinned
CUDA packages. The statistical analysis stages can run on CPU, although CLIP
feature extraction is substantially slower.


INSTALLATION
======================================================================

Run all commands from a terminal. Replace the example repository URL with the
URL of this repository or a fork.

  git clone <repository-url>
  cd Discover-then-Name-extended

Option A - Conda

  conda create -n dncbm-sun python=3.10 -y
  conda activate dncbm-sun

Option B - Python virtual environment

  python3 -m venv .venv
  source .venv/bin/activate

Install the project:

  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install -e sparse_autoencoder/
  python -m pip install -e .

Do not run "pip install -e clip/". The bundled clip/ directory has no package
build file. Installing the repository itself in editable mode and running from
the repository root makes the bundled clip module importable.

Quick import check:

  python -c "import clip, dncbm, sparse_autoencoder; print('Imports OK')"


REQUIRED EXTERNAL ARTIFACTS
======================================================================

1. SUN Attribute Database
-------------------------

Download the SUN Attribute Database annotations and corresponding SUN image
archive from the official distribution. Arrange them as follows:

  data/sunattributes/
  |-- SUNAttributeDB/
  |   `-- SUNAttributeDB/
  |       |-- attributes.mat
  |       |-- images.mat
  |       `-- attributeLabels_continuous.mat
  `-- SUNAttributeDB_Images/
      `-- images/
          `-- ...

The loader checks all 14,340 annotated image paths before processing begins.
Alternative locations can be supplied through --annotation_dir and
--image_root to scripts that expose those options.

SUN Attribute Database paper and project:
  https://cs.brown.edu/~gmpatter/sunattributes.html


2. Frozen DN-CBM checkpoint
---------------------------

Download the published CLIP-RN50/CC3M DN-CBM checkpoint from the upstream
release:

  https://nextcloud.mpi-klsb.mpg.de/index.php/s/PZYoKSq29XaEgL6

Store or rename the downloaded checkpoint as:

  data/checkpoints/clip_RN50_sparse_autoencoder_final.pt

Copy it into the exact location expected by the evaluation scripts:

  mkdir -p SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints

  cp data/checkpoints/clip_RN50_sparse_autoencoder_final.pt \
    SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints/sparse_autoencoder_final.pt


3. Original naming vocabulary and CLIP embeddings
--------------------------------------------------

Create the vocabulary directory and download the 20,000-word list:

  mkdir -p data/vocab
  curl -L \
    https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt \
    -o data/vocab/clipdissect_20k.txt

Generate normalized CLIP-RN50 embeddings for the vocabulary:

  python - <<'PY'
  from pathlib import Path
  import torch
  import clip

  vocab_path = Path("data/vocab/clipdissect_20k.txt")
  output_path = Path("data/vocab/embeddings_clip_RN50_clipdissect_20k.pth")
  batch_size = 256

  if not vocab_path.exists():
      raise FileNotFoundError(vocab_path)

  words = [
      line.strip()
      for line in vocab_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if len(words) != 20000:
      raise ValueError(f"Expected 20,000 words, found {len(words)}")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _ = clip.load("RN50", device=device, jit=False)
  model.eval()

  batches = []
  with torch.inference_mode():
      for start in range(0, len(words), batch_size):
          batch = words[start:start + batch_size]
          tokens = clip.tokenize(batch, truncate=True).to(device)
          embeddings = model.encode_text(tokens).float()
          embeddings /= embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
          batches.append(embeddings.cpu())
          print(f"Processed {min(start + batch_size, len(words))}/{len(words)}")

  embeddings = torch.cat(batches, dim=0)
  if embeddings.shape != (20000, 1024):
      raise ValueError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")
  if not torch.isfinite(embeddings).all():
      raise ValueError("Embeddings contain non-finite values")

  torch.save(embeddings, output_path)
  print(f"Saved {output_path} with shape {tuple(embeddings.shape)}")
  PY


4. Original DN-CBM concept names
--------------------------------

If concept_names.csv was supplied with the upstream artifact, place it here:

  SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/concept_names.csv

Otherwise, generate it from the checkpoint and the vocabulary embeddings:

  python scripts/assign_names.py \
    --lr 5e-4 \
    --l1_coeff 3e-5 \
    --expansion_factor 8 \
    --img_enc_name clip_RN50 \
    --num_epochs 200 \
    --resample_freq 10 \
    --train_sae_bs 4096 \
    --device cuda

For a CPU-only machine, replace "--device cuda" with "--device cpu".


5. Verify the required files
----------------------------

  test -f data/vocab/clipdissect_20k.txt
  test -f data/vocab/embeddings_clip_RN50_clipdissect_20k.pth
  test -f SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints/sparse_autoencoder_final.pt
  test -f SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/concept_names.csv

If all commands exit silently with status 0, the four required files exist.


REPRODUCING THE EVALUATION
======================================================================

Run every command from the repository root. The order below preserves the
intended separation between training, validation, and test decisions.


EXP-00 - Frozen-model sanity check
----------------------------------

Place a small set of ordinary image files in data/exp00_images/, then run:

  python scripts/project/00_extract_pilot_features.py \
    --image_dir data/exp00_images \
    --device cuda

  python scripts/project/00_check_frozen_dncbm.py \
    --features_path data/exp00/clip_RN50_pilot_features.pth \
    --device cuda

The pilot verifies checkpoint loading, tensor dimensions, reconstruction, and
activation behavior. It is a sanity check, not a population-level estimate.


EXP-01 - Data split and naming protocols
-----------------------------------------

  python scripts/project/01_prepare_sun_splits.py
  python scripts/project/01_embed_sun_vocabulary.py --device cuda
  python scripts/project/01_assign_sun_names.py

  python scripts/project/01_recover_original_names.py --device cuda
  python scripts/project/01_prepare_protocol_a_mapping.py
  python scripts/project/01_freeze_protocol_a_mapping.py

The Protocol A review file contains lexical suggestions for inspection, but the
primary mapping accepts exact string matches only. Mapping decisions should not
be changed after inspecting semantic test scores.


EXP-02 and EXP-03 - Cache, atom scoring, and geometry
-----------------------------------------------------

  python scripts/project/02_cache_sun_activations.py \
    --device cuda \
    --batch_size 32 \
    --num_workers 4

  python scripts/project/02_score_individual_atoms.py
  python scripts/project/03_geometry_semantics.py
  python scripts/project/03_bootstrap_geometry.py
  python scripts/project/03_plot_geometry_semantics.py

The activation cache is large and ignored by Git. Geometry bootstrap intervals
resample attributes and atoms; they do not resample images.


EXP-04 - Individual-atom case studies
--------------------------------------

  python scripts/project/04_prepare_case_candidates.py
  python scripts/project/04_build_inspection_manifest.py
  python scripts/project/04_render_case_sheets.py
  python scripts/project/04_audit_case_context.py


EXP-05 and EXP-06 - Fixed baselines and sparse group readouts
----------------------------------------------------------------

  python scripts/project/05_select_validation_atoms.py
  python scripts/project/05_evaluate_fixed_baselines.py
  python scripts/project/05_bootstrap_fixed_baselines.py

  python scripts/project/06_select_sparse_models.py
  python scripts/project/06_evaluate_sparse_models.py
  python scripts/project/06_bootstrap_sparse_models.py

Sparse models are fitted on training images, selected by validation AUROC, and
evaluated once on the test split. The selected training fit is not refitted on
validation images.


EXP-07 - Group structure and error analysis
--------------------------------------------

  python scripts/project/07_select_group_cases.py
  python scripts/project/07_audit_group_structure.py
  python scripts/project/07_build_group_error_manifest.py
  python scripts/project/07_render_group_error_sheets.py


EXP-08 - Compound nodes and direct CLIP
----------------------------------------

  python scripts/project/08_prepare_compound_groups.py
  python scripts/project/08_build_compound_scores.py
  python scripts/project/08_select_compound_thresholds.py
  python scripts/project/08_evaluate_compound_nodes.py
  python scripts/project/08_evaluate_clip_baseline.py
  python scripts/project/08_compare_sae_clip.py

This is separate from the sparse-readout analysis. It filters atoms by training
activity and cosine similarity, selects thresholds on validation data, and
reports image-level Jaccard on the common 100-attribute label space.


EXP-09 - Protocol population audit
-----------------------------------

  python scripts/project/09_compare_protocol_populations.py
  python scripts/project/09_summarize_protocol_a_groups.py


EXP-10 - Robustness and matched controls
-----------------------------------------

  python scripts/project/10_label_permutation_control.py
  python scripts/project/10_worker_agreement_sensitivity.py

  python scripts/project/10_prepare_matched_random_groups.py
  python scripts/project/10_select_matched_random_models.py --n_jobs 8
  python scripts/project/10_evaluate_matched_random_models.py --n_jobs 8

  python scripts/project/10_correlated_attribute_audit.py

Set --n_jobs to the number of parallel workers appropriate for the machine.
The matched-random analysis fits 20 controls per real group. Matching controls
group size and marginal activation frequency, not the complete correlation
structure.


KEY OUTPUTS
======================================================================

Question / result                        Main file
---------------------------------------------------------------------
Split and eligibility                    experiment-results/exp01/
                                         split_summary.json

Naming populations                       experiment-results/exp01/
                                         protocol_a_mapping_summary.json
                                         sun_naming_summary.json

Individual-atom detection                experiment-results/exp02/
                                         atom_scoring_summary.json

Geometry-detection relationship          experiment-results/exp03/
                                         geometry_semantics_summary.json
                                         geometry_bootstrap_ci.json

Fixed baselines                          experiment-results/exp05/
                                         fixed_baseline_summary.json

Sparse group readouts                    experiment-results/exp06/
                                         sparse_test_summary.json
                                         sparse_bootstrap_summary.json

Group interpretation                     experiment-results/exp07/
                                         group_structure_summary.csv

SAE versus direct CLIP                   experiment-results/exp08/
                                         sae_clip_comparison_summary.json

Protocol comparison                      experiment-results/exp09/
                                         protocol_comparison_summary.json

Matched random controls                  experiment-results/exp10/
                                         matched_random_test_summary.json

Annotation/context sensitivity           experiment-results/exp10/
                                         worker_agreement_summary.json
                                         correlated_attribute_summary.json

CSV files contain per-atom, per-attribute, model-selection, and bootstrap
details. Large prediction tensors and generated caches are intentionally not
committed.


INTERPRETATION BOUNDARIES
======================================================================

  - High decoder-text cosine is a naming signal, not direct evidence that an
    activation detects the corresponding attribute.

  - High sparse-readout AUROC shows that information is recoverable from a
    group. Signed coefficients mean the group need not be a positive union of
    interchangeable concept detectors.

  - Strong random-group performance shows that successful recovery is not
    unique evidence of semantic coherence.

  - Conditional degradation is consistent with contextual dependence but does
    not identify a causal shortcut.

  - Results use one frozen encoder-SAE checkpoint, one deterministic split,
    and one dataset. Bootstrap intervals condition on fitted models and do not
    represent every source of training uncertainty.

  - The experiments do not establish causal steering, intervention
    faithfulness, or cross-dataset generalization.


TROUBLESHOOTING
======================================================================

ModuleNotFoundError: clip
  Confirm that commands are run from the repository root and that
  "python -m pip install -e ." completed successfully. Do not attempt to
  install clip/ directly.

Checkpoint not found
  Confirm the full checkpoint path shown in the required-artifacts section.
  The directory name, including every hyperparameter token, must match exactly.

CUDA requested but unavailable
  Replace "--device cuda" with "--device cpu" where the script exposes a
  --device option. Feature extraction will be considerably slower.

SUN image validation fails
  Check that SUNAttributeDB_Images/images contains the same relative image paths
  listed by SUNAttributeDB/images.mat. Avoid adding an extra nested images/
  directory when extracting the archive.

Out of memory during activation caching
  Reduce --batch_size in 02_cache_sun_activations.py. The completed cache is
  written to data/exp02/sun_frozen_cache.pth.

Multiprocessing issues
  Reduce --num_workers to 0 or 1 for constrained systems or some WSL/Windows
  configurations. Reduce EXP-10 --n_jobs if system memory is limited.


UPSTREAM PROJECT AND ACKNOWLEDGEMENTS
======================================================================

This extension builds on DN-CBM by Sukrut Rao, Sweta Mahajan, Moritz Boehle,
and Bernt Schiele.

  Original repository:
  https://github.com/neuroexplicit-saar/Discover-then-Name

  ECCV 2024 paper:
  https://arxiv.org/abs/2407.14499

  DN-CBM project resources:
  https://sukrutrao.github.io/publication/discover-then-name-task-agnostic-concept-bottlenecks-via-automated-concept-discovery/

The repository also incorporates code from:

  OpenAI CLIP:
  https://github.com/openai/CLIP

  ai-safety-foundation/sparse_autoencoder:
  https://github.com/ai-safety-foundation/sparse_autoencoder

SUN annotations and images are provided by the authors of the SUN Attribute
Database.


CITATION
======================================================================

If this code is used as a DN-CBM implementation, cite:

  @inproceedings{rao2024discover,
    author    = {Rao, Sukrut and Mahajan, Sweta and B{\"o}hle, Moritz and
                 Schiele, Bernt},
    title     = {Discover-then-Name: Task-Agnostic Concept Bottlenecks via
                 Automated Concept Discovery},
    booktitle = {European Conference on Computer Vision},
    year      = {2024}
  }

For the SUN Attribute Database, cite:

  @article{patterson2014sun,
    author  = {Patterson, Genevieve and Xu, Chen and Su, Hang and Hays, James},
    title   = {The SUN Attribute Database: Beyond Categories for Deeper Scene
               Understanding},
    journal = {International Journal of Computer Vision},
    volume  = {108},
    number  = {1--2},
    pages   = {59--81},
    year    = {2014}
  }


LICENSE
======================================================================

The repository retains the upstream license in LICENSE. Dataset, CLIP, SAE,
and pretrained-model artifacts may have separate licenses and terms of use.
