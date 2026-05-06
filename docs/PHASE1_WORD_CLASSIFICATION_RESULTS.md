# Phase 1 Word-Classification Modal Results

Date: 2026-04-29

## Summary

The detached full Phase 1 word-classification run completed successfully. Modal app `ap-lA3y1U06dHYydNNQXeQhaZ` stopped after writing the expected full-run artifacts at `13:40 PDT`:

```text
/checkpoints/motion_conformer_phase1_words_full/phase1_words.pt
/checkpoints/motion_conformer_phase1_words_full/word_vocab.json
/checkpoints/motion_conformer_phase1_words_full/train_summary.json
```

The full validation evaluation also completed successfully in Modal app `ap-jYT1mNsp2eZ1rcX9od39nC`, writing:

```text
/checkpoints/motion_conformer_phase1_words_full/eval_validation/metrics.json
/checkpoints/motion_conformer_phase1_words_full/eval_validation/predictions.jsonl
```

The model beat the untrained baseline by a wide margin on word classification, but it still has a strong prediction-diversity problem: only `19` unique top-5 predicted word sets across `1,739` validation utterances.

## Full Training Run

Command:

```bash
modal run --detach modal_motion_conformer_app.py::train_phase1_words_full \
  --batch-size 2 \
  --top-k-words 512 \
  --max-windows-per-utterance 12
```

Configuration:

| Setting | Value |
|---|---:|
| Dataset | `bdanko/how2sign-landmarks-front-raw-parquet` |
| Split | `train` |
| Max samples | full train split |
| Utterances | `31,046` |
| Windows | `120,734` |
| Steps | `15,523` |
| Batch size | `2` utterances |
| Window size / stride | `60 / 30` frames |
| Max windows per utterance | `12` |
| Word vocabulary size | `512` |
| LR | `1e-4` |
| Epochs | `1` |

Training summary:

| Metric | Value |
|---|---:|
| `avg_total_loss` | `0.420955` |
| `avg_full_recon_loss` | `0.010821` |
| `avg_pos_recon_loss` | `0.015603` |
| `avg_vel_recon_loss` | `0.006038` |
| `avg_word_bce` | `0.410088` |
| `avg_latent_smooth_loss` | `0.004048` |
| `avg_latent_reg_loss` | `0.000574` |

Top vocabulary words were dominated by common instructional words, including:

```text
going, want, it's, like, get, you're, one, little, i'm, make, we're, right
```

## Full Validation Results

Command:

```bash
modal run modal_motion_conformer_app.py::evaluate_phase1_words_checkpoint \
  --split validation \
  --batch-size 4 \
  --max-samples 0
```

Validation covered `1,739` utterances and `6,607` selected windows.

| Metric | Trained model | Untrained baseline |
|---|---:|---:|
| `micro_precision@0.5` | `0.082587` | `0.009372` |
| `micro_recall@0.5` | `0.196653` | `0.515952` |
| `micro_f1@0.5` | `0.116323` | `0.018409` |
| `precision@5` | `0.086717` | `0.004255` |
| `recall@5` | `0.099037` | `0.005344` |
| `hit@5` | `0.308223` | `0.021277` |

Reconstruction and latent metrics:

| Metric | Value |
|---|---:|
| `val/full_recon_loss` | `0.011042` |
| `val/pos_recon_loss` | `0.016495` |
| `val/vel_recon_loss` | `0.005588` |
| `val/latent_smooth_loss` | `0.001888` |
| `val/latent_reg_loss` | `0.000187` |
| `val/word_bce` | `0.398264` |

Sanity checks:

| Check | Value |
|---|---:|
| `beats_untrained_baseline` | `true` |
| `predictions_vary` | `true` |
| `unique_predicted_word_sets` | `19` |
| `baseline_unique_predicted_word_sets` | `248` |

## Prediction Analysis

The model improved substantially over random initialization, but the prediction distribution is still too concentrated. The most common predicted top-5 set appeared in `1,378 / 1,739` validation utterances:

```text
little, we're, i'm, see, going
```

Most common top-5 sets:

| Count | Predicted words |
|---:|---|
| `1,378` | `little`, `we're`, `i'm`, `see`, `going` |
| `70` | `good`, `okay`, `that's`, `right`, `i'm` |
| `36` | `good`, `i'm`, `that's`, `we're`, `see` |
| `35` | `good`, `that's`, `okay`, `i'm`, `right` |
| `35` | `good`, `that's`, `i'm`, `see`, `right` |

Example validation rows:

| Reference words | Predicted words | Sentence preview |
|---|---|---|
| `show`, `i'll` | `little`, `we're`, `i'm`, `see`, `going` | `I'll show you.` |
| `one`, `right`, `way`, `let` | `little`, `we're`, `i'm`, `see`, `going` | `So that is one way, you can just let them go right by.` |
| `like`, `little`, `bit`, `look`, `something`, `might` | `little`, `we're`, `i'm`, `see`, `going` | `If you need to deal with the situation a little bit more...` |
| `good`, `kind`, `getting`, `definitely`, `recommend` | `little`, `we're`, `i'm`, `see`, `going` | `I will definitely recommend getting this kind of yogurt...` |

## Comparison To Previous T5 Run

The previous Motion-Conformer-T5 generation run failed completely: all `7,088` validation windows generated the same punctuation-only string and scored BLEU `0.0074`, ROUGE-L `0.0`, and exact match `0.0`.

This Phase 1 word-classification run is a clear improvement because it learns a measurable visual-to-word signal:

```text
micro_f1@0.5: 0.1163 trained vs 0.0184 baseline
hit@5:        0.3082 trained vs 0.0213 baseline
```

It also preserves the strong reconstruction behavior:

```text
previous Phase 1-style val/full_recon_loss: 0.01257
new Phase 1 word val/full_recon_loss:       0.01104
```

## Interpretation

What went well:

- The full run completed and produced all expected artifacts.
- Reconstruction stayed strong while adding word supervision.
- Word classification beats the untrained baseline clearly on micro-F1, precision@5, recall@5, and hit@5.
- Latent regularization and smoothness are much better than the earlier combined T5 run.

What still went poorly:

- Top-5 predictions are heavily biased toward frequent generic words.
- Only `19` unique predicted word sets across the full validation split is too low.
- The classifier improves ranking enough to beat baseline, but it is not yet discriminative enough for reliable per-utterance word recovery.

Recommended next iteration:

- Use stronger class balancing, such as focal BCE or logit-adjusted BCE.
- Exclude very generic words like `going`, `it's`, `i'm`, `we're`, `that's`, and `you're` from the target vocabulary.
- Add train/validation per-class AP or AUROC to separate thresholding issues from ranking issues.
- Train for more than one epoch after fixing generic-word dominance.
- Consider a second head for rarer content nouns/verbs so frequent instructional filler words cannot dominate the shared top-5 predictions.
