# Phase 8A -- Validation Report

Overall status: **PASS**

- Fairness checks: PASS
- Run 1 integrity vs. documented reference: PASS
- Run 1 integrity vs. stored metrics.json (exact): PASS
- Frozen files unchanged: True
- Deterministic repeated computation: True

## Fairness check detail

{
  "same_threshold_selection_rule_used_for_all_models": true,
  "threshold_selected_using_validation_only": true,
  "test_labels_never_used_to_select_threshold": true,
  "test_probabilities_never_used_during_threshold_selection": true,
  "sample_counts_match_expected_partitions": true,
  "positive_prevalence_identical_across_models": true,
  "predictions_align_to_same_samples": true,
  "validation_positive_counts": {
    "logistic_regression": 2592,
    "weighted_lstm": 2592,
    "unweighted_lstm": 2592,
    "temporal_transformer": 2592,
    "vector_world_model_run1": 2592
  },
  "test_positive_counts": {
    "logistic_regression": 1763,
    "weighted_lstm": 1763,
    "unweighted_lstm": 1763,
    "temporal_transformer": 1763,
    "vector_world_model_run1": 1763
  },
  "issues": [],
  "status": "PASS"
}

## Run 1 integrity detail

{
  "reference_comparison": {
    "threshold": {
      "reference": 0.335242,
      "recomputed": 0.335242003202,
      "diff": 3.201999998481142e-09
    },
    "validation_fpr": {
      "reference": 0.0497,
      "recomputed": 0.04968082153760755,
      "diff": 1.9178462392452356e-05
    },
    "validation_recall": {
      "reference": 0.4726,
      "recomputed": 0.47260802469135804,
      "diff": 8.024691358021752e-06
    },
    "test_precision": {
      "reference": 0.755,
      "recomputed": 0.7546296296296297,
      "diff": 0.00037037037037035425
    },
    "test_recall": {
      "reference": 0.74,
      "recomputed": 0.7396483267158253,
      "diff": 0.0003516732841747361
    },
    "test_f1": {
      "reference": 0.747,
      "recomputed": 0.7470638785448296,
      "diff": 6.387854482958755e-05
    },
    "test_fpr": {
      "reference": 0.096,
      "recomputed": 0.09566787003610108,
      "diff": 0.00033212996389891947
    }
  },
  "reference_mismatches": {},
  "reference_status": "PASS",
  "exact_stored_value_comparison": {
    "threshold": {
      "recomputed": 0.335242003202,
      "stored": 0.33524200320243835
    },
    "validation_fpr": {
      "recomputed": 0.04968082153760755,
      "stored": 0.04968082153760755
    },
    "validation_recall": {
      "recomputed": 0.47260802469135804,
      "stored": 0.47260802469135804
    },
    "test_precision": {
      "recomputed": 0.7546296296296297,
      "stored": 0.7546296296296297
    },
    "test_recall": {
      "recomputed": 0.7396483267158253,
      "stored": 0.7396483267158253
    },
    "test_f1": {
      "recomputed": 0.7470638785448296,
      "stored": 0.7470638785448296
    },
    "test_fpr": {
      "recomputed": 0.09566787003610108,
      "stored": 0.09566787003610108
    }
  },
  "exact_stored_value_mismatches": {},
  "exact_status": "PASS"
}

