# Assumptions

## Dataset Usage

- The breast-cancer data set is used strictly as a technical ML benchmark. No
  clinical validity, diagnosis, treatment guidance, or real-world medical
  claim is implied by notebooks, API responses, or agent review content.
- The 30-feature binary target structure provided by sklearn is treated as the
  benchmark contract for the classical and boosting workflows.

## Reproducibility and Data Handling

- Randomized operations use the fixed seed 42 through src.config.RANDOM_SEED
  where shared configuration is available.
- Notebook 02 injects approximately one percent missing values only into a
  copy of post-split training data. That copy validates imputation behavior and
  is not used for actual model training or evaluation.
- The container trainer uses the selected notebook 03 XGBoost hyperparameters
  as constants instead of repeating a search at every container start. This
  prioritizes deterministic, fast artifact generation for Docker and CI.

## Transformer and LLM Scope

- Transformer fine-tuning is intentionally capped at at most 4,000 training
  records, 1,000 validation records, sequence length 128, and one or two
  epochs with a lightweight checkpoint. These are assessment resource
  constraints, not a claim that a larger model or longer training would not
  improve performance.
- The recommendation chain defaults to a deterministic mock LLM. It makes no
  external API calls and has no personal cost, while retaining an interface
  that can accept a LangChain-compatible LLM later.

## Agent and Deployment Guardrails

- The CrewAI workflow has explicit per-agent iteration limits and a
  wall-clock timeout around kickoff. Missing or malformed artifacts return a
  structured failure response rather than fabricated review content.
- The Azure deployment document is design-only. No cloud resources, accounts,
  subscriptions, or infrastructure-as-code have been provisioned.
- Docker Compose uses a one-shot trainer followed by an API service sharing
  artifacts through a named volume. Its cloud design maps that concept to
  versioned Blob Storage artifacts.

## Limitations

- The API explain endpoint calculates SHAP values for one record per request.
  It avoids generating a full global plot, but it still adds more latency than
  serving precomputed explanations.
- The resource-capped transformer result should not be compared directly with
  published results that use larger checkpoints, broader data, or longer
  training schedules.
- The explainability workflow may use a conservative-threshold fallback when
  the saved model has no default-threshold false negatives, so that local
  false-negative-style analysis remains transparent rather than inventing a
  nonexistent error.
