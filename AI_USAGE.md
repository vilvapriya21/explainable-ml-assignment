# AI Usage Disclosure

## Tool Used

OpenAI Codex was used during this repository's development as an
implementation and review assistant. The maintainer supplied the assessment
requirements, scoped the requested changes, and retained responsibility for
accepting the resulting work.

## What It Was Used For

- Drafting initial implementations of requested source modules from detailed
  written specifications, including trainers, evaluation, configuration,
  utilities, agents, API endpoints, and Docker support.
- Drafting notebook cells for Parts B through H, including explanatory
  markdown, runnable examples, and artifact-writing steps.
- Drafting and updating pytest coverage for the requested functions,
  workflows, configuration helpers, and FastAPI endpoints.
- Drafting repository documentation, including README.md, ASSUMPTIONS.md,
  AI_USAGE.md, architecture documentation, and the cloud deployment design.
- Running local validation commands, interpreting failures, and adjusting
  generated code to match installed package behavior and the saved model
  artifact format.

## What Was Reviewed or Adjusted

Every AI-generated file was checked against the assessment's stated
requirements before being retained. Concrete adjustments during assisted
development included aligning shared FN_COST and FP_COST configuration imports,
supporting the saved full sklearn Pipeline artifact in the FastAPI startup
loader, and making the Docker trainer reproduce notebook 03's selected XGBoost
configuration rather than rerunning a tuning search.

Before a final external submission, the project maintainer should add any
additional human-specific review example that is not captured in repository
history, such as a domain decision, test case, or wording change they made
independently.
