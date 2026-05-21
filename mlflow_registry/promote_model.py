"""
mlflow_registry/promote_model.py
---------------------------------
Registers the merged model in the MLflow Model Registry and promotes it
through: None → Staging → Production.

Run this after merge_adapter.py has completed and logged the merged model.
"""

import yaml
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

MLFLOW_CFG     = cfg["mlflow"]
TRACKING_URI   = MLFLOW_CFG["tracking_uri"]
EXPERIMENT     = MLFLOW_CFG["experiment_name"]
REGISTERED_NAME = MLFLOW_CFG["registered_model_name"]

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def get_merge_run_id() -> str:
    """Find the most recent 'merge-adapter' run in the experiment."""
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT}' not found. Run merge_adapter.py first.")

    runs = client.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string  = "tags.mlflow.runName = 'merge-adapter'",
        order_by       = ["start_time DESC"],
        max_results    = 1,
    )
    if not runs:
        raise ValueError("No 'merge-adapter' run found. Run merge_adapter.py first.")

    run_id = runs[0].info.run_id
    print(f"Found merge run: {run_id}")
    return run_id


def promote():
    run_id = get_merge_run_id()

    # ── Register model ────────────────────────────────────────────────────────
    artifact_uri = f"runs:/{run_id}/merged_model"
    print(f"\nRegistering model from: {artifact_uri}")

    result = mlflow.register_model(
        model_uri = artifact_uri,
        name      = REGISTERED_NAME,
    )
    version = result.version
    print(f"  ✅ Registered as '{REGISTERED_NAME}' version {version}")

    # ── Transition: None → Staging ────────────────────────────────────────────
    print(f"\nTransitioning v{version} → Staging ...")
    client.transition_model_version_stage(
        name    = REGISTERED_NAME,
        version = version,
        stage   = "Staging",
    )
    print(f"  ✅ v{version} is now in Staging")

    # ── Transition: Staging → Production ──────────────────────────────────────
    print(f"\nTransitioning v{version} → Production ...")
    client.transition_model_version_stage(
        name                         = REGISTERED_NAME,
        version                      = version,
        stage                        = "Production",
        archive_existing_versions    = True,   # demotes any previous Production version
    )
    print(f"  ✅ v{version} is now in Production")

    # ── Summary ───────────────────────────────────────────────────────────────
    model_version = client.get_model_version(REGISTERED_NAME, version)
    print(f"""
══════════════════════════════════════════════
  MLflow Registry Summary
══════════════════════════════════════════════
  Model name : {model_version.name}
  Version    : {model_version.version}
  Stage      : {model_version.current_stage}
  Run ID     : {model_version.run_id}
  Source     : {model_version.source}
══════════════════════════════════════════════
""")
    print("View in MLflow UI → http://127.0.0.1:5000/#/models")


if __name__ == "__main__":
    promote()