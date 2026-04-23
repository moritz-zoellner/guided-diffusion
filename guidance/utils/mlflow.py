import mlflow

def run_artifact_path(run, name):
    return mlflow.MlflowClient()._log_artifact_helper(run.info.run_id, name)
