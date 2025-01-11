import pandas as pd

# Results from models are copied here for easy access and comparison.
model_results = [("Baseline", 0.5011190233977619), ("OneR (boc)", 0.5812817904374364)]
model_results = pd.DataFrame(
    [{"Model": m, "Accuracy": a} for m, a in model_results]
).set_index("Model")

# Multiclass classification results.
mc_model_results = [("Baseline", 0.1997151576805697), ("OneR (boc)", pd.NA)]
mc_model_results = pd.DataFrame(
    [{"Model": m, "Accuracy": a} for m, a in mc_model_results]
).set_index("Model")


def get_results(models=None):
    if models is None:
        return model_results.copy()
    return model_results[model_results.index.isin(models)].copy()


def get_multiclass_results(models=None):
    if models is None:
        return mc_model_results.copy()
    return mc_model_results[mc_model_results.index.isin(models)].copy()
