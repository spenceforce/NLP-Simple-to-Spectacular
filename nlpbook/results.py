import pandas as pd

# Results from models are copied here for easy access and comparison.
model_results = [
    ("Baseline", 0.5011190233977619),
    ("OneR (length)", 0.5026653102746694),
    ("OneR (boc)", 0.5812817904374364),
    ("Decision Tree (boc + accuracy)", 0.5877924720244151),
    ("Decision Tree (boc + gini)", 0.5558087487283825),
    ("Decision Tree (bow)", 0.7185350966429298),
]
model_results = pd.DataFrame(
    [{"Model": m, "Accuracy": a} for m, a in model_results]
).set_index("Model")


def get_results(models=None):
    if models is None:
        return model_results.copy()
    return model_results[model_results.index.isin(models)].copy()
