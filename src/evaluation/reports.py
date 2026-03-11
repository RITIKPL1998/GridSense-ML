import pandas as pd

def get_feature_importance(model, feature_names):

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df