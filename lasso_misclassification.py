"""
Extension to the base repo: LASSO-regularized logistic regression +
misclassification demographic breakdown, mirroring the methodology from
the Mini-REU opioid-risk-metric paper (LASSO variable selection -> refit
interpretable logistic regression -> compare false positive / false
negative groups against concordant/correct predictions).

Run this against the real loan_data.csv once downloaded from Kaggle;
column schema matches exactly.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_and_prep(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace(".", "_", regex=False)
    df['purpose'] = df['purpose'].astype('category')
    return df


def build_lasso_pipeline(numeric_features, categorical_features):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ])

    # LogisticRegressionCV with L1 penalty = LASSO logistic regression,
    # with built-in cross-validated selection of the regularization strength
    # (same idea as lambda.min in the R glmnet workflow from the Mini-REU paper)
    lasso_logit = LogisticRegressionCV(
        l1_ratios=[1.0], solver='liblinear', cv=5,
        scoring='roc_auc', max_iter=5000, random_state=42,
        class_weight='balanced'  # penalizes missing an actual default more heavily
                                  # than misclassifying a safe borrower, correcting
                                  # for the ~84%/16% class imbalance in this dataset
    )

    return Pipeline([('preprocess', preprocessor), ('model', lasso_logit)])


def misclassification_breakdown(df, y_true, y_pred, group_cols):
    """
    Splits predictions into concordant / false positive / false negative,
    then reports how each group_col differs across those three groups.
    Same structure as Tables 5-7 in the Mini-REU paper (mean pain scores,
    overdose rates, health status by misclassification group) -- here
    applied to loan applicant profile instead of patient health profile.
    """
    result = df.copy()
    result['actual'] = y_true.values
    result['predicted'] = y_pred

    def classify(row):
        if row['actual'] == row['predicted']:
            return 'Concordant'
        elif row['predicted'] == 1 and row['actual'] == 0:
            return 'False Positive'  # predicted default, actually repaid
        else:
            return 'False Negative'  # predicted repaid, actually defaulted

    result['group'] = result.apply(classify, axis=1)

    print("\n=== Misclassification distribution ===")
    print(result['group'].value_counts())
    print((result['group'].value_counts(normalize=True) * 100).round(1))

    for col in group_cols:
        print(f"\n=== Mean {col} by misclassification group ===")
        print(result.groupby('group')[col].mean().round(2))

    return result


if __name__ == "__main__":
    df = load_and_prep("loan_data.csv")  # swap for real loan_data.csv

    numeric_features = ['int_rate', 'installment', 'log_annual_inc', 'dti', 'fico',
                         'days_with_cr_line', 'revol_bal', 'revol_util',
                         'inq_last_6mths', 'delinq_2yrs', 'pub_rec']
    categorical_features = ['purpose']

    X = df[numeric_features + categorical_features]
    y = df['not_fully_paid']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipeline = build_lasso_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=== LASSO Logistic Regression performance ===")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))
    print("Selected regularization strength (C):", pipeline.named_steps['model'].C_)

    # which features survived LASSO's shrinkage (non-zero coefficients)
    feature_names = (numeric_features +
                      list(pipeline.named_steps['preprocess']
                           .named_transformers_['cat']
                           .get_feature_names_out(categorical_features)))
    coefs = pipeline.named_steps['model'].coef_[0]
    nonzero = [(name, round(c, 3)) for name, c in zip(feature_names, coefs) if abs(c) > 1e-4]
    print(f"\n=== LASSO-selected features ({len(nonzero)} of {len(feature_names)}) ===")
    for name, c in sorted(nonzero, key=lambda x: -abs(x[1])):
        print(f"  {name}: {c}")

    misclassification_breakdown(
        X_test.reset_index(drop=True), y_test.reset_index(drop=True), y_pred,
        group_cols=['fico', 'dti', 'int_rate', 'inq_last_6mths']
    )


# ============================================================
# SHAP explainability — per-applicant "reason codes"
#
# US lending law (the Equal Credit Opportunity Act) requires
# lenders to give applicants specific reasons when they're denied
# credit ("adverse action notices"). SHAP values are well suited
# to exactly this: for any single applicant, they show which
# features pushed the prediction toward "will default" and by
# how much, giving a concrete, individualized explanation rather
# than just a black-box score.
# ============================================================

def explain_predictions(pipeline, X_test, feature_names, n_examples=3):
    import shap
    import numpy as np

    X_test_transformed = pipeline.named_steps['preprocess'].transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    model = pipeline.named_steps['model']

    # LinearExplainer is exact (not approximate) for linear models like
    # logistic regression, and much faster than the general-purpose
    # KernelExplainer -- appropriate here since our model IS linear.
    # max_samples lives on the masker object, not on LinearExplainer itself --
    # building it explicitly here uses the full background set instead of
    # SHAP's default 100-sample subsample. Still fast since LinearExplainer
    # is closed-form math, not iterative approximation.
    masker = shap.maskers.Independent(X_test_transformed, max_samples=len(X_test_transformed))
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer(X_test_transformed)
    shap_values.feature_names = feature_names

    # Global feature importance (mirrors what the base repo already shows
    # for XGBoost/RF, but here for our LASSO logistic model specifically)
    print("\n=== Mean |SHAP value| by feature (global importance) ===")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    for name, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {val:.4f}")

    # Per-applicant "reason codes" -- the adverse-action-notice framing.
    # Pick a few applicants the model flagged as likely to default and
    # show the top factors driving that specific prediction.
    proba = model.predict_proba(X_test_transformed)[:, 1]
    flagged_idx = np.argsort(-proba)[:n_examples]  # highest predicted default risk

    print(f"\n=== Reason codes for {n_examples} applicants flagged as highest risk ===")
    for i in flagged_idx:
        print(f"\nApplicant (predicted default probability: {proba[i]:.2f}):")
        contributions = list(zip(feature_names, shap_values.values[i]))
        top_factors = sorted(contributions, key=lambda x: -abs(x[1]))[:4]
        for name, val in top_factors:
            direction = "increased" if val > 0 else "decreased"
            print(f"  - {name} {direction} default risk (SHAP: {val:+.3f})")

    return shap_values


if __name__ == "__main__":
    # (re-running here so this section can be executed independently too;
    # relies on `pipeline`, `X_test`, `feature_names` already defined above
    # when the whole file is run top to bottom)
    explain_predictions(pipeline, X_test, feature_names, n_examples=3)