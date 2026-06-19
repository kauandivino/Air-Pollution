from __future__ import annotations

from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency guard
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency guard
    XGBClassifier = None


DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    family: str
    description: str


def build_model_registry(random_seed: int = DEFAULT_RANDOM_SEED) -> dict[str, ModelSpec]:
    """Create the initial model registry used by baseline experiments."""
    registry = {
        "majority_class": ModelSpec(
            name="majority_class",
            family="baseline",
            description="Predicts the most frequent class in the training split.",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
        ),
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            family="linear",
            description="Regularized logistic regression with median imputation and scaling.",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        ),
        "decision_tree": ModelSpec(
            name="decision_tree",
            family="tree",
            description="Shallow-ish decision tree with balanced class weights.",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        DecisionTreeClassifier(
                            class_weight="balanced",
                            max_depth=12,
                            min_samples_leaf=50,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            family="ensemble",
            description="Random forest baseline with balanced class weights.",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=120,
                            class_weight="balanced_subsample",
                            min_samples_leaf=20,
                            n_jobs=-1,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        ),
    }

    if LGBMClassifier is not None:
        registry["lightgbm"] = ModelSpec(
            name="lightgbm",
            family="boosting",
            description="Gradient boosting trees via LightGBM.",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=300,
                            learning_rate=0.05,
                            num_leaves=31,
                            min_child_samples=40,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            class_weight="balanced",
                            objective="binary",
                            n_jobs=-1,
                            random_state=random_seed,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
        )

    if XGBClassifier is not None:
        registry["xgboost"] = ModelSpec(
            name="xgboost",
            family="boosting",
            description="Gradient boosting trees via XGBoost.",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=6,
                            min_child_weight=10,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            n_jobs=-1,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        )

    return registry


def get_model_specs(
    model_names: list[str] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[ModelSpec]:
    registry = build_model_registry(random_seed=random_seed)
    if model_names is None:
        return list(registry.values())

    unknown = [name for name in model_names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown model names: {', '.join(unknown)}")

    return [registry[name] for name in model_names]
