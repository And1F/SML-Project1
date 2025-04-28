from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np
import pandas as pd

# Custom wrapper so scaler can be swapped inside GridSearchCV
class ScalerSwitcher(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=StandardScaler()):
        self.scaler = scaler

    def fit(self, X, y=None):
        return self.scaler.fit(X, y)

    def transform(self, X):
        return self.scaler.transform(X)

if __name__ == "__main__":
    # Load configs and dataset
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Define RationalQuadratic kernel base
    kernel_base = C(1.0, (1e-3, 1e3))

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', ScalerSwitcher()),
        ('select', SelectKBest(score_func=f_regression)),
        ('pca', PCA()),
        ('regressor', GaussianProcessRegressor(random_state=42, normalize_y=True))
    ])

    # Define hyperparameter grid
    param_grid = {
        'scaler__scaler': [StandardScaler(), RobustScaler()],
        'select__k': [300, 500, 700],
        'pca__n_components': [30, 50, 70],
        'regressor__alpha': [1e-3, 5e-3, 1e-2, 5e-2],
        'regressor__kernel': [
            kernel_base * RationalQuadratic(length_scale=ls, alpha=a)
            for ls in [0.5, 1.0, 2.0]
            for a in [0.5, 1.0, 2.0]
        ]
    }

    # 7-Fold cross-validation
    cv = KFold(n_splits=7, shuffle=True, random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters: {grid_search.best_params_}")

    # Evaluate on validation set
    val_preds = grid_search.predict(X_val)
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(grid_search.best_estimator_, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Best pipeline saved.")

    # Save GridSearch results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "gridsearch_results.csv", index=False)
    print("[INFO]: GridSearchCV results saved.")

    # Cross-validation predictions on training set
    cv_preds = cross_val_predict(grid_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_errors = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_errors
    })
    cv_results_df.to_csv(config["data_dir"] / "cv_scores.csv", index=False)
    print("[INFO]: Cross-validation scores saved.")

    # Load test data and predict
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)
    test_preds = grid_search.predict(test_images)

    # Save test predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved.")
