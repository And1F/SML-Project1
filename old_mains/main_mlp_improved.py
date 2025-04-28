from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Load configs
    config = load_config()

    # Load dataset
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Check number of features
    n_features = X_train.shape[1]
    print(f"[INFO]: Number of flattened features: {n_features}")

    # Dynamically define sensible values for SelectKBest and PCA
    k_values = [int(r * n_features) for r in [0.3, 0.5, 0.7] if int(r * n_features) >= 10]
    if len(k_values) == 0:
        k_values = ['all']  # fallback if dataset is tiny

    pca_values = [int(r * n_features) for r in [0.3, 0.5, 0.7] if int(r * n_features) >= 10]

    # Define pipeline: scaler -> SelectKBest -> PCA -> MLP Regressor
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('selectk', SelectKBest(score_func=f_regression, k='all')),
        ('pca', PCA(n_components=min(pca_values) if pca_values else None)),
        ('regressor', MLPRegressor(random_state=42, max_iter=1000, early_stopping=True))
    ])

    # Define hyperparameter grid
    param_grid = {
        'selectk__k': k_values,
        'pca__n_components': pca_values,
        'regressor__hidden_layer_sizes': [(64, 64), (128, 64), (128, 128)],
        'regressor__activation': ['relu', 'tanh'],
        'regressor__alpha': [1e-5, 1e-4, 1e-3],
        'regressor__learning_rate_init': [0.001, 0.01],
        'regressor__learning_rate': ['constant', 'adaptive'],
        'regressor__validation_fraction': [0.1, 0.15],
        'regressor__n_iter_no_change': [10, 20],
    }

    # 7-Fold cross-validation setup
    cv = KFold(n_splits=7, shuffle=True, random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_absolute_error',
                               n_jobs=-1, verbose=2, return_train_score=True)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {grid_search.best_params_}")

    # Evaluate on validation set
    val_preds = grid_search.predict(X_val)
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(grid_search.best_estimator_, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline.joblib")

    # Save GridSearchCV result summary to CSV
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "gridsearch_results.csv", index=False)
    print("[INFO]: GridSearchCV results saved to gridsearch_results.csv")

    # Cross-validated predictions for training set (for inspection)
    cv_preds = cross_val_predict(grid_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / "cv_scores.csv", index=False)
    print("[INFO]: Cross-validation scores saved to cv_scores.csv")

    # Load test data
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)

    # Predict on test set
    test_preds = grid_search.predict(test_images)

    # Save test predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction.csv")
