from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

if __name__ == "__main__":
    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configs
    config = load_config()

    # Load dataset
    images, distances = load_dataset(config)
    logger.info(f"Dataset loaded with {len(images)} samples.")

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Determine optimal number of PCA components for 95% variance
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA().fit(X_train_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_pca_components = np.argmax(explained_variance >= 0.95) + 1
    logger.info(f"95% variance reached with {optimal_pca_components} components.")

    # Define candidate kernels
    kernel_candidates = [
        C(1.0, (1e-2, 1e2)) * RBF(length_scale=ls) for ls in [0.1, 0.5, 1.0, 2.0, 5.0]
    ] + [
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=ls, nu=1.5) for ls in [0.1, 0.5, 1.0, 2.0, 5.0]
    ]

    # Define pipeline: scaler -> feature selection -> PCA -> Gaussian Process Regressor
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selection', SelectKBest(score_func=f_regression, k=300)),  # tune k later
        ('pca', PCA(n_components=optimal_pca_components)),
        ('regressor', GaussianProcessRegressor(random_state=42, normalize_y=True))
    ])

    # Hyperparameter grid
    param_grid = {
        'feature_selection__k': [200, 300, 400],
        'pca__n_components': [optimal_pca_components - 10, optimal_pca_components, optimal_pca_components + 10],
        'regressor__alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        'regressor__kernel': kernel_candidates
    }

    # K-Fold cross-validation setup (15 folds)
    cv = KFold(n_splits=15, shuffle=True, random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_absolute_error',
                               n_jobs=-1, verbose=2, return_train_score=True)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters found: {grid_search.best_params_}")

    # Evaluate on validation set
    val_preds = grid_search.predict(X_val)
    logger.info(f"Validation MAE: {mean_absolute_error(y_val, val_preds)}")
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(grid_search.best_estimator_, config["data_dir"] / "pipeline.joblib")
    logger.info("Best pipeline saved to data/pipeline.joblib")

    # Save GridSearchCV result summary with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(config["data_dir"] / f"gridsearch_results_{timestamp}.csv", index=False)
    logger.info("GridSearchCV results saved.")

    # Cross-validated predictions for training set (for inspection)
    cv_preds = cross_val_predict(grid_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / f"cv_scores_{timestamp}.csv", index=False)
    logger.info("Cross-validation scores saved.")

    # Load test data
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)

    # Predict on test set with progress bar
    logger.info("Predicting on test set...")
    test_preds = np.array([grid_search.predict(img.reshape(1, -1))[0] for img in tqdm(test_images)])

    # Save test predictions
    save_results(test_preds)
    logger.info("Predictions saved to prediction.csv")
