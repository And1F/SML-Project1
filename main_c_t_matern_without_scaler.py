from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
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

    # Define Matern * Constant kernel
    matern_kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)

    # Define pipeline without scaler
    pipeline = Pipeline([
        ('pca', PCA(n_components=50)),
        ('regressor', GaussianProcessRegressor(kernel=matern_kernel, alpha=1e-2, random_state=42, normalize_y=True))
    ])

    # Hyperparameter distributions
    param_dist = {
        'pca__n_components': [30, 50, 70],
        'regressor__alpha': uniform(1e-3, 0.1),
        'regressor__kernel__k1__constant_value': uniform(0.1, 5.0),
        'regressor__kernel__k2__length_scale': uniform(0.1, 5.0)
    }

    # K-Fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )

    # Fit
    random_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {random_search.best_params_}")

    # Evaluate on validation set
    val_preds = random_search.predict(X_val)
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(random_search.best_estimator_, config["data_dir"] / "pipeline_random_matern.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline_random_matern.joblib")

    # Save RandomizedSearchCV results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "randomsearch_results_matern.csv", index=False)
    print("[INFO]: RandomizedSearchCV results saved to randomsearch_results_matern.csv")

    # Cross-validated predictions
    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / "cv_scores_random_matern.csv", index=False)
    print("[INFO]: Cross-validation scores saved to cv_scores_random_matern.csv")

    # Predict on test data
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)
    test_preds = random_search.predict(test_images)

    # Save predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction_random_matern.csv")
