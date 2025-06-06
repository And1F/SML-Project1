from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import uniform, randint
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

    # Define kernel (RationalQuadratic + WhiteKernel)
    base_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1.0)
    regressor = GaussianProcessRegressor(kernel=base_kernel, normalize_y=True, random_state=42)

    # Build pipeline with SelectKBest before PCA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selectk', SelectKBest(score_func=f_regression)),
        ('pca', PCA()),
        ('regressor', regressor)
    ])

    # Define param distribution for RandomizedSearchCV
    param_dist = {
        'selectk__k': randint(20, 150),  # Number of features to select
        'pca__n_components': randint(30, 200),  # Number of PCA components
        'regressor__alpha': uniform(1e-10, 1),  # Alpha parameter for GPR
        'regressor__kernel__k1__length_scale': uniform(0.1, 5.0),  # RationalQuadratic length_scale
        'regressor__kernel__k1__alpha': uniform(0.1, 5.0),  # RationalQuadratic alpha
        'regressor__kernel__k2__noise_level': uniform(1e-5, 1e-1)  # WhiteKernel noise_level
    }

    # K-Fold cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=-1,
        return_train_score=True,
        random_state=42
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {random_search.best_params_}")

    # Evaluate on validation set
    val_preds = random_search.predict(X_val)
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(random_search.best_estimator_, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline.joblib")

    # Save all search results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "randomsearch_results.csv", index=False)
    print("[INFO]: RandomSearchCV results saved to randomsearch_results.csv")

    # Cross-validated predictions on training set
    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
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
    test_preds = random_search.predict(test_images)

    # Save test predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction.csv")
