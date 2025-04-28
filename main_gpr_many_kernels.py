from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, RationalQuadratic, Matern, DotProduct, WhiteKernel, ConstantKernel as C
)
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd
from scipy.stats import uniform

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    def kernel_sampler():
        kernels = []
        for _ in range(50):  # Sample 50 different kernel combinations
            ls = np.random.uniform(0.1, 5.0)
            alpha = np.random.uniform(0.1, 5.0)
            nu = np.random.choice([0.5, 1.5, 2.5])
            noise = np.random.uniform(1e-3, 1.0)

            kernels.extend([
                C(1.0) * RBF(length_scale=ls),
                C(1.0) * RationalQuadratic(length_scale=ls, alpha=alpha),
                C(1.0) * Matern(length_scale=ls, nu=nu),
                C(1.0) * (RBF(length_scale=ls) + WhiteKernel(noise_level=noise)),
                C(1.0) * (RationalQuadratic(length_scale=ls, alpha=alpha) + WhiteKernel(noise_level=noise)),
                C(1.0) * (DotProduct() + WhiteKernel(noise_level=noise)),
            ])
        return kernels

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('regressor', GaussianProcessRegressor(normalize_y=True, random_state=42))
    ])

    param_dist = {
        'pca__n_components': [20, 30, 40, 50, 60, 70],
        'regressor__alpha': uniform(1e-4, 1e-1),
        'regressor__kernel': kernel_sampler()
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,  # You can increase this to explore more combinations
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {random_search.best_params_}")

    # Evaluate
    val_preds = random_search.predict(X_val)
    print_results(y_val, val_preds)

    joblib.dump(random_search.best_estimator_, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline.joblib")

    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "randomsearch_results.csv", index=False)
    print("[INFO]: RandomizedSearchCV results saved to randomsearch_results.csv")

    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / "cv_scores.csv", index=False)
    print("[INFO]: Cross-validation scores saved to cv_scores.csv")

    test_images = np.array(load_test_dataset(config))
    test_preds = random_search.predict(test_images)
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction.csv")
