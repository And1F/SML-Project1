from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ExpSineSquared, RBF, RationalQuadratic,
    ConstantKernel as C, WhiteKernel
)
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Define individual kernels
    periodic_kernel = ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                     length_scale_bounds=(1e-5, 1e5),
                                     periodicity_bounds=(0.1, 10.0))
    rbf_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
    rq_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0,
                                  length_scale_bounds=(1e-5, 1e5),
                                  alpha_bounds=(0.1, 10.0))
    white_kernel = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))

    # Kernel: C * (periodic * rbf + rq) + white
    combined_kernel = C(1.0, (1e-3, 1e3)) * ((periodic_kernel * rbf_kernel) + rq_kernel)
    full_kernel = combined_kernel + white_kernel

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=50, random_state=42)),
        ('regressor', GaussianProcessRegressor(kernel=full_kernel, alpha=1e-2, random_state=42, normalize_y=True))
    ])

    param_dist = {
        'pca__n_components': [30, 50, 70, 100],
        'regressor__alpha': uniform(1e-3, 0.5),

        # ConstantKernel
        'regressor__kernel__k1__k1__constant_value': uniform(0.1, 5.0),

        # ExpSineSquared
        'regressor__kernel__k1__k2__k1__k1__length_scale': uniform(0.1, 5.0),
        'regressor__kernel__k1__k2__k1__k1__periodicity': uniform(0.5, 5.0),

        # RBF
        'regressor__kernel__k1__k2__k1__k2__length_scale': uniform(0.1, 5.0),

        # RationalQuadratic
        'regressor__kernel__k1__k2__k2__length_scale': uniform(0.1, 5.0),
        'regressor__kernel__k1__k2__k2__alpha': uniform(0.1, 10.0),

        # WhiteKernel
        'regressor__kernel__k2__noise_level': uniform(1e-5, 1e2)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {random_search.best_params_}")

    val_preds = random_search.predict(X_val)
    print_results(y_val, val_preds)

    joblib.dump(random_search.best_estimator_, config["data_dir"] / "pipeline_random_exp_rbf_rq_white.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline_random_exp_rbf_rq_white.joblib")

    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "randomsearch_results_exp_rbf_rq_white.csv", index=False)
    print("[INFO]: RandomizedSearchCV results saved to randomsearch_results_exp_rbf_rq_white.csv")

    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / "cv_scores_random_exp_rbf_rq_white.csv", index=False)
    print("[INFO]: Cross-validation scores saved to cv_scores_random_exp_rbf_rq_white.csv")

    test_images = load_test_dataset(config)
    test_images = np.array(test_images)
    test_preds = random_search.predict(test_images)
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction_random_exp_rbf_rq_white.csv")
