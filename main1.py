from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import uniform, randint
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    config = load_config()

    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Kernel: (RBF * RQ) + White
    kernel = (RBF(length_scale=1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selectk', SelectKBest(score_func=f_regression)),
        ('pca', PCA()),
        ('regressor', gpr)
    ])

    # Get kernel param keys for debugging
    print("\n[DEBUG]: Kernel parameters:")
    for key in gpr.get_params().keys():
        print(key)

    # Parameter distributions
    param_dist = {
        'selectk__k': randint(20, 150),
        'pca__n_components': randint(10, 100),
        'regressor__alpha': uniform(1e-4, 1e-1),
        'regressor__kernel__k1__k1__length_scale': uniform(0.1, 10.0),   # RBF
        'regressor__kernel__k1__k2__length_scale': uniform(0.1, 10.0),   # RQ
        'regressor__kernel__k1__k2__alpha': uniform(0.1, 5.0),           # RQ
        'regressor__kernel__k2__noise_level': uniform(1e-5, 1e-1)        # WhiteKernel
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

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

    random_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters: {random_search.best_params_}")

    val_preds = random_search.predict(X_val)
    print_results(y_val, val_preds)

    joblib.dump(random_search.best_estimator_, config["data_dir"] / "pipeline_rbf_rq.joblib")
    pd.DataFrame(random_search.cv_results_).to_csv(config["data_dir"] / "randomsearch_results_rbf_rq.csv", index=False)

    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": np.abs(y_train - cv_preds)
    }).to_csv(config["data_dir"] / "cv_scores_rbf_rq.csv", index=False)

    test_images = load_test_dataset(config)
    save_results(random_search.predict(np.array(test_images)))
