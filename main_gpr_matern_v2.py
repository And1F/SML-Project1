<<<<<<< HEAD
from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, ConstantKernel as C, WhiteKernel
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Combined kernel
    kernel = C(1.0, (1e-3, 1e3)) * (
        Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-5, 1e5)) +
        RationalQuadratic(length_scale=1.0, alpha=1.0,
                          length_scale_bounds=(1e-5, 1e5),
                          alpha_bounds=(0.1, 10.0))
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('kbest', SelectKBest(score_func=f_regression, k=100)),  # Select top 100 by default
        ('pca', PCA(n_components=50, random_state=42)),
        ('regressor', GaussianProcessRegressor(kernel=kernel, alpha=1e-2, random_state=42, normalize_y=False))
    ])

    param_dist = {
        'kbest__k': randint(50, 300),
        'pca__n_components': [30, 50, 70, 100],
        'regressor__alpha': uniform(1e-4, 0.2),

        # ConstantKernel
        'regressor__kernel__k1__k1__constant_value': uniform(0.1, 10.0),

        # Matern
        'regressor__kernel__k1__k2__k1__length_scale': uniform(0.1, 10.0),

        # RationalQuadratic
        'regressor__kernel__k1__k2__k2__length_scale': uniform(0.1, 10.0),
        'regressor__kernel__k1__k2__k2__alpha': uniform(0.1, 20.0),

        # WhiteKernel
        'regressor__kernel__k2__noise_level': uniform(1e-5, 1e1)
    }

    cv = KFold(n_splits=8, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,  # more iterations for finer search
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = config["data_dir"] / f"pipeline_best_{timestamp}.joblib"
    joblib.dump(random_search.best_estimator_, model_path)
    print(f"[INFO]: Best pipeline saved to {model_path}")

    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(config["data_dir"] / f"randomsearch_results_{timestamp}.csv", index=False)

    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / f"cv_scores_{timestamp}.csv", index=False)

    test_images = load_test_dataset(config)
    test_images = np.array(test_images)
    test_preds = random_search.predict(test_images)
    save_results(test_preds)
    print("[INFO]: Predictions saved.")
=======
from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, ConstantKernel as C, WhiteKernel
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Combined kernel
    kernel = C(1.0, (1e-3, 1e3)) * (
        Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-5, 1e5)) +
        RationalQuadratic(length_scale=1.0, alpha=1.0,
                          length_scale_bounds=(1e-5, 1e5),
                          alpha_bounds=(0.1, 10.0))
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('kbest', SelectKBest(score_func=f_regression, k=100)),  # Select top 100 by default
        ('pca', PCA(n_components=50, random_state=42)),
        ('regressor', GaussianProcessRegressor(kernel=kernel, alpha=1e-2, random_state=42, normalize_y=False))
    ])

    param_dist = {
        'kbest__k': randint(50, 300),
        'pca__n_components': [30, 50, 70, 100],
        'regressor__alpha': uniform(1e-4, 0.2),

        # ConstantKernel
        'regressor__kernel__k1__k1__constant_value': uniform(0.1, 10.0),

        # Matern
        'regressor__kernel__k1__k2__k1__length_scale': uniform(0.1, 10.0),

        # RationalQuadratic
        'regressor__kernel__k1__k2__k2__length_scale': uniform(0.1, 10.0),
        'regressor__kernel__k1__k2__k2__alpha': uniform(0.1, 20.0),

        # WhiteKernel
        'regressor__kernel__k2__noise_level': uniform(1e-5, 1e1)
    }

    cv = KFold(n_splits=8, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,  # more iterations for finer search
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = config["data_dir"] / f"pipeline_best_{timestamp}.joblib"
    joblib.dump(random_search.best_estimator_, model_path)
    print(f"[INFO]: Best pipeline saved to {model_path}")

    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(config["data_dir"] / f"randomsearch_results_{timestamp}.csv", index=False)

    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / f"cv_scores_{timestamp}.csv", index=False)

    test_images = load_test_dataset(config)
    test_images = np.array(test_images)
    test_preds = random_search.predict(test_images)
    save_results(test_preds)
    print("[INFO]: Predictions saved.")
>>>>>>> 65311bbca24a89bc8980352add82757395146059
