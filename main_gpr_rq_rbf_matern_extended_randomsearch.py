from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
import joblib
import numpy as np
import pandas as pd

# Load configs
config = load_config()

# Load dataset
images, distances = load_dataset(config)
print(f"[INFO]: Dataset loaded with {len(images)} samples.")

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

# Define combined kernel: (C * (RQ + RBF + Matern)) + WhiteKernel
rq = RationalQuadratic(length_scale=1.0, alpha=1.0)
rbf = RBF(length_scale=1.0)
matern = Matern(length_scale=1.0, nu=1.5)

sum_kernel = rq + rbf + matern

combined_kernel = C(1.0) * sum_kernel + WhiteKernel(noise_level=1.0)


# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # overwritten during search if needed
    ('pca', PCA(n_components=100)),
    ('regressor', GaussianProcessRegressor(
        kernel=combined_kernel,
        alpha=1e-2,
        random_state=42,
        normalize_y=True,
        n_restarts_optimizer=10
    ))
])

# Randomized hyperparameter distributions
param_dist = {
    'scaler': [StandardScaler(), RobustScaler()],
    'pca__n_components': [20, 30, 50, 70, 90],

    'regressor__alpha': uniform(1e-5, 1e-1),

    # ConstantKernel (scaling factor)
    'regressor__kernel__k1__k1__constant_value': uniform(0.1, 10.0),

    # RationalQuadratic (inside k1 -> k2 -> k1 -> k1)
    'regressor__kernel__k1__k2__k1__k1__length_scale': uniform(0.1, 10.0),
    'regressor__kernel__k1__k2__k1__k1__alpha': uniform(0.1, 10.0),

    # RBF (inside k1 -> k2 -> k1 -> k2)
    'regressor__kernel__k1__k2__k1__k2__length_scale': uniform(0.1, 10.0),

    # Matern (inside k1 -> k2 -> k2)
    'regressor__kernel__k1__k2__k2__length_scale': uniform(0.1, 10.0),
    'regressor__kernel__k1__k2__k2__nu': [0.5, 1.5, 2.5],

    # WhiteKernel (noise)
    'regressor__kernel__k2__noise_level': uniform(1e-5, 1e-1),
}


# K-Fold cross-validation
cv = KFold(n_splits=7, shuffle=True, random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=300,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)
print(f"[INFO]: Best parameters found: {random_search.best_params_}")

# Evaluate on validation set
val_preds = random_search.predict(X_val)
print_results(y_val, val_preds)

# Save best pipeline
joblib.dump(random_search.best_estimator_, config["data_dir"] / "pipeline_random_rq_rbf_matern_white.joblib")
print("[INFO]: Best pipeline saved to data/pipeline_random_rq_rbf_matern_white.joblib")

# Save RandomizedSearchCV results to CSV
results_df = pd.DataFrame(random_search.cv_results_)
results_df.to_csv(config["data_dir"] / "randomsearch_results_rq_rbf_matern_white.csv", index=False)
print("[INFO]: RandomizedSearchCV results saved to randomsearch_results_rq_rbf_matern_white.csv")

# Cross-validated predictions for training set
cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
cv_scores = np.abs(y_train - cv_preds)

cv_results_df = pd.DataFrame({
    "GroundTruth": y_train,
    "Prediction": cv_preds,
    "AbsoluteError": cv_scores
})
cv_results_df.to_csv(config["data_dir"] / "cv_scores_random_rq_rbf_matern_white.csv", index=False)
print("[INFO]: Cross-validation scores saved to cv_scores_random_rq_rbf_matern_white.csv")

# Predict on test data
test_images = load_test_dataset(config)
test_images = np.array(test_images)
test_preds = random_search.predict(test_images)

# Save test predictions
save_results(test_preds)
print("[INFO]: Predictions saved to prediction_random_rq_rbf_matern_white.csv")
