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

# ------------- Define your pipeline exactly like in your main code -------------
kernel = C(1.0) * (RationalQuadratic() + RBF() + Matern(nu=1.5)) + WhiteKernel()

pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=100)),
    ('regressor', GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.01,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42
    ))
])

# ------------- Define your param_distributions like you do for RandomizedSearchCV -------------
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

# ------------- Standalone Checking Code -------------
print("\n--- Available parameter names in pipeline ---")
valid_params = list(pipeline.get_params().keys())
for param in valid_params:
    print(param)

print("\n--- Checking param_distributions keys ---")
invalid_keys = []
for key in param_dist.keys():
    if key not in valid_params:
        invalid_keys.append(key)

if invalid_keys:
    print("\n⚠️ Invalid parameter keys found:")
    for key in invalid_keys:
        print(f" - {key}")
else:
    print("\n✅ All param_distributions keys are valid!")
