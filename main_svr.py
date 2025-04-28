from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, uniform
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

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('regressor', SVR())
    ])

    # Randomized hyperparameter distributions for SVR
    param_dist = {
        'pca__n_components': [30, 50, 70],
        'regressor__C': loguniform(1e-2, 1e3),
        'regressor__epsilon': uniform(0.01, 1.0),
        'regressor__kernel': ['rbf', 'linear'],
        'regressor__gamma': ['scale', 'auto']
    }

    # K-Fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # RandomizedSearchCV
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

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {random_search.best_params_}")

    # Evaluate on validation set
    val_preds = random_search.predict(X_val)
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(random_search.best_estimator_, config["data_dir"] / "pipeline_random_svr.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline_random_svr.joblib")

    # Save RandomizedSearchCV results to CSV
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "randomsearch_results_svr.csv", index=False)
    print("[INFO]: RandomizedSearchCV results saved to randomsearch_results_svr.csv")

    # Cross-validated predictions for training set
    cv_preds = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / "cv_scores_random_svr.csv", index=False)
    print("[INFO]: Cross-validation scores saved to cv_scores_random_svr.csv")

    # Predict on test data
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)
    test_preds = random_search.predict(test_images)

    # Save test predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction_random_svr.csv")
