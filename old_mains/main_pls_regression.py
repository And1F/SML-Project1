from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
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

    # Define pipeline: scaler -> PLS Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', PLSRegression())
    ])

    # Define hyperparameter grid
    param_grid = {
        'regressor__n_components': [5, 10, 20, 30]
    }

    # K-Fold cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2, return_train_score=True)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {grid_search.best_params_}")

    # Evaluate on validation set
    val_preds = grid_search.predict(X_val)
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(grid_search.best_estimator_, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline.joblib")

    # Save GridSearchCV results to CSV
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(config["data_dir"] / "gridsearch_results.csv", index=False)
    print("[INFO]: GridSearchCV results saved to gridsearch_results.csv")

    # Cross-validated predictions for training set
    cv_preds = cross_val_predict(grid_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
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
    test_preds = grid_search.predict(test_images)

    # Save test predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction.csv")
