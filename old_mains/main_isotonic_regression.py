from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np
import pandas as pd

# Wrapper to integrate IsotonicRegression into sklearn Pipeline
class IsotonicRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, X, y):
        self.model.fit(X.ravel(), y)
        return self

    def predict(self, X):
        return self.model.predict(X.ravel())

if __name__ == "__main__":
    # Load configs
    config = load_config()

    # Load dataset
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Define pipeline: scaler -> PCA (1 comp) -> Isotonic Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=1)),
        ('regressor', IsotonicRegressorWrapper())
    ])

    # K-Fold cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated predictions for training set
    cv_preds = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=-1)
    cv_scores = np.abs(y_train - cv_preds)

    # Calculate and print cross-validation MAE
    cv_mae = mean_absolute_error(y_train, cv_preds)
    print(f"[INFO]: Cross-validated MAE: {cv_mae:.3f}")

    # Save CV results to CSV
    cv_results_df = pd.DataFrame({
        "GroundTruth": y_train,
        "Prediction": cv_preds,
        "AbsoluteError": cv_scores
    })
    cv_results_df.to_csv(config["data_dir"] / "cv_scores.csv", index=False)
    print("[INFO]: Cross-validation scores saved to cv_scores.csv")

    # Fit final model on full training data
    pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = pipeline.predict(X_val)
    print_results(y_val, val_preds)

    # Save final pipeline
    joblib.dump(pipeline, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Pipeline saved to data/pipeline.joblib")

    # Load test data
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)

    # Predict test set
    test_preds = pipeline.predict(test_images)

    # Save test predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction.csv")
