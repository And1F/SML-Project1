from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

if __name__ == "__main__":
    # Load configs
    config = load_config()

    # Load dataset
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Build pipeline: scaler -> PCA -> Ridge Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),  # adjust n_components based on explained variance if you like
        ('regressor', Ridge(alpha=1.0))
    ])

    # Train pipeline
    pipeline.fit(X_train, y_train)
    print("[INFO]: Model trained.")

    # Evaluate on validation set
    val_preds = pipeline.predict(X_val)
    print_results(y_val, val_preds)

    # Save pipeline
    joblib.dump(pipeline, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Pipeline saved to data/pipeline.joblib")

    # Load test data
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)

    # Predict test set
    test_preds = pipeline.predict(test_images)

    # Save test results
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction.csv")