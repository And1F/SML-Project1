from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

if __name__ == "__main__":
    # Load configs
    config = load_config()

    # Load dataset
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Define pipeline steps
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),  # you can adjust this number later if needed
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=2000))
    ])

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'pca__n_components': [30, 50, 70],
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [None, 10, 20]
    }

    # K-Fold cross-validation strategy
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Define GridSearchCV with the pipeline
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)
    print(f"[INFO]: Best parameters found: {grid_search.best_params_}")

    # Evaluate on validation set
    val_preds = grid_search.predict(X_val)
    print_results(y_val, val_preds)

    # Save best pipeline
    joblib.dump(grid_search.best_estimator_, config["data_dir"] / "pipeline.joblib")
    print("[INFO]: Best pipeline saved to data/pipeline.joblib")

    # Load test data
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)

    # Predict test set distances
    test_preds = grid_search.predict(test_images)

    # Save test predictions
    save_results(test_preds)
    print("[INFO]: Predictions saved to prediction.csv")
