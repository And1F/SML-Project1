from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import r2_score

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    training_images, training_distances = load_dataset(config)
    testing_images, testing_distances = load_dataset(config, "public_test")

    # Split the training dataset into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(training_images, training_distances, test_size=0.03, random_state=42)

    # Pipeline
    #scaler = QuantileTransformer() (7, 0.01) 12.6
    #scaler = Normalizer() #(C= 10, epsilon=0.001) ~13.8 (test) (35 scale)
    #scaler = MinMaxScaler() #(C=9.5, epsilon=0.011) ~13.2 ~13.8 (26 Scale)
    #scaler = StandardScaler() #(C=5.75, epsilon = [0.0014-0.0017]) ~12.4 (public) (31 Scale)
    #scaler = RobustScaler() #(C = 6.5, epsilon=0.002, gamma=0.021)  (public)  (34 Scale)
    scaler = PowerTransformer() #(C = 2, epsilon=0.0008, gamma=0.009)
    #pca = PCA(whiten=False)
    svr_regressor = SVR(C=3, epsilon=0.002, gamma=0.01)  # Using RBF kernel
    #stdimagines
    
    pipe = Pipeline(steps=[("scaler", scaler), ("svr", svr_regressor)])
    #("scaler", scaler),

    
    # Fit the pipeline with training data
    pipe.fit(X_train, y_train)
    #joblib.dump(pipe, 'trained_model_Robust.pkl')
    """
    
    # Grid search parameters
    param_grid = {
        #"pca__n_components": [60, 61, 62],
        #"scaler": [QuantileTransformer(), MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer(), MaxAbsScaler()],
        #"svr__C": [4,4.5, 5, 5.5, 6, 7],  # Regularization parameter
        #"svr__epsilon": [0.01, 0.015, 0.02, 0.025, 0.03, 0.04],  # Epsilon parameter in the SVR model
        #"svr__kernel": ['rbf', 'linear', 'poly'],
        "svr__gamma": [0.005, 0.01, 0.015, 0.02]
    }
    
    # Grid search
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters found during grid search
    print("Best parameters found during grid search:")
    print(grid_search.best_params_)
    """

    # Evaluation X_test
    print("X_testset:")
    distances_pred = pipe.predict(X_test)
    score = r2_score(y_test, distances_pred)
    print_results(y_test, distances_pred)

    
    # Evalution public_test
    print("public_testset:")
    distances_pred_public = pipe.predict(testing_images)
    score = r2_score(testing_distances, distances_pred_public)
    print_results(testing_distances, distances_pred_public)
    

    """
    #Evalution saved model
    print("saved Model:")
    loaded_model = joblib.load('trained_model_without_pca.pkl')#inport model
    
    # Evaluation X_test
    print("X_testset:")
    distances_pred_2 = loaded_model.predict(X_test)
    score = r2_score(y_test, distances_pred_2)
    print_results(y_test, distances_pred_2)

    # Evalution public_test
    print("public_testset:")
    distances_pred_public_2 = loaded_model.predict(testing_images)
    score = r2_score(testing_distances, distances_pred_public_2)
    print_results(testing_distances, distances_pred_public_2)
    """

    # Save the results
    private_images = load_private_test_dataset(config)
    pred_private = pipe.predict(private_images)
    save_results(pred_private)
    # save_results(private_test_pred)