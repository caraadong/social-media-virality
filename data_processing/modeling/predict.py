from sklearn.metrics import mean_squared_error, accuracy_score

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression model
    Args:
        model: Trained LinearRegression
        X_test: Features
        y_test: Target
    Returns:
        mse 
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate classification model
    Args:
        model: Trained DecisionTreeClassifier
        X_test: Features
        y_test: Labels
    Returns:
        accuracy
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy
