from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_regression_model(X_train, y_train):
    """
    Train a multiple linear regression model
    Args:
        X_train: Features
        y_train: Target
    Returns:
        model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Train a decision tree classifier with hyperparameter tuning
    Args:
        X_train: Features
        y_train: Labels
    Returns:
        model
    """
    param_grid = {
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_split': [2, 5, 10]
    }
    tree = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(tree, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

