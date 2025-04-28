
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import pandas as pd

def plot_regression_coefficients(model, feature_names, output_path=None):
    """ 
    Plot regression coefficients
    Args:
        model: trained LinearRegression model
        feature_names: Feature names
        output_path: optional save path
    """
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
    plt.figure(figsize=(10,6))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature')
    plt.title('Regression Coefficients')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def plot_decision_tree_model(tree_model, feature_names, class_names, output_path = None):
    """
    Plot decision tree diagram
    Args: 
        tree_model: Trained DecisionTreeClassifier
        feature_names: feature names
        class_names: class names
        output_path: optional save path
    """
    plt.figure(figsize=(12, 8))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True
    )
    plt.title("Decision Tree Classifier for Engagement Prediction")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
