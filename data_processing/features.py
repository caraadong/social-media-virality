def select_features(df, target_variable='Likes'):
    """
    Select features and target variable for regression
    Args:
        df: Preprocessed dataset
        target_variable: Target variable
    Returns:
        X, y
    """
    X = df.drop(columns=[target_variable, 'Engagement_Label'])
    y = df[target_variable]
    return X, y

def select_classification_features(df):
    """
    Select features and binary target variable for classification
    Args:
        df: Preprocessed dataset
    Returns:
        X, y
    """
    X = df.drop(columns=['Likes', 'Engagement_Label'])
    y = df['Engagement_Label']
    return X, y
