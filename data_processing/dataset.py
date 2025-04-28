import pandas as pd

def load_raw_data(filepath):
    """
    Load the raw social media dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Load Dataset.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Perform basic preprocessing on the dataset:
    - create dummy variables
    - create engagement label

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Processed dataset
            
    """
    df = pd.get_dummies(df, columns=['Platform', 'Content_Type'])
    df['Engagement_Label'] = (df['Likes'] > df['Likes'].median()).astype(int)
    return df

