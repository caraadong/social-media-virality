import scipy.stats as stats
import pandas as pd

def perform_anova(df, group_col, value_col):
    """
    Perform one way ANOVA test
    Args:
        df: Dataset
        group_col: Grouping variable
        value_col: Numeric target
    Returns:
        p_value
    """
    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)
    return p_value

def perform_chi_square(df, col1, col2):
    """
    Perform chi-square test of independence
    Args:
        df: Dataset
        col1: First categorical variable
        col2: Second categorical variable
    Returns:
        p_value
    """
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return p
