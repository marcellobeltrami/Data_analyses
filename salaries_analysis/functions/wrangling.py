import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Takes a df with removed/processed Nans and output a scaled and encoded dataframe.
def preprocessing(df: pd.DataFrame): 
    df_cat_col = []
    df_num_col = []

    for col in df.columns: 
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            df_cat_col.append(col)
        else:
            df_num_col.append(col)

    preprocessed_df = pd.DataFrame()

    for cat in df_cat_col:
        preprocessed_df[cat] = (LabelEncoder().fit_transform(df[cat]))

    for num in df_num_col:
        preprocessed_df[num] = (RobustScaler().fit_transform(df[[num]]))

    return (preprocessed_df, LabelEncoder())



