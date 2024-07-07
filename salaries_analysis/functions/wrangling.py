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
    encodings_cat = pd.DataFrame()
    label_encoders = {}

    # Encode categorical columns
    for cat in df_cat_col:
        le = LabelEncoder()
        encoded_label = le.fit_transform(df[cat])
        preprocessed_df[cat] = encoded_label
        encodings_cat[cat] = encoded_label
        label_encoders[cat] = le

    # Scale numerical columns
    for num in df_num_col:
        preprocessed_df[num] = RobustScaler().fit_transform(df[[num]])

    return preprocessed_df, encodings_cat, label_encoders

# Function to decode categorical columns
def decode_column(encoded_df: pd.DataFrame, column: str, label_encoder: LabelEncoder):
    decoded_column = label_encoder.inverse_transform(encoded_df[column])
    return decoded_column





