import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

def preprocessing(df):
    df_cat_col = []
    df_num_col = []

    for col in df.columns: 
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            df_cat_col.append(col)
        else:
            df_num_col.append(col)

    preprocessed_df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    encodings_cat = pd.DataFrame()
    label_encoders = {}

    # Encode categorical columns
    for cat in df_cat_col:
        le = LabelEncoder()
        encoded_label = le.fit(df[cat])
        preprocessed_df[cat] = encoded_label.transform(df[cat])
        encodings_cat[cat] = encoded_label
        label_encoders[cat] = le

    scalings_num = pd.DataFrame()
    num_scalers = {}

    # Scale numerical columns
    for num in df_num_col:
        rs = RobustScaler()
        scaled_values = rs.fit_transform(df[[num]])  # This returns a 2D array
        preprocessed_df[num] = scaled_values.flatten()  # Flatten to ensure it's 1D
        scalings_num[num] = scaled_values.flatten()
        num_scalers[num] = rs

    return preprocessed_df, encodings_cat, label_encoders, num_scalers

# Function to decode categorical columns
def decode_column(encoded_df: pd.DataFrame, column: str, label_encoder: LabelEncoder):
    decoded_column = label_encoder.inverse_transform(encoded_df[column])
    return decoded_column

# Concatenates dataframe by accordingly resetting indeces.
def c_concatenate(df1, df2): 
    df_1 = df1.reset_index(drop=True)
    df_2= df2.reset_index(drop=True)

    combined_df = pd.concat([df_1, df_2], axis=1, join='outer')

    return combined_df





