from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carries out Random Forest after splitting for training and testing. 
def RF_train_test (preprocessed_df: pd.DataFrame, prediction_col:str, randomizer:str = None, type="X"):
        
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_df.drop(columns=prediction_col), 
                                                    preprocessed_df[prediction_col], 
                                                    test_size=0.2, 
                                                    random_state=randomizer)
    # Carries out RF classifier
    if type == "C":
        rgrs = RandomForestClassifier()

        fitted_model = rgrs.fit(X_train,y_train)

        # Feature importances
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': fitted_model.feature_importances_
        })

        return fitted_model.score(X_test,y_test), importance_df.sort_values(by="Importance"), fitted_model
    
    # Carries out RF regressor
    elif  type == "R":
        rgrs = RandomForestRegressor()

        fitted_model = rgrs.fit(X_train,y_train)

        # Feature importances
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': fitted_model.feature_importances_
        })

        return fitted_model.score(X_test,y_test), importance_df.sort_values(by="Importance", ascending=True), fitted_model
    
    else: 
        KeyError("Specify correct RF model: R or C")


# Implement another supervised algorithm
def STR_train_test (preprocessed_df: pd.DataFrame, prediction_col:str, randomizer:str = None):
        
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_df.drop(columns=prediction_col), 
                                                    preprocessed_df[prediction_col], 
                                                    test_size=0.2, 
                                                    random_state=randomizer)
    
    '''
    rgrs = ...insert algorthm

    fitted_model = rgrs.fit(X_train,y_train)

    # Feature importances
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': fitted_model.feature_importances_
    })

    return (fitted_model.score(X_test,y_test), importance_df)
    '''

