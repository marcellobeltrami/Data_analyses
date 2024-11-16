import streamlit as st

st.title("Salary prediction")
st.write("""Predicting salaries based on various features is crucial for informed decision-making in HR, finance, and economic planning. 
         This assignment aims to develop a robust salary prediction model using unsupervised learning, supervised learning, and neural networks. 
         Each method plays a specific role in achieving our goals.""")

st.write("""**Feature selection and Supervised Learning**
    
- Achieve Predictive Accuracy: Use algorithms like linear regression and decision trees.
- Evaluate Models: Measure performance with metrics like MSE and cross-validation. 
- Determine Feature Importance: Identify key predictors of salary. 
- Prepare Data: Detect and handle anomalies. 

**Unsupervised Learning**

- Identify Patterns and Clusters: Group similar job roles or experience levels. 
- Reduce Dimensionality: Focus on significant features. 


**Neural Networks**

- Model Complex Relationships: Capture intricate patterns and interactions.
- Ensure Flexibility and Scalability: Handle large datasets and many features.
- Goal: Create a neural network that is capable of predicting salary given the most important features. """)


st.title("Correlation Matrix")
st.image("./salaries_analysis/images/Corr_mx.png", use_column_width=True)


st.title("Feature Importance")
st.image("./salaries_analysis/images/Feat_importance.png", use_column_width=True)
st.write(""" Some features have been randomly removed, now use a RF algorithms to determine which ones are the most important ones.

Training a Random Forest Regressor to predict salary, removing least important features results in improved accuracy.

On average, the features selected model results in greater accuracy, with 4/5 iteration performing better than non features selected model.""")

st.title("Data structure and patterns")
st.image("./salaries_analysis/images/Designation_UMAP.png", use_column_width=True)
st.image("./salaries_analysis/images/UMAP_visualization.png", use_column_width=True)
st.image("./salaries_analysis/images/salary_umap.png", use_column_width=True)
st.write("""
Relationship between salary and designation can be noticed by connecting the two plots, clustering observed in UMAP is further confirmed using various clustering algorithms.  


- There are 2 main clusters of data analysts with a broad range of salaries.
- Senior analysts are payed more than data analysts.
- Associates make as much as Senior Analysts.  
- Managers have the second highest salary.
- Directors earn the most money. 

The top payed person is a director.

Furthermore, it can be inferred that age play a important role in salary earning, which is in line with what found with Gini scoring.""")

st.title("Conclusions and applications")

st.write(""" 
Using a Sequential Neural Network we can use the most important features to predict a likely salary someone  is likely to obtain givent the most important features provided in this dataset.
This is achieved by using a 4 layer regression model, trained from scratch. 
""")


