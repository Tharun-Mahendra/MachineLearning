# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import(confusion_matrix,accuracy_score,classification_report,roc_curve,roc_auc_score)
import seaborn as sns


st.title("Decision Tree Classifier")

# Uploading File
file=st.file_uploader('Upload Your File for Model Building',type=['csv'])
if file is not None:
    # Load 
    data=pd.read_csv(file)
    st.write('- Preview')
    st.dataframe(data.head())
    
    # FeatureSelection
    x=data.iloc[:,2:4]
    y=data.iloc[:,-1]
    
    # SplittingData
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

    # model building
    model=DecisionTreeClassifier()
    model.fit(x_train,y_train)

    # prediction
    y_pred=model.predict(x_test)
    y_prob=model.predict_proba(x_test)[:,1]

    # Metrics
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig_cm)

    ac = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {ac:.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write(f"**Training Accuracy (Bias):** {model.score(x_train, y_train):.2f}")
    st.write(f"**Testing Accuracy (Variance):** {model.score(x_test, y_test):.2f}")

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    st.write(f"**AUC Score:** {auc_score:.2f}")

    st.subheader("ROC Curve")
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc_score:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig_roc)