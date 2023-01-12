# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:57:56 2023

@author: ameimand
"""

import streamlit as st

import numpy as np
import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

from omnixai.explainers.tabular import TabularExplainer
from omnixai.data.tabular import Tabular

import dill as pickle

filename = 'Profitability_Prediction.sav'
model = pickle.load(open(filename, 'rb'))

filename = 'explainers.sav'
explainers = pickle.load(open(filename, 'rb'))


st.title("Resturant Profitibility Insight")
st.subheader("How to imporve the annual profitibility of your resturant")

#col1, col2 = st.columns([1, 1])


#with col1:
#    st.write("""To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate""")

#with col2:
st.write("""Please enter the characteristic of your restaurant to generate insight""")


st.subheader("Below you could find the profitibility prediction result along with factors contribution: ")



######################
#sidebar layout
######################

st.sidebar.title("Resturant Parameters")

#input features
Area  = st.sidebar.radio("Select restaurant location: ", ('Downtown', 'Suburbs'))
Age  = st.sidebar.radio("Select restaurant age: ", ('Developed', 'Well-Known','Recent'))
Type  = st.sidebar.radio("Select restaurant type: ", ('Cafeteria', 'Banquet','Full Service Restaurant'))
Price_Range =st.sidebar.selectbox('Please the pricen range', ("$","$$","$$$","$$$$","$$$$"))
Capacity =st.sidebar.selectbox('Please capacity', ("No seat","<10","10-15","15-20","20-30","30-50","50-70","70-90","90-100","100+"))
Item =st.sidebar.selectbox('Please number of items in the menu', ("Salad bar only","Salad & Sandwich only","< 5","5-7","7-10","10-15","15+"))



#predict button

btn_predict = st.sidebar.button("Generate Insight")

Insight = st.sidebar.selectbox('Insighit generation method', ("shap","lime"))

if btn_predict:
    
    input_data = np.column_stack([Area, Age, Type, Price_Range, Capacity, Item])
    X = pd.DataFrame(input_data,columns=['Area', 'Age', 'Type','Price Range','Capacity','Number of Menu Items'])
    pred = model.predict(X)
    value = "{:,.2f}".format(pred[0]/1000)
    
    tabular_data = Tabular(
    data=X,
    categorical_columns=['Area', 'Age', 'Type','Price Range','Capacity','Number of Menu Items']
 
        )
    explanations = explainers.explain(tabular_data)
    
    result = pd.DataFrame(data=np.column_stack([explanations[Insight].get_explanations()[0]['features'],explanations[Insight].get_explanations()[0]['scores']]),columns=['Features','Scores'])
    result = result.astype({'Scores': 'float64'})
    data = pd.DataFrame([[float(i)] for i in explanations[Insight].get_explanations()[0]['scores']],
                    index=[i for i in explanations[Insight].get_explanations()[0]['features']],
                    columns=['values'])
    data['positive'] = data['values'] > 0
    
    st.success('Proftibiltiy is $'+value+"K")
    st.pyplot(data['values'].plot(kind='barh', color=data.positive.map({True: 'b', False: 'r'})).figure)


    
