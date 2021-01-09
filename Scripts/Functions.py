# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:09:23 2021

@author: Rajesh Sharma <https://github.com/Rajesh-ML-Engg>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def customize_plots(plt_cust_prop):
    """
    Description: This function is created for providing the customized plotting properties.

    Parameters
    ----------
    custom_dict : str
        It serves as an indicator for providing the customized plotting properties of label,
        title, wedges and texts for various plots.

    Returns
    -------
    Dictionary with cutomized values of the required plot property.
    """
    
    if plt_cust_prop == 'label':
        label_dict = {'family':'Calibri','size':21,'style':'oblique','color':'coral'}
        return label_dict
    elif plt_cust_prop == 'title':
        title_dict = {'family':'Calibri','size':23,'style':'oblique','color':'magenta'}
        return title_dict
    elif plt_cust_prop =='wedge':
        wedge_dict = {'linewidth': 1, 'edgecolor': 'black'}
        return wedge_dict
    elif plt_cust_prop == 'txt':
        txt_dict = {'family':'Calibri','size':16,'style':'oblique','color':'k'}
        return txt_dict
    else:
        return label_dict
    
def compute_null_percentage(data_df,missing_type=False):
    """
    Description: This function is created for calculating the NULL values percentage in every feature.
    
    Input: It accepts below parameters:
        - `data_df: pandas dataframe`
                Dataframe with entire dataset
        - `missing_type`: str or boolean
                This acts as an indicator of missing values.
            
            NOTE :: If missing_type = False then,
                it will consider NaN as missing value 
            Else, consider the provided string as missing value.
            
    Return: It returns below object:
        -  `nulls_info_df: pandas dataframe`
                Dataframe containing nulls percentage feature-wise.    
    """
    cols = []
    null_percentage = []

    for col in data_df.columns:
        if missing_type:
            no_of_nulls = data_df[data_df[col] == missing_type].shape[0]
        else:
            no_of_nulls = data_df[data_df[col].isna()].shape[0]
        tot_rec = data_df.shape[0]
        nulls_percent = np.round((np.divide(no_of_nulls,tot_rec,dtype= np.float) * 100),2)
        cols.append(col)
        null_percentage.append(nulls_percent)
    
    nulls_info_df = pd.DataFrame({'Feature_Name':cols,'NULL Percentage':null_percentage})
    return nulls_info_df

def plot_null_values(df):
    """
    Description: This function is created for plotting the Nulls occurences in the dataset.
    
    Input: It accepts one parameter:
        `df`: Pandas Dataframe
            Dataframe containing the observations
            
    Return: None
        It plots the heatmap of null values in the dataset.
    """
    with plt.style.context('seaborn'):
        plt.figure(figsize=(15,12))
        sns.heatmap(data=pd.DataFrame(df.isnull()),cmap=ListedColormap(sns.color_palette('GnBu',10)),cbar=False)
        plt.xlabel('Features',fontdict=customize_plots(plt_cust_prop='label'))
        plt.ylabel('Record Indices',fontdict=customize_plots(plt_cust_prop='label'))
        plt.title('Missing Values in the Dataset',fontdict=customize_plots(plt_cust_prop='title'))
        plt.xticks(color='black',size=12,style='oblique')
        plt.yticks(color='black',size=10,style='oblique')
    plt.show()

def impute_std_hc_iud(df,cols):
    """
    Description: 
        This function is created for imputing -1 in STDs, HCs and IUDs features where a record contains blank value for them.
        `The idea behind this imputation is that if all these features are NULL then
         not even a doctor can diagnose any disease in such a patient.`
    
    Input Parameter: It accepts below two parameters:
        - df: pandas dataframe
            Dataframe containing observations
        
        - cols: list
            Python list of features that needs to be imputed 
    """
    for col in cols:
        df[col] = df.apply(lambda row: -1 if row['STDs'] in ['?',-1] and row['Hormonal Contraceptives'] in ['?',-1] and row['IUD'] in ['?',-1]\
                            else row[col],axis=1)