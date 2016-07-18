# Logistic regression 
# week4 assignment for Regression modeling in practice

# explanator variable: income per person, alcohol consumption, urban rate
# response variable: life expectancy

"""
Created on Mon Jul 18 11:34:27 2016

@author: taehee jeong
"""
# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sb

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)


#%% Load data
path='C:/Bigdata/Data Analysis and Interpretation/Dataset/GapMinder/'
data = pd.read_csv(path+'gapminder.csv', low_memory=False)

print data.columns

#convert to numeric format
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['alcconsumption'] = pd.to_numeric(data['alcconsumption'], errors='coerce')
data['urbanrate'] = pd.to_numeric(data['urbanrate'], errors='coerce')
data['lifeexpectancy'] = pd.to_numeric(data['lifeexpectancy'], errors='coerce')

data['lifeexpectancy'].describe()

# subset for only selecting variables
sub1 = data[['incomeperperson','alcconsumption','urbanrate', 'lifeexpectancy']]

#print 'life expectancy - 2 categories - binary'
#sub1['life_bin']=pd.qcut(sub1.lifeexpectancy, 2, labels=["0","1"])
#c1 = sub1['life_bin'].value_counts(sort=False, dropna=True)
#print(c1)
#sub1.drop('life_bin', axis=1, inplace=True)

def lifegroup(row):
    if row['lifeexpectancy'] < sub1['lifeexpectancy'].mean():
        return 0
    elif row['lifeexpectancy'] >= sub1['lifeexpectancy'].mean():
        return 1

sub1['life_bin']=sub1.apply(lambda row:lifegroup(row),axis=1)

sub1.head()

#cleaning na
sub2=sub1.dropna()

#%% logistic regression

lreg1 = smf.logit(formula = 'life_bin ~ alcconsumption', data = sub2).fit()
print (lreg1.summary())
# odds ratios
print ("Odds Ratios")
print (np.exp(lreg1.params))

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

# logistic regression with alcohol consumption and income
lreg2 = smf.logit(formula = 'life_bin ~ alcconsumption + incomeperperson', data = sub2).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

# logistic regression with alcohol consumption and urban rate
lreg2 = smf.logit(formula = 'life_bin ~ alcconsumption + urbanrate', data = sub2).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))


# logistic regression with alcol consumption and income, urban rate
lreg3 = smf.logit(formula = 'life_bin ~ alcconsumption + incomeperperson+ urbanrate', data = sub2).fit()
print (lreg3.summary())

# odd ratios with 95% confidence intervals
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))
