#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:54:05 2019

@author: pierre
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import uniform
from plotnine import *

#Create URL for Coaches dataset from github
url = 'https://raw.githubusercontent.com/2SUBDA/IST_718/master/Coaches9.csv'
#Create dataframe of Coaches dataset
df = pd.read_csv(url)
#View first 5 rows of dataset
print(df.head(5))

#Remove '$' and ',' from columns in order to change string to numeric in this 
#function
def cleanColumns(a):
  a = a.map(lambda x: x.lstrip('$').rstrip())
  a = [x.replace(',','') for x in a]
  a = [x.replace('--','') for x in a]
  a = pd.to_numeric(a)
  return a

#Run columns through created function
df['SchoolPay'] = cleanColumns(df['SchoolPay'])
df['TotalPay'] = cleanColumns(df['TotalPay'])
df['Bonus'] = cleanColumns(df['Bonus'])
df['BonusPaid'] = cleanColumns(df['BonusPaid'])
df['AssistantPay'] = cleanColumns(df['AssistantPay'])
df['Buyout'] = cleanColumns(df['Buyout'])

#Read in stadium size data
url1 = 'https://raw.githubusercontent.com/gboeing/data-visualization/master/ncaa-football-stadiums/data/stadiums-geocoded.csv'
stadiumSize = pd.read_csv(url1)

#Read in win/loss records
record = pd.read_csv('wl.csv')

#Read in graduation rates
gradRates = pd.read_csv('graduation.csv')

#Rename columns in order to merge dataframes
stadiumSize.rename(columns={'team':'School'},inplace=True)
gradRates.rename(columns={'SCL_NAME':'School'},inplace=True)

#Merge coach and record dataframes
df2 = df.merge(record, left_on='School',
               right_on='School',
               how='outer',
               suffixes=["","_match"])

#Find values that didnt match
missingSchools = df2[df2.W.isnull()]
missingSchools

#Change school names in the record dataframe and re-merge dataframes
record.School[110] = 'Alabama at Birmingham'
record.School[111] = 'Central Florida'
record.School[58] = 'Miami (Fla.)'
record.School[59] = 'Miami (Ohio)'
record.School[82] = 'Mississippi'
record.School[113] = 'Nevada-Las Vegas'
record.School[114] = 'Southern California'
record.School[96] = 'Southern Mississippi'
record.School[99] = 'Texas Christian'
record.School[117] = 'Texas-El Paso'
record.School[118] = 'Texas-San Antonio'

#Re-merge dataframes 
df2 = df.merge(record, left_on='School',
               right_on='School',
               how='outer',
               suffixes=["","_match"])

#Delete row not in original data frame
df2 = df2.drop([129])

#Clean colums from grad rates dataframe
gradRates.columns
gradRates = gradRates.drop(columns=['SCL_UNITID', 'SCL_DIVISION', 'SCL_SUBDIVISION',
       'SCL_CONFERENCE', 'DIV1_FB_CONFERENCE', 'SCL_HBCU', 'SCL_PRIVATE',
       'SPORT', 'SPONSORED',])

#Drop part of string "University", "of", "," and whitespace
gradRates.School = [x.replace('University','') for x in gradRates.School]
gradRates.School = [x.replace(' of ','') for x in gradRates.School]
gradRates.School = [x.replace('The','') for x in gradRates.School]
gradRates['School'] = gradRates['School'].str.split(',').str[0]
gradRates.School = gradRates.School.str.rstrip() 
gradRates.School = gradRates.School.str.lstrip() 

#Merge coach and grad rates dataframes
df3 = df2.merge(gradRates, left_on='School',
               right_on='School',
               how='outer',
               suffixes=["","_match"])

#Find schools that didnt match
missingSchools = df3[df3.GSR.isnull()]
missingSchools

#Adjust names for missing schools and remerge data frames
gradRates.School[62] = 'Air Force'
gradRates.School[63] = 'Army'
gradRates.School[92] = 'Miami (Fla.)'
gradRates.School[30] = 'Miami (Ohio)'
gradRates.School[8] = 'Bowling Green'
gradRates.School[98] = 'Nevada-Las Vegas'
gradRates.School[65] = 'Buffalo'
gradRates.School[57] = 'Charlotte'
gradRates.School[10] = 'Fresno State'
gradRates.School[113] = 'Texas-El Paso'
gradRates.School[114] = 'Texas-San Antonio'
gradRates.School[20] = 'Georgia Tech'
gradRates.School[82] = 'Illinois'
gradRates.School[86] = 'Louisiana-Lafayette'
gradRates.School[87] = 'Louisiana-Monroe'
gradRates.School[27] = 'LSU'
gradRates.School[32] = 'Middle Tennessee'
gradRates.School[64] = 'Navy'
gradRates.School[42] = 'Penn State'
gradRates.School[112] = 'Texas'
gradRates.School[72] = 'UCLA'
gradRates.School[123] = 'Virginia Tech'
gradRates.School[119] = 'Wisconsin'

#Re-merge coach and grad rates dataframes
df3 = df2.merge(gradRates, left_on='School',
               right_on='School',
               how='outer',
               suffixes=["","_match"])

#Liberty and Coastal Carolina were not included in my grad #rates dataset. Manually add the 2 scores
df3.FED_RATE[50] = 66
df3.GSR[50] = 77

df3.FED_RATE[24] = 58
df3.GSR[24] = 73

#Delete rows not in original data frame
df3 = df3.drop([129,130])

#Merge coach and stadium size dataframes
df4 = df3.merge(stadiumSize, left_on='School',
               right_on='School',
               how='outer',
               suffixes=["","_match"])

#Find schools that didnt match
missingSchools = df4[df4.capacity.isnull()]
missingSchools

#Add missing schools to stadium capacity data frame
stadiumSize.School[68] = 'Central Florida'
stadiumSize.School[46] = 'North Carolina State'
stadiumSize.School[31] = 'Miami (Fla.)'
stadiumSize.School[117] = 'Miami (Ohio)'
stadiumSize.School[33] = 'Brigham Young'
stadiumSize.School[83] = 'Nevada-Las Vegas'
stadiumSize.School[121] = 'Florida International'
stadiumSize.School[114] = 'Northern Illinois'
stadiumSize.School[30] = 'South Florida'
stadiumSize.School[56] = 'Texas-El Paso'
stadiumSize.School[32] = 'Texas-San Antonio'
stadiumSize.School[71] = 'Texas Christian'
stadiumSize.School[92] = 'Southern Methodist'
stadiumSize.School[84] = 'Southern Mississippi'
stadiumSize.School[125] = 'Massachusetts'
stadiumSize.School[157] = 'Liberty'
stadiumSize.School[213] = 'Coastal Carolina'

#Re-merge coach and stadium size dataframes
df4 = df3.merge(stadiumSize, left_on='School',
               right_on='School',
               how='outer',
               suffixes=["","_match"])

#Find schools that didnt match
missingSchools = df4[df4.TotalPay.isnull()]
missingSchools

#UAB didnt have a football program during that year. Manually entered data points
df4.capacity[3] = 71594
df4.latitude[3] = 33.5115
df4.longitude[3] = -86.8425

#Drop extra rows
df4 = df4.drop(df4.index[129:254,])

#Drop rows with NA data
df4 = df4.drop([12,16,91,99],axis=0)

#Drop unnecessary columns
df4 = df4.drop(['AssistantPay','conference','built','expanded','div'],axis=1)

#Add a point diferential column
df4['pointDiff'] = df4['PF'] - df4['PA']

#Quick description
df4.describe()

#Summary stats
print('School Pay mean is ', round(df4.SchoolPay.mean(),2))
print('Total Pay mean is ', round(df4.TotalPay.mean(),2))
print('Bonus mean is ', round(df4.Bonus.mean(),2))
print('Bonus Paid mean is ', round(df4.BonusPaid.mean(),2))
print('Buyout mean is ', round(df4.Buyout.mean(),2))
print('Points scored mean is ', round(df4.PF.mean(),2))
print('Stadium capacity mean is ', round(df4.capacity.mean(),2))

#Top 5 paid coaches
top5 = df4.sort_values(['TotalPay'], ascending=False).head()
print('The top 5 paid coaches are:', '\n', top5[['Coach','TotalPay','Conference']])

#Bottom 5 paid coaches
bottom5 = df4.sort_values(['TotalPay'], ascending=False).tail()
print('The bottom 5 paid coaches are:', '\n', bottom5[['Coach','TotalPay','Conference']])

#Boxplot by conference vs pay
bp = ggplot(df4, aes('Conference','SchoolPay')) + geom_boxplot() + labs(x='Conference',y='School Pay')
bp

#Scatter plot by PPG vs pay with total points
sp = (ggplot(df4, aes('PF','SchoolPay')) + geom_point(aes(size = 'W')) + 
labs(x='Total Points Scored',y='School Pay'))
sp = sp + geom_smooth(method = 'lm')
sp

#Explore data by conference
conferenceView = df4.groupby(['Conference'])['TotalPay', 'W', 'L', 'PF', 'PA', 'FED_RATE','GSR', 'capacity', 'latitude', 'longitude'].agg('sum')

conferenceView = conferenceView.reset_index()

#Create charts exploring conference stats
cpf = ggplot(conferenceView,aes('Conference','PF')) + geom_col()

cw = ggplot(conferenceView,aes('Conference','W')) + geom_col()

cgr = ggplot(conferenceView,aes('Conference','GSR')) + geom_col()

cs = ggplot(conferenceView,aes('Conference','capacity')) + geom_col()

print(cpf,cw,cgr,cs)

#Create corralation matrix
plt.matshow(df4[['SchoolPay','TotalPay','Bonus','BonusPaid','W','L','PF','PA','pointDiff','FED_RATE','GSR',
                 'capacity']].corr())
plt.xticks(range(len(df4[['SchoolPay','TotalPay','Bonus',
                          'BonusPaid','W','L','PF','PA','pointDiff','FED_RATE','GSR',
                          'capacity']].columns)), df4[['SchoolPay','TotalPay','Bonus','BonusPaid',
    'W','L','PF','PA','pointDiff','FED_RATE','GSR',
    'capacity']].corr().columns, rotation='vertical');
plt.yticks(range(len(df4[['SchoolPay','TotalPay','Bonus',
                          'BonusPaid','W','L','PF','PA','pointDiff','FED_RATE','GSR','capacity']].columns)), df4[['SchoolPay','TotalPay','Bonus','BonusPaid','W','L','PF','PA','pointDiff','FED_RATE','GSR','capacity']].corr().columns);

#Create a linear model
#Create inputs 
myModel = str('(TotalPay-BonusPaid) ~ Conference +  W + L + PF + PA + pointDiff + FED_RATE + GSR + capacity')

#Create model and print
train_model_fit = smf.ols(myModel, data = df4).fit()
print(train_model_fit.summary())

#Create recommendations
df4['Prediction'] = train_model_fit.predict(df4)



#Create dataframe for map by median salary
confMap = df4.groupby(['Conference'])['TotalPay'].agg('median')
confMap = pd.DataFrame(confMap)
confMap = confMap.reset_index()

#Add lat and lon columns
confMap['ZIP'] = ('02903','27407','75062','60018','75019','46204','44113','80921', '94107','35203','70112')

confMap['ZIP'] = pd.to_numeric(confMap['ZIP'])

#Import zip file with lat and long data
zip = pd.read_csv('zip_lat_long.csv')

#Merge zip with confMap
confMap = confMap.merge(zip, left_on='ZIP',
               right_on='ZIP',
               how='outer',
               suffixes=["","_match"])

confMap = confMap.drop(confMap.index[12:33144,],axis=0)

confMap.rename(columns={'ZIP':'Code'},inplace=True)




