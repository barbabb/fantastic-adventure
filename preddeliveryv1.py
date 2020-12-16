import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st

st.write("""
# NFL Margin and Confidence Predictor Models

Margin: Positive numbers are margins predicted in favor of the Home Team, negative margins for the Away Team

Confidence:

1 = Home Team classified as winner

0 = Pass, too close to classify either as outright winner

-1 = Away Team classified as winner


""")

team_dict = {'atl':'Atlanta Falcons', 'buf':'Buffalo Bills', 'car':'Carolina Panthers', 'chi':'Chicago Bears', 'cin':'Cincinnati Bengals', 'cle':'Cleveland Browns', 'clt':'Indianapolis Colts', 'crd':'Arizona Cardinals',
         'dal':'Dallas Cowboys', 'den':'Denver Broncos', 'det':'Detroit Lions', 'gnb':'Green Bay Packers', 'htx':'Houston Texans', 'jax':'Jacksonville Jaguars', 'kan':'Kansas City Chiefs', 'mia':'Miami Dolphins', 
         'min':'Minnesota Vikings', 'nor':'New Orleans Saints', 'nwe':'New England Patriots', 'nyg':'New York Giants', 'nyj':'New York Jets', 'oti':'Tennessee Titans', 'phi':'Philadelphia Eagles',
         'pit':'Pittsburgh Steelers', 'rai':'Las Vegas Raiders', 'ram':'Los Angeles Rams', 'rav':'Baltimore Ravens', 'sdg':'Los Angeles Chargers', 'sea':'Seattle Seahawks', 'sfo':'San Francisco 49ers',
         'tam':'Tampa Bay Buccaneers', 'was':'Washington Football Team'}

df_url = 'https://raw.githubusercontent.com/barbabb/fantastic-adventure/master/2020df_week14.csv'
df = pd.read_csv(df_url)
df['Opp_Name'] = df['Opp_Name'].astype('category')
df['Team'] = df['Team'] .astype('category')

week = 14
df1 = df[df['Week'].between(week-3, week-1)]
df1 = df1[~df1.Opp_Name.str.contains("Bye")]
df1.reset_index(inplace=True)
# print(df1.head())
dfavg = df1.groupby(['Team']).agg([np.average]).copy()
dfavg.columns = ['index', 'Unnamed: 0','Week',	'Result',	'Home',	'Tm',	'Opp',	'OFF1stD',	'OFFTotYd',	'OFFPassY',	'OFFRushY',	'TOOFF',	'DEF1stD',	'DEFTotYd',
                 'DEFPassY',	'DEFRushY',	'TODEF',	'OffenseEP',	'DefenseEP',	'Sp_TmsEP']
dfavg = dfavg.reset_index()
dfavg.drop('index', axis=1, inplace=True)
preddf = df[df['Week'].between(week, week)].copy()
preddf.reset_index(inplace=True)
preddf.columns = df1.columns
preddf = preddf.drop(['index','Unnamed: 0'], axis=1).copy()
preddf.sort_values('Team', inplace=True)
preddf.reset_index(inplace=True)
preddf = preddf.drop('index', axis=1)
replace_cols = ['Tm',	'Opp',	'OFF1stD',	'OFFTotYd',	'OFFPassY',	'OFFRushY',	'TOOFF',	'DEF1stD',	'DEFTotYd',	'DEFPassY',	'DEFRushY',	'TODEF',	'OffenseEP',	'DefenseEP',	'Sp_TmsEP']
preddf[replace_cols] = dfavg[replace_cols].copy()

to_pred = preddf.copy()

X_reg = to_pred.drop(['Week','Result','Tm'], axis=1)
y_reg = to_pred['Tm'].copy()

X_class = to_pred.drop(['Week','Result','Tm'], axis=1)
y_reg = to_pred['Result'].copy()



regmodel = lgb.Booster(model_file='https://raw.githubusercontent.com/barbabb/fantastic-adventure/master/regmodelv2_train18.txt')
classmodel = lgb.Booster(model_file='https://raw.githubusercontent.com/barbabb/fantastic-adventure/master/classmodel_train18.txt')

reg_preds = regmodel.predict(X_reg)
class_preds = classmodel.predict(X_class)

totals = [to_pred['Team'], to_pred['Opp_Name'], pd.Series(reg_preds), pd.Series(np.round(class_preds))]
df_del = pd.concat(totals, axis=1)
df_del.columns = ['Team','Opp_Name','Pts','W/L']


pts_dict = dict(zip(df_del.Team,df_del.Pts))
w_l_dict = dict(zip(df_del.Team,df_del['W/L']))
df_del.Team = df_del.Team.map(team_dict)
opp_pts_dict = dict(zip(df_del.Team,df_del.Pts))
opp_w_l_dict = dict(zip(df_del.Team,df_del['W/L']))


new_dict = dict(zip(to_pred.Team,to_pred.Opp_Name))


games = pd.DataFrame.from_dict(new_dict, orient='index',
                       columns=['Opps'])
games.index.rename('Team', inplace=True)
games.reset_index(inplace=True)
to_cat = [games, preddf['Home']]
games_df = pd.concat(to_cat, axis=1)


prepared_df = pd.DataFrame()
prepared_df['Home_Team'] = games_df.Team.where(games_df.Home == 1)
prepared_df['Away_Team'] = games_df.Opps.where(games_df.Home == 1)
prepared_df.dropna(how='all', inplace=True)

prepared_df['Home_Score'] = prepared_df.Home_Team.map(pts_dict)
prepared_df['Away_Score'] = prepared_df.Away_Team.map(opp_pts_dict)
prepared_df['Home_W/L'] = prepared_df.Home_Team.map(w_l_dict)
prepared_df['Away_W/L'] = prepared_df.Away_Team.map(opp_w_l_dict)

prepared_df['Margin'] = prepared_df.Home_Score - prepared_df.Away_Score
prepared_df['Confidence'] = prepared_df['Home_W/L'] - prepared_df['Away_W/L']
prepared_df.Margin = prepared_df.Margin.round(1)

finished_df = prepared_df.drop(['Home_Score', 'Away_Score', 'Home_W/L', 'Away_W/L'], axis=1)
finished_df.Home_Team = finished_df.Home_Team.map(team_dict)

st.dataframe(finished_df)
