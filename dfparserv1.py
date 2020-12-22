from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from random import randint

# !pip install -U selenium\\
# !apt-get update 
# !apt install chromium-chromedr/iver
from selenium import webdriver
from time import sleep

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)

teams = ['htx', 'kan', 'nyj', 'buf', 'sea', 'atl', 'phi', 'was', 'cle', 'rav', 'mia', 'nwe', 'gnb', 'min', 'clt', 'jax', 'chi', 
         'det', 'rai', 'car', 'sdg', 'cin', 'crd', 'sfo', 'tam', 'nor', 'dal', 'ram', 'pit', 'nyg', 'oti', 'den']

no_table = []
data = pd.DataFrame()
url = 'https://www.pro-football-reference.com'
year = 2020

for team in teams:
  driver.get(url + '/teams/' + str(team) + '/2020.htm') 
  sleep(randint(2,10))
  table = pd.read_html(driver.page_source)
  week = 16
  cols = ['Week', 'Day', 'Date', 'Time', 'BoxS', 'Result', 'OT',	'Rec', 'Home', 'Opp_Name',	'Tm',	'Opp',	'OFF1stD',	'OFFTotYd',	'OFFPassY',	'OFFRushY',	'TOOFF',	'DEF1stD',	'DEFTotYd',	'DEFPassY',	'DEFRushY',	'TODEF',	'OffenseEP',	'DefenseEP',	'Sp_TmsEP']
  dft = table[2]
  dft = dft[0:week]
  dft.columns = cols
  dft = dft[~dft.Opp_Name.str.contains("Bye")]
  dft = dft.drop(['Day', 'Date', 'Time', 'BoxS', 'OT', 'Rec'], axis=1)
  dft['Result'] = [0 if r=='L' else 1 for r in dft['Result']]
  dft['Home'] = [0 if r=='@' else 1 for r in dft['Home']]
  dft['TOOFF'] = dft['TOOFF'].fillna(0)
  dft['TODEF'] = dft['TODEF'].fillna(0)
  dft['Team'] = str(team)
  dft = dft.set_index('Team')
  dft.reset_index(inplace=True)
  no_table.append(dft)
    
df = pd.concat(no_table)
df['Opp_Name'] = df['Opp_Name'].astype('category')
df['Team'] = df['Team'] .astype('category')
df.to_csv('D:/Documents/ML DOCS/ML Project Files/preds/2020df_week16.csv')