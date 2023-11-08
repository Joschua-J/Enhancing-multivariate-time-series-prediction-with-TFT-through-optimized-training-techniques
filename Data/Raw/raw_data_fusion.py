#Imports
import pandas as pd

#Read in data from SMARD.de
_50Hertz = pd.read_csv('50Hertz.csv', sep=';')
_Amprion = pd.read_csv('Amprion.csv', sep=';')
_TenneT = pd.read_csv('TenneT.csv', sep=';')
_TransnetBW = pd.read_csv('TransnetBW.csv', sep=';')
_Gesamt = pd.read_csv('Gesamt.csv', sep=';')

#Delete unnecessary columns
for i in [_50Hertz, _Amprion, _TenneT, _TransnetBW, _Gesamt]:
    i.drop(columns=['Ende', 'Residuallast [MWh] Originalauflösungen'], inplace=True)
    i['Gesamt (Netzlast) [MWh] Originalauflösungen'] = i['Gesamt (Netzlast) [MWh] Originalauflösungen'].str.replace('.', '', regex=True)
    i['Gesamt (Netzlast) [MWh] Originalauflösungen'] = i['Gesamt (Netzlast) [MWh] Originalauflösungen'].str.replace(',', '.', regex=True)
    i['Gesamt (Netzlast) [MWh] Originalauflösungen'] = i['Gesamt (Netzlast) [MWh] Originalauflösungen'].str.replace('-', '0', regex=True).astype(float)

#Create new DataFrame
data = pd.DataFrame()
data['Date'] = _50Hertz['Datum']
data['Time'] = _50Hertz['Anfang']
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d.%m.%Y %H:%M')

#Add data to the new DataFrame
for i in [(_50Hertz, '50Hertz'), (_Amprion, 'Amprion'), (_TenneT, 'TenneT'), (_TransnetBW, 'TransnetBW'), (_Gesamt, 'Total')]:
    data[i[1]] = i[0]['Gesamt (Netzlast) [MWh] Originalauflösungen']
data['Delta'] = abs(data['Total'] - data['50Hertz'] - data['Amprion'] - data['TenneT'] - data['TransnetBW'])

#Save new DataFrame as csv
data.to_csv('raw.csv', index=False)