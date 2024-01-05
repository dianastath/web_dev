
import io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
from datetime import datetime
import geopandas
import imageio
import time 
import sys
from scipy.ndimage import gaussian_filter


# Getting world map data from geo pandas
worldmap = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

#FUNCTION  USED TO UPLOAD FILES
def load_data(data_path, data_name):
    csv_path = os.path.join(data_path, data_name)
    return pd.read_csv(csv_path)  

#FUNCTION USED TO GROUP BY SAME COUNTRIES
def processing_file(data_name, name):
    # KATARGHSH KOLONAS Province/State
    data_name = data_name.drop(['Province/State'], axis=1)
    # Prostetoyme mia nea kolona me 1
    data_name.insert(0, 'Assos', 1)
    # Atrizoyme tis kolones se mia gia kathe xora
    data_name = data_name.groupby(['Country/Region'], as_index=False).sum()
    data_name['Lat'] = data_name['Lat']/data_name['Assos']
    # Mesos oros Lat
    data_name['Long'] = data_name['Long']/data_name['Assos']
    # Mesos oros Long
    data_name = data_name.drop('Assos', axis=1)
    #data_name = data_name.set_index(['Country/Region'])
    print('------------------------------------------------------------')
    print('THE FILE', name, 'HAS ',data_name.shape[0], 'LINES AND ', data_name.shape[1], 'COLUMNS')
    print('THE FILE', name, ' HAS ', data_name.size, 'DATA')
    print('****HEAD OF THE FILE',name,'****')
    print(data_name.head())
    print('------------------------------------------------------------')
    csv_path = os.path.join(ORIGINAL_DATA_PATH, name)
    data_name.to_csv(csv_path)
    return data_name

#################################################################################
#FUNCTION TO PLOT THE GLOBAL MAP FOR QUESTION 4
def plot1(data_name, name, date, MaxValue, thres,savetodisk=False):
    
    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(16, 10))
    worldmap.plot(color="lightgrey", ax=ax)
    
    x = []
    y = []
    z = []
    z2= []
    all_x = list(data_name['Long'])
    all_y = list(data_name['Lat'])
    all_z = list(data_name[date]) #list with number of cases for date
    #all_z2 = list(MinMaxScaler(feature_range=(0, 1000)).fit_transform(np.array(data_name[date]).reshape(-1,1)))
    for i in range(0, len(all_z)):
        if all_z[i] > thres:
            x.append(all_x[i])
            y.append(all_y[i])
            z.append(all_z[i])  
            z2.append(all_z[i]/MaxValue *1000)
            
    plt.scatter(x, y, s=z2, c=z, alpha=0.6, cmap='autumn')
    plt.colorbar(label=f'Number of {name}')
    plt.clim(0, MaxValue)
    # Creating axis limits and title
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.title(f"Covid 19 {name} on {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    new_date = date.replace("/", "-")
    if savetodisk:
        plt.savefig(f'./img/img_{new_date}_{name}.png', transparent=False, facecolor='white')
    stream = io.BytesIO()  #make it faster
    plt.savefig(stream, transparent=False, facecolor='white')
    stream.seek(0)  #absolute file positioning (start_position)
    image = imageio.v2.imread(stream)
    plt.show()
    plt.close()
    return image

#FUNCTION TO MAKE GIF FOR GUESTION 5
def CreateGifImage(df,name,thresold):
    df =df.drop('Country/Region',axis=1)
    #df = df.set_index(['Country/Region'])
    pictures=[]
    ListWithColumnsDates = list(df.columns[4:])    
    MaxValue = df.values.max()
    for i in ListWithColumnsDates:
        print('Creating Image '+name+' for '+i)
        image=plot1(df, name, i, MaxValue, thresold,False)
        pictures.append(image)
    imageio.mimsave('./Data_'+name+'.gif', pictures,  fps=30, loop=1)

#FUNCTION TO MAKE THE CHART FOR QUESTION 6
def ChartForTheTop10NumberOfCases(data_name,name,maxdate):
    data_name = data_name.set_index(['Country/Region'])
    max_10=data_name[maxdate].nlargest(n=10)
    top_10_name_countries=list(max_10.index)
    print('TOP 10 COUNTRIES NAMES' ,top_10_name_countries)
    chart = (max_10.sort_values(ascending=True))
    plt.title(f'Covid 19 {name} by Countrh/Region-Top 10 As of 29-May-2021',fontweight=10,pad='2.0')
    chart.plot.barh() 
    plt.savefig(f'./img/img_{name}.png', transparent=False, facecolor='white')
    plt.show()
    plt.close()

#FUNCTON TO MAKE A DATAFRAME WITH THE TOP 10 COUNTRIES 
def GetDataFrameWithTop10Countries(data_name,maxdate,name):
    df = data_name.set_index(['Country/Region'])
    max_10=df[maxdate].nlargest(n=10)
    ListWithTop10Names = list(max_10.index)
    #data_name.reset_index()
    Top10DataFrame = data_name.loc[data_name['Country/Region'].isin(ListWithTop10Names)]
    Top10DataFrame = Top10DataFrame.drop(['Lat'], axis=1)
    Top10DataFrame = Top10DataFrame.drop(['Long'], axis=1)
    Top10DataFrame = Top10DataFrame.set_index(['Country/Region'])
    
    csv_path = os.path.join(ORIGINAL_DATA_PATH, name)
    Top10DataFrame.to_csv(csv_path)
    return Top10DataFrame

#FUNCTION TO FIND THE NEW CASES-DEATHS-RECOVERS
def AddDifferenceToTop10(data_name,name):
    ListWithNames = list(data_name.index)
    for j in ListWithNames:

        L=data_name.loc[j].values
        D=[]
        for i in range(1,len(L)):
            D.append(L[i]-L[i-1])
        D.insert(0, 0)
           
        data_name.loc[j+'_d']=D
    csv_path = os.path.join(ORIGINAL_DATA_PATH, name)
    data_name.to_csv(csv_path)
    return data_name
        
def NewPlot(data_name,ListWithNames,name):
    
    for j in ListWithNames:
        row = data_name.loc[j+'_d']
        #row2 = data_name.loc[j] #+'_d']
        row.plot()
        #row2.plot(secondary_y=True)
        plt.title(f"{j}")
        plt.xlabel('Date')
        plt.ylabel('Cases Per Day')
        plt.savefig(f'./img3/img_{j}_{name}.png',transparent=False, facecolor='white')
        plt.show()
        plt.close()

def AddGausianFilteredDataToTop10DataFrame(data_name,name,ListWithNames):
    for j in ListWithNames:
         row = data_name.loc[j+'_d']
         #L=row.values
         L=list(gaussian_filter(row.values,sigma=15))
         data_name.loc[j+'_G']=L  #ftiaxno mia kainoyria grammh kai tiw bazo mesa ta L
         row2=data_name.loc[j+'_G']
         row2.plot()
         plt.title(f"{j}")
         plt.xlabel('Date')
         plt.ylabel('Cases Per Day')
         #plt.savefig(f'./img3/img_{j}_{name}.png',transparent=False, facecolor='white')
         plt.show()
         plt.close()
    csv_path = os.path.join(ORIGINAL_DATA_PATH, 'Top10Gauss'+name+'.csv')
    data_name.to_csv(csv_path)
    return data_name
                 

    

    
###############################################################################
start_time=time.time()

ORIGINAL_DATA_PATH = os.path.join('Covid_19')
FIGURES_PATH = os.path.join('Covid_19','Figures')

#CALL FUNCTION load_data

Data_confirmed = load_data(ORIGINAL_DATA_PATH, 'time_series_covid_19_confirmed.csv')
Data_deaths = load_data(ORIGINAL_DATA_PATH, 'time_series_covid_19_deaths.csv')
Data_recovered = load_data(ORIGINAL_DATA_PATH, 'time_series_covid_19_recovered.csv')


print('------------------------------------------------------------')

print('THE ORIGINAL FILE FOR CONFIRMED CASES HAS',Data_confirmed.shape[0], 'LINES AND ', Data_confirmed.shape[1], 'COLUMNS')
print('THE ORIGINAL FILE FOR CONFIRMED CASES HAS ', Data_confirmed.size, 'DATA')


####################### QUESTION 2 AND 3 ######################################

list_with_countries = Data_confirmed['Country/Region'].unique()

LISTA = pd.DataFrame(list_with_countries) 
csv_path = os.path.join(ORIGINAL_DATA_PATH, "List_with_countries.csv")
LISTA.to_csv(csv_path)


print('LIST WITH COUNTRIES:  ', list_with_countries)
print('WE HAVE', len(list_with_countries),'DIFFERENT COUNTRIES')


ListWithColumnsDates = list(Data_confirmed.columns[4::])


LISTA_2 = pd.DataFrame(ListWithColumnsDates) 
csv_path = os.path.join(ORIGINAL_DATA_PATH, "List_with_dates.csv")
LISTA_2.to_csv(csv_path)

print('LIST WITH DATES: ', ListWithColumnsDates)


StartDate = ListWithColumnsDates[0]
EndDate   = ListWithColumnsDates[-1]
a = datetime.strptime(StartDate, '%m/%d/%y')  # .date
b = datetime.strptime(EndDate, '%m/%d/%y')  # .date

print('START DAY=', a)
print('FINAL DAY=', b)

delta = b - a

print(f'Difference is {delta.days} days')
print('THE PERIOD IS ', delta,'DAYS')
print('------------------------------------------------------------')

# CALL FUNCTION processing_file

Data_confirmed = processing_file(Data_confirmed, 'New_time_series_covid_19_confirmed.csv')
Data_deaths    = processing_file(Data_deaths,    'New_time_series_covid_19_deaths.csv')
Data_recovered = processing_file(Data_recovered, 'New_time_series_covid_19_recovered.csv')

#CALL FUNCTION plot1

thresold = 100
MaxValue = Data_confirmed[StartDate].values.max()
plot1(Data_confirmed, 'confirmed_cases', StartDate, MaxValue, thresold,True)
MaxValue = Data_confirmed[EndDate].values.max()
plot1(Data_confirmed, 'confirmed_cases', EndDate, MaxValue, thresold,True)


thresold = 50
MaxValue = Data_deaths[StartDate].values.max()
plot1(Data_deaths, 'deaths', StartDate, MaxValue, thresold,True)
MaxValue = Data_deaths[EndDate].values.max()
plot1(Data_deaths, 'deaths', EndDate, MaxValue, thresold,True)


thresold = 0
MaxValue = Data_recovered[StartDate].values.max()
plot1(Data_recovered, 'recovered_cases', StartDate, MaxValue, thresold,True)
MaxValue = Data_recovered[EndDate].values.max()
plot1(Data_recovered, 'recovered_cases', EndDate, MaxValue, thresold,True)

#CALL FUNCTION ChartForTheTop10NumberOfCases

ChartForTheTop10NumberOfCases(Data_confirmed,'confirmed_cases',EndDate)
ChartForTheTop10NumberOfCases(Data_deaths,'deaths',EndDate)
ChartForTheTop10NumberOfCases(Data_recovered,'recovered_cases',EndDate)

#CALL FUNCTION Top10ConfirmedDataFrame

Top10ConfirmedDataFrame = GetDataFrameWithTop10Countries(Data_confirmed,EndDate,'Top10ConfirmedDataFrame.csv')
Top10DeathsDataFrame    = GetDataFrameWithTop10Countries(Data_deaths,EndDate,'Top10DeathsDataFrame.csv')
Top10RecoveredDataFrame = GetDataFrameWithTop10Countries(Data_recovered,EndDate,'Top10RecoveredDataFrame.csv')

ListWithTopTenConfirmedCountries=list(Top10ConfirmedDataFrame.index)
ListWithTopTenDeathsCountries=list(Top10DeathsDataFrame.index)
ListWithTopTenRecoveredCountries=list(Top10RecoveredDataFrame.index)


Top10ConfirmedDataFrame=AddDifferenceToTop10(Top10ConfirmedDataFrame,'Top10ConfirmedDataFrameDifference.csv')
Top10DeathsDataFrame=AddDifferenceToTop10(Top10DeathsDataFrame,'Top10DeathsDataFrameDifference.csv')
Top10RecoveredDataFrame=AddDifferenceToTop10(Top10RecoveredDataFrame,'Top10RecoveredDataFrameDifference.csv')

print(Top10ConfirmedDataFrame)
print(Top10DeathsDataFrame)
print(Top10RecoveredDataFrame)

#CALL FUNCTION NewPlot

NewPlot(Top10ConfirmedDataFrame,ListWithTopTenConfirmedCountries,'confirmed_cases')
NewPlot(Top10DeathsDataFrame,ListWithTopTenDeathsCountries,'deaths')
NewPlot(Top10RecoveredDataFrame,ListWithTopTenRecoveredCountries,'recovered_cases')

#CALL FUNCTION AddGausianFilteredDataToTop10DataFrame

AddGausianFilteredDataToTop10DataFrame(Top10ConfirmedDataFrame,'confirmed_cases',ListWithTopTenConfirmedCountries)
AddGausianFilteredDataToTop10DataFrame(Top10DeathsDataFrame,'deaths',ListWithTopTenDeathsCountries)
AddGausianFilteredDataToTop10DataFrame(Top10RecoveredDataFrame,'recovered_cases',ListWithTopTenRecoveredCountries)

end_time=time.time()
print('time=',end_time-start_time,'sec')

sys.exit(0)



CreateGifImage(Data_confirmed,'Confirmed_cases',0)
CreateGifImage(Data_deaths,'Deaths',0)
CreateGifImage(Data_recovered,'Recovered',0)


end_time=time.time()
print('time=',end_time-start_time,'sec')


