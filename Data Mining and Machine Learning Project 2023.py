import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

df=pd.read_csv('data.csv')

datetime_is_numeric=True
df['Date']=df['Date'].astype('datetime64[ns]')

print(df)
print(df.info())
print(df.describe(include='all'))

NaNs_df=df.isnull().sum().sort_values(ascending=False)
print(NaNs_df)

df['Daily Cases'] = abs(df.groupby('Entity')['Cases'].diff())
df['Daily Deaths'] = abs(df.groupby('Entity')['Deaths'].diff())

Tests_mean_values=df.groupby('Entity')['Daily tests'].mean()
Cases_mean_values=df.groupby('Entity')['Cases'].mean()
Deaths_mean_values=df.groupby('Entity')['Deaths'].mean()
Daily_Cases_mean_values=df.groupby('Entity')['Daily Cases'].mean()
Daily_Deaths_mean_values=df.groupby('Entity')['Daily Deaths'].mean()

df['Daily Cases'] = df['Daily Cases'].fillna(df['Entity'].map(Daily_Cases_mean_values))
df['Daily Deaths'] = df['Daily Deaths'].fillna(df['Entity'].map(Daily_Deaths_mean_values))
df['Daily tests'] = df['Daily tests'].fillna(df['Entity'].map(Tests_mean_values))
df['Cases'] = df['Cases'].fillna(df['Entity'].map(Cases_mean_values))
df['Deaths'] = df['Deaths'].fillna(df['Entity'].map(Deaths_mean_values))


countries_list=list(df.Entity.unique())
population_list=list(df.Population.unique())
continent_list=list(df.Continent.unique())

df_daily_tests=df.groupby(['Date'])['Daily tests'].sum().reset_index()
dates=df.sort_values(by='Date').reset_index(drop=True)
date_array=dates.Date.unique()
print(len(date_array))

NaNs_df=df.isnull().sum().sort_values(ascending=False)
print(NaNs_df)

for country in countries_list:
    print(country+":\n")
    print(df[df['Entity']==country].describe())



df['total_deaths']=df.groupby('Entity')['Deaths'].transform('max')
df['total_cases']=df.groupby('Entity')['Cases'].transform('max')
df['total_tests']=df.groupby('Entity')['Daily tests'].transform('sum')

Africa = df[df.iloc[:, 1] == 'Africa']
Asia = df[df.iloc[:, 1] == 'Asia']
Europe = df[df.iloc[:, 1] == 'Europe']
North_America = df[df.iloc[:, 1] == 'North America']
Oceania = df[df.iloc[:, 1] == 'Oceania']
South_America = df[df.iloc[:, 1] == 'South America']



continent_list = df['Continent'].unique()

for continent in continent_list:
    data_of_each_country_per_continent = df[df['Continent'] == continent]
    
    plt.figure(figsize=(10, 6))
    for index, row in data_of_each_country_per_continent.iterrows():
        plt.bar(row['Entity'], row['total_deaths'])
        
    plt.xticks(rotation=90)
    plt.xlabel('Countries')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths per Country in {continent}')
    plt.tight_layout()
    plt.show()


for continent in continent_list:
    data_of_each_country_per_continent = df[df['Continent'] == continent]
    
    plt.figure(figsize=(10, 6))
    for index, row in data_of_each_country_per_continent.iterrows():
        plt.bar(row['Entity'], row['total_cases'])
        
    plt.xticks(rotation=90)
    plt.xlabel('Countries')
    plt.ylabel('Total Cases')
    plt.title(f'Total Cases per Country in {continent}')
    plt.tight_layout()
    plt.show()

for continent in continent_list:
    data_of_each_country_per_continent = df[df['Continent'] == continent]
    
    plt.figure(figsize=(10, 6))
    for index, row in data_of_each_country_per_continent.iterrows():
        plt.bar(row['Entity'], row['total_tests'])
        
    plt.xticks(rotation=90)
    plt.xlabel('Countries')
    plt.ylabel('Total Tests')
    plt.title(f'Total Tests per Country in {continent}')
    plt.tight_layout()
    plt.show()



corr_df_pearson=df.corr(method='pearson')

corr_df_kendall=df.corr(method='kendall')

corr_df_spearman=df.corr(method='spearman')

plt.figure(figsize=(10,8))
sn.heatmap(corr_df_pearson, annot=True)
plt.show()

plt.figure(figsize=(10,8))
sn.heatmap(corr_df_kendall, annot=True)

plt.show()
plt.figure(figsize=(10,8))

sn.heatmap(corr_df_spearman, annot=True)
plt.show()



def calculate_death_percentage(group):
    group['Death %'] = group['total_deaths'] / group['total_cases'] * 100
    return group

death_precentage_df = df.groupby('Entity').apply(calculate_death_percentage)


def calculate_positivity_percentage(group):
    group['Positivity %'] = group['total_cases'] / group['total_tests'] * 100
    return group

positivity_precentage_df = df.groupby('Entity').apply(calculate_positivity_percentage)


def calculate_cases_to_population_percentage(group):
    group['Cases to Population %'] = group['total_cases'] / group['Population'] * 100
    return group

cases_to_population_precentage_df = df.groupby('Entity').apply(calculate_cases_to_population_percentage)




for continent in continent_list:
    data_of_each_country_per_continent_death_prc = death_precentage_df[death_precentage_df['Continent'] == continent]
    
    plt.figure(figsize=(10, 6))
    for index, row in data_of_each_country_per_continent_death_prc.iterrows():
        plt.bar(row['Entity'], row['Death %'])
        
    plt.xticks(rotation=90)
    plt.xlabel('Countries')
    plt.ylabel('Death %')
    plt.title(f'Death precentage per Country in {continent}')
    plt.tight_layout()
    plt.show()



for continent in continent_list:
    data_of_each_country_per_continent_positivity_prc = positivity_precentage_df[positivity_precentage_df['Continent'] == continent]
    
    plt.figure(figsize=(10, 6))
    for index, row in data_of_each_country_per_continent_positivity_prc.iterrows():
        plt.bar(row['Entity'], row['Positivity %'])
        
    plt.xticks(rotation=90)
    plt.xlabel('Countries')
    plt.ylabel('Positivity %')
    plt.title(f'Positivity precentage per Country in {continent}')
    plt.tight_layout()
    plt.show()



for continent in continent_list:
    data_of_each_country_per_continent_positivity_prc = cases_to_population_precentage_df[cases_to_population_precentage_df['Continent'] == continent]
    
    plt.figure(figsize=(10, 6))
    for index, row in data_of_each_country_per_continent_positivity_prc.iterrows():
        plt.bar(row['Entity'], row['Cases to Population %'])
        
    plt.xticks(rotation=90)
    plt.xlabel('Countries')
    plt.ylabel('Cases to Population  %')
    plt.title(f'Cases to Population precentage per Country in {continent}')
    plt.tight_layout()
    plt.show()


evaluation_df=pd.DataFrame()


evaluation_df['Entity'] = death_precentage_df['Entity']

evaluation_df['Death %'] = death_precentage_df['Death %']

evaluation_df['Positivity %'] = positivity_precentage_df['Positivity %']

evaluation_df['Cases to Population %'] = cases_to_population_precentage_df['Cases to Population %']



grouped_evaluation_df = evaluation_df.groupby('Entity').agg({
    'Death %': 'first',
    'Positivity %': 'first', 
    'Cases to Population %': 'first' 
}).reset_index()

final_df=pd.DataFrame()

final_df=grouped_evaluation_df

grouped_evaluation_df.drop(columns='Entity', inplace=True)

final_evaluation_df=evaluation_df.groupby('Entity').agg({
    'Death %': 'first',
    'Positivity %': 'first', 
    'Cases to Population %': 'first' 
}).reset_index()

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(grouped_evaluation_df)
normalized_df = pd.DataFrame(normalized_data, columns=grouped_evaluation_df.columns)


neighbors = NearestNeighbors(n_neighbors=3)
neighbors_fit = neighbors.fit(normalized_df)
distances, indices = neighbors_fit.kneighbors(normalized_df)
distances = np.sort(distances, axis=0)
distances = distances[:,2]
plt.plot(distances)



dbscan = DBSCAN(eps=0.1088, min_samples=3)
cluster_labels = dbscan.fit_predict(normalized_df)

labels=np.unique(cluster_labels)

cluster_labels_df= pd.DataFrame({'Cluster': cluster_labels})
result_df_features = pd.concat([final_df,cluster_labels_df], axis=1)


entities=df['Entity'].unique()
entities_df=pd.DataFrame({'Entities': entities})

entities_df=pd.DataFrame({'Entity': entities})

result_df = pd.concat([entities_df,cluster_labels_df], axis=1)

value_counts =result_df['Cluster'].value_counts()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


scatter = ax.scatter(result_df_features['Death %'], result_df_features['Positivity %'], result_df_features['Cases to Population %'], c=result_df_features['Cluster'], cmap='viridis')


ax.set_xlabel('Death %')
ax.set_ylabel('Positivity %')
ax.set_zlabel('Cases to Population %')
ax.set_title('DBSCAN')


cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')


cluster_counts=result_df['Cluster'].value_counts()
print(cluster_counts) 






Greece_df = df[df['Entity'] == 'Greece']

prediction_df = Greece_df[Greece_df['Date'] < '2021-01-01']
prediction_df['positivity_ratio'] = prediction_df['Daily Cases'] / prediction_df['Daily tests']

columns = ['Date', 'positivity_ratio']
Greece_prediction_df = prediction_df[columns]


Greece_prediction_df['prediction'] = Greece_prediction_df['positivity_ratio'].shift(3)
x = Greece_prediction_df['positivity_ratio']
y = Greece_prediction_df['prediction']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

svm_regressor = SVR(C=200, epsilon=0.001)


svm_regressor.fit(X_train, y_train)


prediction_days = min(49, len(X_test))
  


start_date = datetime.strptime('2021-01-01', '%Y-%m-%d').date()

 
predicted_ratios = []

actual_ratios = []

 
for i in range(prediction_days):
      target_date = start_date + timedelta(days=3+i)
      
      prediction_input = X_test.iloc[i].to_frame().T

      
      predicted_ratio = svm_regressor.predict(prediction_input)[0]

      
      predicted_ratios.append(predicted_ratio)
      actual_ratios.append(y_test.iloc[i])

      
      print(f"Predicted positivity ratio for {target_date}: {predicted_ratio}")

 
mse = mean_squared_error(actual_ratios, predicted_ratios)

  
accuracy = svm_regressor.score(X_test, y_test)


print(f"Mean Squared Error: {mse}")
print(f"Accuracy: {accuracy}")











scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

    
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=False)

    
x_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
x_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

  
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

   
model.fit(x_train, y_train, epochs=10, batch_size=32)

    
prediction_days = min(49, len(x_test))

   
predicted_ratios = []

    
actual_ratios = []
 
start_date = datetime.strptime('2021-01-01', '%Y-%m-%d').date()
   
for i in range(prediction_days):
    target_date = start_date + timedelta(days=3+i)
      
    prediction_input = np.reshape(X_test[i], (1, 1, X_test.shape[2]))

     
    predicted_ratio = model.predict(prediction_input)[0][0]

      
    predicted_ratios.append(predicted_ratio)
    actual_ratios.append(y_test.iloc[i])

        
    target_date_index = i 
        
    print(f"Predicted positivity ratio for {target_date}: {predicted_ratio}")

   
mse = mean_squared_error(actual_ratios, predicted_ratios)

  
print(f"Mean Squared Error: {mse}")
   


