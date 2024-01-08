import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data_path = 'GDP per capita.csv'


data = pd.read_csv(data_path, skiprows=4)
print(data.head())
#print(data.info())

# Selecting recent 10 years of data
recent_years_data = data.iloc[:, -12:-2]  

# Replacing NaN with 0 for clustering purposes
recent_years_data = recent_years_data.fillna(0)
#print(recent_years_data.head())
#print(recent_years_data.info())

# Normalizing the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(recent_years_data)

# Applying KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(normalized_data)

data['Cluster'] = clusters
#print(data.head())

# Visualizing the clusters for a couple of recent years
plt.figure(figsize=(12, 6))
plt.scatter(data['2020'], data['2021'], c=data['Cluster'], cmap='viridis')
plt.title('Cluster of Countries by GDP per Capita (2020 vs 2021)')
plt.xlabel('GDP per Capita 2020')
plt.ylabel('GDP per Capita 2021')
plt.colorbar(label='Cluster')
plt.show()


# Calling data again
data = pd.read_csv(data_path, skiprows=4)

median_gdp_index = data['2021'].median()
representative_country = data.iloc[(data['2021'] - median_gdp_index).
                                   abs().argsort()[:1]]

# Assuming data from 1960 to 2021
years = np.arange(1960, 2022)  

# Taking GDP values and handling NaNs
gdp_values = representative_country.iloc[0, 4:-2].fillna(0).values  

# Defining a simple polynomial model function
def poly_model(x, a, b, c):
    return a * x**2 + b * x + c

# Fitting the model to the data
popt, pcov = curve_fit(poly_model, years, gdp_values, maxfev=10000)

# Predicting future values for the next 20 years
future_years = np.arange(2022, 2042)
predictions = poly_model(future_years, *popt)

# Calculating confidence intervals
perr = np.sqrt(np.diag(pcov))
ci = 1.96 * perr

# Visualizing the fitting and prediction
plt.figure(figsize=(12, 6))
plt.plot(years, gdp_values, 'b-', label="Historical GDP")
plt.plot(future_years, predictions, 'r--', label="Predicted GDP")
plt.fill_between(future_years, (predictions-ci[0]), (predictions+ci[0]), 
                 color='gray', alpha=0.2)
plt.title(f'GDP per Capita Prediction for {representative_country.iloc[0, 0]}')
plt.xlabel('Year')
plt.ylabel('GDP per Capita')
plt.legend()
plt.show()

representative_country.iloc[0, 0], popt, ci



# Focusing on recent 10 years and handling NaNs
recent_years_data = data.iloc[:, -12:-2].fillna(0)  

# Normalizing recent years data again
scaler = StandardScaler()
normalized_data = scaler.fit_transform(recent_years_data)

# Applying KMeans Clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(normalized_data)
data['Cluster'] = clusters


representative_countries = data.groupby('Cluster').\
apply(lambda x: x.iloc[(x['2021'] - x['2021'].median()).abs().argsort()[:1]])

representative_countries = representative_countries.reset_index(drop=True)

years = np.arange(2011, 2021)  # Adjusting to the last 10 years of data

plt.figure(figsize=(14, 7))
for index, row in representative_countries.iterrows():
    country_gdp = row.iloc[-12:-2].values
    plt.plot(years, country_gdp, label=f"Cluster {row['Cluster']} - \
             {row['Country Name']}")

plt.title('GDP per Capita Trends of Representative Countries from Each \
          Cluster')
plt.xlabel('Year')
plt.ylabel('GDP per Capita')
plt.legend()
plt.grid(True)
plt.show()