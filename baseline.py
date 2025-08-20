# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# Set plot style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Step 2: Load the dataset
# Load the dataset
file_path = 'global_air_pollution_data.csv'
air_df = pd.read_csv(file_path)

# Display basic information
print(f"Dataset shape: {air_df.shape}")
print("\nFirst 5 rows:")
air_df.head()

# Check column names and data types
print("Column names:")
print(air_df.columns.tolist())
print("\nData types:")
air_df.dtypes

# Check for missing values
print("Missing values per column:")
air_df.isnull().sum()

# Summary statistics
air_df.describe().round(2)

# Check unique categories for AQI
print("AQI Categories:")
print(air_df['aqi_category'].value_counts())

# Create dominant_pollutant column based on highest pollutant AQI values
pollutants = {
    'CO': 'co_aqi_value\t',  # Note: there appears to be a tab character in this column name
    'Ozone': 'ozone_aqi_value',
    'NO2': 'no2_aqi_value',
    'PM2.5': 'pm2.5_aqi_value'
}

# Function to find the dominant pollutant for each row
def get_dominant_pollutant(row):
    max_val = -1
    dominant = 'Unknown'
    for pollutant, column in pollutants.items():
        if pd.notna(row[column]) and row[column] > max_val:
            max_val = row[column]
            dominant = pollutant
    return dominant

# Apply the function to create the dominant_pollutant column
air_df['dominant_pollutant'] = air_df.apply(get_dominant_pollutant, axis=1)
print("\nDominant Pollutant Distribution:")
print(air_df['dominant_pollutant'].value_counts())


# Step 3: Data Visualization

# Define AQI category mapping for proper sorting
aqi_category_map = {
    'Good': 1,
    'Moderate': 2,
    'Unhealthy for Sensitive Groups': 3,
    'Unhealthy': 4,
    'Very Unhealthy': 5,
    'Hazardous': 6
}

# Distribution of AQI values
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(air_df['aqi_value'], bins=30, kde=True)
plt.title('Distribution of AQI Values')
plt.xlabel('AQI Value')

plt.subplot(1, 2, 2)
sns.countplot(y='aqi_category', data=air_df, order=sorted(air_df['aqi_category'].unique(),
                                                        key=lambda x: aqi_category_map.get(x, 0)))
plt.title('Count of Cities by AQI Category')
plt.xlabel('Count')
plt.ylabel('AQI Category')

plt.tight_layout()
plt.show()

# Top 10 countries with worst air quality (highest average AQI)
country_aqi = air_df.groupby('country_name')['aqi_value'].agg(['mean', 'count'])
country_aqi = country_aqi[country_aqi['count'] >= 5]  # Consider only countries with at least 5 cities
top_10_worst = country_aqi.sort_values('mean', ascending=False).head(10)

plt.figure(figsize=(14, 6))
sns.barplot(x=top_10_worst.index, y=top_10_worst['mean'], palette='YlOrRd')
plt.title('Top 10 Countries with Worst Air Quality')
plt.xlabel('Country')
plt.ylabel('Average AQI Value')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=50, color='green', linestyle='--', label='Good AQI Threshold')
plt.axhline(y=100, color='yellow', linestyle='--', label='Moderate AQI Threshold')
plt.axhline(y=150, color='orange', linestyle='--', label='Unhealthy for Sensitive Groups Threshold')
plt.legend()
plt.tight_layout()
plt.show()

# Boxplot of AQI by dominant pollutant
plt.figure(figsize=(12, 6))
sns.boxplot(x='dominant_pollutant', y='aqi_value', data=air_df)
plt.title('AQI Values by Dominant Pollutant')
plt.xlabel('Dominant Pollutant')
plt.ylabel('AQI Value')
plt.show()

# Top 20 cities with worst air quality
top_20_cities = air_df.sort_values('aqi_value', ascending=False).head(20)
plt.figure(figsize=(15, 8))
bars = sns.barplot(x='city_name', y='aqi_value', hue='country_name', dodge=False, data=top_20_cities)
plt.title('Top 20 Cities with Worst Air Quality')
plt.xlabel('City')
plt.ylabel('AQI Value')
plt.xticks(rotation=90)

# Add horizontal lines for AQI thresholds
plt.axhline(y=50, color='green', linestyle='--', label='Good')
plt.axhline(y=100, color='yellow', linestyle='--', label='Moderate')
plt.axhline(y=150, color='orange', linestyle='--', label='Unhealthy for Sensitive Groups')
plt.axhline(y=200, color='red', linestyle='--', label='Unhealthy')
plt.axhline(y=300, color='purple', linestyle='--', label='Very Unhealthy')

# Handle legend placement to avoid overlap
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Correlation heatmap of different pollutant values
correlation_columns = ['aqi_value', 'co_aqi_value', 'ozone_aqi_value', 'no2_aqi_value', 'pm2.5_aqi_value']
correlation_matrix = air_df[correlation_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Between Different Pollutants')
plt.tight_layout()
plt.show()

# Distribution of dominant pollutants by AQI category
pollutant_by_category = pd.crosstab(air_df['aqi_category'], air_df['dominant_pollutant'], normalize='index')

# Order the categories
ordered_categories = sorted(air_df['aqi_category'].unique(),
                          key=lambda x: aqi_category_map.get(x, 0))
pollutant_by_category = pollutant_by_category.loc[ordered_categories]

plt.figure(figsize=(12, 8))
pollutant_by_category.plot(kind='bar', stacked=True, colormap='tab10')
plt.title('Dominant Pollutants by AQI Category')
plt.xlabel('AQI Category')
plt.ylabel('Proportion')
plt.legend(title='Dominant Pollutant', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
