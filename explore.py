import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@st.cache_data
def get_data():
    # Load dataset
    data = pd.read_csv("vehicle_collisions.csv", low_memory = False)

    # Setting the decimal to 6 points
    pd.options.display.float_format = '{:.6f}'.format

    # Dropping unwanted columns
    data = data.drop(columns = ['VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5', 'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5'])

    '''
    Since number of null values are very few in NUMBER OF PERSONS INJURED, NUMBER OF PERSONS KILLED, CONTRIBUTING FACTOR VEHICLE 1,
    VEHICLE TYPE CODE 1, Let us drop those columns
    '''
    columns_to_drop = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1']
    data.dropna(subset=columns_to_drop, how='any', inplace=True)

    '''
    Removing redudant columns.
    Since Location can be traced from Latitude and longitude, lets drop other columns related to location
    '''
    data = data.drop(columns=['ZIP CODE', 'ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME', 'LOCATION'])

    # Get all the unique values of Latitude, Longitude and Borough
    unique_location_values = data[['LATITUDE', 'LONGITUDE', 'BOROUGH']].drop_duplicates()
    unique_location_values.dropna(inplace=True)

    # Out of these values, lets pick first unique occurance of borough and its corresponding latitude and longitude
    single_location_values = unique_location_values.groupby('BOROUGH').first().reset_index()

    # Get all samples where borough, latitude and longitude are null
    location_null = data['BOROUGH'].isnull() & data['LATITUDE'].isnull() & data['LONGITUDE'].isnull()

    # Replace boorugh with mode where all the 3 parameters are null
    data.loc[location_null, 'BOROUGH'] = data['BOROUGH'].mode().iloc[0]

    # Fetch the latitude and longitude for the mode borough
    borough_mode = data['BOROUGH'].mode().iloc[0]
    borough_mode_lat = single_location_values[single_location_values['BOROUGH'] == borough_mode]['LATITUDE'].values[0]
    borough_mode_lon = single_location_values[single_location_values['BOROUGH'] == borough_mode]['LONGITUDE'].values[0]

    # Replace LATITUDE and LONGITUDE values for these records using the single_location_values DataFrame
    data.loc[location_null, 'LATITUDE'] = borough_mode_lat
    data.loc[location_null, 'LONGITUDE'] = borough_mode_lon

    # Identify records where either LATITUDE or LONGITUDE is null, but BOROUGH has a value
    latitude_longitude_null = (data['LATITUDE'].isnull() | data['LONGITUDE'].isnull()) & ~data['BOROUGH'].isnull()

    # Iterate over these rows and impute the LATITUDE and LONGITUDE based on single_location_values
    for index, row in data[latitude_longitude_null].iterrows():
        borough_val = row['BOROUGH']

        # Fetch the latitude and longitude for the given borough from single_location_values
        borough_lat = single_location_values[single_location_values['BOROUGH'] == borough_val]['LATITUDE'].values[0]
        borough_lon = single_location_values[single_location_values['BOROUGH'] == borough_val]['LONGITUDE'].values[0]

        # Update LATITUDE and LONGITUDE values for these records using the single_location_values DataFrame
        data.loc[index, 'LATITUDE'] = borough_lat
        data.loc[index, 'LONGITUDE'] = borough_lon

    # Identify records where LATITUDE and LONGITUDE have values, but BOROUGH is null
    borough_null = ~data['LATITUDE'].isnull() & ~data['LONGITUDE'].isnull() & data['BOROUGH'].isnull()

    # Iterate over these rows and impute the BOROUGH based on unique_location_values
    for index, row in data[borough_null].iterrows():
        latitude_val = row['LATITUDE']
        longitude_val = row['LONGITUDE']

        # Fetch the borough for the given latitude and longitude from unique_location_values
        borough_value = unique_location_values[
            (unique_location_values['LATITUDE'] == latitude_val) &
            (unique_location_values['LONGITUDE'] == longitude_val)]['BOROUGH'].values

        # Check if we found a matching borough
        if borough_value.size > 0:
            # Update BOROUGH value for this record using the unique_location_values DataFrame
            data.loc[index, 'BOROUGH'] = borough_value[0]

    # Replace null borough values with mode
    data['BOROUGH'].fillna(data['BOROUGH'].mode()[0], inplace=True)

    # Drop duplicate records
    data.drop_duplicates(inplace=True)

    # Instead of having 2 separate columns for CRASH DATE and CRASH TIME, lets merge them into a single column of datatime format
    data['CRASH DATE TIME'] = pd.to_datetime(data['CRASH DATE'] + ' ' + data['CRASH TIME'])
    data.drop(columns=['CRASH DATE', 'CRASH TIME'], inplace=True)

    # Standerdize Categorical data
    def standerdizer(column_name):
        data.loc[:,column_name] = data[column_name].str.upper()

    categorical_columns=['BOROUGH','VEHICLE TYPE CODE 1','CONTRIBUTING FACTOR VEHICLE 1']

    for column in categorical_columns:
        standerdizer(column)

    return data

data = get_data()

def view_explore_page():
    st.title("Number of Collisions per Month (by Year)")
    # Exploring Number of collisions per month of a year
    data.loc[:,'Year'] = data['CRASH DATE TIME'].dt.year
    data.loc[:,'Month'] = data['CRASH DATE TIME'].dt.month

    # Group the data by year and month and count the number of collisions in each group
    monthly_collisions = data.groupby(['Year', 'Month']).size().unstack(fill_value=0)

    # Create a list of month names for labeling the x-axis
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot the number of collisions per month for each year
    plt.figure(figsize=(12, 6))
    for year in monthly_collisions.index:
        plt.bar(month_names, monthly_collisions.loc[year], label=str(year))
    
    plt.xlabel('Month')
    plt.ylabel('Number of Collisions')
    plt.title('Number of Collisions per Month (by Year)')
    plt.legend(title='Year')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

    # Exploring number of collision per year
    st.title("Number of Collisions per Year")
    data.loc[:,'Year'] = data['CRASH DATE TIME'].dt.year

    # Group the data by year and count the number of collisions in each year
    yearly_collisions = data['Year'].value_counts().sort_index()

    # Plot the number of collisions per year
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_collisions.index, yearly_collisions.values)
    plt.xlabel('Year')
    plt.ylabel('Number of Collisions')
    plt.title('Number of Collisions per Year')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

    # Count the number of crashes for each borough
    st.title("Number of Crashes by Borough")
    borough_counts = data['BOROUGH'].value_counts()

    plt.figure(figsize=(10,6))
    borough_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Number of Crashes by Borough')
    plt.xlabel('Borough')
    plt.ylabel('Number of Crashes')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    st.pyplot(plt)

    # Plot to visualize the top 20 contribution factor for a crash
    st.title("Top 20 Contributing Factors for Crashes")
    factor_counts = data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
    top_factors = factor_counts.head(20)

    plt.figure(figsize=(12, 6))
    top_factors.plot(kind='barh', color='teal', edgecolor='black')
    plt.title('Top 20 Contributing Factors for Crashes')
    plt.xlabel('Number of Crashes')
    plt.ylabel('Contributing Factor')
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.tight_layout()
    st.pyplot(plt)

    # Visualize top 20 vehicles involved in a crash
    st.title("Top 20 Vehicle Types Involved in Crashes")
    vehicle_counts = data['VEHICLE TYPE CODE 1'].value_counts()
    top_vehicles = vehicle_counts.head(20)

    plt.figure(figsize=(12, 6))
    top_vehicles.plot(kind='bar', color='lightblue', edgecolor='black')
    plt.title('Top 20 Vehicle Types Involved in Crashes')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Number of Crashes')
    plt.xticks(rotation=75)
    plt.grid(axis='y')

    plt.tight_layout()
    st.pyplot(plt)