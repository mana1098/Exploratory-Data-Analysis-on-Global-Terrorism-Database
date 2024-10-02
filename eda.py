import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Load the data
df = pd.read_csv('global_terrorism_db.csv', encoding='latin1', low_memory=False)

# Data Cleaning and Preprocessing
def clean_data(df):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['iyear'].astype(str) + '-' + df['imonth'].astype(str) + '-' + df['iday'].astype(str))
    
    # Filter for recent years (e.g., last 20 years)
    df = df[df['iyear'] >= 2000]
    
    # Handle missing values
    df['nkill'] = df['nkill'].fillna(0)
    df['nwound'] = df['nwound'].fillna(0)
    
    return df

df = clean_data(df)

# Exploratory Data Analysis Functions

def plot_attacks_over_time(df):
    attacks_by_year = df.groupby('iyear').size()
    plt.figure(figsize=(12, 6))
    attacks_by_year.plot(kind='line')
    plt.title('Number of Terrorist Attacks Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Attacks')
    plt.show()

def plot_top_countries(df, n=10):
    top_countries = df['country_txt'].value_counts().head(n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_countries.index, y=top_countries.values)
    plt.title(f'Top {n} Countries by Number of Terrorist Attacks')
    plt.xlabel('Country')
    plt.ylabel('Number of Attacks')
    plt.xticks(rotation=45)
    plt.show()

def plot_attack_types(df):
    attack_types = df['attacktype1_txt'].value_counts()
    plt.figure(figsize=(12, 6))
    attack_types.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Attack Types')
    plt.ylabel('')
    plt.show()

def plot_casualty_trends(df):
    yearly_casualties = df.groupby('iyear')[['nkill', 'nwound']].sum()
    plt.figure(figsize=(12, 6))
    yearly_casualties.plot(kind='line')
    plt.title('Yearly Trends in Casualties')
    plt.xlabel('Year')
    plt.ylabel('Number of Casualties')
    plt.legend(['Killed', 'Wounded'])
    plt.show()

def plot_geospatial_distribution(df):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    attack_counts = df['country_txt'].value_counts()
    world['attacks'] = world['name'].map(attack_counts)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.plot(column='attacks', ax=ax, legend=True, 
               legend_kwds={'label': 'Number of Attacks', 'orientation': 'horizontal'},
               missing_kwds={'color': 'lightgrey'}, cmap='YlOrRd')
    ax.set_title('Geospatial Distribution of Terrorist Attacks')
    plt.axis('off')
    plt.show()

def analyze_causal_factors(df):
    corr = df[['nkill', 'nwound', 'property', 'success']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Potential Causal Factors')
    plt.show()

# Main analysis
def main_analysis(df):
    print("Performing Exploratory Data Analysis on Global Terrorism Database...")
    
    plot_attacks_over_time(df)
    plot_top_countries(df)
    plot_attack_types(df)
    plot_casualty_trends(df)
    plot_geospatial_distribution(df)
    analyze_causal_factors(df)
    
    print("\nKey Insights:")
    print(f"1. Total number of attacks: {len(df)}")
    print(f"2. Most common attack type: {df['attacktype1_txt'].mode().values[0]}")
    print(f"3. Country with most attacks: {df['country_txt'].value_counts().index[0]}")
    print(f"4. Year with most attacks: {df['iyear'].value_counts().index[0]}")
    print(f"5. Total casualties: {df['nkill'].sum()} killed, {df['nwound'].sum()} wounded")

    # Additional analyses can be added here

if __name__ == "__main__":
    main_analysis(df)

# Strategic Insights and Recommendations (to be added based on the analysis results)
def generate_recommendations(df):
    # This function would analyze the results and generate strategic recommendations
    # For example:
    high_risk_countries = df['country_txt'].value_counts().head(5).index.tolist()
    print("\nStrategic Recommendations:")
    print(f"1. Focus on high-risk countries: {', '.join(high_risk_countries)}")
    print("2. Implement targeted prevention strategies based on the most common attack types")
    print("3. Enhance international cooperation for cross-border terrorism prevention")
    # Add more recommendations based on the analysis

generate_recommendations(df)
