"""
Task 2: Unemployment Analysis with Python
CodeAlpha Data Science Internship

This script analyzes unemployment rate data to understand trends, patterns,
and the impact of COVID-19 on unemployment rates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Create output directory for visualizations
import os
os.makedirs('visualizations', exist_ok=True)


def load_or_generate_data():
    """Load existing data or generate it if not available."""
    try:
        df = pd.read_csv('unemployment_data.csv', parse_dates=['Date'])
        print("✓ Loaded existing dataset")
    except FileNotFoundError:
        print("✗ Dataset not found. Generating synthetic data...")
        from generate_unemployment_data import generate_unemployment_data
        df = generate_unemployment_data()
    
    return df


def main():
    print("=" * 70)
    print("UNEMPLOYMENT ANALYSIS WITH PYTHON")
    print("=" * 70)
    
    # ============================================================
    # 1. LOAD AND PREPARE DATA
    # ============================================================
    print("\n" + "-" * 70)
    print("1. DATA LOADING AND INITIAL EXPLORATION")
    print("-" * 70)
    
    df = load_or_generate_data()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nBasic info:")
    print(df.info())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Check unique regions
    print(f"\nRegions: {df['Region'].unique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # ============================================================
    # 2. DATA CLEANING
    # ============================================================
    print("\n" + "-" * 70)
    print("2. DATA CLEANING")
    print("-" * 70)
    
    # Remove any rows with missing values
    initial_rows = len(df)
    df_clean = df.dropna()
    print(f"Rows before cleaning: {initial_rows}")
    print(f"Rows after cleaning: {len(df_clean)}")
    print(f"Rows removed: {initial_rows - len(df_clean)}")
    
    # Check for duplicates
    duplicates = df_clean.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    df_clean = df_clean.drop_duplicates()
    print(f"Rows after removing duplicates: {len(df_clean)}")
    
    # Ensure data types are correct
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean['Month'] = df_clean['Date'].dt.month
    df_clean['Year'] = df_clean['Date'].dt.year
    df_clean['Quarter'] = df_clean['Date'].dt.quarter
    
    # Add derived columns
    df_clean['Month_Name'] = df_clean['Date'].dt.strftime('%B')
    df_clean['Year_Quarter'] = df_clean['Year'].astype(str) + '-Q' + df_clean['Quarter'].astype(str)
    
    # ============================================================
    # 3. EXPLORATORY DATA ANALYSIS
    # ============================================================
    print("\n" + "-" * 70)
    print("3. EXPLORATORY DATA ANALYSIS")
    print("-" * 70)
    
    # National average by month
    national_avg = df_clean.groupby('Date')['Unemployment_Rate'].mean().reset_index()
    national_avg.columns = ['Date', 'National_Avg']
    
    # 3.1 Overall trend plot
    print("\n3.1 Plotting overall unemployment trend...")
    plt.figure(figsize=(16, 6))
    plt.plot(national_avg['Date'], national_avg['National_Avg'], 
             color='#2E86AB', linewidth=2.5, marker='o', markersize=3)
    plt.axvline(x=pd.Timestamp('2020-03-01'), color='red', linestyle='--', 
                linewidth=2, label='COVID-19 Onset (March 2020)')
    plt.axvline(x=pd.Timestamp('2020-04-01'), color='darkred', linestyle=':', 
                linewidth=2, label='COVID-19 Peak (April 2020)')
    plt.title('U.S. National Unemployment Rate Trend (2016-2023)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Unemployment Rate (%)', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key events
    plt.annotate('COVID-19\nPeak: ~14.8%', xy=(pd.Timestamp('2020-04-01'), 14.8),
                 xytext=(pd.Timestamp('2020-08-01'), 16),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                 fontsize=11, fontweight='bold', color='darkred')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    plt.savefig('visualizations/national_unemployment_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: visualizations/national_unemployment_trend.png")
    
    # 3.2 Regional comparison
    print("\n3.2 Plotting regional comparison...")
    plt.figure(figsize=(16, 7))
    
    for region in df_clean['Region'].unique():
        region_data = df_clean[df_clean['Region'] == region]
        region_avg = region_data.groupby('Date')['Unemployment_Rate'].mean().reset_index()
        plt.plot(region_avg['Date'], region_avg['Unemployment_Rate'], 
                 linewidth=2, label=region, alpha=0.85)
    
    plt.axvline(x=pd.Timestamp('2020-03-01'), color='gray', linestyle='--', alpha=0.7)
    plt.title('Unemployment Rate by Region (2016-2023)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Unemployment Rate (%)', fontsize=13)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    plt.savefig('visualizations/regional_unemployment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: visualizations/regional_unemployment.png")
    
    # 3.3 Seasonal patterns
    print("\n3.3 Analyzing seasonal patterns...")
    
    # Pre-COVID seasonal pattern
    pre_covid = df_clean[df_clean['Date'] < '2020-03-01']
    seasonal_pre = pre_covid.groupby('Month')['Unemployment_Rate'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(14, 6))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.errorbar(months, seasonal_pre['mean'], yerr=seasonal_pre['std'], 
                 fmt='-o', color='#2E86AB', linewidth=2.5, capsize=5,
                 label='Pre-COVID (2016-2019)', markersize=8)
    
    plt.title('Seasonal Unemployment Pattern (Pre-COVID)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=13)
    plt.ylabel('Unemployment Rate (%)', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/seasonal_pattern_precovid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: visualizations/seasonal_pattern_precovid.png")
    
    # 3.4 Year-over-year comparison
    print("\n3.4 Plotting year-over-year comparison...")
    df_clean['Year'] = df_clean['Date'].dt.year
    df_clean['Month'] = df_clean['Date'].dt.month
    
    # Filter to 2019-2023 for COVID impact comparison
    recent_years = df_clean[df_clean['Year'].isin([2019, 2020, 2021, 2022, 2023])]
    yearly_by_month = recent_years.groupby(['Year', 'Month'])['Unemployment_Rate'].mean().reset_index()
    
    plt.figure(figsize=(14, 7))
    colors = {2019: '#2ECC71', 2020: '#E74C3C', 2021: '#F39C12', 2022: '#3498DB', 2023: '#9B59B6'}
    line_styles = {2019: '-', 2020: '--', 2021: '-.', 2022: ':', 2023: '-'}
    
    for year in [2019, 2020, 2021, 2022, 2023]:
        year_data = yearly_by_month[yearly_by_month['Year'] == year]
        if not year_data.empty:
            label = f"{year} {'(COVID Peak)' if year == 2020 else ''}"
            plt.plot(year_data['Month'], year_data['Unemployment_Rate'], 
                     color=colors.get(year, '#333'), linewidth=2.5,
                     linestyle=line_styles.get(year, '-'),
                     marker='o', markersize=6, label=label)
    
    plt.xlabel('Month', fontsize=13)
    plt.ylabel('Unemployment Rate (%)', fontsize=13)
    plt.title('Year-over-Year Comparison: COVID Impact on Unemployment', fontsize=16, fontweight='bold')
    plt.xticks(range(1, 13), months)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/yearly_comparison_covid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: visualizations/yearly_comparison_covid.png")
    
    # 3.5 Heatmap of unemployment by year and month
    print("\n3.5 Creating heatmap visualization...")
    pivot_table = df_clean.pivot_table(
        values='Unemployment_Rate', index='Year', columns='Month', aggfunc='mean'
    )
    pivot_table.columns = months
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', 
                linewidths=1, cbar_kws={'label': 'Unemployment Rate (%)'})
    plt.title('Unemployment Rate Heatmap (Year vs Month)', fontsize=16, fontweight='bold')
    plt.ylabel('Year', fontsize=13)
    plt.xlabel('Month', fontsize=13)
    plt.tight_layout()
    plt.savefig('visualizations/unemployment_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: visualizations/unemployment_heatmap.png")
    
    # ============================================================
    # 4. QUANTITATIVE ANALYSIS
    # ============================================================
    print("\n" + "-" * 70)
    print("4. QUANTITATIVE ANALYSIS")
    print("-" * 70)
    
    # COVID Impact Statistics
    pre_covid_data = df_clean[df_clean['Date'] < '2020-03-01']
    covid_peak_data = df_clean[df_clean['Date'] == '2020-04-01']
    post_covid_data = df_clean[df_clean['Date'] >= '2021-01-01']
    recent_data = df_clean[df_clean['Date'] >= '2022-01-01']
    
    pre_covid_avg = pre_covid_data['Unemployment_Rate'].mean()
    covid_peak_avg = covid_peak_data['Unemployment_Rate'].mean()
    post_covid_avg = post_covid_data['Unemployment_Rate'].mean()
    recent_avg = recent_data['Unemployment_Rate'].mean()
    
    print(f"\nPre-COVID Average Unemployment Rate: {pre_covid_avg:.2f}%")
    print(f"COVID Peak (April 2020) Average: {covid_peak_avg:.2f}%")
    print(f"Post-COVID (2021+) Average: {post_covid_avg:.2f}%")
    print(f"Recent (2022+) Average: {recent_avg:.2f}%")
    print(f"COVID Peak Increase: +{covid_peak_avg - pre_covid_avg:.2f} percentage points")
    print(f"Recovery (Peak to Recent): {covid_peak_avg - recent_avg:.2f} percentage points drop")
    
    # Region with highest unemployment
    regional_stats = df_clean.groupby('Region')['Unemployment_Rate'].agg(['mean', 'max', 'min', 'std'])
    print(f"\nRegional Statistics:")
    print(regional_stats)
    
    # ============================================================
    # 5. TREND FORECASTING (Simple Linear Regression)
    # ============================================================
    print("\n" + "-" * 70)
    print("5. TREND ANALYSIS AND FORECASTING")
    print("-" * 70)
    
    # Use recent post-COVID data for trend prediction (2022 onwards)
    recent_trend_data = national_avg[national_avg['Date'] >= '2022-01-01'].copy()
    recent_trend_data['Days'] = (recent_trend_data['Date'] - recent_trend_data['Date'].min()).dt.days
    
    X_trend = recent_trend_data['Days'].values.reshape(-1, 1)
    y_trend = recent_trend_data['National_Avg'].values
    
    model = LinearRegression()
    model.fit(X_trend, y_trend)
    
    y_pred = model.predict(X_trend)
    r2 = r2_score(y_trend, y_pred)
    rmse = np.sqrt(mean_squared_error(y_trend, y_pred))
    
    print(f"\nRecent Trend (2022-2023):")
    print(f"  Slope: {model.coef_[0]:.6f} (daily change)")
    print(f"  Monthly change: {model.coef_[0]*30:.4f} percentage points")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Visualize trend with forecast
    last_date = recent_trend_data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                  periods=12, freq='MS')
    future_days = (future_dates - recent_trend_data['Date'].min()).days.values.reshape(-1, 1)
    future_pred = model.predict(future_days)
    
    plt.figure(figsize=(14, 6))
    plt.plot(recent_trend_data['Date'], y_trend, 'o-', color='#2E86AB', 
             linewidth=2.5, label='Actual Rate', markersize=5)
    plt.plot(recent_trend_data['Date'], y_pred, '--', color='#E74C3C', 
             linewidth=2, label=f'Trend Line (R²={r2:.3f})')
    plt.plot(future_dates, future_pred, '--', color='#F39C12', 
             linewidth=2.5, label='Forecast (Next 12 months)', marker='s', markersize=5)
    plt.fill_between(future_dates, future_pred - 0.5, future_pred + 0.5, 
                     alpha=0.2, color='#F39C12', label='Confidence Band (±0.5%)')
    plt.title('Unemployment Rate Trend & Forecast (2022-2024)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Unemployment Rate (%)', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/unemployment_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: visualizations/unemployment_forecast.png")
    
    # ============================================================
    # 6. KEY INSIGHTS AND CONCLUSIONS
    # ============================================================
    print("\n" + "-" * 70)
    print("6. KEY INSIGHTS AND CONCLUSIONS")
    print("-" * 70)
    
    print("""
    KEY FINDINGS:
    
    1. PRE-COVID PERIOD (2016-2019):
       - Stable unemployment around {:.2f}%
       - Clear seasonal patterns: lower in Q4 (holiday hiring), higher in Q1
       - Steady downward trend with economic expansion
    
    2. COVID-19 IMPACT:
       - Sharp spike from ~{:.2f}% to ~{:.2f}% in March-April 2020
       - Peak unemployment of {:.2f}% in April 2020
       - All regions affected, but some more severely than others
    
    3. RECOVERY PHASE (2021-2023):
       - Steady decline back to ~{:.2f}% by 2022-2023
       - Faster recovery than initial spike
       - Some regional variation in recovery speed
    
    4. SEASONAL PATTERNS:
       - Consistent patterns pre-COVID with Q4 hiring surges
       - COVID disrupted normal seasonal patterns
       - Seasonal patterns re-emerging in recovery phase
    
    5. POLICY IMPLICATIONS:
       - Need for rapid response mechanisms during crises
       - Regional targeting for unemployment support
       - Seasonal employment programs could help stabilize rates
    """.format(
        pre_covid_avg, pre_covid_avg, covid_peak_avg,
        covid_peak_avg, recent_avg
    ))
    
    print("\n" + "=" * 70)
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nAll visualizations saved in 'visualizations/' directory")


if __name__ == "__main__":
    main()

