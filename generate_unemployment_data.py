"""
Generate realistic unemployment data based on historical U.S. patterns
including the impact of COVID-19.
"""

import numpy as np
import pandas as pd
import os


def generate_unemployment_data(start_year=2016, end_year=2023, output_file="unemployment_data.csv"):
    """
    Generate realistic monthly unemployment rate data.
    
    Parameters:
    - start_year: Starting year for data generation
    - end_year: Ending year for data generation
    - output_file: Path to save the CSV file
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')
    n_months = len(dates)
    
    # Base unemployment rate with moderate seasonal component
    base_rate = 4.5
    
    # Seasonal component (higher in Jan/Feb, lower in Nov/Dec)
    seasonal = np.array([
        0.4, 0.3, 0.1, -0.1, -0.2, -0.3,
        -0.3, -0.2, 0.0, 0.1, 0.2, 0.3
    ])
    
    # Year-over-year trend (slight decline pre-COVID, recovery post-COVID)
    yearly_trend = np.linspace(0, 0, n_months)  # Will be overridden
    
    # Pre-COVID trend: slight decline
    pre_covid_mask = dates < pd.Timestamp("2020-03-01")
    pre_covid_indices = np.where(pre_covid_mask)[0]
    if len(pre_covid_indices) > 0:
        yearly_trend[pre_covid_indices] = -np.linspace(0.8, 0, len(pre_covid_indices))
    
    # COVID-19 impact (March 2020 spike with gradual decline)
    covid_start = pd.Timestamp("2020-03-01")
    covid_peak = pd.Timestamp("2020-04-01")
    recovery_end = pd.Timestamp("2023-01-01")
    
    unemployment_rates = []
    
    for date, base, season in zip(dates, np.full(n_months, base_rate), 
                                   np.tile(seasonal, (n_months + 11) // 12)[:n_months]):
        rate = base + season
        
        # Pre-COVID trend
        if date < covid_start:
            months_from_start = (date.year - start_year) * 12 + (date.month - 1)
            rate -= 0.04 * months_from_start  # Gradual decline
        
        # COVID period
        if date >= covid_start:
            if date <= covid_peak:
                # Sharp increase to peak
                days_to_peak = (covid_peak - date).days
                spike = 10.3 * (1 - max(0, days_to_peak) / 31)  # Up to ~14.8%
                rate += spike
            elif date <= recovery_end:
                # Gradual recovery
                days_since_peak = (date - covid_peak).days
                recovery_days = (recovery_end - covid_peak).days
                recovery_factor = min(1, days_since_peak / recovery_days)
                # Recovery: from ~14.8% back to ~4% but with bumps
                spike_remaining = 10.3 * (1 - recovery_factor) * (1 + 0.1 * np.sin(days_since_peak / 90))
                rate += spike_remaining
            else:
                # Post-recovery with some noise
                rate += np.random.normal(0, 0.15)
        
        # Add random noise
        rate += np.random.normal(0, 0.15)
        
        # Ensure rate is not negative
        rate = max(0.5, rate)
        unemployment_rates.append(round(rate, 2))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Month': dates.month,
        'Year': dates.year,
        'Unemployment_Rate': unemployment_rates
    })
    
    # Add region column (simulate multiple regions)
    regions = ['Northeast', 'Midwest', 'South', 'West']
    all_data = []
    
    for region in regions:
        region_df = df.copy()
        region_df['Region'] = region
        
        # Different regions have slightly different rates
        if region == 'Northeast':
            region_df['Unemployment_Rate'] += np.random.normal(0.2, 0.1, n_months)
        elif region == 'Midwest':
            region_df['Unemployment_Rate'] += np.random.normal(-0.1, 0.15, n_months)
        elif region == 'South':
            region_df['Unemployment_Rate'] += np.random.normal(0.1, 0.12, n_months)
        elif region == 'West':
            region_df['Unemployment_Rate'] += np.random.normal(0.0, 0.18, n_months)
        
        region_df['Unemployment_Rate'] = region_df['Unemployment_Rate'].clip(0.5, 20).round(2)
        all_data.append(region_df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"✓ Unemployment data saved to '{output_file}'")
    print(f"  Records: {len(final_df)}")
    print(f"  Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"  Regions: {final_df['Region'].nunique()}")
    print(f"  Unemployment range: {final_df['Unemployment_Rate'].min():.1f}% - {final_df['Unemployment_Rate'].max():.1f}%")
    
    # Also save a national-level aggregated dataset
    national_df = df.copy()
    national_df.to_csv("unemployment_data_national.csv", index=False)
    print(f"✓ National-level data saved to 'unemployment_data_national.csv'")
    
    return final_df


if __name__ == "__main__":
    generate_unemployment_data()

