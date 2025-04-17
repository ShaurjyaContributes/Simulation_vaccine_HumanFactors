import pandas as pd
import numpy as np # Needed for infinity check

def calculate_summary_metrics(results_df, params):
    """Calculates key summary metrics from the simulation results."""
    if results_df is None or results_df.empty:
        return {}

    N = params['N']
    final_week = results_df['Week'].max()
    final_row = results_df.iloc[-1]

    # Disease Outcomes
    # Total ever infected = Initial Infected + sum of New_Infections_Week
    # Or approximate as N - S(end) - V(end) if simulation runs long enough for E, I to go near zero
    # Let's use R(end) + I(end) + E(end) - R(0) - I(0) - E(0) + Initial_Infected
    initial_infected_compartments = params.get('Initial_Infected', 0) # Assuming they start in I
    total_ever_infected = final_row['R'] + final_row['I'] + final_row['E'] # R(0)=E(0)=0

    attack_rate = total_ever_infected / N if N > 0 else 0
    peak_infected_count = results_df['I'].max()
    # Handle potential NaNs or infinities if simulation fails
    if pd.isna(peak_infected_count) or np.isinf(peak_infected_count):
        peak_infected_count = -1 # Indicate error
        peak_infection_week = -1
    else:
        peak_infection_week = results_df['I'].idxmax() if peak_infected_count > 0 else 0


    # Vaccination & Supply Chain Outcomes
    vaccines_used_cumulative = final_row['Used_Total']
    # Total vaccine produced is based on the initial order in this simple model
    total_vaccine_produced = params['Desired_Order']
    # Handle division by zero
    vaccine_yield = (vaccines_used_cumulative / total_vaccine_produced) if total_vaccine_produced > 0 else 0

    # Cost Outcomes
    per_vaccine_cost = params['Per_Vaccine_Cost']
    # Cost based on doses produced/ordered
    total_vaccine_cost = total_vaccine_produced * per_vaccine_cost

    # Ensure results are serializable (convert numpy types if needed)
    peak_infected_count = int(peak_infected_count) if peak_infected_count != -1 else 'Error'
    peak_infection_week = int(peak_infection_week) if peak_infection_week != -1 else 'Error'
    vaccines_used_cumulative = int(vaccines_used_cumulative)


    return {
        "Attack Rate (%)": f"{attack_rate * 100:.2f}",
        "Peak Infected Count": f"{peak_infected_count:,}" if isinstance(peak_infected_count, int) else peak_infected_count,
        "Peak Infection Week": peak_infection_week,
        "Vaccines Used (Cumulative)": f"{vaccines_used_cumulative:,.0f}",
        "Total Vaccine Produced (Ordered)": f"{total_vaccine_produced:,.0f}",
        "Vaccine Yield (%)": f"{vaccine_yield * 100:.2f}", # Used / Produced
        "Total Vaccine Cost (Euros)": f"{total_vaccine_cost:,.2f}"
    }
