import pandas as pd

def calculate_summary_metrics(results_df, params):
    """Calculates key summary metrics from the simulation results."""
    if results_df is None or results_df.empty:
        return {}

    N = params['N']
    final_week = results_df['Week'].max()
    final_row = results_df.iloc[-1]

    # Disease Outcomes
    total_infected = N - final_row['S'] - final_row['V'] # Everyone not S or V at the end got infected at some point (approx)
    # More accurate: Sum of New_Infected over time, or N - S(end) - V(end) - E(end) - I(end) - R(0)? Need careful definition. Let's use peak I.
    # Attack Rate based on paper's likely definition (cumulative incidence proportion)
    # Need cumulative infections. Can estimate from R + I + E at the end? Or sum New_Infected? Let's use R(end) + I(end) + E(end)
    total_ever_infected = final_row['R'] + final_row['I'] + final_row['E']
    attack_rate = total_ever_infected / N if N > 0 else 0
    peak_infected_count = results_df['I'].max()
    peak_infection_week = results_df['I'].idxmax() if peak_infected_count > 0 else 0

    # Vaccination & Supply Chain Outcomes
    vaccines_used_cumulative = final_row['Used_Total']
    # total_vaccine_produced = params['Desired_Order'] # Assuming production meets order exactly in this simplified model
    # Let's refine: track production output instead. Need to add this tracking to sim.
    # For now, assume Desired Order = Produced eventually
    total_vaccine_produced = params['Desired_Order']
    vaccine_yield = (vaccines_used_cumulative / total_vaccine_produced) if total_vaccine_produced > 0 else 0

    # Cost Outcomes
    per_vaccine_cost = params['Per_Vaccine_Cost']
    # Cost based on doses produced/ordered
    total_vaccine_cost = total_vaccine_produced * per_vaccine_cost

    return {
        "Attack Rate (%)": f"{attack_rate * 100:.2f}",
        "Peak Infected Count": f"{peak_infected_count:,.0f}",
        "Peak Infection Week": peak_infection_week,
        "Vaccines Used (Cumulative)": f"{vaccines_used_cumulative:,.0f}",
        "Total Vaccine Produced (Ordered)": f"{total_vaccine_produced:,.0f}",
        "Vaccine Yield (%)": f"{vaccine_yield * 100:.2f}",
        "Total Vaccine Cost (Euros)": f"{total_vaccine_cost:,.2f}"
    }