import streamlit as st
import pandas as pd
import plotly.express as px
import copy

# Import your simulation modules
from parameters import get_default_params
from simulation_model import VaccineSimulation
from analysis import calculate_summary_metrics
# from utils import plot_results (or define plotting here)

st.set_page_config(layout="wide")
st.title("💉 SEIR & Vaccine Supply Chain Simulation")

# --- Session State Initialization ---
if 'params' not in st.session_state:
    st.session_state.params = get_default_params()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'scenario_results' not in st.session_state:
    st.session_state.scenario_results = {} # Store results for comparison

# --- Sidebar for Parameter Configuration ---
st.sidebar.header("Configuration")
st.sidebar.subheader("Edit Parameters")

# Make parameters editable in the sidebar
# Use a deepcopy to allow editing without modifying the default dict directly
editable_params = copy.deepcopy(st.session_state.params)

# Display key parameters for editing (Add more as needed)
editable_params["Ro"] = st.sidebar.number_input("Basic Reproduction Number (Ro)", value=editable_params["Ro"], step=0.1, format="%.1f")
editable_params["Time_to_Vaccination"] = st.sidebar.number_input("Vaccination Start Week", value=editable_params["Time_to_Vaccination"], step=1, min_value=0)
editable_params["Desired_Order"] = st.sidebar.number_input("Vaccine Doses Ordered", value=editable_params["Desired_Order"], step=10000, min_value=0)
editable_params["Max_Vaccine_Dispensed"] = st.sidebar.number_input("Max Weekly Dispensing Capacity", value=editable_params["Max_Vaccine_Dispensed"], step=1000, min_value=0)
editable_params["Vaccine_Efficacy"] = st.sidebar.slider("Vaccine Efficacy", min_value=0.0, max_value=1.0, value=editable_params["Vaccine_Efficacy"], step=0.05)
editable_params["Production_to_Shipping_Delay"] = st.sidebar.number_input("Production/Shipping Delay (Weeks)", value=editable_params["Production_to_Shipping_Delay"], step=1, min_value=0)
# Add more parameters... N, ID, ED, Time_to_Infection etc. for full control

# Update button in sidebar
if st.sidebar.button("Update Parameters"):
    st.session_state.params = editable_params
    st.session_state.results = None # Clear results when params change
    st.session_state.metrics = None
    st.rerun() # Rerun the script to reflect changes

st.sidebar.divider()
st.sidebar.subheader("Run Simulation")
run_button = st.sidebar.button("Run Simulation")

st.sidebar.divider()
st.sidebar.subheader("Scenario Management")
scenario_name = st.sidebar.text_input("Save current run as scenario:", "")
save_scenario_button = st.sidebar.button("Save Scenario")

# --- Main Area for Displaying Results ---
st.header("Simulation Parameters (Current)")
st.json(st.session_state.params) # Display current parameters

if run_button:
    try:
        # Create and run simulation with current parameters
        simulation = VaccineSimulation(st.session_state.params)
        results_df = simulation.run()
        metrics = calculate_summary_metrics(results_df, st.session_state.params)

        # Store results in session state
        st.session_state.results = results_df
        st.session_state.metrics = metrics

        if scenario_name: # Auto-save if name provided when running
            st.session_state.scenario_results[scenario_name] = {'params': copy.deepcopy(st.session_state.params), 'results': results_df, 'metrics': metrics}
            st.sidebar.success(f"Scenario '{scenario_name}' saved.")


    except ValueError as e:
         st.error(f"Simulation Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


if save_scenario_button and scenario_name and st.session_state.results is not None:
     st.session_state.scenario_results[scenario_name] = {'params': copy.deepcopy(st.session_state.params), 'results': st.session_state.results, 'metrics': st.session_state.metrics}
     st.sidebar.success(f"Scenario '{scenario_name}' saved.")
elif save_scenario_button and not scenario_name:
     st.sidebar.warning("Please enter a name for the scenario.")
elif save_scenario_button and st.session_state.results is None:
     st.sidebar.warning("Please run a simulation before saving.")


# Display results if available
if st.session_state.results is not None:
    st.header("Simulation Results")

    # --- Summary Metrics ---
    st.subheader("Summary Metrics")
    if st.session_state.metrics:
        st.json(st.session_state.metrics)
    else:
        st.warning("Metrics could not be calculated.")

    # --- Plots ---
    st.subheader("Plots")
    results_df = st.session_state.results

    # Plot SEIRV compartments
    st.write("SEIRV Dynamics Over Time")
    seirv_df = results_df[['Week', 'S', 'E', 'I', 'R', 'V']]
    seirv_melted = seirv_df.melt(id_vars=['Week'], var_name='Compartment', value_name='Count')
    fig_seirv = px.line(seirv_melted, x='Week', y='Count', color='Compartment',
                        title="Population Compartments Over Time",
                        labels={'Count': 'Number of Individuals'})
    st.plotly_chart(fig_seirv, use_container_width=True)

    # Plot Supply Chain Data
    st.write("Vaccine Supply Chain Dynamics")
    supply_df = results_df[['Week', 'Arrived_Stock', 'Vaccines_Used_Week']] # Add more tracked variables
    supply_melted = supply_df.melt(id_vars=['Week'], var_name='Metric', value_name='Count')
    fig_supply = px.line(supply_melted, x='Week', y='Count', color='Metric',
                         title="Vaccine Stock and Usage Over Time",
                         labels={'Count': 'Number of Doses'})
    st.plotly_chart(fig_supply, use_container_width=True)


    # --- Raw Data ---
    st.subheader("Raw Simulation Data")
    st.dataframe(results_df)

# --- What-if / Scenario Comparison ---
st.header("What-If Analysis / Scenario Comparison")
if len(st.session_state.scenario_results) > 1:
    scenarios_to_compare = st.multiselect("Select scenarios to compare:", options=list(st.session_state.scenario_results.keys()))

    if len(scenarios_to_compare) > 1:
        st.subheader("Comparison")
        comparison_data = []
        for name in scenarios_to_compare:
            metrics = st.session_state.scenario_results[name]['metrics']
            metrics['Scenario'] = name
            # Could also add key differing parameters here
            # params = st.session_state.scenario_results[name]['params']
            # metrics['Ro'] = params.get('Ro', 'N/A')
            # metrics['Order'] = params.get('Desired_Order', 'N/A')
            comparison_data.append(metrics)

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.set_index('Scenario'))

        # Add comparative plots (e.g., Infected curves for selected scenarios)
        st.write("Comparison Plots (Example: Infected Count)")
        plot_data = []
        for name in scenarios_to_compare:
             temp_df = st.session_state.scenario_results[name]['results'][['Week', 'I']].copy()
             temp_df['Scenario'] = name
             plot_data.append(temp_df)
        combined_plot_df = pd.concat(plot_data)
        fig_compare_i = px.line(combined_plot_df, x='Week', y='I', color='Scenario', title="Infected Count Comparison")
        st.plotly_chart(fig_compare_i, use_container_width=True)

        # Add more comparison plots as needed (Attack Rate bar chart, Cost bar chart etc.)

else:
    st.info("Save at least two scenarios using the sidebar to enable comparison.")


# --- Placeholder sections for future features ---
st.header("Advanced Features (Future Implementation)")
with st.expander("Automated Web Search"):
    search_query = st.text_input("Search the web:")
    if st.button("Search"):
        st.info("Web search functionality not yet implemented.")
        # Placeholder: Add call to web_search.py function here
        # try:
        #     tool_code
        #     from web_search import search # Assuming web_search.py exists
        #     results = search(search_query)
        #     st.write(results)
        #     print(results) # tool output
        #     code_output
        # except ImportError:
        #      st.error("Web search module not found.")
        # except Exception as e:
        #      st.error(f"Search failed: {e}")


with st.expander("Geospatial Reasoning"):
    st.info("Geospatial reasoning functionality not yet implemented. Requires specific requirements (e.g., map data, regional parameters).")

with st.expander("Bias Check"):
    st.info("Bias check functionality not yet implemented. Requires definition of specific biases to check.")
    st.markdown("""
    **Potential Model Biases/Assumptions:**
    *   **Homogeneous Mixing:** Assumes everyone has an equal chance of interacting with everyone else. Ignores population structure, geography, social networks.
    *   **Uniform Parameters:** Assumes parameters (Ro, Efficacy, Durations) are constant for the entire population and season. Ignores age/risk group variations.
    *   **Perfect Reporting/Data:** Assumes input parameters and initial conditions are accurate.
    *   **Simplified Supply Chain:** Doesn't model complex logistics, wastage due to cold chain failure, provider capacity variations in detail.
    *   **No Behavioral Changes:** Assumes population behavior (contact rates) doesn't change in response to the outbreak or vaccination campaign.
    """)