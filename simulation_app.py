import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # For more control over plots
import copy
import time # For simulation animation delay

# Import your simulation modules
from parameters import get_default_params, SENSITIVITY_RANGES
from simulation_model import VaccineSimulation
from analysis import calculate_summary_metrics
# from utils import plot_results (plotting defined here)

st.set_page_config(layout="wide", page_title="Vaccine Sim")
st.title("💉 SEIR & Vaccine Supply Chain Simulation")
st.markdown("Integrates SEIR disease spread with vaccine logistics and behavioral adoption.")

# --- Session State Initialization ---
if 'params' not in st.session_state:
    st.session_state.params = get_default_params()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'scenario_results' not in st.session_state:
    st.session_state.scenario_results = {} # Store results for comparison
if 'sensitivity_results' not in st.session_state:
    st.session_state.sensitivity_results = None # Store sensitivity analysis data

# --- Sidebar for Parameter Configuration ---
st.sidebar.header("Simulation Configuration")
st.sidebar.subheader("Edit Core Parameters")

# Use a deepcopy to allow editing without modifying the default dict directly
editable_params = copy.deepcopy(st.session_state.params)

# Display key parameters for editing (Add more as needed)
col1, col2 = st.sidebar.columns(2)
with col1:
    editable_params["N"] = st.number_input("Population (N)", value=editable_params["N"], step=100000, min_value=1000)
    editable_params["Ro"] = st.number_input("Ro", value=editable_params["Ro"], step=0.1, format="%.1f")
    editable_params["Time_to_Infection"] = st.number_input("Infection Start Week", value=editable_params["Time_to_Infection"], step=1, min_value=0)
    editable_params["Vaccine_Efficacy"] = st.slider("Vaccine Efficacy", min_value=0.0, max_value=1.0, value=editable_params["Vaccine_Efficacy"], step=0.05)

with col2:
    editable_params["Desired_Order"] = st.number_input("Vaccine Doses Ordered", value=editable_params["Desired_Order"], step=10000, min_value=0)
    editable_params["Time_to_Vaccination"] = st.number_input("Vaccination Start Week", value=editable_params["Time_to_Vaccination"], step=1, min_value=0)
    editable_params["Max_Vaccine_Dispensed"] = st.number_input("Max Wkly Dispensing", value=editable_params["Max_Vaccine_Dispensed"], step=1000, min_value=0)
    # **** NEW: Acceptance Rate ****
    editable_params["Vaccine_Acceptance_Rate"] = st.slider("Vaccine Acceptance Rate", min_value=0.0, max_value=1.0, value=editable_params["Vaccine_Acceptance_Rate"], step=0.05)


# Add expander for more parameters
with st.sidebar.expander("Advanced Parameters"):
     editable_params["ED"] = st.number_input("Incubation Duration (ED, wks)", value=editable_params["ED"], step=0.1, format="%.1f")
     editable_params["ID"] = st.number_input("Infection Duration (ID, wks)", value=editable_params["ID"], step=0.1, format="%.1f")
     editable_params["Time_to_Order"] = st.number_input("Order Placement Week", value=editable_params["Time_to_Order"], step=1, min_value=0)
     editable_params["Production_to_Shipping_Delay"] = st.number_input("Prod./Ship Delay (wks)", value=editable_params["Production_to_Shipping_Delay"], step=1, min_value=0)
     editable_params["Shipping_Duration"] = st.number_input("Shipping Duration (wks)", value=editable_params["Shipping_Duration"], step=1, min_value=0)
     editable_params["Vaccine_Multiplier"] = st.slider("Deployment Multiplier", min_value=0.1, max_value=1.0, value=editable_params["Vaccine_Multiplier"], step=0.05)
     editable_params["Per_Vaccine_Cost"] = st.number_input("Cost per Dose (€)", value=editable_params["Per_Vaccine_Cost"], step=0.5, format="%.2f")

# Update button in sidebar
if st.sidebar.button("Update Parameters", key="update_params_button"):
    # Basic validation
    if editable_params["ED"] <= 0 or editable_params["ID"] <= 0:
        st.sidebar.error("Durations (ED, ID) must be positive.")
    else:
        st.session_state.params = editable_params
        st.session_state.results = None # Clear results when params change
        st.session_state.metrics = None
        st.rerun() # Rerun the script to reflect changes

st.sidebar.divider()
st.sidebar.subheader("Run Simulation")
run_button = st.sidebar.button("Run Simulation", key="run_sim_button")

st.sidebar.divider()
st.sidebar.subheader("Scenario Management")
scenario_name = st.sidebar.text_input("Save current run as scenario:", key="scenario_name_input")
save_scenario_button = st.sidebar.button("Save Scenario", key="save_scenario_button")

# --- Main Area ---

# Tabs for different views
tab_main, tab_scenarios, tab_sensitivity, tab_adoption_viz, tab_advanced = st.tabs([
    "📊 Main Simulation", "↔️ Scenario Comparison", "🔬 Sensitivity Analysis", "🧑‍🤝‍🧑 Adoption Viz", "💡 Advanced Features"
])

with tab_main:
    st.header("Current Simulation Run")
    st.subheader("Parameters")
    # Display current parameters in a more readable format
    param_df = pd.DataFrame(st.session_state.params.items(), columns=['Parameter', 'Value'])
    st.dataframe(param_df)

    if run_button:
        try:
            with st.spinner("Running Simulation..."):
                # Create and run simulation with current parameters
                simulation = VaccineSimulation(st.session_state.params)
                results_df = simulation.run()
                metrics = calculate_summary_metrics(results_df, st.session_state.params)

            # Store results in session state
            st.session_state.results = results_df
            st.session_state.metrics = metrics
            st.success("Simulation Complete!")

            if scenario_name: # Auto-save if name provided when running
                st.session_state.scenario_results[scenario_name] = {'params': copy.deepcopy(st.session_state.params), 'results': results_df, 'metrics': metrics}
                st.sidebar.success(f"Scenario '{scenario_name}' saved.")

        except ValueError as e:
             st.error(f"Simulation Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during simulation: {e}")
            st.exception(e) # Show full traceback for debugging


    if save_scenario_button and scenario_name and st.session_state.results is not None:
         st.session_state.scenario_results[scenario_name] = {'params': copy.deepcopy(st.session_state.params), 'results': st.session_state.results, 'metrics': st.session_state.metrics}
         st.sidebar.success(f"Scenario '{scenario_name}' saved.")
    elif save_scenario_button and not scenario_name:
         st.sidebar.warning("Please enter a name for the scenario.")
    elif save_scenario_button and st.session_state.results is None:
         st.sidebar.warning("Please run a simulation before saving.")


    # Display results if available
    if st.session_state.results is not None:
        st.subheader("Summary Metrics")
        if st.session_state.metrics:
            # Display metrics in columns for better layout
            m = st.session_state.metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Attack Rate", m.get("Attack Rate (%)", "N/A")+" %")
            col_m1.metric("Peak Infected", m.get("Peak Infected Count", "N/A"))
            col_m1.metric("Peak Week", m.get("Peak Infection Week", "N/A"))

            col_m2.metric("Vaccines Used", m.get("Vaccines Used (Cumulative)", "N/A"))
            col_m2.metric("Vaccines Produced", m.get("Total Vaccine Produced (Ordered)", "N/A"))
            col_m2.metric("Vaccine Yield", m.get("Vaccine Yield (%)", "N/A")+" %")

            col_m3.metric("Total Cost", "€ " + m.get("Total Vaccine Cost (Euros)", "N/A").replace('€ ',''))
        else:
            st.warning("Metrics could not be calculated.")

        # --- Plots ---
        st.subheader("Charts")
        results_df = st.session_state.results

        # Plot SEIRV compartments
        st.write("SEIRV Dynamics Over Time")
        seirv_df = results_df[['Week', 'S', 'E', 'I', 'R', 'V']]
        seirv_melted = seirv_df.melt(id_vars=['Week'], var_name='Compartment', value_name='Count')
        fig_seirv = px.line(seirv_melted, x='Week', y='Count', color='Compartment',
                            title="Population Compartments Over Time",
                            labels={'Count': 'Number of Individuals'})
        fig_seirv.update_layout(yaxis_title="Population Count")
        st.plotly_chart(fig_seirv, use_container_width=True)

        # Plot Supply Chain Data
        st.write("Vaccine Supply Chain Dynamics")
        # Ensure columns exist before plotting
        plot_cols = ['Week']
        if 'Arrived_Stock' in results_df.columns: plot_cols.append('Arrived_Stock')
        if 'Vaccines_Used_Week' in results_df.columns: plot_cols.append('Vaccines_Used_Week')

        if len(plot_cols) > 1:
            supply_df = results_df[plot_cols]
            supply_melted = supply_df.melt(id_vars=['Week'], var_name='Metric', value_name='Count')
            fig_supply = px.line(supply_melted, x='Week', y='Count', color='Metric',
                                title="Vaccine Stock and Usage Over Time",
                                labels={'Count': 'Number of Doses'})
            fig_supply.update_layout(yaxis_title="Dose Count")
            st.plotly_chart(fig_supply, use_container_width=True)
        else:
            st.warning("Supply chain data columns missing for plotting.")


        # --- Raw Data ---
        with st.expander("Show Raw Simulation Data"):
            st.dataframe(results_df)
    elif not run_button:
         st.info("Click 'Run Simulation' in the sidebar after configuring parameters.")


with tab_scenarios:
    st.header("What-If Analysis / Scenario Comparison")
    if len(st.session_state.scenario_results) > 0:
        st.write("Saved Scenarios:")
        st.write(list(st.session_state.scenario_results.keys()))
        scenarios_to_compare = st.multiselect("Select scenarios to compare:",
                                              options=list(st.session_state.scenario_results.keys()),
                                              key="scenario_multiselect")

        if len(scenarios_to_compare) > 1:
            st.subheader("Metric Comparison")
            comparison_data = []
            param_diffs = {} # Track differing parameters

            # Identify differing parameters
            first_params = st.session_state.scenario_results[scenarios_to_compare[0]]['params']
            for name in scenarios_to_compare:
                 params = st.session_state.scenario_results[name]['params']
                 for key, val in params.items():
                     if key not in first_params or first_params[key] != val:
                         if key not in param_diffs: param_diffs[key] = set()
                         param_diffs[key].add(val)

            # Build comparison table
            for name in scenarios_to_compare:
                data = {'Scenario': name}
                metrics = st.session_state.scenario_results[name]['metrics']
                params = st.session_state.scenario_results[name]['params']
                # Add differing params to table
                for key in param_diffs:
                    data[key] = params.get(key, 'N/A')
                # Add key metrics
                data.update({k: metrics.get(k, 'N/A') for k in [
                    "Attack Rate (%)", "Peak Infected Count", "Vaccines Used (Cumulative)", "Vaccine Yield (%)", "Total Vaccine Cost (Euros)"
                ]})
                comparison_data.append(data)

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.set_index('Scenario'))

            # Add comparative plots
            st.subheader("Plot Comparison")
            plot_metric = st.selectbox("Select metric to plot:", ["I", "S", "V", "Vaccines_Used_Week", "Arrived_Stock"], key="compare_plot_metric")

            plot_data = []
            for name in scenarios_to_compare:
                 results_df = st.session_state.scenario_results[name]['results']
                 if plot_metric in results_df.columns:
                     temp_df = results_df[['Week', plot_metric]].copy()
                     temp_df['Scenario'] = name
                     plot_data.append(temp_df)

            if plot_data:
                combined_plot_df = pd.concat(plot_data)
                fig_compare = px.line(combined_plot_df, x='Week', y=plot_metric, color='Scenario', title=f"{plot_metric} Comparison Across Scenarios")
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.warning(f"Metric '{plot_metric}' not available in selected scenario results for plotting.")

    else:
        st.info("Save at least two scenarios using the sidebar to enable comparison.")

with tab_sensitivity:
    st.header("Sensitivity Analysis")
    st.markdown("""
    Explore how changing key input parameters simultaneously affects outcomes.
    Select parameter pairs and the outcome metric to visualize (using bubble plots like in the paper).
    """)

    # Define sensitivity scenarios based on paper's figures
    sensitivity_scenarios = {
        "Ro vs Efficacy -> Attack Rate": ("Ro", "Vaccine_Efficacy", "Attack Rate (%)"),
        "Ro vs Max Dispensing -> Attack Rate": ("Ro", "Max_Vaccine_Dispensed_Sensitivity", "Attack Rate (%)"),
        "Ro vs Max Dispensing -> Vaccine Yield": ("Ro", "Max_Vaccine_Dispensed_Sensitivity", "Vaccine Yield (%)"),
        "Ro vs Desired Order -> Attack Rate": ("Ro", "Desired_Order_Fraction", "Attack Rate (%)"),
        "Ro vs Desired Order -> Vaccine Yield": ("Ro", "Desired_Order_Fraction", "Vaccine Yield (%)"),
        "Ro vs Desired Order -> Cost & Attack Rate": ("Ro", "Desired_Order_Fraction", "Total Vaccine Cost (Euros)", "Attack Rate (%)"), # 4D plot
    }

    selected_sensitivity = st.selectbox("Select Sensitivity Scenario:", options=list(sensitivity_scenarios.keys()), key="sensitivity_select")

    num_runs = st.slider("Number of Sensitivity Runs:", min_value=10, max_value=500, value=100, step=10, key="sensitivity_runs")

    if st.button("Run Sensitivity Analysis", key="run_sensitivity_button"):
        st.session_state.sensitivity_results = None # Clear previous results
        results_list = []
        param_config = sensitivity_scenarios[selected_sensitivity]
        param_x, param_y = param_config[0], param_config[1]
        metric_color = param_config[2]
        metric_size = param_config[3] if len(param_config) > 3 else None


        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(num_runs):
            temp_params = copy.deepcopy(st.session_state.params) # Start with current settings
            # Sample parameters
            sampled_params = {}
            for p, r in SENSITIVITY_RANGES.items():
                sampled_params[p] = np.random.uniform(r[0], r[1])

            # Override the parameters being varied
            temp_params["Ro"] = sampled_params["Ro"]
            if param_x == "Vaccine_Efficacy" or param_y == "Vaccine_Efficacy":
                temp_params["Vaccine_Efficacy"] = sampled_params["Vaccine_Efficacy"]
            if param_x == "Max_Vaccine_Dispensed_Sensitivity" or param_y == "Max_Vaccine_Dispensed_Sensitivity":
                 temp_params["Max_Vaccine_Dispensed"] = int(sampled_params["Max_Vaccine_Dispensed_Sensitivity"])
            if param_x == "Desired_Order_Fraction" or param_y == "Desired_Order_Fraction":
                 temp_params["Desired_Order"] = int(sampled_params["Desired_Order_Fraction"] * temp_params["N"])

            try:
                # Run simulation
                sim = VaccineSimulation(temp_params)
                res_df = sim.run()
                metrics = calculate_summary_metrics(res_df, temp_params)

                # Store sampled params and results
                run_data = {
                    "Run": i,
                    param_x: temp_params[param_x] if 'Sensitivity' not in param_x else int(sampled_params[param_x]), # Adjust for sensitivity param names
                    param_y: temp_params[param_y] if 'Sensitivity' not in param_y else int(sampled_params[param_y]), # Adjust for sensitivity param names
                    # Extract numerical value for plotting
                    metric_color: float(str(metrics.get(metric_color, '0')).replace('%','').replace(',','').replace('€ ','')),
                }
                # Handle Desired_Order_Fraction display
                if param_x == "Desired_Order_Fraction": run_data[param_x] = sampled_params[param_x]
                if param_y == "Desired_Order_Fraction": run_data[param_y] = sampled_params[param_y]

                if metric_size:
                     run_data[metric_size] = float(str(metrics.get(metric_size, '0')).replace('%','').replace(',','').replace('€ ',''))

                results_list.append(run_data)

            except Exception as e:
                st.warning(f"Run {i+1} failed: {e}. Skipping.") # Log error but continue

            progress = (i + 1) / num_runs
            progress_bar.progress(progress)
            status_text.text(f"Running Sensitivity Analysis: {i+1}/{num_runs}")

        status_text.text("Sensitivity Analysis Complete!")
        st.session_state.sensitivity_results = pd.DataFrame(results_list)

    # Display sensitivity plot if results exist
    if st.session_state.sensitivity_results is not None and not st.session_state.sensitivity_results.empty:
        sens_df = st.session_state.sensitivity_results
        param_config = sensitivity_scenarios[selected_sensitivity]
        param_x, param_y = param_config[0], param_config[1]
        metric_color = param_config[2]
        metric_size = param_config[3] if len(param_config) > 3 else None

        # Use specific sensitivity names for labels if needed
        label_x = param_x.replace('_Sensitivity','').replace('_Fraction','')
        label_y = param_y.replace('_Sensitivity','').replace('_Fraction','')

        plot_title = f"Sensitivity: {label_x} vs {label_y}"
        color_label = metric_color.replace('(%)','').strip()
        size_label = metric_size.replace('(%)','').strip() if metric_size else None

        hover_data = [param_x, param_y, metric_color]
        if metric_size: hover_data.append(metric_size)

        try:
            fig_sens = px.scatter(
                sens_df,
                x=param_x,
                y=param_y,
                color=metric_color,
                size=metric_size if metric_size else None, # Only add size if specified
                color_continuous_scale=px.colors.sequential.Viridis_r, # Reversed Viridis (like paper) or try 'Plasma' 'RdYlBu_r' etc.
                hover_name="Run",
                hover_data=hover_data,
                title=plot_title,
                labels={param_x: label_x, param_y: label_y, metric_color: color_label, metric_size: size_label}
            )
            # Adjust bubble size if size parameter is used
            if metric_size and sens_df[metric_size].max() > 0:
                 fig_sens.update_traces(marker=dict(sizemin=4, sizeref=2.*sens_df[metric_size].max()/(40.**2))) # Adjust sizeref based on data range

            st.plotly_chart(fig_sens, use_container_width=True)
            with st.expander("Show Sensitivity Data"):
                st.dataframe(sens_df)
        except Exception as e:
             st.error(f"Error generating sensitivity plot: {e}")
             st.dataframe(sens_df) # Show data even if plot fails
    elif not st.session_state.sensitivity_results:
         st.info("Click 'Run Sensitivity Analysis' to generate results.")

with tab_adoption_viz:
    st.header("Illustrative Vaccine Adoption Visualization")
    st.markdown("""
    This is a simplified visual concept showing how a population (represented by 100 dots, each = 10k people) might adopt vaccination over time based *only* on the **Vaccine Acceptance Rate** set in the sidebar.
    It **does not** show disease spread or vaccine availability impact. The *actual epidemiological impact* of the acceptance rate is simulated in the main SEIR model (see Main Simulation tab).
    """)

    acceptance_rate_viz = st.session_state.params.get("Vaccine_Acceptance_Rate", 0.7)
    st.write(f"Current Acceptance Rate (for illustration): {acceptance_rate_viz*100:.0f}%")

    num_dots = 100
    sim_duration_viz = 20 # Simulate 20 weeks of adoption process

    # Initialize dot positions and states
    if 'dot_positions' not in st.session_state:
        st.session_state.dot_positions = np.random.rand(num_dots, 2)
        st.session_state.dot_states = np.zeros(num_dots) # 0: Unadopted, 1: Adopted

    if st.button("Run Adoption Animation", key="run_adoption_anim"):
        st.session_state.dot_states = np.zeros(num_dots) # Reset states
        fig_placeholder = st.empty()

        for week_viz in range(sim_duration_viz):
            # Determine how many new adoptions this week
            unadopted_indices = np.where(st.session_state.dot_states == 0)[0]
            num_unadopted = len(unadopted_indices)
            if num_unadopted == 0: break # Stop if everyone adopted

            # Simple model: constant proportion of remaining susceptibles adopt each week until target reached
            # Target number to adopt = acceptance_rate * num_dots
            # Num already adopted = num_dots - num_unadopted
            # Additional needed = max(0, int(acceptance_rate * num_dots) - (num_dots - num_unadopted))
            # Let's make it simpler: X% of *currently* unadopted become adopted this week, capped by overall rate.
            target_adoptions_total = int(acceptance_rate_viz * num_dots)
            current_adoptions = int(st.session_state.dot_states.sum())
            max_new_adoptions = target_adoptions_total - current_adoptions

            # Assume fixed fraction adopts per week (e.g. 10% of remaining susceptible)
            adopters_this_week = min(int(num_unadopted * 0.15), max_new_adoptions) # Example: 15% adopt rate per week
            adopters_this_week = max(0, adopters_this_week) # Ensure non-negative

            if adopters_this_week > 0 and len(unadopted_indices) >= adopters_this_week:
                chosen_indices = np.random.choice(unadopted_indices, adopters_this_week, replace=False)
                st.session_state.dot_states[chosen_indices] = 1 # Change state to adopted

            # Create plot data
            plot_df = pd.DataFrame({
                'x': st.session_state.dot_positions[:, 0],
                'y': st.session_state.dot_positions[:, 1],
                'status': ['Adopted' if s == 1 else 'Not Yet Adopted' for s in st.session_state.dot_states]
            })

            # Update plot
            fig_adopt = px.scatter(plot_df, x='x', y='y', color='status',
                                 title=f"Adoption Visualization (Week {week_viz+1}) - Concept Only",
                                 color_discrete_map={'Adopted': 'green', 'Not Yet Adopted': 'blue'},
                                 labels={'status': 'Vaccination Status'})
            fig_adopt.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            fig_placeholder.plotly_chart(fig_adopt, use_container_width=True)
            time.sleep(0.2) # Slow down animation

        st.success("Adoption animation complete.")

    # Static plot showing initial state if not run
    elif 'dot_states' in st.session_state :
         plot_df = pd.DataFrame({
                'x': st.session_state.dot_positions[:, 0],
                'y': st.session_state.dot_positions[:, 1],
                'status': ['Adopted' if s == 1 else 'Not Yet Adopted' for s in st.session_state.dot_states]
            })
         fig_adopt = px.scatter(plot_df, x='x', y='y', color='status',
                                 title=f"Adoption Visualization (Current State) - Concept Only",
                                 color_discrete_map={'Adopted': 'green', 'Not Yet Adopted': 'blue'},
                                 labels={'status': 'Vaccination Status'})
         fig_adopt.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
         st.plotly_chart(fig_adopt, use_container_width=True)


with tab_advanced:
    st.header("Advanced Features & Model Notes")
    with st.expander("Automated Web Search (Placeholder)"):
        search_query = st.text_input("Search the web:")
        if st.button("Search", key="web_search_button"):
            st.info("Web search functionality not yet implemented.")
            # Placeholder for web search tool code

    with st.expander("Geospatial Reasoning (Placeholder)"):
        st.info("Geospatial reasoning functionality not yet implemented. Requires specific requirements (e.g., map data, regional parameters).")

    with st.expander("Bias Check & Model Assumptions"):
        st.info("Bias check functionality requires further definition. Below are key model assumptions:")
        st.markdown("""
        *   **Homogeneous Mixing:** Assumes uniform interaction probability across the population. Ignores spatial/social structure.
        *   **Uniform Parameters:** Assumes constant parameters (Ro, Efficacy, Durations, Acceptance) for everyone. No age/risk stratification.
        *   **Deterministic Model:** Ignores random chance in transitions (uses average rates).
        *   **Perfect Parameter Knowledge:** Assumes input parameters are accurate.
        *   **Simplified Supply Chain:** Basic delays, no complex logistics/wastage modeling (e.g., cold chain).
        *   **Static Behavior:** Assumes population behavior (contact rates, vaccine acceptance) doesn't change dynamically *during* the simulation in response to the outbreak, unless explicitly modeled (like the acceptance rate parameter itself).
        *   **Vaccine Effect:** Models only reduced susceptibility (`V` compartment). Does not model reduced infectiousness or severity if vaccinated person still gets infected.
        """)
