import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import copy
import time
from scipy.spatial.distance import cdist # Efficient distance calculation

# Import your simulation modules
from parameters import get_default_params, SENSITIVITY_RANGES
from simulation_model import VaccineSimulation
from analysis import calculate_summary_metrics

st.set_page_config(layout="wide", page_title="Vaccine Sim")
st.title("💉 SEIR & Vaccine Supply Chain Simulation")
st.markdown("Integrates SEIR disease spread with vaccine logistics and behavioral adoption.")

# --- Session State Initialization ---
# (Keep previous initializations for params, results, metrics, scenarios, sensitivity)
if 'params' not in st.session_state:
    st.session_state.params = get_default_params()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'scenario_results' not in st.session_state:
    st.session_state.scenario_results = {}
if 'sensitivity_results' not in st.session_state:
    st.session_state.sensitivity_results = None
# ** NEW for adoption viz state **
if 'dot_positions' not in st.session_state:
    st.session_state.dot_positions = None
if 'dot_states' not in st.session_state:
    st.session_state.dot_states = None


# --- Sidebar (Parameter Configuration - keep as before) ---
st.sidebar.header("Simulation Configuration")
st.sidebar.subheader("Edit Core Parameters")
editable_params = copy.deepcopy(st.session_state.params)
# ... (Keep all parameter inputs from the previous version, including Vaccine_Acceptance_Rate)
col1, col2 = st.sidebar.columns(2)
with col1:
    editable_params["N"] = st.number_input("Population (N)", value=editable_params["N"], step=100000, min_value=1000, key="sim_N")
    editable_params["Ro"] = st.number_input("Ro", value=editable_params["Ro"], step=0.1, format="%.1f", key="sim_Ro")
    editable_params["Time_to_Infection"] = st.number_input("Infection Start Week", value=editable_params["Time_to_Infection"], step=1, min_value=0, key="sim_tti")
    editable_params["Vaccine_Efficacy"] = st.slider("Vaccine Efficacy", min_value=0.0, max_value=1.0, value=editable_params["Vaccine_Efficacy"], step=0.05, key="sim_veff")
with col2:
    editable_params["Desired_Order"] = st.number_input("Vaccine Doses Ordered", value=editable_params["Desired_Order"], step=10000, min_value=0, key="sim_order")
    editable_params["Time_to_Vaccination"] = st.number_input("Vaccination Start Week", value=editable_params["Time_to_Vaccination"], step=1, min_value=0, key="sim_ttv")
    editable_params["Max_Vaccine_Dispensed"] = st.number_input("Max Wkly Dispensing", value=editable_params["Max_Vaccine_Dispensed"], step=1000, min_value=0, key="sim_maxdisp")
    # **** Acceptance Rate is CRITICAL ****
    editable_params["Vaccine_Acceptance_Rate"] = st.slider("Vaccine Acceptance Rate", min_value=0.0, max_value=1.0, value=editable_params["Vaccine_Acceptance_Rate"], step=0.05, key="sim_acceptrate")

with st.sidebar.expander("Advanced Parameters"):
    # ... (Keep advanced parameter inputs)
    editable_params["ED"] = st.number_input("Incubation Duration (ED, wks)", value=editable_params["ED"], step=0.1, format="%.1f", key="sim_ed")
    editable_params["ID"] = st.number_input("Infection Duration (ID, wks)", value=editable_params["ID"], step=0.1, format="%.1f", key="sim_id")
    editable_params["Time_to_Order"] = st.number_input("Order Placement Week", value=editable_params["Time_to_Order"], step=1, min_value=0, key="sim_tto")
    editable_params["Production_to_Shipping_Delay"] = st.number_input("Prod./Ship Delay (wks)", value=editable_params["Production_to_Shipping_Delay"], step=1, min_value=0, key="sim_proddelay")
    editable_params["Shipping_Duration"] = st.number_input("Shipping Duration (wks)", value=editable_params["Shipping_Duration"], step=1, min_value=0, key="sim_shipdur")
    editable_params["Vaccine_Multiplier"] = st.slider("Deployment Multiplier", min_value=0.1, max_value=1.0, value=editable_params["Vaccine_Multiplier"], step=0.05, key="sim_depmult")
    editable_params["Per_Vaccine_Cost"] = st.number_input("Cost per Dose (€)", value=editable_params["Per_Vaccine_Cost"], step=0.5, format="%.2f", key="sim_cost")


if st.sidebar.button("Update Parameters", key="update_params_button"):
    if editable_params["ED"] <= 0 or editable_params["ID"] <= 0:
        st.sidebar.error("Durations (ED, ID) must be positive.")
    else:
        st.session_state.params = editable_params
        st.session_state.results = None
        st.session_state.metrics = None
        # Reset viz state if parameters change? Optional, maybe not needed.
        # st.session_state.dot_positions = None
        # st.session_state.dot_states = None
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Run Simulation")
run_button = st.sidebar.button("Run Simulation", key="run_sim_button")

st.sidebar.divider()
st.sidebar.subheader("Scenario Management")
scenario_name = st.sidebar.text_input("Save current run as scenario:", key="scenario_name_input")
save_scenario_button = st.sidebar.button("Save Scenario", key="save_scenario_button")


# --- Main Area ---
tab_main, tab_scenarios, tab_sensitivity, tab_adoption_viz, tab_advanced = st.tabs([
    "📊 Main Simulation", "↔️ Scenario Comparison", "🔬 Sensitivity Analysis", "🧑‍🤝‍🧑 Adoption Viz", "💡 Advanced Features"
])

# --- Tab Main (Simulation Results - Keep as before) ---
with tab_main:
    st.header("Current Simulation Run")
    # ... (Keep all the code for running simulation, showing params, metrics, plots, raw data)
    st.subheader("Parameters")
    param_df = pd.DataFrame(st.session_state.params.items(), columns=['Parameter', 'Value'])
    st.dataframe(param_df)

    if run_button:
        try:
            with st.spinner("Running Simulation..."):
                simulation = VaccineSimulation(st.session_state.params)
                results_df = simulation.run()
                metrics = calculate_summary_metrics(results_df, st.session_state.params)
            st.session_state.results = results_df
            st.session_state.metrics = metrics
            st.success("Simulation Complete!")
            if scenario_name:
                st.session_state.scenario_results[scenario_name] = {'params': copy.deepcopy(st.session_state.params), 'results': results_df, 'metrics': metrics}
                st.sidebar.success(f"Scenario '{scenario_name}' saved.")
        except ValueError as e:
             st.error(f"Simulation Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during simulation: {e}")
            st.exception(e)

    if save_scenario_button:
         # ... (Keep save scenario logic)
         if scenario_name and st.session_state.results is not None:
             st.session_state.scenario_results[scenario_name] = {'params': copy.deepcopy(st.session_state.params), 'results': st.session_state.results, 'metrics': st.session_state.metrics}
             st.sidebar.success(f"Scenario '{scenario_name}' saved.")
         elif not scenario_name:
             st.sidebar.warning("Please enter a name for the scenario.")
         elif st.session_state.results is None:
             st.sidebar.warning("Please run a simulation before saving.")

    if st.session_state.results is not None:
        st.subheader("Summary Metrics")
        if st.session_state.metrics:
            # ... (Keep metric display logic)
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

        st.subheader("Charts")
        results_df = st.session_state.results
        # ... (Keep SEIRV plot logic)
        st.write("SEIRV Dynamics Over Time")
        seirv_df = results_df[['Week', 'S', 'E', 'I', 'R', 'V']]
        seirv_melted = seirv_df.melt(id_vars=['Week'], var_name='Compartment', value_name='Count')
        fig_seirv = px.line(seirv_melted, x='Week', y='Count', color='Compartment', title="Population Compartments Over Time", labels={'Count': 'Number of Individuals'})
        fig_seirv.update_layout(yaxis_title="Population Count")
        st.plotly_chart(fig_seirv, use_container_width=True)

        # ... (Keep Supply Chain plot logic)
        st.write("Vaccine Supply Chain Dynamics")
        plot_cols = ['Week']
        if 'Arrived_Stock' in results_df.columns: plot_cols.append('Arrived_Stock')
        if 'Vaccines_Used_Week' in results_df.columns: plot_cols.append('Vaccines_Used_Week')
        if len(plot_cols) > 1:
            supply_df = results_df[plot_cols]
            supply_melted = supply_df.melt(id_vars=['Week'], var_name='Metric', value_name='Count')
            fig_supply = px.line(supply_melted, x='Week', y='Count', color='Metric', title="Vaccine Stock and Usage Over Time", labels={'Count': 'Number of Doses'})
            fig_supply.update_layout(yaxis_title="Dose Count")
            st.plotly_chart(fig_supply, use_container_width=True)
        else:
            st.warning("Supply chain data columns missing for plotting.")

        with st.expander("Show Raw Simulation Data"):
            st.dataframe(results_df)
    elif not run_button:
         st.info("Click 'Run Simulation' in the sidebar after configuring parameters.")


# --- Tab Scenario Comparison (Keep as before) ---
with tab_scenarios:
    st.header("What-If Analysis / Scenario Comparison")
    # ... (Keep all scenario comparison logic)
    if len(st.session_state.scenario_results) > 0:
        st.write("Saved Scenarios:")
        st.write(list(st.session_state.scenario_results.keys()))
        scenarios_to_compare = st.multiselect("Select scenarios to compare:",
                                              options=list(st.session_state.scenario_results.keys()),
                                              key="scenario_multiselect")
        if len(scenarios_to_compare) > 1:
            st.subheader("Metric Comparison")
            # ... (comparison table generation)
            comparison_data = []
            param_diffs = {}
            first_params = st.session_state.scenario_results[scenarios_to_compare[0]]['params']
            for name in scenarios_to_compare:
                 params = st.session_state.scenario_results[name]['params']
                 for key, val in params.items():
                     if key not in first_params or first_params[key] != val:
                         if key not in param_diffs: param_diffs[key] = set()
                         param_diffs[key].add(val)
            for name in scenarios_to_compare:
                data = {'Scenario': name}
                metrics = st.session_state.scenario_results[name]['metrics']
                params = st.session_state.scenario_results[name]['params']
                for key in param_diffs: data[key] = params.get(key, 'N/A')
                data.update({k: metrics.get(k, 'N/A') for k in ["Attack Rate (%)", "Peak Infected Count", "Vaccines Used (Cumulative)", "Vaccine Yield (%)", "Total Vaccine Cost (Euros)"]})
                comparison_data.append(data)
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.set_index('Scenario'))

            st.subheader("Plot Comparison")
            # ... (comparison plot generation)
            plot_metric = st.selectbox("Select metric to plot:", ["I", "S", "V", "Vaccines_Used_Week", "Arrived_Stock"], key="compare_plot_metric")
            plot_data = []
            for name in scenarios_to_compare:
                 results_df = st.session_state.scenario_results[name]['results']
                 if plot_metric in results_df.columns:
                     temp_df = results_df[['Week', plot_metric]].copy(); temp_df['Scenario'] = name; plot_data.append(temp_df)
            if plot_data:
                combined_plot_df = pd.concat(plot_data)
                fig_compare = px.line(combined_plot_df, x='Week', y=plot_metric, color='Scenario', title=f"{plot_metric} Comparison Across Scenarios")
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.warning(f"Metric '{plot_metric}' not available for plotting.")
    else:
        st.info("Save at least two scenarios using the sidebar to enable comparison.")


# --- Tab Sensitivity Analysis (Keep as before) ---
with tab_sensitivity:
    st.header("Sensitivity Analysis")
    # ... (Keep all sensitivity analysis logic - scenario selection, run button, results plotting)
    st.markdown("Explore how changing key input parameters simultaneously affects outcomes.")
    sensitivity_scenarios = {
        "Ro vs Efficacy -> Attack Rate": ("Ro", "Vaccine_Efficacy", "Attack Rate (%)"),
        "Ro vs Max Dispensing -> Attack Rate": ("Ro", "Max_Vaccine_Dispensed_Sensitivity", "Attack Rate (%)"),
        "Ro vs Max Dispensing -> Vaccine Yield": ("Ro", "Max_Vaccine_Dispensed_Sensitivity", "Vaccine Yield (%)"),
        "Ro vs Desired Order -> Attack Rate": ("Ro", "Desired_Order_Fraction", "Attack Rate (%)"),
        "Ro vs Desired Order -> Vaccine Yield": ("Ro", "Desired_Order_Fraction", "Vaccine Yield (%)"),
        "Ro vs Desired Order -> Cost & Attack Rate": ("Ro", "Desired_Order_Fraction", "Total Vaccine Cost (Euros)", "Attack Rate (%)"),
    }
    selected_sensitivity = st.selectbox("Select Sensitivity Scenario:", options=list(sensitivity_scenarios.keys()), key="sensitivity_select")
    num_runs = st.slider("Number of Sensitivity Runs:", min_value=10, max_value=500, value=100, step=10, key="sensitivity_runs")

    if st.button("Run Sensitivity Analysis", key="run_sensitivity_button"):
        # ... (Keep sensitivity run loop and result storage)
        st.session_state.sensitivity_results = None
        results_list = []; param_config = sensitivity_scenarios[selected_sensitivity]
        param_x, param_y, metric_color = param_config[0], param_config[1], param_config[2]
        metric_size = param_config[3] if len(param_config) > 3 else None
        progress_bar = st.progress(0); status_text = st.empty()
        for i in range(num_runs):
            temp_params = copy.deepcopy(st.session_state.params)
            sampled_params = {p: np.random.uniform(r[0], r[1]) for p, r in SENSITIVITY_RANGES.items()}
            temp_params["Ro"] = sampled_params["Ro"]
            if "Efficacy" in param_x or "Efficacy" in param_y: temp_params["Vaccine_Efficacy"] = sampled_params["Vaccine_Efficacy"]
            if "Dispensed" in param_x or "Dispensed" in param_y: temp_params["Max_Vaccine_Dispensed"] = int(sampled_params["Max_Vaccine_Dispensed_Sensitivity"])
            if "Order" in param_x or "Order" in param_y: temp_params["Desired_Order"] = int(sampled_params["Desired_Order_Fraction"] * temp_params["N"])
            try:
                sim = VaccineSimulation(temp_params); res_df = sim.run(); metrics = calculate_summary_metrics(res_df, temp_params)
                run_data = {"Run": i, param_x: temp_params[param_x] if 'Sensitivity' not in param_x else int(sampled_params[param_x]), param_y: temp_params[param_y] if 'Sensitivity' not in param_y else int(sampled_params[param_y])}
                run_data[metric_color] = float(str(metrics.get(metric_color, '0')).replace('%','').replace(',','').replace('€ ',''))
                if param_x == "Desired_Order_Fraction": run_data[param_x] = sampled_params[param_x]
                if param_y == "Desired_Order_Fraction": run_data[param_y] = sampled_params[param_y]
                if metric_size: run_data[metric_size] = float(str(metrics.get(metric_size, '0')).replace('%','').replace(',','').replace('€ ',''))
                results_list.append(run_data)
            except Exception as e: st.warning(f"Run {i+1} failed: {e}. Skipping.")
            progress = (i + 1) / num_runs; progress_bar.progress(progress); status_text.text(f"Running Sensitivity Analysis: {i+1}/{num_runs}")
        status_text.text("Sensitivity Analysis Complete!"); st.session_state.sensitivity_results = pd.DataFrame(results_list)

    if st.session_state.sensitivity_results is not None and not st.session_state.sensitivity_results.empty:
        # ... (Keep sensitivity plot generation)
        sens_df = st.session_state.sensitivity_results; param_config = sensitivity_scenarios[selected_sensitivity]
        param_x, param_y, metric_color = param_config[0], param_config[1], param_config[2]
        metric_size = param_config[3] if len(param_config) > 3 else None
        label_x = param_x.replace('_Sensitivity','').replace('_Fraction',''); label_y = param_y.replace('_Sensitivity','').replace('_Fraction','')
        plot_title = f"Sensitivity: {label_x} vs {label_y}"; color_label = metric_color.replace('(%)','').strip(); size_label = metric_size.replace('(%)','').strip() if metric_size else None
        hover_data = [param_x, param_y, metric_color];
        if metric_size: hover_data.append(metric_size)
        try:
            fig_sens = px.scatter(sens_df, x=param_x, y=param_y, color=metric_color, size=metric_size if metric_size else None, color_continuous_scale=px.colors.sequential.Viridis_r, hover_name="Run", hover_data=hover_data, title=plot_title, labels={param_x: label_x, param_y: label_y, metric_color: color_label, metric_size: size_label})
            if metric_size and sens_df[metric_size].max() > 0: fig_sens.update_traces(marker=dict(sizemin=4, sizeref=2.*sens_df[metric_size].max()/(40.**2)))
            st.plotly_chart(fig_sens, use_container_width=True)
            with st.expander("Show Sensitivity Data"): st.dataframe(sens_df)
        except Exception as e: st.error(f"Error generating sensitivity plot: {e}"); st.dataframe(sens_df)
    elif 'sensitivity_results' in st.session_state and st.session_state.sensitivity_results is None:
        st.info("Click 'Run Sensitivity Analysis' to generate results.")


# --- Tab Adoption Viz (UPDATED) ---
with tab_adoption_viz:
    st.header("Illustrative Vaccine Adoption Visualization (with Movement & Influence)")
    st.markdown("""
    This is a **conceptual animation** showing how social influence might affect vaccine adoption in a small group (dots).
    Dots move randomly. Unadopted (blue) dots are more likely to become adopted (green) if they are near already adopted dots.
    The **Overall Acceptance Rate** (set in the sidebar) acts as the *maximum target* percentage of dots that can become adopted in this visualization.
    **Note:** This viz is separate from the main SEIR simulation, which uses the acceptance rate differently (as a population-level willingness limit).
    """)

    # --- Visualization Parameters ---
    st.subheader("Visualization Parameters")
    col_v1, col_v2, col_v3 = st.columns(3)
    num_dots = col_v1.slider("Number of Dots (People x 10k)", 50, 200, 100, key="viz_numdots")
    influence_radius = col_v2.slider("Influence Radius", 0.01, 0.3, 0.1, step=0.01, key="viz_radius", help="How close dots need to be to influence each other.")
    movement_speed = col_v3.slider("Movement Speed", 0.0, 0.05, 0.01, step=0.005, key="viz_speed", help="How much dots move each step.")
    base_adoption_prob = st.slider("Base Adoption Probability / week", 0.0, 0.2, 0.02, step=0.01, key="viz_baseprob", help="Intrinsic chance of adoption without influence.")
    influence_strength = st.slider("Influence Strength / neighbor", 0.0, 0.3, 0.05, step=0.01, key="viz_strength", help="Added chance of adoption per nearby adopted neighbor.")
    sim_duration_viz = st.slider("Animation Duration (weeks)", 10, 100, 40, key="viz_duration")

    # Get the target acceptance rate from the main simulation parameters
    acceptance_rate_viz = st.session_state.params.get("Vaccine_Acceptance_Rate", 0.7)
    st.info(f"Target Max Adoption based on Sidebar Parameter: {acceptance_rate_viz*100:.0f}%")
    target_adoptions_total = int(acceptance_rate_viz * num_dots)

    # --- Animation Control ---
    if st.button("Run Adoption Animation", key="run_adoption_anim"):
        # Initialize or reset state
        st.session_state.dot_positions = np.random.rand(num_dots, 2) # Random x, y in [0, 1]
        st.session_state.dot_states = np.zeros(num_dots, dtype=int) # 0: Unadopted, 1: Adopted
        fig_placeholder = st.empty()
        progress_bar_viz = st.progress(0)
        status_text_viz = st.empty()

        for week_viz in range(sim_duration_viz):
            # 1. Movement
            noise = (np.random.rand(num_dots, 2) - 0.5) * 2 * movement_speed # Random vector
            st.session_state.dot_positions += noise
            # Boundary conditions (wrap around)
            st.session_state.dot_positions = st.session_state.dot_positions % 1.0

            # 2. Adoption Logic
            current_adopted_count = int(st.session_state.dot_states.sum())
            if current_adopted_count >= target_adoptions_total:
                 status_text_viz.text(f"Week {week_viz+1}/{sim_duration_viz} - Target adoption reached.")
                 # Continue loop for movement? Or break? Let's continue movement.
                 # break # Optionally stop early if target reached
            else:
                status_text_viz.text(f"Week {week_viz+1}/{sim_duration_viz} - Simulating adoption...")

            # Identify who *could* adopt
            unadopted_indices = np.where(st.session_state.dot_states == 0)[0]
            adopted_indices = np.where(st.session_state.dot_states == 1)[0]
            newly_adopted_this_step = []

            if len(unadopted_indices) > 0 and len(adopted_indices) >= 0: # Need potential adopters
                # Calculate all pairwise distances (efficiently)
                # Only need distances between unadopted and adopted points
                if len(adopted_indices) > 0:
                    distances = cdist(st.session_state.dot_positions[unadopted_indices],
                                      st.session_state.dot_positions[adopted_indices])
                    # Count neighbors within radius for each unadopted dot
                    neighbor_counts = np.sum(distances < influence_radius, axis=1)
                else:
                    # If no one is adopted yet, neighbor count is 0 for all
                    neighbor_counts = np.zeros(len(unadopted_indices))


                # Calculate adoption probability for each unadopted dot
                adoption_probs = base_adoption_prob + neighbor_counts * influence_strength
                adoption_probs = np.clip(adoption_probs, 0.0, 1.0) # Ensure probability is valid

                # Decide who adopts
                random_draws = np.random.rand(len(unadopted_indices))
                would_adopt_mask = random_draws < adoption_probs
                potential_new_adopters = unadopted_indices[would_adopt_mask]

                # Apply overall acceptance cap
                num_can_still_adopt = target_adoptions_total - current_adopted_count
                num_to_adopt_now = min(len(potential_new_adopters), num_can_still_adopt)

                if num_to_adopt_now > 0:
                    # Randomly choose from potential adopters if count exceeds cap
                    final_new_adopters = np.random.choice(potential_new_adopters, num_to_adopt_now, replace=False)
                    newly_adopted_this_step = final_new_adopters


            # Update states
            if len(newly_adopted_this_step) > 0:
                st.session_state.dot_states[newly_adopted_this_step] = 1

            # 3. Update Plot
            plot_df_viz = pd.DataFrame({
                'x': st.session_state.dot_positions[:, 0],
                'y': st.session_state.dot_positions[:, 1],
                'status': ['Adopted' if s == 1 else 'Not Yet Adopted' for s in st.session_state.dot_states]
            })

            fig_adopt = px.scatter(plot_df_viz, x='x', y='y', color='status',
                                 title=f"Adoption Dynamics - Week {week_viz+1}",
                                 color_discrete_map={'Adopted': 'mediumseagreen', 'Not Yet Adopted': 'cornflowerblue'}, # Nicer colors
                                 labels={'status': 'Vaccination Status'},
                                 range_x=[-0.05, 1.05], range_y=[-0.05, 1.05]) # Add slight margin
            fig_adopt.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     legend_title_text='Status')
            fig_adopt.update_traces(marker=dict(size=8, opacity=0.8)) # Adjust marker style
            fig_placeholder.plotly_chart(fig_adopt, use_container_width=True)

            progress_bar_viz.progress((week_viz + 1) / sim_duration_viz)
            time.sleep(0.1) # Animation speed control

        status_text_viz.text("Adoption animation complete.")
        st.success(f"Final Adoption: {int(st.session_state.dot_states.sum())}/{num_dots} ({int(st.session_state.dot_states.sum())/num_dots*100:.0f}%)")

    # Display initial state or last state if animation hasn't run/finished
    elif st.session_state.dot_positions is not None:
         plot_df_viz = pd.DataFrame({
                'x': st.session_state.dot_positions[:, 0],
                'y': st.session_state.dot_positions[:, 1],
                'status': ['Adopted' if s == 1 else 'Not Yet Adopted' for s in st.session_state.dot_states]
            })
         fig_adopt = px.scatter(plot_df_viz, x='x', y='y', color='status',
                                 title=f"Adoption Visualization (Current State) - Concept Only",
                                 color_discrete_map={'Adopted': 'mediumseagreen', 'Not Yet Adopted': 'cornflowerblue'},
                                 labels={'status': 'Vaccination Status'},
                                 range_x=[-0.05, 1.05], range_y=[-0.05, 1.05])
         fig_adopt.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  legend_title_text='Status')
         fig_adopt.update_traces(marker=dict(size=8, opacity=0.8))
         st.plotly_chart(fig_adopt, use_container_width=True)
    else:
        st.info("Click 'Run Adoption Animation' to start the visualization.")


# --- Tab Advanced Features (Keep as before) ---
with tab_advanced:
    st.header("Advanced Features & Model Notes")
    # ... (Keep expanders for Web Search, Geospatial, Bias Check/Assumptions)
    with st.expander("Automated Web Search (Placeholder)"):
        search_query = st.text_input("Search the web:", key="web_search_input")
        if st.button("Search", key="web_search_button"): st.info("Web search functionality not yet implemented.")
    with st.expander("Geospatial Reasoning (Placeholder)"):
        st.info("Geospatial reasoning functionality not yet implemented.")
    with st.expander("Bias Check & Model Assumptions"):
        st.info("Bias check functionality requires further definition. Below are key model assumptions:")
        st.markdown("""
        *   **Homogeneous Mixing:** Assumes uniform interaction probability (main simulation).
        *   **Uniform Parameters:** Assumes constant parameters for everyone (main simulation).
        *   **Deterministic Model:** Ignores random chance (main simulation).
        *   **Perfect Parameter Knowledge:** Assumes input parameters are accurate.
        *   **Simplified Supply Chain:** Basic delays, no complex logistics/wastage modeling.
        *   **Static Behavior:** Assumes population behavior doesn't change dynamically *during* the simulation (except for acceptance rate parameter).
        *   **Vaccine Effect:** Models only reduced susceptibility.
        *   **Adoption Viz Simplifications:** The visualization uses a simplified agent-based approach with random movement and proximity influence, separate from the compartmental dynamics of the main SEIR model.
        """)
