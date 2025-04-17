import numpy as np
import pandas as pd
from parameters import calculate_initial_state

class VaccineSimulation:
    def __init__(self, params):
        self.params = params
        self.initial_state = calculate_initial_state(params)
        self.results = None

    def run(self):
        # Initialize state variables
        N = self.params['N']
        S, E, I, R, V = [self.initial_state['S']], [self.initial_state['E']], [self.initial_state['I']], [self.initial_state['R']], [self.initial_state['V']]

        # Initialize Supply Chain state
        orders_placed = 0
        order_fulfillment_schedule = {} # week -> quantity_ready
        vaccines_shipped_schedule = {} # week -> quantity_arriving
        # in_production_shipping = [0] * self.params['Simulation_Duration'] # Less critical to track weekly for this output
        arrived_stock = [0] * (self.params['Simulation_Duration'] + 1) # Stock at START of week
        used_total = [0] * (self.params['Simulation_Duration'] + 1)
        produced_total = 0

        # Timing parameters
        time_to_order = self.params['Time_to_Order']
        shipping_start_week = time_to_order + self.params['Production_to_Shipping_Delay']
        shipping_duration = self.params['Shipping_Duration']
        time_to_vaccination = self.params['Time_to_Vaccination']
        time_to_infection = self.params['Time_to_Infection']
        max_dispensed = self.params['Max_Vaccine_Dispensed']
        deploy_multiplier = self.params['Vaccine_Multiplier'] # Logistic efficiency
        efficacy = self.params['Vaccine_Efficacy']           # Biological effectiveness
        acceptance_rate = self.params['Vaccine_Acceptance_Rate'] # Behavioral willingness
        desired_order = self.params['Desired_Order']

        # SEIR parameters
        Ro = self.params['Ro']
        Ed = self.params['ED'] # Incubation duration
        Id = self.params['ID'] # Infection duration
        if Id <= 0 or Ed <= 0:
             raise ValueError("Infection and Incubation durations must be positive.")
        beta = Ro / Id if Id > 0 else 0
        alpha = 1.0 / Ed if Ed > 0 else float('inf') # Rate E -> I
        gamma = 1.0 / Id if Id > 0 else float('inf') # Rate I -> R

        # --- Simulation Loop ---
        for week in range(self.params['Simulation_Duration']):
            current_S = S[-1]
            current_E = E[-1]
            current_I = I[-1]
            current_R = R[-1]
            current_V = V[-1]
            current_arrived_stock = arrived_stock[week]
            current_used_total = used_total[week]

            # 1. PHA Ordering
            new_orders_this_week = 0
            if week == time_to_order and desired_order > 0:
                new_orders_this_week = desired_order
                orders_placed += new_orders_this_week
                # Schedule when production output is ready to ship
                order_fulfillment_schedule[shipping_start_week] = order_fulfillment_schedule.get(shipping_start_week, 0) + new_orders_this_week
                produced_total += new_orders_this_week # Assume order placed = commitment to produce

            # 2. Production Finishes / Ready to Ship
            vaccines_ready_to_ship = order_fulfillment_schedule.pop(week, 0)

            # 3. Shipping Starts -> Schedule Arrival
            shipped_this_week = 0
            if vaccines_ready_to_ship > 0:
                shipped_this_week = vaccines_ready_to_ship # Assume all ready ships immediately
                arrival_week = week + shipping_duration
                if arrival_week < self.params['Simulation_Duration']: # Avoid scheduling past simulation end
                     vaccines_shipped_schedule[arrival_week] = vaccines_shipped_schedule.get(arrival_week, 0) + shipped_this_week
                # Track stock in transit (for potential detailed view later)

            # 4. Vaccine Arrival
            arrived_this_week = vaccines_shipped_schedule.pop(week, 0)
            next_arrived_stock = current_arrived_stock + arrived_this_week

            # 5. SEIR Dynamics
            # Prevent division by zero if N=0, although unlikely
            effective_N = current_S + current_E + current_I + current_R + current_V
            if week >= time_to_infection and effective_N > 0 and current_I > 0:
                 force_of_infection = beta * current_I / effective_N
            else:
                 force_of_infection = 0

            new_exposed = force_of_infection * current_S #* DTime (DTime=1)
            # Clamp new_exposed to available susceptibles
            new_exposed = min(new_exposed, current_S)

            new_infected = alpha * current_E #* DTime (DTime=1)
            # Clamp new_infected to available exposed
            new_infected = min(new_infected, current_E)

            new_recovered = gamma * current_I #* DTime (DTime=1)
             # Clamp new_recovered to available infected
            new_recovered = min(new_recovered, current_I)

            # --- 6. Vaccination (UPDATED with Acceptance Rate) ---
            actual_vaccinated_used = 0
            actual_vaccinated_effective = 0
            # Conditions: Week is right, stock exists, susceptibles exist
            if week >= time_to_vaccination and next_arrived_stock > 0 and current_S > 0:

                # Max possible doses based on supply and logistics/capacity
                supply_capacity = min(next_arrived_stock, max_dispensed * deploy_multiplier) #* DTime (DTime=1)

                # Max possible demand based on susceptible and willing population
                # Ensure demand doesn't exceed available susceptibles
                demand_willing = min(current_S * acceptance_rate, current_S) #* DTime (DTime=1)

                # Actual doses administered is limited by BOTH supply/capacity AND demand/willingness
                # Also ensure we don't try to vaccinate more people than are susceptible
                actual_vaccinated_used = min(supply_capacity, demand_willing, current_S)

                # Calculate effective vaccinations based on efficacy
                actual_vaccinated_effective = actual_vaccinated_used * efficacy

                # Update stock
                next_arrived_stock -= actual_vaccinated_used

            # Ensure effective vaccinations don't exceed available susceptibles *after* accounting for exposure this week
            # This prevents removing people from S who are already destined for E in this same time step
            actual_vaccinated_effective = min(actual_vaccinated_effective, current_S - new_exposed)
            # If effective vaccinations were capped by (S - new_exposed), should we reduce actual_vaccinated_used?
            # Let's assume the dose is considered 'used' even if the person was exposed nearly simultaneously.
            # This maintains consistency with `used_total`.

            # 7. Update Compartments
            next_S = current_S - new_exposed - actual_vaccinated_effective
            next_E = current_E + new_exposed - new_infected
            next_I = current_I + new_infected - new_recovered
            next_R = current_R + new_recovered
            next_V = current_V + actual_vaccinated_effective

            # Ensure non-negativity and conservation (approximate due to discrete steps)
            next_S, next_E, next_I, next_R, next_V = np.maximum(0, [next_S, next_E, next_I, next_R, next_V])
            # Optional: Rescale slightly if needed due to floating point errors, though better to check model logic

            # 8. Append results for next step
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            V.append(next_V)
            arrived_stock[week+1] = next_arrived_stock
            used_total[week+1] = current_used_total + actual_vaccinated_used

        # --- End Simulation Loop ---

        # Store results in a DataFrame
        weeks = list(range(self.params['Simulation_Duration'] + 1))
        self.results = pd.DataFrame({
            'Week': weeks,
            'S': S,
            'E': E,
            'I': I,
            'R': R,
            'V': V,
            'Arrived_Stock': arrived_stock,
            'Used_Total': used_total,
            # Add other tracked supply chain vars if needed
        })
        # Add derived metrics (e.g., weekly incidence) if desired
        # Approximation of new infections based on outflow from E
        self.results['New_Infections_Week'] = self.results['E'].diff().fillna(0).apply(lambda x: x if x>0 else 0) + self.results['E'] * alpha # Needs careful calc
        # Simpler approx: outflow from E
        e_outflow = pd.Series(E[1:]) * alpha # E[t]*alpha transition for E[t] -> I[t+1]
        self.results['New_Infections_Week'] = e_outflow.reindex(weeks, fill_value=0)

        self.results['Vaccines_Used_Week'] = self.results['Used_Total'].diff().fillna(0)

        return self.results
