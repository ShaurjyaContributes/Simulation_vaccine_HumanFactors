import copy

DEFAULT_PARAMS = {
    # Contextual & Environmental Parameters
    "N": 1_000_000, # Total Population
    "Initial_Infected": 1,
    "Ro": 1.6, # Basic Reproduction Number
    "ED": 0.3, # Average Incubation Duration (Weeks)
    "ID": 1.0, # Average Infection Duration (Weeks)
    "Time_to_Infection": 45, # Week flu season starts
    "Simulation_Duration": 78, # Weeks
    "DTime": 1, # Time Step (Weeks) - Keep as 1 for this model

    # Governance Parameters (PHA Agent Decisions & Rules)
    "Desired_Order": 600_000, # Doses
    "Time_to_Vaccination": 40, # Week deployment starts
    "Time_to_Order": 4, # Week order placed
    # "Vaccine Policy Flag": True, # Implicitly True if Desired_Order > 0
    # "HIT": 0.60, # Target, used to calculate Desired_Order often
    "Vaccine_Multiplier": 0.8, # Deployment efficiency constraint

    # Implementation Process Parameters (Supply Chain & Vaccination Activities)
    "Production_to_Shipping_Delay": 27, # Weeks from order to shipping start
    "Shipping_Duration": 2, # Weeks in transit
    # "Vaccine Deployment": 50_000, # Implicit in Max_Vaccine_Dispensed
    "Max_Vaccine_Dispensed": 50_000, # doses/week capacity
    "Vaccine_Efficacy": 0.55, # 55%

    # Cost Parameters
    "Per_Vaccine_Cost": 10.00, # Euros/dose (applied to produced/ordered amount)
}

def get_default_params():
    """Returns a deep copy of the default parameters."""
    return copy.deepcopy(DEFAULT_PARAMS)

def calculate_initial_state(params):
    """Calculates initial S, E, I, R, V based on N and Initial_Infected."""
    N = params['N']
    I0 = params['Initial_Infected']
    S0 = N - I0
    E0 = 0
    R0 = 0
    V0 = 0
    return {'S': S0, 'E': E0, 'I': I0, 'R': R0, 'V': V0}