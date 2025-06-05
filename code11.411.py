import random
random.seed(42)
import math
import matplotlib.pyplot as plt
import datetime
import os
import json
import glob  # <-- Add this import
import ast
import pandas as pd
import uuid
from datetime import datetime

class FreezeException(Exception):
    pass

class NodeAnchor:
    """
    Represents a symbolic anchor injection for a node.
    Modes:
    - 'linear': bleed rate r fixed at draw-once
    - 'exponential': half-life h fixed at draw-once
    - 'stochastic': per-beat random drop
    """
    def __init__(self, mode, params):
        self.mode = mode
        self.params = params

    def bleed(self, t, efs):
        if self.mode == 'linear':
            r = self.params['r']
            return max(0.0, efs - r)
        elif self.mode == 'exponential':
            h = self.params['h']
            return efs * (2 ** (-1.0 / h))
        elif self.mode == 'stochastic':
            drops = self.params['drops']
            probs = self.params['probs']
            r = random.choices(drops, probs)[0]
            return max(0.0, efs - r)
        else:
            return efs

class NPC:
    """Represents an NPC agent with traits affecting simulation."""
    def __init__(self, name, affiliation, traits, sensitivity=1.0):
        self.name = name
        self.affiliation = affiliation
        self.traits = traits
        self.sensitivity = sensitivity

    def apply_micro_effects(self, state):
        if 'Engineer' in self.traits:
            state['repair'] *= 1.01
        if 'Zealot' in self.traits:
            state['drift'] *= 1.005
        return state

class Simulation:
    def __init__(self, initial_state, weights, alphas, beta, gamma, anchors, npcs=None, filter_factors=None, mode='live', factions=None):
        """
        Initializes the simulation engine with its starting state and parameters.

        Args:
            initial_state (dict): The starting values for all state variables (C, psi, S, R, P, CCI, AS, investment, nodes EFS, etc.).
            weights (dict): Coefficients (w_psi, w_OM, w_SB, w_CCI, w_AS, w_inv_repair, w_inv_panic) that scale the influence of various factors.
            alphas (dict): Alpha parameters (alpha_C, alpha_shock, alpha_order) primarily used in panic calculations.
            beta (dict): Beta parameters (beta1, beta2) influencing the positive impact of inputs on internal states (e.g., psi on CCI, shock on AS).
            gamma (dict): Gamma parameters (gamma1, gamma2) influencing the negative impact of inputs on internal states (e.g., drift on CCI, repair on AS).
            anchors (dict): Dictionary of NodeAnchor objects, representing symbolic anchor points with their own bleed modes.
            npcs (list, optional): A list of NPC objects that can apply micro-effects to inputs. Defaults to an empty list.
            filter_factors (dict, optional): Factors (per faction) that attenuate psi based on belief systems. Defaults to an empty dict.
            mode (str, optional): The operational mode of the simulation ('live' for shadow validation). Defaults to 'live'.
            factions (dict, optional): A dictionary of all faction data, used for applying faction-specific effects in the step function. Defaults to an empty dict.
        """
        self.state = initial_state.copy()  # The current state of the simulation (C, psi, S, R, P, etc.)
        self.weights = weights             # Weights used to adjust the influence of different factors
        self.alphas = alphas               # Alpha parameters for panic calculations (influence of changes)
        self.beta = beta                   # Beta parameters (influence of inputs on internal state)
        self.gamma = gamma                 # Gamma parameters (influence of inputs on internal state)
        self.anchors = anchors             # Anchor nodes in the system
        self.npcs = npcs or []             # Non-Player Characters in the simulation
        self.filter_factors = filter_factors or {} # Factors affecting belief filtering
        self.mode = mode                   # The current mode of the simulation
        self.beat = 0                      # Current simulation step
        self.history = []                  # History of the simulation state, logged per beat
        self.contradictions = []           # List to store detected contradictions or anomalies
        self.factions = factions or {}     # All faction data, for applying effects in step

    def update_flux(self, d, I_ritual, R_phi):
        """Calculates psi residual based on decay and ritual inputs."""
        psi = self.state['psi']
        return psi * (1 - d) + I_ritual * R_phi
    def apply_belief_filter(self, psi):
        """Reduces psi based on faction filter factors."""
        # attenuate psi by combined filter_factors
        total_filter = sum(self.filter_factors.values()) / max(1, len(self.filter_factors))
        return psi * (1 - total_filter)

    def update_CCI(self, psi, drift):
        """Updates belief coherence based on psi and drift."""
        cci = self.state['CCI']
        return cci + self.beta['beta1'] * psi - self.gamma['gamma1'] * drift
    def update_AS(self, S_shock, repair):
        """Increases disorder buffer from shocks minus repairs."""
        as_buf = self.state['AS']
        return as_buf + self.beta['beta2'] * S_shock - self.gamma['gamma2'] * repair
    def update_stability(self, S, drift, repair):
        """
        Calculates effective stability after shocks and repairs.
        It is affected by belief drift and increased by repair efforts,
        also accounting for the current anomaly stress.

        Args:
            S (float): The current stability (unitless).
            drift (float): The current belief drift (unitless).
            repair (float): The current repair capacity (unitless).

        Returns:
            float: The calculated new stability value (unitless).
        """
        as_buf = self.state['AS']  # Current Anomaly Stress
        # The amount of shock that stability couldn't buffer
        remaining_shock = max(0, drift - as_buf)
        # Update AS based on the buffered shock (this line was missing in your previous version)
        self.state['AS'] = max(0, as_buf - drift)
        # Total repair includes base repair and investment-related repair
        total_repair = repair + self.weights.get('w_inv_repair', 0) * self.state.get('investment', 0)

        # Calculate the new stability value
        return S - remaining_shock + total_repair


    def update_reserves(self, R, E, D):
        """Tracks net reserves after expenses and gains."""
        return max(0, R - E + D)
    def update_investment(self, I, inflow, outflow):
        """Alters investment pool with inflow/outflow."""
        return max(0, I + inflow - outflow)
    def update_convergence(self, C, psi_eff, S, E, R_max):
        """Aggregates symbolic alignment score across all signals."""
        term1 = self.weights['w_psi'] * (1 - math.exp(-psi_eff / self.state['psi_max']))
        term2 = self.weights['w_OM'] * (1 - S)
        term3 = self.weights['w_SB'] * (E / R_max)
        term4 = self.weights['w_CCI'] * (self.state['CCI'] / self.state['cci_max'])
        term5 = self.weights['w_AS'] * (self.state['AS'] / self.state['as_max'])
        return C + term1 + term2 + term3 + term4 + term5
    def update_panic(self, P, C_old, C_new, S_shock, O_action):
        """Quantifies system volatility from recent changes."""
        delta = (
            self.alphas['alpha_C'] * (C_new - C_old)
            + self.alphas['alpha_shock'] * S_shock
            - self.alphas['alpha_order'] * O_action
            - self.weights.get('w_inv_panic', 0) * self.state.get('investment', 0)
        )
        return min(10, max(0, P + delta))

    def evolve_filter_factors(self, events):
        # Example: increase a faction's filter_factor by 0.01 per anchor event
        for faction, count in events.items():
            self.filter_factors[faction] = min(1.0, self.filter_factors.get(faction, 0) + 0.01 * count)

    def step(self, inputs):
        """
        Advances the simulation by one beat (time step), updating all state variables.

        Args:
            inputs (dict): A dictionary of input values for the current beat.
        """
        self.beat += 1  # Increment the beat counter

        # Shadow snapshot: Create a copy of the current state for later validation
        # This allows checking for significant divergences after updates
        shadow = self.state.copy() if self.mode == 'live' else None

        # Store the old Convergence value before updating for Panic calculation
        C_old = self.state['C']

        # 1. NPC micro-effects: Apply trait-based modifications to inputs
        # (e.g., Engineers boost repair, Zealots increase drift)
        for npc in self.npcs:
            inputs = npc.apply_micro_effects(inputs)

        # 2. Update psi (Psionic Potential) based on decay and inputs
        self.state['psi'] = self.update_flux(inputs['d'], inputs['I_ritual'], inputs['R_phi'])

        # 3. Apply belief filter to psi, reducing it based on faction belief factors
        self.state['psi_eff'] = self.apply_belief_filter(self.state['psi'])

        # 4. Update Collective Cognition Index (CCI) based on filtered psi and belief drift
        self.state['CCI'] = self.update_CCI(self.state['psi_eff'], inputs['drift'])

        # 5. Update Anomaly Stress (AS) based on shock and repair
        self.state['AS'] = self.update_AS(inputs['S_shock'], inputs['repair'])

        # 6. Update Stability (S) based on drift, repair, and anomaly stress
        # Call update_stability and assign its return value to self.state['S']
        self.state['S'] = self.update_stability(self.state['S'], inputs['drift'], inputs['repair'])

        # 7. Update Reserves (R) based on expenditure and deposits
        self.state['R'] = self.update_reserves(self.state['R'], inputs['E'], inputs['D'])

        # 8. Update Investment (if any) based on inflow and outflow
        if 'investment' in self.state:
            self.state['investment'] = self.update_investment(
                self.state['investment'], inputs.get('inflow', 0), inputs.get('outflow', 0))

        # 9. Update Convergence (C) based on several factors, including effective psi, stability, expenditure, and max reserves
        C_new = self.update_convergence(C_old, self.state['psi_eff'], self.state['S'], inputs['E'], inputs.get('R_max', 1))
        self.state['C'] = C_new

        # 10. Update Panic (P) based on changes in convergence, shock, and order action
        self.state['P'] = self.update_panic(self.state['P'], C_old, C_new, inputs['S_shock'], inputs['O_action'])

        # 11. Apply faction-specific effects (psi multiplier and CCI adjustment)
        # This loop iterates through all factions and applies their modifiers to global state
        if hasattr(self, 'factions') and self.factions: # Ensure factions exist
            for faction_id, faction_data in self.factions.items():
                self.state['psi'] *= faction_data.get('psi_multiplier', 1.0) # Apply psi multiplier
                self.state['CCI'] += faction_data.get('cci_adjustment', 0.0) # Apply CCI adjustment

        # 12. Anchor bleed: Apply decay/change to specific node's Emotional Field Signature (EFS)
        for node, anchor in self.anchors.items():
            # Check if the node exists in self.state before trying to access it
            if node in self.state and 'EFS' in self.state[node]:
                self.state[node]['EFS'] = anchor.bleed(self.beat, self.state[node]['EFS'])
            else:
                # Handle cases where node or EFS might be missing (e.g., initialize it)
                print(f"Warning: Node '{node}' or its 'EFS' not found in state. Initializing EFS to 0.")
                self.state.setdefault(node, {})['EFS'] = 0 # Initialize EFS for missing nodes

        # 13. Evolve filter_factors if 'filter_events' are provided in inputs
        if 'filter_events' in inputs:
            self.evolve_filter_factors(inputs['filter_events'])

        # 14. Shadow validation: Check for significant divergences between current and shadow state
        if shadow:
            # Calculate absolute differences for key state variables
            divergences = {k: abs(self.state[k] - shadow.get(k, 0)) for k in ['C', 'psi', 'S', 'R']}
            # If any divergence is greater than 5, print a warning
            if any(v > 5 for v in divergences.values()):
                print(f"[Warning] Divergence at beat {self.beat}: {divergences}")

        # 15. Log the current state and inputs to history
        self.history.append({
            'beat': self.beat,
            'state': self.state.copy(),
            'inputs': inputs.copy()
        })

        # 16. Check for contradictions (assuming a method `check_contradictions` exists)
        # If you don't have a `check_contradictions` method, you can comment this out or define it.
        # self.check_contradictions() # Uncomment if you have this method

        return self.state # Return the updated state of the simulation


def plot_simulation(history):
    beats = [entry['beat'] for entry in history]
    C_vals = [entry['state']['C'] for entry in history]
    psi_vals = [entry['state']['psi'] for entry in history]
    S_vals = [entry['state']['S'] for entry in history]
    R_vals = [entry['state']['R'] for entry in history]
    P_vals = [entry['state']['P'] for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(beats, C_vals, label='C (Convergence)')
    plt.plot(beats, psi_vals, label='ψ (Flux)')
    plt.plot(beats, S_vals, label='S (Stability)')
    plt.plot(beats, R_vals, label='R (Reserves)')
    plt.plot(beats, P_vals, label='P (Panic)')

    plt.xlabel('Beat')
    plt.ylabel('Value')
    plt.title('Simulation Dynamics Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save with timestamp in sim_plots folder
    folder = "sim_plots"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"simulation_plot_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    return filename

def plot_faction_comparison(factions_data, initial_state, weights, alphas, beta, gamma, anchors, npcs, filter_factors, inputs, beats=10):
    # ...
    for faction_id, faction_info in factions_data.items():
        print(f"Debugging faction: {faction_id}")
        print(f"  beta keys: {beta.keys() if isinstance(beta, dict) else type(beta)}")
        print(f"  gamma keys: {gamma.keys() if isinstance(gamma, dict) else type(gamma)}")
        print(f"  initial_state keys: {initial_state.keys() if isinstance(initial_state, dict) else type(initial_state)}")
        print(f"  weights keys: {weights.keys() if isinstance(weights, dict) else type(weights)}")
        print(f"  alphas keys: {alphas.keys() if isinstance(alphas, dict) else type(alphas)}")
        # ... other prints if necessary
        sim = Simulation(
            initial_state=initial_state.copy(),
            weights=weights,
            alphas=alphas,
            beta=beta,
            gamma=gamma,
            anchors=anchors,
            npcs=npcs,
            filter_factors=filter_factors.copy(),
            factions={faction_id: faction_info}
        )
        # ...
        
        try:
            for _ in range(beats):
                sim.step(inputs)
        except FreezeException:
            pass
        C_vals = [entry['state']['C'] for entry in sim.history]
        plt.plot(range(1, len(C_vals)+1), C_vals, label=f"{faction_id}: {faction_info['name']}")
    plt.xlabel('Beat')
    plt.ylabel('C (Convergence)')
    plt.title('Convergence Comparison Across Factions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    folder = "sim_plots"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"faction_comparison_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    return filename  # <-- Add this line

def load_factions(filename="factions.json"):
    with open(filename) as f:
        data = json.load(f)
    return {f['id']: f for f in data['factions']}

def get_faction(faction_id, factions):
    return factions.get(faction_id, None)

def validate_goods(record, factions):
    faction = get_faction(record['producer'], factions)
    if not faction:
        record['reasons'].append('Unknown faction')
        return False

    # Apply multipliers
    record['psionic_signature'] *= faction['psi_multiplier']
    record['convergence_score'] += faction['cci_adjustment']

    # Use faction-specific thresholds
    thresholds = faction['validation_thresholds']
    if record['psionic_signature'] < thresholds['Psi_min']:
        record['reasons'].append('Ψ below faction-min')
    if record['convergence_score'] < thresholds['C_min']:
        record['reasons'].append('C below faction-min')
    if record['stability'] < thresholds['S_min']:
        record['reasons'].append('S below faction-min')
    if record['anomaly_score'] > thresholds['AS_max']:
        record['reasons'].append('AS above faction-max')

    return len(record['reasons']) == 0

def update_currency_supply(M, psi, faction):
    beta_psi = 1.0  # Example coefficient
    return M + beta_psi * (psi * faction['psi_multiplier']) * faction['investment_pref']

def main():
    # Clear previous simulation plots
    folder = "sim_plots"
    files = glob.glob(os.path.join(folder, "*"))
    for file in files:
        os.remove(file)

    factions = load_factions("factions.json")
    faction = factions['faction-01']

    initial_state = {
        'C': 0.1, 'psi': 11, 'S': 0.1, 'R': 200, 'P': 1,
        'CCI': 5, 'AS': 2, 'investment': 50,
        'psi_max': 50, 'cci_max': 20, 'as_max': 10,
        'Node-12': {'EFS': 29},
        'Node-51': {'EFS': 99},
        'Node-23': {'EFS': 18}
    }
    weights = {
        'w_psi': 0.9,
        'w_OM': 0.6,
        'w_SB': 0.01,
        'w_CCI': 0.01,      # <-- Add this line
        'w_AS': 0.02        # <-- (Optional: if you use term5 in update_convergence)
    }
    alphas = {'alpha_C': 1.0, 'alpha_shock': 1.5, 'alpha_order': 0.5}
    beta = {'beta1': 0.1, 'beta2': 0.2}
    gamma = {'gamma1': 0.05, 'gamma2': 0.1}
    anchors = {
        'Node-12': NodeAnchor('linear', {'r': 0.053}),
        'Node-51': NodeAnchor('exponential', {'h': 4.8}),
        'Node-23': NodeAnchor('stochastic', {'drops': [0.03, 0.05, 0.07], 'probs': [0.5, 0.3, 0.2]})
    }
    npcs = [NPC('Aida', 'Starwatch', ['Engineer']), NPC('Ignis', 'Voidwardens', ['Zealot'])]
    filter_factors = {'Starwatch': 0.1, 'Voidwardens': 0.05}
    sim = Simulation(
    initial_state=initial_state,
    weights=weights,
    alphas=alphas,
    beta=beta,
    gamma=gamma,
    anchors=anchors,
    npcs=npcs,
    filter_factors=filter_factors
)

    input_sets = [
          
    {"d": 0.001, "I_ritual": 0.001, "R_phi": 0.001, "drift": 0.001, "repair": 0.001, "E": 0.001, "D": 0.001, "R_max": 200.05, "S_shock": 0.001, "O_action": 0.001},
    {"d": 0.999, "I_ritual": 0.999, "R_phi": 0.999, "drift": 0.999, "repair": 0.999, "E": 0.999, "D": 0.999, "R_max": 249.95, "S_shock": 0.999, "O_action": 0.999},
    {"d": 0.500, "I_ritual": 0.500, "R_phi": 0.500, "drift": 0.500, "repair": 0.500, "E": 0.500, "D": 0.500, "R_max": 225.00, "S_shock": 0.500, "O_action": 0.500},
    {"d": 0.750, "I_ritual": 0.750, "R_phi": 0.750, "drift": 0.750, "repair": 0.750, "E": 0.750, "D": 0.750, "R_max": 237.50, "S_shock": 0.750, "O_action": 0.750},
    {"d": 0.250, "I_ritual": 0.250, "R_phi": 0.250, "drift": 0.250, "repair": 0.250, "E": 0.250, "D": 0.250, "R_max": 212.50, "S_shock": 0.250, "O_action": 0.250},
    {"d": 0.600, "I_ritual": 0.600, "R_phi": 0.600, "drift": 0.600, "repair": 0.600, "E": 0.600, "D": 0.600, "R_max": 230.00, "S_shock": 0.600, "O_action": 0.600},
    {"d": 0.400, "I_ritual": 0.400, "R_phi": 0.400, "drift": 0.400, "repair": 0.400, "E": 0.400, "D": 0.400, "R_max": 220.00, "S_shock": 0.400, "O_action": 0.400},
    {"d": 0.850, "I_ritual": 0.850, "R_phi": 0.850, "drift": 0.850, "repair": 0.850, "E": 0.850, "D": 0.850, "R_max": 242.50, "S_shock": 0.850, "O_action": 0.850},
    {"d": 0.150, "I_ritual": 0.150, "R_phi": 0.150, "drift": 0.150, "repair": 0.150, "E": 0.150, "D": 0.150, "R_max": 207.50, "S_shock": 0.150, "O_action": 0.150},
    {"d": 0.650, "I_ritual": 0.650, "R_phi": 0.650, "drift": 0.650, "repair": 0.650, "E": 0.650, "D": 0.650, "R_max": 232.50, "S_shock": 0.650, "O_action": 0.650},
    {"d": 0.350, "I_ritual": 0.350, "R_phi": 0.350, "drift": 0.350, "repair": 0.350, "E": 0.350, "D": 0.350, "R_max": 217.50, "S_shock": 0.350, "O_action": 0.350},
    {"d": 0.900, "I_ritual": 0.900, "R_phi": 0.900, "drift": 0.900, "repair": 0.900, "E": 0.900, "D": 0.900, "R_max": 245.00, "S_shock": 0.900, "O_action": 0.900},
    {"d": 0.100, "I_ritual": 0.100, "R_phi": 0.100, "drift": 0.100, "repair": 0.100, "E": 0.100, "D": 0.100, "R_max": 205.00, "S_shock": 0.100, "O_action": 0.100},
    {"d": 0.700, "I_ritual": 0.700, "R_phi": 0.700, "drift": 0.700, "repair": 0.700, "E": 0.700, "D": 0.700, "R_max": 235.00, "S_shock": 0.700, "O_action": 0.700},
    {"d": 0.300, "I_ritual": 0.300, "R_phi": 0.300, "drift": 0.300, "repair": 0.300, "E": 0.300, "D": 0.300, "R_max": 215.00, "S_shock": 0.300, "O_action": 0.300},
    {"d": 0.800, "I_ritual": 0.800, "R_phi": 0.800, "drift": 0.800, "repair": 0.800, "E": 0.800, "D": 0.800, "R_max": 240.00, "S_shock": 0.800, "O_action": 0.800},
    {"d": 0.200, "I_ritual": 0.200, "R_phi": 0.200, "drift": 0.200, "repair": 0.200, "E": 0.200, "D": 0.200, "R_max": 210.00, "S_shock": 0.200, "O_action": 0.200},
    {"d": 0.550, "I_ritual": 0.550, "R_phi": 0.550, "drift": 0.550, "repair": 0.550, "E": 0.550, "D": 0.550, "R_max": 227.50, "S_shock": 0.550, "O_action": 0.550},
    {"d": 0.450, "I_ritual": 0.450, "R_phi": 0.450, "drift": 0.450, "repair": 0.450, "E": 0.450, "D": 0.450, "R_max": 222.50, "S_shock": 0.450, "O_action": 0.450},
    {"d": 0.995, "I_ritual": 0.995, "R_phi": 0.995, "drift": 0.995, "repair": 0.995, "E": 0.995, "D": 0.995, "R_max": 249.75, "S_shock": 0.995, "O_action": 0.995},
]
    

    try:
        for inputs in input_sets:
            sim.step(inputs)
    except FreezeException as e:
        print(f"❄️ Simulation frozen: {e}")
    with open("full_log.txt", "a", encoding="utf-8") as f:
        for entry in sim.history:
            f.write(f"{entry}\n")

    plot_file = plot_simulation(sim.history)
    print(f"Plot saved as {plot_file}")

    # Only compare Governance Resonance and Iron Accord
    factions_to_compare = {
      k: v for k, v in factions.items() if k in ['faction-01', 'faction-02', 'faction-03', 'faction-04']

    }
    comparison_file = plot_faction_comparison(
        factions_to_compare,
        initial_state,
        weights,
        alphas,
        beta,          # Use the 'beta' dictionary
        gamma,         # Use the 'gamma' dictionary
        anchors,
        npcs,          # You're also missing 'npcs' here
        filter_factors, # And 'filter_factors'
        inputs=inputs,
        beats=10
    )
    print(f"Faction comparison plot saved as {comparison_file}")

    # Extract and save only the beats information to a new file
    input_file = "simulation_output.txt"
    output_file = "simulation_output_clean.txt"

    with open(input_file, "r") as infile, open(output_file, "a") as outfile:
        for line in infile:
            if "Beat" in line:
                outfile.write(line)

    print(f"Cleaned output saved to {output_file}")

    results = []

    for i, inputs in enumerate(input_sets, 1):
        # (Re)initialize your simulation state here if needed
        sim = Simulation(
    initial_state=initial_state,
    weights=weights,
    alphas=alphas,
    beta=beta,
    gamma=gamma,
    anchors=anchors,
    npcs=npcs,
    filter_factors=filter_factors
)
        try:
            sim.step(inputs)
            results.append((i, sim.state.copy()))
        except FreezeException as e:
            results.append((i, f"Frozen: {e}"))

    # Print or process results
    for i, result in results:
        print(f"Set {i}: {result}")

    # Save results to a file
    with open("simulation_batch_results.txt", "a", encoding="utf-8") as f:
        for i, result in results:
            f.write(f"Set {i}: {result}\n")

    print("Batch simulation results saved to simulation_batch_results.txt")

    # Use the first input set for comparison and final step
    inputs = input_sets[0]

    try:
        sim.step(inputs)
    except FreezeException as e:
        print(f"❄️ Simulation frozen during final step: {e}")

    # Save the final state to a file
    with open("final_state.txt", "a") as f:
        f.write(f"Final state after last inputs: {sim.state}\n")
        f.write(f"Contradictions: {sim.contradictions}\n")

    # Paste your full log as a string (replace ... with actual values)
    data_text = """
[
{'beat': 1, 'state': {'C': 0.7, 'psi': 10, 'S': 0.1, 'R': 200, 'P': 1.6, 'Node-12': {'EFS': 27.9}, 'Node-51': {'EFS': 5.5}, 'Node-23': {'EFS': 17.0}}, 'inputs': {'d': 0.001, 'I_ritual': 0.001, 'R_phi': 0.001, 'drift': 0.001, 'repair': 0.001, 'E': 0.001, 'D': 0.001, 'R_max': 200.05, 'S_shock': 0.001, 'O_action': 0.001}},
{'beat': 2, 'state': {'C': 1.2, 'psi': 1.0, 'S': 0.1, 'R': 200, 'P': 3.1, 'Node-12': {'EFS': 27.9}, 'Node-51': {'EFS': 5.5}, 'Node-23': {'EFS': 17.0}}, 'inputs': {'d': 0.999, 'I_ritual': 0.999, 'R_phi': 0.999, 'drift': 0.999, 'repair': 0.999, 'E': 0.999, 'D': 0.999, 'R_max': 249.95, 'S_shock': 0.999, 'O_action': 0.999}}
]
"""

    # Parse into Python list of dicts
    beats = ast.literal_eval(data_text)

    # Build a flat table
    records = []
    for entry in beats:
        rec = {'beat': entry['beat']}
        state = entry['state']
        # Add main state variables
        rec.update({k: state[k] for k in ['C', 'psi', 'S', 'R', 'P']})
        # Add each node's EFS
        for node in ['Node-12', 'Node-51', 'Node-23']:
            rec[node + '_EFS'] = state[node]['EFS']
        records.append(rec)

    df = pd.DataFrame(records).set_index('beat')
    print(df)

    # Plotting
    for col in ['C', 'psi', 'S', 'R', 'P', 'CCI', 'AS', 'Node-12_EFS', 'Node-51_EFS', 'Node-23_EFS']:
        if col in df.columns:
            plt.figure()
            df[col].plot(marker='o')
            plt.xlabel('Beat')
            plt.ylabel(col)
            plt.title(f'{col} over Beats')
            plt.show()
        



import os

def safe_write(filepath, content_lines, mode="a", encoding="utf-8", sync=True):
    try:
        abs_path = os.path.abspath(filepath)
        print(f"📁 Writing to: {abs_path} | Mode: {mode}")
        with open(filepath, mode, encoding=encoding) as f:
            for line in content_lines:
                f.write(line)
            f.flush()
            if sync:
                os.fsync(f.fileno())
        print(f"✅ Wrote {len(content_lines)} lines to {filepath}")
    except Exception as e:
        print(f"❌ Failed to write to {filepath}: {e}")



class Currency:
    """Defines a type of currency (e.g. gold, credits)."""
    def __init__(self, code: str, name: str):
        self.code = code
        self.name = name

class Transaction:
    """A single transaction entry on a wallet."""
    def __init__(self, wallet_id: str, currency: Currency, amount: float, tx_type: str, description: str = ''):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.wallet_id = wallet_id
        self.currency = currency
        self.amount = amount  # positive value
        self.type = tx_type   # 'credit' or 'debit'
        self.description = description

class Wallet:
    """Manages balances and history for one wallet."""
    def __init__(self, owner_id: str):
        self.owner_id = owner_id
        self.balances = {}  # currency code -> amount
        self.history = []

    def credit(self, currency: Currency, amount: float, description: str = '') -> Transaction:
        if amount <= 0:
            raise ValueError('Credit amount must be positive')
        prev = self.balances.get(currency.code, 0)
        self.balances[currency.code] = prev + amount
        tx = Transaction(self.owner_id, currency, amount, 'credit', description)
        self.history.append(tx)
        return tx

    def debit(self, currency: Currency, amount: float, description: str = '') -> Transaction:
        if amount <= 0:
            raise ValueError('Debit amount must be positive')
        prev = self.balances.get(currency.code, 0)
        if amount > prev:
            raise ValueError(f'Insufficient funds: have {prev}, need {amount}')
        self.balances[currency.code] = prev - amount
        tx = Transaction(self.owner_id, currency, amount, 'debit', description)
        self.history.append(tx)
        return tx

    def get_balance(self, currency: Currency) -> float:
        return self.balances.get(currency.code, 0)

    def get_history(self):
        # Return a copy to prevent external mutation
        return list(self.history)

class WalletService:
    """Simple in-memory wallet registry."""
    def __init__(self):
        self.wallets = {}

    def create_wallet(self, owner_id: str) -> Wallet:
        w = Wallet(owner_id)
        self.wallets[owner_id] = w
        return w

    def get_wallet(self, owner_id: str) -> Wallet:
        w = self.wallets.get(owner_id)
        if not w:
            raise ValueError(f'Wallet not found for {owner_id}')
        return w

    def list_wallets(self):
        return list(self.wallets.values())

# Example usage:
if __name__ == "__main__":
    main()
