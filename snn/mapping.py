import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import torch

G = nx.Graph()

class Neuron:
    def __init__(self, name, dorsal=False, ventral=False, excitatory=True, inhibitory=False):
        self.name = name
        self.dorsal = dorsal
        self.ventral = ventral
        self.excitatory = excitatory
        self.inhibitory = inhibitory
        self.membrane_potential = -70.0  # Resting potential in mV
        self.membrane_potential_history = []
        self.spike_times = []
        self.is_spiking = False
        self.position = None  # For circular layout

    def encounters_vesicle(self, x):
        """
        Increase the membrane potential by a specified amount.
        """
        if x=="GABA":
            self.membrane_potential = -0.25 # in mV
        elif x=="ACh":
            self.membrane_potential = 0.25 # in mV

    def update(self, time_step):
        """
        Update the neuron's state (e.g., check for spikes, reset potential).
        """
        if self.membrane_potential >= -55.0:  # Threshold for spiking
            self.spike_times.append(time_step)
            self.membrane_potential = -65.0  # Reset after spike
            print(f"{self.name} spiked at time {time_step}!")

    def __repr__(self):
        """
        String representation of the neuron.
        """
        return (f"{self.name}")
        # return (f"Neuron(name={self.name}, dorsal={self.dorsal}, ventral={self.ventral}, "
        #         f"excitatory={self.excitatory}, inhibitory={self.inhibitory}, "
        #         f"membrane_potential={self.membrane_potential:.2f} mV)")


# there are 75 motor neurons in C. elegans
MOTOR_NEURONS = {
    'DA': range(1, 10),
    'DB': range(1, 8),
    'DD': range(1, 7),  
    'VA': range(1, 12),  
    'VB': range(1, 12),  
    'VD': range(1, 14),  
    'AS': range(1, 12)   
}

# manually instantiate neurons
def instantiate_neurons():
    neurons = {}
    
    # Create dorsal motor neurons (DA1-9, DB1-7)
    for i in range(1, 10):
        name = f'DA{i}'
        neurons[name] = Neuron(name, dorsal=True, excitatory=True)
    for i in range(1, 8):
        name = f'DB{i}'
        neurons[name] = Neuron(name, dorsal=True, excitatory=True)
    
    # Create ventral motor neurons (VA1-12, VB1-11)
    for i in range(1, 13):
        name = f'VA{i}'
        neurons[name] = Neuron(name, ventral=True, excitatory=True)
    for i in range(1, 12):
        name = f'VB{i}'
        neurons[name] = Neuron(name, ventral=True, excitatory=True)
    
    # Create D-type motor neurons (DD1-6, VD1-13)
    for i in range(1, 7):  # DD1-6 only
        name = f'DD{i}'
        neurons[name] = Neuron(name, dorsal=True, inhibitory=True)
    for i in range(1, 14):
        name = f'VD{i}'
        neurons[name] = Neuron(name, ventral=True, inhibitory=True)
    
    # Create AS-type motor neurons (AS1-11)
    for i in range(1, 12):
        name = f'AS{i}'
        neurons[name] = Neuron(name, dorsal=True, excitatory=True)
    
    # Create circular layout for neurons
    num_neurons = len(neurons)
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, num_neurons, endpoint=False)
    
    # Assign positions to neurons
    for i, (name, neuron) in enumerate(neurons.items()):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        neuron.position = (x, y)
    
    return neurons

# Biological connections
# simplification
connections = []
# Generate list of all neuron names
all_neurons = []
for class_name, numbers in MOTOR_NEURONS.items():
    class_neurons = [f"{class_name}{num}" for num in numbers]
    all_neurons.extend(class_neurons)
    # Connect sequential neurons within same class
    for i in range(len(class_neurons)-1):
        connections.append((class_neurons[i], class_neurons[i+1]))

# Add connections to graph
G.add_edges_from(connections)


def map_to_color(value):
    """Map a value to a color between red and green"""
    # Ensure value is between 0 and 1
    normalized = (value - (-70)) / ((-50) - (-70))  # Map from -70..-50 to 0..1
    normalized = max(0, min(1, normalized))  # Clamp to 0..1
    
    # Interpolate between red (inhibitory) and green (excitatory)
    r = int(255 * (1 - normalized))
    g = int(255 * normalized)
    b = 0
    
    return f'rgb({r}, {g}, {b})'

def map_spikes_to_energy(spikes, threshold=0.8, decay_rate=0.1):
    """
    Map spike activity to worm energy levels with a threshold
    Args:
        spikes: numpy array of spike activity over time
        threshold: maximum energy depletion threshold (0-1)
        decay_rate: rate at which energy depletes per spike
    Returns:
        energy_levels: numpy array of energy levels over time
    """
    energy = 1.0  # Start with full energy
    energy_levels = []
    
    for time_step in spikes:
        # Sum spikes across neurons at this time step
        total_spikes = np.sum(time_step)
        # Deplete energy based on spike activity
        energy_depletion = total_spikes * decay_rate
        # Apply threshold
        energy = max(1 - threshold, energy - energy_depletion)
        energy_levels.append(energy)
    
    return np.array(energy_levels)

def membrane_to_neurotransmitter(membrane_potentials, neuron_types):
    """
    Convert membrane potentials to neurotransmitter release
    Args:
        membrane_potentials: numpy array of shape (time_steps, num_neurons)
        neuron_types: list of strings indicating neuron type ('GABA' or 'ACh')
    Returns:
        dict with neurotransmitter levels for each type
    """
    num_timesteps = len(membrane_potentials)
    gaba_release = np.zeros(num_timesteps)
    ach_release = np.zeros(num_timesteps)
    
    for t in range(num_timesteps):
        for n, neuron_type in enumerate(neuron_types):
            # Get membrane potential for this neuron at this timestep
            potential = membrane_potentials[t, n % membrane_potentials.shape[1]]  # Use modulo to wrap around if needed
            # Convert potential to scalar if it's an array
            if isinstance(potential, np.ndarray):
                potential = potential.mean()  # Take mean if it's an array
            # Normalize potential to 0-1 range for release probability
            release_prob = float((np.tanh(potential) + 1) / 2)  # Ensure it's a scalar
            
            if neuron_type == 'GABA':
                gaba_release[t] += release_prob  # Accumulate contributions
            else:  # ACh
                ach_release[t] += release_prob  # Accumulate contributions
    
    # Normalize the accumulated values
    if len(neuron_types) > 0:
        num_gaba = sum(1 for nt in neuron_types if nt == 'GABA')
        num_ach = len(neuron_types) - num_gaba
        if num_gaba > 0:
            gaba_release /= num_gaba
        if num_ach > 0:
            ach_release /= num_ach
    
    return {
        'GABA': gaba_release,
        'ACh': ach_release
    }

class WormStateVisualizer:
    def __init__(self):
        # Initialize the figure with subplots
        self.fig = make_subplots(rows=3, cols=1, 
                                subplot_titles=('Worm Energy Level', 
                                              'GABA Release', 
                                              'ACh Release'))
        self.fig.update_layout(height=900, title_text="Neural Activity and Worm State")
        
        # Initialize empty traces
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Energy'), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='GABA'), row=2, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='ACh'), row=3, col=1)
        
        # Initialize data storage
        self.energy_data = []
        self.gaba_data = []
        self.ach_data = []
        self.time_points = []
        self.current_time = 0
        
        # Show the initial empty figure
        self.fig.show()
    
    def update(self, spikes, membrane):
        """Update visualization with new neural activity data"""
        # Process the new data
        energy = map_spikes_to_energy(spikes)
        neuron_types = ['GABA' if i % 2 == 0 else 'ACh' for i in range(membrane.shape[-1])]
        neurotransmitters = membrane_to_neurotransmitter(membrane, neuron_types)
        
        # Append new data
        self.energy_data.extend(energy.tolist())
        self.gaba_data.extend(neurotransmitters['GABA'].tolist())
        self.ach_data.extend(neurotransmitters['ACh'].tolist())
        
        # Update time points
        new_times = list(range(self.current_time, self.current_time + len(energy)))
        self.time_points.extend(new_times)
        self.current_time = new_times[-1] + 1
        
        # Update the plots
        with self.fig.batch_update():
            self.fig.data[0].x = self.time_points
            self.fig.data[0].y = self.energy_data
            
            self.fig.data[1].x = self.time_points
            self.fig.data[1].y = self.gaba_data
            
            self.fig.data[2].x = self.time_points
            self.fig.data[2].y = self.ach_data

def process_activity(spikes, membrane):
    """Process neural activity data and return computed states"""
    energy = map_spikes_to_energy(spikes)
    neuron_types = ['GABA' if i % 2 == 0 else 'ACh' for i in range(membrane.shape[-1])]
    neurotransmitters = membrane_to_neurotransmitter(membrane, neuron_types)
    
    return {
        'energy': energy,
        'neurotransmitters': neurotransmitters
    }

# Load and process the latest neural activity
def visualize_in_subplots():
    try:
        spikes = np.load('spikes_latest.npy')
        membrane = np.load('membrane_latest.npy')
        
        # Define neuron types (alternating between GABA and ACh)
        neuron_types = ['GABA' if i % 2 == 0 else 'ACh' for i in range(membrane.shape[1])]
        
        # Calculate energy levels from spikes
        energy_levels = map_spikes_to_energy(spikes)
        
        # Calculate neurotransmitter release
        neurotransmitters = membrane_to_neurotransmitter(membrane, neuron_types)
        
        # Create visualization
        fig = make_subplots(
            rows=3, 
            cols=1,
            subplot_titles=('Worm Energy Level', 
                          'GABA Release', 
                          'ACh Release')
        )
        
        # Plot energy levels
        fig.add_trace(
            go.Scatter(y=energy_levels, mode='lines', name='Energy'),
            row=1, 
            col=1
        )
        
        # Plot GABA release
        fig.add_trace(
            go.Scatter(y=neurotransmitters['GABA'], mode='lines', name='GABA'),
            row=2, 
            col=1
        )
        
        # Plot ACh release
        fig.add_trace(
            go.Scatter(y=neurotransmitters['ACh'], mode='lines', name='ACh'),
            row=3, 
            col=1
        )
        
        fig.update_layout(height=900, title_text="Neural Activity and Worm State")
        fig.show()
        
        return energy_levels, neurotransmitters
        
    except FileNotFoundError:
        print("No neural activity data found. Run training first.")
        return None, None

# Set random seed for reproducibility
np.random.seed(42)

def create_circular_layout(neurons):
    """Create a circular layout for neurons"""
    num_neurons = len(neurons)
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, num_neurons, endpoint=False)
    
    # Create positions dictionary
    positions = {}
    for i, (name, neuron) in enumerate(neurons.items()):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        positions[name] = (x, y)
        neuron.position = (x, y)  # Store position in neuron object
    
    return positions

def create_network_visualization(neurons, membrane_record=None):
    """Create network visualization with the given neurons in a circle"""
    neuron_list = list(neurons.keys())
    
    def create_frame(frame_idx=None):
        # Create edges between neurons
        edge_x = []
        edge_y = []
        for i, name1 in enumerate(neuron_list):
            # Connect to next neuron in circle
            next_name = neuron_list[(i + 1) % len(neuron_list)]
            x0, y0 = neurons[name1].position
            x1, y1 = neurons[next_name].position
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Connect dorsal to ventral neurons
            if neurons[name1].dorsal:
                # Find closest ventral neuron
                for name2 in neuron_list:
                    if neurons[name2].ventral:
                        x2, y2 = neurons[name2].position
                        edge_x.extend([x0, x2, None])
                        edge_y.extend([y0, y2, None])
                        break
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='rgba(128, 128, 128, 0.3)'),
            hoverinfo='none',
            mode='lines'
        )

        # Get current membrane potentials
        if frame_idx is not None and membrane_record is not None:
            current_potentials = membrane_record[frame_idx][:len(neurons)]  # Take first N neurons
            current_potentials = -70 + 20 * (current_potentials + 1) / 2  # Scale to mV range
        else:
            current_potentials = np.array([neuron.membrane_potential for neuron in neurons.values()])

        # Create node trace
        node_trace = go.Scatter(
            x=[neuron.position[0] for neuron in neurons.values()],
            y=[neuron.position[1] for neuron in neurons.values()],
            mode='markers',
            marker=dict(
                size=15,
                color=[map_to_color(potential) for potential in current_potentials],
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            hovertext=[f"Membrane Potential: {potential:.1f} mV" for potential in current_potentials]
        )
        
        return [edge_trace, node_trace]
    
    # Create base figure
    base_data = create_frame()
    fig = go.Figure(
        data=base_data,
        layout=dict(
            title=dict(
                text="Neural Network Visualization",
                font=dict(color='black', size=24),
                y=0.95
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5], color='black'),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5], color='black'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')  # Set all text to black by default
        )
    )
    if membrane_record is not None:
        frames = []
        for i in range(len(membrane_record)):
            frame_data = create_frame(i)
            frames.append(go.Frame(
                data=frame_data,
                name=f'frame{i}'
            ))
        fig.frames = frames
        
    return fig

def plot_membrane_and_neurotransmitter(neurons, time_points, neuron_names):
    """
    Create plots showing membrane potentials and energy levels over time.
    """
    # Create subplots
    fig = make_subplots(
        rows=len(neuron_names), 
        cols=1,
        subplot_titles=[f"{name} Activity" for name in neuron_names],
        vertical_spacing=0.05
    )
    
    # Set figure height
    fig.update_layout(height=200 * len(neuron_names))
    
    # Add trace for each neuron
    for i, neuron_name in enumerate(neuron_names, 1):
        neuron = neurons[neuron_name]
        
        # Create color based on neuron type
        if neuron.inhibitory:
            line_color = 'rgba(255, 0, 0, 0.8)'  # Red for inhibitory (GABA)
        else:
            line_color = 'rgba(0, 255, 0, 0.8)'  # Green for excitatory (ACh)
        
        # Plot membrane potential as a line
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=[neuron.membrane_potential_history[t] for t in time_points],
                name=f"{neuron_name} Membrane",
                line=dict(color=line_color, width=2),
                mode='lines',
                showlegend=False
            ),
            row=i, 
            col=1
        )
        
        # Add spike markers if any
        if neuron.spike_times:
            fig.add_trace(
                go.Scatter(
                    x=[time_points[t] for t in neuron.spike_times],
                    y=[-55] * len(neuron.spike_times),  # Plot at threshold
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='yellow',
                        line=dict(color='white', width=1)
                    ),
                    name=f"{neuron_name} spikes",
                    showlegend=False
                ),
                row=i,
                col=1
            )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=min(time_points),
            x1=max(time_points),
            y0=-55,
            y1=-55,
            line=dict(
                color="red",
                width=1,
                dash="dash",
            ),
            row=i,
            col=1
        )
        
        # Add energy bar annotation
        current_potential = neuron.membrane_potential_history[-1] if neuron.membrane_potential_history else -70
        fig.add_annotation(
            x=1.1,
            y=0,
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            font=dict(size=20),
            row=i,
            col=1
        )
        
        # Update y-axis
        fig.update_yaxes(
            range=[-90, -40], 
            title="Membrane Potential (mV)", 
            row=i, 
            col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(color='white'),
            title_font=dict(color='white')
        )
        
        # Add neuron type annotation
        neuron_type = "Inhibitory (GABA)" if neuron.inhibitory else "Excitatory (ACh)"
        location = "Dorsal" if neuron.dorsal else "Ventral"
        fig.add_annotation(
            x=0.02,
            y=0.98,
            text=f"{location} {neuron_type}",
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            font=dict(size=10, color='black'),
            row=i,
            col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="C. elegans Neural Activity",
            font=dict(color='black', size=24)
        ),
        showlegend=False,
        xaxis_title=dict(text="Time (ms)", font=dict(color='black')),
        template="plotly_dark",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black')
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='black', size=14)
    
    return fig

def get_neurotransmitter_activity(membrane, instantiated_neurons):
    """
    Calculate neurotransmitter activity based on membrane states from hidden layer
    Args:
        membrane: Tensor or numpy array of shape [time_steps, 256] containing membrane potentials
        instantiated_neurons: List of motor neuron objects
    Returns:
        dict: Mapping from neuron name to activity array
    """
    # Convert membrane to numpy if it's a tensor
    membrane_np = membrane.detach().cpu().numpy() if torch.is_tensor(membrane) else membrane
    num_hidden = membrane_np.shape[1]  # Should be 256
    
    neurotransmitter_activity = {}
    
    # For each motor neuron, map hidden neurons to it using modulo
    for i, neuron in enumerate(instantiated_neurons):
        neuron_name = neuron
        # Map this motor neuron to a subset of hidden neurons using modulo
        hidden_indices = [j for j in range(num_hidden) if j % len(instantiated_neurons) == i]
        
        # Average the membrane potentials of the mapped hidden neurons
        neuron_activity = np.mean(membrane_np[:, hidden_indices], axis=1)
        
        # Scale to reasonable range (-70 to -50 mV)
        neuron_activity = -70 + 20 * (neuron_activity - np.min(neuron_activity)) / (np.max(neuron_activity) - np.min(neuron_activity))
        
        neurotransmitter_activity[neuron_name] = neuron_activity
    
    return neurotransmitter_activity

# def get_energy_bar(membrane_potential):
#     """Convert membrane potential to energy bar with chicken leg emojis"""
#     # Map membrane potential from -70..-50 to 0..10
#     energy = ((membrane_potential + 70) / 20) * 10
#     energy = max(0, min(10, int(energy)))
#     return "🍗" * energy + "⚪" * (10 - energy)

def simulate_membrane_potentials(neurons, time_points):
    """
    Visualize membrane potentials of neurons over time.
    Args:
        neurons: Dictionary of neuron objects
        time_points: List of time points to visualize
    """
    # Representative neurons to track
    neuron_names = ['DA1', 'DB1', 'DD1', 'VA1', 'VB1', 'VD1', 'AS1']
    
    # Create and show the visualization
    fig = plot_membrane_and_neurotransmitter(neurons, time_points, neuron_names)
    fig.show()

def create_network_visualization(neurons, membrane_history=None):
    """Create network visualization with the given neurons"""
    # Create initial edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Add edges to the trace
    for edge in G.edges():
        x0, y0 = neurons[edge[0]].position
        x1, y1 = neurons[edge[1]].position
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Create node trace
    node_trace = go.Scatter(
        x=[neuron.position[0] for neuron in neurons.values()],
        y=[neuron.position[1] for neuron in neurons.values()],
        mode='markers+text',
        text=[str(neuron) for neuron in neurons.values()],
        marker=dict(
            size=30,
            color=[map_to_color(neuron.membrane_potential) for neuron in neurons.values()],
            line=dict(width=2, color='black')
        ),
        textposition="top center",
        textfont=dict(color='black'),  # White text
        hoverinfo='text+text',
        hovertext=[f"{str(neuron)}<br>Potential: {neuron.membrane_potential:.1f}" 
                  for neuron in neurons.values()]
    )

    # Create figure with black background
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="C. elegans Neural Network",
                font=dict(color='black')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                font=dict(color='black') 
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color='white'),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color='white'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    )
    
    return fig

# Run the simulation and visualization
if __name__ == "__main__":
    time_points = list(range(100))
    instantiated_neurons = instantiate_neurons()
    membrane_history = {name: [] for name in instantiated_neurons}
    for t in time_points:
        for neuron in instantiated_neurons.values():
            neuron.membrane_potential_history.append(neuron.membrane_potential)
            if np.random.random() < 0.3:  # 30% chance of releasing neurotransmitter
                if neuron.inhibitory:
                    neurotransmitter = "GABA"
                else:
                    neurotransmitter = "ACh"
                neuron.encounters_vesicle(neurotransmitter)
            neuron.update(t)
            decay_rate = 0.1
            neuron.membrane_potential += decay_rate * (-70.0 - neuron.membrane_potential)
            membrane_history[neuron.name].append(neuron.membrane_potential)
    simulate_membrane_potentials(instantiated_neurons, time_points)
    create_network_visualization(instantiated_neurons, membrane_history)