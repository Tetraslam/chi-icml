from train import train_model, get_default_config, SNN, LIFCell
from mapping import WormStateVisualizer, simulate_membrane_potentials, instantiate_neurons, plot_membrane_and_neurotransmitter, create_network_visualization
import torch
import numpy as np

def process_snn_output(hidden_states, neurons):
    """Process SNN output to update neuron states"""
    # Get membrane potentials from hidden states
    membrane_potentials = hidden_states.detach().numpy()
    
    # Scale membrane potentials to biological range (-70 to -50 mV)
    scaled_potentials = -70 + 20 * (membrane_potentials + 1) / 2
    
    # Update neuron membrane potentials
    for i, neuron_name in enumerate(neurons.keys()):
        neuron = neurons[neuron_name]
        potential = scaled_potentials[i]
        
        # Add some inertia to make changes more gradual
        if neuron.membrane_potential_history:
            last_potential = neuron.membrane_potential_history[-1]
            potential = 0.7 * last_potential + 0.3 * potential  # Smooth changes
        
        neuron.membrane_potential = potential
        neuron.membrane_potential_history.append(potential)
        
        # Check for spikes (threshold crossing)
        if potential >= -55 and not neuron.is_spiking:
            neuron.spike_times.append(len(neuron.membrane_potential_history) - 1)
            neuron.is_spiking = True
            # After spike, membrane potential drops
            neuron.membrane_potential = -70.0
        elif potential < -55:
            neuron.is_spiking = False

def main():
    # Initialize neurons
    neurons = instantiate_neurons()
    
    # Load the trained model
    model = SNN(
        input_features=4096,  # Input size
        hidden_features=256,  # Hidden layer size
        output_features=5,  # Output size
        recurrent_cell=LIFCell()  # Leaky Integrate and Fire
    )
    try:
        model.load_state_dict(torch.load('trained_model.pth'))
    except:
        print("No saved model found, using untrained model")
    
    # Create a simple input sequence with some strong inputs to trigger spikes
    time_steps = 100
    batch_size = 1
    input_sequence = torch.zeros(time_steps, batch_size, 4096)  # [time_steps, batch_size, input_features]
    
    # Add some strong inputs at specific times to trigger spikes
    spike_times = [20, 40, 60, 80]
    for t in spike_times:
        input_sequence[t] = torch.randn(batch_size, 4096) * 2  # Stronger input
    
    # Add some background noise
    input_sequence += torch.randn(time_steps, batch_size, 4096) * 0.1
    
    # Run the model
    with torch.no_grad():
        model(input_sequence)
    
    # Get membrane record from model
    membrane_record = model.membrane_record
    
    # Process each timestep for motor neurons visualization
    time_points = []
    for t, hidden_state in enumerate(input_sequence):
        process_snn_output(hidden_state[0], neurons)  # Take first batch
        time_points.append(t)
    
    # Plot membrane potentials
    membrane_fig = plot_membrane_and_neurotransmitter(neurons, time_points, list(neurons.keys())[:5])
    membrane_fig.show()
    
    # Create network visualization with membrane record
    network_fig = create_network_visualization(neurons, membrane_record)
    network_fig.show()

if __name__ == "__main__":
    main()