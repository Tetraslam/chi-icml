import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import queue
import time

class NeuronVisualizer:
    def __init__(self, hidden_size=256, buffer_size=100):
        self.hidden_size = hidden_size
        self.buffer_size = buffer_size
        
        # Queues for communication between threads
        self.spike_queue = queue.Queue()
        self.voltage_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Buffers for storing historical data
        self.spike_buffer = deque(maxlen=buffer_size)
        self.voltage_buffer = deque(maxlen=buffer_size)
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Live Neuronal Activity')
        
        # Initialize plots
        self.spike_plot = self.ax1.imshow(np.zeros((buffer_size, hidden_size)), 
                                        aspect='auto', cmap='binary')
        self.voltage_plot = self.ax2.imshow(np.zeros((buffer_size, hidden_size)), 
                                          aspect='auto', cmap='viridis')
        
        # Labels and colorbar
        self.ax1.set_title('Spike Raster Plot')
        self.ax2.set_title('Membrane Potentials')
        plt.colorbar(self.voltage_plot, ax=self.ax2, label='Voltage')
        
        # Start visualization thread
        self.vis_thread = threading.Thread(target=self._run_visualization)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def update_state(self, spikes, voltages):
        """Update neuronal states (call this from training loop)"""
        if not self.stop_event.is_set():
            self.spike_queue.put(spikes.detach().cpu().numpy())
            self.voltage_queue.put(voltages.detach().cpu().numpy())

    def _update_plot(self, frame):
        """Update function for animation"""
        # Get new data if available
        while not self.spike_queue.empty() and not self.voltage_queue.empty():
            spikes = self.spike_queue.get()
            voltages = self.voltage_queue.get()
            
            self.spike_buffer.append(spikes)
            self.voltage_buffer.append(voltages)
        
        if len(self.spike_buffer) > 0:
            # Update spike raster plot
            spike_data = np.array(list(self.spike_buffer))
            self.spike_plot.set_array(spike_data)
            
            # Update voltage plot
            voltage_data = np.array(list(self.voltage_buffer))
            self.voltage_plot.set_array(voltage_data)
            
            # Update plot limits
            self.voltage_plot.set_clim(voltage_data.min(), voltage_data.max())
        
        return self.spike_plot, self.voltage_plot

    def _run_visualization(self):
        """Run the visualization in a separate thread"""
        self.anim = FuncAnimation(self.fig, self._update_plot, interval=100, 
                                blit=True)
        plt.show()

    def close(self):
        """Clean up resources"""
        self.stop_event.set()
        plt.close(self.fig)