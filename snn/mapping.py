import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx


G = nx.Graph()

neurons = ['AVA', 'AVB', 'AVD', 'AVE', 'AS1', 'AS2', 'DA1', 'DA2', 'DB1', 'DB2']
G.add_nodes_from(neurons)

# biological conncetions
connections = [('AVA', 'AVB'), ('AVB', 'AVD'), ('AVD', 'AVE'), 
               ('AS1', 'AS2'), ('DA1', 'DA2'), ('DB1', 'DB2'),
               ('AVA', 'DA1'), ('AVB', 'DB1'), ('AVD', 'AS1')]  
G.add_edges_from(connections)

def map_to_color(activity):
    """Map neural activity to color (red for high activity, blue for low)"""
    # Normalize activity to [0,1] range
    activity = (activity + 1) / 2
    # Red for high activity, blue for low
    return f'rgb({int(activity * 255)}, 100, {int((1-activity) * 255)})'

# Set random seed for reproducibility
np.random.seed(42)

pos = nx.circular_layout(G)

num_frames = 100
time = np.linspace(0, 2 * np.pi, num_frames)
# Create different patterns for different neuron types
neurotransmitter_activity_over_time = {}
for i, neuron in enumerate(neurons):
    if neuron.startswith('AV'):  # Command interneurons
        neurotransmitter_activity_over_time[neuron] = np.sin(2*time + i*np.pi/4)
    elif neuron.startswith('AS'):  # Motor neurons
        neurotransmitter_activity_over_time[neuron] = np.sin(3*time + i*np.pi/3)
    else:  # DA/DB motor neurons
        neurotransmitter_activity_over_time[neuron] = np.sin(1.5*time + i*np.pi/2)

# Create the initial edge trace
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Add edges to the trace
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)

# Create initial node trace
node_trace = go.Scatter(
    x=[pos[neuron][0] for neuron in neurons],
    y=[pos[neuron][1] for neuron in neurons],
    mode='markers+text',
    text=neurons,
    marker=dict(
        size=30,
        color=[map_to_color(neurotransmitter_activity_over_time[neuron][0]) for neuron in neurons],
        line=dict(width=2, color='black')
    ),
    textposition="top center",
    hoverinfo='text+text',
    hovertext=[f"{neuron}<br>Activity: {neurotransmitter_activity_over_time[neuron][0]:.2f}" 
               for neuron in neurons]
)

# Create figure
fig = go.Figure(data=[edge_trace, node_trace])

# Add frames for animation
frames = [go.Frame(
    data=[
        edge_trace,
        go.Scatter(
            x=[pos[neuron][0] for neuron in neurons],
            y=[pos[neuron][1] for neuron in neurons],
            mode='markers+text',
            text=neurons,
            marker=dict(
                size=30,
                color=[map_to_color(neurotransmitter_activity_over_time[neuron][i]) for neuron in neurons],
                line=dict(width=2, color='black')
            ),
            textposition="top center",
            hoverinfo='text+text',
            hovertext=[f"{neuron}<br>Activity: {neurotransmitter_activity_over_time[neuron][i]:.2f}" 
                      for neuron in neurons]
        )
    ],
    name=f'frame{i}'
) for i in range(num_frames)]

fig.frames = frames

# Add play/pause button and layout settings
fig.update_layout(
    title=dict(
        text='C. elegans Neural Network Activity',
        x=0.5,
        y=0.95,
        font=dict(size=24)
    ),
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=60),
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        x=0.1,
        y=1.0,
        xanchor='right',
        yanchor='top',
        pad=dict(t=0, r=10),
        buttons=[dict(
            label='Play',
            method='animate',
            args=[None, dict(frame=dict(duration=50, redraw=True), 
                           fromcurrent=True,
                           mode='immediate',
                           transition=dict(duration=50))]
        )]
    )],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
    plot_bgcolor='white'
)

# Show figure
fig.show()