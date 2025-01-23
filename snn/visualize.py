import numpy as np
import plotly.graph_objects as go



class Worm2D:
    def __init__(self, num_circles, radius_increase, start_pos=[], end_pos=[], seconds=10):
        self.num_circles = num_circles
        self.radius_increase = radius_increase
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.circles = []
        self.time = seconds
        self.fw_velocity = 0.1  
        self.up_velocity = 0.1
        # can then update the starting and end positions
        # based on neuron activity

    def update_start_end(self, start_pos, end_pos, instantiated_neurons):
        self.start_pos = start_pos
        self.end_pos = end_pos
        
        for neuron in instantiated_neurons:
            if neuron.dorsal and neuron.excitatory or neuron.ventral and neuron.inhibitory:
                # head down, update x, y for all circles
                # Calculate new positions with downward sine wave
                for i, circle in enumerate(self.circles):
                    t = i / (self.num_circles - 1)  # Normalized position along body
                    # Forward movement
                    x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t + self.fw_velocity
                    # Vertical movement with phase-shifted sine wave
                    wave = np.sin(2 * np.pi * t - np.pi/2)  # Phase shift for downward movement
                    y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t + wave * self.up_velocity
                    circle.center = [x, y]
                
            elif neuron.dorsal and neuron.inhibitory or neuron.ventral and neuron.excitatory:
                # head up, update x, y for all circles
                # Calculate new positions with upward sine wave
                for i, circle in enumerate(self.circles):
                    t = i / (self.num_circles - 1)  # Normalized position along body
                    # Forward movement
                    x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t + self.fw_velocity
                    # Vertical movement with phase-shifted sine wave
                    wave = np.sin(2 * np.pi * t + np.pi/2)  # Phase shift for upward movement
                    y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t + wave * self.up_velocity
                    circle.center = [x, y]

    def draw_frame(self):
        '''
        draws a series of circles with increasing and then decreasing radius
        representing the worm's body, following a sinusoidal pattern
        from start_pos to end_pos [x, y]
        '''
        if not self.circles:
            # Initialize circles with sinusoidal pattern
            for i in range(self.num_circles):
                t = i / (self.num_circles - 1)  # Normalized position
                
                # Calculate x, y positions along curve
                x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t
                y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t
                
                # Calculate radius with smooth increase and decrease
                if i <= self.num_circles // 2:
                    radius = 1 + (i * self.radius_increase * np.sin(np.pi * i / self.num_circles))
                else:
                    radius = 1 + ((self.num_circles - i) * self.radius_increase * np.sin(np.pi * (self.num_circles - i) / self.num_circles))
                
                self.circles.append(Circle([x, y], radius))
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add all circles to the figure
        for circle in self.circles:
            trace = circle.draw()
            fig.add_trace(trace)
        
        # Update layout with dynamic ranges
        x_range = [min(c.center[0] - c.radius for c in self.circles) - 1,
                  max(c.center[0] + c.radius for c in self.circles) + 1]
        y_range = [min(c.center[1] - c.radius for c in self.circles) - 1,
                  max(c.center[1] + c.radius for c in self.circles) + 1]
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range, scaleanchor="x", scaleratio=1),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig

    def update_circles(self):
        for circle in self.circles:
            circle.update()

    def animate(self, frames=100):
        '''
        Creates an animated figure with a play button showing the worm's movement
        Args:
            frames: Number of animation frames
        Returns:
            Plotly figure with animation
        '''
        # Initialize figure with first frame
        fig = self.draw_frame()
        
        # Create frames for animation
        animation_frames = []
        for frame_idx in range(frames):
            # Calculate phase for this frame
            phase = 2 * np.pi * frame_idx / frames
            
            # Update circle positions with wave motion
            frame_data = []
            for i, circle in enumerate(self.circles):
                t = i / (self.num_circles - 1)
                # Forward movement
                x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t
                # Add sinusoidal wave motion
                wave = np.sin(2 * np.pi * t + phase)
                y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t + wave * self.up_velocity
                
                # Generate circle points
                theta = np.linspace(0, 2*np.pi, 50)
                circle_x = x + circle.radius * np.cos(theta)
                circle_y = y + circle.radius * np.sin(theta)
                
                frame_data.append(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    fill='toself',
                    line=dict(color='rgba(0,100,255,0.8)'),
                    fillcolor='rgba(0,100,255,0.3)',
                    showlegend=False
                ))
            
            # Add frame to animation
            animation_frames.append(go.Frame(
                data=frame_data,
                name=f'frame{frame_idx}'
            ))
        
        # Add frames to figure
        fig.frames = animation_frames
        
        # Add animation buttons and slider
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }, {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Frame: '},
                'steps': [
                    {
                        'method': 'animate',
                        'label': str(k),
                        'args': [[f'frame{k}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                    for k in range(frames)
                ]
            }]
        )
        
        return fig

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def draw(self):
        # Generate points for circle using parametric equations
        theta = np.linspace(0, 2*np.pi, 50)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        
        return go.Scatter(
            x=x, 
            y=y,
            mode='lines',
            fill='toself',
            line=dict(color='rgba(0,100,255,0.8)'),
            fillcolor='rgba(0,100,255,0.3)'
        )

    def update(self):
        # Update method for future animation implementations
        # Currently handled by Worm2D class
        pass

if __name__ == '__main__':
    # Example usage with animation
    worm = Worm2D(num_circles=20, radius_increase=0.1, 
                  start_pos=[0, 0], end_pos=[10, 0])
    
    # Create animated figure
    fig = worm.animate(frames=60)  # 60 frames for smooth animation
    fig.show()