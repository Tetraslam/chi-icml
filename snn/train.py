# preprocess and train

import numpy as np
import torch
import torch.nn as nn
from norse.torch import PoissonEncoder, ConstantCurrentLIFEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange
from PIL import Image
import os
import matplotlib.pyplot as plt


from fluoresce import NeuronVisualizer

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),          # input is 64 x 64
    transforms.ToTensor(),                # Convert the PIL image to a tensor
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (remove grayscale later to increase accuracy)
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),         
    transforms.ToTensor(),               
    transforms.Grayscale(num_output_channels=1), 
])

# define paths
flowers_data_dir = "datasets/flowers"
flowers_train_dir = flowers_data_dir + "/train"
flowers_test_dir = flowers_data_dir + "/test"

class UnlabeledImageDataset(Dataset): # define custom datset for the unlabeled test images
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 1  # Return a dummy label

class LIFCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.tau_m = 20.0  # Membrane time constant
        self.v_threshold = -55.0  # Firing threshold
        self.v_reset = -70.0  # Reset potential
        
    def forward(self, x, s):
        # Initialize state if None
        if s is None:
            s = torch.zeros_like(x)
        
        # Update membrane potential
        dv = (x - s) / self.tau_m
        v = s + dv
        
        # Check for spikes
        spikes = (v >= self.v_threshold).float()
        
        # Reset membrane potential where spikes occurred
        v = (1 - spikes) * v + spikes * self.v_reset
        
        return v, v

class SNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, recurrent_cell):
        super(SNN, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.fc_in = nn.Linear(input_features, hidden_features)
        self.recurrent_cell = recurrent_cell
        self.fc_out = nn.Linear(hidden_features, output_features)
        self.spikes_record = []
        self.membrane_record = []  # Will only store membrane states of spiking neurons
        self.spiking_neuron_indices = []  # Track which neurons are spiking
        
    def forward(self, x):
        """
        Forward pass of SNN
        x shape: (seq_length, batch_size, input_features)
        returns: (seq_length, batch_size, output_features)
        """
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        
        # Initialize hidden states and outputs
        s = None
        membrane_potentials = []  # Store membrane potentials
        spikes = []  # Store spikes
        outputs = []  # Store outputs
        
        for ts in range(seq_length):
            z = self.fc_in(x[ts, :])
            z, s = self.recurrent_cell(z, s)
            
            # Get membrane potentials
            if isinstance(s, tuple):
                v = s[0]  # Get membrane potential
            else:
                v = s
            
            # Convert to numpy for processing
            v_np = v.detach().cpu().numpy()
            z_np = z.detach().cpu().numpy()
            
            # Store membrane potentials and spikes
            membrane_potentials.append(v_np)
            spikes.append(z_np)
            
            # Forward pass through output layer
            z = self.fc_out(z)
            outputs.append(z)
        
        # Store records for later analysis
        self.membrane_record = np.array(membrane_potentials)
        self.spikes_record = np.array(spikes)
        
        # Stack outputs along time dimension
        return torch.stack(outputs, dim=0)  # (seq_length, batch_size, output_features)

class Model(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.shape[0]
        # Flatten spatial dimensions
        x = x.reshape(batch_size, -1)  # (batch_size, channels*height*width)
        # Encode into spike sequence
        x = self.encoder(x)  # (seq_length, batch_size, input_features)
        # Process through SNN
        x = self.snn(x)  # (seq_length, batch_size, num_classes)
        # Decode to predictions
        log_p_y = self.decoder(x)  # (batch_size, num_classes)
        return log_p_y


def decode(x):
    """decode SNN output to predictions
    x shape: (seq_length, batch_size, num_classes)
    returns: (batch_size, num_classes)
    """
    # average over time steps first
    x = x.mean(dim=0)  # (batch_size, num_classes)
    # ensure we have 2D tensor (batch_size, num_classes)
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing
    # log softmax over classes dimension (last dimension)
    log_p_y = torch.nn.functional.log_softmax(x, dim=-1)
    return log_p_y

def train(model, device, train_loader, optimizer, epoch, max_epochs=5, worm_visualizer=None):
    """Train the model for one epoch"""
    model.train() # set the model to training mode
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass - data is already in correct shape (batch_size, channels, height, width)
        output = model(data)
        
        # Calculate loss and backpropagate
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch}/{max_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # Get the latest activity records
            spikes = model.snn.spikes_record
            membrane = model.snn.membrane_record
            
            # Update visualization if available
            if worm_visualizer is not None:
                # Convert tensors to numpy arrays and move to CPU if needed
                spikes_np = spikes.detach().cpu().numpy() if torch.is_tensor(spikes) else spikes
                membrane_np = membrane.detach().cpu().numpy() if torch.is_tensor(membrane) else membrane
                worm_visualizer.update(spikes_np, membrane_np)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    # Create progress bar for testing
    pbar = tqdm(test_loader, desc='Testing', leave=True)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{test_loss/len(test_loader.dataset):.4f}',
                'acc': f'{100.*correct/len(test_loader.dataset):.1f}%'
            })
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.1f}%)\n')
    return test_loss, accuracy

def run_training(model, optimizer, train_loader, test_loader, epochs=5, worm_visualizer=None):
    """Complete training pipeline"""
    print("Starting training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training metrics
    training_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Train for one epoch
        train(model, device, train_loader, optimizer, epoch, epochs, worm_visualizer)
        
        # Test the model
        test_loss, test_accuracy = test(model, device, test_loader)
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {100. * test_accuracy:.1f}%\n')
        
        # Record metrics
        test_accuracies.append(test_accuracy)
    
    return model, {
        'test_accuracies': test_accuracies
    }

def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(device)
        # Get model prediction
        output = model(image_tensor)
        # Get predicted class
        pred = output.argmax(dim=1)
        return pred.item()

def encode_image_to_spikes(img, T=32, f_max=20):
    """Convert image to spike trains using Poisson encoding"""
    poisson_encoder = PoissonEncoder(T, f_max=f_max)
    spike_probabilities = poisson_encoder(img)
    # Option 1: Use just the first channel
    spikes = spike_probabilities[:, 0].reshape(T, 64*64).to_sparse().coalesce()
    return spikes

def get_default_config():
    """Return default configuration for training"""
    return {
        'T': 32,  # number of time steps
        'LR': 0.001,  # learning rate
        'INPUT_FEATURES': 64 * 64,  # flattened input size
        'HIDDEN_FEATURES': 256,
        'BATCH_SIZE': 32,
        'EPOCHS': 2,
        'F_MAX': 20  # for Poisson encoding
    }

def setup_model(config, num_classes, device):
    """Setup the complete model architecture"""
    poisson_encoder = PoissonEncoder(seq_length=config['T'], f_max=config['F_MAX'])
    lif_snn = SNN(
        input_features=config['INPUT_FEATURES'],
        hidden_features=config['HIDDEN_FEATURES'],
        output_features=num_classes,
        recurrent_cell=LIFCell()
    )
    
    model = Model(
        encoder=poisson_encoder,
        snn=lif_snn,
        decoder=decode
    ).to(device)
    
    return model

def setup_data_loaders(train_dir, test_dir, batch_size, train_transform, test_transform):
    """Setup data loaders for training and testing"""
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = UnlabeledImageDataset(root_dir=test_dir, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader, len(train_dataset.classes)

def train_model(config=None, data_dirs=None, transforms=None, device=None, worm_visualizer=None):
    """Main training function that can be called from other scripts"""
    # Use default config if none provided
    if config is None:
        config = get_default_config()
    
    # Use default device if none provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use default data directories if none provided
    if data_dirs is None:
        data_dirs = {
            'train': "datasets/flowers/train",
            'test': "datasets/flowers/test"
        }
    
    # Use default transforms if none provided
    if transforms is None:
        transforms = {
            'train': train_transform,
            'test': test_transform
        }
    
    # Setup data loaders
    train_loader, test_loader, num_classes = setup_data_loaders(
        data_dirs['train'],
        data_dirs['test'],
        config['BATCH_SIZE'],
        transforms['train'],
        transforms['test']
    )
    
    # Setup model
    model = setup_model(config, num_classes, device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
    
    # Train the model
    trained_model, metrics = run_training(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config['EPOCHS'],
        worm_visualizer=worm_visualizer
    )
    
    return trained_model, metrics

if __name__ == '__main__':
    # This code only runs if train.py is run directly
    print("Training model with default configuration...")
    model, metrics = train_model()
    
    # Save the model
    torch.save(model.state_dict(), "trained_model.pth")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.plot(metrics['test_accuracies'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()