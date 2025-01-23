# preprocess and train

import numpy as np
import torch
import torch.nn as nn
from norse.torch import LIFCell, LICell, PoissonEncoder, ConstantCurrentLIFEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
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

class SNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, recurrent_cell):
        super(SNN, self).__init__()
        self.input_features = input_features
        # First linear layer to transform input to hidden size
        self.fc_in = torch.nn.Linear(input_features, hidden_features, bias=False)
        # Recurrent cell
        self.cell = recurrent_cell
        # Output layer
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell()
        
        # For visualization
        self.visualizer = None
        self.record_activity = False
                             
    def forward(self, x):
        """
        x shape: (seq_length, batch_size, channels, height, width)
        returns: (seq_length, batch_size, num_classes)
        """
        seq_length, batch_size, channels, height, width = x.shape
        s1 = so = None  # Initialize neuron states
        voltages = []  # Store membrane potentials
        
        hidden_states = []  # Store hidden states for visualization
        spike_states = []   # Store spike states for visualization

        for ts in range(seq_length):
            # Flatten spatial dimensions but keep batch dimension
            z = x[ts].reshape(batch_size, -1)  # (batch_size, channels * height * width)
            # Transform to hidden size
            z = self.fc_in(z)  # (batch_size, hidden_features)
            # Process through recurrent cell
            z, s1 = self.cell(z, s1)
            
            if self.record_activity and self.visualizer is not None:
                # Record neuronal activity for visualization
                # Take first batch item for visualization
                spikes = (z > 0).float()[0]  # Binary spikes
                voltages = s1.v[0] if hasattr(s1, 'v') else z[0]  # Membrane potential
                self.visualizer.update_state(spikes, voltages)
            
            # Transform to output size
            z = self.fc_out(z)  # (batch_size, output_features)
            # Final LIF layer
            vo, so = self.out(z, so)
            voltages += [vo]
        
        return torch.stack(voltages)  # (seq_length, batch_size, output_features)
    
    def enable_recording(self, hidden_size=None):
        """Enable recording of neuronal activity"""
        if hidden_size is None:
            hidden_size = self.fc_in.out_features
        self.visualizer = NeuronVisualizer(hidden_size=hidden_size)
        self.record_activity = True
    
    def disable_recording(self):
        """Disable recording of neuronal activity"""
        if self.visualizer is not None:
            self.visualizer.close()
        self.visualizer = None
        self.record_activity = False

class Model(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y

def decode(x):
    """decode SNN output to predictions
    x shape: (seq_length, batch_size, num_classes)
    returns: (batch_size, num_classes)
    """
    # average over time steps first
    x = x.mean(dim=0)  # (batch_size, num_classes)
    # log softmax
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

def train(model, device, train_loader, optimizer, epoch, max_epochs=5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # move the progress abars
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{max_epochs}', leave=True)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        mean_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{mean_loss:.4f}',
            'acc': f'{accuracy:.1f}%'
        })
    
    mean_loss = total_loss / len(train_loader)
    return total_loss, mean_loss

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
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
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

def run_training(model, optimizer, train_loader, test_loader, epochs=5):
    """Complete training pipeline"""
    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []
    
    # Create progress bar for epochs
    epoch_pbar = trange(epochs, desc='Training Progress', leave=True)
    
    for epoch in epoch_pbar:
        # Train
        training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch + 1, epochs)
        training_losses.append(training_loss)
        mean_losses.append(mean_loss)
        
        # Test
        test_loss, accuracy = test(model, DEVICE, test_loader)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{mean_loss:.4f}',
            'test_loss': f'{test_loss:.4f}',
            'accuracy': f'{accuracy:.1f}%'
        })
    
    return model, {'training_losses': training_losses,
                  'mean_losses': mean_losses,
                  'test_losses': test_losses,
                  'accuracies': accuracies}

def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
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

if __name__ == '__main__':
    T = 32  # number of time steps
    LR = 0.001  # reduced learning rate for stability
    INPUT_FEATURES = 64 * 64  # flattened input size (grayscale)
    HIDDEN_FEATURES = 256  # increased hidden size
    OUTPUT_FEATURES = len(datasets.ImageFolder(root=flowers_train_dir).classes)  # number of flower classes
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    flowers_train_dataset = datasets.ImageFolder(root=flowers_train_dir, transform=train_transform)
    flowers_test_dataset = UnlabeledImageDataset(root_dir=flowers_test_dir, transform=test_transform)

    # Create data loaders
    batch_size = 32
    flowers_train_loader = DataLoader(
        flowers_train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    flowers_test_loader = DataLoader(
        flowers_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 
    )

    print(f"Training dataset size: {len(flowers_train_dataset)}")
    print(f"Test dataset size: {len(flowers_test_dataset)}")
    print(f"Number of classes: {len(flowers_train_dataset.classes)}")
    print(f"Classes: {flowers_train_dataset.classes}")

    # instantiate the model
    poisson_encoder = PoissonEncoder(seq_length=T, f_max=20)
    lif_snn = SNN(
        input_features=INPUT_FEATURES,      # 64x64 = 4096
        hidden_features=HIDDEN_FEATURES,    # 256 hidden neurons
        output_features=OUTPUT_FEATURES,    # number of classes
        recurrent_cell=LIFCell()           # Leaky Integrate-and-Fire neuron
    )

    model = Model(
        encoder=poisson_encoder,
        snn=lif_snn,
        decoder=decode).to(DEVICE)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train the model
    print("Starting training...")
    lif_snn.enable_recording()
    trained_model, training_metrics = run_training(
        model=model,
        optimizer=optimizer,
        train_loader=flowers_train_loader,
        test_loader=flowers_test_loader
    )
    lif_snn.disable_recording()

    # Plot training results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(training_metrics['mean_losses'], label='Training Loss')
    plt.plot(training_metrics['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(training_metrics['accuracies'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Test Accuracy')

    plt.tight_layout()
    plt.show()

    torch.save(trained_model.state_dict(), 'snn_model.pth')
    print("Training completed and model saved!")

    # Try prediction on a sample image
    sample_img, true_label = flowers_train_dataset[0]
    predicted_class = predict_image(trained_model, sample_img)
    print(f"True label: {flowers_train_dataset.classes[true_label]}")
    print(f"Predicted class: {flowers_train_dataset.classes[predicted_class]}")