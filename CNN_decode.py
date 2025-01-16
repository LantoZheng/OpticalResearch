import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# 1. Define the Neural Network Architecture
class SpaceVaryingDeconvNet(nn.Module):
    def __init__(self):
        super(SpaceVaryingDeconvNet, self).__init__()
        # This is a simplified architecture, you might need more complex layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)  # Output is the deconvolved image
        return x

# 2. Create a Custom Dataset
class ScatteringDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Generate a random "original" image
            original_image = np.random.rand(self.image_size, self.image_size).astype(np.float32)

            # Simulate space-varying convolution
            convolved_image = self._apply_space_varying_convolution(original_image)

            data.append((original_image[None, :, :], convolved_image[None, :, :])) # Add channel dimension
        return data

    def _apply_space_varying_convolution(self, image):
        size = image.shape[0]
        convolved = np.zeros_like(image)
        for i in range(size):
            for j in range(size):
                # Define a convolution kernel that changes based on the location (i, j)
                # This is a simple example, you can define more complex kernels
                kernel_size = 5
                center_x, center_y = kernel_size // 2, kernel_size // 2
                kernel = np.zeros((kernel_size, kernel_size))

                # Example: Kernel becomes more like a horizontal edge detector at the right side
                if j > size * 0.7:
                    kernel[center_x, :] = [0, 0.25, 0.5, 0.25, 0]
                # Example: Kernel becomes more like a vertical edge detector at the top
                elif i < size * 0.3:
                    kernel[:, center_y] = [0, 0.25, 0.5, 0.25, 0]
                else:
                    kernel[:] = 1/kernel_size**2  # Simple blur kernel

                # Apply the local convolution
                convolved_patch = convolve2d(image[max(0, i - center_x):min(size, i + center_x + 1),
                                                 max(0, j - center_y):min(size, j + center_y + 1)],
                                           kernel[max(0, center_x - i):kernel_size - max(0, i + center_x + 1 - size),
                                                  max(0, center_y - j):kernel_size - max(0, j + center_y + 1 - size)],
                                           mode='same', boundary='fill', fillvalue=0)
                convolved[i, j] = convolved_patch[i - max(0, i - center_x), j - max(0, j - center_y)]
        return convolved

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        original, convolved = self.data[idx]
        return torch.tensor(convolved, dtype=torch.float32), torch.tensor(original, dtype=torch.float32)

# 3. Instantiate the Network, Dataset, and DataLoader
model = SpaceVaryingDeconvNet()
dataset = ScatteringDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    for i, (convolved_batch, original_batch) in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(convolved_batch)

        # Calculate the loss
        loss = criterion(outputs, original_batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# 6. Evaluation (Optional) - You would typically have a separate test dataset
# Here's a simple way to visualize some results from the training data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_convolved, test_original = dataset[0]  # Get a single sample
    predicted_original = model(test_convolved.unsqueeze(0)) # Add batch dimension

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(test_original.squeeze().numpy(), cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(test_convolved.squeeze().numpy(), cmap='gray')
    axs[1].set_title('Convolved Image (Input)')
    axs[2].imshow(predicted_original.squeeze().numpy(), cmap='gray')
    axs[2].set_title('Predicted Original Image (Output)')
    plt.show()