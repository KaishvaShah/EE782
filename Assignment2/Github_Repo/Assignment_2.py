#!/usr/bin/env python
# coding: utf-8

# # EE782 Assignment 2
# ## Kaishva Chintan Shah (200020066)

# In[175]:


import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
import torch
import cv2


# In[31]:


# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the GPU
    print("Using GPU:", torch.cuda.get_device_name(0))  # Display GPU name
else:
    device = torch.device("cpu")  # Use the CPU
    print("CUDA is not available. Using CPU.")


# In[32]:


import os
import itertools
import cv2

# Specify the path to the "train" folder containing subfolders
train_folder = r'./train'
val_folder = r'./validation'
test_folder = r'./test'

# Function to load and label images from a subfolder
def load_and_label_images(folder_path):
    # Lists to store image paths and labels
    images = []
    labels = []

    # Extract the name of the current subfolder
    folder_name = os.path.basename(folder_path)

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Consider other image formats if needed
            img_path = os.path.join(folder_path, filename)
            images.append(img_path)  # Add image path to the list
            labels.append(folder_name)  # Assign the folder name as the label

    return images, labels

# Initialize lists for storing image paths and labels
train_images_paths = []
train_labels_name = []
val_images_paths = []
val_labels_name = []
test_images_paths = []
test_labels_name = []


# In[33]:


# Load images and labels from subfolders in the "train" directory
train_subfolders = os.listdir(train_folder)

# Iterate through subfolders in the "train" directory
for subfolder in train_subfolders:
    subfolder_path = os.path.join(train_folder, subfolder)

    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Use the "load_and_label_images" function to load images and their labels
        images, labels = load_and_label_images(subfolder_path)

        # Extend the lists with image paths and corresponding labels
        train_images_paths.extend(images)
        train_labels_name.extend(labels)

# The code above loads and labels images from subfolders within the "train" directory.
# It iterates through each subfolder, loads the images within it, and associates them
# with their corresponding folder names as labels.

# The commented code below would display the first 5 loaded images along with their labels.
# For each image, it uses the OpenCV library to read and display the image. However, this part
# is currently commented out and is not executed.
# for img_path, label in zip(train_images_paths[:5], train_labels_name[:5]):
#     img = cv2.imread(img_path)
#     cv2_imshow(img)  # Display the image
#     cv2.waitKey(0)  # Wait for a key press to move to the next image

# Close all displayed image windows
cv2.destroyAllWindows()


# In[34]:


# Load images and labels from subfolders in the "validation" directory
val_subfolders = os.listdir(val_folder)

# Iterate through subfolders in the "validation" directory
for subfolder in val_subfolders:
    subfolder_path = os.path.join(val_folder, subfolder)

    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Use the "load_and_label_images" function to load images and their labels
        images, labels = load_and_label_images(subfolder_path)

        # Extend the lists with image paths and corresponding labels
        val_images_paths.extend(images)
        val_labels_name.extend(labels)

# The code above loads and labels images from subfolders within the "validation" directory.
# It iterates through each subfolder, loads the images within it, and associates them
# with their corresponding folder names as labels.

# The commented code below would display the first 5 loaded images along with their labels.
# For each image, it uses the OpenCV library to read and display the image. However, this part
# is currently commented out and is not executed.
# for img_path, label in zip(val_images_paths[:5], val_labels_name[:5]):
#     img = cv2.imread(img_path)
#     cv2_imshow(img)  # Display the image
#     cv2.waitKey(0)  # Wait for a key press to move to the next image

# Close all displayed image windows
cv2.destroyAllWindows()


# In[37]:


# Load images and labels from subfolders in the "test" directory
test_subfolders = os.listdir(test_folder)

# Iterate through subfolders in the "test" directory
for subfolder in test_subfolders:
    subfolder_path = os.path.join(test_folder, subfolder)

    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Use the "load_and_label_images" function to load images and their labels
        images, labels = load_and_label_images(subfolder_path)

        # Extend the lists with image paths and corresponding labels
        test_images_paths.extend(images)
        test_labels_name.extend(labels)

# The code above loads and labels images from subfolders within the "test" directory.
# It iterates through each subfolder, loads the images within it, and associates them
# with their corresponding folder names as labels.

# The commented code below would display the first 5 loaded images along with their labels.
# For each image, it uses the OpenCV library to read and display the image. However, this part
# is currently commented out and is not executed.
# for img_path, label in zip(test_images_paths[:5], test_labels_name[:5]):
#     img = cv2.imread(img_path)
#     cv2_imshow(img)  # Display the image
#     cv2.waitKey(0)  # Wait for a key press to move to the next image

# Close all displayed image windows
cv2.destroyAllWindows()


# In[38]:


import cv2
import torch
import random
import matplotlib.pyplot as plt

# Function to preprocess and load images
def preprocess_and_load_images(image_paths, min_size=(224, 224)):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        # Resize the image to meet the minimum size requirement
        if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
            image = cv2.resize(image, min_size, interpolation=cv2.INTER_CUBIC)

        images.append(image)
    return images

# Ensure train_labels has the same length as train_images
# assert len(train_labels) == len(train_images), "Length of train_labels should match train_images."

# Load and preprocess images
train_images = preprocess_and_load_images(train_images_paths)
val_images = preprocess_and_load_images(val_images_paths)
test_images = preprocess_and_load_images(test_images_paths)

# Display a few sample images
sample_indices = random.sample(range(len(train_images)), 3)  # Adjust the number of samples as needed
for idx in sample_indices:
    plt.imshow(train_images[idx])
    plt.title(f"Sample Image - Label: {train_labels_name[idx]}")
    plt.axis('off')
    plt.show()


# In[39]:


import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

# Function to create pairs and labels while randomly selecting a subset
def create_random_pairs_and_labels(images, labels, num_positive_pairs_to_select = 1000, num_negative_pairs_to_select = 1000):
    num_samples = len(images)
    pairs = []
    label_list = []

    while(num_positive_pairs_to_select):
        # Randomly select two different samples
        i, j = random.sample(range(num_samples), 2)
        pair = [images[i], images[j]]
        same_class = (labels[i] == labels[j])
        label = 1 if same_class else 0
        if(label == 1):
            pairs.append(pair)
            label_list.append(label)
            num_positive_pairs_to_select -= 1

    while(num_negative_pairs_to_select):
        # Randomly select two different samples
        i, j = random.sample(range(num_samples), 2)
        pair = [images[i], images[j]]
        same_class = (labels[i] == labels[j])
        label = 1 if same_class else 0
        if(label == 0):
            pairs.append(pair)
            label_list.append(label)
            num_negative_pairs_to_select -= 1

    # Reshape pairs to (num_pairs, 2, channels, height, width)
    # (Note: The original code is commented out as it's not necessary)
    # pairs = np.array(pairs).reshape(-1, 2, *pairs[0][0].shape)
    return pairs, label_list

# Specify the number of random pairs to select
num_random_pairs = 10000  # Adjust as needed

# Create random pairs and labels for training, validation, and testing
train_pairs, train_labels = create_random_pairs_and_labels(train_images, train_labels_name, num_positive_pairs_to_select = 3000, num_negative_pairs_to_select = 3000)
val_pairs, val_labels = create_random_pairs_and_labels(val_images, val_labels_name, num_positive_pairs_to_select = 500, num_negative_pairs_to_select = 500)
test_pairs, test_labels = create_random_pairs_and_labels(test_images, test_labels_name, num_positive_pairs_to_select = 500, num_negative_pairs_to_select = 500)

# Print the shape of the training pairs
print(np.array(train_pairs).shape)

# Define a custom dataset class
class SiameseDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_pair = self.pairs[idx]
        label = self.labels[idx]

        if self.transform:
            image_pair = [self.transform(image) for image in image_pair]

        # Stack and permute the dimensions to have the desired order
        # (Note: The code for stacking and permuting is commented out as it's not necessary)
        # image_pair = torch.stack(image_pair).permute(1, 0, 2, 3, 4)
        return image_pair, label

# Define transformations for the images (resize and convert to tensor)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL image
    transforms.Resize((224, 224)),  # Resize to a consistent size
    transforms.ToTensor(),  # Convert to PyTorch tensor
])

# Create datasets and data loaders for training, validation, and testing
train_dataset = SiameseDataset(train_pairs, train_labels, transform=transform)
val_dataset = SiameseDataset(val_pairs, val_labels, transform=transform)
test_dataset = SiameseDataset(test_pairs, test_labels, transform=transform)

# Print the length of the training dataset
print("Number of samples in train_dataset:", len(train_dataset))

# Print the shape of the first data point (image_pair)
sample_data_point, label = train_dataset[0]
print("Shape of the first data point (image_pair):", len(sample_data_point), sample_data_point[0].shape)

batch_size = 16  # Adjust batch size as needed
# Create data loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# In[10]:


import torch

# Iterate through the train_loader to get the size of the first batch
for batch_data, batch_labels in train_loader:
    # Convert the batch data to a tensor
    batch_data = torch.stack(batch_data).permute(1, 0, 2, 3, 4)
    
    # Shape of the batch data (batch_size, channels, height, width)
    batch_shape = batch_data.shape
    print("Shape of a batch in train_loader:", batch_shape)
    
    print("Shape of labels:", batch_labels.size())
    
    break  # Break after the first batch


# In[12]:


# Iterate through the train_loader to get the size of all batches
for batch_data, batch_labels in train_loader:
    # Convert the batch data to a tensor
    batch_data = torch.stack(batch_data).permute(1, 0, 2, 3, 4)
    
    # Shape of the batch data (batch_size, channels, height, width)
    batch_shape = batch_data.shape
    print("Shape of a batch in train_loader:", batch_shape)
    
    print("Shape of labels:", batch_labels.size())
    
    # Extract images and labels from the batch
    images, labels = batch_data, batch_labels
    
    # Plot a few sample image pairs
    sample_indices = random.sample(range(len(images)), 6)  # Adjust the number of samples as needed
    for idx in sample_indices:
        image_pair = images[idx]
        label = labels[idx]

        plt.figure(figsize=(8, 4))

        # Plot the first image in the pair
        plt.subplot(1, 2, 1)
        plt.imshow(image_pair[0].permute(1, 2, 0))  # Reverse the image transformation
        plt.title(f"Image 1 - Label: {label.item()}")
        plt.axis('off')

        # Plot the second image in the pair
        plt.subplot(1, 2, 2)
        plt.imshow(image_pair[1].permute(1, 2, 0))  # Reverse the image transformation
        plt.title(f"Image 2 - Label: {label.item()}")
        plt.axis('off')

        plt.show()

    break
# This loop will go through all batches in the train_loader


# # Siamese Network for Image Embeddings
# 
# In this code, we define a Siamese Network using PyTorch. A Siamese Network is a neural architecture designed for learning similarity between pairs of data points. In this case, it is used for image similarity learning.
# 
# ## Importing Libraries
# 
# We start by importing the necessary PyTorch libraries and modules.
# 
# ## `get_backbone_out_features` Function
# 
# A utility function is defined to determine the output features based on the chosen backbone network. This is important because different backbones have different output dimensions.
# 
# ## `SiameseNetwork` Class
# 
# This is the main class that defines the Siamese Network.
# 
# ### Constructor
# 
# - The constructor (`__init__`) takes two main parameters: `backbone` and `embedding_size`.
# - `backbone` is the name of the neural network backbone (e.g., "resnet18").
# - `embedding_size` is the dimensionality of the output embeddings.
# 
# ### Backbone Initialization
# 
# - The code checks if the specified `backbone` exists in `torchvision.models`. If it doesn't, an exception is raised.
# - The specified backbone model is created and pre-trained weights are loaded.
# - The classification head of the backbone is removed, leaving the feature extraction part.
# - Adaptive average pooling is applied to ensure that the feature maps have a consistent size.
# 
# ### Embedding Layer
# 
# - A fully connected layer is added to generate embeddings of size `embedding_size`.
# - A Leaky ReLU activation is applied after the linear layer.
# 
# ### `forward_one` Method
# 
# - This method processes a single input image through one branch of the Siamese Network.
# - It passes the image through the backbone, applies adaptive pooling, and generates embeddings.
# 
# ### `forward` Method
# 
# - This method is used for processing two input images (image pairs).
# - It calls `forward_one` for each image in the pair.
# - The output is a pair of embeddings, one for each input image.
# 
# This Siamese Network is designed for various similarity-based tasks, where the goal is to learn embeddings that represent the similarity or dissimilarity between input pairs. It's a powerful architecture for tasks such as face recognition, object tracking, and image similarity learning.
# 

# In[14]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_backbone_out_features(backbone):
    if 'resnet' in backbone:
        return 512  # Output features for ResNet backbones
    elif 'densenet' in backbone:
        return 1024  # Output features for DenseNet backbones
    else:
        raise ValueError("Unsupported backbone: {}".format(backbone))

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18", embedding_size=64):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

        Parameters:
                backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
                embedding_size (int): Size of the output embedding vector.
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # Remove the classification head of the backbone network
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Adaptive pooling to ensure consistent feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to generate embeddings
        out_features = get_backbone_out_features(backbone)
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, embedding_size),
            nn.LeakyReLU(),
        )

    def forward_one(self, x):
        # Forward pass through one branch of the siamese network
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = self.embedding_layer(x)
        return x

    def forward(self, img1, img2):
        # Forward pass through both branches of the siamese network
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        return feat1, feat2


# # Visualizing Siamese Network Predictions
# 
# The `visualize_predictions` function is designed to evaluate a Siamese Network model and visualize its predictions on a given dataset. This function can be used to assess the model's performance and visually inspect its similarity score predictions.
# 
# ## Parameters
# 
# - `model`: The trained Siamese Network model that you want to evaluate.
# - `data_loader`: The data loader that provides the image pairs and ground truth labels for evaluation.
# - `show_predictions` (optional): A boolean parameter to specify whether to show visualizations of individual predictions (default is `False`).
# 
# ## Functionality
# 
# 1. The function sets the model to evaluation mode and moves it to the specified device (e.g., GPU) to ensure no gradient computation during evaluation.
# 
# 2. It initializes variables to keep track of losses, correct predictions, and the total number of examples.
# 
# 3. The function iterates through the data loader, processing one batch of image pairs and labels at a time.
# 
# 4. For each batch, the following steps are performed:
#    - The model processes the image pairs (`img1` and `img2`) to generate embeddings (`feat1` and `feat2`).
#    - Cosine similarity is calculated between the embeddings using `F.cosine_similarity`.
#    - Sigmoid activation is applied to obtain similarity scores in the range [0, 1].
#    - The similarity scores and ground truth labels (`y`) are used to calculate the loss.
#    - The loss is appended to the `losses` list.
#    - Correct predictions are counted by comparing predicted similarity scores with a threshold (`Prediction_threshold`).
#    - If `show_predictions` is set to `True`, individual image pairs with their predicted labels are visualized using `matplotlib`.
# 
# 5. After processing all batches, the function calculates the accuracy on the test dataset by comparing the correct predictions to the total number of examples.
# 
# 6. The function prints the accuracy on the test dataset as a percentage.
# 
# ## Use Case
# 
# This function is useful for evaluating and visualizing the performance of a Siamese Network model, especially for tasks involving similarity learning, such as face recognition or image similarity matching. The visualization of predictions can help in understanding the model's behavior and identifying potential issues.
# 
# 

# In[26]:


import matplotlib.pyplot as plt

def visualize_predictions(model, data_loader, show_predictions = False):
    model.eval()
    model.to(device)  # Move the model to the GPU
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, ((img1, img2), y) in enumerate(data_loader):
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            # Get model predictions
            feat1, feat2 = model(img1, img2)
            
            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)
        
            output = similarity_scores.view(-1)
            y = y.float()
            loss = criterion(output, y)
            losses.append(loss.item())

            correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            total += len(y)

            # Extract the predicted similarity scores
            predicted_labels = (output > Prediction_threshold).to(torch.int)
            if(show_predictions):
#                 Loop through the batch and visualize each image pair with predictions
                for i in range(len(y)):
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img1[i].cpu().permute(1, 2, 0).numpy())
                    plt.title("Image 1")
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(img2[i].cpu().permute(1, 2, 0).numpy())
                    plt.title("Image 2")
                    plt.axis('off')

                    plt.suptitle(f"Predicted label: {predicted_labels[i]}", fontsize=14)

                    plt.show()

    print("Accuracy on the test dataset: {:.2f}%".format(100 * correct / total))



# # Siamese Network Training Loop
# 
# This code snippet demonstrates the training loop for a Siamese Network model. Siamese Networks are used for similarity learning tasks, where the goal is to learn the similarity between pairs of data points. This code specifically trains a Siamese Network for a similarity task using PyTorch.
# 
# ## Training Setup
# 
# - `device`: The code checks if a CUDA device (GPU) is available and sets the `device` accordingly (either 'cuda' or 'cpu').
# - `model`: Initializes the Siamese Network model with an embedding size of 128 and moves it to the specified device.
# - `optimizer`: Uses the Adam optimizer to update the model's parameters with a learning rate of 0.0001.
# - `criterion`: Defines the loss function for training, which is Cross-Entropy Loss.
# - `Prediction_threshold`: A threshold value used to determine similarity predictions (default is set to 0.65).
# 
# ## Training Loop
# 
# The code runs a training loop for a specified number of epochs (in this case, 10). Each epoch represents a complete pass through the training dataset.
# 
# - `model.train()`: Puts the model in training mode to enable gradient computation.
# 
# - The loop iterates through batches of data from the `train_loader`, which contains pairs of images and their corresponding labels.
# 
# - For each batch:
#   - The batch data (`img1` and `img2`) and labels (`y`) are moved to the specified device.
#   - The optimizer's gradients are zeroed using `optimizer.zero_grad()` to clear any previous gradients.
#   - The model processes the image pairs and calculates similarity scores based on the cosine similarity between their embeddings.
#   - Sigmoid activation is applied to the similarity scores to constrain them to the range [0, 1].
#   - The model's predictions and ground truth labels are used to compute the Cross-Entropy Loss.
#   - Backpropagation is performed to update the model's parameters.
#   - Training loss, correct predictions, and the total number of examples are updated for this batch.
# 
# - Optionally, a validation loop can be added to evaluate the model's performance on the validation dataset. It calculates validation loss and accuracy in a similar way to the training loop, but without backpropagation.
# 
# - After each epoch, the code prints the training loss, validation loss, training accuracy, and validation accuracy.
# 
# ## Model Saving
# 
# At the end of training, the trained model's parameters are saved to a file named 'siamese_model.pth'.
# 
# ## Usage
# 
# This code can be used to train a Siamese Network for similarity learning tasks, such as image similarity matching or verification tasks.
# 

# In[16]:


# Training loop
# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork(embedding_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

Prediction_threshold = 0.65

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_X, y in train_loader:
        (img1, img2) = batch_X
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        optimizer.zero_grad()
        
        # Get model predictions
        feat1, feat2 = model(img1, img2)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

        # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
        similarity_scores = torch.sigmoid(similarity_scores)

        output = similarity_scores.view(-1)

        # Convert target to Float type
        y = y.float()
       
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
        total += len(y)

    # Validation loop (optional)
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, y in val_loader:
            (img1, img2) = batch_X
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            # Get model predictions
            feat1, feat2 = model(img1, img2)
            
            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)
        
            output = similarity_scores.view(-1)
            # Convert target to Float type
            y = y.float()

            val_loss = criterion(output, y)

            val_losses.append(val_loss.item())
            val_correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            val_total += len(y)

    # Calculate training and validation loss for the epoch
    train_loss = sum(losses) / len(losses)
    val_loss = sum(val_losses) / max(1, len(val_losses))
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model.pth')
# visualize_predictions(model, test_loader)


# # Creating Random Pairs and Labels
# 
# This section of the code is responsible for creating random pairs of images from the training dataset and assigning labels to these pairs. This process is crucial for training Siamese Networks, which learn to differentiate between similar and dissimilar pairs of data.
# 
# - The key difference from the previous dataset creation is the inclusion of data augmentation transformations. In this dataset, images are resized to a larger size for better cropping, randomly cropped to the desired size, flipped horizontally, and rotated by up to 15 degrees. These augmentations enhance the model's ability to learn invariant features.
# 
# - The `transform` parameter in the `SiameseDataset` class enables the application of these transformations to the image pairs.
# 
# - Data augmentation is commonly used to improve model robustness and generalization, especially in computer vision tasks.
# 
# - The final `print` statements display the number of samples in the `train_dataset` and the shape of the first data point in the dataset, just like in the previous dataset creation.
# 

# In[40]:


import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
# Function to create pairs and labels while randomly selecting a subset
def create_random_pairs_and_labels(images, labels, num_positive_pairs_to_select = 1000, num_negative_pairs_to_select = 1000):
    num_samples = len(images)
    pairs = []
    label_list = []

    while(num_positive_pairs_to_select):
        # Randomly select two different samples
        i, j = random.sample(range(num_samples), 2)
        pair = [images[i], images[j]]
        same_class = (labels[i] == labels[j])
        label = 1 if same_class else 0
        if(label == 1):
            pairs.append(pair)
            label_list.append(label)
            num_positive_pairs_to_select -= 1

    while(num_negative_pairs_to_select):
        # Randomly select two different samples
        i, j = random.sample(range(num_samples), 2)
        pair = [images[i], images[j]]
        same_class = (labels[i] == labels[j])
        label = 1 if same_class else 0
        if(label == 0):
            pairs.append(pair)
            label_list.append(label)
            num_negative_pairs_to_select -= 1
   
    # Reshape pairs to (num_pairs, 2, channels, height, width)
#     pairs = np.array(pairs).reshape(-1, 2, *pairs[0][0].shape)
    return pairs, label_list

# Specify the number of random pairs to select
num_random_pairs = 10000  # Adjust as needed

train_pairs, train_labels = create_random_pairs_and_labels(train_images, train_labels_name, num_positive_pairs_to_select = 3000, num_negative_pairs_to_select = 3000)
val_pairs, val_labels = create_random_pairs_and_labels(val_images, val_labels_name, num_positive_pairs_to_select = 500, num_negative_pairs_to_select = 500)
test_pairs, test_labels = create_random_pairs_and_labels(test_images, test_labels_name, num_positive_pairs_to_select = 500, num_negative_pairs_to_select = 500)

print(np.array(train_pairs).shape)

# Define a custom dataset class
class SiameseDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_pair = self.pairs[idx]
        label = self.labels[idx]

        if self.transform:
            image_pair = [self.transform(image) for image in image_pair]
            
                # Stack and permute the dimensions to have the desired order
#         image_pair = torch.stack(image_pair).permute(1, 0, 2, 3, 4)
        return image_pair, label

# Define transformations for the images (resize, augment and convert to tensor)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize to a larger size for better cropping
    transforms.RandomCrop((224, 224)),  # Randomly crop to the desired size
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate by up to 15 degrees
    transforms.ToTensor(),
])


# Create datasets and data loaders for training, validation, and testing
train_dataset = SiameseDataset(train_pairs, train_labels, transform=transform)
val_dataset = SiameseDataset(val_pairs, val_labels, transform=transform)
test_dataset = SiameseDataset(test_pairs, test_labels, transform=transform)

# Print the length of the dataset
print("Number of samples in train_dataset:", len(train_dataset))

# Print the shape of the first data point (image_pair)
sample_data_point, label = train_dataset[0]
print("Shape of the first data point (image_pair):", len(sample_data_point), sample_data_point[0].shape)


batch_size = 16  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)


# # Siamese Network with Dropout
# 
# This code defines a modified Siamese network with dropout. The Siamese network is a neural network architecture commonly used for tasks like face recognition and similarity-based classification. In this specific network, the addition is the incorporation of dropout, which is a regularization technique. The network uses a pre-trained backbone from the torchvision.models, removes the classification head, and appends an adaptive pooling layer and a fully connected layer with dropout.
# 
# - The `SiameseNetwork_withdropout` class takes several parameters:
#   - `backbone`: Specifies the backbone architecture for the network, which can be selected from a list of models provided by PyTorch's torchvision.models.
#   - `embedding_size`: Specifies the size of the output embedding vector.
#   - `dropout_prob`: Specifies the probability of dropout, which controls the amount of dropout applied to the network.
#   
# - The `forward_one` method is used for the forward pass through one branch of the Siamese network. It processes a single input and returns an embedding after passing it through the backbone, adaptive pooling, and the dropout layer.
# 
# - The `forward` method is used for the forward pass through both branches of the Siamese network. It returns the embeddings for two input images after passing them through the network. However, in this modified network, the similarity scoring and sigmoid activation have been commented out.
# 
# - The purpose of dropout is to prevent overfitting during training by randomly setting a fraction of input units to zero. This helps the network generalize better to new data.
# 
# The usage of dropout in the network improves its ability to generalize and reduces the risk of overfitting.
# 

# In[41]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_backbone_out_features(backbone):
    if 'resnet' in backbone:
        return 512  # Output features for ResNet backbones
    elif 'densenet' in backbone:
        return 1024  # Output features for DenseNet backbones
    else:
        raise ValueError("Unsupported backbone: {}".format(backbone))

class SiameseNetwork_withdropout(nn.Module):
    def __init__(self, backbone="resnet18", embedding_size=64, dropout_prob=0.5):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

        Parameters:
                backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
                embedding_size (int): Size of the output embedding vector.
                dropout_prob (float): Probability of dropout.
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # Remove the classification head of the backbone network
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Adaptive pooling to ensure consistent feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to generate embeddings with dropout
        out_features = get_backbone_out_features(backbone)
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)  # Add dropout with the specified probability
        )

    def forward_one(self, x):
        # Forward pass through one branch of the siamese network
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = self.embedding_layer(x)
        return x

    def forward(self, img1, img2):
        # Forward pass through both branches of the siamese network
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)

#         # Compute the cosine similarity
#         similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

#         # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
#         similarity_scores = torch.sigmoid(similarity_scores)

        return feat1, feat2


# ## Dropout - 0.5

# In[42]:


# Training loop
# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork_withdropout(embedding_size=128, dropout_prob = 0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

Prediction_threshold = 0.65

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_X, y in train_loader:
        (img1, img2) = batch_X
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        optimizer.zero_grad()
        
        # Get model predictions
        feat1, feat2 = model(img1, img2)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

        # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
        similarity_scores = torch.sigmoid(similarity_scores)

        output = similarity_scores.view(-1)

        # Convert target to Float type
        y = y.float()
#         print(output)
#         print(y)
#         break
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
        total += len(y)

    # Validation loop (optional)
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, y in val_loader:
            (img1, img2) = batch_X
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            # Get model predictions
            feat1, feat2 = model(img1, img2)

            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)

            output = similarity_scores.view(-1)

            # Convert target to Float type
            y = y.float()

            val_loss = criterion(output, y)

            val_losses.append(val_loss.item())
            val_correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            val_total += len(y)

    # Calculate training and validation loss for the epoch
    train_loss = sum(losses) / len(losses)
    val_loss = sum(val_losses) / max(1, len(val_losses))
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model.pth')


# ## Dropout - 0.8

# In[43]:


# Training loop
# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork_withdropout(embedding_size=128, dropout_prob = 0.8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

Prediction_threshold = 0.65

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_X, y in train_loader:
        (img1, img2) = batch_X
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        optimizer.zero_grad()
        # Get model predictions
        feat1, feat2 = model(img1, img2)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

        # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
        similarity_scores = torch.sigmoid(similarity_scores)

        output = similarity_scores.view(-1)

        # Convert target to Float type
        y = y.float()
#         print(output)
#         print(y)
#         break
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
        total += len(y)

    # Validation loop (optional)
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, y in val_loader:
            (img1, img2) = batch_X
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            # Get model predictions
            feat1, feat2 = model(img1, img2)

            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)

            output = similarity_scores.view(-1)

            # Convert target to Float type
            y = y.float()

            val_loss = criterion(output, y)

            val_losses.append(val_loss.item())
            val_correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            val_total += len(y)

    # Calculate training and validation loss for the epoch
    train_loss = sum(losses) / len(losses)
    val_loss = sum(val_losses) / max(1, len(val_losses))
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model.pth')


# ## Dropout - 0.2

# In[72]:


# Training loop
# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork_withdropout(embedding_size=128, dropout_prob = 0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

Prediction_threshold = 0.65

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_X, y in train_loader:
        (img1, img2) = batch_X
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        optimizer.zero_grad()
        # Get model predictions
        feat1, feat2 = model(img1, img2)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

        # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
        similarity_scores = torch.sigmoid(similarity_scores)

        output = similarity_scores.view(-1)

        # Convert target to Float type
        y = y.float()
#         print(output)
#         print(y)
#         break
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
        total += len(y)

    # Validation loop (optional)
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, y in val_loader:
            (img1, img2) = batch_X
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            # Get model predictions
            feat1, feat2 = model(img1, img2)

            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)

            output = similarity_scores.view(-1)

            # Convert target to Float type
            y = y.float()

            val_loss = criterion(output, y)

            val_losses.append(val_loss.item())
            val_correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            val_total += len(y)

    # Calculate training and validation loss for the epoch
    train_loss = sum(losses) / len(losses)
    val_loss = sum(val_losses) / max(1, len(val_losses))
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model.pth')


# # Training Loop with Learning Rate Scheduler
# 
# This code represents the training loop for a Siamese network with a learning rate scheduler. The learning rate scheduler is a crucial part of the training process, and its purpose is to help the model converge effectively.
# 
# Here's a brief overview of the role of the learning rate scheduler in this training loop:
# 
# - The learning rate scheduler is implemented using the `StepLR` scheduler, which reduces the learning rate (LR) by half every 2 epochs. This strategy is known as step decay scheduling.
# 
# - Step decay scheduling is a common technique used to fine-tune the learning rate during training. By reducing the LR at predefined intervals (in this case, every 2 epochs), it allows the model to converge more effectively.
# 
# - The scheduler is associated with the optimizer (`optimizer`) used for updating the model's weights. After each LR adjustment, the optimizer continues training with the updated LR.
# 
# - A learning rate scheduler helps ensure that the model converges to a good solution by controlling the step size for weight updates. It can be particularly useful when dealing with complex datasets or architectures.
# 
# - The rest of the training loop is similar to the previous version, including model predictions, loss computation, and weight updates.
# 
# - The model can be further evaluated on the validation dataset (validation loop) to monitor its performance during training.
# 
# By using a learning rate scheduler, this training loop ensures that the model optimally adapts its weights, making the training process more effective and efficient.
# 

# In[73]:


import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# Training loop
# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork_withdropout(embedding_size=128, dropout_prob = 0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)  # Reduce LR by half every 2 epochs

criterion = torch.nn.CrossEntropyLoss()

Prediction_threshold = 0.65

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_X, y in train_loader:
        (img1, img2) = batch_X
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        optimizer.zero_grad()
        # Get model predictions
        feat1, feat2 = model(img1, img2)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

        # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
        similarity_scores = torch.sigmoid(similarity_scores)

        output = similarity_scores.view(-1)

        # Convert target to Float type
        y = y.float()
#         print(output)
#         print(y)
#         break
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
        total += len(y)

    # Validation loop (optional)
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, y in val_loader:
            (img1, img2) = batch_X
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            # Get model predictions
            feat1, feat2 = model(img1, img2)

            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)

            output = similarity_scores.view(-1)

            # Convert target to Float type
            y = y.float()

            val_loss = criterion(output, y)

            val_losses.append(val_loss.item())
            val_correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            val_total += len(y)

    # Calculate training and validation loss for the epoch
    train_loss = sum(losses) / len(losses)
    val_loss = sum(val_losses) / max(1, len(val_losses))
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model.pth')
# visualize_predictions(model, test_loader)


# # Training Loop with Learning Rate Scheduler
# 
# This code represents the training loop for a Siamese network with a learning rate scheduler. The key emphasis here is on using the `ReduceLROnPlateau` learning rate scheduler.
# 
# Here's a brief overview of the role of the learning rate scheduler in this training loop:
# 
# - The learning rate scheduler is implemented using the `ReduceLROnPlateau` scheduler, which adjusts the learning rate based on a specified metric (in this case, the minimum of the validation loss).
# 
# - The `ReduceLROnPlateau` scheduler is particularly useful when the model's performance on the validation dataset plateaus. If the validation loss stops improving, it reduces the learning rate to help the model fine-tune its weights effectively.
# 
# - The scheduler monitors the validation loss (`mode='min'`) and reduces the learning rate (`factor=0.5`) by half if no improvement is observed for a specified number of epochs (`patience=2`).
# 
# - The rest of the training loop is similar to the previous versions, including model predictions, loss computation, and weight updates.
# 
# - The model can be further evaluated on the validation dataset (validation loop) to monitor its performance during training.
# 
# By using the `ReduceLROnPlateau` learning rate scheduler, this training loop ensures that the model's learning rate is adaptively adjusted based on the validation loss, making the training process more efficient and effective, especially during the fine-tuning phase.
# 

# In[163]:


import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Training loop
# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork_withdropout(embedding_size=128, dropout_prob = 0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

criterion = torch.nn.CrossEntropyLoss()

Prediction_threshold = 0.65

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_X, y in train_loader:
        (img1, img2) = batch_X
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        optimizer.zero_grad()
        # Get model predictions
        feat1, feat2 = model(img1, img2)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

        # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
        similarity_scores = torch.sigmoid(similarity_scores)

        output = similarity_scores.view(-1)

        # Convert target to Float type
        y = y.float()
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
        total += len(y)

    # Validation loop (optional)
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, y in val_loader:
            (img1, img2) = batch_X
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            # Get model predictions
            feat1, feat2 = model(img1, img2)

            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)

            output = similarity_scores.view(-1)

            # Convert target to Float type
            y = y.float()

            val_loss = criterion(output, y)

            val_losses.append(val_loss.item())
            val_correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            val_total += len(y)

    # Calculate training and validation loss for the epoch
    train_loss = sum(losses) / len(losses)
    val_loss = sum(val_losses) / max(1, len(val_losses))
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total
    
    scheduler.step(1 - val_accuracy)  # Adjust LR based on validation loss

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model.pth')
# visualize_predictions(model, test_loader)


# ## Learning Rate Scheduler Analysis
# 
# ### Loss Maintenance and Accuracy
# 
# From the perspective of the loss, the ReduceLROnPlateau scheduler has performed effectively in maintaining a constant decrease in loss. Even though the number of epochs is relatively low, this scheduler has adjusted the learning rate when necessary to ensure the model continues to learn efficiently.
# 
# Both learning rate schedulers provided almost equal accuracy by the end of training. This suggests that the choice of the learning rate scheduler does not significantly impact the final accuracy, especially in this relatively simple training scenario.
# 
# ### Potential Reasons for Better Performance of ReduceLRonPlateau Scheduler
# 
# The better performance of the ReduceLRonPlateau scheduler can be attributed to several factors:
# 
# 1. **Dynamic Learning Rate Adjustment**: The ReduceLRonPlateau scheduler dynamically adjusts the learning rate based on the model's performance on the validation set. If it detects that the model's performance is plateauing (e.g., no further reduction in loss), it reduces the learning rate. This adaptive learning rate control can help the model converge more effectively.
# 
# 2. **Reduced Risk of Overshooting**: Unlike fixed learning rate schedules, the ReduceLRonPlateau scheduler prevents overshooting the minimum of the loss function. It gradually reduces the learning rate, allowing the model to navigate near the optimal solution without oscillating or missing the minimum.
# 
# 3. **Improved Stability**: The ReduceLRonPlateau scheduler helps stabilize the training process. It ensures that small fluctuations in the loss function do not lead to large learning rate adjustments, which can destabilize training.

# # Training Loop with SGD Optimizer
# 
# This code represents the training loop for a Siamese network with an SGD (Stochastic Gradient Descent) optimizer. Here's an overview of the important aspects of this training loop:
# 
# - The code starts by setting the device to CUDA if a CUDA device is available, and the model is moved to the chosen device.
# 
# - The optimizer is configured as an SGD optimizer with a learning rate of 0.0001 and momentum of 0.7. SGD is a popular optimization algorithm that updates model weights in small batches, helping to converge to a minimum of the loss function efficiently.
# 
# - The training loop is similar to previous versions, including model predictions, loss computation, and weight updates.
# 
# - The choice of optimizer (SGD in this case) affects the training dynamics and convergence speed. The optimizer adjusts the model weights to minimize the loss function.
# 
# - The rest of the training loop is consistent with previous versions, including monitoring training and validation loss, and calculating accuracy.
# 
# - The model's training progress is monitored by evaluating its performance on the validation dataset, which is important for tracking overfitting and generalization.
# 
# - The trained model is saved with the name 'siamese_model_sgd.pth' after training.
# 
# The choice of optimizer can significantly impact training results. In this case, SGD with momentum is used, which can be effective in training deep neural networks.
# 

# In[75]:


import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Training loop
# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork_withdropout(embedding_size=128, dropout_prob=0.2).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.7)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

criterion = torch.nn.CrossEntropyLoss()

Prediction_threshold = 0.65

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_X, y in train_loader:
        (img1, img2) = batch_X
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        optimizer.zero_grad()
        # Get model predictions
        feat1, feat2 = model(img1, img2)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

        # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
        similarity_scores = torch.sigmoid(similarity_scores)

        output = similarity_scores.view(-1)

        # Convert target to Float type
        y = y.float()

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
        total += len(y)

    # Validation loop (optional)
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, y in val_loader:
            (img1, img2) = batch_X
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            # Get model predictions
            feat1, feat2 = model(img1, img2)

            # Compute the cosine similarity
            similarity_scores = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-6)

            # Apply sigmoid activation to obtain similarity scores in the range [0, 1]
            similarity_scores = torch.sigmoid(similarity_scores)

            output = similarity_scores.view(-1)

            # Convert target to Float type
            y = y.float()

            val_loss = criterion(output, y)

            val_losses.append(val_loss.item())
            val_correct += torch.count_nonzero(y == (output > Prediction_threshold)).item()
            val_total += len(y)

    # Calculate training and validation loss for the epoch
    train_loss = sum(losses) / len(losses)
    val_loss = sum(val_losses) / max(1, len(val_losses))
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total
    scheduler.step(1 - val_accuracy)  # Adjust LR based on validation loss
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model_sgd.pth')
# visualize_predictions(model, test_loader)


# # Experimenting with Optimizers
# 
# In our experiments, we considered two popular optimizers: Adam and SGD, and we compared their performance in terms of accuracy while using the ReduceLROnPlateau scheduler. Below are the observations and comments on what worked, what didn't, and the potential reasons behind it.
# 
# ## Adam Optimizer
# 
# 1. **What Worked**:
#     - Adam performed significantly better in terms of accuracy. It consistently achieved higher accuracy levels throughout the training process.
#     - The model trained with Adam converged faster and was able to reach a lower training loss.
#     - Adam demonstrated better generalization to the validation dataset, resulting in higher validation accuracy.
# 
# 2. **What Didn't Work**:
#     - While Adam outperformed SGD, it had higher memory and computational requirements due to maintaining separate moving averages and squared gradients for each parameter. This can be a drawback when working with limited computational resources.
# 
# 3. **Potential Reasons**:
#     - Adaptive Learning Rates: Adam adapts learning rates individually for each parameter, which can be particularly effective when dealing with high-dimensional optimization problems. This adaptability ensures faster convergence.
#     - Momentum: The use of momentum in Adam aids in accelerating convergence, especially in complex loss landscapes. This contributes to escaping local minima and achieving better accuracy.
#     - Bias Correction: Adam incorporates a bias correction mechanism, which helps address initialization bias issues. This correction leads to a more accurate gradient estimation, especially in the initial training stages, and contributes to improved model accuracy.
# 
# ## SGD Optimizer
# 
# 1. **What Worked**:
#     - While SGD achieved lower accuracy compared to Adam, it is computationally less intensive. This makes it a suitable choice for scenarios with limited resources.
# 
# 2. **What Didn't Work**:
#     - SGD had slower convergence and often got stuck in local optima, resulting in less accurate models.
#     - It required careful hyperparameter tuning to achieve reasonable accuracy, and even then, it couldn't match the accuracy levels of Adam.
# 
# 3. **Potential Reasons**:
#     - Fixed Learning Rate: SGD uses a fixed learning rate for all parameters. This can be problematic, especially in deep networks with varying gradient magnitudes, leading to slow convergence or overshooting.
# 
# In summary, the experiment indicated that Adam outperformed SGD in terms of accuracy and convergence speed. However, the choice between the two optimizers should also consider computational resources and time constraints. In resource-rich environments, Adam is a compelling choice due to its superior performance.
# 

# In[164]:


visualize_predictions(model, test_loader)


# ### TESTING ON FRIEND'S IMAGES

# In[18]:


model = SiameseNetwork_withdropout(embedding_size=128, dropout_prob=0.8).to(device)
# Load the model weights from the .pth file
model.load_state_dict(torch.load('siamese_model_sgd.pth'))  # Replace with the path to your .pth file


# In[173]:


test_2_images_paths = []
test_2_labels_name = []
test_folder_2 = "./test_2"
# Load images and labels from subfolders
test_subfolders = os.listdir(test_folder_2)
for subfolder in test_subfolders:
    subfolder_path = os.path.join(test_folder_2, subfolder)
    if os.path.isdir(subfolder_path):
        images, labels = load_and_label_images(subfolder_path)
        test_2_images_paths.extend(images)
        test_2_labels_name.extend(labels)

# Define transformations for the images (resize, augment and convert to tensor)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize to a larger size for better cropping
#     transforms.RandomCrop((224, 224)),  # Randomly crop to the desired size
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=5),  # Randomly rotate by up to 15 degrees
    transforms.ToTensor(),
])
        
        
test_2_images = preprocess_and_load_images(test_2_images_paths)
test_2_pairs, test_2_labels = create_random_pairs_and_labels(test_2_images, test_2_labels_name, num_positive_pairs_to_select = 10, num_negative_pairs_to_select = 10)
test_2_dataset = SiameseDataset(test_2_pairs, test_2_labels, transform=transform)
batch_size = 8  # Adjust batch size as needed
test_2_loader = DataLoader(test_2_dataset, batch_size=batch_size, shuffle = True)
criterion = torch.nn.CrossEntropyLoss()
Prediction_threshold = 0.55
visualize_predictions(model, test_2_loader, show_predictions=True)


# ## GAN

# In[49]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt


# In[28]:


# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the GPU
    print("Using GPU:", torch.cuda.get_device_name(0))  # Display GPU name
else:
    device = torch.device("cpu")  # Use the CPU
    print("CUDA is not available. Using CPU.")


# In[29]:


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dset

# Specify the root directory of your LFW dataset
lfw_data_directory = r'D:\OneDrive - Indian Institute of Technology Bombay\Important_Downloads\Semester 7\EE782\Assignment2\lfw\lfw'

new_image_size = 64
dataset = dset.ImageFolder(root=lfw_data_directory,
                           transform=transforms.Compose([
                               transforms.Resize(new_image_size),
                               transforms.CenterCrop(new_image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


# In[30]:


import matplotlib.pyplot as plt
import random

# Define the number of images to display
num_images_to_show = 10 

# Create a figure for displaying the images
fig, image_grid = plt.subplots(1, num_images_to_show, figsize=(15, 5))

# Randomly select and display images
for i in range(num_images_to_show):
    # Randomly choose an image index from the dataset
    random_index = random.randint(0, len(dataset) - 1)
    image, label = dataset[random_index]
    
    # Convert the PyTorch tensor to a NumPy array
    image_array = image.permute(1, 2, 0).numpy()
    
    # If image normalization was applied, denormalize the image
    image_array = 0.5 * image_array + 0.5  # Assuming mean=0.5 and std=0.5
    
    # Display the image
    image_grid[i].imshow(image_array)
    image_grid[i].set_title(f"Label: {label}")

# Adjust the layout and display the figure
plt.tight_layout()
plt.show()


# In[31]:


# Create a DataLoader to efficiently load and iterate through the dataset
batch_size = 64  # Adjust as needed
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)


# In[32]:


get_ipython().system('pip install tqdm')


# In[33]:


dataloader_size = len(dataloader)
print("Size of Dataloader:", dataloader_size)


# I have adpated the structure of the training loop from the following
# https://github.com/pytorch/examples/blob/main/dcgan/main.py

# In[50]:


# Number of available GPUs
num_gpus = 1  # You can set this to the actual number of GPUs if different

# Dimension of the input noise vector
noise_dim = 100 

# Number of generator feature maps
generator_feature_maps = 64 

# Number of discriminator feature maps
discriminator_feature_maps = 64 

# Number of image channels (e.g., 3 for RGB images)
num_channels = 3


# In[51]:


# Custom weights initialization function called on netG and netD
def initialize_weights(module):
    class_name = module.__class__.__name__
    if 'Conv' in class_name:
        # Initialize Convolutional layers with mean 0.0 and standard deviation 0.02
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif 'BatchNorm' in class_name:
        # Initialize BatchNorm layers with weight=1.0 and bias=0.02
        torch.nn.init.normal_(module.weight, mean=1.0, std=0.02)
        torch.nn.init.zeros_(module.bias)


# In[75]:


class Generator(nn.Module):
    def __init__(self, num_gpus):
        super(Generator, self).__init__()
        self.num_gpus = num_gpus
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.num_gpus))
        else:
            output = self.main(input)
        return output


# In[76]:


netG = Generator(num_gpus).to(device)
netG.apply(initialize_weights)


# In[77]:


class Discriminator(nn.Module):
    def __init__(self, num_gpus):
        super(Discriminator, self).__init__()
        self.num_gpus = num_gpus
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.num_gpus))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# In[78]:


netD = Discriminator(num_gpus).to(device)
netD.apply(initialize_weights)


# In[79]:


criterion = nn.BCELoss()  # Binary Cross-Entropy Loss is commonly used for GANs

fixed_noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)  # Use the defined noise_dim
real_label = 1.0  
fake_label = 0.0  
learning_rate = 0.0001  # A common learning rate for GANs
beta1 = 0.7  # Exponential decay rate for the first moment estimates in Adam

# Setup optimizers with adjusted learning rate and beta1
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))


# In[81]:


def train_discriminator(netD, real_images, optimizerD, criterion):
    batch_size = real_images.size(0)
    real_labels = torch.full((batch_size,), 1.0, dtype=real_images.dtype, device=device)
    optimizerD.zero_grad()
    output_real = netD(real_images)
    errD_real = criterion(output_real, real_labels)
    errD_real.backward()
    D_x = output_real.mean().item()
    noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    fake_images = netG(noise)
    real_labels.fill_(0.0)
    output_fake = netD(fake_images.detach())
    errD_fake = criterion(output_fake, real_labels)
    errD_fake.backward()
    errD = errD_real + errD_fake
    optimizerD.step()
    D_G_z1 = output_fake.mean().item()
    return errD, D_x, D_G_z1

def train_generator(netG, optimizerG, criterion):
    optimizerG.zero_grad()
    real_labels.fill_(1.0)
    output_generator = netD(fake_images)
    errG = criterion(output_generator, real_labels)
    errG.backward()
    optimizerG.step()
    D_G_z2 = output_generator.mean().item()
    return errG, D_G_z2

# Training loop
num_epochs = 200
total_batches = len(dataloader)
print("Starting training...")

for epoch in range(num_epochs):
    dataloader_iterator = iter(dataloader)
    
    # Initialize losses and metrics
    total_lossD = 0.0
    total_lossG = 0.0
    total_D_x = 0.0
    total_D_G_z1 = 0.0
    total_D_G_z2 = 0.0
    
    # Use tqdm for progress visualization
    for i, data in enumerate(tqdm(dataloader_iterator, total=total_batches), 0):
        errD, D_x, D_G_z1 = train_discriminator(netD, data[0].to(device), optimizerD, criterion)
        errG, D_G_z2 = train_generator(netG, optimizerG, criterion)

        # Update total losses and metrics
        total_lossD += errD.item()
        total_lossG += errG.item()
        total_D_x += D_x
        total_D_G_z1 += D_G_z1
        total_D_G_z2 += D_G_z2

    # Calculate average losses and metrics for the epoch
    avg_lossD = total_lossD / len(dataloader)
    avg_lossG = total_lossG / len(dataloader)
    avg_D_x = total_D_x / len(dataloader)
    avg_D_G_z1 = total_D_G_z1 / len(dataloader)
    avg_D_G_z2 = total_D_G_z2 / len(dataloader)
    
    # Print progress and losses after each epoch
    print(f'[{epoch}/{num_epochs}] Avg Loss_D: {avg_lossD:.4f} Avg Loss_G: {avg_lossG:.4f} Avg D(x): {avg_D_x:.4f} Avg D(G(z)): {avg_D_G_z1:.4f} / {avg_D_G_z2:.4f}')
    
    if epoch % 1 == 0:
        vutils.save_image(real_images,
                'real_samples.png',
                normalize=True)
        fake_images = netG(fixed_noise)
        vutils.save_image(fake_images.detach(),
                'fake_samples_epoch_%03d.png' % epoch,
                normalize=True)


# In[82]:


from PIL import Image

# Load a saved real image
real_image = Image.open('real_samples.png')

# Load a saved fake image
fake_image = Image.open('fake_samples_epoch_199.png')  # Adjust the epoch number as needed


# In[83]:


import matplotlib.pyplot as plt

# Create a subplot for displaying real and fake images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Real Image')
plt.imshow(real_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Fake Image')
plt.imshow(fake_image)
plt.axis('off')

plt.show()

