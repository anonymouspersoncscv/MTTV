import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to extract features from the model
def extract_features(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu())
            labels.append(targets)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

# Main function
def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    save_path = "tsne_visualization.png"

    # Load pre-trained model and modify for feature extraction
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the classification head
    model.to(device)

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract features and labels
    print("Extracting features...")
    features, labels = extract_features(model, dataloader, device)

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    features_2d = tsne.fit_transform(features)

    # Plot t-SNE
    print("Plotting t-SNE...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter, label='Labels')
    plt.title("2D t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE visualization saved to {save_path}")

if __name__ == "__main__":
    main()

