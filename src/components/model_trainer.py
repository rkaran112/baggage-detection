import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm  # Import tqdm for progress bars

class ModelTrain:
    def __init__(self, train_loader, test_loader, valid_loader, num_classes=5, learning_rate=0.001):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_path = "model_weights.pth"  # Specify the file path for saving the model

        # Initialize the model, loss function, and optimizer
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, self.num_classes)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)

    def _to_device(self, images, targets):
        # squeeze(1) only, not squeeze(): a size-1 final batch would otherwise
        # also collapse the batch dimension, turning labels into a 0-d scalar.
        return images.to(self.device), targets['labels'].to(self.device).squeeze(1)

    def train_model(self, num_epochs=20, patience=5):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.model.train()  # validate_model() leaves the model in eval mode
            running_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

            for batch_idx, (images, targets) in progress_bar:
                images, labels = self._to_device(images, targets)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Update the progress bar with the current loss
                progress_bar.set_postfix(loss=loss.item())

            average_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Completed. Average Loss: {average_loss:.4f}")

            # Validate the model
            val_loss = self.validate_model()

            # Step the scheduler
            self.scheduler.step(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_model(self.file_path)  # Save the best model
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    def validate_model(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.valid_loader, desc='Validating', leave=False)

            for images, targets in progress_bar:
                images, labels = self._to_device(images, targets)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_val_loss = val_loss / len(self.valid_loader)
        accuracy = 100 * correct / total



        print(f"Validation Loss: {average_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return average_val_loss 

        
    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    
    def test_model(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='Testing', leave=False)

            for images, targets in progress_bar:
                images, labels = self._to_device(images, targets)

                # Forward pass
                outputs = self.model(images)

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy