import torchvision

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torchvision import datasets
from torchvision.transforms import v2, autoaugment
from torch.utils.data import DataLoader
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")
print("Execution device", device)

def loaders(parts, batch_size):
    """ One part gives None, None, test

    Two parts gives train, validate, and None

    Three parts gives train, validate, and test

    i.e. always three tuple returned
    """

    if len(parts) == 1:
        test_loader = DataLoader(parts[0], batch_size=1, shuffle=False)
        return None, None, test_loader

    train_loader = DataLoader(parts[0], batch_size=batch_size, shuffle=True)
    print("Train", type(train_loader))
    validate_loader = DataLoader(parts[1], batch_size=batch_size, shuffle=False)

    if len(parts) <= 2:
        return train_loader, validate_loader, None

    test_loader = DataLoader(parts[2], batch_size=1, shuffle=False)
    return train_loader, validate_loader, test_loader

# Add training augmentations here, remember: we do not want to transform the validation images.
# For information about augmentation see: https://pytorch.org/vision/stable/transforms.html
train_transformations = v2.AutoAugment() # autoaugment.AutoAugmentPolicy.IMAGENET,
                                     #  autoaugment.InterpolationMode.BILINEAR)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, early_stop=20):
    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    best_validation_accuracy = 0
    train_loss = []
    val_loss = []
    seen_no_improvements = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # Training
        training_number = 0
        training_loss = 0.0
        training_accuracy = 0.0
        model.eval()
        for image, target in train_loader:
            image, target = train_transformations(image, target)
            target = target.to(device)
            estimate = model.forward(image.to(device))
            guess = torch.argmax(estimate, dim=1)
            training_accuracy += torch.sum(guess == target)
            training_number += target.shape[0]
            loss = criterion(estimate, nn.functional.one_hot(target, 10).type_as(estimate))
            training_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        validation_number = 0
        validation_loss = 0.0
        validation_accuracy = 0.0
        model.train(True)
        for image, target in val_loader:
            estimate = model.forward(image.to(device))
            guess = torch.argmax(estimate, dim=1)
            target = target.to(device)
            validation_accuracy += torch.sum(guess == target)
            validation_number += target.shape[0]
            loss = criterion(estimate, nn.functional.one_hot(target, 10).type_as(estimate))
            validation_loss += loss

        if validation_loss/validation_number < best_loss:
            best_loss = validation_loss/validation_number
            best_model = copy.deepcopy(model)
            best_validation_accuracy = int(validation_accuracy)/validation_number
            best_training_accuracy = int(training_accuracy)/training_number
            print(f"Saving model... validation loss: {best_loss:.4f} ({100*best_validation_accuracy:.1f}%) training ({100*best_training_accuracy:.1f}%)")
            seen_no_improvements = 0

        seen_no_improvements += 1
        if seen_no_improvements >= early_stop:
            print("Early stopping...")
            break

    return best_model, best_loss, best_validation_accuracy, best_training_accuracy

#alexnet_transformations = torchvision.models.AlexNet.DEFAULT.transforms()
basic_transformations = v2.Compose([v2.ToImage(),
                                    v2.ToDtype(torch.float32, scale=True)])

labeled_images_ = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=basic_transformations)
print(labeled_images_)

torch.manual_seed(1)
basic_parts = torch.utils.data.random_split(labeled_images_, [0.80, 0.20])
print("train and validate images", list(map(len, basic_parts)))

train_loader, validate_loader, _ = loaders(basic_parts, batch_size=64)

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=5),
    nn.MaxPool2d(kernel_size=2),
    nn.LeakyReLU(),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10),
).to(device)
print(model)


optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

trained_model, best_loss, best_validation_accuracy, best_training_accuracy = train_model(model,
                                                                                         criterion=criterion,
                                                                                         optimizer=optimizer,
                                                                                         train_loader=train_loader,
                                                                                         val_loader=validate_loader,
                                                                                         num_epochs=1000)

labeled_test_images_ = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=basic_transformations)
_, _, test_loader = loaders(labeled_test_images_)