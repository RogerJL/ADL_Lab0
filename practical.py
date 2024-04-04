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
import matplotlib.pyplot as plt

CLASSES=10

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
            loss = criterion(estimate, target)
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
            loss = criterion(estimate, target)
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

def plot_confusion(cm, ax=None):
    if ax is None:
        ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='jet');  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('True labels');ax.set_ylabel('Predicted labels');
    ax.set_title('Confusion Matrix');
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    ax.xaxis.set_ticklabels(labels);
    ax.yaxis.set_ticklabels(labels);
    plt.pause(0.001)

def evaluate_model(model, criterion, test_loader):
    total_loss = 0
    confusion_matrix = torch.zeros((CLASSES, CLASSES)).to(device)
    model.to(device)
    model.eval()
    losses_when_wrong = []
    for index, (image, target) in enumerate(test_loader):
        outv = model.forward(image.to(device))
        target = target.to(device)
        guess = torch.argmax(outv).to(device)
        confusion_matrix[guess, target] += 1
        loss = criterion(outv, target)
        if guess != target:
            losses_when_wrong.append((loss.item(), index))
        total_loss += loss
    losses_when_wrong = sorted(losses_when_wrong, reverse=True)
    return confusion_matrix.cpu(), total_loss.cpu() / len(test_loader), losses_when_wrong

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
    nn.Linear(64 * 8 * 8, 2048),
    nn.LeakyReLU(),
    nn.Linear(2048, CLASSES),
).to(device)
print(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

m_ft = torch.nn.Sigmoid()
loss_ft = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
def criterion(y_est, y_true):
    y_true = nn.functional.one_hot(y_true.long(), num_classes=CLASSES).float().to(device)
    result = loss_ft(m_ft(y_est),
                     y_true)
    return result

trained_model, best_loss, best_validation_accuracy, best_training_accuracy = train_model(model,
                                                                                         criterion=criterion,
                                                                                         optimizer=optimizer,
                                                                                         train_loader=train_loader,
                                                                                         val_loader=validate_loader,
                                                                                         num_epochs=1000)
# Test model
labeled_test_images_ = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=basic_transformations)
_, _, test_loader = loaders([labeled_test_images_], batch_size=1)

confusion_matrix, test_loss, losses_when_wrong = evaluate_model(trained_model,
                                                                criterion=criterion,
                                                                test_loader=test_loader)
print(confusion_matrix, test_loss, losses_when_wrong)
print(f"Test {100 * float(sum(torch.diagonal(confusion_matrix, 0)) / torch.sum(confusion_matrix)):.1f}%, loss={test_loss}")
plot_confusion(confusion_matrix)