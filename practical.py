import torchvision
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torchvision import datasets
from torchvision.transforms import v2, autoaugment
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import model_utils as mu

#experiment = " simple CNN network with AdamW optimizer"
experiment = " fine tune Alex network with AdamW optimizer"
#experiment = " fine tune Alex network with SGD optimizer"
writer = SummaryWriter(comment=experiment)

CLASSES=10
BATCH_SIZE = 64

device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")
writer.add_text('device', f"Execution device {device}")

def loaders(parts, batch_size):
    """ One part gives None, None, test

    Two parts gives train, validate, and None

    Three parts gives train, validate, and test

    i.e. always three tuple returned
    """

    if len(parts) == 1:
        test_loader = DataLoader(parts[0], batch_size=1, shuffle=False)
        return None, None, test_loader

    train_loader = DataLoader(parts[0], batch_size=batch_size, shuffle=True, num_workers=1)
    print("Train", type(train_loader))
    validate_loader = DataLoader(parts[1], batch_size=batch_size, shuffle=False, num_workers=1)

    if len(parts) <= 2:
        return train_loader, validate_loader, None

    test_loader = DataLoader(parts[2], batch_size=1, shuffle=False)
    return train_loader, validate_loader, test_loader


# Add training augmentations here, remember: we do not want to transform the validation images.
# For information about augmentation see: https://pytorch.org/vision/stable/transforms.html
train_transformations = v2.AutoAugment() # autoaugment.AutoAugmentPolicy.IMAGENET,
                                     #  autoaugment.InterpolationMode.BILINEAR)



if "simple" in experiment:
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=5),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh() if "Tanh" in experiment else nn.LeakyReLU(),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 2048),
        nn.Tanh() if "Tanh" in experiment else nn.LeakyReLU(),
        nn.Linear(2048, CLASSES),
    )
    input_transformations = v2.Compose([v2.ToImage(),
                                        v2.ToDtype(torch.float32, scale=True)])

else:
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    if "FE" in experiment:
        mu.freeze_weights(model)
    model = nn.Sequential(
        model,
        nn.Linear(1000, CLASSES),
    )
    input_transformations = torchvision.models.AlexNet_Weights.DEFAULT.transforms()

model = model.to(device)
print(model)
writer.add_text('model', str(model))

if "SGD" in experiment:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-5)
elif "AdamW" in experiment:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
else:
    raise NotImplementedError("Optimizer is not given")

#m_ft = torch.nn.Sigmoid()
#loss_ft = torch.nn.CrossEntropyLoss(reduction='sum')
loss_ft = torch.nn.BCEWithLogitsLoss()
def criterion(y_est, y_true):
    """CrossEntropyLoss(Sigmoid(y_est), one_hot(y_true))"""
    y_true = nn.functional.one_hot(y_true.long(), num_classes=CLASSES).float().to(device)
#    result = loss_ft(m_ft(y_est),
#                     y_true)
    result = loss_ft(y_est,
                     y_true)
    return result

# Data loaders
labeled_images_ = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=input_transformations)
print(labeled_images_)

torch.manual_seed(1)
basic_parts = torch.utils.data.random_split(labeled_images_, [0.80, 0.20])
print("train and validate images", list(map(len, basic_parts)))

train_loader, validate_loader, _ = loaders(basic_parts, batch_size=BATCH_SIZE)

# Train... ... ...
mu.train_model(model,
               criterion=criterion,
               optimizer=optimizer,
               train_loader=train_loader,
               train_transformations=train_transformations,
               val_loader=validate_loader,
               num_epochs=1000,
               device=device,
               early_stop=100,
               writer=writer)

#%% Test model
labeled_test_images_ = torchvision.datasets.CIFAR10(root='./data',
                                                    train=False,
                                                    download=True,
                                                    transform=input_transformations)
_, _, test_loader = loaders([labeled_test_images_], batch_size=1)

trained_model = torch.load('best_model.pt')
confusion_matrix, test_loss, losses_when_wrong = mu.evaluate_model(trained_model,
                                                                   criterion=criterion,
                                                                   classes=CLASSES,
                                                                   test_loader=test_loader,
                                                                   device=device,
                                                                   writer=writer)
print(confusion_matrix, test_loss, losses_when_wrong)
print(f"Test {100 * float(sum(torch.diagonal(confusion_matrix, 0)) / torch.sum(confusion_matrix)):.1f}%, loss={test_loss}")
figure = mu.plot_confusion(confusion_matrix)
writer.add_figure('confusion_matrix', figure)

writer.flush()
