import torchinfo
import torchvision

import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import model_utils as mu

NUM_EPOCHS = 1000
EARLY_STOP = 20
CLASSES=10
BATCH_SIZE = 64

device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")


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
    validate_loader = DataLoader(parts[1], batch_size=batch_size, shuffle=False, num_workers=1)

    if len(parts) <= 2:
        return train_loader, validate_loader, None

    test_loader = DataLoader(parts[2], batch_size=1, shuffle=False)
    return train_loader, validate_loader, test_loader


# Add training augmentations here, remember: we do not want to transform the validation images.
# For information about augmentation see: https://pytorch.org/vision/stable/transforms.html
train_transformations = v2.AutoAugment() # autoaugment.AutoAugmentPolicy.IMAGENET,
                                     #  autoaugment.InterpolationMode.BILINEAR)


def build_model(model_specification, writer):
    if "simple" in model_specification:
        input_shape = (BATCH_SIZE, 3, 32, 32)
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Tanh() if "Tanh" in model_specification else nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 2048),
            nn.Tanh() if "Tanh" in model_specification else nn.LeakyReLU(),
            nn.Linear(2048, CLASSES),
        )
    elif "complex" in model_specification:
        input_shape = (BATCH_SIZE, 1, 32, 32)
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=2, padding=5),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 10),
        )
    else:
        input_shape = (BATCH_SIZE, 3, 224, 224)
        model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
        if "FE" in model_specification:
            mu.freeze_weights(model)
        model = nn.Sequential(
            model,
            nn.Linear(NUM_EPOCHS, CLASSES),
        )

    torchinfo.summary(model, input_size=input_shape)
    model = model.to(device)
    writer.add_text('model', str(model))
    return model

def build_optimizer(model, optimizer_specification):
    if "SGD" in optimizer_specification:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-5)
    elif "AdamW" in optimizer_specification:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    else:
        raise NotImplementedError("Optimizer is not given")
    return optimizer

m_ft = torch.nn.Sigmoid()
ce_loss_ft = torch.nn.CrossEntropyLoss(reduction='sum')
def criterion_CE(y_est, y_true):
    """CrossEntropyLoss(Sigmoid(y_est), one_hot(y_true))"""
    y_true = nn.functional.one_hot(y_true.long(), num_classes=CLASSES).float().to(device)
    return ce_loss_ft(m_ft(y_est), y_true)

bce_loss_ft = torch.nn.BCEWithLogitsLoss(reduction='sum')
def criterion_BCE(y_est, y_true):
    """BCEWithLogitsLoss(y_est, one_hot(y_true))"""
    y_true = nn.functional.one_hot(y_true.long(), num_classes=CLASSES).float().to(device)
    return bce_loss_ft(y_est, y_true)

# Data loaders
def build_loaders(model_specification, data="CIFAR10", seed=1):
    if "simple" in model_specification:
        input_transformations = v2.Compose([v2.ToImage(),
                                            v2.ToDtype(torch.float32, scale=True)])
    elif "AlexNet" in model_specification:
        input_transformations = torchvision.models.AlexNet_Weights.DEFAULT.transforms()
    elif "complex" in model_specification:
        # model_specification = "MNIST"
        # data = "MNIST" or "SVHN"
        input_transformations = v2.Compose([v2.ToImage(),
                                            v2.Grayscale(),
                                            v2.Resize(32),
                                            v2.ToDtype(torch.float32, scale=True)])
    else:
        raise NotImplementedError("Loader for model: " + model_specification)

    print("Input transform", input_transformations)

    if data == "CIFAR10":
        labeled_images_ = torchvision.datasets.CIFAR10(root='./data',
                                                       train=True,
                                                       download=True,
                                                       transform=input_transformations)
        labeled_test_images_ = torchvision.datasets.CIFAR10(root='./data',
                                                            train=False,
                                                            download=True,
                                                            transform=input_transformations)
    elif data == "MNIST":
        labeled_images_ = torchvision.datasets.MNIST(root='./data',
                                                     train=True,
                                                     download=True,
                                                     transform=input_transformations)
        labeled_test_images_ = torchvision.datasets.MNIST(root='./data',
                                                            train=False,
                                                            download=True,
                                                            transform=input_transformations)
    elif data == "SVHN":
        labeled_images_ = torchvision.datasets.SVHN(root='./data',
                                                    split="train",
                                                    download=True,
                                                    transform=input_transformations)
        labeled_test_images_ = torchvision.datasets.SVHN(root='./data',
                                                         split="test",
                                                         download=True,
                                                         transform=input_transformations)
    else:
        raise NotImplementedError("Data is not given")
    print(labeled_images_)

    torch.manual_seed(seed)
    basic_parts = torch.utils.data.random_split(labeled_images_, [0.80, 0.20])
    print("train and validate images", list(map(len, basic_parts)))

    train_loader, validate_loader, _ = loaders(basic_parts, batch_size=BATCH_SIZE)

    _, _, test_loader = loaders([labeled_test_images_], batch_size=1)

    return train_loader, validate_loader, test_loader

#%%
# Setup
for model_, optimizer_ in [('simple', 'SGD'),
                           ('simple with Tanh', 'SGD'),
                           ('simple', 'AdamW'),
                           ('AlexNet FE', 'AdamW'),
                           ('AlexNet FE', 'SGD'),
                           ('AlexNet FT', 'AdamW'),
                           ('AlexNet FT', 'SGD'),
                           ]:
    writer = SummaryWriter(f"runs/{model_}/{optimizer_}")
    model = build_model(model_, writer)
    train_loader, validate_loader, test_loader = build_loaders(model_)
    optimizer = build_optimizer(model, optimizer_)

    # Train... ... ...
    trained_model, _, _ = mu.train_model(model,
                                         criterion=criterion_BCE,
                                         optimizer=optimizer,
                                         train_loader=train_loader,
                                         train_transformations=train_transformations,
                                         val_loader=validate_loader,
                                         num_epochs=NUM_EPOCHS,
                                         device=device,
                                         early_stop=EARLY_STOP,
                                         writer=writer)
    # Save model
    torch.save(trained_model, f'trained {model_} {optimizer_}.pt')

    # Test
    confusion_matrix, test_loss, losses_when_wrong = mu.evaluate_model(trained_model,
                                                                       # always evaluate with same loss function
                                                                       criterion=criterion_CE,
                                                                       classes=CLASSES,
                                                                       test_loader=test_loader,
                                                                       device=device,
                                                                       writer=writer)
    figure = mu.plot_confusion(confusion_matrix)
    writer.add_figure('confusion_matrix', figure)

    writer.flush()

#%% Test with other dataset
# Prepare a CNN of your choice and train it on the MNIST data. Report the accuracy
writer = SummaryWriter(f"runs/MNIST")
model_ = "complex"
optimizer_ = "AdamW"

model = build_model(model_, writer)
mnist_train_loader, mnist_validate_loader, mnist_test_loader = build_loaders(model_, "MNIST")
optimizer = build_optimizer(model, optimizer_)
mnist_model, _, _ = mu.train_model(model,
                                   criterion=criterion_BCE,
                                   optimizer=optimizer,
                                   train_loader=mnist_train_loader,
                                   train_transformations=train_transformations,
                                   val_loader=mnist_validate_loader,
                                   num_epochs=NUM_EPOCHS,
                                   device=device,
                                   early_stop=EARLY_STOP,
                                   writer=writer,
                                   )
torch.save(mnist_model, f'trained {model_} {optimizer_}.pt')

mnist_model = torch.load(f'trained {model_} {optimizer_}.pt')
confusion_matrix, _, _, = mu.evaluate_model(mnist_model,
                                            criterion=criterion_CE,
                                            classes=CLASSES,
                                            test_loader=mnist_test_loader,
                                            device=device,
                                            writer=writer,
                                            )
figure = mu.plot_confusion(confusion_matrix)
writer.add_figure('confusion_matrix', figure)

#%% Use the above model as a pre-trained CNN for the SVHN dataset. Report the accuracy

mnist_model = torch.load(f'trained {model_} {optimizer_}.pt')
writer = SummaryWriter(f"runs/SVHN with MNIST trained")
svhn_loader, svhn_validate_loader, svhn_test_loader = build_loaders(model_, "SVHN")
confusion_matrix, _, _, = mu.evaluate_model(mnist_model,
                                            criterion=criterion_CE,
                                            classes=CLASSES,
                                            test_loader=svhn_test_loader,
                                            device=device,
                                            writer=writer,
                                            )

figure = mu.plot_confusion(confusion_matrix)
writer.add_figure('confusion_matrix', figure)

writer.flush()

#%% In the third step you are performing transfer learning from MNIST to SVHN (optional)
writer = SummaryWriter(f"runs/SVHN from MNIST trained")
optimizer = build_optimizer(mnist_model, optimizer_)
svhn_model, _, _ = mu.train_model(mnist_model,
                                  criterion=criterion_BCE,
                                  optimizer=optimizer,
                                  train_loader=svhn_loader,
                                  train_transformations=train_transformations,
                                  val_loader=svhn_validate_loader,
                                  num_epochs=NUM_EPOCHS,
                                  device=device,
                                  early_stop=EARLY_STOP,
                                  writer=writer,
                                  )
confusion_matrix, _, _, = mu.evaluate_model(svhn_model,
                                            criterion=criterion_CE,
                                            classes=CLASSES,
                                            test_loader=svhn_test_loader,
                                            device=device,
                                            writer=writer,
                                            )
figure = mu.plot_confusion(confusion_matrix)
writer.add_figure('confusion_matrix', figure)
writer.flush()