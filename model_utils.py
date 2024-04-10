import copy
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt

def freeze_weights(model_: nn.Module):
    for param in model_.parameters(recurse=True):
        param.requires_grad = False

def unfreeze_weights(model_: nn.Module):
    for param in model_.parameters(recurse=True):
        param.requires_grad = True

def freeze_running_stats(model_: nn.Module):
    def disable_running_stat(m: nn.Module) -> None:
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
    model_.apply(disable_running_stat)

def unfreeze_running_stats(model_: nn.Module):
    def disable_running_stat(m: nn.Module) -> None:
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
    model_.apply(disable_running_stat)

def describe_loader(loader):
    return f"{len(loader)}/{loader.batch_size}"

def train_model(model:nn.Module,
                criterion,
                optimizer,
                train_loader, train_transformations,
                val_loader,
                num_epochs,
                device,
                early_stop=20,
                writer=None) -> tuple[nn.Module, float, float]:
    if writer is not None:
        writer.add_text('device', f"Execution device {device}")

        writer.add_hparams({'optimizer': str(optimizer),
                            'criterion': criterion.__doc__,
                            'train_loader': describe_loader(train_loader),
                            'val_loader':describe_loader(val_loader),
                            'num_epochs': num_epochs,
                            'early_stop': early_stop,
                            },
                           {})

    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    best_validation_accuracy = 0.0
    train_loss = []
    val_loss = []
    seen_no_improvements = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # Training
        print(" - Training")
        training_number = 0
        training_loss = 0.0
        training_accuracy = 0.0
        model.train(True)
        for image, target in train_loader:
            print(image.shape)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                image, target = train_transformations(image, target)
                image = image.to(device)
                target = target.to(device)
                estimate = model.forward(image)
                del image
                guess = torch.argmax(estimate, dim=1)
                training_accuracy += torch.sum(guess == target)
                training_number += target.shape[0]
                loss = criterion(estimate, target)
                training_loss += loss.item()

            # Do not autocast during backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if writer is not None:
            writer.add_scalar('training_loss', training_loss / training_number, epoch + 1)
            writer.add_scalar('training_accuracy', training_accuracy / training_number, epoch + 1)

        # Validation
        print(" - Validation")
        validation_number = 0
        validation_loss = 0.0
        validation_accuracy = 0.0
        model.eval()
        for image, target in val_loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                image = image.to(device)
                estimate = model.forward(image)
                del image
                guess = torch.argmax(estimate, dim=1)
                target = target.to(device)
                validation_accuracy += torch.sum(guess == target)
                validation_number += target.shape[0]
                loss = criterion(estimate, target)
                validation_loss += loss.item()
        if writer is not None:
            writer.add_scalar('validation_loss', validation_loss / validation_number, epoch + 1)
            writer.add_scalar('validation_accuracy', validation_accuracy / validation_number, epoch + 1)

        if validation_loss/validation_number < best_loss:
            best_loss = validation_loss/validation_number
            best_model = copy.deepcopy(model)
            best_validation_accuracy = int(validation_accuracy)/validation_number
            best_training_accuracy = int(training_accuracy)/training_number
            torch.save(model, 'best_model.pt')
            print(f"Saving model... validation loss: {best_loss:.4f} ({100*best_validation_accuracy:.1f}%) training ({100*best_training_accuracy:.1f}%)")
            seen_no_improvements = 0

        seen_no_improvements += 1
        if seen_no_improvements >= early_stop:
            if writer is not None:
                writer.add_text('early_stopping', "Early stopping", epoch + 1)
            print("Early stopping...")
            break

        if writer is not None:
            writer.flush()

    return best_model, best_loss, best_validation_accuracy

def plot_confusion(cm, ax=None):
    if ax is None:
        ax= plt.subplot()
    image = sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='jet')  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('True labels');ax.set_ylabel('Predicted labels');
    ax.set_title('Confusion Matrix');
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    ax.xaxis.set_ticklabels(labels);
    ax.yaxis.set_ticklabels(labels);
    plt.pause(0.001)
    return image.get_figure()

def evaluate_model(model:nn.Module, criterion, classes, test_loader, device, writer:SummaryWriter=None):
    total_loss = 0
    confusion_matrix = torch.zeros((classes, classes)).to(device)
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
    if writer is not None:
        writer.add_scalar('test_loss', total_loss.cpu() / len(test_loader))

    print("losses when wrong", losses_when_wrong)

    correct = sum(torch.diagonal(confusion_matrix, 0))
    test_loss = total_loss.cpu() / len(test_loader)
    print(f"Test {100 * float(correct / torch.sum(confusion_matrix)):.1f}%, loss={test_loss}")

    return confusion_matrix.cpu(), test_loss, losses_when_wrong
