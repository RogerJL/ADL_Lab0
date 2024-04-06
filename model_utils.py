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
