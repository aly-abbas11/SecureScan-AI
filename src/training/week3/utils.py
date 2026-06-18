
import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    path
):

    torch.save({

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict()

    }, path)

    print(f"Checkpoint saved to {path}")
