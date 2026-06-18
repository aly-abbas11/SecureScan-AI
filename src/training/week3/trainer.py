
import torch

from tqdm import tqdm

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        device
    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.scheduler = scheduler

        self.device = device

        self.scaler = GradScaler()

    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0

        for batch in tqdm(dataloader):

            input_ids = batch["input_ids"].to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)

            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision
            with autocast():

                outputs = self.model(
                    input_ids,
                    attention_mask
                )

                loss = self.criterion(
                    outputs,
                    labels
                )

            # Backpropagation
            self.scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0
            )

            self.scaler.step(self.optimizer)

            self.scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        return avg_loss


    def validate(self, dataloader):

        self.model.eval()

        total_loss = 0

        correct = 0

        total = 0

        with torch.no_grad():

            for batch in dataloader:

                input_ids = batch["input_ids"].to(self.device)

                attention_mask = batch["attention_mask"].to(self.device)

                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask
                )

                loss = self.criterion(
                    outputs,
                    labels
                )

                total_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)

        accuracy = correct / total

        return avg_loss, accuracy
