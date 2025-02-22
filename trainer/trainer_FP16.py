
from ctypes import Union
from typing import Any
import torch

from base.base_trainer import BaseTrainer
from tqdm import tqdm
from torch.cuda.amp import autocast
from logger.pbar import PBar
from typing import Dict, Union

# Added from here by jimmy
import gc
import time
# Added till here by jimmy

from torch.cuda.amp import GradScaler, autocast

class Trainer(BaseTrainer):
    def __init__(self, 
                dist,
                rank,
                n_gpus,
                config,
                resume,
                preload,
                epochs,
                steps_per_epoch,
                model,
                compute_metric,
                processor,
                train_dl,
                val_dl,
                train_sampler,
                val_sampler,
                optimizer,
                scheduler,
                save_dir,
                log_dir,
                gradient_accumulation_steps,
                use_amp,
                max_clip_grad_norm
                ):
        super(Trainer, self).__init__(
                                        dist, 
                                        rank, 
                                        config,
                                        resume, 
                                        preload, 
                                        epochs, 
                                        steps_per_epoch,
                                        model, 
                                        processor,
                                        train_dl,
                                        val_dl,
                                        train_sampler,
                                        val_sampler,
                                        optimizer, 
                                        scheduler,
                                        save_dir, 
                                        log_dir,
                                        use_amp,
                                        gradient_accumulation_steps
                                        )
        self.compute_metric = compute_metric
        self.sr = config["meta"]["sr"]
        self.n_gpus = n_gpus
        self.max_clip_grad_norm = max_clip_grad_norm
        self.stateful_metrics = ["train_loss", "train_lr", "train_grad_norm", "train_wer", "val_loss", "val_wer"]
        self.scaler = GradScaler()  # Initialize GradScaler for FP16

    def get_grad_norm(self, params, scale=1) -> torch.tensor:
        """Compute the norm of gradients across all model parameters."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = (p.grad.detach().data / scale).norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5 if total_norm > 0 else 0.0  # Handle case where no gradients exist
        return total_norm

    def _train_epoch(self, epoch) -> None:
        self.train_sampler.set_epoch(epoch)
        if self.rank == 0:
            print(f"Epoch {epoch+1}: ")
            gc.collect()
            torch.cuda.empty_cache()
            pbar = PBar(self.steps_per_epoch, 10, stateful_metrics=self.stateful_metrics)

        for dl_step, batch in enumerate(self.train_dl):
            with autocast(enabled=self.use_amp):
                # Forward pass with FP16
                self.model.train()
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps

            # Backward pass with GradScaler
            self.scaler.scale(loss).backward()
            wer = torch.tensor(self.compute_metric(outputs.logits.detach().float(), batch['labels']))

            if (dl_step + 1) % self.gradient_accumulation_steps == 0 or dl_step == len(self.train_dl) - 1:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Logging and metrics
                if self.n_gpus > 1:
                    loss = self.gather(loss).mean()
                    wer = self.gather(wer).mean()

                grad_norm = self.get_grad_norm(self.model.parameters())
                grad_norm = grad_norm if grad_norm is not None else 0.0  # Handle invalid grad_norm

                train_logs = {
                    "loss": loss.item() * self.gradient_accumulation_steps,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "grad_norm": grad_norm,
                    "wer": wer.item()
                }

                if self.rank == 0:
                    self.writer.update(self.completed_steps, 'Train', train_logs)

                    # Validate and update progress bar
                    if all(isinstance(v, (int, float)) for v in train_logs.values()):
                        pbar.update(self.pbar_step + 1, "train_", train_logs)
                    else:
                        print("Skipping progress bar update due to invalid values in train_logs.")

                self.completed_steps += 1
        self.pbar_step = 0

    def _valid_epoch(self, step) -> Dict[str, Union[Any, float]]:
        self.val_sampler.set_epoch(step)
        val_logs = {"loss": 0, "wer": 0}

        for batch in tqdm(self.val_dl, total=len(self.val_dl), disable=not self.rank == 0):
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    outputs = self.model(**batch)

            val_logs["loss"] += outputs.loss / len(self.val_dl)
            val_logs["wer"] += torch.tensor(self.compute_metric(outputs.logits.float(), batch['labels'])) / len(self.val_dl)

        if self.n_gpus > 1:
            val_logs = {k: self.gather(v).mean() for k, v in val_logs.items()}

        return {k: v.item() if hasattr(v, 'item') else v for k, v in val_logs.items()}
