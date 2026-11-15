import os
import math
from datetime import datetime
import warnings
import lightning.pytorch as pl

from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from lightning.pytorch.callbacks import Callback


class LambdaLRByStep(LambdaLR):
    """
    Override of torch.optim.lr_scheduler.LambdaLR to call get_lr()
    with the current step instead of the current epoch.

    Also, the lr is directly computed from the lambda function,
    instead of multiplying the base_lr with the lambda function.
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [lmbda(self._step_count) for lmbda in self.lr_lambdas]
    

def linear_warmup_constant(warmup_steps, warmup_lr, init_lr):
    """
    Linear warmup from warmup_lr to init_lr over warmup_steps, then stay constant over the rest of the training.
    """
    
    def f(current_step):
        if current_step < warmup_steps:
            new_lr = warmup_lr + (init_lr - warmup_lr) * (current_step / warmup_steps)
        else:
            new_lr = init_lr

        return new_lr

    return f


class LinearWarmupConstantLRScheduler(LambdaLRByStep):
    def __init__(
            self,
            optimizer,
            warmup_steps,
            warmup_lr,
            init_lr,
            last_epoch=-1,
            verbose=False
        ):
        super().__init__(optimizer, linear_warmup_constant(warmup_steps, warmup_lr, init_lr), last_epoch, verbose)


class CosineAnnealingLRByStep(LRScheduler):
    """
    Override of CosineAnnealingLR to work based on steps
    """

    def __init__(self, optimizer, num_step, eta_min=0, last_epoch=-1, verbose=False, override_lr=None):
        self.num_step = num_step
        self.eta_min = eta_min
        self.local_step_count = 0
        
        # workaround to resume training from a checkpoint and not use the previous LR
        self.override_lr = override_lr

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        ratio = min(self.local_step_count / self.num_step, 1)

        # temp hack (since we are resuming for previously trained epoch, the LR is automatically updated
        # to the last user LR, which is not what we want.)
        if self.override_lr is not None and self.local_step_count <= 5:
            for i in range(len(self.base_lrs)):
                self.base_lrs[i] = self.override_lr

        ret = [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * ratio)) / 2
                for base_lr in self.base_lrs]
        
        self.local_step_count += 1

        return ret
    

class SaveCheckpointAfterValidation(Callback):
    def __init__(self, training_name, save_dir, save_last_only=True):
        self.training_name = training_name
        self.save_dir = save_dir
        self.save_last_only = save_last_only

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # save a checkpoint after validation
        ckpt_path = "last.ckpt" \
            if self.save_last_only \
            else f"post_val_epoch_{pl_module.current_epoch}_step_{pl_module.global_step}.ckpt"
        
        trainer.save_checkpoint(os.path.join(self.save_dir, ckpt_path))

        # update log file with the datetime and last training step
        with open(f"{self.save_dir}/last_ckpt_info.txt", "w") as f:
            f.write(f"{datetime.now()}\n")
            f.write(f"training: {self.training_name}\n")
            f.write(f"epoch: {pl_module.current_epoch}\n")
            f.write(f"training_step: {pl_module.global_step}\n")


class SaveCheckpointAtEpochEnd(Callback):
    def __init__(self, training_name, save_dir):
        self.training_name = training_name
        self.save_dir = save_dir

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # save a checkpoint at the end of each epoch

        print(f"Saving checkpoint at epoch {pl_module.current_epoch}")
        ckpt_path = f"epoch_{pl_module.current_epoch}.ckpt"
        trainer.save_checkpoint(os.path.join(self.save_dir, ckpt_path))


class LogLastTrainingStep(Callback):
    def __init__(self, training_name, save_dir):
        self.training_name = training_name
        self.log_file = f"{save_dir}/last_ckpt_info.txt"

    def setup(self, trainer, pl_module, stage=None):
        # look up the log file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def on_train_step_end(self, trainer, pl_module, outputs):
       self._log_current_training_state(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_current_training_state(trainer)

    def _log_current_training_state(self, trainer):
        # update log file with the datetime and last training step
        with open(self.log_file, "w") as f:
            f.write(f"{datetime.now()}\n")
            f.write(f"training: {self.training_name}\n")
            f.write(f"epoch: {trainer.current_epoch}\n")
            f.write(f"training_step: {trainer.global_step}\n")


def make_training_dir(root_dir: str, training_name: str) -> str:
    """
    Create a new directory to store training related data for different runs.

    Returns:
        str: The name of the new directory, relative to the root_dir.
    """

    run_name = 0
    while True:
        try:
            os.makedirs(f"{root_dir}/{training_name}/{run_name}")
            break
        except FileExistsError:
            run_name += 1

    return os.path.join(root_dir, training_name, str(run_name))
