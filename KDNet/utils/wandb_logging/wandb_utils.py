# utils/wandb_logging/wandb_utils.py


class WandbLogger:
    def __init__(self, *args, **kwargs):
        self.wandb = None  # so wandb_logger.wandb won't break
        self.wandb_run = None  # so wandb_logger.wandb_run won't break
        self.current_epoch = 0
        self.data_dict = {}

    def log(self, data):
        pass  # do nothing

    def end_epoch(self, best_result=False):
        pass

    def finish_run(self):
        pass

    def log_model(self, *args, **kwargs):
        pass


def check_wandb_resume(opt):
    # always return False or None if you're not using W&B
    return False
