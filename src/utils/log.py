import hydra
import logging
import time
import wandb
from codetiming import Timer
from contextlib import ContextDecorator
from datetime import datetime
from functools import wraps
from humanfriendly import format_timespan
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from types import SimpleNamespace


logger = rich_logger = logging.getLogger()
rich_handler = RichHandler(
    rich_tracebacks=False,
    tracebacks_suppress=[hydra],
    console=Console(width=165),
    enable_link_path=False,
)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[rich_handler],
)

# Keep third-party network debug/info logs out of training output.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class ExpLogger:
    """
    Customized Logger that supports:
    - Wandb integration
    - Rich logger console outputs
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_wandb = cfg.use_wandb

        # Initialize rich logger
        self.logger = rich_logger
        self.console = rich_handler.console
        self.logger.setLevel(getattr(logging, cfg.logging.level.upper()))

        self.logger.info("Initializing Logger")
        self.rule = self.console.rule
        self.print = self.console.print

        self.info = self.logger.info
        self.critical = self.logger.critical
        self.warning = self.logger.warning
        self.debug = self.logger.debug
        self.error = self.logger.error
        self.exception = self.logger.exception

        self.results = list()

    # Log functions
    def log(self, *args, level="", **kwargs):
        self.logger.log(getattr(logging, level.upper(), logging.INFO), *args, **kwargs)

    def log_metrics(self, metrics, step=None, level="info", use_pretty_repr=False):
        """Logs metrics to Wandb if enabled, otherwise to stdout.
           Also saves metrics in self.results.
        """
        if self.use_wandb:
            wandb.log(metrics, step=step)
        if not self.use_wandb or self.cfg.logging.log_wandb_metric_to_stdout:
            self.log(pretty_repr(metrics) if use_pretty_repr else metrics, level=level)
        # Update self.results with the new metrics.
        self.results.append(metrics)

    def load_and_log_previous_metrics(self, previous_results, wandb_log=True):
        """
        Loads previous metrics results and logs them.
        If wandb_log is True, previous results are also logged to Wandb step-by-step.
        """
        if len(self.results):
            self.warning(f'The current result is {self.results}. Overwriting with previous results.')
        self.results = previous_results

        if wandb_log and self.use_wandb and self.results:
            max_steps = self.cfg.get('max_steps', 1e9)
            for metrics in self.results:
                if metrics.get('step', 0) > max_steps:
                    break
                wandb.log(metrics)
            self.critical(f'Resumed from previous checkpoints {self.results[-1]}')

    def wandb_config_update(self, config_updates):
        """Updates the Wandb config and optionally finishes the run."""
        if self.use_wandb:
            wandb.config.update(config_updates, allow_val_change=True)

    def save_file_to_wandb(self, file, base_path, policy="now", **kwargs):
        """Saves a file to Wandb storage."""
        if self.use_wandb:
            wandb.save(file, base_path=base_path, policy=policy, **kwargs)

    @property
    def experiment(self):
        """Returns the Wandb experiment object."""
        if not hasattr(self, "_experiment"):
            if self.use_wandb:
                self._experiment = wandb.run
            else:
                self._experiment = SimpleNamespace(log=self.logger.info, id=self.cfg.uid, name=self.cfg.alias)
        return self._experiment

def get_cur_time(timezone=None, t_format="%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(int(time.time()), timezone).strftime(t_format)


class timer(ContextDecorator):
    def __init__(self, name=None, log_func=logger.info):
        self.name = name
        self.log_func = log_func
        self.timer = Timer(name=name, logger=None)  # Disable internal logging

    def __enter__(self):
        self.timer.start()
        self.log_func(f"Started {self.name} at {get_cur_time()}")
        return self

    def __exit__(self, *exc):
        elapsed_time = self.timer.stop()
        formatted_time = format_timespan(elapsed_time)
        self.log_func(
            f"Finished {self.name} at {get_cur_time()}, running time = {formatted_time}."
        )
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator
