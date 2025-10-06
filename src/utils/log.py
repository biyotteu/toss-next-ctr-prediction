from rich.console import Console
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
import csv, os, time

class Logger:
    def __init__(self, log_dir: str, tb: bool=True, csv_log: bool=True):
        self.console = Console()
        self.tb = SummaryWriter(log_dir) if tb else None
        self.csv_path = os.path.join(log_dir, "train_log.csv") if csv_log else None
        if self.csv_path:
            os.makedirs(log_dir, exist_ok=True)
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(["time","fold","epoch","split","loss","AP","WLL","Score","lr","bs","K","tau"])

    def scalars(self, tag, step, **kwargs):
        if self.tb:
            for k,v in kwargs.items():
                self.tb.add_scalar(f"{tag}/{k}", v, step)

    def row(self, **kwargs):
        msg = "  ".join(f"{k}={v}" for k,v in kwargs.items())
        self.console.print(msg)

    def csv(self, **kwargs):
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                row = [time.strftime("%Y-%m-%d %H:%M:%S")]
                for k in ["fold","epoch","split","loss","AP","WLL","Score","lr","bs","K","tau"]:
                    row.append(kwargs.get(k,""))
                csv.writer(f).writerow(row)
