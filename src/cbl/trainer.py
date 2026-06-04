import torch
import lightning as L
from lightning.pytorch.cli import LightningCLI

from cbl.data.ir_dataset import IRDataModule

def mute_console_noise():
    import logging
    import warnings

    # Disable pytree deprecation warnings introduced by Pytorch Lightning
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Disable Pytorch Lightning tips until 2.6.5 releases
    class TipFilter(logging.Filter):
        def filter(self, record):
            return "💡 Tip" not in record.getMessage()

    logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(TipFilter())

def cli_main():
    mute_console_noise()
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(datamodule_class=IRDataModule)

if __name__ == "__main__":
    cli_main()
