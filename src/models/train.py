import pytorch_lightning as pl

from fastapi_service.s3.utils import save_checkpoint
from fastapi_service.utils import get_checkpoint_dir_and_name


def train(
    model: pl.LightningModule,
    epochs: int,
    dataset_folder_name: str,
    model_filename: str,
):
    checkpoint_dir, checkpoint_name = get_checkpoint_dir_and_name(
        dataset_folder_name, model_filename
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        save_top_k=1,
        monitor="Validation loss",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        default_root_dir=checkpoint_dir,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)

    save_checkpoint(
        checkpoint_dir / checkpoint_name, dataset_folder_name, model_filename
    )

    return
