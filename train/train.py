from os import name
import pytorch_lightning as pl
import sys
from model.models import MetricLearner

if __name__ == "__main__":
    dataset_root = '/home/jszumski/mulran_dataset'
    batch_size = 32
    embedding_size = 256

    tb_logger = pl.loggers.TensorBoardLogger("./lightning_logs", name="logs")

    print("Initiating Training")
    trainer = pl.Trainer(max_epochs=5, gpus=1, progress_bar_refresh_rate=80, logger=tb_logger)
    ckpt_path = "/home/jszumski/RadarRandy/lightning_logs/logs/version_2/checkpoints/epoch=1-step=769.ckpt"
    model = MetricLearner.load_from_checkpoint(ckpt_path)
    # model = MetricLearner(dataset_root, embedding_size, learning_rate=9e-6)
    trainer.fit(model)
    print("Training completed")
