import pytorch_lightning as pl
from model.model import MetricLearner


if __name__ == "__main__":
    batch_size = 32
    embedding_size = 128

    trainer = pl.Trainer(max_epochs=10, gpus=1, progress_bar_refresh_rate=20)
    model = MetricLearner(dataset_root, embedding_size, 0.4, batch_size)
