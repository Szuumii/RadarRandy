import pytorch_lightning as pl
import sys
from models import MetricLearner

if __name__ == "__main__":
    dataset_root = '/home/jszumski/mulran_dataset'
    batch_size = 32
    embedding_size = 256

    print("Initiating Training")
    trainer = pl.Trainer(max_epochs=20, gpus=1, progress_bar_refresh_rate=20)
    model = MetricLearner(dataset_root, embedding_size)
    trainer.fit(model)
    print("Training completed")
