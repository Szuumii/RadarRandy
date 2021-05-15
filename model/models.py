import pytorch_lightning as pl
from torchvision import models, transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
from dataset.MulRan import MulRanDataset
from dataset.sampler import BatchSampler
from pytorch_metric_learning import losses, reducers
from model.miners import HardTripletMiner
from pytorch_metric_learning.distances import LpDistance



class MetricLearner(pl.LightningModule):
    def __init__(self, dataset_root, embeding_size=256, margin=0.4, batch_size=32, small_train=False, learning_rate= 1e-6):
        super().__init__()

        self.save_hyperparameters()
        self.small_train = small_train
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.margin = margin
        self.net = models.resnet34(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, embeding_size)

        self.miner = HardTripletMiner(distance=LpDistance(power=2))
        self.reducer = reducers.DoNothingReducer(collect_stats=True)
        self.loss = losses.TripletMarginLoss(margin=self.margin)

        self.learning_rate = learning_rate

    @pl.core.decorators.auto_move_data
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        embeddings = self.forward(x)
        positive_mask, negative_mask = self.get_masks_for_batch(
            self.train_dataset, labels)
        triplets = self.miner(embeddings, positive_mask, negative_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss(embeddings, dummy_labels, triplets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_num_non_zero_triplets', loss.reducer.triplets_past_filter, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        embeddings = self.forward(x)
        positive_mask, negative_mask = self.get_masks_for_batch(
            self.val_dataset, labels)
        triplets = self.miner(embeddings, positive_mask, negative_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss(embeddings, dummy_labels, triplets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([400, 400])])
        # test_transform = transforms.ToTensor()

        train_querries_filename = "train_processed_queries.pickle"
        val_querries_filename = "val_processed_queries.pickle"

        if self.small_train is True:
            train_querries_filename = "small_" + train_querries_filename
            val_querries_filename = "small_" + val_querries_filename

        train_queries_filepath = os.path.join(
            self.dataset_root, train_querries_filename)
        val_querries_fielpath = os.path.join(
            self.dataset_root, val_querries_filename)

        # test_queries_filepath = os.path.join(dataset_root, "test_processed_queries.pickle")
        self.train_dataset = MulRanDataset(
            self.dataset_root, train_queries_filepath, train_transform)

        self.val_dataset = MulRanDataset(
            self.dataset_root, val_querries_fielpath, train_transform)

        print("train = ", end="")
        self.train_dataset.print_info()

        print("val = ", end="")
        self.val_dataset.print_info()

        self.train_sampler = BatchSampler(self.train_dataset, self.batch_size)
        self.val_sampler = BatchSampler(self.val_dataset, self.batch_size)
        # self.samplers["test"] = BatchSampler(self.datasets["test"], self.batch_size)

    def get_masks_for_batch(self, dataset, labels):
        positive_mask = []
        negative_mask = []

        for label in labels:
            label_positives = dataset.get_positives(label.item())
            label_negatives = dataset.get_negatives(label.item())
            positives = []
            negatives = []
            for other_label in labels:
                if other_label.item() in label_positives:
                    positives.append(True)
                    negatives.append(False)
                elif other_label.item() in label_negatives:
                    negatives.append(True)
                    positives.append(False)
                else:
                    positives.append(False)
                    negatives.append(False)

            positive_mask.append(positives)
            negative_mask.append(negatives)

        return torch.tensor(positive_mask), torch.tensor(negative_mask)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=24)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=24) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    print("Succesfull imports")