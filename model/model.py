import pytorch_lightning as pl
from torchvision import models
import torch.nn as nn
from pytorch_metric_learning import losses


class MetricLearner(pl.LightningModule):
    def __init__(self, dataset_root, embeding_size=256, margin=0.4, batchsize=32):
        super().__init__()

        self.save_hyperparameters()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.margin = margin
        self.net = models.resnet34(pretrained=True)
        self.net.fc = nn.Linear(net.fc.in_features, embeding_size)

        self.miner = HardTripletMiner(distance=LpDistance(power=2))
        self.loss = losses.TripletMarginLoss(margin=self.margin)

    @pl.core.decorators.auto_move_data
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        embeddings = self.forward(x)
        positive_mask, negative_mask = get_masks_for_batch(
            self.train_dataset, labels)
        triplets = self.miner(embeddings, positive_mask, negative_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss(embeddings, labels, triplets)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def prepare_data(self):
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([400, 400])])
        # test_transform = transforms.ToTensor()
        train_queries_filepath = os.path.join(
            dataset_root, "train_processed_queries.pickle")
        # test_queries_filepath = os.path.join(dataset_root, "test_processed_queries.pickle")
        self.train_dataset = MulRanDataset(
            dataset_root, train_queries_filepath, train_transform)

        print("train = ", end="")
        self.train_dataset.print_info()

        # self.datasets["test"] = MulRanDataset(dataset_root, test_queries_filepath, test_transform)

        self.train_sampler = BatchSampler(self.train_dataset, self.batch_size)
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
        dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        next(iter(dataloader))
        return dataloader

    def show_dataloader(self):
        self.train_dataset.print_info()
        dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
