import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class MulRanDataset(Dataset):
    def __init__(self, dataset_path, queries_filepath, transform=None, test=False):
        self.dataset_path = dataset_path
        self.queries_filepath = queries_filepath
        self.transform = transform
        self.test = test
        self.queries = self.load_queries_file(self.queries_filepath)

    def __len__(self):
        return len(self.queries)

    def load_queries_file(self, queries_filepath):
        print(f"Loading file {queries_filepath}")
        assert os.path.exists(
            self.queries_filepath), f"Cannot access queries file {queries_filepath}"
        queries = {}
        with open(queries_filepath, 'rb') as handle:
            queries = pickle.load(handle)

        if self.test is False:
            for idx in queries:
                queries[idx]["positives"] = np.where(
                    np.array(queries[idx]["positives"]) == 1)[0]
                queries[idx]["negatives"] = np.where(
                    np.array(queries[idx]["negatives"]) == 1)[0]

        return queries

    def __getitem__(self, idx):
        query_filename = self.queries[idx]["query"]
        query_image = self.load_picture(query_filename)
        query_image = query_image.convert("RGB")
        if self.transform is not None:
            query_tensor = self.transform(query_image)

        if self.test is False:
            return query_tensor, idx
        else:
            return query_tensor, idx, self.queries[idx]["position"]

    def load_picture(self, query_filename):
        file_path = os.path.join(
            self.dataset_path, "images", str(query_filename) + ".png")
        picture = Image.open(file_path)
        return picture

    def get_positives(self, idx):
        if self.test is False:
            return self.queries[idx]["positives"]
        else:
            return None

    def get_negatives(self, idx):
        if self.test is False:
            return self.queries[idx]["negatives"]
        else:
            return None

    def print_info(self):
        print(f"Dataset contains {len(self.queries)} queries")


if __name__ == "__main__":
    dataset_path = '/home/jszumski/mulran_dataset'
    queries_file = '/home/jszumski/mulran_dataset/train_processed_queries.pickle'
    # queries_file = '/home/jszumski/mulran_dataset/Sejong02_test_processed_queries.pickle'
    # queries_file = '/home/jszumski/mulran_dataset/Sejong01_test_processed_queries.pickle'

    transform = transforms.ToTensor()

    dataset = MulRanDataset(dataset_path, queries_file, transform)

    dataset.print_info()
