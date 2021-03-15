import os
import pickle
import numpy as np
import random
import torch
from torch.utlis.data import Dataset
import torchvision.transforms as transforms
from PIL import Image



class MulRanDataset(Dataset):
    def __init__(self, dataset_path, queries_filepath, transform=transforms.ToTensor()):
        self.dataset_path = dataset_path
        self.queries_filepath = queries_filepath
        self.transform = transform
        self.queries = self.load_queries_file(self.queries_filepath)

    def __len__():
        return len(self.queries)

    def load_queries_file(self, queries_filepath):
        print(f"Loading file {queries_filepath}")
        assert os.path.exists(self.queries_filepath), f"Cannot access queries file {queries_filepath}"
        queries = {}
        with open(queries_filepath, 'rb') as handle:
            queries = pickle.load(handle)

        for idx in queries:
            queries[idx]["positives"] = [ndx for ndx, pos_elem in enumerate(queries[idx]["positives"]) if pos_elem == 1]
            queries[idx]["negatives"] = [ndx for ndx, neg_elem in enumerate(queries[idx]["positives"]) if neg_elem == 1]

        return queries

    def __getitem__(self, idx):
        query_filename = self.queries[idx]["query"]
        query_image = self.load_picture(query_filename)
        query_tensor = self.transform(query_image)
        return query_tensor, idx
    
    def load_picture(self, query_filename):
        file_path = os.path.join(self.dataset_path, "polar", str(query_filename) + ".png")
        picture = Image.open(file_path)
        return picture
    
    def get_positives(self, idx):
        return self.queries[idx]["positives"]

    def get_negatives(self, idx):
        return self.queries[idx]["negatives"]
    