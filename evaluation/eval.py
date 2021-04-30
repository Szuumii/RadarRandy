from eval_utils import compute_embeddings, find_nearest_neighbours
from MulRan import MulRanDataset
from models import MetricLearner
import random
import numpy as np
from torchvision import transforms
import torch
from dataset_preprocessing import is_positive


def evaluate(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = '/home/jszumski/mulran_dataset'
    database_queries_file = '/home/jszumski/mulran_dataset/Sejong01_test_processed_queries.pickle'
    querries_file = '/home/jszumski/mulran_dataset/Sejong02_test_processed_queries.pickle'

    transform = transforms.ToTensor()

    database_set = MulRanDataset(
        dataset_path, database_queries_file, transform, True)
    querries_set = MulRanDataset(dataset_path, querries_file, transform, True)

    print("Initiating database calculations")
    database_embeddings = compute_embeddings(database_set, device,model)
    print("Database calculations finished")

    model.to(device)
    model.eval()

    indexes = list(range(0,len(querries_set)))

    test_cases = 1000

    test_indexes = random.sample(indexes, test_cases)

    true_positives = 0

    print(f"Initiating success rate calculation for {test_cases} number of test cases")

    for i in range(test_cases):
        idx = test_indexes[i]
        
        x, ndx, position = querries_set[idx]

        x = x.to(device)
        x = x.unsqueeze(0).contiguous()

        querry_embedding = model(x).squeeze(0)

        nn_ndx, nn_dist = find_nearest_neighbours(database_embeddings, querry_embedding, k=5)

        nearest_neighbour = nn_ndx[0]
        nearest_neighbour_distance = nn_dist[0]
        _, _, querry_position = querries_set[idx]
        _, _, nearest_neighbour_position = database_set[nearest_neighbour]

        if is_positive(querry_position, nearest_neighbour_position):
            true_positives += 1

        # print(f"Nearest neighbour index: {nearest_neighbour}")

        # print(f"Nearest neighbour distance: {nearest_neighbour_distance}")

    
    success_rate = true_positives / test_cases * 100 # IN %

    print(f"Success rate of this model is {success_rate}%")

    # print(database_embeddings[0])


if __name__ == "__main__":
    ckpt_path = "/home/jszumski/RadarRandy/lightning_logs/version_8/checkpoints/epoch=19-step=259.ckpt"
    model = MetricLearner.load_from_checkpoint(ckpt_path)
    
    evaluate(model)
