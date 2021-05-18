from evaluation.eval_utils import compute_embeddings, find_nearest_neighbours
from dataset.MulRan import MulRanDataset
from model.models import MetricLearner
import random
import numpy as np
import math
from torchvision import transforms
import torch
from dataset.dataset_preprocessing import is_positive


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
    database_embeddings = compute_embeddings(database_set, device, model)
    print("Database calculations finished")

    model.to(device)
    model.eval()


    # indexes = list(range(0,len(querries_set)))

    # test_cases = 2000

    # test_indexes = random.sample(indexes, test_cases)

    n = 10
    count = 0
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    for i in range(0, len(querries_set)):

        x, _, querry_position = querries_set[i]

        # print(f"Querry position: {querry_position}")

        x = x.to(device)
        x = x.unsqueeze(0).contiguous()

        querry_embedding = model(x).squeeze(0)

        nn_idx, nn_dist = find_nearest_neighbours(database_embeddings, querry_embedding, k=n)

        # for j in range(n):


        # if i % 100 == 0:
        #     print(f"Closest distances: {nn_dist[0]}, {nn_dist[1]}, {nn_dist[3]}")
        # if i % 500 == 0:
        #     temp = np.power(querry_position - neighbour_position, 2)
        #     real_distance = math.sqrt(temp[0] + temp[1] + temp[2])
        #     print(f"Nearest neighbour idx: {neighbour} calculated distance is {nn_dist[0]} and real-life distance is {real_distance}")
            # print(f"For index {nn_idx[j]} calculated distance is {nn_dist[j]} and real-life distance is {real_distance}")

        for j in range(len(nn_idx)):
            neighbour = nn_idx[j]
            _, _, neighbour_position = database_set[neighbour]
            if is_positive(neighbour_position, querry_position):                
                if j == 0:
                    recall_at_1 += 1
                    recall_at_3 += 1
                    recall_at_5 += 1
                    recall_at_10 += 1
                    break
                elif j < 3 :
                    recall_at_3 += 1
                    recall_at_5 += 1
                    recall_at_10 += 1
                    break
                elif j < 5:
                    recall_at_5 += 1
                    recall_at_10 += 1
                    break
                elif j < 10:
                    recall_at_10 += 1
                    break

        count += 1

        # print(f"Nearest neighbour index: {nearest_neighbour}")

        # print(f"Nearest neighbour distance: {nearest_neighbour_distance}")

    
    recall_1 = (recall_at_1 / count) * 100

    recall_3 = recall_at_3 / (count) * 100

    recall_5 = recall_at_5 / (count) * 100

    recall_10 = (recall_at_10 / count) * 100


    print(f"Recall at 1: {recall_1} %")
    print(f"Recall at 3: {recall_3} %")
    print(f"Recall at 5: {recall_5} %")
    print(f"Recall at 10: {recall_10} %")

    # print(database_embeddings[0])


if __name__ == "__main__":
    ckpt_path = "/home/jszumski/RadarRandy/lightning_logs/logs/version_1/checkpoints/epoch=3-step=1539.ckpt"
    model = MetricLearner.load_from_checkpoint(ckpt_path)
    
    evaluate(model)
