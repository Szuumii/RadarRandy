import matplotlib.pyplot as plt
import torch
import math
from dataset.dataset_utils import tensor_to_image

def compute_embeddings(dataset, device,model):
    model.to(device)
    model.eval()

    embeddings = None
    for ndx, (x, _, _) in enumerate(dataset):
        # if ndx % 100 == 0:
            # print(f"Going through {ndx}-th example of database set")
        x = x.to(device)
        x = x.unsqueeze(0).contiguous()
        with torch.no_grad():
            y = model(x).squeeze(0)
        if embeddings is None:
            ds_len = len(dataset)
            embeddings = torch.zeros(
                (ds_len, y.shape[0]), dtype=y.dtype, device=x.device)

        embeddings[ndx] = y

    assert len(embeddings) == len(dataset)

    return embeddings


def is_true_positive(position_1, position_2):
    pass

def show_n_search_results(n, database_set, querries_set, db_embeddings, querry_embedding ,query_ndx):
    nn_ndx, nn_dist = find_nearest_neighbours(db_embeddings, querry_embedding, k=n)
    
    fig, axis = plt.subplots(1, 1 + len(nn_ndx), figsize=(16,8))
#     fig.suptitle(f'{n} nearest neighbours')
    query_img, query_id, querry_position = querries_set[query_ndx]
    axis[0].imshow(tensor_to_image(query_img))
    axis[0].set_xlabel(f'id: {query_id}\nbase')

    for i, neighbour_ndx in enumerate(nn_ndx):
        img, id, neighbour_position = database_set[neighbour_ndx]
        temp = np.power(querry_position - neighbour_position, 2)
        real_distance = math.sqrt(temp[0] + temp[1] + temp[2])
        axis[i+1].imshow(tensor_to_image(img))
        axis[i+1].set_xlabel(f'id: {id}\nembedding_dist {nn_dist[i]:0.3f}\nreal-life_dist {real_distance:0.2f}')

def find_nearest_neighbours(database_embeddings, querry_embedding, k=5):
    dist = torch.norm(database_embeddings - querry_embedding, p=2, dim=1)
    values, nn_ndx = torch.topk(dist, k + 1, largest=False, sorted=True)
    return nn_ndx, dist[nn_ndx]


def show_search_results(ds, embeddings, query_ndx):
    nn_ndx, nn_dist = find_nearest_neighbours(embeddings, query_ndx, k=5)
    print(nn_dist)
    fig, axis = plt.subplots(1, 1 + len(nn_ndx))

    fig.suptitle('Nearest Neighbours')
    query_img, query_id = ds[query_ndx]
    axis[0].imshow(tensor_to_image(query_img))
    axis[0].set_xlabel(f'id: {query_id}\nbase')

    for i, neighbour_ndx in enumerate(nn_ndx):
        img, id = ds[neighbour_ndx]
        axis[i+1].imshow(tensor_to_image(img))
        axis[i+1].set_xlabel(f'id: {id}\ndist {nn_dist[i]:0.3f}')
