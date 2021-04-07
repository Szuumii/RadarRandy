import matplotlib.pyplot as plt


def compute_embeddings(dataset, model):
    model.to(device)
    model.eval()

    embeddings = None
    for ndx, (x, _) in enumerate(dataset):
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


def find_nearest_neighbours(embeddings, query_ndx, k=10):
    y = embeddings[query_ndx]
    dist = torch.norm(embeddings - y, p=2, dim=1)
    values, nn_ndx = torch.topk(dist, k + 1, largest=False, sorted=True)
    nn_ndx = nn_ndx[1:]
    return nn_ndx, dist[nn_ndx]


def show_search_results(ds, embeddings, query_ndx):
    nn_ndx, nn_dist = find_nearest_neighbours(embeddings, query_ndx, k=5)
    print(nn_dist)
    fig, axis = plt.subplots(1, 1 + len(nn_ndx))

    fig.suptitle('Nearenst Neighbours')
    query_img, query_id = ds[query_ndx]
    axis[0].imshow(tensor_to_image(query_img))
    axis[0].set_xlabel(f'id: {query_id}\nbase')

    for i, neighbour_ndx in enumerate(nn_ndx):
        img, id = ds[neighbour_ndx]
        axis[i+1].imshow(tensor_to_image(img))
        axis[i+1].set_xlabel(f'id: {id}\ndist {nn_dist[i]:0.3f}')
