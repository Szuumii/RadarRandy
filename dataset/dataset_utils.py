import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from MulRan import MulRanDataset


def make_datasets(dataset_root):
    datasets = {}
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    train_queries_filepath = os.path.join(
        dataset_root, "mulran_train_processed_queries.pickle")
    test_queries_filepath = os.path.join(
        dataset_root, "mulran_test_processed_queries.pickle")
    datasets["train"] = MulRanDataset(
        dataset_root, train_queries_filepath, train_transform)
    datasets["test"] = MulRanDataset(
        dataset_root, train_queries_filepath, test_transform)
    return datasets


def tensor_to_image(tensor_image):
    img_size = (400, 400)
    to_pil_image = Compose(
        [transforms.Resize(img_size), transforms.ToPILImage()])
    image = to_pil_image(tensor_image)
    return image


def display_batch(batch):
    imgs, classes = batch
    n_images_per_row = 8
    n_rows = int(np.ceil(len(imgs) / n_images_per_row))
    fig, axis = plt.subplots(n_rows, n_images_per_row)
    for ndx in range(len(imgs)):
        row = ndx // n_images_per_row
        col = ndx % n_images_per_row
        axis[row, col].imshow(tensor_to_image(imgs[ndx]))
        axis[row, col].set_xlabel(classes[ndx].item())


def polar_to_carthesian(polar_image_tensor):
    radius, theta = image_tensor[0].size()
    cart_image_tensor = torch.zeros(
        torch.Size([3, 2 * radius + 1, 2 * radius + 1]))
    for i in range(0, 3):
        for r in range(0, radius):
            for t in range(0, theta):
                x = radius - r * math.cos(t * 2 * math.pi / theta) + 1
                y = radius + r * math.sin(t * 2 * math.pi / theta) + 1
                cart_image_tensor[i][math.floor(x)][math.floor(
                    y)] = polar_image_tensor[i][r][t]
    return cart_image_tensor


def display_triplet(batch, triplets, i):
    x, labels = batch
    anchors, positives, negatives = triplets

    fig, axis = plt.subplots(1, 3)
    anchor_ndx = anchors[i]
    positive_ndx = positives[i]
    negative_ndx = negatives[i]

    anchor2pos_dist = torch.norm(x[anchor_ndx] - x[positive_ndx], p=2)
    anchor2neg_dist = torch.norm(x[anchor_ndx] - x[negative_ndx], p=2)
    anchor2pos_dist = torch.norm(
        embeddings[anchor_ndx] - embeddings[positive_ndx], p=2)
    anchor2neg_dist = torch.norm(
        embeddings[anchor_ndx] - embeddings[negative_ndx], p=2)

    fig.suptitle('Triplet: anchor, positive and negative')
    axis[0].imshow(tensor_to_image(x[anchor_ndx]))
    axis[0].set_xlabel(f'Anchor\n: {labels[anchor_ndx].item()}')
    axis[1].imshow(tensor_to_image(x[positives[i]]))
    axis[1].set_xlabel(
        f'Positive\n: {labels[positive_ndx].item()}\nDistance {anchor2pos_dist:0.3f}')
    axis[2].imshow(tensor_to_image(x[negatives[i]]))
    axis[2].set_xlabel(
        f'Negative\n : {labels[negative_ndx].item()}\Distance {anchor2neg_dist:0.3f}')


if __name__ == "__main__":
    dataset_root = "/data3/mulran"

    datasets = make_datasets(dataset_root)

    x, label = datasets["train"][0]
    print(x)
