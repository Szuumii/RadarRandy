from torch.utils.data import DataLoader, Sampler
import copy
import random
from MulRan import MulRanDataset
from dataset_utils import display_batch


class BatchSampler(Sampler):
    def __init__(self, dataset: MulRanDataset, batch_size: int, batch_size_limit: int = None):
        self.batch_size = batch_size
        self.dataset = dataset
        self.k = 2  # No of positive examples
        if self.batch_size < 2 * self.k:
            self.batch_size = 2 * self.k
            print(
                f"Warning: Batch too small. Batch size increased to {self.batch}")

        self.batch_idx = []
        self.elems_idx = {}
        for idx in self.dataset.queries:
            self.elems_idx[idx] = True

    def __len__(self):
        return len(self.batch_idx)

    def __iter__(self):
        self.generate_batches()
        for batch in self.batch_idx:
            for element in batch:
                yield element

    def generate_batches(self):
        self.batch_idx = []
        unused_elements_idx = copy.deepcopy(self.elems_idx)
        current_batch = []
        assert self.k == 2, "Sampler can sample only k=2 elements from th same class"

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_idx) == 0:
                if len(current_batch) >= 2 * self.k:
                    assert len(
                        current_batch) % self.k == 0, f"Incorrect batch size: {len(current_batch)}"
                    self.batch_idx.append(current_batch)
                    current_batch = []
                if len(unused_elements_idx) == 0:
                    break

            selected_element = random.choice(list(unused_elements_idx))
            unused_elements_idx.pop(selected_element)
            positives = self.dataset.get_positives(selected_element)
            if len(positives) == 0:
                # No positives found, try another element
                continue

            unused_positives = [
                e for e in positives if e in unused_elements_idx]

            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_idx.pop(second_positive)
            else:
                second_positive = random.choice(positives)

            current_batch += [selected_element, second_positive]

        for batch in self.batch_idx:
            assert len(
                batch) % self.k == 0, f"Incorrect batch size: {len(batch)}"


if __name__ == '__main__':
    dataset_path = ""
    query_filename = ''

    ds = MulRanDataset(dataset_path, query_filename)
    sampler = BatchSampler(ds, batch_size=16)
    dataloader = DataLoader(ds, batch_sampler=sampler)
    e = ds[0]
    res = next(iter(dataloader))
    display_batch(res)
