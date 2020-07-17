from .dataset.coco import COCO
from .sample.ctdet import CTDetDataset


dataset_factory = {'coco': COCO,}
_sample_factory = {'ctdet': CTDetDataset,}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass
    return Dataset

