from vitaa.data import datasets

from .cuhkpedes import cuhkpedes_evaluation
from .market1501 import market1501_evaluation
from .dukemtmc import dukemtmc_evaluation


def evaluate(dataset, predictions, output_folder):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(dict): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder
    )
    if isinstance(dataset, datasets.CUHKPEDESDataset):
        return cuhkpedes_evaluation(**args)
    elif isinstance(dataset, datasets.Market1501Dataset):
        return market1501_evaluation(**args)
    elif isinstance(dataset, datasets.DukeMTMCDataset):
        return dukemtmc_evaluation(**args)
    elif isinstance(dataset, datasets.DemoDataset):
        return cuhkpedes_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
