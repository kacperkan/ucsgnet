from torch.utils.data import DataLoader
from typing_extensions import Literal

from ucsgnet.ucsgnet.net_2d import Net as Net2D
from ucsgnet.dataset import CADDataset, get_simple_2d_transforms


class Net(Net2D):
    def build(self, data_path: str, **kwargs):
        self.data_path_ = data_path

    def _dataloader(
        self, training: bool, split_type: Literal["train", "valid"]
    ) -> DataLoader:
        batch_size = self.hparams.batch_size
        transforms = get_simple_2d_transforms()
        loader = DataLoader(
            dataset=CADDataset(self.data_path_, split_type, transforms),
            batch_size=batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=0,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(True, "train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(False, "valid")
