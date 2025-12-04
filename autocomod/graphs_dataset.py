from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.data import BaseData


class GraphsDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        files: list[str] | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        super().__init__(root, transform, pre_transform)
        if files is None:
            self._files = sorted(Path(self.root).glob("*.pt"))
        else:
            self._files = [Path(root) / file for file in files]

        data_list = self._load_data_list()
        self._data_len = len(data_list)
        self._data_dict = {data.name: data for data in data_list}

        self.data, self.slices = self.collate(data_list)

    def _load_data_list(self) -> list[Data]:
        data_list = []
        for pt_file in self._files:
            data: Data = torch.load(pt_file, weights_only=False)
            data.name = pt_file.stem
            data_list.append(data)
        return data_list

    def len(self) -> int:
        return self._data_len

    def get(self, idx: int) -> BaseData:
        return super().get(idx)

    def metadata(self, name: str):
        return getattr(self._data_dict[name], "_metadata", None)
