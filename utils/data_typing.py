from typing import NewType, Union, Dict

import pydicom as pyd

__all__ = [
    'CardiacPhase', 'FilePathPack'
]

CardiacPhase = NewType('CardiacPhase', Union[float, int])
FilePathPack = NewType('FilePathPack', tuple[pyd.FileDataset, str])
IspCtPair = NewType('IspCtPair', Dict[str, str | CardiacPhase])
