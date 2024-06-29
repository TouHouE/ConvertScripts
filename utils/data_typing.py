from typing import NewType, Union

import pydicom as pyd

__all__ = [
    'CardiacPhase', 'FilePathPack'
]

CardiacPhase = NewType('CardiacPhase', Union[float, int])
FilePathPack = NewType('FilePathPack', tuple[pyd.FileDataset, str])