import enum
from typing import NewType, Union, Dict

import pydicom as pyd

__all__ = [
    'CardiacPhase', 'FilePathPack', 'IspCtPair', 'PatientId', 'Gender'
]

CardiacPhase = NewType('CardiacPhase', Union[float, int])
FilePathPack = NewType('FilePathPack', tuple[pyd.FileDataset, str])
IspCtPair = NewType('IspCtPair', Dict[str, str | CardiacPhase])
PatientId = NewType('PatientId', str)


class Gender(enum.Enum):
    Female = enum.auto()
    Male = enum.auto()

    @classmethod
    def declare(cls, gender: str | None):
        if gender is None:
            return None
        if '男' in gender:
            return Gender.Male
        else:
            return Gender.Female

    def __repr__(self):
        if self == Gender.Male:
            return 'male'
        if self == Gender.Female:
            return 'female'
        return 'unk'
if __name__ == '__main__':
    gender = Gender.declare(None)
    print(gender)