from dataclasses import dataclass
from typing import Iterable

@dataclass
class Field():
    name: str
    long_name: str
    standard_name: str
    units: str
    pre_transformations: Iterable[str]
    post_transformations: Iterable[str]        