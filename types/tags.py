### Core modules ###
from enum import Enum


### Type hints ###


### Internal modules ###



"""
https://fastapi.tiangolo.com/tutorial/path-operation-configuration/?h=enum#tags-with-enums
"""
class APITag(Enum):
    """docstring for APITag."""
    models = "Models API Endpoint"
    cosmic = "CoSMIC API Endpoint"
    memory = "Memory API Endpoint"