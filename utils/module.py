### Core modules ###


### Type hints ###
from typing import Any


### Internal modules ###


def get_instance(
    instances,
    instance_name: str
) -> Any:
    """
    Get a class instance from a file which is imported as instances.

    Args:
        instances       (object):   imported file containing all functions, classes, etc.
        instance_name   (str):      the name of function, class, etc., in the file.

    Returns:
        instance: the function, class, etc.
    """
    instance: Any = getattr(
        instances,
        instance_name
    )

    return instance