### Core modules ###
from re import (
    match,
    sub
)
from shlex import quote
from keyword import iskeyword


### Type hints ###


### Internal modules ###


# https://docs.python.org/3/library/shlex.html#shlex.quote


def sanitise_input(
    input_string: str
) -> str:
    """
    Sanitizes input string by escaping special characters and validating it.
    Input string can only contain alphanumeric characters, '.', '_', ':', and '-'.

    Args:
        input_string    (str):  Input string to be sanitized.
        pattern         (str):  Regex pattern to validate the input.

    Raises:
        ValueError: If input string contains invalid characters.

    Returns:
        str: Sanitized input string.
    """
    pattern: str = r"^[a-zA-Z0-9._:-]+$"

    # Validate the input against the provided pattern
    if not match(
        pattern=pattern,
        string=input_string,
        flags=0
    ):
        raise ValueError(f"Invalid input: {input_string}")

    # Escape the input for safe shell usage
    return quote(s=input_string)


def sanitise_variable_name(
    name: str
) -> str:
    """
    Fixes a string to make it a valid Python variable name.

    Parameters:
        name (str): The input string to sanitize.

    Returns:
        str: A valid Python variable name.
    """
    # Replace invalid characters with underscores
    new_name: str = sub(
        pattern=r"[^a-zA-Z0-9_]",
        repl="_",
        string=name,
        count=0,
        flags=0
    )

    # Ensure the name doesn't start with a digit
    if name and new_name[0].isdigit():
        new_name: str = f"_{name}"

    # Ensure the name isn't a Python reserved keyword
    if iskeyword(name):
        new_name: str = f"{name}_var"

    return new_name