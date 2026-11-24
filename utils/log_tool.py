class BCOLORS:
    HEADER:     str = "\033[95m"
    OKBLUE:     str = "\033[94m"
    OKCYAN:     str = "\033[96m"
    OKGREEN:    str = "\033[92m"
    WARNING:    str = "\033[93m"
    FAIL:       str = "\033[91m"
    ENDC:       str = "\033[0m"
    BOLD:       str = "\033[1m"
    UNDERLINE:  str = "\033[4m"


INFOR_DICT: dict[str, dict[str, str]] = {
    "success": {
        "color": BCOLORS.OKGREEN,
        "comment": "Success"
    },
    "fail": {
        "color": BCOLORS.FAIL,
        "comment": "Fail"
    },
    "warning": {
        "color": BCOLORS.WARNING,
        "comment": "Warning"
    },
    "info": {
        "color": BCOLORS.HEADER,
        "comment": "Info"
    },
    "error": {
        "color": BCOLORS.FAIL,
        "comment": "Error"
    },
    "hint": {
        "color": BCOLORS.OKGREEN,
        "comment": "Hint"
    },
}


def set_color(
    status: str,
    information: str
):
    """
    Set color to display information on terminal.

    Args:
        status (str): information type, see INFOR_DICT.keys.
        information (str): information to be printed.

    Returns:
        information (str): colorized information.
    """
    normalise_status: str = status.lower()

    msg: str = "{0:s}[{1:s}]{2:s} {3:s}".format(
        INFOR_DICT[normalise_status]['color'],
        INFOR_DICT[normalise_status]['comment'],
        BCOLORS.ENDC,
        information
    )

    return msg