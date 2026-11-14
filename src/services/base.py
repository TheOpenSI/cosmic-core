### Core modules ###
from pathlib import Path


### Type hints ###


### Internal modules ###


class ServiceBase:
    def __init__(
        self,
        log_file: str | None = None
    ):
        """
        Base class for the services in OpenSI-CoSMIC.

        Args:
            log_file (str, optional): (relative) log file path for storing printed information.
                Defaults to None.
        """
        # Set log file path.
        self.log_file = log_file

        # Get the root of the current file to set log file path as absolute path if it is a relative path.
        self.root = Path(__file__).resolve(strict=True).parent.parent.parent


    def set_log_file(
        self,
        log_file: str
    ):
        """
        Set log file externally.

        Args:
            log_file (str): (relative) log file path.
        """
        self.log_file = log_file