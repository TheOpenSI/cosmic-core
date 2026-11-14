### Core modules ###
from fastapi import Request


### Type hints ###


### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC



def get_opensi_cosmic(request: Request) -> OpenSICoSMIC:
    return request.app.state.opensi_cosmic