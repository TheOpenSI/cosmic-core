### Core modules ###
from requests import post


### Type hints ###
from requests import (
    Response,
    RequestException
)



### Internal modules ###
from ...src.services.base import ServiceBase


class CodeGenerator(ServiceBase):
    def __init__(
        self,
        service_container_name: str = "pycapsule",
        model_name: str = "qwen2.5-coder",
        **kwargs
    ) -> None:
        """
        Code generation module.
        Sends http requests at port 8780 to the PyCapsule service.

        Args:
            service_container_name (str): Name of the container, set to localhost for development.
                Set this to the container name for docker network.
            model_name (str): Model name.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.PORT = 8780 # PyCapsule service default port
        self.service_url = f"http://{service_container_name}:{self.PORT}"


    def _check_service_status(self):
        """
        Check if the PyCapsule service is running.
        """
        try:
            # Send a health check request to the service
            response: Response = post(f"{self.service_url}/health")
            if response.status_code != 200:
                raise Exception(f"PyCapsule service is not running. Status code: {response.status_code}")

        except RequestException as e:
            raise Exception(f"Failed to connect to PyCapsule service: {str(e)}")


    def __call__(
        self,
        query:str,
        timeout: int = 60
    ) -> tuple[str, str]:
        """
        Send the user query to the PyCapsule service.

        Args:
            query (str): User query.
            timeout (int): Timeout for the request. Defaults to 30 seconds.
        """
        try:
            # Check service availability
            self._check_service_status()

            # Send the user query to the PyCapsule service
            pycaspsule_response: Response = post(
                f"{self.service_url}/query",
                json={"query": query},
                timeout=timeout
            )

            if pycaspsule_response.status_code != 200:
                raise Exception(
                    (f"PyCapsule service returned error. "
                    f"Status code: {pycaspsule_response.status_code}")
                )

            # Extract respone, error, code and status from the response
            response: str = pycaspsule_response.json()[0].get("code", "")
            raw_response: str = pycaspsule_response.json()[0].get("response", "") + "\n\n" + response

        except Exception as e:
            raw_response = response = "PyCapsule service encountered an error."
            print(str(e))

        return (response, raw_response)