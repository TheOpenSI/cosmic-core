from .base import ServiceBase

class FallbackService(ServiceBase):
    def __init__(self,
                 **kwargs) -> None:
        """
        Service -1: fallback service for when no other service can answer the query.

        Args:
            **kwargs: additional keyword arguments.
        """
        super().__init__(**kwargs)


    def _fix_service_name(self,
                          service_name: str) -> str:
        """
        Fix the service name to be more user-friendly e.g. "service_1" -> "Service 1".

        Args:
            service_name (str): the original service name.

        Returns:
            fixed_service_name (str): the fixed service name.
        """
        return service_name.replace('_', ' ').title()


    def _get_service_names(self,
                           services: dict[str, dict[str, str]]) -> str:
        """
        Get the names of the available services excluding the system information service.

        Returns:
            service_names (str): names of the available services.
        """
        return ", ".join(
            self._fix_service_name(service.get("name", "undefined_service_name"))
            for service in services.values()
            if not "system_information" in service.get("name", "")
        )


    def __call__(self,
                 services: dict[str, dict[str, str]]) -> tuple[str, str]:
        """
        Answer a query when no other service can answer it.

        Returns:
            response (str): truncated answer.
            raw_response (str): original answer from LLM.
        """
        service_string = self._get_service_names(services) or "None"
        raw_response = response = (f"I am unable to assist with that request at this time. "
                                   f"It may be outside my current capabilities, or the required service is not enabled. "
                                   f"For your reference, I currently have access to the following {len(services) - 1} service(s): "
                                   f"{service_string}. "
                                   f"Please feel free to rephrase your request, or contact the OpenSI team for further support.")

        return (response, raw_response)