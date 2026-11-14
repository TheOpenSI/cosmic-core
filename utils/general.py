### Core modules ###
import openai


### Type hints ###


### Internal modules ###


def validate_openai_api_key(
    api_key: str
) -> bool:
    """
    Validates the given OpenAI API key by making a test request.

    Args:
        api_key (str): The OpenAI API key to validate.

    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    openai.api_key = api_key

    try:
        # Make a simple API call to validate the key
        openai.models.list()

        return True

    except openai.AuthenticationError:
        return False

    except Exception as e:
        print(f"An error occurred during validation: {e}")

        return False