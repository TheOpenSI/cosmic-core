### Core modules ###
from os import environ, getenv


### Type hints ###


### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC


if __name__ == "__main__":
    # Switch on this to avoid massive warning.
    environ["TOKENIZERS_PARALLELISM"] = "false"

    # # # Build the system for a specific LLM.
    opensi_cosmic = OpenSICoSMIC()

    # # Get query from arguments.
    query = getenv(
        key="query",
        default=""
    )

    # Run for each question/query, return the truncated response if applicable.
    answer, _, _ = opensi_cosmic(query)

    # Print the answer.
    print(f"Query: {query}\nAnswer: {answer}")

    # Remove memory cached in the system.
    opensi_cosmic.quit()