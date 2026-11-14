### Core modules ###
from pathlib import Path
from pandas import read_csv
from numpy import mean


### Type hints ###


### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC
from ...src.services.qa import QABase
from ...utils.log_tool import set_color


class QABaseTest(QABase):
    def __init__(self, *args, **kwargs):
        """
        Base class for QA test.
        """
        super().__init__(*args, **kwargs)


    def __call__(
        self,
        query: str,
        context: str = "",
        is_rag: bool = False,
        verbose: bool = False
    ):
        """
        Process each QA.

        Args:
            query   (str):                  a question.
            context (str|dict, optional):   contex associated with the question. Defaults to "".
            is_rag  (bool, optional):       if retrieve context for the question. Defaults to False.
            verbose (bool, optional):       debug mode. Default to False.

        Returns:
            response    (str): service number.
            empty       (str): None. Just align the number of outputs.
        """
        # Get service option through query analyser.
        service_option, _ = self.query_analyser(query)

        return service_option, None, None


class OpenSICoSMICTest(OpenSICoSMIC):
    def __init__(self, *args, **kwargs):
        """
        Construct OpenSICoSMIC test instance. It contains LLM and services including vector database
        and RAG, where RAG includes context retriever and vector database update.
        Chess services are induced in PuzzleAnalyse and QualityEval, called on demand, not as global instance.
        """
        super().__init__(*args, **kwargs)

        # Release unused LLM and database.
        self.llm.quit()
        self.rag.vector_database.quit()

        # Use QA test version.
        self.qa = QABaseTest(self.query_analyser, None, None)


if __name__ == "__main__":
    # Get the file's absolute path.
    root = Path(__file__).resolve(strict=True).parent.parent.parent

    # Load test dataset.
    df = read_csv(f"{root}/data/test.csv")
    queries = df["Question"]
    answers = df["Label"].astype('str')

    # Build CoSMIC test version.
    opensi_cosmic = OpenSICoSMICTest()

    # Initialize statistics.
    fail_list = []
    statistics_dict = {}
    label_list = ["0.0", "0.1", "1", "2", "3"]

    # Update to remove no answers.
    valid_indices = [idx for idx, v in enumerate(answers) if v != "nan"]
    queries = queries[valid_indices]
    answers = answers[valid_indices]

    for label in label_list:
        statistics_dict.update({label: {"success": 0, "fail": 0, "success_rate": -1}})

    # Run over each sample and compare with the ground truth service number.
    for idx, (query, gt) in enumerate(zip(queries, answers)):
        # Align gt.
        if gt in ["1.0", "2.0", "3.0"]:
            gt = gt.replace(".0", "")

        # Print the progress.
        if idx % 50 == 0 or idx == len(queries) - 1:
            print(f"{idx + 1}/{len(queries)}, fail list: {fail_list}.")

        # Get service option.
        predicted_service, _, _ = opensi_cosmic(query)

        # Statistics for success and fail samples.
        if predicted_service == gt:
            statistics_dict[gt]["success"] += 1

        else:
            statistics_dict[gt]["fail"] += 1
            fail_list.append(idx)

    # Calculate the success rate.
    average_success_rate = []

    for label in label_list:
        total_num = statistics_dict[label]["success"] + statistics_dict[label]["fail"]

        if total_num == 0:
            success_rate = 0

        else:
            success_rate = 100 * statistics_dict[label]["success"] / total_num
            average_success_rate.append(success_rate)

        statistics_dict[label]["success_rate"] = success_rate

    # Print the statistics.
    print(set_color(
        "info",
        f"{statistics_dict}.\n" \
        f"Average success rate: {mean(average_success_rate)}.\n" \
        f"Error list: {fail_list}.")
    )