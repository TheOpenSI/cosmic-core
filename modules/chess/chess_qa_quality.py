### Core modules ###
from pandas import read_csv
from numpy import (
   nonzero,
    mean,
    array
)
from numbers import Number


### Type hints ###


### Internal modules ###
from ...utils.log_tool import set_color
from ...utils.num2word import convert_number2word
from ...src.services.qa import QABase


class QualityEval(QABase):
    def __init__(
        self,
        is_rag: bool = False,
        **kwargs
    ):
        """
        Evaluate OpenSI-CoSMIC's qualities except for the reasoning which is evaluated in
        src/modules/chess_qa_puzzle.py.

        Args:
            is_rag (bool, optional): retrieve context from vector database or not. Defaults to False.
        """
        super().__init__(**kwargs)

        # Set config.
        self.is_rag = is_rag


    def parse_quality_csv(
        self,
        csv_path: str
    ):
        """
        Get query information from .csv file.

        Args:
            csv_path (str): .csv file path.

        Returns:
            info (dict): a dictionary containing query information.
        """
        # Read data.
        df = read_csv(csv_path)

        # Set a dictionary.
        info = {
            "question": df["Question"],
            "answer": df["Answer"]
        }

        return info


    def batch_process(
        self,
        query_csv: str
    ):
        """Process multiple queries in a .csv file.

        Args:
            query_csv (str): .csv file path.

        Returns:
            average_score (float): average score over queries having a ground truth answer.
        """
        # Get questions and ground truth answers.
        data_info = self.parse_quality_csv(query_csv)
        questions = data_info["question"]
        answers = data_info["answer"]

        num_questions = len(questions)
        score_list = []

        # Write information head for log file.
        if self.log_file is not None:
            self.log_file.writerow(["Question", "Answer", "Label", "Score", "Comment", "Raw Answer"])

        print(set_color("info", f"Processing {query_csv}..."))

        for idx, (question, gt_answer) in enumerate(zip(questions, answers)):
            # Print progress.
            if idx % 10 == 0 or idx == num_questions - 1:
                print(set_color("info", f"Solving {idx + 1}/{num_questions}."))

            # Process each query.
            result, raw_result, retriever_score = super().__call__(
                question,
                is_rag=self.is_rag
            )

            # For exit, skip, __update__store__, the result is None.
            if result is None: continue

            # Change None to "N/A" for comparison.
            if (gt_answer is None) or (isinstance(gt_answer, numbers.Number) and np.isnan(gt_answer)):
                gt_answer = "N/A"

            # Case insensitive and remove line change for better readability.
            if isinstance(gt_answer, Number) or gt_answer.isdigit():
                # Convert number to word and compare both number and string format answer.
                gt_answer = [str(int(gt_answer)), str(convert_number2word(int(gt_answer)))]

            elif isinstance(gt_answer, str):
                # A number can be read as a string, so convert it to a number.
                gt_answer = gt_answer.lower().replace("\n", " ")

            if isinstance(result, str):
                result = result.lower().replace("\n", " ")
                raw_result = raw_result.lower().replace("\n", " ")

            # Check if the answer is in the analysis.
            if isinstance(gt_answer, list):
                score_per = float(len(nonzero([float(result.find(v) > -1) for v in gt_answer])[0]) > 0)
            else:
                score_per = float(result.find(gt_answer) > -1)

            # Push each question score to the list.
            score_list.append(score_per)

            # Save to log.
            if self.log_file is not None:
                # Take the first retriever score even if multiple documents are retrieved from vector database,
                # that is when topk>1.
                if isinstance(retriever_score, list): retriever_score = retriever_score[0]

                self.log_file.writerow([question, result, gt_answer, score_per, retriever_score, raw_result])

                print(set_color(
                    "info",
                    f"Question: {question}, GT: {gt_answer}, Result: {result}, " \
                    f"Retrieve score: {retriever_score:.4f}.")
                )

        # Calculate the average score for queries having a ground truth answer.
        average_score = mean(array(score_list))

        return average_score