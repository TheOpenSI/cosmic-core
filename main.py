# -------------------------------------------------------------------------------------------------------------
# File: main.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
#
# Copyright (c) 2024 Open Source Institute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import os, csv
import pandas as pd

from src.opensi_cosmic import OpenSICoSMIC
from utils.log_tool import set_color

# =============================================================================================================

if __name__ == "__main__":
    # Switch on this to avoid massive warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get the file's absolute path.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = f"{current_dir}"

    # Set a bunch of questions, can also read from .csv.
    df = pd.read_csv(f"{root}/data/test.csv")
    queries = df["Question"]
    answers = df["Answer"]

    # Build the system for a specific LLM.
    config_path = os.path.join(root, "scripts/configs/config.yaml")
    opensi_cosmic = OpenSICoSMIC(config_path=config_path)

    # Loop over questions to get the answers.
    for idx, (query, gt) in enumerate(zip(queries, answers)):
        # Skip marked questions.
        if query.find("skip") > -1: continue

        # Create a log file.
        if query.find(".csv") > -1:
            # Remove all namespace.
            query = query.replace(" ", "")

            # Return if file is invalid.
            if not os.path.exists(query):
                set_color("error", f"!!! Error, {query} not exist.")
                continue

            # Change the data folder to results for log file.
            log_file = query.replace("/data/", f"/results/{opensi_cosmic.config.llm_name}/")

            # Create a folder to store log file.
            log_file_name = log_file.split("/")[-1]
            log_dir = log_file.replace(log_file_name, "")
            os.makedirs(log_dir, exist_ok=True)
            log_file_pt = open(log_file, "w")
            log_file = csv.writer(log_file_pt)
        else:
            log_file_pt = None
            log_file = None

        # Run for each question/query, return the truncated response if applicable.
        answer, _, _ = opensi_cosmic(query, log_file=log_file)

        # Print the answer.
        if answer is not None and isinstance(gt, str):  # compare with GT string
            # Assign to q variables.
            status = "success" if (answer.find(gt) > -1) else "fail"

            print(set_color(status, f"\nQuestion: '{query}' with GT: {gt}.\nAnswer: '{answer}'."))
        elif answer is not None and answer != '':
            print(set_color("info", f"\nQuestion: '{query}'.\nAnswer: '{answer}'."))

        # Close log file pointer.
        if log_file_pt is not None:
            log_file_pt.close()

    # Remove memory cached in the system.
    opensi_cosmic.quit()