import os
import requests
from pydantic import BaseModel, Field
from typing import List
from main import reload

baseUrl = os.environ.get("OPENSI_COSMIC_API_BASE_URL", "http://cosmic:3000")

class Pipeline:
    class Valves(BaseModel):
        pass
        # WORD_LIMIT: int = Field(title="Word Limit", default=300, description="Word limit when getting page summary")
        # WIKI_ROOT: str = Field(title="Wiki", default="https://en.wikipedia.org/wiki", description="Wikipedia root URL")
        # SOURCE_TYPE: str = Field(title="Source Type", default="Wikipedia", description="Source type for the pipeline", enum=["Wikipedia", "Google", "Custom"])
    # A class-level dictionary to keep track of user queries.
    # Key: user_id, Value: number of queries asked.
    user_queries_count = {}

    def __init__(self):
        self.id = "cosmic_pipeline"
        self.name = "OpenSI-CoSMIC"

        # self.valves = self.Valves(
        #     **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        # )

        # print("TEST", self.valves.RATE_LIMIT)

        self.MAX_QUERIES_PER_USER = int(os.environ.get("MAX_QUERIES_PER_USER", "5"))

    # async def on_valves_updated(self):
    #     pass


    def quit_cosmic(self):
        try:
            response = requests.get(f"{baseUrl}/quit")
            if response.status_code != 200:
                raise Exception(f"Error during /quit API call: {response.text}")
            print(response.json()["message"])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error during /quit API call: {e}")

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        self.quit_cosmic()

    def post_to_cosmic(self, user_message: str, body: dict) -> str:
        try:
            response = requests.post(
                f"{baseUrl}/cosmic",
                json={"user_message": user_message, "body": body}
            )
            if response.status_code == 200:
                return response.json().get("result", "Something went wrong.")

            raise Exception(f"Error: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error during /cosmic API call: {e}")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ):
        # Some will run twice for history, just ignore.
        if user_message[:3] == "###": return ""

        # # Extract user_id from body. Adjust if user_id is available elsewhere.
        user_id = body["user"]["id"]
        user_role = body["user"]["role"]

        current_count = self.user_queries_count.get(user_id, 0)

        if (user_role != "admin") and (current_count >= self.MAX_QUERIES_PER_USER):
            # Return a message indicating the limit has been reached
            return "You have reached the maximum number of queries allowed."

        # Increment the count for this user
        self.user_queries_count[user_id] = current_count + 1

        answer = self.post_to_cosmic(user_message, body)
        if answer is None: answer = 'Successfully!'

        return answer

# pipeline = Pipeline()

# body = {
#     "user": {
#         "id": 1,
#         "role": "admin",
#         "email": "opensi@canberra.edu.au"
#     }
# }

# user_message = "Do you know what was my first question?"

# print(pipeline.pipe(user_message=user_message, model_id="model_id", messages=[], body=body))