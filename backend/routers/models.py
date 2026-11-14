### Core modules ###
import threading
import sys
import uuid
from fastapi import (
    APIRouter,
    status
)
from ollama import Client
from fastapi import HTTPException
from pydantic import BaseModel


### Type hints ###
from ...types.tags import APITag


### Internal modules ###
from ...src.services.llms.Ollama import Ollama


router: APIRouter = APIRouter(
    prefix="/api/v1/models",
    tags=[APITag.models]
)


ollama_client = Client(
    host="http://ollama:11434",
    headers={"Content-Type": "application/json"}
)

# track the job_id
download_jobs = {}


class PullModelRequest(BaseModel):
    model: str


@router.get(
    path="",
    status_code=status.HTTP_200_OK
)
async def get_ollama_models():
    try:
        result = ollama_client.list()

        models = []

        for model in result.get("models", []):
            details = model.get("details", {})
            models.append(
                {
                    "model": model.get("model"),
                    "id": model.get("digest", "")[:12],
                    "size": f"{round(model.get('size', 0) / (1000 ** 3), 1)} GB",
                    "modified_at": model.get("modified_at"),
                    "family": details.get("family", "N/A"),
                }
            )

        return {"models": models, "total": len(models)}
        # return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    path="/{model_name}",
    status_code=status.HTTP_200_OK
)
async def delete_ollama_model(model_name: str):
    try:
        ollama_client.delete(model_name)

        return {"message": f"Model '{model_name}' deleted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# fix downloading not persistent by giving frontend a job_id
@router.post(
    path="/pull",
    status_code=status.HTTP_201_CREATED
)
async def pull_ollama_model(request: PullModelRequest):

    job_id = str(uuid.uuid4())
    download_jobs[job_id] = {"status": "running", "logs": [], "error": None}

    def run():

        class PrintMessagesCapture:
            def write(self, message):
                # sys.__stdout__.write(f"raw: {repr(message)}\n") # just want to see how the raw message looks like

                sys.__stdout__.write(message)
                clean = message.strip()
                if not clean:
                    return
                download_jobs[job_id]["logs"].append(clean)

            def flush(self):
                pass

        sys.stdout = PrintMessagesCapture()

        # must handle Ollama errors carefully because it can't have exception as it would break the Ollama code
        # if new errors arise, can add in here
        try:
            Ollama(llm_name=request.model)
            logs = download_jobs[job_id]["logs"]

            for log in logs:
                if "invalid model name" in log:
                    download_jobs[job_id]["status"] = "error"
                    download_jobs[job_id]["error"] = "Invalid model name!"
                    break

                if "does not exist on the Ollama registry" in log:
                    download_jobs[job_id]["status"] = "error"
                    download_jobs[job_id]["error"] = "Invalid model name!"
                    break

                if "no space left on device" in log:
                    download_jobs[job_id]["status"] = "error"
                    download_jobs[job_id]["error"] = "No space left!"
                    break

            else:
                download_jobs[job_id]["status"] = "done"

        except Exception as e:
            download_jobs[job_id]["status"] = "error"
            download_jobs[job_id]["error"] = str(e)

        finally:
            sys.stdout = sys.__stdout__

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


# create get api to check if there is any download,
# if there is show progress for frontend to do polling
@router.get(
    path="/pull/{job_id}",
    status_code=status.HTTP_200_OK
)
async def get_pull_status(job_id: str):
    job = download_jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}

    return job


# @app.post("/models/pull")
# async def pull_ollama_model(request: PullModelRequest):
#     try:
#         Ollama(llm_name=request.model)
#         # ollama_client.pull(request.model)
#         return {"message": f"Model '{request.model}' downloaded successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/models/pull")
# async def pull_ollama_model(request: PullModelRequest):
#     # set up a queue [] (like an array, but can get and put messages in without crashing if empty)
#     # thread will DROP messages in
#     # stream will PICK messages out
#     queue = asyncio.Queue()
#     # needed so the thread can safely talk to asyncio
#     loop = asyncio.get_running_loop()

#     # Class that has "write" method to change from showing messages in terminal to showing in the frontend via stream
#     class PrintMessagesCapture:
#         def write(self, message):
#             sys.__stdout__.write(message)
#             loop.call_soon_threadsafe(queue.put_nowait, message)

#         def flush(self):
#             pass

#     # Thread - allows background jobs to run all together
#     def run():
#         sys.stdout = PrintMessagesCapture()
#         try:
#             Ollama(llm_name=request.model)
#             loop.call_soon_threadsafe(queue.put_nowait, "__DONE__")
#         except Exception as e:
#             loop.call_soon_threadsafe(queue.put_nowait, f"__ERROR__:{e}")

#     threading.Thread(target=run, daemon=True).start()

#     # Stream - picks messages and sends to user live
#     async def stream():
#         while True:
#             message = await queue.get()
#             if message == "__DONE__":
#                 yield f"data: {json.dumps({'type': 'done'})}\n\n"
#                 break

#             else:
#                 clean = message.strip()
#                 if not clean:
#                     continue

#                 if "Download error" in clean:
#                     yield f"data: {json.dumps({'type': 'error', 'message': clean})}\n\n"
#                     break
#                 yield f"data: {json.dumps({'type': 'log', 'message': clean})}\n\n"

#     return StreamingResponse(stream(), media_type="text/event-stream")