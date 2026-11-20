### Core modules ###
import os
from datetime import (
    datetime,
    timezone
)
from fastapi import (
    Depends,
    HTTPException,
    APIRouter,
    Request,
    status
)
from httpx import (
    AsyncClient,
    Response
)
from codecarbon import EmissionsTracker


### Type hints ###
from typing import Any
from ...types.tags import APITag


### Internal modules ###
from ..cores.dependencies import get_opensi_cosmic
from ...src.opensi_cosmic import OpenSICoSMIC
from ...types.json_schemas import (
    ChatHistorySchema,
    ChatHistorySchemaUpdate,
    ChatSessionPayload,
    LLM_ROLE
)
from ...utils.chat_history import (
    build_context_from_chat_history,
    build_messages_from_chat_history
)
from ...utils.log_tool import set_color



router: APIRouter = APIRouter(
    prefix="/api/v1/cosmic",
    tags=[APITag.cosmic]
)


CHAT_API_URL: str = os.getenv(
    "CHAT_API_URL",
    "http://backend:8000/api/v1/chatboxes/"
)
EMISSIONS_API_URL: str = os.getenv(
    "EMISSIONS_API_URL",
    "http://backend:8000/api/v1/emissions/"
)


async def _chat_session_create(
    client:     AsyncClient,
    user_id:    str,
    name:       str
) -> str:
    """Create an empty chatbox and return its id (used before QA needs a session)."""
    response: Response = await client.post(
        url=CHAT_API_URL,
        json={
            "user_id":  user_id,
            "name":     name,
            "details":  [],
        }
    )
    response.raise_for_status()
    return response.json()["created"]["id"]


async def _chat_session_patch(
    client:             AsyncClient,
    chat_session_id:    str,
    payload:            dict[str, Any]
) -> None:
    """Append the completed chat history block to an existing chatbox."""
    response: Response = await client.patch(
        url=f"{CHAT_API_URL}{chat_session_id}",
        json=payload
    )

    response.raise_for_status()
    return None


async def _chat_session_delete(
    client:             AsyncClient,
    chat_session_id:    str
) -> None:
    """Best-effort rollback for a chatbox created during a failed request."""
    try:
        response: Response = await client.delete(url=f"{CHAT_API_URL}{chat_session_id}")
        response.raise_for_status()

    except Exception as rollback_exc:
        print(
            set_color(
                status="error",
                information=f"[Chatbox Rollback] Failed to delete {chat_session_id}: {rollback_exc}"
            )
        )


class EmptyServiceError(Exception):
    """
    Raised when no active service is available for the Query Analyser.
    """


async def send_emissions_to_db(
    user_id: str,
    tracker: EmissionsTracker,
) -> None:
    """Send emissions data directly via POST request to database."""
    ed = tracker.final_emissions_data

    if ed is None:
        print(set_color(status="error", information="[Emissions DB] No emissions data available"))
        return

    emission_payload: dict[str, Any] = {
        "run_id":                   ed.run_id,
        "duration":                 ed.duration,
        "emissions":                ed.emissions,
        "emissions_rate":           ed.emissions_rate,
        "cpu_power":                ed.cpu_power,
        "gpu_power":                ed.gpu_power,
        "ram_power":                ed.ram_power,
        "cpu_energy":               ed.cpu_energy,
        "gpu_energy":               ed.gpu_energy,
        "ram_energy":               ed.ram_energy,
        "energy_consumed":          ed.energy_consumed,
        "water_consumed":           ed.water_consumed,
        "region":                   ed.region,
        "cloud_provider":           ed.cloud_provider,
        "cloud_region":             ed.cloud_region,
        "os":                       ed.os,
        "cpu_count":                ed.cpu_count,
        "cpu_model":                ed.cpu_model,
        "gpu_count":                ed.gpu_count,
        "gpu_model":                ed.gpu_model,
        "longitude":                ed.longitude,
        "latitude":                 ed.latitude,
        "ram_total_size":           ed.ram_total_size,
        "tracking_mode":            ed.tracking_mode,
        "cpu_utilization_percent":  ed.cpu_utilization_percent,
        "gpu_utilization_percent":  ed.gpu_utilization_percent,
        "ram_utilization_percent":  ed.ram_utilization_percent,
        "ram_used_gb":              ed.ram_used_gb,
        "on_cloud":                 ed.on_cloud,
        "pue":                      ed.pue,
        "wue":                      ed.wue,
        "user_id":                  str(user_id),
    }

    try:
        async with AsyncClient() as client:
            response: Response = await client.post(
                url=EMISSIONS_API_URL,
                json=emission_payload
            )

            response.raise_for_status()
            print(
                set_color(
                    status="info",
                    information=f"[Emissions DB] Successfully stored emissions for user {user_id}"
                )
            )
            return None

    except Exception as emission_exc:
        print(
            set_color(
                status="error",
                information=f"[Emissions DB] Failed to store emissions: {str(emission_exc)}"
            )
        )


@router.post(
    path="",
    status_code=status.HTTP_200_OK
)
async def process_cosmic(
    chat_session_data:  ChatSessionPayload,
    opensi_cosmic:      OpenSICoSMIC = Depends(get_opensi_cosmic)
):
    """
    Receive FE's partial chat session payload, run QA, then send the
    completed payload to the Chatboxes API.

    When a file is attached, FE pre-creates an empty chat session, uploads the
    file into that session, then sends the tag as part of user query. At this
    point, CoSMIC just simply need to parse it.
    """
    try:
        user_id:            str                             = str(chat_session_data.user_id)
        chat_history:       list[ChatHistorySchemaUpdate]   = chat_session_data.details[:-1]
        chat_history_block: ChatHistorySchemaUpdate         = chat_session_data.details[-1]

        chat_session_id: str | None = (
            str(chat_session_data.chat_session_id)
            if chat_session_data.chat_session_id
            else None
        )

        exist_chat_session: bool = False

        async with AsyncClient() as client:
            # A session must exist before QA can run; the FE normally creates
            # it up-front only when it needs to upload an attachment.
            if chat_session_id is None:
                chat_session_id = await _chat_session_create(
                    client=client,
                    user_id=user_id,
                    name=chat_session_data.name
                )
                exist_chat_session = True

            try:
                # The current inquiry cycle carries the leading `<files>` tag
                # when a file was attached. Strip it inline (legacy approach) so
                # QA keeps receiving bare question text, while the stored
                # `user_query` keeps the tag for the UI chip.
                stored_user_query:  str         = (chat_history_block.user_query or "")
                has_files:          bool        = stored_user_query.find("</files>") > -1
                file_refs:          list[str]   = []
                question:           str         = stored_user_query
                if has_files:
                    splits: list[str] = stored_user_query.split("</files>")
                    file_refs = [
                        ref.strip()
                        for ref in splits[0].split("<files>")[-1].split(",")
                        if ref.strip()
                    ]
                    question = splits[1]

                chat_history_context: str = build_context_from_chat_history(details=chat_history)
                history_messages: list[dict[str, str]] = build_messages_from_chat_history(
                    details=chat_history
                )

                # Start CodeCarbon emission tracking process (for General user queries).
                general_tracker: EmissionsTracker = EmissionsTracker(
                    project_name="cosmic-chat",
                    save_to_file=False,
                    allow_multiple_runs=True,
                    tracking_mode="process",  # track only the current process (not the whole machine)
                    log_level="error"
                )
                general_tracker.start()

                try:
                    (
                        llm_response,
                        _llm_raw_response,
                        llm_response_timestamp,
                        llm_input_token,
                        llm_output_token,
                        _llm_retrieve_score
                    ) = opensi_cosmic(
                        question=question,
                        context=chat_history_context,
                        history_messages=history_messages,
                        session_id=chat_session_id,
                        has_files=has_files,
                        user_id=user_id,
                        file_refs=file_refs
                    )
                finally:
                    general_emissions: float | None = general_tracker.stop()

                final_payload: ChatHistorySchema = ChatHistorySchema(
                    inquiry_cycle_id=chat_history_block.inquiry_cycle_id,   # pyright: ignore[reportArgumentType]
                    user_role=chat_history_block.user_role,                 # pyright: ignore[reportArgumentType]
                    user_query=stored_user_query,
                    query_create_on=chat_history_block.query_create_on,     # pyright: ignore[reportArgumentType]
                    llm_role=LLM_ROLE,
                    llm_response=llm_response,
                    response_create_on=llm_response_timestamp,
                    input_token=llm_input_token,
                    output_token=llm_output_token
                )

                await _chat_session_patch(
                    client=client,
                    chat_session_id=chat_session_id,
                    payload={
                        "user_id":  user_id,
                        "name":     chat_session_data.name,
                        "details":  [
                            final_payload.model_dump(
                                mode="json",
                                exclude_unset=True
                            )
                        ]
                    }
                )

                await send_emissions_to_db(
                    user_id=user_id,
                    tracker=general_tracker
                )
                if general_emissions is not None:
                    print(
                        set_color(
                            status="info",
                            information=f"[CodeCarbon] General chat query emissions: {general_emissions:.8f} kg CO₂"
                        )
                    )

            except Exception as fastapi_exc:
                if exist_chat_session:
                    await _chat_session_delete(
                        client=client,
                        chat_session_id=chat_session_id
                    )

                raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail={
                            "status": "500 - Internal Server Error",
                            "message": f"{fastapi_exc}"
                        }
                )

        return {
            "status": "success",
            "result": llm_response,
            "chat_session_id": chat_session_id
        }

    except HTTPException as http_exc:
        raise http_exc


    except EmptyServiceError as empty_service_exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": "404 - Not Found",
                "message": f"EmptyServiceError: {empty_service_exc}"
            }
        )


    except Exception as fastapi_exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "500 - Internal Server Error",
                "message": f"{fastapi_exc}"
            }
        )


@router.patch(
    path="",
    status_code=status.HTTP_200_OK
)
async def rebuild_cosmic(
    request:        Request,
    opensi_cosmic:  OpenSICoSMIC = Depends(get_opensi_cosmic)
):
    try:
        latest_configs: list[dict[str, dict[str, Any]]] = opensi_cosmic.get_configs()

        if not latest_configs:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "503 - API Services Unavailable",
                    "message:": "Could not reach Configurations API."
                }
            )

        else:
            # Similar trick to prevent having to perform expensive for-loop. Take a
            # look at `src/opensi_cosmic.py` [Line 85]
            latest_config_data:             dict[str, Any] = list(latest_configs[0].values())[0]

            latest_general_config_data:     dict[str, Any] = latest_config_data["general"]
            default_general_config_data:    dict[str, Any] = request.app.state.default_configs["general"]

            latest_qa_config_data:          dict[str, Any] = latest_config_data["query_analyser"]
            default_qa_config_data:         dict[str, Any] = request.app.state.default_configs["query_analyser"]

            if  (latest_general_config_data == default_general_config_data) \
            and (latest_qa_config_data == default_qa_config_data):
                # Accidentally click 'Save' button? No worries, nothing will
                # happens except a friendly "warning" meesage :)
                return {
                    "status": "success",
                    "message": "`OpenSICoSMIC()` stay the same due to exact config data found during update request."
                }

            else:
                # Clear cached memories from currently used SLMs first
                opensi_cosmic.quit()

                # Then re-state it again to allow `OpenSICoSMIC()` receiving new config data
                request.app.state.opensi_cosmic     = OpenSICoSMIC()
                request.app.state.default_configs   = latest_config_data

                return {
                    "status": "success",
                    "message": "`OpenSICoSMIC()` reconstructing due to new config data found during update request..."
                }


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )
