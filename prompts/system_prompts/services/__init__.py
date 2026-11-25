from pathlib import Path


### Internal modules ###
from ....utils.env import PROJECT_ROOT


# Persona/service name -> system prompt file. Only services with a text persona
# are listed; other services rely on their SLM prompt class.
PROMPT_FILES: dict[str, Path] = {
    "academic_governance": PROJECT_ROOT / "prompts" / "system_prompts" / "services" / "academic_governance" / "academic_governance.txt",
    "document_qa": PROJECT_ROOT / "prompts" / "system_prompts" / "services" / "general_qa_answering" / "document_qa.txt",
    "system_information": PROJECT_ROOT / "prompts" / "system_prompts" / "services" / "system_information" / "system_information.txt",
}