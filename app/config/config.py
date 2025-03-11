from pathlib import Path
from dotenv import dotenv_values

CONFIG_PATH = Path.cwd()
APP_PATH = CONFIG_PATH.parent
SERVICES_PATH = APP_PATH / "services"
ROOT_PATH = APP_PATH.parent
QA_RAG = True

openai_api_key = dotenv_values(ROOT_PATH / ".env").get("OPENAI_API_KEY")
model_name = "gpt-4o-mini"