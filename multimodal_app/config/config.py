from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import os

# Get the absolute path to the project root
CONFIG_PATH = Path(__file__).resolve().parent
APP_PATH = CONFIG_PATH.parent
ROOT_PATH = APP_PATH.parent

# Load environment variables from .env file
env_path = ROOT_PATH / ".env"
load_dotenv(dotenv_path=env_path)

TASK = {"text": "text", "qa_rag": "qa_rag", "multimodal": "multimodal"}

# Get API key using os.environ (safer approach as it also works with system environment variables)
openai_api_key = os.environ.get("OPENAI_API_KEY")
# Fallback to dotenv_values if not found in environment
if not openai_api_key:
    openai_api_key = dotenv_values(env_path).get("OPENAI_API_KEY")

model_name = "gpt-4o-mini"

if __name__ == "__main__":
    print(openai_api_key)
    print(model_name)
