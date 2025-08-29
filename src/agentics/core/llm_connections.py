import os

from crewai import LLM
from dotenv import load_dotenv
from openai import AsyncOpenAI
from loguru import logger

#Turbo Models gpt-oss:20b, deepseek-v3.1:671b

load_dotenv()

available_llms = {}

ollama_turbo_llm = LLM(
    host="https://ollama.com",
    headers={'Authorization': os.getenv("OLLAMA_TURBO_API_KEY")},
    model='ollama/gpt-oss:20b',
    )

gemini_llm  = LLM(
    model=os.getenv("GEMINI_MODEL_ID"),
    temperature=0.7,
)

ollama_llm = LLM(
    model=os.getenv("OLLAMA_MODEL_ID"),
    base_url="http://localhost:11434"
)



openai_llm = LLM(
    model=os.getenv("OPENAI_MODEL_ID"), # call model by provider/model_name
    temperature=0.8,
    top_p=0.9,
    stop=["END"],
    api_key=os.getenv("OPENAI_API_KEY"),
    seed=42
)

watsonx_llm = LLM(
    model=os.getenv("MODEL_ID"),
    base_url=os.getenv("WATSONX_URL"),
    project_id=os.getenv("WATSONX_PROJECTID"),
    api_key=os.getenv("WATSONX_APIKEY"),
    max_tokens=8000,
    temperature=0.9,
)

vllm_llm = AsyncOpenAI(
    api_key="EMPTY",
    base_url=os.getenv("VLLM_URL"),
    default_headers={
        "Content-Type": "application/json",
    },
)

vllm_crewai = LLM(
    model=os.getenv("VLLM_MODEL_ID"),
    api_key="EMPTY",
    base_url=os.getenv("VLLM_URL"),
    max_tokens=8000,
    temperature=0.0,
)

logger.debug("AGENTICS is connecting to the following LLM API providers:")
i =0
if os.getenv("WATSONX_APIKEY"):
    logger.debug(f"{i} - WatsonX")
    available_llms["watsonx"] = watsonx_llm
    i +=1
if os.getenv("GEMINI_API_KEY"):
    available_llms["gemini"] =gemini_llm
    logger.debug(f"{i} - Gemini")
    i +=1
if os.getenv("OPENAI_API_KEY"):
    available_llms["openai"] = openai_llm
    logger.debug(f"{i} - OpenAI")

logger.debug("Please add API keys in .env file to add or disconnect providers.")
