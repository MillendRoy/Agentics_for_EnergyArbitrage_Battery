import os

from crewai import LLM
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


watsonx_crewai_llm = LLM(
    model=os.getenv("MODEL_ID"),
    base_url=os.getenv("WATSONX_URL"),
    project_id=os.getenv("WATSONX_PROJECTID"),
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
