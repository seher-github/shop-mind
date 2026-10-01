"""The only file that knows about Groq. Change the model here if needed."""
from crewai import LLM

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-20b"  # model name exactly as Groq lists it


def get_llm(api_key: str, base_url: str = GROQ_BASE_URL) -> LLM:
    # Same idea as the OpenAI(...) + client.responses.create(...) snippet:
    # Groq speaks the OpenAI protocol, so we point CrewAI's OpenAI client at Groq
    # and use the Responses API.
    #  - provider="openai"  keeps the model name "openai/gpt-oss-20b" untouched
    #  - reasoning_effort="low" keeps this reasoning model fast and cheap
    return LLM(
        model=MODEL,
        provider="openai",
        api="responses",
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_tokens=1500,
        reasoning_effort="low",
    )
