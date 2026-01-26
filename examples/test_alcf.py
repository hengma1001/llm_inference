from llm_inference.llm_interface import alcfLLM
import logging

# Configure logger
logger = logging.getLogger(__name__)

llm = alcfLLM(model="openai/gpt-oss-20b")

prompt = "application of machine learning in cancer diagnosis"


schema = {
            "hypothesis": {
                "title": "string",
                "content": "string",
                "summary": "string",
                "key_novelty_aspects": ["string"],
                "testable_predictions": ["string"]
            },
            "explanation": "string",
            "generation_strategy": "string"
        }


responese, _, _ = llm.generate_with_json_output(
    prompt=prompt,
    json_schema=schema,
    max_tokens=5000,
)

print(responese)


