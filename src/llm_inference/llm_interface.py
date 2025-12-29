"""
Base interface and implementations for different LLM backends.
This allows the co-scientist system to work with any LLM provider.

This is the ProtoGnosis LLM interface integrated into Jnana.
"""
import os
import json
import requests
import openai
from abc import ABC, abstractmethod
from transformers import pipeline
from typing import Dict, List, Optional, Any, Union, Tuple
from .inference_auth_token import get_access_token

import logging

# Configure logger
logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Abstract base class defining the interface for LLM providers."""

    def __init__(self, model: str, model_adapter: Optional[Dict] = None):
        """Initialize the LLM interface."""
        self.model = model
        self.model_adapter = model_adapter
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The main prompt text
            system_prompt: Optional system instructions
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum response length

        Returns:
            Generated text from the LLM
        """
        pass

    @abstractmethod
    def generate_with_json_output(self, prompt: str, json_schema: Dict,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7, max_tokens: int = 1024) -> Union[Dict, Tuple[Dict, int, int]]:
        """
        Generate a response that conforms to a specific JSON schema.

        Args:
            prompt: The main prompt text
            json_schema: Dictionary describing the expected JSON structure
            system_prompt: Optional system instructions
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum response length

        Returns:
            Tuple containing:
            - Response as a structured dictionary matching the given schema
            - Number of prompt tokens used
            - Number of completion tokens used
        """
        pass


class OpenAILLM(LLMInterface):
    """Implementation for OpenAI's API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", model_adapter: Optional[Dict] = None):
        """
        Initialize the OpenAI LLM interface.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            model: Model identifier to use
            model_adapter: Optional configuration for model adaptation
        """
        super().__init__(model, model_adapter)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenAI API key provided")

        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response from OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def generate_with_json_output(self, prompt: str, json_schema: Dict,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7, max_tokens: int = 1024) -> Tuple[Dict, int, int]:
        """Generate a structured JSON response from OpenAI."""
        schema_prompt = f"""
        Your response must be formatted as a JSON object according to this schema:
        {json_schema}

        Ensure your response can be parsed by Python's json.loads().
        """

        full_prompt = f"{prompt}\n\n{schema_prompt}"
        system = system_prompt or "You output only valid JSON according to the specified schema."

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": full_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        # Extract JSON string and parse
        try:
            content = response.choices[0].message.content
            parsed_response = json.loads(content)

            # Get token counts from OpenAI response
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            return parsed_response, prompt_tokens, completion_tokens
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Response was: {response.choices[0].message.content}")


class alcfLLM(LLMInterface):
    """Implementation for OpenAI's API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b", model_adapter: Optional[Dict] = None):
        """
        Initialize the OpenAI LLM interface.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            model: Model identifier to use
            model_adapter: Optional configuration for model adaptation
        """
        super().__init__(model, model_adapter)
        if 'metis' in model:
            base_url = "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
            self.model = model.replace('metis/', '')
        else:
            base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
            self.model = model

        self.model_adapter = model_adapter
        api_key = get_access_token()

        self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response from OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response

    def generate_reponse(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response from OpenAI."""
        if system_prompt is None:
            system_prompt = "No reasoning. Final answer only."
        else:
            system_prompt = system_prompt + "No reasoning. Final answer only."
        response = self.generate(prompt, system_prompt, temperature, max_tokens)

        while response.choices[0].message.content is None:
            if response.choices[0].message.reasoning_content is not None:
                prompt = f"USER PROMPT: {prompt}\n\nASSISTANT REASONING: {response.choices[0].message.reasoning_content}\n\nNow provide the final answer only."
                response = self.generate(prompt, system_prompt, temperature, max_tokens)
        return response

    def generate_with_json_output(self, prompt: str, json_schema: Dict,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7, max_tokens: int = 1024) -> Tuple[Dict, int, int]:
        """Generate a structured JSON response from OpenAI."""
        schema_prompt = f"""
        Your response must be formatted as a JSON object according to this schema:
        {json_schema}

        Ensure your response can be parsed by Python's json.loads().
        """

        system_prompt = system_prompt or schema_prompt

        response = self.generate_reponse(prompt, system_prompt, temperature, max_tokens)

        # Extract JSON string and parse
        try:
            content = response.choices[0].message.content
            parsed_response = json.loads(content)

            # Get token counts from OpenAI response
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            return parsed_response, prompt_tokens, completion_tokens
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Response was: {response.choices[0].message.content}")


class vllmLLM(LLMInterface):
    """Implementation for OpenAI's API."""

    def __init__(self, api_key: str = 'EMPTY', model: str = "openai/gpt-oss-120b", model_adapter: Optional[Dict] = None):
        """
        Initialize the OpenAI LLM interface.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            model: Model identifier to use
            model_adapter: Optional configuration for model adaptation
        """
        super().__init__(model, model_adapter)
        self.model = model
        self.model_adapter = model_adapter

        self.client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key=api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response from OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def generate_with_json_output(self, prompt: str, json_schema: Dict,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7, max_tokens: int = 1024) -> Tuple[Dict, int, int]:
        """Generate a structured JSON response from OpenAI."""
        schema_prompt = f"""
        Your response must be formatted as a JSON object according to this schema:
        {json_schema}

        Ensure your response can be parsed by Python's json.loads().
        """

        full_prompt = f"{prompt}\n\n{schema_prompt}"
        system = system_prompt or "You output only valid JSON according to the specified schema."

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": full_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        # Extract JSON string and parse
        import json
        try:
            content = response.choices[0].message.content
            parsed_response = json.loads(content)

            # Get token counts from OpenAI response
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            return parsed_response, prompt_tokens, completion_tokens
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Response was: {response.choices[0].message.content}")


class huggingfaceLLM(LLMInterface):
    """Implementation for hugging face API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b", model_adapter: Optional[Dict] = None):
        """
        Initialize the OpenAI LLM interface.

        Args:
            api_key: no longer used, kept for compatibility
            model: Model identifier to use
            model_adapter: Optional configuration for model adaptation
        """
        super().__init__(model, model_adapter)

        self.model_adapter = model_adapter

        self.client = pipeline(
            "text-generation",
            model=model,
            torch_dtype="auto",
            device_map="auto"  # Automatically place on available GPUs
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response from OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client(
            messages,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        last_message = response[0]['generated_text'][-1]
        # final_content = last_message.get('content', last_message.get('reasoning_content'))

        return last_message


    def generate_reponse(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response from OpenAI."""
        if system_prompt is None:
            system_prompt = "No reasoning. Final answer only."
        else:
            system_prompt = system_prompt + "No reasoning. Final answer only."
        response = self.generate(prompt, system_prompt, temperature, max_tokens)

        return response

    def generate_with_json_output(self, prompt: str, json_schema: Dict,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7, max_tokens: int = 1024) -> Tuple[Dict, int, int]:
        """Generate a structured JSON response from OpenAI."""
        schema_prompt = f"""
        Your response must be formatted as a JSON object according to this schema:
        {json_schema}

        Ensure your response can be parsed by Python's json.loads().
        """

        system_prompt = system_prompt or schema_prompt

        response = self.generate_reponse(prompt, system_prompt, temperature, max_tokens)

        # Extract JSON string and parse
        try:
            parsed_response = parse_hf_json_response(response['content'])

            # Get token counts from OpenAI response
            prompt_tokens = 0 # response.usage.prompt_tokens
            completion_tokens = 0 # response.usage.completion_tokens

            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            return parsed_response, prompt_tokens, completion_tokens
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Response was: {response['content']}")

def parse_hf_json_response(response: str) -> Dict:
    """Parse a JSON response from Hugging Face LLM output."""
    if '```json' in response:
        json_str = response.split('```json')[1].rsplit('```')[0]
    elif '```' in response:
        json_str = response.split('```')[1]
    else:
        json_str = response.split('assistantfinal')[1].rsplit('```')[0]
    return json.loads(json_str)



def typify_schema(in_schema):
    out_schema = {}

    for k, v in in_schema.items():
        # logging.info((k, v, type(k), type(v)))

        if isinstance(v, list):
            out_schema[k] = {
                "type": "array",
                "items": {"type": v[0]}
            }

        elif isinstance(v, dict):
            out_schema[k] = typify_schema(v)

        else:
            out_schema[k] = {"type": v}

    return out_schema


def translate_cerebras_schema(in_schema):
    out_schema = {
        "name": "coscientist_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "required": list(in_schema.keys())
        }
    }

    typified_schema = typify_schema(in_schema)

    from jsonschema import validate, ValidationError, SchemaError

    try:
        validate(instance={}, schema=typified_schema)
    except SchemaError as e:
        logging.error(f"JSON schema is invalid: {e}.")
        raise e
    except Exception as e:
        logging.error(f"Other JSON schema issue: {e}")
        raise e

    out_schema["schema"]["properties"] = typified_schema

    return out_schema

