import json
import os
import re
from typing import Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - handled at runtime for clearer setup errors
    genai = None
    types = None


class GemmaChat:
    def __init__(
        self,
        question_type: int = 1,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: int = 500,
    ):
        if genai is None or types is None:
            raise ImportError("Install google-genai to use GemmaChat: pip install google-genai")

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY before using GemmaChat.")

        self.model = model or os.getenv("GEMMA_MODEL", "gemma-4-32b-it")
        self.max_output_tokens = max_output_tokens
        self.client = genai.Client(api_key=self.api_key)
        self.context = self.set_context(question_type)

    def set_context(self, question_type: int) -> str:
        base_context = (
            "You are a scientific medical assistant designed to synthesize responses "
            "from specific medical documents. Only use the information provided in the "
            "documents to answer questions. The first documents should be the most relevant. "
            "Do not use any other information except for the documents provided. "
            "When answering questions, always format your response as a JSON object with "
            "fields for 'response' and 'used_PMIDs'. Cite all PMIDs your response is based "
            "on in the 'used_PMIDs' field. Think step-by-step internally, but only return "
            "the JSON object."
        )

        question_specific_context = {
            1: " Provide a detailed answer to the question in the 'response' field.",
            2: " Your response should only be 'yes' or 'no'. If no relevant documents are found, return 'no_docs_found'.",
            3: " Choose between the given options 1 to 4 and return the chosen number as 'response'. If no relevant documents are found, return the number 5.",
            4: " Respond with keywords and list each keyword separately as a list element. If no relevant documents are found, return an empty list.",
        }

        return base_context + question_specific_context.get(question_type, "")

    def set_initial_message(self) -> List[dict]:
        return [{"role": "system", "content": self.context}]

    def create_chat(self, user_message: str, retrieved_documents: Dict) -> str:
        prompt = self._build_prompt(user_message, retrieved_documents)

        try:
            response_text = self._generate(prompt)
            response_data = self._parse_json(response_text)
            formatted_response = {
                "response": response_data.get("response"),
                "used_PMIDs": response_data.get("used_PMIDs", []),
                "retrieved_PMIDs": [doc["PMID"] for doc in retrieved_documents.values()],
            }
            return json.dumps(formatted_response)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _build_prompt(self, user_message: str, retrieved_documents: Dict) -> str:
        documents = {
            key: {
                "PMID": doc.get("PMID"),
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "relevance_score": doc.get("score"),
            }
            for key, doc in retrieved_documents.items()
        }
        return (
            f"Answer the following question: {user_message}\n\n"
            "Here are the documents:\n"
            f"{json.dumps(documents, ensure_ascii=False)}"
        )

    def _generate(self, prompt: str) -> str:
        config_kwargs = {
            "temperature": 0.0,
            "max_output_tokens": self.max_output_tokens,
            "system_instruction": self.context,
        }
        try:
            config = types.GenerateContentConfig(
                **config_kwargs,
                response_mime_type="application/json",
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception:
            config = types.GenerateContentConfig(**config_kwargs)
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

        return getattr(response, "text", "") or ""

    def _parse_json(self, response_text: str) -> dict:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise ValueError(f"Gemma response was not valid JSON: {response_text}")
            return json.loads(match.group(0))