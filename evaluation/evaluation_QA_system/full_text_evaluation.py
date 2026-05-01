import os
import re

from google import genai
from google.genai import types


class evaluateResponseGemma:
    def __init__(self, response, answer, api_key=None, model=None):
        self.response = response
        self.correct_answer = answer
        self.model = model or os.getenv("GEMMA_MODEL", "gemma-4-32b-it")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY before using evaluateResponseGemma.")
        self.client = genai.Client(api_key=self.api_key)
        self.context = self.set_context()

    def set_context(self) -> str:
        return (
            "You will evaluate a response by comparing it to an expert's optimal answer in the biomedical domain. "
            "The evaluation process should include the following steps:"
            "1. Identification of key terms and concepts in both the provided response and the expert's optimal answer. "
            "2. Assessment of the context of the used terms and concepts in the response and the expert's answer. "
            "3. Determination of the accuracy and completeness of the provided response. "
            "Score the response on a scale from 0 to 10, where 0 means completely no overlap with the expert's answer and 10 means a perfect match."
            "Provide only the discrete numerical score as your response."
        )
    
    def set_initial_message(self):
        return [{"role": "system", "content": self.context}]
    
    def get_evaluation(self) -> float:
        messages = self.set_initial_message()
        messages.append({"role": "user", "content": f"Response: {self.response}"})
        messages.append({"role": "user", "content": f"Correct answer: {self.correct_answer}. Please score the response above from 0 to 1 based on its accuracy and completeness."})
        try:
            prompt = "\n".join(message["content"] for message in messages)
            completion = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.context,
                    temperature=0.0,
                    max_output_tokens=16,
                ),
            )

            response_content = getattr(completion, "text", "") or ""
            try:
                score_match = re.search(r"\d+(?:\.\d+)?", response_content)
                score = float(score_match.group(0)) if score_match else 0
            except ValueError:
                score = 0
        except Exception as e:
            print(f"An error occurred during response evaluation: {e}")
            score = 0

        if score > 1:
            score = score / 10
        return max(0, min(score, 1))


evaluateResponseGPT = evaluateResponseGemma


if __name__ == "__main__":
    response = "Standard treatment for type 2 diabetes is insulin injections and does emphasize lifestyle changes or oral medications like metformin."
    correct_answer = "The standard treatment for type 2 diabetes involves lifestyle modifications such as diet and exercise, complemented by medications like metformin to regulate blood sugar levels."
    evaluator = evaluateResponseGemma(response, correct_answer)
    score = evaluator.get_evaluation()
    print(f"Response score: {score}")