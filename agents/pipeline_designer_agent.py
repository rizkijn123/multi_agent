import json
import re
from datetime import datetime
from config import client
import os

class PipelineDesignerAgent:
    def __init__(self):
        self.agent_name = "Pipeline Designer Agent"
        self.prompt_path = "prompts/prompt_pipeline_designer.txt"
        self.log_path = "logs/agent_logs.txt"
        self.output_path = "logs/pipeline_strategy.json"
        self.feedback_path = "logs/evaluator_feedback.json"

        # Load sistem prompt
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = (
                "You are a pipeline designer AI. Based on the previous pipeline strategy and "
                "feedback from the evaluator, propose a new ML pipeline strategy in JSON format."
            )

    def run(self, goal):
        # Load pipeline history
        if os.path.exists(self.output_path):
            with open(self.output_path, "r", encoding="utf-8") as f:
                try:
                    pipeline_history = json.load(f)
                    if not isinstance(pipeline_history, list):
                        pipeline_history = [pipeline_history]
                except Exception:
                    pipeline_history = []
        else:
            pipeline_history = []

        last_pipeline = pipeline_history[-1] if pipeline_history else {}

        # Load evaluator feedback
        if os.path.exists(self.feedback_path):
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                try:
                    feedback_history = json.load(f)
                    feedback = feedback_history[-1] if feedback_history else {}
                except Exception:
                    feedback = {}
        else:
            feedback = {}

        suggestions = feedback.get("suggestions", [])
        goals_achieved = feedback.get("goals_achieved", True)

        # Prompt LLM
        llm_prompt = f"""
Goal: {goal}

Previous Pipeline Strategy:
{json.dumps(last_pipeline, indent=2)}

Evaluator Feedback:
{json.dumps(feedback, indent=2)}

Instructions:
- Generate a new pipeline strategy in JSON format as an array of objects.
- Each object must have keys: "pipeline_strategy" and "justification".
- Implement suggestions from the evaluator.
- Keep improvements incremental; do not remove effective parts of the previous pipeline.
- Format example:
[
  {{
    "pipeline_strategy": {{
      "preprocessing": ["remove_special_characters","convert_to_lowercase"],
      "feature_extraction": "TF-IDF",
      "model": "RandomForestClassifier",
      "training_strategy": "StratifiedKFoldCrossValidation",
      "evaluation_metric": ["accuracy","f1_score","precision","recall"]
    }},
    "justification": "Explain why these changes were made."
  }}
]

Suggestions to implement: {json.dumps(suggestions, indent=2)}
"""

        # Generate output via GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": llm_prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )

        raw_output = response.choices[0].message.content

        try:
            json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if json_match:
                new_strategy = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON array found in LLM output")
        except Exception:
            new_strategy = [{
                "pipeline_strategy": {},
                "justification": f"LLM output could not be parsed. Raw output:\n{raw_output}"
            }]

        # Append ke history pipeline
        pipeline_history.append(new_strategy[0])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_history, f, indent=2)

        # Logging
        self._log(new_strategy)

        print(f"[PIPELINE DESIGNER] New strategy appended. Total iterations: {len(pipeline_history)}")

    # LOGGING
    def _log(self, output):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Agent: {self.agent_name}\n")
            f.write("New pipeline strategy appended to pipeline_strategy.json\n")
            f.write(json.dumps(output, indent=2))
            f.write("\n" + "=" * 60 + "\n")
