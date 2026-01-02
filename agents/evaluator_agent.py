import json
from datetime import datetime
from config import client  # OpenAI client
import os

class EvaluatorAgent:
    def __init__(self):
        self.agent_name = "Evaluator Agent"
        self.eval_path = "logs/evaluation_results.json"
        self.strategy_path = "logs/pipeline_strategy.json"
        self.feedback_path = "logs/evaluator_feedback.json"
        self.log_path = "logs/agent_logs.txt"
        self.prompt_path = "prompts/evaluator_prompt.txt"
        self.iteration = 1

        # Load iterasi
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    prev_feedback_history = json.load(f)
                    if isinstance(prev_feedback_history, list) and prev_feedback_history:
                        last_feedback = prev_feedback_history[-1]
                        self.iteration = last_feedback.get("iteration", 0) + 1
                        self.prev_accuracy = last_feedback.get("metrics", {}).get("Accuracy", 0)
                    else:
                        self.prev_accuracy = 0
            except Exception:
                self.prev_accuracy = 0
        else:
            self.prev_accuracy = 0

        # read evaluator from file
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        else:
            # fallback prompt alternative
            self.prompt_template = (
                "You are an expert ML evaluator. Analyze the following pipeline and metrics:\n\n"
                "Iteration: {iteration}\n"
                "Goal: {goal}\n\n"
                "Pipeline Strategy:\n{pipeline_strategy}\n\n"
                "Evaluation Metrics:\n{metrics}\n\n"
                "Previous Best Accuracy: {prev_accuracy}\n\n"
                "Suggest concrete improvements in suggestions (preprocessing, algorithm, hyperparameters, data augmentation)"
            )

    def run(self, goal=None):
        # Load result evaluation ML Engineer
        try:
            with open(self.eval_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = {}


        # Load pipeline strategy
        try:
            with open(self.strategy_path, "r", encoding="utf-8") as f:
                pipeline_strategy = json.load(f)
        except FileNotFoundError:
            pipeline_strategy = {}

        example_feedback = {
            "iteration": 1,
            "summary": "Pipeline has basic preprocessing, accuracy can improve with hyperparameter tuning.",
            "suggestions": [
                "Add normalization for numeric features",
                "Try RandomForest instead of LogisticRegression",
                "Tune learning rate and number of estimators"
            ],
            "goals_achieved": False
        }

        # Generate prompt dari template file
        llm_prompt = (
                self.prompt_template.format(
                    iteration=self.iteration,
                    goal=goal or "Maximize accuracy",
                    pipeline_strategy=json.dumps(pipeline_strategy, indent=2),
                    metrics=json.dumps(metrics, indent=2),
                    prev_accuracy=self.prev_accuracy
                )
                + "\nExample of expected JSON output:\n"
                + json.dumps(example_feedback, indent=2)
        )

        # CALL LLM dan parse feedback
        feedback = self._get_llm_feedback(
            llm_prompt,
            pipeline_strategy=pipeline_strategy,
            metrics=metrics
        )

        # EVALUATION = ACCURATION > BEFORE?
        current_acc = metrics.get("Accuracy", 0)
        feedback["metrics"] = metrics
        feedback["iteration"] = self.iteration

        # Evaluasi goals_achieved
        if self.iteration == 1 or current_acc < self.prev_accuracy:
            feedback["goals_achieved"] = False
            feedback["summary"] += f" | Goals not achieved."

            # handling llm not give feedback
            if not feedback.get("suggestions"):
                feedback["suggestions"] = [
                    "Add more step good preprocessing",
                    "Change other algorithm",
                    "Use tunning hyperparameter."
                ]
        else:
            feedback["goals_achieved"] = True
            feedback[
                "summary"] += f" | Accuracy improved from {self.prev_accuracy:.4f} to {current_acc:.4f}. Goals achieved."

        # Load existing feedback history
        feedback_history = []
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    feedback_history = json.load(f)
                    if not isinstance(feedback_history, list):
                        feedback_history = []
            except Exception:
                feedback_history = []

        feedback_history.append(feedback)


        # Save feedback (history)
        os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(feedback_history, f, indent=2)

        # log
        self._log(feedback)
        print(f"[EVALUATOR] Iteration {self.iteration} feedback saved to {self.feedback_path}")

        return feedback

    # helper: get feedback from LLM (use retry if JSON invalid)
    def _get_llm_feedback(self, prompt, pipeline_strategy, metrics, max_retries=3):
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional ML pipeline evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens = 800
            )
            llm_output = response.choices[0].message.content.strip()
            try:
                feedback = json.loads(llm_output)
                return feedback
            except json.JSONDecodeError:
                # Retry regenerate JSON valid
                prompt = (
                    "Previous response was invalid JSON. "
                    "Regenerate JSON strictly with keys: iteration, summary, suggestions, goals_achieved. "
                    "Analyze the pipeline and metrics below:\n"
                    f"{pipeline_strategy}\nMetrics: {metrics}\nPrev Accuracy: {self.prev_accuracy}"
                )
        # if retry failed
        return {
            "iteration": self.iteration,
            "summary": "LLM feedback could not be parsed after retries.",
            "suggestions": [],
            "goals_achieved": False
        }

    # LOGGING
    def _log(self, feedback):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {self.agent_name} - Iteration {feedback.get('iteration', self.iteration)}\n")
            f.write("=== FEEDBACK ===\n")
            f.write(json.dumps(feedback, indent=2))
            f.write("\n" + "=" * 50 + "\n")
