import json
import os
import subprocess
import sys
from datetime import datetime
from config import client
import re
import time

class MLEngineerAgent:
    def __init__(self):
        self.agent_name = "ML Engineer Agent"
        self.prompt_path = "prompts/prompt_ml_engineer.txt"
        self.log_path = "logs/agent_logs.txt"
        self.code_path = "generated_ml_pipeline.py"
        self.strategy_path = "logs/pipeline_strategy.json"
        self.eval_path = "logs/evaluation_results.json"
        self.max_attempts = 5

        # read prompt
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    # MAIN FUNCTION
    def run(self, goal):
        # Load strategy new pipeline
        if os.path.exists(self.strategy_path):
            with open(self.strategy_path, "r", encoding="utf-8") as f:
                pipeline_history = json.load(f)
                if isinstance(pipeline_history, list) and len(pipeline_history) > 0:
                    pipeline_strategy = pipeline_history[-1]
                elif isinstance(pipeline_history, dict):
                    pipeline_strategy = pipeline_history
                else:
                    pipeline_strategy = {}
        else:
            pipeline_strategy = {}

        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            print(f"[ML ENGINEER] ATTEMPT {attempt}")

            # Prompt alternatife: use new pipeline + error if any
            user_prompt = f"""
Goal:
{goal}

Pipeline Strategy:
{json.dumps(pipeline_strategy, indent=2)}

Instructions:
- The previous error (if any) is included below.
- Please generate Python code that adapts to any runtime errors.
- Ensure the code handles missing values, text preprocessing, or other runtime issues automatically.
- Return code that runs successfully in Python 3.11 environment.
- Implement the ML pipeline based on the given strategy
- Generate executable Python code using scikit-learn
- Load dataset from: data/data.csv
- Perform preprocessing, training, and evaluation
- Print evaluation metrics to stdout
- Proses columns name = text and label
"""
            if last_error:
                user_prompt += f"\nPrevious Error:\n{last_error}"

            # Generate code Python
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            full_code = response.choices[0].message.content.strip()

            # Log
            self._log(full_code, stdout="", stderr="")

            # clear only python
            executable_code = self._extract_python_code(full_code)

            with open(self.code_path, "w", encoding="utf-8") as f:
                f.write(executable_code)

            # run code
            result = subprocess.run(
                [sys.executable, self.code_path],
                capture_output=True,
                text=True
            )

            # handle runtime error dinamis
            handled = self._handle_runtime_error(result.stderr or "", executable_code)
            if handled:
                last_error = "Auto-handled runtime error"
                continue  # countinue without adding attempt

            # Log
            self._log(executable_code, result.stdout, result.stderr)

            if result.returncode == 0:
                print("[ML ENGINEER] SUCCESS: Pipeline executed without errors")

                # Save to JSON
                metrics = self._extract_metrics(result.stdout)
                if metrics:
                    os.makedirs(os.path.dirname(self.eval_path), exist_ok=True)
                    with open(self.eval_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2)
                    print(f"[ML ENGINEER] Evaluation results saved to {self.eval_path}")

                return

            # note error for next prompt
            last_error = (result.stderr or "") + "\n" + (result.stdout or "")
            print(f"[ML ENGINEER] ERROR detected, regenerating code...\n{last_error}")
            time.sleep(1)

        raise RuntimeError("ML Engineer Agent failed after max attempts")

    # DETECT PACKAGE NOT INSTALLED
    def _parse_missing_package(self, stderr_text: str):
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr_text)
        if match:
            return match.group(1)
        return None

    # HANDLE RUNTIME ERROR ADAPTIVE
    def _handle_runtime_error(self, stderr_text: str, code: str) -> bool:
        # received Log error
        if stderr_text.strip():
            print(f"[ML ENGINEER] Handling runtime error:\n{stderr_text}")
            self._log(code, stdout="", stderr=stderr_text)

        pkg = self._parse_missing_package(stderr_text)
        if pkg:
            print(f"[ML ENGINEER] Installing missing package: {pkg}")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
            return True

        if ("Resource punkt_tab not found" in stderr_text
            or "Resource punkt not found" in stderr_text
            or "LookupError" in stderr_text):
            try:
                import nltk
                print("[ML ENGINEER] Downloading NLTK punkt, punkt_tab, stopwords, wordnet")
                nltk.download('punkt_tab')
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('wordnet')
                return True
            except Exception as e:
                print(f"[ML ENGINEER] Failed to download NLTK resources: {e}")
                return False

        if "name 'csv' is not defined" in stderr_text:
            if "import csv" not in code:
                lines = code.splitlines()
                lines.insert(0, "import csv")
                new_code = "\n".join(lines)
                with open(self.code_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                print("[ML ENGINEER] Added missing import 'csv' at the top")
                return True

        return False


    # EXTRACT PURE PYTHON CODE
    def _extract_python_code(self, code: str) -> str:
        code = code.replace("```python", "").replace("```", "").strip()
        import_match = re.search(r"(^import\s.+?$|^from\s.+?$)", code, re.MULTILINE)
        if import_match:
            code = code[import_match.start():]
        lines = code.splitlines()
        pure_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("###") or stripped.lower().startswith("explanation:"):
                break
            pure_lines.append(line)
        return "\n".join(pure_lines)

    # EXTRACT METRICS FROM STDOUT
    def _extract_metrics(self, stdout: str):
        metrics = {}
        for line in stdout.splitlines():
            if "Accuracy" in line:
                metrics["Accuracy"] = float(line.split(":")[-1].strip())
            elif "F1 Score" in line:
                metrics["F1 Score"] = float(line.split(":")[-1].strip())
            elif "Precision" in line:
                metrics["Precision"] = float(line.split(":")[-1].strip())
            elif "Recall" in line:
                metrics["Recall"] = float(line.split(":")[-1].strip())
        return metrics if metrics else None

    # LOGGING
    def _log(self, code, stdout, stderr):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {self.agent_name}\n")
            f.write("=== CODE ===\n")
            f.write(code)
            f.write("\n=== STDOUT ===\n")
            f.write(stdout or "(empty)")
            if stderr:
                f.write("\n=== STDERR ===\n")
                f.write(stderr)
            f.write("\n" + "=" * 50 + "\n")
