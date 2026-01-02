import json
from datetime import datetime
from config import client


class RecruiterAgent:
    def __init__(self, prompt_path, log_path="logs/agent_logs.txt"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        self.log_path = log_path
        self.agent_name = self.__class__.__name__

    def recruit(self, goal):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Goal: {goal}"}
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        # Convert JSON string → Python dict
        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Recruiter output is not valid JSON")

        self._log_output(plan)

        return plan

    def _log_output(self, output):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] Agent: {self.agent_name}\n")
            log_file.write(json.dumps(output, indent=2))
            log_file.write("\n" + "-" * 50 + "\n")
