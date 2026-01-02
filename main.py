from agents.recruiter_agent import RecruiterAgent
from orchestrator import Orchestrator

GOAL = "Building an online gambling text comment classification model with a pipeline that can produce a good model"

if __name__ == "__main__":
    recruiter = RecruiterAgent(
        prompt_path="prompts/prompt_recruiter.txt",
        log_path="logs/agent_logs.txt"
    )

    plan = recruiter.recruit(GOAL)

    orchestrator = Orchestrator(plan["execution_order"])
    orchestrator.run(GOAL)
