from agents.pipeline_designer_agent import PipelineDesignerAgent
from agents.ml_engineer_agent import MLEngineerAgent
from agents.evaluator_agent import EvaluatorAgent
import time

class Orchestrator:
    def __init__(self, execution_order):
        self.execution_order = execution_order

        self.agent_map = {
            "Pipeline Designer Agent": PipelineDesignerAgent,
            "ML Engineer Agent": MLEngineerAgent,
            "Evaluator Agent": EvaluatorAgent
        }

    def run(self, goal):
        iteration_count = 0
        goals_achieved = False

        print("\n=== ORCHESTRATION START ===\n")

        while not goals_achieved:
            iteration_count += 1
            print(f"\n--- Orchestration Iteration {iteration_count} ---\n")

            for agent_name in self.execution_order:
                agent_class = self.agent_map.get(agent_name)
                if not agent_class:
                    raise ValueError(f"Agent {agent_name} not supported yet")

                print(f"[ORCHESTRATOR] Running {agent_name}")
                agent = agent_class()
                if agent_name == "Evaluator Agent":
                    feedback = agent.run(goal)
                    goals_achieved = feedback.get("goals_achieved", False)
                else:
                    agent.run(goal)

                print(f"[ORCHESTRATOR] {agent_name} finished\n")

            if iteration_count < 2:
                goals_achieved = False

            if not goals_achieved:
                print(f"[ORCHESTRATOR] Goals not achieved yet, preparing next iteration...\n")
                time.sleep(1)

        print(f"\n=== ORCHESTRATION END: Goals achieved in iteration {iteration_count} ===")
