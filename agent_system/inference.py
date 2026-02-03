from agent_system.inferencer import AKBClient, RecursiveInferencer
from agent_system.slm import SLMRunnerHF
from agent_system.tools import build_tool_registry
from agent_system.reconstructor import SubtaskRewriter, SLMRunnerHFAdapter

from dotenv import load_dotenv
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    # Instantiate dependencies
    akb = AKBClient()
    slm = SLMRunnerHF(model_id="Qwen/Qwen3-4B-Instruct-2507")

    rewriter_model = SLMRunnerHFAdapter(slm_runner_hf=slm)
    rewriter = SubtaskRewriter(model=rewriter_model)
    tools = build_tool_registry()

    # Create the inferencer with rewrite enabled
    inferencer = RecursiveInferencer(
        akb_client=akb,
        slm_runner=slm,
        max_subtasks=4,
        max_depth=5,
        max_retries_per_depth=1,
        subtask_rewriter=rewriter,
        rewrite_before_recursive=True,
        rewrite_retry_budget=1,
        rewrite_guidance="Make it more precise for automated tools.",
    )

    # Run a task
    original_task = """
        Here\'s a fun riddle that I think you\'ll enjoy.\n\nYou have been selected to play the final round of the hit new game show "Pick That Ping-Pong". 
        In this round, you will be competing for a large cash prize. Your job will be to pick one of several different numbered ping-pong balls, and then the game will commence. 
        The host describes how the game works.\n\nA device consisting of a winding clear ramp and a series of pistons controls the outcome of the game. 
        The ramp feeds balls onto a platform. The platform has room for three ping-pong balls at a time.
         The three balls on the platform are each aligned with one of three pistons. 
         At each stage of the game, one of the three pistons will randomly fire, ejecting the ball it strikes. 
         If the piston ejects the ball in the first position on the platform the balls in the second and third position on the platform each advance one space, and the next ball on the ramp advances to the third position. 
         If the piston ejects the ball in the second position, the ball in the first position is released and rolls away, the ball in the third position advances two spaces to occupy the first position, and the next two balls on the ramp advance to occupy the second and third positions on the platform. 
         If the piston ejects the ball in the third position, the ball in the first position is released and rolls away, the ball in the second position advances one space to occupy the first position, and the next two balls on the ramp advance to occupy the second and third positions on the platform.
         The ramp begins with 100 numbered ping-pong balls, arranged in ascending order from 1 to 100. 
         The host activates the machine and the first three balls, numbered 1, 2, and 3, advance to the platform. 
         Before the random firing of the pistons begins, you are asked which of the 100 balls you would like to pick. 
         If your pick is ejected by one of the pistons, you win the grand prize, $10,000.
        Which ball should you choose to maximize your odds of winning the big prize? Please provide your answer as the number of the ball selected.
    """
    res = inferencer.run(
        root_task=original_task,
        task=original_task,
        tools=tools,  # pass your tool registry
        difficulty_threshold=2.0,
        stop_on_first_failure=True,
    )
    #  nohup python -m agent_system.inference > inference.log 2>&1
    print("RESULT:")
    import pprint

    pprint.pprint(res)
