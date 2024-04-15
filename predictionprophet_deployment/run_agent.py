"""
Entrypoint for running the agent in GKE.
If the agent adheres to PMAT standard (subclasses DeployableAgent), 
simply add the agent to the `RunnableAgent` enum and then `RUNNABLE_AGENTS` dict.

Can also be executed locally, simply by running `python predictionprophet_deployment/run_agent.py <agent> <market_type>`.
"""

from enum import Enum

import typer
from prediction_market_agent_tooling.markets.markets import MarketType

from predictionprophet_deployment.agents.prophet_agent.deploy import (
    DeployableOlasEmbeddingOAAgent,
    DeployablePredictionProphetGPT3Agent,
    DeployablePredictionProphetGPT4TurboFinalAgent,
    DeployablePredictionProphetGPT4TurboPreviewAgent,
)


class RunnableAgent(str, Enum):
    prophet_gpt3 = "prophet_gpt3"
    prophet_gpt4 = "prophet_gpt4"
    prophet_gpt4_final = "prophet_gpt4_final"
    olas_embedding_oa = "olas_embedding_oa"


RUNNABLE_AGENTS = {
    RunnableAgent.prophet_gpt3: DeployablePredictionProphetGPT3Agent,
    RunnableAgent.prophet_gpt4: DeployablePredictionProphetGPT4TurboPreviewAgent,
    RunnableAgent.prophet_gpt4_final: DeployablePredictionProphetGPT4TurboFinalAgent,
    RunnableAgent.olas_embedding_oa: DeployableOlasEmbeddingOAAgent,
}


def main(agent: RunnableAgent, market_type: MarketType) -> None:
    RUNNABLE_AGENTS[agent]().run(market_type)


if __name__ == "__main__":
    typer.run(main)
