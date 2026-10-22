"""PerovAgents — agentic framework for perovskite solar cell research.

Public API:
    from perovagents import Agent
    agent = Agent()
    answer = agent.answer("What is the average PCE of devices using SnO2?")

Entry points:
    CLI:    python -m perovagents "question"
    Demo:   python examples/demo_agent.py
"""
from .agent import Agent, Plan, PlanStep, PlanResult
from .core import default_registry, Tool

__version__ = "0.1.0"

__all__ = ["Agent", "Plan", "PlanStep", "PlanResult", "default_registry",
           "Tool", "__version__"]