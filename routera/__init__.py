"""ROUTERA: saves perishable farm exports when routes, markets or tariffs break.

One real agent (the Disruption Sentinel in agents/sentinel) senses; everything after it is
deterministic: shelf_life, reroute, market, compliance and planner score the options, a human
approves, and audit keeps a hash-chained record. orchestrator wires them; app serves the UI.
"""
