# ROUTERA monorepo targets for the Disruption Sentinel (agents/sentinel). Everything runs offline.
# No make on Windows? Run the command after each target name directly.

PY ?= python

.PHONY: routera routera-test routera-lint sentinel-demo sentinel-replay sentinel-check sentinel-test sentinel-lint

# The whole ROUTERA control room at http://localhost:8000: Sentinel + shelf-life clocks + planner
# + approval with a hash-chained audit. Plays the India-inside scenario when the page opens.
routera:
	$(PY) -m routera

# ROUTERA engine and app tests, then the Sentinel's own.
routera-test:
	$(PY) -m pytest routera/tests agents/sentinel/tests -q

routera-lint:
	$(PY) -m ruff check routera agents/sentinel

# Serve the prototype UI at http://localhost:8001 and replay the India-inside Nashik corridor
# scenario when the page opens (pick the Gulf and other global shocks from the dropdown).
sentinel-demo:
	$(PY) -m agents.sentinel.api --demo nashik_corridor_strike --port 8001

# Print every Sentinel decision for the India-inside scenario at 60x.
sentinel-replay:
	$(PY) -m agents.sentinel.replay nashik_corridor_strike --speed 60

# Replay all five scenarios instantly and compare with data/expected.
sentinel-check:
	$(PY) -m agents.sentinel.replay all --speed 0 --check

sentinel-test:
	$(PY) -m pytest agents/sentinel/tests -q

sentinel-lint:
	$(PY) -m ruff check agents/sentinel
	$(PY) -m ruff format --check agents/sentinel
