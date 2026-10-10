"""Run the Discharge Planning Agent on one or more test-set patients.

    python demo_agent.py 0 1 2          # specific patient ids
    python demo_agent.py --top 3        # the 3 highest-risk patients in the test set
Set ANTHROPIC_API_KEY to let Claude drive the tools; otherwise the offline planner runs.
"""
import sys

import numpy as np

from src.agent import DischargeAgent

agent = DischargeAgent()
args = sys.argv[1:]
if args[:1] == ["--top"]:
    n = int(args[1]) if len(args) > 1 else 3
    sysm = agent.sys
    X = sysm.pre.transform(sysm.test_df[[c for c in sysm.ref_df.columns]])
    ids = list(np.argsort(-sysm._ensemble_fn(X))[:n])
else:
    ids = [int(a) for a in args] or [0]
print(f"Agent mode: {agent.mode}")
for pid in ids:
    agent.run(int(pid))
