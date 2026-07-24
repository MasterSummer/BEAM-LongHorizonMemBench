# Online long-horizon execution track

The frozen v0.14 qualification artifacts remain replay-backed and are not
retroactively relabelled.  The online track is a separate execution artifact.

An episode is called `online_long_horizon_agent_execution` only if the policy
model makes at least 200 real structured calls and every counted call satisfies
the closed-loop contract:

1. the model selects an opaque implementation option;
2. the selected patch changes the mutable workspace and execution state;
3. the next request contains the resulting workspace/state/effect digest; and
4. a later policy step consumes the previous effect through the dependency
   chain.

At every session handoff the working context is cleared.  Native memory, when
enabled, is written and queried across the boundary independently.  The
evaluator records exact policy requests, responses, workspace hashes, state
digests, handoffs, storage IDs, retrieval IDs, and drift categories.  State IDs
and validity labels are never placed in the model-visible surface.

The command-line entry point is:

```bash
python -m lhmsb.longhorizon.online_cli \
  --dataset <frozen-dataset> \
  --config configs/experiments/systems_controlled_gpt_only_longitudinal_v014_shengsuanyun_writer.yaml \
  --condition workspace_only \
  --out <run-directory>
```

Each run writes one episode JSON, `MANIFEST.json`, `report.json`, and
`report.md`.  Attribution probes remain sparse and are sampled only at the
pre-registered critical checkpoints; they are not required to establish the
online-execution claim.
