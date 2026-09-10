# Evolution configuration examples

Copy these files to a local configuration before use. `discovery.example.json`
contains illustrative paths and owners; replace them with the intended repository
and stable source identity. An empty hotspot list is valid but supplies no runtime
evidence, so such candidates may require workbench investigation.

`correctness-policy.example.json` is only the validation portion of an approved
Plan. Its zero hashes and REPLACE identifiers are placeholders, not collected
evidence. Replace them with actual suite/environment hashes and execution IDs;
freeze the complete plan before running baseline/candidate checks.

See [the operations guide](../../docs/EVOLUTION_V2_OPERATIONS_CN.md) and generate
current contracts with `python -m hmopt.evolution.cli schema discovery` or
`schema correctness-policy`. These examples do not authorize execution or activate
research patterns.
