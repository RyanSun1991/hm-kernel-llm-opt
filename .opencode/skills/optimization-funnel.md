# Optimization Funnel

## Ideation Protocol

1. generate exactly five ideas
2. drop repeated bad plans
3. rank first by likely instruction-count reduction on the hot path, then by risk and implementation cost
4. show only the top idea
5. wait for explicit approval
6. write the detailed plan only after approval

## Minimum Ranking Questions

- does the idea plausibly remove instructions from the measured or suspected hot path
- does it remove repeated work, branches, loads/stores, or redundant synchronization
- is the expected instruction-count gain likely to survive real build and runtime validation
- does it keep correctness, locking, lifetime, and logic boundaries defensible
