# Validation Flight Check

## Validation Ladder

1. static sanity and dependency radius
2. plan review
3. code review
4. build validation (stock + feature images)
5. relay health check and device visibility
6. flash stock image to device and auto-test validation on stock
7. flash feature image to device and auto-test validation on feature
8. instruction-count or trace comparison (stock vs feature A/B delta)
9. independent summary and memory update

## Required Outputs

- validation plan
- plan review note
- code review note
- after-patch summary
- tester A/B validation summary (stock baseline + feature candidate + delta analysis + verdict)
