# Instruction-Count First

## Default Objective

Unless the staged task explicitly says otherwise, optimize for lower instruction count on the hot path first.

## What Counts As Instruction-Count Work

- redundant branches
- repeated loads and stores
- repeated pointer chasing
- duplicated bookkeeping
- avoidable synchronization overhead
- repeated allocation, copy, and serialization work
- control-flow churn that does not change externally visible behavior

## Required Framing

Every research note, plan, code review, and tester report should answer:

- where instructions are currently being spent
- what exact mechanism is expected to reduce them
- what correctness or lifecycle constraints limit the change
- what artifact will later confirm or falsify the expected win

## Tie-Breakers

When two ideas look similar, prefer the one that:

1. touches the measured hot path more directly
2. removes instructions with less concurrency or lifetime risk
3. is easier to validate with Build MCP, Auto-Test MCP, and trace artifacts
