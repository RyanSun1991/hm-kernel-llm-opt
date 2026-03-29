# Research Discipline

## Non-Negotiable Order

1. Sequential Thinking MCP
2. Kernel Index MCP
3. local file reading
4. design doc update
5. instruction-count hypothesis update
6. optimization only after the model is stable

## Minimum Questions

- what are the entry points
- what data is protected
- what ownership or lifecycle boundaries exist
- what cross-file dependencies matter
- what is likely hot versus incidental
- where instruction count is likely being spent
- which repeated work, branches, loads/stores, synchronization, or copying dominate the hot path
- what proof artifact can later validate an instruction-count win
