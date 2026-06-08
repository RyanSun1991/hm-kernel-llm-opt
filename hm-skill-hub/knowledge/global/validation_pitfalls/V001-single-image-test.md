# Validation Pitfalls

Gotchas in measurement / A-B comparison / flashing. Stable id prefix `V`.

---

### V001 — single-image test mistaken for A/B
- **lesson**: A test result on the feature image alone is not a verdict — it lacks
    the stock baseline and proves nothing about delta.
- **kind**: validation_pitfall
- **applies_when**: tester returns results without paired stock+feature runs
- **do_or_dont**: "watch out for: any verdict citing only feature-image numbers;
    require both flash+test cycles and a delta computation"
- **tags**: [validation, ab-test, flash]
- **evidence**:
    - {kind: review, ref: reviews/hyperhold_io__iter1_code_review.md}
- **confidence**: confirmed
- **added_on**: 2026-04-15
- **added_by**: kernel-code-reviewer
- **status**: active
