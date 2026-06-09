## Candidate source

- [ ] Local sediment bundle path / run id is listed.
- [ ] Candidate records declare source evidence.

## Engine classification

- [ ] Knowledge changes went through dedup/conflict review.
- [ ] Skill changes include eval scorecard and bounded edit manifest.

## Safety gates

- [ ] `python tools/lint.py`
- [ ] `python tools/redact.py --check`
- [ ] `python tools/dedup.py --fail-on-conflict <candidate>` if knowledge changed.

## Reviewers

- [ ] Domain reviewer approved.
- [ ] Process / hub governance reviewer approved.
