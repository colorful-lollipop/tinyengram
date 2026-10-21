## Summary

<!-- What does this PR change, and why? Link the issue if applicable. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New lab / feature
- [ ] Documentation
- [ ] Experiment / benchmark
- [ ] Tooling / CI

## For a new or changed lab

- [ ] Lecture added/updated under `docs/`
- [ ] Skeleton updated under `src/tinyengram/` with `TODO(Lx)`
- [ ] Reference implementation updated under `solutions/tinyengram/`
- [ ] Property-based tests added under `tests/`
- [ ] `scripts/grade.py` stage updated (if a new lab)
- [ ] Difficulty levels (`必做 / 进阶 / 挑战`) and question tags are present

## Verification

```bash
# Paste the commands you ran and their results.
TINYENGRAM_IMPL=solutions pytest -q
python scripts/check_docs.py
ruff check solutions tests experiments perf scripts
```

## Results / screenshots

<!-- For experiments or performance changes, include numbers and hardware. -->

## Checklist

- [ ] I did not tune the baseline unfairly to make Engram look better.
- [ ] I did not commit generated artifacts, checkpoints, datasets, or credentials.
- [ ] New claims are supported by a test, a measurement, or a cited source.
- [ ] I have read `CONTRIBUTING.md` and agree to the Code of Conduct.
