# Security Policy

## Supported versions

TinyEngram is an educational project. Security fixes are applied to the latest `main` branch.

| Version | Supported |
|---|---|
| `0.1.x` / `main` | ✅ |
| Older commits | ❌ |

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Use GitHub's private vulnerability reporting:

<https://github.com/colorful-lollipop/tinyengram/security/advisories/new>

Include:

- a description of the issue and its impact;
- steps to reproduce, or a minimal proof of concept;
- affected files/versions and any suggested fix;
- whether you would like to be credited.

You can expect an initial response within 7 days. Because this is a volunteer-run teaching
project, timelines for fixes may vary; we will keep you informed.

## Scope

This repository contains no production service and no secrets. Relevant reports include:

- arbitrary code execution or path traversal in scripts;
- unsafe deserialization in checkpoints/utilities;
- supply-chain issues in dependencies or CI;
- accidentally committed credentials or personal data.

General correctness bugs, documentation issues, and model-performance questions should use the
public issue tracker instead.

## Safe usage

- Treat downloaded checkpoints and tokenizers as untrusted; prefer `safetensors` or trusted sources.
- Do not run `experiments/` with untrusted data paths.
- This project is not a safety-critical system and has not been evaluated for deployment.
