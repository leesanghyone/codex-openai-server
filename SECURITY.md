# Security Policy

## Supported use

`codex_openai_server` is intended for local use on a trusted machine or private network segment. It is not positioned as a hardened multi-tenant or public internet-facing service.

Security issues that affect that intended use are in scope. Misconfiguration risks called out in the README, such as exposing the server directly to the public internet without additional controls, should still be reported if you find a concrete bypass or unexpected impact.

## Reporting a vulnerability

Please do not open a public GitHub issue for suspected vulnerabilities.

Use GitHub's private vulnerability reporting for this repository if it is available. If private reporting is unavailable, contact the maintainer through GitHub before sharing details publicly.

Include:

- a short description of the issue and likely impact
- affected version or commit
- reproduction steps or a minimal proof of concept
- any required environment details, including whether Docker or a local Python run was involved

Do not include live secrets, `auth.json`, bearer tokens, or other sensitive credentials in the report.

## Response expectations

Reports are handled on a best-effort basis. Once a report is confirmed, the goal is to coordinate a fix and disclosure responsibly.
