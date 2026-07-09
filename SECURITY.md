# Security Policy

## Supported Versions

mixle is pre-1.0 and releases frequently. Security fixes are made against the latest release only;
please upgrade to the newest version before reporting an issue if you are not already on it.

| Version | Supported |
| ------- | --------- |
| Latest  | ✅ |
| Older   | ❌ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for a security vulnerability. Instead, use
[GitHub's private vulnerability reporting](https://github.com/gmboquet/mixle/security/advisories/new)
for this repository, or email **grant.boquet@gmail.com** with:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal example, if possible).
- The mixle version and Python version affected.

You should expect an initial response within **5 business days**. Once a fix is confirmed, we will
coordinate a release and a public advisory, and credit the reporter unless anonymity is requested.

## Scope

mixle is a modeling/inference library, not a network service. In-scope issues include things like:
deserialization of untrusted model artifacts leading to code execution, dependency vulnerabilities
that affect mixle's own code paths, and unsafe handling of untrusted input data. Out of scope:
vulnerabilities in optional third-party backends (torch, Spark, Dask, Ray, MPI, ...) that are not
specific to how mixle uses them — please report those upstream.
