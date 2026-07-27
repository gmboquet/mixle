# Repository manifests

These committed JSON files are generated compatibility and governance baselines, not runtime data:

- `api_manifest.json` records declared public exports;
- `maturity_manifest.json` records maturity tiers for public surfaces;
- `module_ownership.json` records review ownership; and
- `stable_api_signatures.json` records parameter order, kinds, defaults, and annotations for every
  callable owned by an explicitly stable module; and
- `serialization_schema_manifest.json` records exact dependency-attested generation profiles and a
  version/stability/codec/migration record for every registered serialization type;
- `development_policy.json` is the authoritative release-target, maturity, compatibility, and
  validation policy used to render contributor instructions; and
- `public_claims.json` records material public claims, their evidence grades, and the exact public
  prose occurrences that are permitted by the claim-hygiene gate.

Run the corresponding generator in `scripts/` after changing a governed surface. Drift tests compare
the generated result with the reviewed file in this directory. Keeping the manifests together avoids
cluttering the repository root and makes their generated status explicit.

Serialization profiles are updated independently and explicitly:

``python scripts/gen_schema_manifest.py --profile base``
``python scripts/gen_schema_manifest.py --profile full``

The generator refuses a profile whose required imports are unavailable and never overwrites the other
profile from a partial environment.

Run ``python scripts/check_public_claims.py`` after changing public prose or release evidence. The
checker scans README, changelog, Sphinx prose, example module docstrings, and committed benchmark
results. New quantitative or comparative product claims fail until they are either removed or
registered with retained evidence.
