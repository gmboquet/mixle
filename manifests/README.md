# Repository manifests

These committed JSON files are generated compatibility and governance baselines, not runtime data:

- `api_manifest.json` records declared public exports;
- `maturity_manifest.json` records maturity tiers for public surfaces;
- `module_ownership.json` records review ownership; and
- `serialization_schema_manifest.json` records registered serialization type identifiers.

Run the corresponding generator in `scripts/` after changing a governed surface. Drift tests compare
the generated result with the reviewed file in this directory. Keeping the manifests together avoids
cluttering the repository root and makes their generated status explicit.
