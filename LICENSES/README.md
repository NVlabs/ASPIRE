# Licensing and Compliance

This directory is the central index for ASPIRE licensing, third-party
attribution, source provenance, and release-compliance evidence.

## Project license

- [`../LICENSE`](../LICENSE) contains the Apache License 2.0 text governing
  ASPIRE material owned by NVIDIA or contributed under the project license.
- [`../NOTICE`](../NOTICE) contains the project notice and directs recipients
  to the retained third-party terms.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) contains the contribution
  procedure and Developer Certificate of Origin requirements.

The project-level Apache-2.0 license does not replace component-specific terms.

## Third-party materials

- [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) is the canonical
  component inventory, including incorporated or modified material, pinned Git
  submodules, and external runtime components.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) contains the associated
  notices and attribution.
- The `.txt` files in this directory retain the license texts distributed with
  incorporated or modified third-party material.

## Compliance evidence

- [`evidence/OSRB_COMPLIANCE_REPORT.md`](evidence/OSRB_COMPLIANCE_REPORT.md)
  summarizes the completed repository checks and remaining approval gates.
- [`evidence/SOURCE_PROVENANCE.md`](evidence/SOURCE_PROVENANCE.md) and
  [`evidence/SOURCE_PROVENANCE.tsv`](evidence/SOURCE_PROVENANCE.tsv) record the
  source-ownership and header classification.
- [`evidence/SOURCE_MODIFICATION_EVIDENCE.tsv`](evidence/SOURCE_MODIFICATION_EVIDENCE.tsv)
  records per-path Git history and blob evidence for inherited source.
- [`evidence/SUBMODULE_AUDIT.md`](evidence/SUBMODULE_AUDIT.md) and
  [`evidence/SUBMODULE_AUDIT.tsv`](evidence/SUBMODULE_AUDIT.tsv) record the
  recursive submodule license and asset audit.

## Distribution boundary

The parent ASPIRE source artifact contains parent-tree files, `.gitmodules`,
and Git submodule links only. It does not contain populated submodule
repositories. A recursive checkout fetches those repositories separately from
their configured remotes, and their own source, model, dataset, media, and
asset terms apply. Review the third-party inventory and submodule audit before
using or redistributing the full stack.
