# __NVIDIA_OSS__ Standard Repo Template

This README file is from the NVIDIA_OSS standard repo template of [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file). It provides a list of files in the PLC-OSS-Template and guidelines on how to use (clone and customize) them.

**Upon completing the customization for the project repo, the repo admin should replace this README template with the project specific README file.**

- Files (org-wide templates in the NVIDIA .github org repo; per-repo overrides allowed) in [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file)

   - Root 
     - README.md skeleton (CTA + Quickstart + Support/Security/Governance links) 
     - LICENSE (Apache 2.0 by default)
        - For other licenses, see the [Confluence page](https://confluence.nvidia.com/pages/viewpage.action?pageId=788418816) for other licenses
        - CLA.md file (delete if not using MIT or BSD licenses)
     - CODE_OF_CONDUCT.md 
     - SECURITY.md (vuln reporting path) 
     - CONTRIBUTING.md (base; repo can add specifics)
     - SUPPORT.md (Support levels/channels)
     - GOVERNANCE.md (baseline; repo may extend)
     - CITATION.md (for projects that need citation)

   - .github/ 
     - ISSUE_TEMPLATE/ (<https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository>)
       - bug.yml, feature.yml, task.yml, config.yml 
     - PULL_REQUEST_TEMPLATE.md (<https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository>)
     - workflows/
     - Note: workflow-templates/ for starter workflows should live in the org-level .github repo, not per-repo

   - Repo-specific (not org-template, maintained by the team)
     - CODEOWNERS (place at .github/CODEOWNERS or repo root)
     - CHANGELOG.md (or RELEASE.md) 
     - ROADMAP.md 
     - MAINTAINERS.md 
     - NOTICE or THIRD_PARTY_NOTICES / THIRD_PARTY_LICENSES (dependency specific)
     - Build/package files (CMake, pyproject, Dockerfile, etc.)

   - Recommended structure and hygiene
     - docs/
     - examples/
     - tests/
     - scripts/
     - Container/dev env: Dockerfile, docker/, .devcontainer/ (optional)
     - Build/package (language-specific):
       - Python: pyproject.toml, setup.cfg/setup.py, requirements.txt, environment.yml
       - C++: CMakeLists.txt, cmake/, vcpkg.json
     - Repo hygiene: .gitignore, .gitattributes, .editorconfig, .pre-commit-config.yaml, .clang-format


## Usage of [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file) for NEW NVIDIA OSS repos

1. Clone the [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file)
2. Find/replace all in the clone of `___PROJECT___` and `__PROJECT_NAME__` with the name of the specific project.
3. Inspect all files to make sure all replacements work and update text as needed


**What you can reuse immediately**
- CODE_OF_CONDUCT.md
- SECURITY.md
- CONTRIBUTING.md (base)
- .github/ISSUE_TEMPLATE/.yml (bug/feature/task + config.yml)
- .github/PULL_REQUEST_TEMPLATE.md
- Reusable workflows 

**What you must customize per repo**
- README.md: copy the skeleton and fill in product-specific details (Quickstart, Requirements, Usage, Support level, links)
- LICENSE: check file is correct, update year, consult Confluence for alternatives https://confluence.nvidia.com/pages/viewpage.action?pageId=788418816, add CLA.md only if your license/process requires it
- CODEOWNERS: replace <TEAM> with your GitHub team handle(s). Place at .github/CODEOWNERS (or repo root)
- MAINTAINERS.md: list maintainers names/roles, escalation path
- CHANGELOG.md (or RELEASE.md): track releases/changes
- SUPPORT.md: Update for your project
- ROADMAP.md (optional): upcoming milestones
- NOTICE / THIRD_PARTY_NOTICES (if you ship third‑party content)
- Build/package files (CMake/pyproject/Dockerfile/etc.), tests/, docs/, examples/, scripts/ as appropriate
- Workflows: Edit if you need custom behavior 


4. Change git origin to point to new repo and push
5. Remove the line break below and everything above it

## Usage for existing NVIDIA OSS repos

1. Follow the steps above, but add the files to your existing repo and merge

<!-- REMOVE THE LINE BELOW AND EVERYTHING ABOVE -->
-----------------------------------------
# [Project Title]
One-sentence value proposition for users. Who is it for, and why it matters. 

# Overview
What the project does? Why the project is useful?
Provide a brief overview, highlighting key features or problem-solving capabilities.

# Getting Started
Guide users on how they can get started with the project. This should include basic installation step, quick-start examples 
```bash
# Option A: Package manager (pip/conda/npm/etc.)
<copy-paste install>

# Option B: Container
docker run <image> <args>

# Verify (hello world)
<one-liner or ~10-line example>
```
# Requirements
Include a list of pre-requisites. 
- OS/Arch: <summary or link to full matrix>
- Runtime/Compiler: <versions>
- GPU/Drivers (if applicable): CUDA <ver>, driver <ver>, etc.

# Usage
```bash
# Minimal runnable snippet (≤20 lines)
<code>
```
- More examples/tutorials: <link>
- API reference: <link>

# Performance (Optional)
Summary of benchmarks; link to detailed results and hardware used.

## Releases & Roadmap 
- Releases/Changelog: <link>
- (Optional) Next milestones or link to `ROADMAP.md`.
  
# Contribution Guidelines
- Start here: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Development quickstart (build/test):
```bash
<clone> && <deps> && <build/test>
```
## Governance & Maintainers
- Governance: `GOVERNANCE.md`
- Maintainers: <team/handles>
- Labeling/triage policy: <link>

## Security
- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.

## Support
- Level: <Experimental | Maintained | Stable>
- How to get help: Issues/Discussions/<channel link>
- Response expectations (if any).

# Community
Provide the channel for community communications.

# References
Provide a list of related references

# License
This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
- License: <link>
