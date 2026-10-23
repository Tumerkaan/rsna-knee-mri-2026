# Manual publication steps — NOT EXECUTED

Prepared on 23 October 2026. Run only after a separate explicit author approval.
No remote repository, push, Pages site or Kaggle post was created by this task.
The protected entry page already contains the final result table; it was not
edited here. Review the retained historical snapshots and naming exceptions in
RELEASE_CHECKS.md before approving Pages. Do not silently edit the entry page.

## 1. Use a standalone copy and verify it

Copy this package into a new standalone directory, outside the private research
checkout. Open PowerShell in that copy. Do not copy the research repository's
`.git`, data, outputs, credentials or environment. Python with the package's
dependencies, Git, GitHub CLI and RTK must already be installed.

```powershell
rtk proxy python -m pytest
rtk proxy python scripts/audit_release.py
rtk proxy git init --initial-branch main
rtk proxy git rev-parse --show-toplevel
```

Stop if a check fails or the Git root is not the standalone package. If Windows
denies the default pytest temp directory, use a fresh writable `--basetemp`
outside the release directory, never an existing directory with valuable files.
The release manifest inventories the local package, including retained historical
HTML; the staging list below intentionally excludes those reference snapshots.

## 2. Authenticate to your personal account

```powershell
rtk proxy gh auth login --hostname github.com --git-protocol https --web
rtk proxy gh auth status
rtk proxy gh auth setup-git
$releaseOwner = rtk proxy gh api user --jq .login
```

Confirm the returned owner is your personal account. The recommended browser
login requires no pasted personal access token. If your environment instead
requires a classic PAT, enter it only through `gh auth login --with-token`
standard input in your local terminal, never in this document, chat, command-line
arguments or repository files. GitHub CLI documents `repo`, `read:org` and `gist`
as the minimum classic-PAT scopes for that login method. Prefer credential-store
storage; do not use `--insecure-storage`.

## 3. Stage and inspect the intended public files

```powershell
rtk proxy git add -- .gitignore README.md REPRODUCE.md ATTRIBUTION.md LICENSE NOTICE RELEASE_POLICY.md PUBLISH_STEPS.md RELEASE_CHECKS.md pyproject.toml requirements-tested.txt SOURCE_MANIFEST.json EVIDENCE_MANIFEST.json NARRATIVE_EVIDENCE.json RELEASE_MANIFEST.json MEDIA_MANIFEST.json configs src scripts tests results
rtk proxy git add -- docs/WRITEUP.md docs/KAGGLE_DISCUSSION.md docs/EVIDENCE.md docs/index.html docs/assets
rtk proxy git diff --cached --stat
rtk proxy git diff --cached --name-only
rtk proxy git diff --cached --check
```

Review the staged list and diff. No raw data, weights, private manifests, OOF
rows, identifiers, credentials, local caches, HTML history or build templates
belong in the commit. Preserve the external image attribution and licenses.
The unused assets are retained as requested; this checklist does not delete them.

## 4. Create and push the repository (publishing action)

Only after approval and the preceding review:

```powershell
rtk proxy git commit -m "Prepare RSNA knee MRI research release"
rtk proxy gh repo create "$releaseOwner/rsna-knee-mri-2026" --public --source . --remote origin
rtk proxy git push -u origin main
```

Stop if the repository already exists; do not force-push or overwrite it. The
explicit owner avoids accidentally creating an organization repository.

## 5. Enable GitHub Pages from main:/docs (publishing action)

```powershell
rtk proxy gh api --method POST "repos/$releaseOwner/rsna-knee-mri-2026/pages" -f build_type=legacy -f 'source[branch]=main' -f 'source[path]=/docs'
rtk proxy gh api "repos/$releaseOwner/rsna-knee-mri-2026/pages" --jq '{status: .status, url: .html_url, source: .source}'
```

If Pages is already configured, inspect its settings before changing anything.
UI alternative: repository **Settings → Pages → Deploy from a branch → main →
/docs → Save**. Check the build result and live page after deployment; an API
acceptance response alone is not a successful rendered-site check.

Command references checked on 23 October 2026:
[GitHub CLI repository creation](https://cli.github.com/manual/gh_repo_create),
[authentication](https://cli.github.com/manual/gh_auth_login), and
[Pages REST API](https://docs.github.com/en/rest/pages/pages#create-a-github-pages-site).
