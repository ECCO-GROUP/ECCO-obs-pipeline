# Release Process — ECCO-obs-pipeline

## 1. Decide the version number

Use [Semantic Versioning](https://semver.org/):

| Change type | Bump | Example |
|---|---|---|
| Breaking config, schema, or API changes | `MAJOR` | v1.0.0 → v2.0.0 |
| New features, backwards-compatible | `MINOR` | v2.0.0 → v2.1.0 |
| Bug fixes, small improvements | `PATCH` | v2.0.0 → v2.0.1 |

---

## 2. Pick a marine animal name

Each release gets a marine animal name. Pick one that hasn't been used:

| Version | Name | Animal |
|---|---|---|
| v1.0.0 | *Aurelia aurita* | Moon jellyfish |
| v2.0.0 | _TBD_ | _TBD_ |

Some ideas for future releases:
- *Stenella longirostris* — spinner dolphin
- *Architeuthis dux* — giant squid
- *Mobula birostris* — giant oceanic manta ray
- *Bathysaurus ferox* — deep-sea lizardfish
- *Vampyroteuthis infernalis* — vampire squid

---

## 3. Review commits since the last release

```bash
git log <last-tag>..HEAD --oneline
# e.g.
git log v1.0.0..HEAD --oneline
```

Categorize them into: Breaking Changes, New Features, Dataset Updates, Improvements, Bug Fixes.

---

## 4. Update CHANGELOG.md

1. Fill in the `[Unreleased]` section at the top with:
   - The version number and today's date
   - The marine animal name
   - All categorized changes from step 3
2. Add a new blank `[Unreleased]` section above it for the next release.
3. Update the comparison links at the bottom of the file.

---

## 5. Commit, tag, and push

```bash
# Make sure you're on master and it's clean
git checkout master && git pull

# Commit the changelog
git add CHANGELOG.md
git commit -m "chore: update CHANGELOG for v2.0.0"

# Create an annotated tag
git tag -a v2.0.0 -m "v2.0.0 — [Marine Animal Name]"

# Push branch and tag
git push
git push origin v2.0.0
```

---

## 6. Create the GitHub release

1. Go to [Releases → Draft a new release](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/releases/new).
2. Select the tag you just pushed (`v2.0.0`).
3. Set the **title** to the marine animal name (e.g. `Aurelia aurita`).
4. Paste the relevant `CHANGELOG.md` section as the release description.
5. Click **Publish release**.

GitHub automatically attaches `.zip` and `.tar.gz` source archives.

---

## Quick checklist

```
[ ] git log <last-tag>..HEAD --oneline  — review what's changed
[ ] Decide version bump (major / minor / patch)
[ ] Pick marine animal name
[ ] Update CHANGELOG.md (fill in version, date, name, categorized changes)
[ ] Add new [Unreleased] block at top of CHANGELOG.md
[ ] git add CHANGELOG.md && git commit -m "chore: update CHANGELOG for vX.Y.Z"
[ ] git tag -a vX.Y.Z -m "vX.Y.Z — [Name]"
[ ] git push && git push origin vX.Y.Z
[ ] GitHub → Releases → Draft → select tag → paste changelog → Publish
```
