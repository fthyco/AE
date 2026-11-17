"""Conservatively prune GitHub branches already merged into main.

Only exact merged PR heads are eligible. Published release build branches require
a verified GitHub Release, all three distribution assets, and an ancestor tag.
Dry-run unless --apply is specified; never deletes a tag or GitHub Release.
"""
from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

MAIN = "main"
MERGED_PREFIXES = ("feat/", "docs/", "test/", "chore/")
RELEASE_BRANCHES = {
    "release/v026-verified-github": "v0.26.0",
    "release/v027-verified-github": "v0.27.0",
}
ASSETS = (".whl", ".tar.gz", "SHA256SUMS.txt")


def request(path: str, *, method: str = "GET"):
    repo = os.environ["GITHUB_REPOSITORY"]
    url = "https://api.github.com/repos/" + repo + path
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    req = Request(url, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except HTTPError as exc:
        raise RuntimeError(f"GitHub {method} {path}: HTTP {exc.code}") from exc


def pages(path: str):
    results = []
    for page in range(1, 51):
        part = request(f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
        if not isinstance(part, list):
            raise TypeError(f"Expected list at {path}")
        results.extend(part)
        if len(part) < 100:
            return results
    raise RuntimeError("Page limit exceeded")


def is_main_descendant(sha: str) -> bool:
    # A merged PR's squash commit or release target must be reachable from main.
    match = request(f"/compare/{quote(sha, safe='')}...{MAIN}")
    return match["status"] in ("identical", "ahead")


def verify_release(branch: str) -> bool:
    tag = RELEASE_BRANCHES[branch]
    release = request("/releases/tags/" + quote(tag, safe=""))
    if release.get("draft") or release.get("prerelease") or release.get("tag_name") != tag:
        return False
    names = {a["name"] for a in release.get("assets", [])}
    if "SHA256SUMS.txt" not in names or not any(x.endswith(".whl") for x in names) or not any(x.endswith(".tar.gz") for x in names):
        return False
    target = release.get("target_commitish", "")
    return len(target) == 40 and is_main_descendant(target)


def candidates(branches, pull_requests):
    heads = {}
    for pr in pull_requests:
        if pr.get("merged_at") and pr.get("base", {}).get("ref") == MAIN:
            heads.setdefault(pr.get("head", {}).get("ref"), []).append(pr)
    result = []
    for branch in branches:
        name, sha = branch["name"], branch["commit"]["sha"]
        if name == MAIN or branch.get("protected"):
            continue
        if name in RELEASE_BRANCHES:
            if verify_release(name):
                result.append((name, sha, "published-release-verified"))
            continue
        if not name.startswith(MERGED_PREFIXES):
            continue
        for pr in heads.get(name, []):
            if pr["head"]["sha"] == sha and is_main_descendant(pr["merge_commit_sha"]):
                result.append((name, sha, f"merged-pr-{pr['number']}"))
                break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Delete eligible branch refs")
    args = parser.parse_args()
    branches = pages("/branches")
    prs = pages("/pulls?state=closed")
    planned = candidates(branches, prs)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run",
                      "current_branches": len(branches),
                      "eligible": [{"name": n, "sha": sha, "reason": why}
                                   for n, sha, why in planned]}, indent=2), flush=True)
    if not args.apply:
        return
    for name, sha, _ in planned:
        # Re-read immediately before deletion to avoid deleting a moving ref.
        latest = request("/branches/" + quote(name, safe="/"))
        if latest.get("protected") or latest["commit"]["sha"] != sha:
            raise RuntimeError(f"Branch moved/protected, refusing deletion: {name}")
        request("/git/refs/heads/" + quote(name, safe="/"), method="DELETE")
        print(f"Deleted merged branch: {name}", flush=True)
    print(f"Deleted {len(planned)} eligible stale branches", flush=True)


if __name__ == "__main__":
    main()
