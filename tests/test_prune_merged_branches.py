"""Safety checks for the merged-branch pruning selection."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "repo_maintenance", Path(__file__).resolve().parents[1] /
    "scripts/maintenance/prune_merged_branches.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_exact_merged_head_and_ancestry_can_be_deleted():
    branches = [
        {"name": "main", "commit": {"sha": "m"}, "protected": False},
        {"name": "feat/merged", "commit": {"sha": "a"}, "protected": False},
        {"name": "feat/moved", "commit": {"sha": "new"}, "protected": False},
        {"name": "feat/unmerged", "commit": {"sha": "b"}, "protected": False},
        {"name": "feat/protected", "commit": {"sha": "c"}, "protected": True},
        {"name": "unknown/reference", "commit": {"sha": "d"}, "protected": False},
    ]
    def pr(n, branch, sha, merged=True):
        return {"number": n, "merged_at": "2026-11-17" if merged else None,
                "base": {"ref": "main"}, "head": {"ref": branch, "sha": sha},
                "merge_commit_sha": f"merge-{n}"}
    prs = [pr(1, "feat/merged", "a"), pr(2, "feat/moved", "old"),
           pr(3, "feat/unmerged", "b", False), pr(4, "feat/protected", "c")]
    with patch.object(module, "is_main_descendant", return_value=True):
        assert module.candidates(branches, prs) == [
            ("feat/merged", "a", "merged-pr-1")
        ]


def test_release_branches_require_verified_published_artifacts():
    branches = [
        {"name": "release/v027-verified-github", "commit": {"sha": "x"},
         "protected": False}
    ]
    with patch.object(module, "verify_release", return_value=False):
        assert module.candidates(branches, []) == []
    with patch.object(module, "verify_release", return_value=True):
        assert module.candidates(branches, []) == [
            ("release/v027-verified-github", "x", "published-release-verified")
        ]
