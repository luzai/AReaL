# Historical C1 runtime snapshot — DRAFT

This independent branch contains the selected **third-stage** AReaL runtime source for
reconstructing the historical C1 Iter25/global-step-24 lineage. It has no earlier Git
history. It is **not** a completed three-stage reproduction runtime, a bug-fixed
replacement, or a validated training release.

Recipe companion:
[areal-pacman repro/c1-iter25-lineage](https://github.com/luzai/areal-pacman/tree/repro/c1-iter25-lineage).
Game companion:
[pacman-python repro/c1-iter25-game](https://github.com/luzai/pacman-python/tree/repro/c1-iter25-game).

## Source scope

The source archive SHA256 is
`f0fdf3961fd2bab89ef5f69d9e9cf65f95095d1cacd83d4842ef4e73aa0e508f`. Its included
historical files were verified against the original source manifest. This publication
selects runtime, tests, packaging/lock metadata and supporting tools. Backup files,
private logs, checkpoints, environment directories, original Git history and
investigation records are excluded. Existing Apache-2.0 notices and LICENSE are
retained. AReaL was developed by the AReaL team and contributors.

The first and second stages used different source snapshots. Their selection and the
three-stage launcher/checkpoint handoff remain unfinished. **Do not run all three recipe
YAMLs against this third-stage runtime as if they were equivalent.** Historical
dependency metadata is retained, not certified against the current server environment.
No install or training command is endorsed by this draft.

## Verification boundary

- Synthetic four-process CPU Gloo dispatch and two-level packing passed three cases for
  this snapshot. This was not a real-model GPU forward/backward test.
- Nonfinite JSON codec loss was reproduced: inf/-inf/NaN become null/None.
- Qwen3.5 GDN row isolation (1b31d73d) and multimodal position IDs (98028b2b) are not
  backported here. Real-model verification is still required.
- Validation remains OFF in the companion recipe. No new GPU training is started by
  publishing this branch.

Keep this historical reference distinct from any later correctness-fixed variant.

## Publication-only maintenance

The publication applies formatting and two import-order fixes. AST comparison of 719
Python files found no differences outside top-level imports; the two import changes were
separately reviewed. This is not a byte-identical archive.

Historical dependency lockfiles are retained unchanged. The lockfile hook checks them
offline with `--locked`, rather than regenerating/upgrading dependencies. The large-file
check remains enabled at 1100 KiB to admit the 1037 KiB historical vLLM lockfile. Shell
scripts use LF line endings. None of these adjustments is a training algorithm, reward,
game rule, or checkpoint-lineage change.

The portable lock checker works on temporary copies and asserts unchanged lock digests.
Local CLI documentation generation uses the existing lightweight dataclass-inspection
harness (no model imports), with the referenced constants checked against this snapshot.
This is a documentation check, not runtime evidence.
