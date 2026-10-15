# Reference material

Updated 2026-10-15. This directory contains upstream code, papers and local reading material. Active project code
is at the repository root. Reference model rankings do not override the [current experimental findings](../README.md).

| Path | Purpose |
|---|---|
| `DSE_clustering-main/` | Original DSE implementation snapshot; retain its README and license |
| `REFERENCE_DSE_original_merged.py` | Flattened original implementation for inspection |
| `2504.09970v2.pdf`, `sun24g.pdf` | Local DSE/LSEnet papers |
| `urbanity/` | Separate upstream checkout recorded as a Git link; has its own environment and local changes |

The root graph environment does not install or synchronize `urbanity/`. See [UV_ENVIRONMENTS.txt](../UV_ENVIRONMENTS.txt)
for its separate workflow. This repository currently records that checkout without a `.gitmodules` mapping, so a
fresh clone cannot initialize it with `git submodule update`; obtain it separately if doing Urbanity-specific work.

Additional locally collected reference folders may be untracked. They are not runtime inputs for the main clustering
code and are not part of the current documentation/code submission. Preserve upstream README files and licenses
as source material. The [reading notes](../reading/README.md) explain the original DSE dataflow.
