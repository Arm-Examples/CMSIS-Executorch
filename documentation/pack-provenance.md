# Where the ExecuTorch CMSIS pack comes from

The `PyTorch::ExecuTorch` pack used by this branch is **vendored** into
`packs/PyTorch.ExecuTorch.1.4.0-rc2/` rather than fetched. This page explains
why, how to rebuild it, and how to move to a newer ExecuTorch version.

## Why it is vendored

The pack's `.pdsc` declares a download location that does not exist:

```xml
<url>https://github.com/pytorch/executorch/releases/download/1.4.0-rc2/</url>
```

There is no `1.4.0-rc2` GitHub release to download from — at the time of
writing ExecuTorch has published no `v1.4*` tag at all. So `cpackget add`,
`cbuild --packs` and the CMSIS Solution extension cannot acquire this pack the
normal way. Committing the pack directory and pointing at it with a local
`path:` is the only arrangement that gives a working clone:

```yaml
packs:
  - pack: PyTorch::ExecuTorch
    path: ./packs/PyTorch.ExecuTorch.1.4.0-rc2
```

That `path:` appears twice — in `cmsis-executorch-simple.csolution.yml` and in
`cmsis-executorch-simple.cproject.yml`. Both must agree.

The cost is ~8 MB and ~940 files in the repository. That is the price of a
clone that builds without a manual pack-building step first, and it goes away
once ExecuTorch publishes the pack as a release artifact.

## What is in the pack

| Path | Contents |
|---|---|
| `PyTorch.ExecuTorch.pdsc` | Component declarations — one component per operator, plus runtime and backend |
| `src/` | ExecuTorch runtime, kernel and Ethos-U backend sources |
| `include/` | Public headers, **including the bundled `include/flatbuffers/`** |
| `armclang_shims/` | Small compatibility shims for Arm Compiler 6 |
| `Documentation/` | Pack README |
| `LICENSE` | Upstream BSD-3-Clause (the example code around it is Apache-2.0) |

It is a **source** pack: nothing is prebuilt. Every operator is a selectable
component, which is what lets `scripts/gen_components.py` narrow the link to
exactly the kernels a given `.pte` needs.

## Rebuilding the pack

The generator lives in the ExecuTorch tree at
`backends/arm/cmsis_pack/scripts/build_pack.sh`.

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch
git checkout release/1.4          # or the tag matching your target version
git submodule update --init --recursive
```

> **Run the CMake cross-compile before `build_pack.sh`.**
>
> This is the one trap worth knowing about. If `build_pack.sh` runs without the
> prior cross-compile, it completes successfully but **silently omits**:
>
> - the bundled `include/flatbuffers/` headers, and
> - the flatc-generated `program_generated.h` and `scalar_type_generated.h`.
>
> The resulting pack looks complete and fails at compile time with missing-header
> errors that point nowhere useful. Those two headers are generated with:
>
> ```bash
> flatc --cpp --cpp-std c++11 --gen-mutable --scoped-enums <schema>.fbs
> ```
>
> `flatc` ships inside the executorch wheel — after `./setup_venv.sh` it is at
> `.venv/bin/flatc`.

Then run the generator; the pack lands in `pack-output/`:

```bash
backends/arm/cmsis_pack/scripts/build_pack.sh
```

### Verifying a rebuilt pack

Before trusting a freshly built pack, check the parts that go missing quietly:

```bash
PACK=pack-output/PyTorch.ExecuTorch.<version>

# 1. The bundled flatbuffers headers must be present.
test -d "$PACK/include/flatbuffers" || echo "MISSING: include/flatbuffers"

# 2. The flatc-generated headers must be present.
for h in program_generated.h scalar_type_generated.h; do
    find "$PACK/include" -name "$h" | grep -q . || echo "MISSING: $h"
done

# 3. The component list should still cover the operators the model uses.
grep -c '<component' "$PACK/PyTorch.ExecuTorch.pdsc"
```

A pack that passes all three is safe to drop in.

## Moving to a new ExecuTorch version

The pack's C++ runtime and the Python exporter must come from the *same*
ExecuTorch version — a `.pte` produced by a different version than the runtime
that loads it will fail at load time, or worse, at inference time. Four steps,
in order:

1. **Rebuild the pack** at the new ref (above) and replace
   `packs/PyTorch.ExecuTorch.<old>/` with the new directory.
2. **Update both `path:` entries** — csolution and cproject.
3. **Update the Python pin** in `requirements-executorch.txt` to the matching
   `executorch` version, and check the new release's `install_requirements.py`
   for the `torch` and `torchao` versions it expects. Update
   `requirements.txt` (torch) accordingly. See the README's
   [Version pinning](../README.md#version-pinning) table for the current set.
4. **Rebuild the venv and the project:**

   ```bash
   ./setup_venv.sh --recreate
   ./build.sh                     # may abort once; see below
   ```

If the new version changes the operator set, the first build stops with the
"operator set changed" notice and rewrites `ai_layer/ai_layer.clayer.yml`.
That is expected — run `./build.sh` again. See
[mlops-flow.md](mlops-flow.md#why-the-clayer-cannot-update-in-place) for why.

Finally, update the pack directory name wherever it is referenced in prose:
`README.md` (Layout table, license note, the manual `gen_components.py`
invocation) and this page.
