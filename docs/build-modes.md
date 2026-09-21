# Build modes and threading

Most users never need this page: `pip install pylmcf` picks the right wheel and
works. It matters if you are building from the sdist, running on a free-threaded
interpreter, working on musl, or debugging a mysterious `TypeError` in a stack
that includes `wnet` or `wnetalign`.

---

## Split vs linked nanobind

Since nanobind 3 an extension is built in one of two modes:

- **split** — carries no nanobind library code, and resolves the shared
  `nanobind_backend` module at import. The extension is a stable-ABI build, so
  one wheel per platform covers every Python from 3.10 up.
- **linked** — embeds its own copy of libnanobind. Version-specific.

`CMakeLists.txt` picks per platform. Linked is used when split mode is
unreachable:

| Condition | Mode | Why |
|---|---|---|
| CPython on a platform the backend publishes wheels for | **split** | the normal case |
| free-threaded CPython 3.15+ | **split** (`abi3t`) | with nanobind's `FREE_THREADED` option |
| PyPy | linked | no backend wheel |
| musl libc | linked | `nanobind-backend` ships no musllinux wheel — *and no sdist* |
| 32-bit Windows | linked | no win32 backend wheel |
| free-threaded CPython below 3.15 | linked | predates the backend's `abi3t` wheels |
| `-DWNET_NB_LINKED=ON` | linked | forced; sanitizer builds need it |

You can ask an installed pylmcf which it got:

```python
>>> import pylmcf
>>> pylmcf.is_nanobind_split()
True
```

### Why a mixed stack breaks

Extensions only see each other's `nb::class_` registrations when they share one
set of nanobind internals. Two split-mode extensions share the backend's; two
linked extensions of the same ABI share the process-global map. **A split
extension and a linked one share nothing.**

`wnetalign` casts a class registered inside `wnet_cpp`
(`nb::cast<Spectrum<DIM>*>`), so a mixed stack does not merely lose a feature —
it fails, and it fails late, as a `TypeError` or a `std::bad_cast` from somewhere
far from the real cause.

`pylmcf.nanobind_mode.check_consistent()` exists to catch that at import time
instead, and every package downstream of pylmcf calls it:

```python
from pylmcf.nanobind_mode import check_consistent

check_consistent([
    ("pylmcf", pylmcf.pylmcf_cpp),
    ("wnet",   wnet.wnet_cpp),
])
```

It raises `ImportError` naming the odd module out. A mixed stack is not something
wheels normally produce — the mode follows the platform, so all four packages
agree by construction. It shows up when the packages come from different places:
one built from an sdist and the rest from wheels, a stale editable install left
over from an experiment, a `WNET_NB_LINKED=ON` sanitizer build sitting in a venv
whose other packages came from wheels.

### The stable-ABI tag caveat on linked builds

Every linked path drops the `wheel.py-api` request, because nanobind refuses a
linked build targeting the classic 3.10/3.11 stable ABI. musl is where this
actually bites — it is the only linked path on which scikit-build-core honours
`py-api` at all.

**scikit-build-core still *tags* such a wheel `cp310-abi3` while the extension
inside is version-specific.** A musl-built wheel must therefore be installed and
discarded — never republished, never reused across Python versions. CMake emits a
warning when it drops the request.

---

## Free-threaded Python

On CPython 3.15t the extension is built in split mode with nanobind's
`FREE_THREADED` option, which declares `Py_MOD_GIL_NOT_USED`. There is a separate
`cp315-abi3t` wheel for it.

The declaration is honest. The extension has:

- **no mutable globals whatsoever** — every `.data`/`.bss` symbol is a vtable,
  typeinfo or libstdc++ fixture. Worth re-checking with
  `nm -C --defined-only` if file-scope state is ever added.
- the optional `PYLMCF_PIVOT_STATS` counters declared `thread_local`.
- all solver state per-`Graph`.

`tests/test_free_threading.py` guards this: it runs 8 threads × 25 concurrent
solves against a serial oracle. The module self-skips unless the GIL is genuinely
off *after* importing the extension, which is why it stays quiet on 3.14t where
the linked fallback is expected to turn the GIL back on.

### What is NOT promised

**A single `CGraph` may not be driven from two threads at once.** `solve()`
mutates the instance, takes no lock, and no binding releases the GIL. Independent
`Graph` objects in different threads are fine; a *shared* object is the caller's
problem.

`Py_MOD_GIL_NOT_USED` says the module does not rely on the GIL for its own
internal state. It says nothing about your objects.

Free-threaded 3.14 and below get no wheel and build from the sdist in linked
mode.

---

## Wheels

Built by `cibuildwheel` in `.github/workflows/build_wheels.yml`, on the
`ci_wheels` branch.

- `CIBW_BUILD` stays at `cp310-*`. The single `cp310-abi3` wheel already covers
  everything up to and including 3.15 — verified, not assumed: a wheel built on
  3.14 passes the full suite on 3.15, which `test_wheel_newest` asserts by
  installing the artifact with `--only-binary`. A `cp315` build would only
  re-emit the same tag.
- `cp315t-*` is also built, giving a second, separate `cp315-abi3t` wheel.
  `test_wheel_freethreaded` fails if importing pylmcf re-enables the GIL, or if
  `tests/test_free_threading.py` self-skips instead of running.
- musl is skipped (`CIBW_SKIP: *-musllinux_*`) and covered by `test_sdist_musl`,
  which runs in an `alpine:3.21` container and asserts that `SOABI` really says
  musl and that the backend is neither declared nor installed.
- `test_sdist_freethreaded` and `test_sdist_pypy` cover the other linked paths.

### Dynamic dependencies

`dependencies` is dynamic, resolved per build by `_pylmcf_metadata.py`:
`nanobind-backend` is required only by a split-mode build, and no PEP 508 marker
can express "not a free-threaded interpreter". A static requirement made
resolution fail on free-threaded CPython 3.14 *before a compiler was reached* —
and building from the sdist is the only install path those interpreters have.

`_pylmcf_metadata._split_mode()` mirrors the `NB_MODE` selection in
`CMakeLists.txt`. **The two must be kept in step.** The libc test lives in one
place only — `_pylmcf_metadata.is_musl()`, which CMake calls through
`Python_EXECUTABLE`, treating a failed probe as a hard error rather than guessing.

---

## Building from source

```bash
./reinstall.sh
```

installs in editable/development mode. It uses
`SKBUILD_BUILD_DIR=_skbuild_<host>_<venv>` so the persistent CMake directory is
keyed on both hostname and active venv — the repo is often shared across machines
over NFS, and each venv has its own Python ABI and nanobind. It falls back to an
isolated build if `scikit_build_core` or `nanobind` are missing from the venv.

Requirements for a source build: **Python 3.10+ and a C++20 compiler.**

---

## See also

- [Testing and diagnostics](testing.md)
- [The C++ header tree](cpp-headers.md)
