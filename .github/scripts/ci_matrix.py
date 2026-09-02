#!/usr/bin/env python3
"""Pick the CI test matrix for this run, and prove it still covers everything.

This file is the single source of truth for *which* combinations of platform,
Python and compiler get tested, and when. It replaced a `os x python x
compiler` cross-product plus ~24 `exclude:` rules in run_tests.yml -- a shape
in which the exclusions were longer than the inclusions, nobody could read the
actual coverage off the page, and the four sibling repositories had silently
drifted to different exclude lists.

The design is a covering array, not a cross-product. Every level of every
factor is exercised; pairs are only covered where the pair actually interacts:

  compiler x platform  interacts, but only on Linux -- gcc/clang is the one
                       place the compiler is a free variable. MSVC and
                       AppleClang are 1:1 with their OS.
  python x platform    interacts at the edges only: 3.10 is the abi3 floor and
                       3.15 is the prerelease/free-threading frontier. To
                       nanobind, 3.11/3.12/3.13 are the same C API, so those
                       rotate one platform each rather than running everywhere.
  python x compiler    does not interact. Codegen does not know which
                       interpreter will dlopen it.
  arch x anything      interacts, and is the axis that has actually found bugs.

The cost asymmetry that drove the numbers: `linux-arm64` is a self-hosted
runner on wloczykij, which is an Opteron 6380. There is no ARM in that machine
-- the lane is qemu, and it measures 65-85 min against 8-16 min for every other
platform. Twelve arm64 legs were 900 of the 1166 job-minutes in a full run:
77% of CI spent asking one emulated architecture the same question twelve
times. Tier B asks it twice.

Tiers, cheapest first, each a superset of the last (asserted below):

  A  every push to a work branch. Self-hosted linux-amd64 only, ~20 min wall.
     Fast enough that nobody starts writing "[skip ci]".
  B  main, the nightly run, and releases of the leaf packages. The covering
     matrix: all 6 platforms, all 6 Pythons, all 4 toolchains, both arches.
  C  v* tags on pylmcf and wnet. The wide matrix, still with arm64 halved --
     gcc-vs-clang at *every* Python on an emulated arch is the least
     informative axis in the design.

Output goes to $GITHUB_OUTPUT as `tier`, `tests` and `sanitize`; the last two
are `{"include": [...]}` objects consumed by
`strategy: matrix: ${{ fromJson(needs.select.outputs.tests) }}`.
"""

import json
import os
import sys

# ── per-repository knobs ─────────────────────────────────────────────────────
# These two lines are the ONLY intended difference between the four copies of
# this file. Everything below is meant to stay byte-identical across pylmcf,
# wnet, wnetalign and wnetdeconv; diff them when in doubt.
RELEASE_TIER = "C"  # pylmcf, wnet. Leaves (wnetalign, wnetdeconv) use "B":
#                     a leaf release rides on the fact that its dependencies
#                     already passed their own Tier C.
HAS_SANITIZE = False  # only wnet has a sanitizer job; elsewhere the `sanitize`
#                     output is emitted and ignored.

TIER_ORDER = {"A": 0, "B": 1, "C": 2}

# ── platforms ────────────────────────────────────────────────────────────────
# `container: ""` means "run directly on the runner" -- the hosted legs have
# always relied on that (they simply omitted the key), so the empty string is
# the same thing said out loud.
PLATFORMS = {
    "linux-amd64": dict(
        runner=["self-hosted", "wloczykij-amd64"],
        container="localhost:5000/ubuntu-ci:24.04",
        arch="x86-64",
        minutes=11,
    ),
    "linux-arm64": dict(
        runner=["self-hosted", "wloczykij-arm64"],
        container="localhost:5000/ubuntu-ci:24.04",
        arch="aarch64",
        minutes=75,  # qemu on an Opteron. This number is the whole design.
    ),
    "windows-latest": dict(runner="windows-latest", container="", arch="x86-64", minutes=8),
    "windows-11-arm": dict(runner="windows-11-arm", container="", arch="aarch64", minutes=10),
    "macos-latest": dict(runner="macos-latest", container="", arch="aarch64", minutes=8),
    "macos-15-intel": dict(runner="macos-15-intel", container="", arch="x86-64", minutes=10),
}

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15"]

OLDEST = "3.10"  # the abi3 floor: the wheel tag every published artifact carries
NEWEST = "3.15"  # prerelease, and the only free-threaded-capable interpreter


def toolchain(platform: str, compiler: str) -> str:
    """The compiler that actually ends up building the extension."""
    if platform.startswith("linux"):
        return "clang" if compiler == "clang" else "gcc"
    if platform.startswith("windows"):
        return "msvc"
    return "appleclang"


def lane(platform, python, compiler="default", tier="B", realdata=False):
    return dict(platform=platform, python=python, compiler=compiler, tier=tier, realdata=realdata)


# ── the test lanes ───────────────────────────────────────────────────────────
# `tier` is the *cheapest* tier at which the lane runs; it also runs in every
# tier above that. Read the table as the coverage argument -- there is nothing
# else to read, which is the point.
LANES = [
    # linux-amd64 is self-hosted, uncontended and ~11 min, so it carries the
    # Python breadth for the whole design.
    lane("linux-amd64", "3.10", "default", "A"),                    # abi3 floor
    lane("linux-amd64", "3.11", "default", "B"),
    lane("linux-amd64", "3.12", "default", "A", realdata=True),     # see below
    lane("linux-amd64", "3.13", "clang",   "B"),
    lane("linux-amd64", "3.14", "clang",   "A"),
    lane("linux-amd64", "3.15", "clang",   "A"),                    # prerelease frontier
    # linux-arm64: 12 lanes -> 2. Both ends of the supported range, one under
    # each compiler. Everything in between is covered on amd64, and the arch
    # difference does not care which minor version of CPython is driving it.
    lane("linux-arm64", "3.10", "default", "B"),
    lane("linux-arm64", "3.15", "clang",   "B"),
    # Windows x64: 6 -> 2, the two ends.
    lane("windows-latest", "3.10", "default", "B"),
    lane("windows-latest", "3.15", "default", "B"),
    # Windows on ARM is a platform, not a Python matrix. One seat. (3.10 has no
    # numpy wheel there and would build numpy from source, so never 3.10.)
    lane("windows-11-arm", "3.14", "default", "B"),
    # macOS.
    lane("macos-latest", "3.13", "default", "B"),
    lane("macos-latest", "3.15", "default", "B"),
    lane("macos-15-intel", "3.12", "default", "B"),  # GitHub is retiring this runner
    # ── Tier C only: the wide release matrix ────────────────────────────────
    lane("linux-amd64", "3.10", "clang",   "C"),
    lane("linux-amd64", "3.11", "clang",   "C"),
    lane("linux-amd64", "3.12", "clang",   "C"),
    lane("linux-amd64", "3.13", "default", "C"),
    lane("linux-amd64", "3.14", "default", "C"),
    lane("linux-amd64", "3.15", "default", "C"),
    # arm64 on a release: every Python once, alternating compiler. Six lanes,
    # not twelve -- see the module docstring.
    lane("linux-arm64", "3.11", "clang",   "C"),
    lane("linux-arm64", "3.12", "default", "C"),
    lane("linux-arm64", "3.13", "clang",   "C"),
    lane("linux-arm64", "3.14", "default", "C"),
    lane("windows-latest", "3.11", "default", "C"),
    lane("windows-latest", "3.12", "default", "C"),
    lane("windows-latest", "3.13", "default", "C"),
    lane("windows-latest", "3.14", "default", "C"),
    lane("windows-11-arm", "3.11", "default", "C"),
    lane("windows-11-arm", "3.12", "default", "C"),
    lane("windows-11-arm", "3.13", "default", "C"),
    lane("windows-11-arm", "3.15", "default", "C"),
    lane("macos-latest", "3.14", "default", "C"),
    lane("macos-15-intel", "3.14", "default", "C"),
    lane("macos-15-intel", "3.15", "default", "C"),
]

# ── the sanitizer lanes ──────────────────────────────────────────────────────
# The six are {gcc, clang+libstdc++, clang+libc++} x {asan+ubsan, hardened}.
# Pairwise on a 3x2 *is* the full six, so there is no honest pairwise cut here.
# Tier A instead takes 1-coverage: three lanes that between them exercise all
# three toolchains and both modes. Tiers B and C run all six.
#
# Every flag string below is carried over verbatim from run_tests.yml; the
# reasoning that produced them is preserved in the comments.
SANITIZE = [
    dict(
        label="gcc asan+ubsan", tier="A",
        cc="gcc", cxx="g++", apt_extra="", build_type="Debug",
        # Debug activates WNET_DO_ASSERTS + -Og -g in CMakeLists.
        # _GLIBCXX_SANITIZE_VECTOR poisons vector storage for ASan;
        # _GLIBCXX_DEBUG is intentionally omitted -- it conflicts with ASan
        # (different allocator hooks, false positives).
        cxx_flags="-fsanitize=address,undefined -fno-sanitize=vptr"
                  " -fno-sanitize-recover=all -fno-omit-frame-pointer"
                  " -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_SANITIZE_VECTOR",
        link_flags="",
        # GCC's driver does not add libasan to DT_NEEDED for `-shared
        # -fsanitize=address` on this container, so the extension imports with
        # unresolved __asan_* symbols. Preload libasan instead. Safe here
        # because the extension is built with libstdc++, so the libstdc++ EH
        # hooks libasan pulls in match the ABI.
        preload_asan=True, pythonmalloc="malloc",
        asan_opts="detect_leaks=0:abort_on_error=1:print_stacktrace=1"
                  ":log_path=/tmp/asan:verify_asan_link_order=0",
        ubsan_opts="print_stacktrace=1:halt_on_error=1:log_path=/tmp/ubsan",
        dump_logs=True,
    ),
    dict(
        label="gcc hardened", tier="B",
        cc="gcc", cxx="g++", apt_extra="",
        # RelWithDebInfo = -O2 -g -DNDEBUG; the internal macros are passed
        # explicitly since the CMakeLists Debug block does not fire.
        # _GLIBCXX_DEBUG: full iterator-validity checking (safe: the whole .so
        # is compiled consistently and no STL objects cross the Python C-API
        # boundary). _FORTIFY_SOURCE=3 requires -O2, unavailable at the -Og
        # that Debug uses.
        build_type="RelWithDebInfo",
        cxx_flags="-DWNET_DO_ASSERTS -DDEBUG_MODE -D_GLIBCXX_ASSERTIONS"
                  " -D_GLIBCXX_DEBUG -D_FORTIFY_SOURCE=3 -fstack-protector-strong"
                  " -fstack-clash-protection -fcf-protection=full",
        link_flags="", preload_asan=False, pythonmalloc="",
        asan_opts="", ubsan_opts="", dump_logs=False,
    ),
    dict(
        label="clang libstdc++ asan+ubsan", tier="B",
        cc="clang", cxx="clang++", apt_extra="clang", install_clang_rt=True,
        build_type="Debug",
        cxx_flags="-fsanitize=address,undefined -fno-sanitize=vptr"
                  " -fno-sanitize-recover=all -fno-omit-frame-pointer"
                  " -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_SANITIZE_VECTOR",
        link_flags="", preload_asan=True, pythonmalloc="malloc",
        asan_opts="detect_leaks=0:abort_on_error=1:print_stacktrace=1"
                  ":log_path=/tmp/asan:verify_asan_link_order=0",
        ubsan_opts="print_stacktrace=1:halt_on_error=1:log_path=/tmp/ubsan",
        dump_logs=True,
    ),
    dict(
        label="clang libstdc++ hardened", tier="A",
        cc="clang", cxx="clang++", apt_extra="clang", build_type="RelWithDebInfo",
        cxx_flags="-DWNET_DO_ASSERTS -DDEBUG_MODE -D_GLIBCXX_ASSERTIONS"
                  " -D_GLIBCXX_DEBUG -D_FORTIFY_SOURCE=3 -fstack-protector-strong"
                  " -fstack-clash-protection -fcf-protection=full",
        link_flags="", preload_asan=False, pythonmalloc="",
        asan_opts="", ubsan_opts="", dump_logs=False,
    ),
    dict(
        # libc++ 18+ supports _LIBCPP_HARDENING_MODE.
        # _LIBCPP_ENABLE_ASSERTIONS is intentionally absent: since LLVM 18 it is
        # deprecated and hard-errors when combined with _LIBCPP_HARDENING_MODE.
        label="clang libc++ asan+ubsan", tier="A",
        cc="clang-22", cxx="clang++-22",
        apt_extra="clang-22 libclang-rt-22-dev libc++-22-dev libc++abi-22-dev",
        # LLVM 22 from apt.llvm.org: Ubuntu 24.04 does not ship it, so there is
        # no epoch-version conflict.
        use_llvm_apt=True, llvm_apt_version="22", build_type="Debug",
        cxx_flags="-fsanitize=address,undefined -fno-sanitize=vptr"
                  " -fno-sanitize-recover=all -fno-omit-frame-pointer -stdlib=libc++"
                  " -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE",
        # Force dynamic ASan runtime linkage for Python extension modules;
        # without -shared-libasan the module may import with unresolved
        # __asan_* symbols when no preload is used.
        link_flags="-stdlib=libc++ -shared-libasan",
        # preload_asan MUST be false for libc++ builds. LD_PRELOAD injects ASan
        # before libc++abi.so.1 is loaded, so ASan's dlsym(RTLD_NEXT,
        # "__cxa_rethrow_primary_exception") returns NULL and the interceptor
        # CHECK-fails the first time std::rethrow_exception is called. With the
        # static ASan runtime the extension loads libc++abi.so.1 as a
        # dependency before ASan initializes, so the dlsym succeeds.
        preload_asan=False, pythonmalloc="malloc",
        asan_opts="detect_leaks=0:abort_on_error=1:print_stacktrace=1"
                  ":log_path=/tmp/asan:verify_asan_link_order=0",
        ubsan_opts="print_stacktrace=1:halt_on_error=1:log_path=/tmp/ubsan",
        dump_logs=True,
    ),
    dict(
        label="clang libc++ hardened", tier="B",
        cc="clang", cxx="clang++", apt_extra="clang libc++-dev libc++abi-18-dev",
        build_type="RelWithDebInfo",
        cxx_flags="-stdlib=libc++ -DWNET_DO_ASSERTS -DDEBUG_MODE"
                  " -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG"
                  " -D_FORTIFY_SOURCE=3 -fstack-protector-strong"
                  " -fstack-clash-protection -fcf-protection=full",
        link_flags="-stdlib=libc++", preload_asan=False, pythonmalloc="",
        asan_opts="", ubsan_opts="", dump_logs=False,
    ),
]

SANITIZE_DEFAULTS = dict(install_clang_rt=False, use_llvm_apt=False, llvm_apt_version="22")


def pick_tier() -> str:
    """Which tier this run gets. CI_TIER (workflow_dispatch) overrides."""
    forced = (os.environ.get("CI_TIER") or "").strip().upper()
    if forced in TIER_ORDER:
        return forced
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    ref = os.environ.get("GITHUB_REF", "")
    if event == "schedule":
        return "B"
    if ref.startswith("refs/tags/"):
        return RELEASE_TIER
    if ref == "refs/heads/main":
        return "B"
    return "A"


def select(tier: str):
    want = TIER_ORDER[tier]
    tests = []
    for ln in LANES:
        if TIER_ORDER[ln["tier"]] > want:
            continue
        plat = PLATFORMS[ln["platform"]]
        tool = toolchain(ln["platform"], ln["compiler"])
        prefix = "[clang] " if ln["compiler"] == "clang" else ""
        tests.append({
            "lane": f"{prefix}{ln['platform']} / py{ln['python']}",
            "os": ln["platform"],
            "python-version": ln["python"],
            "compiler": ln["compiler"],
            "toolchain": tool,
            "arch": plat["arch"],
            "runner": plat["runner"],
            "container": plat["container"],
            # The wnetalign real-data tests pin exact costs and an exact
            # consensus pairing against tests/baselines/. Those numbers come out
            # of floating-point arithmetic feeding an exact integer solver, so a
            # last-ulp difference between architectures, compilers or libm
            # versions can tip a tie between two equally-optimal matchings -- a
            # red square with nothing actually wrong behind it. Exactly one lane
            # is canonical; everywhere else only the portable properties run.
            "realdata": ln["realdata"],
        })
    san = []
    if HAS_SANITIZE:
        for s in SANITIZE:
            if TIER_ORDER[s["tier"]] > want:
                continue
            entry = dict(SANITIZE_DEFAULTS)
            entry.update(s)
            san.append(entry)
    return tests, san


def audit(tier: str, tests):
    """Fail loudly if a lane was deleted that the coverage argument needed.

    The previous shape could lose coverage silently, because coverage was an
    emergent property of two dozen subtraction rules. Here it is an assertion.
    """
    errs = []

    def have(**kw):
        return [t for t in tests if all(t[k] == v for k, v in kw.items())]

    def need(what, **kw):
        if not have(**kw):
            errs.append(f"tier {tier}: nothing covers {what}")

    # Holds in every tier.
    need("the abi3 floor (py%s)" % OLDEST, **{"python-version": OLDEST})
    need("the newest interpreter (py%s)" % NEWEST, **{"python-version": NEWEST})
    need("gcc", toolchain="gcc")
    need("clang", toolchain="clang")
    canon = have(realdata=True)
    if len(canon) != 1:
        errs.append(
            f"tier {tier}: exactly one lane must be canonical for the pinned "
            f"real-data baselines, found {len(canon)}"
        )

    if tier in ("B", "C"):
        for plat in PLATFORMS:
            need(plat, os=plat)
        for py in PYTHONS:
            need(f"py{py}", **{"python-version": py})
        for tool in ("gcc", "clang", "msvc", "appleclang"):
            need(tool, toolchain=tool)
        for arch in ("x86-64", "aarch64"):
            need(arch, arch=arch)
            # The edges of the supported range are where the ABI actually
            # differs, so both ends must be seen on both architectures.
            for py in (OLDEST, NEWEST):
                need(f"py{py} on {arch}", arch=arch, **{"python-version": py})
        # Every Python must reach a compiler that is not MSVC: MSVC-only
        # coverage of a version would leave the C++ that everyone else ships
        # untested at that version.
        for py in PYTHONS:
            if not [t for t in tests
                    if t["python-version"] == py and t["toolchain"] != "msvc"]:
                errs.append(f"tier {tier}: py{py} is only covered under MSVC")

    if tier == "C":
        for py in PYTHONS:
            for comp in ("default", "clang"):
                need(f"linux-amd64 {comp} py{py}",
                     os="linux-amd64", compiler=comp, **{"python-version": py})

    # A subset of B subset of C, always -- a lane promoted out of Tier A by
    # mistake would otherwise stop running on work branches while still looking
    # covered on main.
    for lo, hi in (("A", "B"), ("B", "C")):
        small = {(t["os"], t["python-version"], t["compiler"]) for t in select(lo)[0]}
        big = {(t["os"], t["python-version"], t["compiler"]) for t in select(hi)[0]}
        if not small <= big:
            errs.append(f"tier {lo} is not a subset of tier {hi}: {sorted(small - big)}")

    if errs:
        print("CI matrix coverage audit FAILED:", file=sys.stderr)
        for e in errs:
            print("  - " + e, file=sys.stderr)
        sys.exit(1)


def main():
    tier = pick_tier()
    tests, san = select(tier)
    audit(tier, tests)

    est = sum(PLATFORMS[t["os"]]["minutes"] for t in tests)
    lines = [
        f"### CI tier {tier} - {len(tests)} test jobs"
        + (f" + {len(san)} sanitizer jobs" if san else ""),
        "",
        f"Estimated {est} job-minutes of test lanes.",
        "",
        "| lane | python | toolchain | arch |",
        "|---|---|---|---|",
    ]
    for t in tests:
        lines.append(
            f"| {t['os']} | {t['python-version']} | {t['toolchain']} | {t['arch']} |"
        )
    for s in san:
        lines.append(f"| [sanitize] {s['label']} | 3.14 | {s['cc']} | x86-64 |")
    summary = "\n".join(lines)
    print(summary)

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"tier={tier}\n")
            fh.write("tests=" + json.dumps({"include": tests}, separators=(",", ":")) + "\n")
            fh.write("sanitize=" + json.dumps({"include": san}, separators=(",", ":")) + "\n")
    step = os.environ.get("GITHUB_STEP_SUMMARY")
    if step:
        with open(step, "a", encoding="utf-8") as fh:
            fh.write(summary + "\n")


if __name__ == "__main__":
    main()
