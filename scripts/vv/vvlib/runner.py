"""Run the case matrix into results/tests/test-<timestamp>/ and cache the results.

Layout mirrors build.sh, so a V&V suite looks like every other run in results/:

    results/tests/test-20260731-011530/
        data/<case-id>/     one Athena++ run each: .hst, .tab, athinput.in, run.log
        materials/          provenance: athinput template, pgen source, configure.log
        figs/               figures
        REPORT.md           human-readable verdicts
        results.json        the same verdicts, machine-readable

Cases are declarative (id + parameter overrides), so the matrix is data rather than
control flow and the report can name exactly which run produced which number. A case
whose `done.json` already matches its overrides is skipped, which makes re-running the
suite in an existing directory cheap.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import time

from .core import repo_root


@dataclasses.dataclass
class Case:
    """One Athena++ invocation. `est` is a rough serial cost in seconds, used only for
    scheduling the longest jobs first so the wall clock is not held hostage by one of
    them starting last."""

    cid: str
    overrides: dict
    tags: tuple = ()
    est: float = 10.0
    #: optional second invocation in the same directory, e.g. ["-r", "<file>.rst", ...].
    #: Used only by the restart test, where the point IS that it is two invocations.
    post_argv: tuple = ()
    #: extra text appended to this case's athinput. ParameterInput::ModifyFromCmdline
    #: refuses to create blocks or parameters that the input file does not already have,
    #: so anything new (a restart output block) has to arrive this way.
    extra_input: str = ""
    #: which binary and which input to use. Athena++ compiles one problem generator per
    #: binary, so the Lynden-Bell & Pringle ring needs its own; see build_ring.sh.
    binary: str = "athena"
    athinput: str = "athinput.acc_disk_visc_vv"
    #: command prefix, e.g. ("mpirun", "-n", "4"). Part of the cached argv, so changing
    #: the rank count invalidates the cached run rather than reusing it silently.
    launcher: tuple = ()
    #: wall-clock ceiling. A collapsed timestep does not crash - it burns CPU forever
    #: while model time stands still, and .hst stops growing so nothing looks wrong.
    #: Without this the suite hangs instead of reporting the collapse, which is the
    #: opposite of what it is for. None means max(180 s, 5x est).
    timeout: float | None = None

    def wall_limit(self) -> float:
        return self.timeout if self.timeout else max(180.0, 5.0 * self.est)

    def argv(self):
        return [*self.launcher, *(f"{k}={v}" for k, v in sorted(self.overrides.items()))]


class Suite:
    def __init__(self, root=None, tag="test", reuse=None):
        self.repo = repo_root()
        if reuse:
            self.dir = pathlib.Path(reuse).resolve()
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            base = root or (self.repo / "results" / "tests")
            self.dir = pathlib.Path(base) / f"{tag}-{stamp}"
        self.data = self.dir / "data"
        self.materials = self.dir / "materials"
        self.figs = self.dir / "figs"
        for d in (self.data, self.materials, self.figs):
            d.mkdir(parents=True, exist_ok=True)
        self.athinput = self.repo / "inputs" / "hydro" / "athinput.acc_disk_visc_vv"
        self.binary = self.repo / "bin" / "athena"

    # -- provenance ---------------------------------------------------------

    def record_provenance(self):
        """A convergence study is only meaningful if every point in it demonstrably came
        from the same binary. Copy what identifies that binary next to the results."""
        for src in (self.athinput,
                    self.repo / "src" / "pgen" / "acc_disk_visc_vv.cpp",
                    self.repo / "configure.log",
                    self.repo / ".build_state"):
            if src.exists():
                shutil.copy(src, self.materials / src.name)

        def git(*args):
            try:
                return subprocess.run(["git", "-C", str(self.repo), *args],
                                      capture_output=True, text=True,
                                      timeout=10).stdout.strip()
            except Exception:
                return "unknown"

        info = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": os.uname().nodename,
            "git_sha": git("rev-parse", "HEAD"),
            "git_dirty_files": len([x for x in git("status", "--porcelain").splitlines()
                                    if x]),
            "binary_mtime": (datetime.datetime.fromtimestamp(self.binary.stat().st_mtime)
                             .isoformat(timespec="seconds")
                             if self.binary.exists() else "missing"),
        }
        (self.materials / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
        return info

    # -- running ------------------------------------------------------------

    def case_dir(self, case: Case) -> pathlib.Path:
        return self.data / case.cid

    def case_dir_path(self, cid: str) -> pathlib.Path:
        return self.data / cid

    def _is_done(self, case: Case) -> bool:
        marker = self.case_dir(case) / "done.json"
        if not marker.exists():
            return False
        try:
            prev = json.loads(marker.read_text())
        except json.JSONDecodeError:
            return False
        return prev.get("argv") == case.argv()

    def _run_one(self, case: Case) -> dict:
        cdir = self.case_dir(case)
        if self._is_done(case):
            # tags travel with the record even when nothing is re-run: the checks use
            # them to tell a run that is meant to stay smooth from one deliberately
            # driven unstable, and a cached record without them silently mixes the two
            return {"cid": case.cid, "status": "cached", "seconds": 0.0,
                    "argv": case.argv(), "tags": list(case.tags)}
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir(parents=True)

        # Every case gets its own copy of the input, so the run directory records
        # exactly what was solved without having to re-derive it from command.txt.
        template = self.repo / "inputs" / "hydro" / case.athinput
        local_input = cdir / "athinput.in"
        local_input.write_text(template.read_text() + case.extra_input)

        # Athena++ compiles one problem generator, and one MPI setting, into the binary,
        # so some cases need one of their own. Say which script builds the missing one
        # rather than leaving a traceback.
        binary = self.repo / "bin" / case.binary
        if not binary.exists():
            builder = {"athena_visc_ring": "./scripts/vv/build_ring.sh",
                       "athena_mpi": "./scripts/vv/build_mpi.sh"}.get(case.binary)
            (cdir / "run.log").write_text(
                f"{binary} not found.\n"
                + (f"Build it once with:  {builder}\n" if builder else ""))
            return {"cid": case.cid, "status": "failed", "returncode": -2,
                    "seconds": 0.0, "argv": case.argv(), "tags": list(case.tags)}

        limit = case.wall_limit()
        overrides = [f"{k}={v}" for k, v in sorted(case.overrides.items())]
        argv = [*case.launcher, str(binary), "-i", "athinput.in", *overrides]
        t0 = time.time()
        rc, timed_out = self._invoke(argv, cdir, limit, "w")
        lines = [" ".join(argv)]
        if rc == 0 and case.post_argv:
            argv2 = [*case.launcher, str(binary), *case.post_argv]
            lines.append(" ".join(argv2))
            rc, timed_out = self._invoke(argv2, cdir, limit, "a",
                                         banner="\n=== second invocation ===\n")
        (cdir / "command.txt").write_text("\n".join(lines) + "\n")
        dt = time.time() - t0

        status = "timeout" if timed_out else ("ok" if rc == 0 else "failed")
        rec = {"cid": case.cid, "status": status, "returncode": rc,
               "seconds": round(dt, 2), "wall_limit": limit,
               "argv": case.argv(), "tags": list(case.tags)}
        # A timeout is cached like a success: the partial output is real data, and
        # re-running would only hang again. Delete the case directory to force a retry.
        if status in ("ok", "timeout"):
            (cdir / "done.json").write_text(json.dumps(rec, indent=2) + "\n")
        return rec

    @staticmethod
    def _invoke(argv, cdir, limit, mode, banner=""):
        with open(cdir / "run.log", mode) as log:
            if banner:
                log.write(banner)
            proc = subprocess.Popen(argv, cwd=cdir, stdout=log,
                                    stderr=subprocess.STDOUT)
            try:
                return proc.wait(timeout=limit), False
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                log.write(f"\n### killed by scripts/vv after {limit:.0f} s\n")
                return -1, True

    def run(self, cases, jobs=None, log=print) -> dict:
        """Run the cases, longest first, `jobs` at a time. Returns {cid: record}."""
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not found - build it first")
        jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
        todo = sorted(cases, key=lambda c: -c.est)
        pending = [c for c in todo if not self._is_done(c)]
        est = sum(c.est for c in pending)
        log(f"  {len(todo)} cases, {len(pending)} to run, {jobs} at a time "
            f"(~{est:.0f} s serial, ~{est / jobs + max([c.est for c in pending], default=0):.0f} s expected)")

        out, done = {}, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(self._run_one, c): c for c in todo}
            for fut in concurrent.futures.as_completed(futures):
                rec = fut.result()
                out[rec["cid"]] = rec
                done += 1
                mark = {"ok": "ok", "cached": "--", "failed": "FAILED",
                        "timeout": "KILLED"}[rec["status"]]
                log(f"  [{done:>2}/{len(todo)}] {mark:>6}  {rec['cid']:<34} "
                    f"{rec['seconds']:>7.1f} s")
        return out
