"""REPORT.md, results.json and the figures.

Figures follow one rule: the data's job picks the form. Convergence is magnitude across
an ordered scale, so it is a log-log line with a labelled reference slope; mode growth is
change over time compared against a prediction, so it is semi-log with the prediction
drawn as the thing being tested; a scan over named variants is a dot plot, not a bar
chart, because the quantity has a meaningful zero to sit against.

Colours are the Okabe-Ito set, checked here rather than assumed: adjacent-pair separation
under protanopia, deuteranopia and tritanopia is >= 8 (OKLab dE x100) and >= 15 for normal
vision. Series are assigned in fixed order and never cycled; every figure with two or more
series carries a legend, so identity never rests on colour alone.
"""

from __future__ import annotations

import json

import numpy as np

from .core import OK, WARN, FAIL, INFO

# fixed categorical order - assigned by position, never by rank, never cycled
C_BLUE, C_VERM, C_GREEN, C_PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
SERIES = (C_BLUE, C_VERM, C_GREEN, C_PURPLE)
C_REF = "#E69F00"          # reference/annotation only, and always directly labelled
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

_MARK = {OK: "ok", WARN: "warn", FAIL: "FAIL", INFO: "--"}


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def make_figures(suite, model, data, log=print):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not available - skipping figures")
        return []
    made = []

    # -- 1. residual convergence ------------------------------------------------------
    if data.get("resid_zones"):
        lv = np.array(data["resid_levels"], float)
        z = data["resid_zones"]
        # Direct labels at the right end instead of a legend box: with four series the
        # box has nowhere to sit that does not land on a line, and a label beside its own
        # curve is read without a lookup anyway.
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        # dy separates the two pairs whose curves end within a decade of each other
        series = [("core", "torus core", "o", -10),
                  ("surface", "torus surface", "s", +10),
                  ("funnel", "inner funnel", "^", -10),
                  ("outer", "outer ambient", "D", +10)]
        for i, (key, label, mark, dy) in enumerate(series):
            col = SERIES[i % len(SERIES)]
            ax.loglog(lv, z[key], mark + "-", color=col, lw=1.8, ms=6.5)
            ax.annotate(label, (lv[-1], z[key][-1]), textcoords="offset points",
                        xytext=(10, dy), va="center", color=col, fontsize=8.5)
        # offset the reference so it is visible instead of hiding under the core line
        ref = 4.0 * z["core"][0] * (lv / lv[0]) ** -2.0
        ax.loglog(lv, ref, "--", color=C_REF, lw=1.6, zorder=1)
        ax.annotate(r"$\propto N^{-2}$", (lv[0], ref[0]), textcoords="offset points",
                    xytext=(6, 6), ha="left", color=C_REF, fontsize=9)
        _style(ax, "$N$", r"$\langle |{\rm resid}_r| \rangle \,/\, \rho g$",
               "Discrete equilibrium residual")
        ax.set_xticks(lv)
        ax.set_xticklabels([f"{int(x)}" for x in lv])
        ax.minorticks_off()
        ax.set_xlim(lv[0] * 0.9, lv[-1] * 2.6)
        lo = min(min(z[k]) for k, _, _, _ in series)
        hi = max(max(z[k]) for k, _, _, _ in series)
        ax.set_ylim(lo * 0.05, hi * 20)          # room for the labels at both ends
        fig.tight_layout()
        p = suite.figs / "fig_residual.png"
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        made.append(p)

    # -- 2. grid convergence ----------------------------------------------------------
    if data.get("conv_profiles"):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for i, (nx, rr, prof) in enumerate(zip(data["conv_levels"],
                                               data["conv_radii"],
                                               data["conv_profiles"])):
            axes[0].semilogy(rr, prof, color=SERIES[i % len(SERIES)], lw=1.6,
                             label=f"${nx}^2$")
        _style(axes[0], "$r$", r"$\langle\rho\rangle_\phi$",
               r"Density profile at $t = 2\,P_{\rm orb}$")
        axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED, title="$N$",
                       title_fontsize=9)

        d = np.array(data["conv_drift"])
        db = np.array(data.get("conv_drift_body", d))
        lv = np.array(data["conv_levels"], float)
        g = data.get("conv_gci", {})
        # Absolute band: the exact drift is zero, so a band quoted as a percentage of a
        # number that is itself converging to zero says nothing.
        if np.isfinite(g.get("gci_abs", np.nan)):
            b = g["gci_abs"]
            axes[1].axhspan(d[-1] - b, d[-1] + b, color=C_BLUE, alpha=0.14, lw=0)
            axes[1].annotate(f"GCI $\\pm{b:.1e}$", (lv[0], d[-1] + b),
                             fontsize=9, color=C_BLUE, va="bottom")
        axes[1].semilogx(lv, db, "s--", color=C_VERM, lw=1.4, ms=6,
                         label=r"$\rho > 2\rho_{\rm atm}$")
        axes[1].semilogx(lv, d, "o-", color=C_BLUE, lw=1.8, ms=7,
                         label=r"$\int\rho\,dV$")
        axes[1].axhline(0, color=GRID, lw=1)
        axes[1].legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="upper right")
        _style(axes[1], "$N$", r"$\Delta m/m_0$",
               r"Mass drift at $t = 2\,P_{\rm orb}$")
        axes[1].set_xticks(lv)
        axes[1].set_xticklabels([f"{int(x)}" for x in lv])
        axes[1].minorticks_off()
        fig.tight_layout()
        p = suite.figs / "fig_convergence.png"
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        made.append(p)

    # -- 3. Papaloizou-Pringle growth -------------------------------------------------
    meas, th = data.get("ppi_measured"), data.get("ppi_theory")
    if meas and th:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for i, cid in enumerate(("ppi_m1", "ppi_m2", "ppi_m3")):
            if cid not in meas:
                continue
            d = meas[cid]
            m = d["m"]
            col = SERIES[i % len(SERIES)]
            a = d["amps"][:, m]
            ax.semilogy(d["times"], a, "o-", color=col, lw=1.6, ms=4.5,
                        label=f"$m = {m}$")
            lo, hi = d["window"]
            sel = (d["times"] >= lo) & (d["times"] <= hi) & (a > 0)
            if sel.sum() > 2:
                rate = th[m]["rate"] * model.P_orb
                t0 = d["times"][sel][0]
                tt = np.linspace(lo, hi, 20)
                ax.semilogy(tt, a[sel][0] * np.exp(rate * (tt - t0)), "--",
                            color=col, lw=1.4, alpha=0.8)
        ax.plot([], [], "--", color=MUTED, lw=1.4, label="linear theory")
        _style(ax, r"$t \; [P_{\rm orb}]$", r"$|A_m|$",
               "Papaloizou-Pringle mode growth")
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower right")
        fig.tight_layout()
        p = suite.figs / "fig_ppi.png"
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        made.append(p)

    # -- 3b. what the growing mode actually looks like --------------------------------
    # The raw density is useless for this: the mode is a percent-level perturbation on a
    # profile spanning eight decades, so it is invisible in any colour scale wide enough
    # to show the disk. Dividing out the azimuthal mean is what makes the pattern appear,
    # and it turns a signed quantity into the one case that wants a diverging map -
    # two hues from the categorical set with a neutral midpoint pinned at zero.
    if data.get("ppi_frames"):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "vv_div", [C_BLUE, "#e8e8e8", C_VERM])
        # One figure per mode. Each panel is normalised by its own RMS so a single colour
        # bar serves the row; the absolute amplitude spans two orders of magnitude across
        # the panels and is given in each title instead, which would otherwise leave the
        # early panels blank. The range is clipped at 2 sigma rather than 3, because the
        # RMS is set largely by the bright ring at the inner edge and a wider range
        # leaves the body of the pattern pale.
        for row in data["ppi_frames"]:
            nc = len(row["times"])
            fig, axes = plt.subplots(1, nc, figsize=(3.9 * nc + 1.0, 4.3),
                                     subplot_kw={"projection": "polar"})
            # Wide gutters: the angular labels of one panel sit where the next panel
            # begins, so they need the room rather than being deleted.
            fig.subplots_adjust(wspace=0.45, right=0.88)
            axes = np.atleast_1d(axes)
            for ax, t, d, pk in zip(axes, row["times"], row["resid"], row["peak"]):
                mesh = ax.pcolormesh(row["phi"], row["r"], d.T, cmap=cmap,
                                     vmin=-3, vmax=3, shading="auto")
                ax.set_title(fr"$t = {t:.1f}\,P_{{\rm orb}}$,  "
                             fr"$\max|\delta| = {pk:.2f}$",
                             color=INK, fontsize=9.5, pad=12)
                ax.set_yticklabels([])
                ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
                ax.tick_params(colors=MUTED, labelsize=7.5, pad=2)
                ax.grid(color=GRID, lw=0.5)
            # One bar to the right of the whole row - per-panel bars land on the
            # neighbouring panel's angular labels. Its axes is placed explicitly rather
            # than through `fraction`/`aspect`, because those measure against the
            # three-panel bounding box and make the bar both thinner and shorter than a
            # single panel. This way it keeps a fixed thickness and spans exactly the
            # height of the panels.
            box = axes[-1].get_position()
            cax = fig.add_axes([box.x1 + 0.030, box.y0, 0.014, box.height])
            cb = fig.colorbar(mesh, cax=cax, orientation="vertical", extend="both")
            cb.set_label(r"$\delta / (1.4826\,{\rm MAD})$,   "
                         r"$\delta = \rho/\langle\rho\rangle_\phi - 1$",
                         color=MUTED, fontsize=9)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            fig.suptitle(fr"Density perturbation, $m = {row['m']}$",
                         color=INK, fontsize=11.5, y=0.99)
            p = suite.figs / f"fig_ppi_pattern_m{row['m']}.png"
            fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            made.append(p)

        # All three modes in one grid, twice: once with each mode at a fixed fraction of
        # its own run, and once at equal growth. The second is the fair comparison of
        # SHAPE - at a common instant the slower modes are simply younger, not different.
        for key, name, title in (
                ("ppi_frames", "fig_ppi_pattern_grid",
                 "Density perturbation: rows are modes, columns are times"),
                ("ppi_frames_efold", "fig_ppi_pattern_efolds",
                 "Density perturbation at equal growth"
                 + (f": {data['ppi_efold_budget']:.1f} e-foldings across the row"
                    if data.get("ppi_efold_budget") else ""))):
            rows = data.get(key)
            if not rows:
                continue
            nr, nc = len(rows), max(len(r["times"]) for r in rows)
            fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc + 1.2, 3.5 * nr + 0.9),
                                     subplot_kw={"projection": "polar"})
            axes = np.atleast_2d(axes)
            fig.subplots_adjust(wspace=0.40, hspace=0.34, right=0.88, left=0.07)
            for row, axrow in zip(rows, axes):
                for k, ax in enumerate(axrow):
                    if k >= len(row["times"]):
                        ax.set_visible(False)
                        continue
                    mesh = ax.pcolormesh(row["phi"], row["r"], row["resid"][k].T,
                                         cmap=cmap, vmin=-3, vmax=3, shading="auto")
                    extra = (fr", $n_e = {row['labels'][k]}$"
                             if k < len(row["labels"]) else "")
                    ax.set_title(fr"$t = {row['times'][k]:.1f}\,P_{{\rm orb}}$"
                                 fr"{extra},  $\max|\delta| = {row['peak'][k]:.2f}$",
                                 color=INK, fontsize=9, pad=10)
                    ax.set_yticklabels([])
                    ax.tick_params(colors=MUTED, labelsize=7, pad=1)
                    ax.grid(color=GRID, lw=0.5)
                pos = axrow[0].get_position()
                fig.text(0.018, pos.y0 + pos.height / 2, fr"$m = {row['m']}$",
                         color=INK, fontsize=12, va="center", ha="left", rotation=90)
            top, bot = axes[0, -1].get_position(), axes[-1, -1].get_position()
            cax = fig.add_axes([top.x1 + 0.030, bot.y0, 0.012, top.y1 - bot.y0])
            cb = fig.colorbar(mesh, cax=cax, orientation="vertical", extend="both")
            cb.set_label(r"$\delta / (1.4826\,{\rm MAD})$,   "
                         r"$\delta = \rho/\langle\rho\rangle_\phi - 1$",
                         color=MUTED, fontsize=9)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            fig.suptitle(title, color=INK, fontsize=12, y=0.985)
            p = suite.figs / f"{name}.png"
            fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            made.append(p)

    # -- 3c. Lynden-Bell & Pringle ring -----------------------------------------------
    if data.get("ring_profile"):
        from . import ring as ring_mod
        rp = data["ring_profile"]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

        ana = ring_mod.sigma(rp["r"], rp["tau"], 1.0, rp["norm"])
        axes[0].plot(rp["r"], ana, color=MUTED, lw=2.6, alpha=0.55,
                     label="LBP74")
        axes[0].plot(rp["r"], rp["sim"], color=C_BLUE, lw=1.6,
                     label=fr"$N = {rp['nx']}$")
        axes[0].set_yscale("log")
        axes[0].set_ylim(max(1e-8, ana.max() * 1e-6), ana.max() * 3)
        _style(axes[0], "$R/R_0$", r"$\Sigma$",
               fr"Surface density at $\tau = {rp['tau']:.2f}$")
        axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED)

        # tau - tau0, not tau: the drift is five orders below the value it sits on, and
        # plotting the absolute number would show a flat line with an offset label.
        nums = data.get("ring_numerical") or {}
        for i, nx in enumerate(sorted(nums)):
            d = nums[nx]
            t = np.array(d["times"])
            tau0 = d["taus"][0]
            axes[1].plot(t, np.array(d["taus"]) - tau0, "o-",
                         color=SERIES[i % len(SERIES)], ms=5, lw=1.2,
                         label=fr"$N = {nx}$:  "
                               fr"$\nu_{{\rm num}} \leq {d['bound']:.0e}$")
        axes[1].axhline(0, color=GRID, lw=1)
        _style(axes[1], "$t$", r"$\tau - \tau_0$",
               r"Numerical spreading, $\nu = 0$")
        if nums:
            axes[1].legend(frameon=False, fontsize=8.5, labelcolor=MUTED,
                           loc="upper center", ncol=1,
                           bbox_to_anchor=(0.5, 1.02))
            axes[1].margins(y=0.35)
        fig.tight_layout()
        p = suite.figs / "fig_ring.png"
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        made.append(p)

    # -- 4. sensitivity to parameters of numerical origin -----------------------------
    if data.get("sens_rows"):
        rows = data["sens_rows"]
        fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(rows) + 2.0))
        y = np.arange(len(rows))
        base = data["sens_base"]
        g = data.get("conv_gci", {})
        if np.isfinite(g.get("gci", np.nan)):
            b = abs(g["gci"] * base)
            ax.axvspan(base - b, base + b, color=C_BLUE, alpha=0.12, lw=0)
        ax.axvline(base, color=C_BLUE, lw=1.4)
        ax.annotate("baseline $\\pm$ GCI", (base, len(rows) - 0.4),
                    fontsize=9, color=C_BLUE, ha="center", va="bottom")
        ax.plot([r[1] for r in rows], y, "o", color=C_VERM, ms=8)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
        ax.set_ylim(-0.7, len(rows) - 0.1)
        _style(ax, r"$\Delta m/m_0$ at $t = 2\,P_{\rm orb}$", "",
               "Sensitivity to parameters of numerical origin")
        fig.tight_layout()
        p = suite.figs / "fig_sensitivity.png"
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        made.append(p)

    return made


def write_report(suite, model, checks, records, figures, elapsed, provenance):
    counts = {s: sum(1 for c in checks if c.status == s)
              for s in (FAIL, WARN, OK, INFO)}
    groups = {}
    for c in checks:
        groups.setdefault(c.group, []).append(c)

    L = []
    add = L.append
    add("# Verification & validation suite")
    add("")
    add(f"| | |\n|---|---|")
    add(f"| Generated | {provenance['date']} on `{provenance['host']}` |")
    add(f"| Commit | `{provenance['git_sha'][:12]}` "
        f"({provenance['git_dirty_files']} modified file(s)) |")
    add(f"| Directory | `{suite.dir}` |")
    add(f"| Cases | {len(records)} runs, {elapsed:.0f} s wall |")
    add(f"| Result | **{counts[FAIL]} failure(s), {counts[WARN]} warning(s)**, "
        f"{counts[OK]} passed, {counts[INFO]} recorded |")
    add("")
    add("The model under test is the dimensionless problem "
        f"`beta = {model.beta:.4e}`, `C' = {model.C_prime}`, `gamma = {model.gamma:.4f}` "
        f"(`n_poly = {model.n_poly:.3f}`), `alpha` as stated per case. `T_0` and `mu` set "
        "the code unit of velocity and cancel from every physical result, which the "
        "`units_T0` check demonstrates rather than assumes.")
    add("")

    order = ["health", "static", "mapping", "residual", "convergence", "temporal",
             "invariance", "sensitivity", "viscosity", "ppi"]
    titles = {
        "health": "0. Run health - did every case actually finish?",
        "static": "1. Static verification - the equilibrium's own algebra",
        "mapping": "2. The C++ generator against scripts/theory",
        "residual": "3. Order of accuracy of the discrete residual",
        "convergence": "4. Grid convergence and numerical uncertainty",
        "temporal": "5. Timestep convergence",
        "invariance": "6. Invariance - what must not change the answer",
        "sensitivity": "7. Parameters of numerical origin",
        "viscosity": "8. Viscosity: the alpha-law and what it transports",
        "ppi": "9. Papaloizou-Pringle instability vs linear theory",
    }
    for g in order:
        if g not in groups:
            continue
        add(f"## {titles[g]}")
        add("")
        add("| | check | value | expected | detail |")
        add("|---|---|---|---|---|")
        for c in groups[g]:
            v = "" if c.value is None else f"`{c.value:.4g}`"
            add(f"| {_MARK[c.status]} | `{c.name}` | {v} | {c.expected} | {c.detail} |")
        add("")

    if figures:
        add("## Figures")
        add("")
        for f in figures:
            add(f"![{f.stem}](figs/{f.name})")
            add("")

    add("## Runs")
    add("")
    add("| case | status | seconds | overrides |")
    add("|---|---|---|---|")
    for cid in sorted(records):
        r = records[cid]
        ov = " ".join(a for a in r.get("argv", [])
                      if not a.startswith(("time/ncycle_out", "output")))
        add(f"| `{cid}` | {r['status']} | {r['seconds']:.1f} | `{ov}` |")
    add("")
    add("Every case directory holds the exact `athinput.in` it was run with, the command "
        "line, and its own log; `materials/` holds the generator source, the configure "
        "line and the commit, so any number here can be traced to the binary that "
        "produced it.")

    (suite.dir / "REPORT.md").write_text("\n".join(L) + "\n")

    payload = {
        "provenance": provenance,
        "elapsed_seconds": round(elapsed, 1),
        "counts": {k: counts[k] for k in (FAIL, WARN, OK, INFO)},
        "checks": [dict(group=c.group, name=c.name, status=c.status, value=c.value,
                        expected=c.expected, detail=c.detail) for c in checks],
        "runs": records,
    }
    (suite.dir / "results.json").write_text(json.dumps(payload, indent=2,
                                                       default=str) + "\n")
    return counts
