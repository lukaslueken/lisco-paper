"""
Scaling visualization for the nonlinear double integrator case study.

Loads three solver evaluations trained on the NMPC problem with different
prediction horizons N (10, 20, 30). Verifies that the configurations are
otherwise identical, and overlays the KKT convergence curves.

Allowed differences across the three runs:
    - nlp_cfg["N"]              (prediction horizon)
    - model_cfg["n_in"]          (depends on N via n_p + n_z + 1)
    - model_cfg["n_out"]         (depends on N via n_z)
    - model_cfg["n_neurons"]     (NN width scales with N)
    - train_cfg["predictor_pth"] (each N has its own predictor)
"""

# %% Imports
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

plt.style.use(['science', 'ieee', 'no-latex'])

# %% Configuration
RUN_FOLDER = "results"
SAVE_FOLDER = "visualization"
TOL = 1e-6

# Solver experiment ids per prediction horizon
solver_exps = {
    10: "exp_0",
    20: "exp_2",
    30: "exp_3",
}

FILE_PTH = Path(__file__).parent.resolve()
RESULTS_PATH = FILE_PTH.joinpath(RUN_FOLDER)

# Keys that are allowed to differ across the three solver runs
ALLOWED_DIFFS = {
    ("nlp_cfg", "N"),
    ("model_cfg", "n_in"),
    ("model_cfg", "n_out"),
    ("model_cfg", "n_neurons"),
    ("train_cfg", "predictor_pth"),
    ("train_cfg", "N_epochs"),
}

# Top-level config sections to compare
COMPARE_SECTIONS = ["model_cfg", "train_cfg", "nlp_cfg", "solver_cfg"]


# %% Utility Functions
def load_results(mode, exp_id):
    folder = RESULTS_PATH.joinpath(mode, exp_id)
    with open(folder.joinpath("eval.json"), "r") as f:
        eval_dict = json.load(f)
    with open(folder.joinpath("config.json"), "r") as f:
        config_dict = json.load(f)
    return {"eval": eval_dict, "config": config_dict}


def assert_configs_consistent(results_by_N, allowed_diffs, sections):
    """Assert that all configs match except for the allowed-diff keys."""
    Ns = sorted(results_by_N.keys())
    ref_N = Ns[0]
    ref_cfg = results_by_N[ref_N]["config"]

    for N in Ns[1:]:
        cfg = results_by_N[N]["config"]
        for section in sections:
            ref_section = ref_cfg.get(section, {})
            cur_section = cfg.get(section, {})
            keys = set(ref_section.keys()) | set(cur_section.keys())
            for key in keys:
                if (section, key) in allowed_diffs:
                    continue
                assert ref_section.get(key) == cur_section.get(key), (
                    f"Config mismatch for N={ref_N} vs N={N} at "
                    f"{section}.{key}: {ref_section.get(key)!r} vs {cur_section.get(key)!r}"
                )

    # Sanity: the allowed-diff keys should actually differ for N (and likely the others).
    for N in Ns[1:]:
        assert ref_cfg["nlp_cfg"]["N"] != results_by_N[N]["config"]["nlp_cfg"]["N"], \
            f"Expected different N between {ref_N} and {N}"


def plot_kkt_convergence_overlay(results_by_N, ax=None, tol=None):
    """Overlay max-KKT convergence curves for each N."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    max_iters = 1
    for N in sorted(results_by_N.keys()):
        traj = results_by_N[N]["eval"]["trajectory_evaluation"]
        iters = np.arange(len(traj)) + 1
        kkt = [v["KKT_inf_max"] for v in traj]
        ax.plot(iters, kkt, label=f"N={N}")
        max_iters = max(max_iters, len(traj))

    if tol is not None:
        ax.axhline(tol, color='red', linestyle='--', linewidth=0.3,
                   label=f'Tolerance: {tol:.1e}')

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$||KKT||_{\infty}$ (max)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1, max_iters])
    ax.set_ylim([1e-12, 1e2])
    ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    ax.legend(loc="upper right", fontsize="small")
    return fig, ax


# %% Load Results
results_by_N = {N: load_results("solver", exp_id) for N, exp_id in solver_exps.items()}

# %% Validate Consistency
for N, exp_id in solver_exps.items():
    actual_N = results_by_N[N]["config"]["nlp_cfg"]["N"]
    assert N == actual_N, f"Key {N} maps to {exp_id} but its nlp_cfg.N is {actual_N}"

assert_configs_consistent(results_by_N, ALLOWED_DIFFS, COMPARE_SECTIONS)
print("Config consistency check passed.")
for N in sorted(results_by_N.keys()):
    cfg = results_by_N[N]["config"]
    print(
        f"  N={N:>3}  n_neurons={cfg['model_cfg']['n_neurons']:>5}  "
        f"n_in={cfg['model_cfg']['n_in']:>4}  n_out={cfg['model_cfg']['n_out']:>4}  "
        f"predictor_pth={cfg['train_cfg']['predictor_pth']}"
    )

# %% Summary Table
import pandas as pd

table_spec = [
    ("Speedup over IPOPT", "min",    "cpu_speedup_factor_min"),
    ("Speedup over IPOPT", "1st perc.",    "cpu_speedup_factor_01"),
    ("Speedup over IPOPT", "median",    "cpu_speedup_factor_med"),
    ("Speedup over IPOPT", "99th perc.",    "cpu_speedup_factor_99"),
    ("Speedup over IPOPT", "max",    "cpu_speedup_factor_max"),
    ("Solver Iterations",       "median",    "cpu_n_iter_med"),
    ("Solver Iterations",       "99th perc.",     "cpu_n_iter_99"),
    ("Solver Iterations",       "max",     "cpu_n_iter_max"),
    ("Total Solve Time [s]",     "median",    "cpu_full_solve_time_med"),
    ("Total Solve Time [s]",     "99th perc.",     "cpu_full_solve_time_99"),
    ("Total Solve Time [s]",     "max",     "cpu_full_solve_time_max"),
    ("Success Rate",       "",       "success_rate"),
    # ("Solver Iterations",       "90",     "cpu_n_iter_90"),
    # ("Solver Iterations",       "95",     "cpu_n_iter_95"),
    # ("Total Solve Time [s]",     "90",     "cpu_full_solve_time_90"),
    # ("Total Solve Time [s]",     "95",     "cpu_full_solve_time_95"),
]

Ns_sorted = sorted(results_by_N.keys())
columns = [f"N={N}" for N in Ns_sorted]
index = pd.MultiIndex.from_tuples([(m, s) for m, s, _ in table_spec], names=["Metric", "Stat"])

data = np.array([
    [results_by_N[N]["eval"][key] for N in Ns_sorted]
    for _, _, key in table_spec
])
table_df = pd.DataFrame(data, index=index, columns=columns)

# Prepend model-info rows (n_neurons, n_z, training epochs) from each run's config
INFO_LABEL_NEURONS = "Neurons per Layer"
INFO_LABEL_NZ = "$n_z$"
INFO_LABEL_EPOCHS = "Training Epochs"
info_rows = [(INFO_LABEL_NEURONS, ""), (INFO_LABEL_NZ, ""), (INFO_LABEL_EPOCHS, "")]
info_data = np.array([
    [results_by_N[N]["config"]["model_cfg"]["n_neurons"]   for N in Ns_sorted],
    [results_by_N[N]["config"]["model_cfg"]["n_out"]       for N in Ns_sorted],
    [results_by_N[N]["config"]["N_epochs_trained"]         for N in Ns_sorted],
])
info_df = pd.DataFrame(
    data=info_data,
    index=pd.MultiIndex.from_tuples(info_rows, names=["Metric", "Stat"]),
    columns=columns,
)
table_df = pd.concat([info_df, table_df])

# Append IPOPT reference rows (median and max solve times, loaded per N)
IPOPT_DATA_BASE = FILE_PTH.joinpath("case_study_data_v2")
IPOPT_ROW_LABEL = "IPOPT Solve Time [s]"
ipopt_rows = [(IPOPT_ROW_LABEL, "median"), (IPOPT_ROW_LABEL, "max")]
ipopt_data = {}
for N in Ns_sorted:
    test_data = results_by_N[N]["eval"]["test_data"]
    solve_times = np.load(IPOPT_DATA_BASE.joinpath(test_data, "nmpc_data.npz"))["solve_time"]
    ipopt_data[N] = {"median": float(np.median(solve_times)), "max": float(np.max(solve_times))}

ipopt_df = pd.DataFrame(
    data=np.array([[ipopt_data[N][stat] for N in Ns_sorted] for _, stat in ipopt_rows]),
    index=pd.MultiIndex.from_tuples(ipopt_rows, names=["Metric", "Stat"]),
    columns=columns,
)
table_df = pd.concat([table_df, ipopt_df])

table_formatted = table_df.copy().astype(object)
for metric, stat in table_df.index:
    for col in columns:
        value = table_df.loc[(metric, stat), col]
        if pd.isna(value):
            table_formatted.loc[(metric, stat), col] = "N/A"
        elif metric in (INFO_LABEL_NEURONS, INFO_LABEL_NZ, INFO_LABEL_EPOCHS):
            table_formatted.loc[(metric, stat), col] = f"{int(value)}"
        elif metric == "Success Rate":
            table_formatted.loc[(metric, stat), col] = f"{value:.4f}"
        elif metric == "Solver Iterations":
            table_formatted.loc[(metric, stat), col] = f"{value:.1f}"
        elif metric in ("Total Solve Time [s]", IPOPT_ROW_LABEL):
            table_formatted.loc[(metric, stat), col] = f"{value:.2e}"
        elif metric == "Speedup over IPOPT":
            table_formatted.loc[(metric, stat), col] = f"{value:.2f}"

print("\nScaling Summary Table")
print("=" * 60)
print(table_formatted.to_string())

save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
save_path.mkdir(parents=True, exist_ok=True)

latex_path = save_path.joinpath("table_scaling_N.tex")
latex_string = table_formatted.to_latex(
    escape=False,
    multirow=True,
    column_format='ll' + 'c' * len(table_formatted.columns)
)

# Inject a midrule before the IPOPT reference block to separate it from the solver rows.
latex_lines = latex_string.split('\n')
for i, line in enumerate(latex_lines):
    if IPOPT_ROW_LABEL in line:
        latex_lines.insert(i, '\\midrule')
        break
latex_string = '\n'.join(latex_lines)

with open(latex_path, 'w') as f:
    f.write(latex_string)
print(f"Saved scaling summary table to {latex_path}")

# Compact version: flatten the (Metric, Stat) MultiIndex into a single column.
# Metric goes on its own header row; stats are indented below via \quad.
# Metrics whose only stat is "" stay on a single line with their values.
compact_index = []
compact_data = []
compact_metrics = []  # parallel to compact_index: metric name per row
prev_metric = None
metric_level = table_formatted.index.get_level_values(0)
for (metric, stat) in table_formatted.index:
    n_stats_for_metric = int((metric_level == metric).sum())
    values = [table_formatted.loc[(metric, stat), col] for col in columns]
    if n_stats_for_metric == 1 and stat == "":
        compact_index.append(metric)
        compact_data.append(values)
        compact_metrics.append(metric)
    else:
        if metric != prev_metric:
            compact_index.append(metric)
            compact_data.append(["" for _ in columns])
            compact_metrics.append(metric)
        compact_index.append(f"\\quad {stat}")
        compact_data.append(values)
        compact_metrics.append(metric)
    prev_metric = metric

compact_df = pd.DataFrame(compact_data, index=compact_index, columns=columns)
compact_df.index.name = ""

print("\nScaling Summary Table (compact)")
print("=" * 60)
print(compact_df.to_string())

latex_compact_string = compact_df.to_latex(
    escape=False,
    column_format='l' + 'c' * len(columns),
)

# Inject a \midrule after each major group: the descriptor block
# (Neurons per Layer + $n_z$ as one group) and every other metric as its own.
DESCRIPTOR_METRICS = {INFO_LABEL_NEURONS, INFO_LABEL_NZ, INFO_LABEL_EPOCHS}
def _group_key(metric):
    return "_descriptors" if metric in DESCRIPTOR_METRICS else metric

end_of_group = [False] * len(compact_metrics)
for i in range(len(compact_metrics) - 1):
    if _group_key(compact_metrics[i]) != _group_key(compact_metrics[i + 1]):
        end_of_group[i] = True

latex_compact_lines = latex_compact_string.split('\n')
new_lines = []
data_row_count = 0
seen_first_midrule = False
for line in latex_compact_lines:
    new_lines.append(line)
    if not seen_first_midrule:
        if line.strip() == '\\midrule':
            seen_first_midrule = True
        continue
    stripped = line.strip()
    if stripped.endswith('\\\\') and '&' in stripped:
        if data_row_count < len(end_of_group) and end_of_group[data_row_count]:
            new_lines.append('\\midrule')
        data_row_count += 1
latex_compact_string = '\n'.join(new_lines)

latex_compact_path = save_path.joinpath("table_scaling_N_compact.tex")
with open(latex_compact_path, 'w') as f:
    f.write(latex_compact_string)
print(f"Saved compact scaling summary table to {latex_compact_path}")

# %% KKT Convergence Plot
save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
save_path.mkdir(parents=True, exist_ok=True)

fig, ax = plot_kkt_convergence_overlay(results_by_N, tol=TOL)
fig.savefig(save_path.joinpath("kkt_convergence_scaling_N.png"), bbox_inches='tight', dpi=300)
fig.savefig(save_path.joinpath("kkt_convergence_scaling_N.pdf"), bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved KKT convergence overlay to {save_path}")
