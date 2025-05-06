import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
import glob
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

def load_all_data(file_paths):
    """
    Load every JSON once, return a dict mapping file_path → {'df': DataFrame, 'meta': config dict}.
    We also inject kp, kd, and a guaranteed 'mode' into each DataFrame for easy filtering later.
    """
    store = {}
    for fp in file_paths:
        js = json.loads(Path(fp).read_text())
        meta = js['config']
        # fallback mode from filename if absent
        mode = meta.get('mode') or ('real' if Path(fp).stem.startswith('real_') else 'sim')
        df = pd.DataFrame(js['data'])
        # inject config fields for convenience
        df = df.assign(kp=meta['kp'], kd=meta['kd'], mode=mode)
        store[fp] = {'df': df, 'meta': meta, 'mode': mode}
    return store

def plot_kp_kd_matrix(actuator_files, actuator_id, data_type, output_file):
    """Create a KP×KD matrix of position plots using cached data."""
    # preload everything once
    data_store = load_all_data(actuator_files)

    # discover unique kp/kd
    kp_values = sorted({info['meta']['kp'] for info in data_store.values()})
    kd_values = sorted({info['meta']['kd'] for info in data_store.values()})
    n_rows, n_cols = len(kp_values), len(kd_values)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4*n_cols, 3*n_rows),
                             sharex=True, sharey=True)

    # normalize axes array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # annotate KP/KD labels
    # for i, kp in enumerate(kp_values):
    #     axes[i, 0].set_ylabel("Pos (deg)", fontsize=10)
    #     fig.text(0.08, (i+0.5)/n_rows, f"KP={kp}",
    #              fontsize=12, fontweight='bold',
    #              ha='right', va='center',
    #              bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", alpha=0.6),
    #              transform=fig.transFigure)
    # for j, kd in enumerate(kd_values):
    #     axes[0, j].annotate(f"KD={kd}", xy=(0.5,1.05), xycoords='axes fraction',
    #                         fontsize=12, fontweight='bold',
    #                         ha='center', va='bottom',
    #                         bbox=dict(boxstyle="round,pad=0.3", fc="lightgray",
    #                                   ec="black", alpha=0.6))

    # plot each cell
    for i, kp in enumerate(kp_values):
        for j, kd in enumerate(kd_values):
            ax = axes[i, j]

            # **NEW: title each subplot with its KP/KD**
            ax.set_title(f"KP={kp}, KD={kd}", fontsize=8, pad=4)

            # filter real & sim
            subset = [fp for fp, info in data_store.items()
                      if info['meta']['kp']==kp and info['meta']['kd']==kd]
            if not subset:
                ax.text(0.5,0.5,"No Data", ha='center', va='center',
                        transform=ax.transAxes)
                continue

            # real first
            for fp in subset:
                info = data_store[fp]
                df = info['df']
                if info['mode']=='real':
                    ax.plot(df["time_since_start"], df["commanded_position"],
                            "k:", linewidth=2, label="Commanded")
                    ax.plot(df["time_since_start"], df["position"],
                            color='#FFA07A', linewidth=1.2, label="Real")
            # then sim
            for fp in subset:
                info = data_store[fp]
                df = info['df']
                if info['mode']=='sim':
                    if not any(data_store[other]['mode']=='real' for other in subset):
                        ax.plot(df["time_since_start"], df["commanded_position"],
                                "k:", linewidth=2, label="Commanded")
                    ax.plot(df["time_since_start"], df["position"],
                            color='#6BB7E0', linestyle='--', linewidth=1.5, label="Sim")
                    m = info['meta']
                    sim_params = (
                        f"\nArm={m.get('armature','N/A')}, "
                        f"Fric={m.get('frictionloss','N/A')}, "
                        f"Damp={m.get('damping','N/A')}"
                    )
                    ax.annotate(sim_params, xy=(0.5,-0.18), xycoords='axes fraction',
                                fontsize=7, ha='center', va='bottom')

            ax.grid(True, alpha=0.3)

    # legend, labels, title, spacing
    axes[0,0].legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.04, "Time (s)", ha='center', fontsize=11)
    plt.suptitle(f"Actuator {actuator_id} - {data_type.capitalize()} Response", fontsize=14)
    plt.subplots_adjust(left=0.10, hspace=0.4, wspace=0.3)
    plt.tight_layout(rect=[0.10, 0.05, 1, 0.95])

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved → {output_file}")


def main():
    today = datetime.now().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="KP/KD matrix plots")
    p.add_argument("--date", default="20250505", help="YYYYMMDD")
    p.add_argument("--type", default="chirp", choices=["chirp","step"])
    args = p.parse_args()

    data_dir = Path("DATA")/args.date/args.type
    if not data_dir.exists():
        print(f"Not found: {data_dir}"); sys.exit(1)

    files = list(data_dir.glob("*.json"))
    if not files:
        print(f"No JSON in {data_dir}"); sys.exit(1)

    # group by actuator ID
    actuators = {}
    for fp in files:
        act_id = fp.stem.split("_")[1]
        actuators.setdefault(act_id, []).append(str(fp))

    out_dir = Path(f"kp_kd_plots_{args.date}_{args.type}")
    out_dir.mkdir(exist_ok=True)

    for aid, fps in actuators.items():
        out_file = out_dir/f"actuator_{aid}_{args.type}.png"
        plot_kp_kd_matrix(fps, aid, args.type, str(out_file))

if __name__=="__main__":
    main()
