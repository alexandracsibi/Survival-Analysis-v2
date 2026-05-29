from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def _safe_filename(path: Path) -> str:
    return path.stem.replace("_history", "")


def _plot_single_metric(df, x_col, y_col, out_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col], marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_multiple_metrics(df, x_col, y_cols, out_path, title):
    plt.figure(figsize=(8, 5))

    for y_col in y_cols:
        if y_col in df.columns:
            plt.plot(df[x_col], df[y_col], marker="o", linewidth=1.5, label=y_col)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_history(history_path: Path, output_dir: Path):
    df = pd.read_csv(history_path)

    if "epoch" not in df.columns:
        raise ValueError(f"{history_path} does not contain an 'epoch' column.")

    experiment_name = _safe_filename(history_path)

    relative_parent = history_path.parent.relative_to(PROJECT_ROOT / "results" / "tables")
    out_dir = output_dir / relative_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # 1. Loss plot
    loss_cols = [
        col for col in ["train_loss", "loss", "supervised_loss", "pseudo_loss"]
        if col in df.columns
    ]

    if loss_cols:
        out_path = out_dir / f"{experiment_name}_loss.png"
        _plot_multiple_metrics(
            df=df,
            x_col="epoch",
            y_cols=loss_cols,
            out_path=out_path,
            title=f"{experiment_name} - training losses",
        )
        generated.append(out_path)

    # 2. Validation C-index plot
    val_cols = [
        col for col in ["val_cindex", "val_mean_cindex"]
        if col in df.columns
    ]

    if val_cols:
        out_path = out_dir / f"{experiment_name}_validation.png"
        _plot_multiple_metrics(
            df=df,
            x_col="epoch",
            y_cols=val_cols,
            out_path=out_path,
            title=f"{experiment_name} - validation performance",
        )
        generated.append(out_path)

    # 3. Competing-risk event-specific validation plot
    event_val_cols = [
        col for col in df.columns
        if col.startswith("val_event_") and col.endswith("_cindex")
    ]

    if event_val_cols:
        out_path = out_dir / f"{experiment_name}_event_validation.png"
        _plot_multiple_metrics(
            df=df,
            x_col="epoch",
            y_cols=event_val_cols,
            out_path=out_path,
            title=f"{experiment_name} - event-specific validation C-index",
        )
        generated.append(out_path)

    # 4. Pseudo-label count plot for SSL
    if "selected_pseudo" in df.columns:
        out_path = out_dir / f"{experiment_name}_pseudo_labels.png"
        _plot_single_metric(
            df=df,
            x_col="epoch",
            y_col="selected_pseudo",
            out_path=out_path,
            title=f"{experiment_name} - selected pseudo-labels",
        )
        generated.append(out_path)

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        required=True,
        help="Path to one history CSV, or a directory containing *_history.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory where plots will be saved.",
    )

    args = parser.parse_args()

    history_input = Path(args.history).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    if history_input.is_file():
        history_files = [history_input]
    elif history_input.is_dir():
        history_files = sorted(history_input.rglob("*_history.csv"))
    else:
        raise FileNotFoundError(f"Could not find: {history_input}")

    if not history_files:
        raise FileNotFoundError(f"No *_history.csv files found in {history_input}")

    all_generated = []

    for history_path in history_files:
        print(f"Plotting: {history_path}")
        generated = plot_history(history_path, output_dir)
        all_generated.extend(generated)

        for path in generated:
            print(f"  saved: {path}")

    print(f"\nGenerated {len(all_generated)} plot(s).")


if __name__ == "__main__":
    main()