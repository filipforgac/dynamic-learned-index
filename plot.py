from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_recall_vs_n_probe(csv_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['algo', 'nprobe'])

    plt.figure(figsize=(10, 6))
    for algo, group in df.groupby('algo'):
        plt.plot(group['nprobe'], group['recall'], label=algo, marker='o')
    plt.gca().set_facecolor("#f0f0f0")  # Light grey background
    plt.grid(color="#ffffff", linestyle="-", linewidth=1.5)  # White grid lines
    plt.xlabel('Probe')
    plt.ylabel('Recall')
    plt.title(title)
    plt.ylim(bottom=0)
    plt.legend(title='Learned Index')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_recall_vs_n_probe(Path("res_laion_300k.csv"), title='Recall vs Probe [LAION2B]')
    plot_recall_vs_n_probe(Path("res_agnews_mxbai.csv"), 'Recall vs Probe [agnews-mxbai]')
