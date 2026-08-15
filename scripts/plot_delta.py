import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    # 1. Zielordner festlegen
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "evaluation_runs" / "run_20260815_154808"

    stats_file = run_dir / "data" / "summary_statistics.json"
    if not stats_file.exists():
        print(f"Error: Could not find {stats_file}")
        return

    # 2. Daten laden
    with open(stats_file, 'r') as f:
        data = json.load(f)

    # Relevante NLP Metriken
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
    display_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore F1"]

    baseline_raw = []
    rag_raw = []
    pct_change = []

    # 3. Rohdaten extrahieren und prozentuale Änderung berechnen
    for m in metrics:
        b_mean = data.get(f"baseline_{m}", {}).get("mean", 0)
        r_mean = data.get(f"rag_{m}", {}).get("mean", 0)

        baseline_raw.append(b_mean)
        rag_raw.append(r_mean)

        # Prozentuale Veränderung: (Neu - Alt) / Alt * 100
        if b_mean != 0:
            change = ((r_mean - b_mean) / b_mean) * 100
        else:
            change = 0
        pct_change.append(change)

    # 4. Diverging Bar Chart initialisieren
    fig, ax = plt.subplots(figsize=(10, 6))

    # Farben: Grün für Verbesserung (>0), Rot für Verschlechterung (<0)
    colors = ['tab:green' if val > 0 else 'tab:red' for val in pct_change]

    # Horizontale Balken zeichnen
    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, pct_change, color=colors, height=0.6, edgecolor='black', linewidth=0.5)

    # 5. Styling und Achsen
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=12, weight='bold')

    # Nulllinie (Baseline) fett markieren
    ax.axvline(0, color='black', linewidth=1.5)

    ax.set_xlabel("Percentage Change vs. Baseline (%)", fontsize=12, weight='bold')
    ax.set_title("Relative Performance Impact of RAG Architecture", fontsize=15, weight='bold', pad=20)

    # X-Achse dynamisch erweitern, damit der Text Platz hat
    max_abs_change = max([abs(x) for x in pct_change] + [1.0])  # Mindestens 1% Skala
    ax.set_xlim(-max_abs_change * 1.3, max_abs_change * 1.3)

    # Raster für bessere Lesbarkeit
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # 6. Exakte Prozentwerte an die Balken schreiben
    for bar, change in zip(bars, pct_change):
        # Positionierung des Textes leicht außerhalb des Balkens
        offset = max_abs_change * 0.02
        x_pos = change + offset if change > 0 else change - offset
        ha = 'left' if change > 0 else 'right'

        # Text formatieren (mit + Zeichen für positive Werte)
        text_str = f"{change:+.2f}%"

        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, text_str,
                va='center', ha=ha, fontsize=11, weight='bold',
                color='darkgreen' if change > 0 else 'darkred')

    # 7. Speichern
    out_path = run_dir / "plots" / "nlp_performance_delta.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Diverging bar chart saved to: {out_path}")


if __name__ == "__main__":
    main()