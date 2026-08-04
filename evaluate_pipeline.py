import os
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

from bert_score import score as bert_score_calc

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from rag_pipeline import ClinicalRAGSystem, PROJECT_ROOT


class ClinicalEvaluator:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.system = ClinicalRAGSystem()
        self.test_metadata_path = PROJECT_ROOT / "data" / "test_samples" / "test_metadata.csv"
        self.images_dir = PROJECT_ROOT / "data" / "test_samples"
        
        # Setup Output Directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = PROJECT_ROOT / "evaluation_runs" / f"run_{timestamp}"
        self.data_dir = self.run_dir / "data"
        self.plots_dir = self.run_dir / "plots"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load evaluators
        logger.info("Initializing offline NLP evaluation metrics...")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_smoother = SmoothingFunction().method1
        self.bertscore_model = "allenai/scibert_scivocab_uncased"
        
        # Disable overly verbose BERTScore logging
        logging.getLogger("transformers").setLevel(logging.WARNING)

    def _generate_metadata(self, execution_time: float, num_samples: int):
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "num_samples": num_samples,
            "debug_mode": self.debug,
            "bertscore_model": self.bertscore_model,
            "model_name": "google/medgemma-1.5-4b-it",
            "generation_max_new_tokens": 512,
            "generation_do_sample": False,
            "generation_device": str(self.system.device),
        }
        with open(self.data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def _calculate_length_ratio(self, preds: List[str], refs: List[str]) -> List[float]:
        ratios = []
        for p, r in zip(preds, refs):
            p_len = len(p.split())
            r_len = len(r.split())
            # Prevent division by zero
            r_len = max(r_len, 1)
            ratios.append(p_len / r_len)
        return ratios
        
    def _generate_plots(self, df: pd.DataFrame):
        logger.info("Generating statistical visualizations...")
        sns.set_theme(style="whitegrid")
        
        # 1. Boxplots for ROUGE, BLEU, and BERTScore
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
        melted_df = pd.melt(df, value_vars=[m for m in metrics_to_plot if m in df.columns], 
                            var_name="Metric", value_name="Score")
        sns.boxplot(x="Metric", y="Score", data=melted_df, palette="Set2")
        plt.title("Distribution of NLP Evaluation Metrics")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metrics_boxplot.png", dpi=300)
        plt.close()
        
        # 2. Scatter Plot: Ground Truth Length vs Generated Length
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="gt_length", y="gen_length", hue="category", data=df, s=100, alpha=0.7)
        # Add y=x line
        max_len = max(df["gt_length"].max(), df["gen_length"].max()) + 10
        plt.plot([0, max_len], [0, max_len], 'r--', label='Ideal 1:1 Ratio')
        plt.title("Report Verbosity: Generated vs Ground Truth Length")
        plt.xlabel("Ground Truth Length (Words)")
        plt.ylabel("Generated Length (Words)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "length_scatter.png", dpi=300)
        plt.close()
        
        # 3. Radar Chart
        try:
            from math import pi
            means = df[[m for m in metrics_to_plot if m in df.columns]].mean()
            categories = list(means.index)
            N = len(categories)
            
            values = means.values.tolist()
            values += values[:1] # Repeat first value to close the circular graph
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)
            
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, 'b', alpha=0.25)
            
            plt.title("Normalized Mean Scores")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "radar_chart.png", dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate radar chart: {e}")

    def run_evaluation(self):
        logger.info("Starting Quantitative Evaluation Pipeline...")
        start_time = time.time()
        
        if not self.test_metadata_path.exists():
            raise FileNotFoundError(f"Test metadata not found at {self.test_metadata_path}. Run create_test_split.py first.")
            
        test_df = pd.read_csv(self.test_metadata_path)
        if self.debug:
            test_df = test_df.head(2)
            logger.info("DEBUG MODE: Only processing 2 samples.")
            
        results = []
        predictions = []
        references = []
        
        for idx, row in test_df.iterrows():
            img_filename = row["new_filename"]
            img_path = self.images_dir / img_filename
            category = row["Category"]
            
            gt_text = f"Findings: {row['findings']} Impression: {row['impression']}"
            
            logger.info(f"Processing sample {idx+1}/{len(test_df)}: {img_filename}")
            
            try:
                cases = self.system.retrieve_similar_cases(img_path, k=3)
                sys_result = self.system.generate_report(img_path, cases)
                gen_text = sys_result["generated_report"]
                
                predictions.append(gen_text)
                references.append(gt_text)
                
                results.append({
                    "uid": row["uid"],
                    "filename": img_filename,
                    "category": category,
                    "ground_truth": gt_text,
                    "generated_text": gen_text,
                    "gt_length": len(gt_text.split()),
                    "gen_length": len(gen_text.split())
                })
            except Exception as e:
                logger.error(f"Failed processing {img_filename}: {e}")
                
        if not results:
            logger.error("No results generated. Exiting.")
            return
            
        # Calculate NLP Metrics
        logger.info("Calculating ROUGE scores...")
        rouge1_f, rouge2_f, rougeL_f = [], [], []
        for p, r in zip(predictions, references):
            scores = self.rouge_scorer.score(r, p)
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougeL_f.append(scores['rougeL'].fmeasure)
        
        logger.info("Calculating BLEU scores...")
        bleu_results = []
        for p, r in zip(predictions, references):
            ref_tokens = r.split()
            pred_tokens = p.split()
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.bleu_smoother)
            bleu_results.append(bleu)
        
        logger.info(f"Calculating BERTScore using {self.bertscore_model}...")
        
        # Monkey-patch AutoTokenizer to prevent Windows OverflowError on SciBERT
        import transformers
        _orig_from_pretrained = transformers.AutoTokenizer.from_pretrained
        def _patched_from_pretrained(*args, **kwargs):
            kwargs['model_max_length'] = 512
            kwargs['use_fast'] = False
            return _orig_from_pretrained(*args, **kwargs)
        transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained
        
        P, R, F1 = bert_score_calc(
            predictions, 
            references, 
            model_type=self.bertscore_model, 
            lang="en", 
            verbose=False
        )
        
        transformers.AutoTokenizer.from_pretrained = _orig_from_pretrained
        bert_p = P.tolist()
        bert_r = R.tolist()
        bert_f1 = F1.tolist()
        
        length_ratios = self._calculate_length_ratio(predictions, references)
        
        # Merge metrics back into results dataframe
        for i, r in enumerate(results):
            r["rouge1"] = rouge1_f[i]
            r["rouge2"] = rouge2_f[i]
            r["rougeL"] = rougeL_f[i]
            r["bleu"] = bleu_results[i]
            r["bertscore_precision"] = bert_p[i]
            r["bertscore_recall"] = bert_r[i]
            r["bertscore_f1"] = bert_f1[i]
            r["length_ratio"] = length_ratios[i]
            
        results_df = pd.DataFrame(results)
        
        # Save Sample-level Data
        results_csv_path = self.data_dir / "results.csv"
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Sample-level results saved to {results_csv_path}")
        
        # Save Summary Statistics
        numeric_cols = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1", "length_ratio", "gen_length"]
        summary_stats = {}
        for col in numeric_cols:
            if col in results_df.columns:
                summary_stats[col] = {
                    "mean": float(results_df[col].mean()),
                    "median": float(results_df[col].median()),
                    "std": float(results_df[col].std())
                }
                
        summary_json_path = self.data_dir / "summary_statistics.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary_stats, f, indent=4)
        logger.info(f"Summary statistics saved to {summary_json_path}")
        
        # Generate Visualizations
        self._generate_plots(results_df)
        
        # Save Metadata
        end_time = time.time()
        self._generate_metadata(end_time - start_time, len(results_df))
        
        logger.info(f"Evaluation complete. All files saved to {self.run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIN-RAG Quantitative Evaluation")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (only 2 samples)")
    args = parser.parse_args()
    
    evaluator = ClinicalEvaluator(debug=args.debug)
    evaluator.run_evaluation()
