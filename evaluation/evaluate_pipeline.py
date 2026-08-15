import os
import time
import json
import argparse
import logging
import re
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

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import ClinicalRAGSystem, PROJECT_ROOT

class ClinicalEvaluator:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.system = ClinicalRAGSystem()
        self.test_metadata_path = PROJECT_ROOT / "data" / "test_samples" / "test_metadata.csv"
        self.images_dir = PROJECT_ROOT / "data" / "test_samples"
        
        # Load clinical entities dynamically
        entities_path = PROJECT_ROOT / "data" / "archive" / "indiana_reports_clinical_entities.json"
        if not entities_path.exists():
            logger.error(f"Clinical entities JSON not found at {entities_path}")
            raise FileNotFoundError("Please run `python extract_clinical_entities.py` first to generate the entities file.")
            
        with open(entities_path, "r") as f:
            self.clinical_entities = json.load(f)
        logger.info(f"Loaded {len(self.clinical_entities)} clinical entities for evaluation.")
        
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

    def _parse_report_sections(self, text: str) -> Dict[str, str]:
        """Extracts reasoning, findings, and impression from the generated text using regex."""
        reasoning = ""
        findings = ""
        impression = ""
        
        reasoning_match = re.search(r"<clinical_reasoning>(.*?)</clinical_reasoning>", text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            
        findings_match = re.search(r"(?:###\s*)?FINDINGS:\s*(.*?)\s*(?:(?:###\s*)?IMPRESSION:|$)", text, re.DOTALL | re.IGNORECASE)
        if findings_match:
            findings = findings_match.group(1).strip()
            # Clean up trailing </findings> tag if the model accidentally included it
            findings = re.sub(r"</findings>$", "", findings, flags=re.IGNORECASE).strip()
            
        impression_match = re.search(r"(?:###\s*)?IMPRESSION:\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
        if impression_match:
            impression = impression_match.group(1).strip()
            
        return {
            "reasoning": reasoning,
            "findings": findings,
            "impression": impression
        }

    def _calculate_length_ratio(self, preds: List[str], refs: List[str]) -> List[float]:
        ratios = []
        for p, r in zip(preds, refs):
            p_len = len(p.split())
            r_len = len(r.split())
            # Prevent division by zero
            r_len = max(r_len, 1)
            ratios.append(p_len / r_len)
        return ratios
        
    def _calculate_clinical_recall(self, gt_text: str, gen_text: str) -> float:
        gt_lower = gt_text.lower()
        gen_lower = gen_text.lower()
        
        gt_entities = [ent for ent in self.clinical_entities if ent in gt_lower]
        if not gt_entities:
            return np.nan
            
        matched_entities = [ent for ent in gt_entities if ent in gen_lower]
        return len(matched_entities) / len(gt_entities)
        
    def _generate_plots(self, df: pd.DataFrame):
        logger.info("Generating statistical visualizations...")
        sns.set_theme(style="whitegrid")
        
        # 1. Boxplots for ROUGE, BLEU, and BERTScore (Baseline vs RAG)
        plt.figure(figsize=(12, 6))
        # We need to reshape the dataframe to have 'Metric', 'Score', and 'Model'
        metrics_bases = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1", "clinical_recall"]
        
        plot_data = []
        for index, row in df.iterrows():
            for metric in metrics_bases:
                if f"baseline_{metric}" in row:
                    plot_data.append({"Metric": metric, "Score": row[f"baseline_{metric}"], "Model": "Baseline"})
                if f"rag_{metric}" in row:
                    plot_data.append({"Metric": metric, "Score": row[f"rag_{metric}"], "Model": "RAG"})
                    
        melted_df = pd.DataFrame(plot_data)
        if not melted_df.empty:
            sns.boxplot(x="Metric", y="Score", hue="Model", data=melted_df, palette="Set2", order=metrics_bases)
            plt.title("Distribution of NLP Evaluation Metrics (Baseline vs RAG)")
            plt.ylim(-0.05, 1.05)
            plt.legend(title="Model")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "metrics_boxplot.png", dpi=300)
        plt.close()
        
        # 2. Scatter Plot: Ground Truth Length vs Generated Length
        plt.figure(figsize=(8, 6))
        if "baseline_gen_length" in df.columns and "rag_gen_length" in df.columns:
            sns.scatterplot(x="gt_length", y="baseline_gen_length", data=df, label="Baseline", alpha=0.7)
            sns.scatterplot(x="gt_length", y="rag_gen_length", data=df, label="RAG", alpha=0.7)
        # Add y=x line
        max_len = max(df["gt_length"].max(), df.get("baseline_gen_length", df["gt_length"]).max(), df.get("rag_gen_length", df["gt_length"]).max()) + 10
        plt.plot([0, max_len], [0, max_len], 'r--', label='Ideal 1:1 Ratio')
        plt.title("Report Verbosity: Generated vs Ground Truth Length")
        plt.xlabel("Ground Truth Length (Words)")
        plt.ylabel("Generated Length (Words)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "length_scatter.png", dpi=300)
        plt.close()
        
        # 3. Radar Chart (Overlapping Polygons)
        try:
            from math import pi
            categories = metrics_bases
            N = len(categories)
            
            baseline_means = []
            rag_means = []
            
            for cat in categories:
                if f"baseline_{cat}" in df.columns:
                    val = df[f"baseline_{cat}"].mean()
                    baseline_means.append(0 if pd.isna(val) else val)
                else:
                    baseline_means.append(0)
                if f"rag_{cat}" in df.columns:
                    val = df[f"rag_{cat}"].mean()
                    rag_means.append(0 if pd.isna(val) else val)
                else:
                    rag_means.append(0)
            
            baseline_values = baseline_means + baseline_means[:1]
            rag_values = rag_means + rag_means[:1]
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)
            
            ax.plot(angles, baseline_values, linewidth=2, linestyle='solid', label='Baseline')
            ax.fill(angles, baseline_values, 'blue', alpha=0.1)
            
            ax.plot(angles, rag_values, linewidth=2, linestyle='solid', label='RAG')
            ax.fill(angles, rag_values, 'orange', alpha=0.1)
            
            plt.title("Normalized Mean Scores (Baseline vs RAG)")
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.savefig(self.plots_dir / "radar_chart.png", dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate radar chart: {e}")

    def run_evaluation(self):
        logger.info("Starting Quantitative Evaluation Pipeline (Ablation Study)...")
        start_time = time.time()
        
        if not self.test_metadata_path.exists():
            raise FileNotFoundError(f"Test metadata not found at {self.test_metadata_path}. Run create_test_split.py first.")
            
        test_df = pd.read_csv(self.test_metadata_path)
        if self.debug:
            test_df = test_df.head(10)
            logger.info("DEBUG MODE: Only processing 10 samples.")
            
        results = []
        baseline_preds = []
        rag_preds = []
        references = []
        
        for idx, row in test_df.iterrows():
            img_filename = row["new_filename"]
            img_path = self.images_dir / img_filename
            category = row["Category"]
            
            gt_text = f"Findings: {row['findings']} Impression: {row['impression']}"
            
            logger.info(f"Processing sample {idx+1}/{len(test_df)}: {img_filename}")
            
            try:
                # Retrieve RAG cases
                cases = self.system.retrieve_similar_cases(img_path, k=3)
                
                # Baseline (Zero-Shot) - empty cases list
                logger.info("Generating Baseline report (No RAG)...")
                baseline_result = self.system.generate_report(img_path, [])
                baseline_text = baseline_result["generated_report"]
                
                # RAG Generation
                logger.info("Generating RAG report...")
                rag_result = self.system.generate_report(img_path, cases)
                rag_text = rag_result["generated_report"]
                
                baseline_preds.append(baseline_text)
                rag_preds.append(rag_text)
                references.append(gt_text)
                
                baseline_parsed = self._parse_report_sections(baseline_text)
                rag_parsed = self._parse_report_sections(rag_text)
                
                results.append({
                    "uid": row["uid"],
                    "filename": img_filename,
                    "category": category,
                    "ground_truth": gt_text,
                    "baseline_text": baseline_text,
                    "rag_text": rag_text,
                    "gt_length": len(gt_text.split()),
                    "baseline_gen_length": len(baseline_text.split()),
                    "rag_gen_length": len(rag_text.split()),
                    "baseline_clinical_recall": self._calculate_clinical_recall(gt_text, baseline_text),
                    "rag_clinical_recall": self._calculate_clinical_recall(gt_text, rag_text),
                    "gt_findings": str(row.get('findings', '')),
                    "baseline_findings": baseline_parsed["findings"],
                    "rag_findings": rag_parsed["findings"],
                    "gt_impression": str(row.get('impression', '')),
                    "baseline_impression": baseline_parsed["impression"],
                    "rag_impression": rag_parsed["impression"],
                    "baseline_reasoning": baseline_parsed["reasoning"],
                    "rag_reasoning": rag_parsed["reasoning"],
                })
            except Exception as e:
                logger.error(f"Failed processing {img_filename}: {e}")
                
        if not results:
            logger.error("No results generated. Exiting.")
            return
            
        # Calculate NLP Metrics
        logger.info("Calculating ROUGE scores...")
        for p_list, prefix in [(baseline_preds, "baseline_"), (rag_preds, "rag_")]:
            rouge1_f, rouge2_f, rougeL_f = [], [], []
            for p, r in zip(p_list, references):
                scores = self.rouge_scorer.score(r, p)
                rouge1_f.append(scores['rouge1'].fmeasure)
                rouge2_f.append(scores['rouge2'].fmeasure)
                rougeL_f.append(scores['rougeL'].fmeasure)
            
            for i, r in enumerate(results):
                r[f"{prefix}rouge1"] = rouge1_f[i]
                r[f"{prefix}rouge2"] = rouge2_f[i]
                r[f"{prefix}rougeL"] = rougeL_f[i]
        
        logger.info("Calculating BLEU scores...")
        for p_list, prefix in [(baseline_preds, "baseline_"), (rag_preds, "rag_")]:
            bleu_results = []
            for p, r in zip(p_list, references):
                ref_tokens = r.split()
                pred_tokens = p.split()
                bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.bleu_smoother)
                bleu_results.append(bleu)
                
            for i, r in enumerate(results):
                r[f"{prefix}bleu"] = bleu_results[i]
        
        logger.info(f"Calculating BERTScore using {self.bertscore_model}...")
        
        # Monkey-patch AutoTokenizer to prevent Windows OverflowError on SciBERT
        import transformers
        _orig_from_pretrained = transformers.AutoTokenizer.from_pretrained
        def _patched_from_pretrained(*args, **kwargs):
            kwargs['model_max_length'] = 512
            kwargs['use_fast'] = False
            return _orig_from_pretrained(*args, **kwargs)
        transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained
        
        for p_list, prefix in [(baseline_preds, "baseline_"), (rag_preds, "rag_")]:
            P, R, F1 = bert_score_calc(
                p_list, 
                references, 
                model_type=self.bertscore_model, 
                lang="en", 
                verbose=False
            )
            bert_p = P.tolist()
            bert_r = R.tolist()
            bert_f1 = F1.tolist()
            
            for i, r in enumerate(results):
                r[f"{prefix}bertscore_precision"] = bert_p[i]
                r[f"{prefix}bertscore_recall"] = bert_r[i]
                r[f"{prefix}bertscore_f1"] = bert_f1[i]
        
        transformers.AutoTokenizer.from_pretrained = _orig_from_pretrained
        
        # Length ratios
        baseline_ratios = self._calculate_length_ratio(baseline_preds, references)
        rag_ratios = self._calculate_length_ratio(rag_preds, references)
        for i, r in enumerate(results):
            r["baseline_length_ratio"] = baseline_ratios[i]
            r["rag_length_ratio"] = rag_ratios[i]
            
        results_df = pd.DataFrame(results)
        
        # Save Sample-level Data
        results_csv_path = self.data_dir / "results.csv"
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Sample-level results saved to {results_csv_path}")
        
        # Save Qualitative Comparison Data
        qualitative_cols = [
            "filename", "category", 
            "gt_findings", "baseline_findings", "rag_findings",
            "gt_impression", "baseline_impression", "rag_impression",
            "baseline_reasoning", "rag_reasoning"
        ]
        # Ensure columns exist to avoid KeyError
        available_qual_cols = [col for col in qualitative_cols if col in results_df.columns]
        qualitative_df = results_df[available_qual_cols]
        
        qualitative_csv_path = self.data_dir / "qualitative_comparison.csv"
        qualitative_df.to_csv(qualitative_csv_path, index=False)
        logger.info(f"Qualitative comparison data saved to {qualitative_csv_path}")
        
        # Save Summary Statistics and Generate LaTeX Table
        numeric_cols_bases = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1", "length_ratio", "gen_length", "clinical_recall"]
        
        summary_stats = {}
        latex_data = []
        
        for col in numeric_cols_bases:
            baseline_col = f"baseline_{col}"
            rag_col = f"rag_{col}"
            
            row_data = {"Metric": col}
            
            if baseline_col in results_df.columns:
                mean_val = float(results_df[baseline_col].mean(skipna=True))
                std_val = float(results_df[baseline_col].std(skipna=True))
                summary_stats[baseline_col] = {
                    "mean": mean_val,
                    "median": float(results_df[baseline_col].median(skipna=True)),
                    "std": std_val
                }
                row_data["Baseline Mean"] = mean_val
                row_data["Baseline Std"] = std_val
            
            if rag_col in results_df.columns:
                mean_val = float(results_df[rag_col].mean(skipna=True))
                std_val = float(results_df[rag_col].std(skipna=True))
                summary_stats[rag_col] = {
                    "mean": mean_val,
                    "median": float(results_df[rag_col].median(skipna=True)),
                    "std": std_val
                }
                row_data["RAG Mean"] = mean_val
                row_data["RAG Std"] = std_val
                
            latex_data.append(row_data)
                
        summary_json_path = self.data_dir / "summary_statistics.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary_stats, f, indent=4)
        logger.info(f"Summary statistics saved to {summary_json_path}")
        
        latex_df = pd.DataFrame(latex_data).set_index("Metric")
        latex_table_path = self.data_dir / "summary_table.tex"
        latex_df.to_latex(latex_table_path, float_format="%.3f")
        logger.info(f"LaTeX summary table saved to {latex_table_path}")
        
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
