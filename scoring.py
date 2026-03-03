import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.cluster import AgglomerativeClustering
import difflib

class ParticipantVisibleError(Exception):
    pass

def _string_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def load_sdrf(sdrf_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    if "PXD" not in sdrf_df.columns:
        raise ParticipantVisibleError("Both solution and submission must include a 'PXD' column.")
    sdrf_dict: Dict[str, Dict[str, List[str]]] = {}
    for pxd, pxd_df in sdrf_df.groupby('PXD'):
        sdrf_dict[pxd] = {}
        for col in pxd_df.columns:
            uniq = pd.Series(pxd_df[col]).dropna().astype(str).unique().tolist()
            if uniq == ['Not Applicable']:
                continue
            values: List[str] = []
            for v in uniq:
                if 'NT=' in v:
                    parts = [r for r in v.split(';') if 'NT=' in r]
                    values.append(parts[0].replace('NT=', '').strip() if parts else v.strip())
                else:
                    values.append(v.strip())
            sdrf_dict[pxd][col] = values
    return sdrf_dict

def Harmonize_and_Evaluate_datasets(A, B, threshold=0.80, CompleteAbsence=float('nan')):
    from sklearn.metrics import precision_score, recall_score, f1_score
    eval_metrics = {'pxd': [], 'AnnotationType': [], 'precision': [], 'recall': [], 'f1': [], 'jacc': []}
    harmonized_A = {}
    harmonized_B = {}
    common_pubs = set(A) & set(B)
    for pub in common_pubs:
        harmonized_A[pub], harmonized_B[pub] = {}, {}
        for category in (set(A[pub]) & set(B[pub])):
            vals_A = list(dict.fromkeys(A[pub][category]))
            vals_B = list(dict.fromkeys(B[pub][category]))
            all_vals = vals_A + [v for v in vals_B if v not in vals_A]
            if len(vals_A) == 0 and len(vals_B) == 0:
                harmA, harmB = [], []
            elif len(all_vals) == 1:
                labels = np.array([0])
                str2cid = {all_vals[0]: 0}
                harmA = [str2cid[s] for s in vals_A]
                harmB = [str2cid[s] for s in vals_B]
            else:
                N = len(all_vals)
                dist = np.zeros((N, N), dtype=float)
                for i in range(N):
                    for j in range(i + 1, N):
                        sim = _string_similarity(all_vals[i], all_vals[j])
                        d = 1.0 - sim
                        dist[i, j] = d
                        dist[j, i] = d
                clusterer = AgglomerativeClustering(
                    n_clusters=None, metric='precomputed',
                    linkage='average', distance_threshold=1.0 - threshold
                )
                labels = clusterer.fit_predict(dist)
                str2cid = {s: int(labels[i]) for i, s in enumerate(all_vals)}
                harmA = [str2cid[s] for s in vals_A]
                harmB = [str2cid[s] for s in vals_B]
            harmonized_A[pub][category] = harmA
            harmonized_B[pub][category] = harmB
            uniq = sorted(set(harmA) | set(harmB))
            if not uniq:
                p = r = f = CompleteAbsence
                j = 1.0
            else:
                y_true = [1 if u in harmA else 0 for u in uniq]
                y_pred = [1 if u in harmB else 0 for u in uniq]
                p = precision_score(y_true, y_pred, average='macro', zero_division=0)
                r = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f = f1_score(y_true, y_pred, average='macro', zero_division=0)
                setA, setB = set(harmA), set(harmB)
                j = 1.0 if (not setA and not setB) else len(setA & setB) / len(setA | setB)
            eval_metrics['pxd'].append(pub)
            eval_metrics['AnnotationType'].append(category)
            eval_metrics['precision'].append(p)
            eval_metrics['recall'].append(r)
            eval_metrics['f1'].append(f)
            eval_metrics['jacc'].append(j)
    return harmonized_A, harmonized_B, pd.DataFrame(eval_metrics)

def score(solution, submission, row_id_column_name):
    if row_id_column_name and row_id_column_name in solution.columns:
        solution = solution.drop(columns=[row_id_column_name])
    if row_id_column_name and row_id_column_name in submission.columns:
        submission = submission.drop(columns=[row_id_column_name])
    sol = load_sdrf(solution)
    sub = load_sdrf(submission)
    _, _, eval_df = Harmonize_and_Evaluate_datasets(sol, sub, threshold=0.80)
    vals = eval_df["f1"].dropna()
    return eval_df, float(vals.mean()) if not vals.empty else 0.0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", default="./eval_output.csv")
    args = parser.parse_args()
    solution_df = pd.read_csv(args.solution)
    submission_df = pd.read_csv(args.submission)
    eval_df, final_score = score(solution_df, submission_df, 'ID')
    print(eval_df.to_string())
    print(f"\nFinal SDRF Average F1 Score: {final_score:.6f}")
    eval_df.to_csv(args.output, index=False)
