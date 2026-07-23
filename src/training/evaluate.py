import time
import numpy as np

class RIFMEvaluator:
    def __init__(self):
        pass

    def compute_rouge_l(self, reference: str, candidate: str) -> float:
        # Standard LCS (Longest Common Subsequence) based ROUGE-L mock/fallback
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        m, n = len(ref_tokens), len(cand_tokens)
        if m == 0 or n == 0:
            return 0.0
            
        # Compute LCS length
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        lcs = dp[m][n]
        precision = lcs / n
        recall = lcs / m
        
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

    def compute_skill_metrics(self, ground_truth: list, extracted: list) -> dict:
        gt_set = set(item.lower().strip() for item in ground_truth)
        ext_set = set(item.lower().strip() for item in extracted)
        
        if not gt_set and not ext_set:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not gt_set or not ext_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        true_positives = len(gt_set.intersection(ext_set))
        precision = true_positives / len(ext_set)
        recall = true_positives / len(gt_set)
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    def compute_ats_mae(self, actual_scores: list, predicted_scores: list) -> float:
        if not actual_scores or len(actual_scores) != len(predicted_scores):
            return -1.0
        errors = [abs(act - pred) for act, pred in zip(actual_scores, predicted_scores)]
        return float(np.mean(errors))

    def profile_latency(self, model_callable, prompt: str) -> dict:
        start_time = time.perf_counter()
        
        # Invoke inference
        response = model_callable(prompt)
        
        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000
        
        # Estimate token count
        token_count = len(response.split()) * 1.3 # rough estimation factor
        tokens_per_sec = token_count / ((end_time - start_time) if (end_time - start_time) > 0 else 1)
        
        return {
            "latency_ms": round(total_latency_ms, 2),
            "estimated_tokens": int(token_count),
            "tokens_per_second": round(tokens_per_sec, 2),
            "response": response
        }

if __name__ == "__main__":
    evaluator = RIFMEvaluator()
    rouge = evaluator.compute_rouge_l("Senior ML Engineer with PyTorch", "Junior ML Engineer using PyTorch")
    skills = evaluator.compute_skill_metrics(["PyTorch", "Python", "Kubernetes"], ["pytorch", "Python", "Docker"])
    print(f"LCS ROUGE-L alignment: {rouge:.4f}")
    print(f"Skill Metrics: {skills}")
