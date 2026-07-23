import os
import unittest
import numpy as np
from src.parser.document_parser import DocumentParser
from src.parser.schema_verifier import SchemaVerifier, ParsedResume, AnalysisReport
from src.search.embedder import BGEM3Embedder
from src.search.hybrid_retriever import QdrantHybridRetriever
from src.search.reranker import BGEReranker
from src.training.train import run_training
from src.training.evaluate import RIFMEvaluator

class TestRIFMPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = DocumentParser()
        cls.verifier = SchemaVerifier()
        cls.embedder = BGEM3Embedder(use_mock=True)
        cls.retriever = QdrantHybridRetriever(cls.embedder)
        cls.reranker = BGEReranker(use_mock=True)
        cls.evaluator = RIFMEvaluator()

    def test_01_parser_and_verifier(self):
        print("\n--- Running Parser and Schema Verifier Tests ---")
        mock_raw_cv = """
        Name: John Doe
        Email: john.doe@email.com
        Phone: 123-456-7890
        GitHub: github.com/johndoe
        
        Education:
        Stanford University, BS in CS, 2021
        
        Experience:
        Google, Software Engineer, 2021-Present
        * Developed scalable neural networks with PyTorch.
        * Maintained deployment containers.
        
        Skills:
        PyTorch, Python, Docker, Kubernetes
        """
        cleaned = self.parser.clean_text(mock_raw_cv)
        self.assertIn("John Doe", cleaned)
        
        # Test schema parsing recovery from raw model outputs
        raw_model_output = """
        Here is the parsed analysis:
        ```json
        {
          "personal_info": {
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "123-456-7890",
            "links": {
              "github": "https://github.com/johndoe"
            }
          },
          "education": [
            {
              "institution": "Stanford University",
              "degree": "BS in CS",
              "graduation_year": 2021
            }
          ],
          "experience": [
            {
              "company": "Google",
              "title": "Software Engineer",
              "start_date": "2021",
              "end_date": "Present",
              "bullet_points": [
                "Developed scalable neural networks with PyTorch."
              ]
            }
          ],
          "skills": ["PyTorch", "Python"]
        }
        ```
        """
        parsed_resume = self.verifier.verify_parsed_resume(raw_model_output)
        self.assertIsInstance(parsed_resume, ParsedResume)
        self.assertEqual(parsed_resume.personal_info.name, "John Doe")
        self.assertEqual(len(parsed_resume.experience), 1)

    def test_02_embeddings_and_hybrid_search(self):
        print("\n--- Running BGE-M3 Retrieval and Reranker Tests ---")
        
        # Index sample resumes
        self.retriever.index_document(
            doc_id="res_01",
            text="Experienced Machine Learning Developer specializing in PyTorch and transformers.",
            metadata={"domain": "ML"}
        )
        self.retriever.index_document(
            doc_id="res_02",
            text="Frontend web designer specializing in React, Next.js and styling sheets.",
            metadata={"domain": "Frontend"}
        )
        
        # Query matching ML
        results = self.retriever.query("Looking for PyTorch ML developer", filters={"domain": "ML"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "res_01")
        
        # Retrieve candidates for global query
        candidates = self.retriever.query("Need experienced engineer", limit=10)
        self.assertEqual(len(candidates), 2)
        
        # Rerank candidates
        reranked = self.reranker.rerank("Machine Learning Neural Nets", candidates)
        self.assertEqual(len(reranked), 2)
        self.assertTrue("rerank_score" in reranked[0])

    def test_03_training_compilation(self):
        print("\n--- Running SFT Training dry-run ---")
        try:
            run_training(dry_run=True)
            success = True
        except Exception as e:
            print(f"Training compile error: {e}")
            success = False
        self.assertTrue(success)

    def test_04_evaluator_metrics(self):
        print("\n--- Running Evaluator Metric Tests ---")
        # 1. ROUGE-L matching
        rouge_score = self.evaluator.compute_rouge_l(
            reference="Experienced PyTorch Deep Learning expert.",
            candidate="Deep Learning expert with PyTorch."
        )
        self.assertGreater(rouge_score, 0.4)
        
        # 2. Skill evaluation
        skills = self.evaluator.compute_skill_metrics(
            ground_truth=["PyTorch", "Python", "Docker"],
            extracted=["pytorch", "Python", "Kubernetes"]
        )
        self.assertEqual(skills["precision"], 0.6667)
        self.assertEqual(skills["recall"], 0.6667)
        self.assertEqual(skills["f1"], 0.6667)
        
        # 3. MAE verification
        mae = self.evaluator.compute_ats_mae(
            actual_scores=[90, 85, 70],
            predicted_scores=[88, 85, 75]
        )
        self.assertAlmostEqual(mae, 7/3)

if __name__ == "__main__":
    unittest.main()
