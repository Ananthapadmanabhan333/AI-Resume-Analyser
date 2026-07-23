import os
import re
import json
import random
import hashlib
from collections import defaultdict

class MinHashLSH:
    def __init__(self, num_perm=128, threshold=0.85, num_bands=16, rows_per_band=8):
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        assert num_bands * rows_per_band == num_perm, "num_bands * rows_per_band must equal num_perm"
        
        # Generate hash function coefficients
        random.seed(42)
        # Using prime numbers for hashing parameters (a * x + b) % c
        self.prime = 4294967311 # A prime number larger than 2^32
        self.a_coeffs = [random.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b_coeffs = [random.randint(0, self.prime - 1) for _ in range(num_perm)]
        
        self.buckets = [defaultdict(list) for _ in range(num_bands)]
        self.signatures = {}

    def get_shingles(self, text, k=5):
        # Normalize text and extract k-character shingles
        text = re.sub(r'\s+', ' ', text.lower().strip())
        shingles = set()
        for i in range(len(text) - k + 1):
            shingles.add(text[i:i+k])
        return shingles

    def compute_signature(self, shingles):
        signature = []
        # Convert shingles to integers using MD5/FNV hash
        shingle_hashes = []
        for s in shingles:
            h = int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)
            shingle_hashes.append(h)
            
        if not shingle_hashes:
            return [self.prime] * self.num_perm

        for i in range(self.num_perm):
            a = self.a_coeffs[i]
            b = self.b_coeffs[i]
            # Compute hash values and find the minimum
            min_val = min((a * h + b) % self.prime for h in shingle_hashes)
            signature.append(min_val)
        return signature

    def insert(self, doc_id, text):
        shingles = self.get_shingles(text)
        signature = self.compute_signature(shingles)
        self.signatures[doc_id] = (signature, text)
        
        # Partition signature into bands and insert into hash buckets
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(signature[start:end])
            
            # Hash the band to bucket it
            band_hash = hash(band_tuple)
            self.buckets[band_idx][band_hash].append(doc_id)

    def query_duplicates(self):
        candidate_pairs = set()
        # Find candidates in the same bucket
        for band_buckets in self.buckets:
            for bucket_id, doc_ids in band_buckets.items():
                if len(doc_ids) > 1:
                    for i in range(len(doc_ids)):
                        for j in range(i + 1, len(doc_ids)):
                            doc_a = doc_ids[i]
                            doc_b = doc_ids[j]
                            candidate_pairs.add(tuple(sorted([doc_a, doc_b])))

        duplicates = []
        # Verify candidate pairs using exact Jaccard Similarity
        for doc_a, doc_b in candidate_pairs:
            sig_a, _ = self.signatures[doc_a]
            sig_b, _ = self.signatures[doc_b]
            
            # Approximate Jaccard via signature matching
            matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
            approx_jaccard = matches / self.num_perm
            
            if approx_jaccard >= self.threshold:
                duplicates.append((doc_a, doc_b, approx_jaccard))
                
        return duplicates

def deduplicate_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    lsh = MinHashLSH(num_perm=128, threshold=0.85)
    
    docs = {}
    print(f"Reading files from {input_dir}...")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.json'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    docs[path] = content
                    lsh.insert(path, content)

    if not docs:
        print("No documents found to deduplicate.")
        return

    duplicates = lsh.query_duplicates()
    print(f"Found {len(duplicates)} duplicate candidate pairs.")

    # Select duplicates to remove (keep the longer file)
    to_remove = set()
    for doc_a, doc_b, score in duplicates:
        if len(docs[doc_a]) >= len(docs[doc_b]):
            to_remove.add(doc_b)
        else:
            to_remove.add(doc_a)

    print(f"Removing {len(to_remove)} redundant documents. Writing remaining to {output_dir}...")
    for path, content in docs.items():
        if path not in to_remove:
            filename = os.path.basename(path)
            out_path = os.path.join(output_dir, filename)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
    print("Deduplication phase complete.")

if __name__ == "__main__":
    import sys
    # For testing, execute with mock folders
    if len(sys.argv) > 2:
        deduplicate_dataset(sys.argv[1], sys.argv[2])
    else:
        # Default mock paths inside workspace for checking
        deduplicate_dataset("./data/raw", "./data/processed")
