import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.parser.document_parser import DocumentParser
from src.parser.schema_verifier import SchemaVerifier, ParsedResume, AnalysisReport
from src.search.embedder import BGEM3Embedder
from src.search.hybrid_retriever import QdrantHybridRetriever
from src.search.reranker import BGEReranker

app = FastAPI(title="Resume Intelligence Foundation Model API")

# Setup static folders
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Instantiate core components
parser = DocumentParser()
verifier = SchemaVerifier()
embedder = BGEM3Embedder(use_mock=True)
retriever = QdrantHybridRetriever(embedder)
reranker = BGEReranker(use_mock=True)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <head><title>RIFM Service</title></head>
        <body style="font-family: sans-serif; text-align: center; margin-top: 100px;">
            <h1>Resume Intelligence Foundation Model Server</h1>
            <p>Static index.html not found yet. Write index.html to serve dashboard.</p>
        </body>
    </html>
    """

@app.post("/api/parse")
async def parse_resume(file: UploadFile = File(...)):
    # Create a temporary file to save the uploaded binary content
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Extract text content
        raw_text = parser.parse_document(tmp_path)
        
        # In a real environment, we'd invoke the Qwen3-8B model here.
        # We'll mock the extraction of structure to map raw_text to ParsedResume.
        name_match = re.search(r"Name:\s*([^\n]+)", raw_text)
        name = name_match.group(1).strip() if name_match else "Jane Doe"
        
        email_match = re.search(r"Email:\s*([^\n]+)", raw_text)
        email = email_match.group(1).strip() if email_match else "jane.doe@example.com"
        
        phone_match = re.search(r"Phone:\s*([^\n]+)", raw_text)
        phone = phone_match.group(1).strip() if phone_match else "+1-555-0100"
        
        # Look for skills in text
        detected_skills = []
        all_known_skills = ["python", "pytorch", "tensorflow", "kubernetes", "docker", "react", "next.js", "typescript", "sql", "aws", "gcp"]
        for s in all_known_skills:
            if s in raw_text.lower():
                detected_skills.append(s.capitalize() if s != "next.js" else "Next.js")
                
        if not detected_skills:
            detected_skills = ["Python", "PyTorch", "Git"]

        parsed_data = {
            "personal_info": {
                "name": name,
                "email": email,
                "phone": phone,
                "links": {"linkedin": "https://linkedin.com/in/janedoe"}
            },
            "education": [
                {
                    "institution": "University of Waterloo",
                    "degree": "BS in Computer Science",
                    "graduation_year": 2022
                }
            ],
            "experience": [
                {
                    "company": "Tech Labs",
                    "title": "Machine Learning Engineer",
                    "start_date": "2022",
                    "end_date": "Present",
                    "bullet_points": ["Designed transformers using PyTorch.", "Deployed containers with Docker."]
                }
            ],
            "skills": detected_skills
        }
        
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

import re

@app.post("/api/analyze")
async def analyze_match(
    resume_text: str = Form(...),
    jd_title: str = Form(...),
    jd_requirements: str = Form(...)
):
    try:
        # Split requirement string into list
        reqs = [r.strip() for r in jd_requirements.split(",") if r.strip()]
        
        # Basic text searching logic to evaluate alignment
        skills = []
        for r in reqs:
            if r.lower() in resume_text.lower():
                skills.append(r)
                
        missing = [r for r in reqs if r.lower() not in resume_text.lower()]
        
        # Calculate metric ratios
        skill_ratio = len(skills) / len(reqs) if reqs else 0.5
        match_pct = skill_ratio
        
        # Add educational prestige points (Stanford, Waterloo, MIT, Waterloo -> boost)
        prestige_institutions = ["stanford", "mit", "waterloo", "berkeley", "iit", "toronto"]
        prestige_boost = 0.0
        for inst in prestige_institutions:
            if inst in resume_text.lower():
                prestige_boost = 0.10
                break
                
        ats_score = int(min(1.0, match_pct + prestige_boost) * 100)
        
        # Strengths and weaknesses formulation
        strengths = [f"Found direct keyword matching for: {', '.join(skills[:3])}."]
        if prestige_boost > 0:
            strengths.append("High educational pedigree background detected.")
            
        weaknesses = []
        if missing:
            weaknesses.append(f"Gaps identified in following skill areas: {', '.join(missing[:2])}.")
        else:
            weaknesses.append("No major skill gaps identified matching requirements.")
            
        # Core structured analysis response
        report = {
            "ats_score": ats_score,
            "job_match": round(match_pct, 2),
            "skill_score": int(skill_ratio * 100),
            "missing_skills": missing,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "reasoning_summary": f"Matching evaluation complete for role '{jd_title}'. Candidate holds {len(skills)} out of {len(reqs)} key requirements.",
            "confidence_score": 0.94
        }
        
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
