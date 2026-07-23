import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

# --- Pydantic Data Schemas ---

class PersonalInfo(BaseModel):
    name: str = Field(..., description="Candidate's full name")
    email: str = Field(..., description="Email address")
    phone: str = Field(..., description="Phone number")
    links: Optional[dict] = Field(default=None, description="Links to GitHub, LinkedIn, etc.")

class Education(BaseModel):
    institution: str = Field(..., description="University or collage name")
    degree: str = Field(..., description="Degree type and specialization")
    graduation_year: int = Field(..., description="Year of graduation")

class Experience(BaseModel):
    company: str = Field(..., description="Name of the employer")
    title: str = Field(..., description="Job title held")
    start_date: str = Field(..., description="Employment start date")
    end_date: str = Field(..., description="Employment end date or 'Present'")
    bullet_points: List[str] = Field(default=[], description="Bullet points detailing achievements")

class ParsedResume(BaseModel):
    personal_info: PersonalInfo
    education: List[Education]
    experience: List[Experience]
    skills: List[str] = Field(default=[], description="Extracted skills")

class AnalysisReport(BaseModel):
    ats_score: int = Field(..., ge=0, le=100, description="ATS Score (0-100)")
    job_match: float = Field(..., ge=0.0, le=1.0, description="Semantic match factor (0.0 to 1.0)")
    skill_score: int = Field(..., description="Skill alignment score")
    missing_skills: List[str] = Field(default=[], description="Skills requested but not found in resume")
    strengths: List[str] = Field(default=[], description="Key candidate strengths")
    weaknesses: List[str] = Field(default=[], description="Key gaps or weaknesses identified")
    reasoning_summary: str = Field(..., description="Executive reasoning summary")
    confidence_score: Optional[float] = Field(default=None, description="Model evaluation confidence metric")

# --- Verification & Extraction Utilities ---

class SchemaVerifier:
    def __init__(self):
        pass

    def extract_json_string(self, raw_response: str) -> str:
        # 1. Search for JSON markdown block
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
            
        # 2. Extract anything between the first { and the last }
        brace_match = re.search(r'(\{.*})', raw_response, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
            
        return raw_response.strip()

    def verify_parsed_resume(self, raw_response: str) -> ParsedResume:
        json_str = self.extract_json_string(raw_response)
        data = json.loads(json_str)
        return ParsedResume(**data)

    def verify_analysis_report(self, raw_response: str) -> AnalysisReport:
        json_str = self.extract_json_string(raw_response)
        data = json.loads(json_str)
        return AnalysisReport(**data)

if __name__ == "__main__":
    verifier = SchemaVerifier()
    print("SchemaVerifier models and parsing logic loaded.")
