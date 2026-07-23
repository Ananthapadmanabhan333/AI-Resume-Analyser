import os
import json
import random

# Core definitions for synthetic profiles
CAREERS = {
    "Machine Learning Engineer": {
        "skills": ["PyTorch", "Python", "Docker", "vLLM", "CUDA optimization", "TensorRT", "Kubernetes", "Transformers", "SQL"],
        "companies": ["Google DeepMind", "Meta AI", "OpenAI", "Anthropic", "Cohere", "Hugging Face"],
        "degrees": ["MS in Computer Science", "Ph.D. in Machine Learning", "B.Tech in Artificial Intelligence"],
        "strengths": [
            "Strong background in transformer optimization and serving engines.",
            "Experienced in building scalable training infrastructure.",
            "Demonstrated ability to implement custom CUDA kernels."
        ],
        "weaknesses": [
            "Lacks extensive front-end dashboard development experience.",
            "Limited exposure to legacy relational database migrations."
        ]
    },
    "Frontend Software Engineer": {
        "skills": ["React", "TypeScript", "Next.js", "TailwindCSS", "Webpack", "CSS3", "HTML5", "GraphQL", "Jest"],
        "companies": ["Vercel", "Stripe", "Airbnb", "Netflix", "Shopify", "Figma"],
        "degrees": ["BS in Software Engineering", "B.Sc. in Computer Science", "Interactive Media Design Degree"],
        "strengths": [
            "Expert in responsive design, accessibility standards (WCAG), and Next.js routers.",
            "Strong micro-animation skill set with CSS and Framer Motion.",
            "Excellent team collaboration and cross-functional communication."
        ],
        "weaknesses": [
            "Lacks deep systems-level database modeling experience.",
            "Minimal familiarity with backend language runtimes like Rust or Go."
        ]
    },
    "Technical Product Manager": {
        "skills": ["Product Roadmap", "Jira", "SQL", "Data Analytics", "Agile", "User Research", "A/B Testing", "System Design"],
        "companies": ["Atlassian", "Slack", "Microsoft", "Amazon", "Uber", "Salesforce"],
        "degrees": ["BS in Computer Science + MBA", "BS in Business Analytics", "Master of Engineering Management"],
        "strengths": [
            "Excellent capability in translating complex engineering concepts to business strategies.",
            "Data-driven mindset with deep SQL analytics capability.",
            "Proven record of launching products from zero to one."
        ],
        "weaknesses": [
            "Not actively coding day-to-day, which may slow down deep architecture audits.",
            "Limited experience working with hardware-in-the-loop environments."
        ]
    }
}

UNIVERSITIES = ["Stanford University", "MIT", "UC Berkeley", "Carnegie Mellon University", "IIT Bombay", "University of Toronto"]
NAMES = ["Alex Chen", "Sarah Jenkins", "Michael Zhao", "Elena Rostova", "Vikram Patel", "Emily Dubois"]

def generate_sample(sample_id):
    # Choose a career path
    role_name = random.choice(list(CAREERS.keys()))
    career = CAREERS[role_name]
    
    # Assemble Candidate details
    name = random.choice(NAMES) + f" {random.randint(10,99)}"
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(career["degrees"])
    grad_year = random.randint(2015, 2025)
    
    # Partition skills
    cand_skills = random.sample(career["skills"], k=random.randint(5, len(career["skills"])))
    missing = [s for s in career["skills"] if s not in cand_skills]
    
    # Assemble Experiences
    exp_years = random.randint(2, 8)
    company_1 = random.choice(career["companies"])
    company_2 = random.choice([c for c in career["companies"] if c != company_1])
    
    experience = [
        {
            "company": company_1,
            "title": f"Senior {role_name}" if exp_years > 5 else role_name,
            "start_date": f"{2026 - exp_years}-01",
            "end_date": "Present",
            "bullet_points": [
                f"Led engineering initiatives specializing in {cand_skills[0]} and {cand_skills[1]}.",
                f"Collaborated with product teams to build highly performant services."
            ]
        },
        {
            "company": company_2,
            "title": role_name,
            "start_date": f"{2026 - exp_years - 2}-01",
            "end_date": f"{2026 - exp_years}-01",
            "bullet_points": [
                f"Implemented core application infrastructure using {cand_skills[-1]}.",
                "Optimized build times and runtime latency by 20%."
            ]
        }
    ]
    
    # Formulate Job Description
    jd_reqs = random.sample(career["skills"], k=random.randint(4, 7))
    jd_missing = [req for req in jd_reqs if req not in cand_skills]
    
    jd = {
        "title": f"Senior {role_name}" if exp_years > 4 else role_name,
        "requirements": jd_reqs
    }
    
    resume = {
        "personal_info": {
            "name": name,
            "email": f"{name.lower().replace(' ', '')}@example.com",
            "phone": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "links": {
                "github": f"https://github.com/{name.lower().replace(' ', '')}",
                "linkedin": f"https://linkedin.com/in/{name.lower().replace(' ', '')}"
            }
        },
        "education": [
            {
                "institution": uni,
                "degree": degree,
                "graduation_year": grad_year
            }
        ],
        "experience": experience,
        "skills": cand_skills
    }
    
    # Generate Target Analysis matching our JSON schema
    match_pct = max(0.2, 1.0 - (len(jd_missing) / len(jd_reqs)))
    ats_score = int(match_pct * 100)
    
    # Formulate Reasoning summary
    reasoning = [
        f"1. Candidate has a strong educational pedigree from {uni}."
    ]
    if jd_missing:
        reasoning.append(f"2. Core skill gaps identified: {', '.join(jd_missing)}.")
    else:
        reasoning.append("2. Candidate holds all requested skills.")
    reasoning.append(f"3. Matching score calculated as {ats_score} based on functional metrics.")
    
    reasoning_block = f"<reasoning>\n" + "\n".join(reasoning) + "\n</reasoning>"
    
    analysis_report = {
        "ats_score": ats_score,
        "job_match": round(match_pct, 2),
        "skill_score": int((1.0 - (len(jd_missing) / len(jd_reqs))) * 100),
        "missing_skills": jd_missing,
        "strengths": random.sample(career["strengths"], k=min(2, len(career["strengths"]))),
        "weaknesses": random.sample(career["weaknesses"], k=min(1, len(career["weaknesses"]))),
        "reasoning_summary": f"Candidate is highly skilled with a solid background from {company_1} and {company_2}.",
        "confidence_score": round(random.uniform(0.85, 0.98), 2)
    }
    
    # Build Instruction sample structure
    instruction_tuning_pair = {
        "messages": [
            {
                "role": "system",
                "content": "You are the Resume Intelligence Foundation Model. You must output analysis in structured JSON matching the provided schema."
            },
            {
                "role": "user",
                "content": f"### JOB DESCRIPTION:\n{json.dumps(jd, indent=2)}\n\n### CANDIDATE RESUME:\n{json.dumps(resume, indent=2)}"
            },
            {
                "role": "assistant",
                "content": f"{reasoning_block}\n{json.dumps(analysis_report, indent=2)}"
            }
        ]
    }
    
    return instruction_tuning_pair

def build_synthetic_corpus(dest_file, count=50):
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    with open(dest_file, 'w', encoding='utf-8') as f:
        for idx in range(count):
            sample = generate_sample(idx)
            f.write(json.dumps(sample) + "\n")
    print(f"Generated {count} synthetic instruction pairs in {dest_file}")

if __name__ == "__main__":
    build_synthetic_corpus("./data/processed/synthetic_sft.jsonl", count=100)
