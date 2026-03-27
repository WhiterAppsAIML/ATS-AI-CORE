#!/usr/bin/env python3
"""
sample_test_ats_model.py

Quick script to test the ATS model with inline text.
You can modify this script to test different resumes and job descriptions.

Usage:
    python sample_test_ats_model.py
"""

import sys
from pathlib import Path

# Add ats-ai-core to path (required for imports)
_PROJECT_ROOT = Path(__file__).resolve().parent
_AI_CORE_ROOT = _PROJECT_ROOT / "ats-ai-core"
sys.path.insert(0, str(_AI_CORE_ROOT))

from src.ats_engine.inference import run_ats_inference
import json

# ============================================================================
# SAMPLE TEST DATA
# ============================================================================

# Sample Resume
SAMPLE_RESUME = """
JOHN DOE
Email: john.doe@example.com | Phone: (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Senior Software Engineer with 6 years of experience developing scalable backend systems
and cloud-native applications. Proven expertise in Python, Docker, Kubernetes, and AWS.

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | New York, NY | 2022 - Present
- Architected microservices platform using Python and Docker, serving 100K+ users
- Led Kubernetes deployment strategy, reducing infrastructure costs by 40%
- Mentored team of 4 junior engineers on best practices
- Implemented CI/CD pipelines with GitHub Actions and Jenkins

Software Engineer | CloudStart Technologies | Boston, MA | 2020 - 2022
- Developed RESTful APIs using Python Flask serving 50M requests/month
- Optimized database queries, improving response time by 60%
- Built monitoring system using Prometheus and Grafana
- Collaborated with product team on feature planning and delivery

Junior Software Engineer | StartupXYZ | Remote | 2018 - 2020
- Developed web applications using Python and JavaScript
- Participated in code reviews and agile ceremonies
- Fixed 200+ production bugs and improved code quality

EDUCATION
Bachelor of Science in Computer Science
University of Technology, Graduation: 2018

SKILLS
Technical: Python, Java, JavaScript, SQL, Docker, Kubernetes, AWS (EC2, S3, RDS)
Frameworks: Flask, Django, FastAPI, React
Databases: PostgreSQL, MongoDB, Redis
Tools: Git, GitHub, Jenkins, Docker, Kubernetes, AWS, Terraform
Soft Skills: Leadership, Communication, Problem-solving, Agile methodology

CERTIFICATIONS
AWS Solutions Architect Associate Certification (2023)
"""

# Sample Job Description
SAMPLE_JD = """
JOB TITLE: Senior Full Stack Engineer

COMPANY: InnovateTech Solutions

LOCATION: San Francisco, CA (Remote)

ABOUT THE ROLE
We are seeking a Senior Full Stack Engineer to join our growing engineering team.
You will be responsible for developing scalable backend services and modern frontend
applications for our SaaS platform serving 500K+ customers.

KEY RESPONSIBILITIES
- Design and implement microservices architecture using Python and Node.js
- Deploy and manage Kubernetes clusters in AWS
- Develop RESTful APIs and GraphQL endpoints
- Collaborate with product and design teams on feature development
- Mentor junior engineers and conduct code reviews
- Participate in oncall rotation and incident response

REQUIRED QUALIFICATIONS
- 5+ years of software engineering experience
- Strong proficiency in Python and/or Java
- Experience with Docker and Kubernetes
- AWS cloud platform experience
- SQL and NoSQL database expertise
- Strong problem-solving and communication skills
- Bachelor's degree in Computer Science or related field

PREFERRED QUALIFICATIONS
- Leadership or mentoring experience
- CI/CD pipeline expertise
- Systems design and architecture knowledge
- Experience with microservices architecture
- Knowledge of observability tools (Prometheus, ELK, etc.)
- AWS certifications

WHAT WE OFFER
- Competitive salary ($180K-$220K based on experience)
- Comprehensive health insurance
- 401(k) matching
- Unlimited PTO
- Remote work flexibility
- Professional development budget
- Equity options
"""

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_model():
    """Run ATS model on sample data and print formatted results."""

    print("\n" + "=" * 70)
    print("  ATS MODEL - SAMPLE TEST")
    print("=" * 70)
    print("\n🔍 Running inference on sample resume and job description...\n")

    # Run inference
    result = run_ats_inference(SAMPLE_RESUME, SAMPLE_JD)

    # Print results
    print("\n" + "-" * 70)
    print("  RESULTS SUMMARY")
    print("-" * 70)

    print(f"\n📊 ATS Score:          {result['ats_score']:.1f} / 100")
    print(f"📈 Score Band:         {result['score_band']}")
    print(f"🏢 Predicted Domain:   {result['domain_name']} (Index: {result['domain_index']})")
    print(f"👤 Fresher:            {'Yes' if result['is_fresher'] else 'No'}")

    # Print missing keywords
    print("\n" + "-" * 70)
    print("  MISSING KEYWORDS & SKILLS")
    print("-" * 70)

    hard_skills = result.get("missing_keywords", {}).get("hard_skills", [])
    soft_skills = result.get("missing_keywords", {}).get("soft_skills", [])

    if hard_skills:
        print("\n❌ Hard Skills Missing (Top 5):")
        for i, skill in enumerate(hard_skills[:5], 1):
            skill_name = skill if isinstance(skill, str) else skill.get('keyword', str(skill))
            print(f"   {i}. {skill_name}")
    else:
        print("\n✅ No significant hard skills missing!")

    if soft_skills:
        print("\n❌ Soft Skills Missing:")
        for i, skill in enumerate(soft_skills[:5], 1):
            skill_name = skill if isinstance(skill, str) else skill.get('keyword', str(skill))
            print(f"   {i}. {skill_name}")
    else:
        print("\n✅ No significant soft skills missing!")

    # Print feedback
    feedback = result.get("feedback", [])
    if feedback:
        print("\n" + "-" * 70)
        print("  RECOMMENDATIONS & FEEDBACK")
        print("-" * 70)
        for i, item in enumerate(feedback, 1):
            print(f"\n💡 [{i}] {item}")
    else:
        print("\n✅ No feedback recommendations at this time.")

    # Print raw JSON output
    print("\n" + "-" * 70)
    print("  FULL JSON RESPONSE")
    print("-" * 70)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70 + "\n")


# ============================================================================
# CUSTOM TEST FUNCTION
# ============================================================================

def test_custom(resume_text: str, jd_text: str, test_name: str = "Custom"):
    """Test model with custom resume and job description.

    Args:
        resume_text: Resume content as string
        jd_text: Job description content as string
        test_name: Name for this test (for output)
    """

    print(f"\n🧪 Testing: {test_name}\n")
    result = run_ats_inference(resume_text, jd_text)

    print(f"Score: {result['ats_score']:.1f} | "
          f"Band: {result['score_band']} | "
          f"Domain: {result['domain_name']}")

    return result


# ============================================================================
# BATCH TEST FUNCTION
# ============================================================================

def test_batch(test_cases: list[dict]):
    """Run multiple test cases and compare results.

    Args:
        test_cases: List of dicts with keys 'name', 'resume', 'jd'

    Returns:
        List of results
    """

    results = []

    print("\n" + "=" * 70)
    print("  BATCH TEST RESULTS")
    print("=" * 70 + "\n")

    for case in test_cases:
        name = case.get('name', 'Unknown')
        resume = case.get('resume', '')
        jd = case.get('jd', '')

        if not resume or not jd:
            print(f"⚠️  Skipping {name}: Missing resume or JD")
            continue

        result = run_ats_inference(resume, jd)
        results.append({
            'name': name,
            'score': result['ats_score'],
            'band': result['score_band'],
            'domain': result['domain_name']
        })

        print(f"✓ {name:30} | Score: {result['ats_score']:6.1f} | "
              f"Band: {result['score_band']:15} | Domain: {result['domain_name']}")

    print("\n" + "=" * 70 + "\n")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Test 1: Run on sample data
    test_model()

    # Test 2: Custom test (uncomment to use)
    # my_resume = "Your resume text here..."
    # my_jd = "Your JD text here..."
    # test_custom(my_resume, my_jd, "My Custom Test")

    # Test 3: Batch test (uncomment to use)
    # batch_cases = [
    #     {
    #         'name': 'Senior Python Dev',
    #         'resume': 'Senior Python dev with 10 years...',
    #         'jd': 'We seek Senior Python engineer...'
    #     },
    #     {
    #         'name': 'Junior Frontend Dev',
    #         'resume': 'Fresh grad with React skills...',
    #         'jd': 'Junior React developer needed...'
    #     }
    # ]
    # test_batch(batch_cases)
