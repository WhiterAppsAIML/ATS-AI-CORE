"""
generate_synthetic.py — Generates synthetic resume-JD pairs for Legal and Education domains.

Addresses Domain F1 gap: Legal (F1=0.68, 228 samples) and Education (F1=0.62, 331 samples).
Generates 600 Legal pairs and 500 Education pairs with controlled score distributions
and fresher representation.

Usage:
    python generate_synthetic.py

Outputs:
    data/synthetic/legal_synthetic.csv      (600 pairs, domain_label=5)
    data/synthetic/education_synthetic.csv   (500 pairs, domain_label=6)
"""

import csv
import random
import os
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).parent.resolve()
SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"

# ── Score distribution tables ────────────────────────────────────────────────

LEGAL_SCORE_DIST = [
    (85, 100, 90),   # Excellent: 15% of 600 = 90
    (65, 84, 150),   # Good:      25% of 600 = 150
    (45, 64, 180),   # Moderate:  30% of 600 = 180
    (25, 44, 120),   # Weak:      20% of 600 = 120
    (0, 24, 60),     # Poor:      10% of 600 = 60
]

EDUCATION_SCORE_DIST = [
    (85, 100, 75),   # Excellent: 15% of 500 = 75
    (65, 84, 125),   # Good:      25% of 500 = 125
    (45, 64, 150),   # Moderate:  30% of 500 = 150
    (25, 44, 100),   # Weak:      20% of 500 = 100
    (0, 24, 50),     # Poor:      10% of 500 = 50
]

# ── Name pools (fictional) ──────────────────────────────────────────────────

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karan", "Meera",
    "Arjun", "Divya", "Rahul", "Ishita", "Aditya", "Nisha", "Siddharth", "Pooja",
    "Nikhil", "Kavya", "Amit", "Ritu", "Harsh", "Simran", "Manish", "Neha",
    "Rajesh", "Swati", "Deepak", "Anjali", "Suresh", "Pallavi", "Gaurav", "Tanya",
    "Varun", "Shruti", "Pranav", "Lakshmi", "Akash", "Bhavna", "Tushar", "Sonal",
    "Vivek", "Rina", "Ajay", "Komal", "Naveen", "Sakshi", "Tarun", "Megha",
    "Ashish", "Jyoti", "Kunal", "Preeti", "Mohit", "Rashmi", "Sahil", "Usha",
    "Dev", "Chitra", "Ravi", "Gayatri", "Sameer", "Namrata", "Ankit", "Hema",
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Reddy", "Nair", "Joshi",
    "Mehta", "Shah", "Rao", "Iyer", "Mishra", "Verma", "Chauhan", "Desai",
    "Pillai", "Malhotra", "Bhat", "Agarwal", "Saxena", "Kapoor", "Thakur", "Das",
    "Banerjee", "Mukherjee", "Sen", "Ghosh", "Chatterjee", "Sinha", "Menon", "Kulkarni",
]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune",
    "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Kochi", "Indore", "Nagpur",
    "Bhopal", "Coimbatore", "Noida", "Gurgaon", "Mysore", "Vizag",
]

# ── Legal domain data ────────────────────────────────────────────────────────

LEGAL_ARCHETYPES = {
    "corporate_lawyer": {
        "jd_titles": [
            "Corporate Lawyer", "In-House Counsel", "Corporate Legal Advisor",
            "Senior Corporate Counsel", "Associate Corporate Lawyer",
        ],
        "jd_keywords": [
            "contract drafting", "corporate law", "M&A", "due diligence",
            "regulatory compliance", "LLB", "LLM", "legal research",
            "client advisory", "negotiation", "corporate governance",
            "shareholder agreements", "board resolutions", "joint ventures",
            "commercial contracts", "intellectual property clauses",
        ],
        "jd_responsibilities": [
            "Draft, review, and negotiate commercial contracts and agreements",
            "Provide legal advisory on corporate governance and compliance matters",
            "Conduct due diligence for mergers, acquisitions, and joint ventures",
            "Advise on regulatory compliance including SEBI and Companies Act",
            "Manage corporate legal documentation and board resolutions",
            "Support M&A transactions including structuring and documentation",
            "Review and negotiate vendor agreements and service level agreements",
            "Advise business teams on legal risks and mitigation strategies",
            "Coordinate with external counsel on complex litigation matters",
            "Ensure compliance with applicable laws and regulatory frameworks",
        ],
        "resume_skills": [
            "Contract Drafting", "Corporate Law", "M&A Due Diligence",
            "Legal Research", "Regulatory Compliance", "Negotiation",
            "Corporate Governance", "SEBI Regulations", "Companies Act",
            "Commercial Agreements", "Joint Ventures", "Board Resolutions",
            "Risk Assessment", "Client Advisory", "Legal Documentation",
            "Arbitration", "Dispute Resolution", "Intellectual Property",
        ],
        "resume_education": [
            "LLB from {university}", "LLM in Corporate Law from {university}",
            "BA LLB (Hons) from {university}", "BBA LLB from {university}",
        ],
        "resume_experience_titles": [
            "Corporate Lawyer", "Associate", "In-House Counsel",
            "Legal Advisor", "Senior Associate", "Legal Manager",
        ],
    },
    "litigation_associate": {
        "jd_titles": [
            "Litigation Associate", "Litigation Lawyer", "Civil Litigation Counsel",
            "Senior Litigation Associate", "Trial Attorney",
        ],
        "jd_keywords": [
            "civil litigation", "court filings", "pleadings", "legal briefs",
            "case management", "discovery", "trial preparation", "oral arguments",
            "Bar admission", "criminal litigation", "writ petitions",
            "arbitration", "mediation", "appellate practice",
        ],
        "jd_responsibilities": [
            "Handle civil and commercial litigation matters before various courts",
            "Draft pleadings, written statements, and legal briefs",
            "Conduct legal research and prepare case strategies",
            "Represent clients in court hearings and oral arguments",
            "Manage case files and maintain litigation tracking systems",
            "Prepare witnesses and evidence for trial proceedings",
            "File writ petitions and appeals before High Courts",
            "Coordinate with clients on case status and legal developments",
            "Conduct discovery and document review processes",
            "Attend mediation and arbitration proceedings",
        ],
        "resume_skills": [
            "Civil Litigation", "Criminal Litigation", "Court Filings",
            "Pleadings", "Legal Briefs", "Trial Preparation",
            "Oral Arguments", "Case Management", "Discovery",
            "Legal Research", "Writ Petitions", "Arbitration",
            "Mediation", "Appellate Practice", "Evidence Review",
            "Client Representation", "Dispute Resolution",
        ],
        "resume_education": [
            "LLB from {university}", "BA LLB (Hons) from {university}",
            "LLM in Litigation from {university}",
        ],
        "resume_experience_titles": [
            "Litigation Associate", "Advocate", "Junior Advocate",
            "Litigation Counsel", "Court Practitioner", "Legal Associate",
        ],
    },
    "compliance_officer": {
        "jd_titles": [
            "Legal Compliance Officer", "Compliance Manager", "Regulatory Compliance Specialist",
            "Senior Compliance Officer", "Chief Compliance Officer",
        ],
        "jd_keywords": [
            "regulatory compliance", "risk assessment", "internal audit",
            "policy drafting", "GDPR", "legal frameworks", "compliance training",
            "reporting", "AML", "KYC", "data privacy", "SOX compliance",
            "POSH compliance", "whistleblower policy",
        ],
        "jd_responsibilities": [
            "Develop and implement compliance policies and procedures",
            "Conduct regular risk assessments and internal compliance audits",
            "Ensure organizational compliance with GDPR and data privacy regulations",
            "Draft and update internal policies aligned with regulatory requirements",
            "Deliver compliance training programs to employees across departments",
            "Monitor regulatory changes and assess impact on business operations",
            "Prepare compliance reports for senior management and board",
            "Manage AML/KYC compliance frameworks and reporting",
            "Investigate compliance breaches and recommend corrective actions",
            "Coordinate with external regulators and auditors",
        ],
        "resume_skills": [
            "Regulatory Compliance", "Risk Assessment", "Internal Audit",
            "Policy Drafting", "GDPR", "Data Privacy", "AML/KYC",
            "Compliance Training", "SOX Compliance", "Legal Frameworks",
            "Whistleblower Policy", "POSH Compliance", "Reporting",
            "Risk Mitigation", "Corporate Governance", "Regulatory Affairs",
        ],
        "resume_education": [
            "LLB from {university}", "LLM in Corporate Law from {university}",
            "MBA with specialization in Compliance from {university}",
            "CS (Company Secretary) from ICSI",
        ],
        "resume_experience_titles": [
            "Compliance Officer", "Legal Compliance Analyst", "Regulatory Affairs Manager",
            "Compliance Manager", "Risk & Compliance Associate", "Senior Compliance Officer",
        ],
    },
    "paralegal": {
        "jd_titles": [
            "Paralegal", "Legal Assistant", "Legal Executive",
            "Senior Paralegal", "Litigation Support Specialist",
        ],
        "jd_keywords": [
            "legal documentation", "case files", "client communication",
            "research support", "drafting correspondence", "scheduling hearings",
            "LLB", "paralegal diploma", "legal database management",
            "document review", "filing", "notarization",
        ],
        "jd_responsibilities": [
            "Maintain and organize case files and legal documentation",
            "Assist lawyers with legal research and case preparation",
            "Draft legal correspondence and routine legal documents",
            "Schedule court hearings and manage litigation calendars",
            "Communicate with clients regarding case updates and requirements",
            "Conduct document review and organize discovery materials",
            "Prepare court filing documents and ensure timely submission",
            "Maintain legal databases and case management systems",
            "Assist in preparation of briefs, pleadings, and contracts",
            "Coordinate with courts, opposing counsel, and third parties",
        ],
        "resume_skills": [
            "Legal Documentation", "Case File Management", "Legal Research",
            "Client Communication", "Court Filing", "Document Review",
            "Legal Correspondence", "Scheduling", "Legal Databases",
            "Notarization", "Case Management Systems", "Filing",
            "Research Support", "Drafting", "Administrative Support",
        ],
        "resume_education": [
            "LLB from {university}", "Diploma in Paralegal Studies from {university}",
            "BA LLB from {university}", "Certificate in Legal Assistantship from {university}",
        ],
        "resume_experience_titles": [
            "Paralegal", "Legal Assistant", "Legal Executive",
            "Litigation Support", "Legal Clerk", "Junior Legal Associate",
        ],
    },
    "ip_specialist": {
        "jd_titles": [
            "Intellectual Property Specialist", "IP Attorney", "Patent Attorney",
            "Trademark Specialist", "IP Counsel",
        ],
        "jd_keywords": [
            "patent filing", "trademark registration", "IP litigation",
            "licensing agreements", "IP law", "prior art search", "WIPO",
            "copyright law", "trade secrets", "IP portfolio management",
            "patent prosecution", "design patents",
        ],
        "jd_responsibilities": [
            "File and prosecute patent applications before the Patent Office",
            "Conduct prior art searches and patentability assessments",
            "Handle trademark registration and opposition proceedings",
            "Draft and negotiate IP licensing and technology transfer agreements",
            "Advise clients on IP protection strategies and portfolio management",
            "Manage IP litigation including patent and trademark infringement cases",
            "Ensure compliance with WIPO regulations and international IP treaties",
            "Conduct IP due diligence for M&A transactions",
            "Draft copyright registration applications and enforce copyright claims",
            "Monitor and enforce IP rights against infringement",
        ],
        "resume_skills": [
            "Patent Filing", "Trademark Registration", "IP Litigation",
            "Licensing Agreements", "Prior Art Search", "WIPO",
            "Copyright Law", "Patent Prosecution", "IP Portfolio Management",
            "Trade Secrets", "Design Patents", "IP Due Diligence",
            "Technology Transfer", "IP Strategy", "Infringement Analysis",
        ],
        "resume_education": [
            "LLB from {university}", "LLM in Intellectual Property Law from {university}",
            "B.Tech + LLB from {university}", "Patent Agent Certification",
        ],
        "resume_experience_titles": [
            "IP Specialist", "Patent Attorney", "Trademark Counsel",
            "IP Associate", "IP Analyst", "Senior IP Counsel",
        ],
    },
}

# ── Education domain data ────────────────────────────────────────────────────

EDUCATION_ARCHETYPES = {
    "school_teacher": {
        "jd_titles": [
            "School Teacher (K-12)", "Primary School Teacher", "Secondary School Teacher",
            "High School Teacher", "Subject Teacher", "Senior Teacher",
        ],
        "jd_keywords": [
            "lesson planning", "classroom management", "curriculum development",
            "student assessment", "subject expertise", "B.Ed", "teaching certification",
            "parent communication", "differentiated instruction", "CBSE", "ICSE",
            "activity-based learning", "formative assessment",
        ],
        "jd_responsibilities": [
            "Plan and deliver engaging lessons aligned with CBSE/ICSE curriculum",
            "Manage classroom activities and maintain a positive learning environment",
            "Develop and implement differentiated instruction strategies",
            "Assess student performance through formative and summative evaluations",
            "Communicate regularly with parents regarding student progress",
            "Participate in curriculum development and review processes",
            "Organize co-curricular activities and school events",
            "Maintain student records and prepare progress reports",
            "Attend faculty meetings and professional development workshops",
            "Implement activity-based and experiential learning methodologies",
        ],
        "resume_skills": [
            "Lesson Planning", "Classroom Management", "Curriculum Development",
            "Student Assessment", "Differentiated Instruction", "CBSE Curriculum",
            "ICSE Curriculum", "Activity-Based Learning", "Formative Assessment",
            "Parent Communication", "Co-curricular Activities", "EdTech Tools",
            "Smart Board Usage", "Student Counseling", "Exam Paper Setting",
        ],
        "resume_education": [
            "B.Ed from {university}", "M.Ed from {university}",
            "BA B.Ed from {university}", "BSc B.Ed from {university}",
            "CTET Certified", "State TET Qualified",
        ],
        "resume_experience_titles": [
            "School Teacher", "Primary Teacher", "Subject Teacher",
            "Senior Teacher", "Class Teacher", "TGT", "PGT",
        ],
        "subjects": [
            "Mathematics", "Science", "English", "Hindi", "Social Studies",
            "Physics", "Chemistry", "Biology", "Computer Science", "History",
            "Geography", "Economics", "Political Science", "Sanskrit",
        ],
    },
    "university_lecturer": {
        "jd_titles": [
            "College Lecturer", "University Lecturer", "Assistant Professor",
            "Associate Professor", "Visiting Faculty", "Senior Lecturer",
        ],
        "jd_keywords": [
            "course design", "lecture delivery", "research publications",
            "PhD", "Master's", "academic writing", "student mentoring",
            "syllabus planning", "peer review", "UGC NET", "GATE",
            "journal publications", "conference presentations",
        ],
        "jd_responsibilities": [
            "Design and deliver undergraduate and postgraduate courses",
            "Conduct independent research and publish in peer-reviewed journals",
            "Mentor students on academic projects and dissertations",
            "Develop course syllabi and assessment methodologies",
            "Participate in academic committees and university governance",
            "Present research at national and international conferences",
            "Guide PhD scholars and evaluate research proposals",
            "Collaborate with industry partners on applied research projects",
            "Maintain academic records and submit examination reports",
            "Contribute to accreditation processes (NAAC, NBA)",
        ],
        "resume_skills": [
            "Course Design", "Lecture Delivery", "Research Publications",
            "Academic Writing", "Student Mentoring", "Syllabus Planning",
            "Peer Review", "Conference Presentations", "PhD Supervision",
            "UGC NET Qualified", "NAAC Accreditation", "Curriculum Design",
            "Research Methodology", "Grant Writing", "Academic Administration",
        ],
        "resume_education": [
            "PhD in {subject} from {university}",
            "M.Phil in {subject} from {university}",
            "MA/MSc in {subject} from {university}",
            "UGC NET Qualified", "SET Qualified",
        ],
        "resume_experience_titles": [
            "Assistant Professor", "Lecturer", "Associate Professor",
            "Visiting Faculty", "Research Associate", "Senior Lecturer",
        ],
        "subjects": [
            "Computer Science", "Mathematics", "Physics", "Chemistry",
            "English Literature", "Economics", "Commerce", "Management",
            "Psychology", "Sociology", "Political Science", "History",
            "Biotechnology", "Electronics", "Mechanical Engineering",
        ],
    },
    "edtech_designer": {
        "jd_titles": [
            "Instructional Designer", "EdTech Specialist", "E-Learning Developer",
            "Learning Experience Designer", "Content Developer (EdTech)",
        ],
        "jd_keywords": [
            "e-learning", "LMS platforms", "Moodle", "Canvas", "content authoring tools",
            "Articulate", "Captivate", "instructional design", "ADDIE model",
            "learning objectives", "multimedia content", "SCORM",
            "xAPI", "storyboarding", "Bloom's taxonomy",
        ],
        "jd_responsibilities": [
            "Design and develop e-learning courses using ADDIE instructional design model",
            "Create interactive multimedia content using Articulate Storyline and Captivate",
            "Manage and configure LMS platforms including Moodle and Canvas",
            "Develop learning objectives aligned with Bloom's taxonomy",
            "Create storyboards and course maps for online learning programs",
            "Implement SCORM and xAPI compliant learning modules",
            "Conduct needs analysis and learner assessments",
            "Collaborate with subject matter experts to design course content",
            "Evaluate learning effectiveness through analytics and feedback",
            "Produce video-based and gamified learning content",
        ],
        "resume_skills": [
            "Instructional Design", "ADDIE Model", "E-Learning Development",
            "Articulate Storyline", "Adobe Captivate", "Moodle", "Canvas",
            "SCORM", "xAPI", "Storyboarding", "Bloom's Taxonomy",
            "Multimedia Content", "Video Production", "Gamification",
            "Learning Analytics", "Needs Analysis", "LMS Administration",
        ],
        "resume_education": [
            "M.Ed in Educational Technology from {university}",
            "MA in Instructional Design from {university}",
            "B.Ed from {university}",
            "Certificate in E-Learning Design from {university}",
            "PG Diploma in Educational Technology from {university}",
        ],
        "resume_experience_titles": [
            "Instructional Designer", "E-Learning Developer", "EdTech Specialist",
            "Content Developer", "Learning Designer", "LMS Administrator",
        ],
    },
    "academic_coordinator": {
        "jd_titles": [
            "Academic Coordinator", "Academic Administrator", "Program Coordinator",
            "Dean of Academics", "Academic Operations Manager",
        ],
        "jd_keywords": [
            "academic planning", "timetable management", "faculty coordination",
            "accreditation compliance", "student records", "program administration",
            "reporting", "NAAC", "NBA", "IQAC", "examination management",
            "academic calendar", "admission processes",
        ],
        "jd_responsibilities": [
            "Plan and manage academic calendars and timetables",
            "Coordinate with faculty on course scheduling and workload allocation",
            "Ensure compliance with accreditation requirements (NAAC, NBA)",
            "Manage student records, enrollment, and admission processes",
            "Prepare academic reports for management and regulatory bodies",
            "Oversee examination scheduling and result processing",
            "Coordinate IQAC activities and quality assurance processes",
            "Manage faculty recruitment and onboarding processes",
            "Organize faculty development programs and workshops",
            "Liaise with university affiliating bodies and regulatory agencies",
        ],
        "resume_skills": [
            "Academic Planning", "Timetable Management", "Faculty Coordination",
            "Accreditation Compliance", "Student Records Management",
            "Program Administration", "NAAC Documentation", "NBA Compliance",
            "IQAC Activities", "Examination Management", "Admission Processes",
            "Academic Calendar Planning", "Reporting", "ERP Systems",
            "Faculty Development", "Quality Assurance",
        ],
        "resume_education": [
            "M.Ed from {university}", "MBA in Education Management from {university}",
            "MA in Education from {university}", "PhD in Education from {university}",
        ],
        "resume_experience_titles": [
            "Academic Coordinator", "Program Coordinator", "Academic Administrator",
            "Vice Principal", "Academic Head", "Examination Controller",
        ],
    },
    "special_education": {
        "jd_titles": [
            "Special Education Teacher", "School Counselor", "Special Needs Educator",
            "Inclusive Education Specialist", "Student Counselor",
        ],
        "jd_keywords": [
            "special needs education", "IEP planning", "student counseling",
            "behavioral support", "inclusive education", "RCI certification",
            "therapeutic communication", "learning disabilities",
            "autism spectrum", "dyslexia", "occupational therapy",
            "psychoeducational assessment", "remedial teaching",
        ],
        "jd_responsibilities": [
            "Develop and implement Individualized Education Programs (IEPs) for special needs students",
            "Provide counseling support to students with behavioral and emotional challenges",
            "Conduct psychoeducational assessments and learning disability screenings",
            "Implement inclusive education practices in mainstream classrooms",
            "Collaborate with parents, teachers, and therapists on student development plans",
            "Design remedial teaching programs for students with learning difficulties",
            "Maintain detailed records of student progress and intervention outcomes",
            "Conduct workshops on inclusive education for faculty and staff",
            "Provide career counseling and guidance to students",
            "Implement behavioral intervention plans and positive reinforcement strategies",
        ],
        "resume_skills": [
            "Special Needs Education", "IEP Planning", "Student Counseling",
            "Behavioral Support", "Inclusive Education", "RCI Certification",
            "Therapeutic Communication", "Learning Disabilities", "Remedial Teaching",
            "Psychoeducational Assessment", "Autism Spectrum Support",
            "Dyslexia Intervention", "Career Counseling", "Parent Counseling",
            "Positive Reinforcement", "Sensory Integration",
        ],
        "resume_education": [
            "B.Ed in Special Education from {university}",
            "M.Ed in Special Education from {university}",
            "Diploma in Learning Disabilities from {university}",
            "MA in Psychology/Counseling from {university}",
            "RCI Registered (Category: Special Education)",
        ],
        "resume_experience_titles": [
            "Special Education Teacher", "School Counselor", "Inclusion Facilitator",
            "Remedial Educator", "Student Counselor", "Special Needs Coordinator",
        ],
    },
}

# ── Universities ─────────────────────────────────────────────────────────────

LAW_UNIVERSITIES = [
    "National Law School of India University, Bangalore",
    "NALSAR University of Law, Hyderabad",
    "National Law University, Delhi",
    "West Bengal National University of Juridical Sciences, Kolkata",
    "Gujarat National Law University, Gandhinagar",
    "Symbiosis Law School, Pune",
    "Faculty of Law, University of Delhi",
    "ILS Law College, Pune",
    "Government Law College, Mumbai",
    "Amity Law School, Noida",
    "Christ University, Bangalore",
    "Jindal Global Law School, Sonipat",
    "Rajiv Gandhi National University of Law, Patiala",
    "Chanakya National Law University, Patna",
    "School of Law, KIIT University, Bhubaneswar",
]

EDUCATION_UNIVERSITIES = [
    "Jamia Millia Islamia, Delhi",
    "Lady Shri Ram College, Delhi",
    "University of Mumbai",
    "Bangalore University",
    "IGNOU",
    "Tata Institute of Social Sciences, Mumbai",
    "Central Institute of Education, Delhi",
    "Loyola College, Chennai",
    "St. Xavier's College, Mumbai",
    "Christ University, Bangalore",
    "Amity University, Noida",
    "Banaras Hindu University, Varanasi",
    "University of Mysore",
    "Savitribai Phule Pune University",
    "University of Calcutta",
    "M.S. University, Baroda",
    "Regional Institute of Education, Mysore",
]

# ── Law firms and organizations ──────────────────────────────────────────────

LAW_FIRMS = [
    "Apex Legal Associates", "Sharma & Partners", "Veritas Law Chambers",
    "LegalEdge Advisors", "Nexus Legal Solutions", "Cornerstone Law Firm",
    "Pinnacle Legal Group", "Sterling Advocates", "Patel & Mehta Associates",
    "Juris Counsel LLP", "Fortis Legal Services", "Equitas Law Partners",
    "Paramount Legal Advisory", "Zenith Law Associates", "Meridian Legal Solutions",
    "Vanguard Counsel", "Atlas Legal Group", "Synergy Law Firm",
    "Keystone Legal Associates", "Summit Advocates",
]

EDUCATION_ORGS = [
    "Delhi Public School", "Kendriya Vidyalaya", "Amity International School",
    "Ryan International School", "DAV Public School", "St. Mary's School",
    "The Heritage School", "Presidium School", "Lotus Valley International",
    "Springdales School", "Modern School", "Sanskriti School",
    "Cambridge International School", "KIIT World School", "Bharatiya Vidya Bhavan",
    "National Institute of Open Schooling", "Brilliant Academy",
    "Vivekananda Education Society", "Greenfield Public School", "Sunrise Academy",
    "TechEd Learning Pvt Ltd", "LearnSphere EdTech", "SkillUp Education",
    "EduNova Solutions", "Digital Classroom India", "BrightPath Academy",
]

# ── Helper functions ─────────────────────────────────────────────────────────

def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def random_phone():
    return f"+91 {random.randint(70000, 99999)}{random.randint(10000, 99999)}"

def random_email(name):
    parts = name.lower().split()
    domain = random.choice(["gmail.com", "outlook.com", "yahoo.com", "mail.com"])
    sep = random.choice([".", "_", ""])
    num = random.choice(["", str(random.randint(1, 99))])
    return f"{parts[0]}{sep}{parts[1]}{num}@{domain}"

def random_years(is_fresher):
    if is_fresher:
        return random.choice([0, 0, 0, 1])
    return random.randint(2, 15)

def pick_n(lst, n, allow_repeat=False):
    if allow_repeat or n > len(lst):
        return [random.choice(lst) for _ in range(n)]
    return random.sample(lst, min(n, len(lst)))


# ── Resume text generators ───────────────────────────────────────────────────

def generate_legal_resume(archetype_key, archetype, is_fresher, score_target):
    """Generate a synthetic legal resume text."""
    name = random_name()
    city = random.choice(CITIES)
    email = random_email(name)
    phone = random_phone()

    university = random.choice(LAW_UNIVERSITIES)
    edu_template = random.choice(archetype["resume_education"])
    education = edu_template.format(university=university)

    years_exp = random_years(is_fresher)

    # Determine how many skills to include based on score target
    if score_target >= 85:
        n_skills = random.randint(8, 12)
        n_resp = random.randint(5, 7)
    elif score_target >= 65:
        n_skills = random.randint(6, 9)
        n_resp = random.randint(4, 6)
    elif score_target >= 45:
        n_skills = random.randint(4, 7)
        n_resp = random.randint(3, 5)
    elif score_target >= 25:
        n_skills = random.randint(2, 5)
        n_resp = random.randint(2, 3)
    else:
        n_skills = random.randint(1, 3)
        n_resp = random.randint(1, 2)

    skills = pick_n(archetype["resume_skills"], n_skills)

    # Build experience section
    if is_fresher:
        if years_exp == 0:
            exp_section = _build_fresher_legal_experience(archetype, name)
        else:
            exp_section = _build_junior_legal_experience(archetype, 1)
    else:
        exp_section = _build_experienced_legal_experience(archetype, years_exp, n_resp)

    # Add some cross-domain noise for lower scores
    extra_noise = ""
    if score_target < 45:
        noise_skills = pick_n([
            "Python Programming", "Data Analysis", "Machine Learning",
            "Digital Marketing", "Graphic Design", "Supply Chain Management",
            "Financial Modeling", "Social Media Marketing", "Cloud Computing",
            "Project Management", "Agile Methodology", "SQL Database",
        ], random.randint(2, 5))
        extra_noise = f" Additional Skills: {', '.join(noise_skills)}."

    # Assemble resume
    header = f"{name} | {city} | {email} | {phone}"
    
    summary_templates = [
        f"Dedicated legal professional with {years_exp} years of experience in {archetype_key.replace('_', ' ')}. Proficient in {', '.join(skills[:3])}. Seeking to contribute expertise in a challenging legal role.",
        f"Results-oriented legal practitioner specializing in {archetype_key.replace('_', ' ')} with demonstrated ability in {', '.join(skills[:3])}. {years_exp} years of professional experience.",
        f"Legal professional with strong background in {', '.join(skills[:2])} and {years_exp} years in the legal domain. Committed to delivering quality legal services.",
    ]
    if is_fresher:
        summary_templates = [
            f"Recent law graduate with strong academic foundation in {archetype_key.replace('_', ' ')}. Eager to apply knowledge of {', '.join(skills[:3])} in a professional legal setting.",
            f"Fresh LLB graduate passionate about {archetype_key.replace('_', ' ')}. Completed internships focusing on {', '.join(skills[:2])}. Looking for entry-level legal positions.",
            f"Motivated law graduate with academic training in {', '.join(skills[:2])}. Seeking to begin career in {archetype_key.replace('_', ' ')}.",
        ]

    summary = random.choice(summary_templates)
    skills_section = f"Skills: {', '.join(skills)}"
    education_section = f"Education: {education}"

    resume = f"{header}\n\nProfessional Summary: {summary}\n\n{exp_section}\n\n{skills_section}\n\n{education_section}{extra_noise}"
    return resume


def _build_fresher_legal_experience(archetype, name):
    firm = random.choice(LAW_FIRMS)
    tasks = pick_n([
        "Assisted senior advocates with legal research and case preparation",
        "Drafted legal notices and correspondence under supervision",
        "Organized case files and maintained documentation",
        "Attended court hearings and recorded proceedings",
        "Conducted research on relevant case laws and statutes",
        "Assisted in drafting contracts and agreements",
        "Participated in client meetings and took detailed notes",
        "Prepared summaries of legal documents and judgments",
    ], random.randint(2, 4))
    duration = random.choice(["2 months", "3 months", "6 months", "4 months"])
    return f"Internship Experience:\n{firm} — Legal Intern ({duration})\n" + "\n".join(f"- {t}" for t in tasks)


def _build_junior_legal_experience(archetype, years):
    firm = random.choice(LAW_FIRMS)
    title = random.choice(archetype["resume_experience_titles"])
    tasks = pick_n(archetype.get("jd_responsibilities", [
        "Supported legal team with research and documentation",
        "Drafted legal documents and correspondence",
    ]), random.randint(3, 4))
    return f"Experience:\n{firm} — {title} ({years} year)\n" + "\n".join(f"- {t}" for t in tasks)


def _build_experienced_legal_experience(archetype, years, n_resp):
    sections = []
    remaining = years
    num_roles = min(random.randint(1, 3), remaining)
    
    for i in range(num_roles):
        firm = random.choice(LAW_FIRMS)
        title = random.choice(archetype["resume_experience_titles"])
        role_years = max(1, remaining // (num_roles - i))
        remaining -= role_years
        tasks = pick_n(archetype.get("jd_responsibilities", [
            "Handled legal matters independently",
            "Provided legal advisory to stakeholders",
        ]), min(n_resp, random.randint(3, 5)))
        sections.append(
            f"{firm} — {title} ({role_years} years)\n" + "\n".join(f"- {t}" for t in tasks)
        )
    
    return "Experience:\n" + "\n\n".join(sections)


def generate_education_resume(archetype_key, archetype, is_fresher, score_target):
    """Generate a synthetic education resume text."""
    name = random_name()
    city = random.choice(CITIES)
    email = random_email(name)
    phone = random_phone()

    university = random.choice(EDUCATION_UNIVERSITIES)
    edu_template = random.choice(archetype["resume_education"])
    education = edu_template.format(
        university=university,
        subject=random.choice(archetype.get("subjects", ["Education"]))
    )

    years_exp = random_years(is_fresher)

    if score_target >= 85:
        n_skills = random.randint(8, 12)
        n_resp = random.randint(5, 7)
    elif score_target >= 65:
        n_skills = random.randint(6, 9)
        n_resp = random.randint(4, 6)
    elif score_target >= 45:
        n_skills = random.randint(4, 7)
        n_resp = random.randint(3, 5)
    elif score_target >= 25:
        n_skills = random.randint(2, 5)
        n_resp = random.randint(2, 3)
    else:
        n_skills = random.randint(1, 3)
        n_resp = random.randint(1, 2)

    skills = pick_n(archetype["resume_skills"], n_skills)

    if is_fresher:
        if years_exp == 0:
            exp_section = _build_fresher_education_experience(archetype)
        else:
            exp_section = _build_junior_education_experience(archetype, 1)
    else:
        exp_section = _build_experienced_education_experience(archetype, years_exp, n_resp)

    extra_noise = ""
    if score_target < 45:
        noise_skills = pick_n([
            "Python Programming", "Financial Analysis", "Supply Chain",
            "Java Development", "Digital Marketing", "Graphic Design",
            "Machine Learning", "Cloud Architecture", "Data Engineering",
            "Network Administration", "DevOps", "Salesforce",
        ], random.randint(2, 5))
        extra_noise = f" Additional Skills: {', '.join(noise_skills)}."

    header = f"{name} | {city} | {email} | {phone}"

    summary_templates = [
        f"Dedicated education professional with {years_exp} years of experience in {archetype_key.replace('_', ' ')}. Proficient in {', '.join(skills[:3])}. Passionate about student development and academic excellence.",
        f"Experienced educator specializing in {archetype_key.replace('_', ' ')} with {years_exp} years of teaching experience. Skilled in {', '.join(skills[:3])}.",
        f"Education professional with strong background in {', '.join(skills[:2])} and {years_exp} years in the education sector. Committed to fostering learning and growth.",
    ]
    if is_fresher:
        summary_templates = [
            f"Recent education graduate passionate about {archetype_key.replace('_', ' ')}. Trained in {', '.join(skills[:3])}. Seeking to begin a career in education.",
            f"Fresh B.Ed graduate with strong academic foundation. Completed teaching internships focusing on {', '.join(skills[:2])}. Eager to contribute to student learning.",
            f"Motivated education graduate with training in {', '.join(skills[:2])}. Looking for entry-level teaching or education roles.",
        ]

    summary = random.choice(summary_templates)
    skills_section = f"Skills: {', '.join(skills)}"
    education_section = f"Education: {education}"

    resume = f"{header}\n\nProfessional Summary: {summary}\n\n{exp_section}\n\n{skills_section}\n\n{education_section}{extra_noise}"
    return resume


def _build_fresher_education_experience(archetype):
    org = random.choice(EDUCATION_ORGS)
    tasks = pick_n([
        "Assisted lead teachers with classroom activities and lesson delivery",
        "Prepared teaching materials and visual aids for lessons",
        "Conducted practice teaching sessions under mentor supervision",
        "Assisted in student assessment and progress tracking",
        "Organized co-curricular activities and student events",
        "Participated in parent-teacher meetings as an observer",
        "Helped manage classroom discipline and student engagement",
        "Created worksheets and supplementary learning materials",
    ], random.randint(2, 4))
    duration = random.choice(["2 months", "3 months", "6 months", "4 months", "1 semester"])
    return f"Internship/Teaching Practice:\n{org} — Teaching Intern ({duration})\n" + "\n".join(f"- {t}" for t in tasks)


def _build_junior_education_experience(archetype, years):
    org = random.choice(EDUCATION_ORGS)
    title = random.choice(archetype["resume_experience_titles"])
    tasks = pick_n(archetype.get("jd_responsibilities", [
        "Delivered lessons and assessed student performance",
        "Maintained classroom discipline and student records",
    ]), random.randint(3, 4))
    return f"Experience:\n{org} — {title} ({years} year)\n" + "\n".join(f"- {t}" for t in tasks)


def _build_experienced_education_experience(archetype, years, n_resp):
    sections = []
    remaining = years
    num_roles = min(random.randint(1, 3), remaining)
    
    for i in range(num_roles):
        org = random.choice(EDUCATION_ORGS)
        title = random.choice(archetype["resume_experience_titles"])
        role_years = max(1, remaining // (num_roles - i))
        remaining -= role_years
        tasks = pick_n(archetype.get("jd_responsibilities", [
            "Delivered instruction and managed academic programs",
            "Contributed to curriculum development and assessment",
        ]), min(n_resp, random.randint(3, 5)))
        sections.append(
            f"{org} — {title} ({role_years} years)\n" + "\n".join(f"- {t}" for t in tasks)
        )
    
    return "Experience:\n" + "\n\n".join(sections)


# ── JD text generators ───────────────────────────────────────────────────────

def generate_legal_jd(archetype_key, archetype):
    """Generate a synthetic legal JD text."""
    title = random.choice(archetype["jd_titles"])
    firm = random.choice(LAW_FIRMS)
    city = random.choice(CITIES)
    
    n_resp = random.randint(5, 8)
    responsibilities = pick_n(archetype["jd_responsibilities"], n_resp)
    
    n_kw = random.randint(4, 7)
    keywords = pick_n(archetype["jd_keywords"], n_kw)
    
    exp_req = random.choice([
        "0-2 years of experience in a relevant legal role",
        "2-5 years of experience in legal practice",
        "3-7 years of experience in a law firm or corporate legal department",
        "5+ years of experience in the relevant legal domain",
        "Freshers with strong academic background may also apply",
        "1-3 years of experience preferred, fresh graduates considered",
    ])
    
    edu_req = random.choice([
        "LLB or equivalent degree from a recognized university",
        "LLB/LLM from a reputed law school",
        "BA LLB (Hons) or equivalent; LLM preferred",
        "LLB with Bar Council registration",
        "Law degree with relevant specialization",
    ])

    qualifications = [
        edu_req,
        exp_req,
        f"Strong knowledge of {', '.join(keywords[:3])}",
        "Excellent written and verbal communication skills",
        random.choice([
            "Ability to work independently and as part of a team",
            "Strong analytical and problem-solving abilities",
            "Detail-oriented with excellent organizational skills",
            "Ability to handle multiple matters simultaneously",
        ]),
    ]

    jd = f"Job Title: {title}\nCompany: {firm}\nLocation: {city}\n\n"
    jd += f"About the Role:\nWe are looking for a {title} to join our team at {firm}. "
    jd += f"The ideal candidate will have expertise in {', '.join(keywords[:3])} "
    jd += f"and will contribute to our {archetype_key.replace('_', ' ')} practice.\n\n"
    jd += "Responsibilities:\n" + "\n".join(f"- {r}" for r in responsibilities) + "\n\n"
    jd += "Qualifications:\n" + "\n".join(f"- {q}" for q in qualifications) + "\n\n"
    jd += f"Key Skills: {', '.join(keywords)}"

    return jd


def generate_education_jd(archetype_key, archetype):
    """Generate a synthetic education JD text."""
    title = random.choice(archetype["jd_titles"])
    org = random.choice(EDUCATION_ORGS)
    city = random.choice(CITIES)
    
    n_resp = random.randint(5, 8)
    responsibilities = pick_n(archetype["jd_responsibilities"], n_resp)
    
    n_kw = random.randint(4, 7)
    keywords = pick_n(archetype["jd_keywords"], n_kw)
    
    exp_req = random.choice([
        "0-2 years of teaching experience",
        "2-5 years of experience in education",
        "3-7 years of experience in teaching or academic administration",
        "5+ years of experience in the relevant education domain",
        "Freshers with B.Ed/M.Ed may also apply",
        "1-3 years of experience preferred, fresh graduates welcome",
    ])
    
    edu_req = random.choice([
        "B.Ed or equivalent teaching qualification",
        "M.Ed or Master's in relevant subject",
        "B.Ed with CTET/TET qualification preferred",
        "PhD or M.Phil for lecturer positions",
        "Relevant degree with teaching certification",
    ])

    qualifications = [
        edu_req,
        exp_req,
        f"Strong knowledge of {', '.join(keywords[:3])}",
        "Excellent communication and interpersonal skills",
        random.choice([
            "Passion for teaching and student development",
            "Ability to use technology in education effectively",
            "Strong classroom management and organizational skills",
            "Commitment to continuous professional development",
        ]),
    ]

    jd = f"Job Title: {title}\nOrganization: {org}\nLocation: {city}\n\n"
    jd += f"About the Role:\nWe are seeking a {title} to join {org}. "
    jd += f"The ideal candidate will have expertise in {', '.join(keywords[:3])} "
    jd += f"and will contribute to our academic programs.\n\n"
    jd += "Responsibilities:\n" + "\n".join(f"- {r}" for r in responsibilities) + "\n\n"
    jd += "Qualifications:\n" + "\n".join(f"- {q}" for q in qualifications) + "\n\n"
    jd += f"Key Skills: {', '.join(keywords)}"

    return jd


# ── Pair generation with score-aware misalignment ────────────────────────────

def generate_legal_pair(archetype_key, archetype, score_range, is_fresher):
    """Generate a single legal resume-JD pair with a target score."""
    lo, hi = score_range
    score = round(random.uniform(lo, hi), 2)
    
    jd = generate_legal_jd(archetype_key, archetype)
    
    if score >= 65:
        # Good/Excellent: resume matches JD archetype
        resume = generate_legal_resume(archetype_key, archetype, is_fresher, score)
    elif score >= 45:
        # Moderate: resume partially matches — sometimes use a different archetype
        if random.random() < 0.4:
            other_key = random.choice([k for k in LEGAL_ARCHETYPES if k != archetype_key])
            other_arch = LEGAL_ARCHETYPES[other_key]
            resume = generate_legal_resume(other_key, other_arch, is_fresher, score)
        else:
            resume = generate_legal_resume(archetype_key, archetype, is_fresher, score)
    else:
        # Weak/Poor: resume is from a different archetype or has heavy noise
        if random.random() < 0.6:
            other_key = random.choice([k for k in LEGAL_ARCHETYPES if k != archetype_key])
            other_arch = LEGAL_ARCHETYPES[other_key]
            resume = generate_legal_resume(other_key, other_arch, is_fresher, score)
        else:
            resume = generate_legal_resume(archetype_key, archetype, is_fresher, score)
    
    return resume, jd, score


def generate_education_pair(archetype_key, archetype, score_range, is_fresher):
    """Generate a single education resume-JD pair with a target score."""
    lo, hi = score_range
    score = round(random.uniform(lo, hi), 2)
    
    jd = generate_education_jd(archetype_key, archetype)
    
    if score >= 65:
        resume = generate_education_resume(archetype_key, archetype, is_fresher, score)
    elif score >= 45:
        if random.random() < 0.4:
            other_key = random.choice([k for k in EDUCATION_ARCHETYPES if k != archetype_key])
            other_arch = EDUCATION_ARCHETYPES[other_key]
            resume = generate_education_resume(other_key, other_arch, is_fresher, score)
        else:
            resume = generate_education_resume(archetype_key, archetype, is_fresher, score)
    else:
        if random.random() < 0.6:
            other_key = random.choice([k for k in EDUCATION_ARCHETYPES if k != archetype_key])
            other_arch = EDUCATION_ARCHETYPES[other_key]
            resume = generate_education_resume(other_key, other_arch, is_fresher, score)
        else:
            resume = generate_education_resume(archetype_key, archetype, is_fresher, score)
    
    return resume, jd, score


# ── Main generation logic ────────────────────────────────────────────────────

def generate_domain_pairs(archetypes, score_dist, total, fresher_pct, domain_label, pair_fn):
    """Generate all pairs for a domain with correct score distribution and fresher ratio."""
    pairs = []
    archetype_keys = list(archetypes.keys())
    pairs_per_archetype = total // len(archetype_keys)
    
    # Track fresher count
    fresher_target = int(total * fresher_pct)
    fresher_count = 0
    
    # Distribute pairs across score bands
    band_assignments = []
    for lo, hi, count in score_dist:
        for _ in range(count):
            band_assignments.append((lo, hi))
    
    random.shuffle(band_assignments)
    
    # Distribute across archetypes (roughly equal)
    archetype_assignments = []
    for i, band in enumerate(band_assignments):
        arch_key = archetype_keys[i % len(archetype_keys)]
        archetype_assignments.append((arch_key, band))
    
    # Determine which pairs are freshers
    fresher_indices = set(random.sample(range(total), fresher_target))
    
    for i, (arch_key, (lo, hi)) in enumerate(archetype_assignments):
        is_fresher = i in fresher_indices
        if is_fresher:
            fresher_count += 1
        
        archetype = archetypes[arch_key]
        resume, jd, score = pair_fn(arch_key, archetype, (lo, hi), is_fresher)
        pairs.append({
            "resume_text": resume,
            "jd_text": jd,
            "ats_score": score,
            "domain_label": domain_label,
        })
    
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Fresher pairs: {fresher_count} ({fresher_count/len(pairs)*100:.1f}%)")
    
    # Verify score distribution
    print("  Score distribution:")
    for lo, hi, expected in score_dist:
        actual = sum(1 for p in pairs if lo <= p["ats_score"] <= hi)
        print(f"    {lo}-{hi}: {actual} pairs (expected {expected})")
    
    return pairs


def save_csv(pairs, filepath):
    """Save pairs to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_text", "jd_text", "ats_score", "domain_label"])
        writer.writeheader()
        writer.writerows(pairs)
    print(f"  Saved to: {filepath}")


def main():
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 60)
    
    # Task 1: Legal pairs
    print("\n[Task 1] Generating 600 Legal domain pairs (domain_label=5)...")
    legal_pairs = generate_domain_pairs(
        archetypes=LEGAL_ARCHETYPES,
        score_dist=LEGAL_SCORE_DIST,
        total=600,
        fresher_pct=0.30,
        domain_label=5,
        pair_fn=generate_legal_pair,
    )
    legal_path = SYNTHETIC_DIR / "legal_synthetic.csv"
    save_csv(legal_pairs, legal_path)
    
    # Task 2: Education pairs
    print("\n[Task 2] Generating 500 Education domain pairs (domain_label=6)...")
    education_pairs = generate_domain_pairs(
        archetypes=EDUCATION_ARCHETYPES,
        score_dist=EDUCATION_SCORE_DIST,
        total=500,
        fresher_pct=0.35,
        domain_label=6,
        pair_fn=generate_education_pair,
    )
    education_path = SYNTHETIC_DIR / "education_synthetic.csv"
    save_csv(education_pairs, education_path)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nLegal:     {legal_path} ({len(legal_pairs)} pairs)")
    print(f"Education: {education_path} ({len(education_pairs)} pairs)")
    print("\nNext step: Run 'python merge_synthetic.py' to merge with training data.")


if __name__ == "__main__":
    main()
