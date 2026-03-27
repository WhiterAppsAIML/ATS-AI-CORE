"""
Phase 2 — Generate IT (400) and Finance (400) supplemental synthetic pairs,
update loss weights, merge all data into merged_final.csv.
"""

import csv
import os
import random
from pathlib import Path

import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

IT_CSV = SYNTHETIC_DIR / "it_supplemental.csv"
FINANCE_CSV = SYNTHETIC_DIR / "finance_supplemental.csv"

# ── Name pools (fictional) ──────────────────────────────────────────────────
FIRST_NAMES = [
    "Aarav", "Aditi", "Aisha", "Amit", "Ananya", "Arjun", "Bhavna", "Chandra",
    "Deepak", "Divya", "Esha", "Farhan", "Gaurav", "Harini", "Ishaan", "Jaya",
    "Karthik", "Kavya", "Lakshmi", "Manish", "Meera", "Naveen", "Neha", "Omkar",
    "Pallavi", "Priya", "Rahul", "Ravi", "Rohit", "Saanvi", "Sahil", "Shreya",
    "Sneha", "Suresh", "Tanvi", "Ujjwal", "Varun", "Vikram", "Yash", "Zara",
    "Akshay", "Bharat", "Chitra", "Diya", "Ekta", "Gauri", "Hari", "Isha",
    "Janaki", "Kunal", "Lata", "Mohan", "Nisha", "Pooja", "Rajesh", "Sanjay",
    "Tarun", "Uma", "Vivek", "Swati", "Anil", "Bala", "Girish", "Hemant",
    "Komal", "Manoj", "Nitya", "Prakash", "Reema", "Shalini",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Reddy", "Nair",
    "Joshi", "Iyer", "Chopra", "Mishra", "Mehta", "Das", "Rao", "Bhat",
    "Kulkarni", "Desai", "Agarwal", "Banerjee", "Thakur", "Malhotra",
    "Srivastava", "Pillai", "Mukherjee", "Ghosh", "Kaur", "Chauhan",
    "Pandey", "Saxena", "Tiwari", "Bose", "Sen", "Dutta", "Roy",
]
CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Noida", "Gurgaon",
    "Chandigarh", "Kochi", "Indore", "Bhopal", "Nagpur", "Coimbatore",
]
COMPANIES = [
    "TechNova Solutions", "DataBridge Pvt Ltd", "Cloudware Systems",
    "SilverPeak Technologies", "NexGen Analytics", "BlueStar Infotech",
    "Optima Digital", "VelocityStack Inc", "PrimeLogic Software",
    "CoreWave Technologies", "Pinnacle Systems", "FusionTech Labs",
    "Meridian Solutions", "QuantumLeap Ltd", "InnovateX Software",
    "AlphaEdge Tech", "CrestPoint Digital", "SwiftCode Solutions",
    "Zenith Infosys", "GreenByte Technologies", "BrightPath IT",
    "CodeSphere Global", "Horizon Financials", "TrustBridge Capital",
    "GlobalFin Services", "PeakVest Advisors", "SilverOak Investments",
    "CreditShield Corp", "FinEdge Analytics", "WealthBridge Partners",
    "AuditPrime LLP", "CapitalWise Group", "FiscalPoint Consulting",
    "NorthStar Finance", "RiskGuard Solutions", "ValueNet Accounting",
]


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_email(name):
    parts = name.lower().split()
    domain = random.choice(["gmail.com", "outlook.com", "yahoo.com", "protonmail.com"])
    sep = random.choice([".", "_", ""])
    return f"{parts[0]}{sep}{parts[1]}{random.randint(10, 99)}@{domain}"


def random_phone():
    return f"+91 {random.randint(7000000000, 9999999999)}"


# ═══════════════════════════════════════════════════════════════════════════
# IT / SOFTWARE ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════

IT_ARCHETYPES = {
    "Backend Engineer": {
        "jd_skills": ["Python", "Java", "Node.js", "REST APIs", "PostgreSQL", "MySQL",
                       "Docker", "microservices", "Redis", "message queues", "RabbitMQ",
                       "Kafka", "Git", "unit testing", "CI/CD", "Linux"],
        "jd_responsibilities": [
            "Design and develop scalable backend services and REST APIs",
            "Write clean, maintainable, and well-tested code",
            "Collaborate with frontend and DevOps teams",
            "Optimize database queries and system performance",
            "Participate in code reviews and architectural discussions",
            "Implement authentication and authorization mechanisms",
            "Monitor and troubleshoot production issues",
            "Design database schemas and data models",
        ],
        "jd_qualifications": [
            "B.Tech/B.E. in Computer Science or related field",
            "Strong proficiency in Python or Java",
            "Experience with relational and NoSQL databases",
            "Understanding of software design patterns",
            "Familiarity with containerization and orchestration tools",
        ],
        "resume_skills_pool": [
            "Python", "Java", "Node.js", "REST APIs", "PostgreSQL", "MySQL",
            "Docker", "microservices", "Redis", "RabbitMQ", "Kafka", "Git",
            "unit testing", "CI/CD", "Linux", "Flask", "FastAPI", "Django",
            "Spring Boot", "Express.js", "MongoDB", "SQLAlchemy", "Celery",
            "Kubernetes", "Nginx", "GraphQL", "gRPC", "AWS", "Jenkins",
            "HTML", "CSS", "Photoshop", "Excel", "PowerPoint", "Tally",
        ],
        "resume_exp_tasks": [
            "Built RESTful APIs serving 10K+ daily requests using {skill}",
            "Designed microservices architecture for order management system",
            "Implemented caching layer with Redis reducing response time by 40%",
            "Migrated monolithic application to containerized microservices",
            "Set up CI/CD pipelines with Jenkins and Docker",
            "Optimized SQL queries reducing page load time by 60%",
            "Developed authentication service using JWT and OAuth2",
            "Led backend team of 3 engineers on e-commerce platform",
        ],
        "fresher_exp_tasks": [
            "Built a REST API project using {skill} as part of coursework",
            "Developed a CRUD application with {skill} and PostgreSQL",
            "Completed online course in backend development with {skill}",
            "Contributed to open-source project on GitHub",
            "Built a containerized todo app using Docker and {skill}",
            "Created a simple microservice for student project",
        ],
    },
    "Frontend Developer": {
        "jd_skills": ["React", "Vue.js", "Angular", "TypeScript", "JavaScript",
                       "CSS", "HTML", "responsive design", "Webpack", "testing",
                       "Jest", "Redux", "REST APIs", "Git", "Figma"],
        "jd_responsibilities": [
            "Build responsive and accessible web applications",
            "Implement pixel-perfect UI components from Figma designs",
            "Write unit and integration tests using Jest and React Testing Library",
            "Optimize frontend performance and bundle size",
            "Collaborate with UX designers and backend engineers",
            "Maintain component library and design system",
            "Implement state management solutions",
        ],
        "jd_qualifications": [
            "B.Tech/B.E. in Computer Science or equivalent",
            "Proficiency in React or Vue.js with TypeScript",
            "Strong understanding of CSS, HTML5, and responsive design",
            "Experience with modern build tools and test frameworks",
        ],
        "resume_skills_pool": [
            "React", "Vue.js", "Angular", "TypeScript", "JavaScript", "CSS", "HTML",
            "responsive design", "Webpack", "Jest", "Redux", "REST APIs", "Git",
            "Figma", "Tailwind CSS", "Material UI", "Next.js", "Nuxt.js",
            "Sass", "Bootstrap", "Storybook", "Cypress", "Node.js",
            "Python", "SQL", "Tally", "Photoshop", "Excel",
        ],
        "resume_exp_tasks": [
            "Developed customer-facing dashboard using React and TypeScript",
            "Implemented responsive design supporting mobile and desktop devices",
            "Built reusable component library with Storybook documentation",
            "Migrated legacy jQuery codebase to React with 90% test coverage",
            "Optimized bundle size by 45% using code splitting and lazy loading",
            "Led frontend development for SaaS product serving 5K+ users",
        ],
        "fresher_exp_tasks": [
            "Built a portfolio website using React and Tailwind CSS",
            "Completed frontend development bootcamp with {skill} specialization",
            "Created a weather app using {skill} and public APIs",
            "Developed a student project e-commerce UI with {skill}",
            "Contributed to open-source UI component library",
        ],
    },
    "Data Engineer / ML": {
        "jd_skills": ["Python", "TensorFlow", "PyTorch", "pandas", "SQL",
                       "machine learning", "data pipeline", "Spark", "Airflow",
                       "scikit-learn", "NumPy", "AWS", "GCP", "ETL", "statistics"],
        "jd_responsibilities": [
            "Design and build data pipelines for ML model training",
            "Develop and deploy machine learning models to production",
            "Perform exploratory data analysis and feature engineering",
            "Optimize model performance and inference latency",
            "Collaborate with data scientists and product teams",
            "Maintain data quality and monitoring dashboards",
            "Implement A/B testing frameworks",
        ],
        "jd_qualifications": [
            "M.Tech/B.Tech in CS, Statistics, or related field",
            "Strong Python skills with ML frameworks (TensorFlow/PyTorch)",
            "Experience with data processing tools (Spark, pandas)",
            "Understanding of statistical methods and ML algorithms",
        ],
        "resume_skills_pool": [
            "Python", "TensorFlow", "PyTorch", "pandas", "SQL", "machine learning",
            "data pipeline", "Spark", "Airflow", "scikit-learn", "NumPy", "AWS",
            "GCP", "ETL", "statistics", "Keras", "NLP", "computer vision",
            "deep learning", "XGBoost", "feature engineering", "Docker",
            "Jupyter", "Matplotlib", "Tableau", "R", "Hadoop",
            "Excel", "PowerPoint", "Tally", "Photoshop",
        ],
        "resume_exp_tasks": [
            "Built end-to-end ML pipeline processing 2M+ records daily",
            "Developed recommendation engine improving CTR by 25%",
            "Implemented NLP classification model with 92% accuracy",
            "Designed real-time feature store using Redis and Spark",
            "Deployed models to production using TensorFlow Serving",
            "Led data science team on customer churn prediction project",
        ],
        "fresher_exp_tasks": [
            "Completed ML specialization on Coursera using {skill}",
            "Built a sentiment analysis project using {skill} and NLP",
            "Participated in Kaggle competition achieving top 20% rank",
            "Developed image classifier using {skill} for academic project",
            "Created data analysis project using pandas and Matplotlib",
        ],
    },
    "DevOps / Cloud": {
        "jd_skills": ["AWS", "GCP", "Azure", "Kubernetes", "Terraform", "CI/CD",
                       "Linux", "Docker", "Jenkins", "Ansible", "monitoring",
                       "Prometheus", "Grafana", "networking", "security"],
        "jd_responsibilities": [
            "Design and maintain cloud infrastructure on AWS/GCP/Azure",
            "Implement and manage CI/CD pipelines",
            "Automate infrastructure provisioning using Terraform and Ansible",
            "Monitor system health and respond to incidents",
            "Implement security best practices and compliance controls",
            "Optimize cloud costs and resource utilization",
            "Manage Kubernetes clusters and container orchestration",
        ],
        "jd_qualifications": [
            "B.Tech in CS/IT or equivalent experience",
            "AWS/GCP/Azure certification preferred",
            "Strong Linux system administration skills",
            "Experience with Infrastructure as Code tools",
        ],
        "resume_skills_pool": [
            "AWS", "GCP", "Azure", "Kubernetes", "Terraform", "CI/CD", "Linux",
            "Docker", "Jenkins", "Ansible", "Prometheus", "Grafana", "networking",
            "security", "Nginx", "HAProxy", "CloudFormation", "Helm",
            "GitOps", "ArgoCD", "Datadog", "ELK Stack", "Bash scripting",
            "Python", "Git", "SQL", "Excel", "PowerPoint",
        ],
        "resume_exp_tasks": [
            "Managed AWS infrastructure serving 50K+ daily active users",
            "Implemented Kubernetes clusters with auto-scaling for microservices",
            "Automated infrastructure provisioning reducing setup time by 80%",
            "Designed CI/CD pipelines achieving 15-minute deployment cycles",
            "Implemented monitoring and alerting using Prometheus and Grafana",
            "Reduced cloud costs by 35% through resource optimization",
        ],
        "fresher_exp_tasks": [
            "Completed AWS Cloud Practitioner certification",
            "Set up a personal Kubernetes cluster on GCP for learning",
            "Built CI/CD pipeline for college project using Jenkins and Docker",
            "Completed Linux administration course and shell scripting",
            "Deployed a sample application on {skill} as part of coursework",
        ],
    },
    "Mobile Developer": {
        "jd_skills": ["Flutter", "React Native", "Android", "iOS", "Swift",
                       "Kotlin", "Dart", "REST APIs", "Firebase", "app deployment",
                       "Git", "UI/UX", "state management", "testing"],
        "jd_responsibilities": [
            "Develop and maintain cross-platform mobile applications",
            "Implement clean UI following Material Design / iOS HIG guidelines",
            "Integrate REST APIs and third-party services",
            "Write unit and widget tests for quality assurance",
            "Publish and maintain apps on Play Store and App Store",
            "Optimize app performance and battery usage",
            "Collaborate with designers and backend team",
        ],
        "jd_qualifications": [
            "B.Tech/B.E. in Computer Science or equivalent",
            "Proficiency in Flutter/React Native or native Android/iOS",
            "Experience with mobile app architecture patterns (MVVM, BLoC)",
            "Understanding of mobile CI/CD and app distribution",
        ],
        "resume_skills_pool": [
            "Flutter", "React Native", "Android", "iOS", "Swift", "Kotlin",
            "Dart", "REST APIs", "Firebase", "Git", "UI/UX", "state management",
            "testing", "MVVM", "BLoC", "Provider", "Redux", "SQLite",
            "Hive", "Play Store", "App Store", "Xcode", "Android Studio",
            "Python", "Java", "HTML", "CSS", "Photoshop",
        ],
        "resume_exp_tasks": [
            "Developed e-commerce mobile app with 100K+ downloads using Flutter",
            "Built real-time chat feature using Firebase and {skill}",
            "Implemented push notifications and deep linking",
            "Published 3 apps on Google Play Store with 4.5+ rating",
            "Led mobile development team building health tracking app",
            "Optimized app startup time by 50% through lazy loading",
        ],
        "fresher_exp_tasks": [
            "Built a weather app using {skill} for college project",
            "Completed mobile development course on Udemy with {skill}",
            "Developed a task manager app and published on Play Store",
            "Participated in hackathon building {skill} based application",
            "Created a personal portfolio app using Flutter/React Native",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# FINANCE / BANKING ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════

FINANCE_ARCHETYPES = {
    "Investment Banking Analyst": {
        "jd_skills": ["DCF", "financial modelling", "Excel", "Bloomberg", "M&A",
                       "valuation", "financial statements", "PowerPoint", "equity research",
                       "pitch books", "due diligence", "capital markets"],
        "jd_responsibilities": [
            "Build and maintain detailed financial models (DCF, LBO, M&A)",
            "Prepare pitch books and client presentation materials",
            "Conduct industry and company research for deal origination",
            "Support senior bankers in deal execution and due diligence",
            "Analyze financial statements and prepare valuation analyses",
            "Monitor capital markets and maintain deal pipeline tracking",
        ],
        "jd_qualifications": [
            "MBA/CA/CFA or B.Com/BBA from a top institution",
            "Strong financial modelling and Excel skills",
            "Understanding of valuation methodologies (DCF, comparable analysis)",
            "Excellent analytical and presentation skills",
        ],
        "resume_skills_pool": [
            "DCF", "financial modelling", "Excel", "Bloomberg", "M&A", "valuation",
            "financial statements", "PowerPoint", "equity research", "pitch books",
            "due diligence", "capital markets", "LBO", "comparable analysis",
            "CFA", "CA", "financial analysis", "VBA", "SQL",
            "Python", "Tableau", "Word", "Tally", "HTML",
        ],
        "resume_exp_tasks": [
            "Built DCF and M&A models for 15+ transactions worth $500M+",
            "Prepared pitch books that secured 3 mandates for the firm",
            "Conducted due diligence on 8 M&A transactions",
            "Analyzed financial statements of 20+ companies for equity research",
            "Supported IPO execution for mid-market technology company",
            "Developed automated financial model templates in Excel VBA",
        ],
        "fresher_exp_tasks": [
            "Completed financial modelling course with DCF and valuation methods",
            "Built a sample DCF model for a public company as coursework",
            "Interned at {company} in investment banking division",
            "Participated in CFA Institute Research Challenge",
            "Completed Bloomberg Market Concepts certification",
        ],
    },
    "Risk Analyst": {
        "jd_skills": ["credit risk", "VaR", "Basel norms", "SQL", "risk reporting",
                       "risk assessment", "stress testing", "regulatory compliance",
                       "SAS", "Python", "probability", "statistics"],
        "jd_responsibilities": [
            "Conduct credit risk assessments for lending portfolios",
            "Calculate and monitor VaR and other risk metrics",
            "Prepare risk reports for senior management and regulators",
            "Implement stress testing scenarios per Basel III/IV norms",
            "Develop and validate risk models",
            "Monitor market risk exposures and credit limits",
            "Ensure compliance with RBI and Basel regulatory requirements",
        ],
        "jd_qualifications": [
            "MBA/M.Sc in Finance, Statistics, or Economics",
            "FRM/PRM certification preferred",
            "Strong quantitative and statistical skills",
            "Experience with risk management tools and SQL",
        ],
        "resume_skills_pool": [
            "credit risk", "VaR", "Basel norms", "SQL", "risk reporting",
            "risk assessment", "stress testing", "regulatory compliance",
            "SAS", "Python", "probability", "statistics", "FRM",
            "Monte Carlo", "Excel", "R", "Tableau", "credit scoring",
            "PD/LGD/EAD", "RAROC", "market risk", "operational risk",
            "HTML", "Photoshop", "Tally", "PowerPoint",
        ],
        "resume_exp_tasks": [
            "Monitored credit risk portfolio worth INR 5,000 Cr",
            "Implemented VaR models reducing risk reporting time by 30%",
            "Developed credit scoring model with 88% accuracy using logistic regression",
            "Conducted stress testing scenarios per RBI guidelines",
            "Prepared quarterly risk reports for board-level review",
            "Validated Basel III capital adequacy calculations",
        ],
        "fresher_exp_tasks": [
            "Completed FRM Level 1 certification covering risk fundamentals",
            "Built a credit scoring model as final year project using {skill}",
            "Interned in risk management division at {company}",
            "Completed coursework in financial risk management and statistics",
            "Analyzed sample loan portfolio for default prediction project",
        ],
    },
    "Financial Analyst / FP&A": {
        "jd_skills": ["budgeting", "forecasting", "Excel", "ERP systems", "variance analysis",
                       "financial reporting", "SAP", "Oracle", "Power BI", "dashboards",
                       "cost analysis", "revenue modelling"],
        "jd_responsibilities": [
            "Prepare monthly and quarterly financial reports and forecasts",
            "Conduct variance analysis comparing actuals to budget",
            "Develop and maintain financial models for business planning",
            "Create dashboards and visualizations for senior leadership",
            "Support annual budgeting and long-range planning processes",
            "Analyze cost structures and identify optimization opportunities",
            "Coordinate with cross-functional teams for data collection",
        ],
        "jd_qualifications": [
            "B.Com/MBA in Finance or CA/CMA qualification",
            "Advanced Excel and ERP system proficiency",
            "Experience with BI tools (Power BI, Tableau)",
            "Strong analytical and communication skills",
        ],
        "resume_skills_pool": [
            "budgeting", "forecasting", "Excel", "ERP systems", "variance analysis",
            "financial reporting", "SAP", "Oracle", "Power BI", "dashboards",
            "cost analysis", "revenue modelling", "Tableau", "VBA", "SQL",
            "Tally", "QuickBooks", "CMA", "CA", "IFRS",
            "Python", "HTML", "PowerPoint", "Word",
        ],
        "resume_exp_tasks": [
            "Managed annual budgeting process for business unit of INR 200 Cr revenue",
            "Built Power BI dashboards tracking 25+ financial KPIs",
            "Reduced variance between forecast and actuals by 15% through model improvements",
            "Prepared monthly management reporting packages for C-suite",
            "Automated financial reporting using Excel VBA saving 20 hours/month",
            "Led FP&A analysis for new product launch business case",
        ],
        "fresher_exp_tasks": [
            "Completed CA articleship focusing on financial reporting",
            "Built a budgeting template and analysis project in Excel",
            "Interned at {company} in finance and accounting team",
            "Completed financial analysis certification on Coursera",
            "Created sample variance analysis report as coursework",
        ],
    },
    "Chartered Accountant / Audit": {
        "jd_skills": ["IFRS", "audit procedures", "tax compliance", "Tally", "Big 4",
                       "statutory audit", "GST", "income tax", "internal audit",
                       "accounting standards", "financial statements", "Excel"],
        "jd_responsibilities": [
            "Conduct statutory and internal audit engagements",
            "Prepare and review financial statements per IFRS/Ind AS",
            "Ensure compliance with tax regulations (GST, Income Tax)",
            "Perform analytical procedures and substantive testing",
            "Prepare audit reports and communicate findings to management",
            "Review internal controls and recommend improvements",
            "Coordinate with regulatory bodies and external auditors",
        ],
        "jd_qualifications": [
            "CA qualification (ICAI) required",
            "Knowledge of IFRS/Ind AS and auditing standards",
            "Experience with Tally, SAP, or similar accounting software",
            "Strong attention to detail and analytical skills",
        ],
        "resume_skills_pool": [
            "IFRS", "audit procedures", "tax compliance", "Tally", "Big 4",
            "statutory audit", "GST", "income tax", "internal audit",
            "accounting standards", "financial statements", "Excel",
            "Ind AS", "SA", "CARO", "SAP", "QuickBooks", "CA",
            "transfer pricing", "TDS", "ROC compliance",
            "Python", "SQL", "PowerPoint", "Word",
        ],
        "resume_exp_tasks": [
            "Led statutory audit engagements for 10+ clients across industries",
            "Prepared financial statements per Ind AS for listed company",
            "Managed GST compliance and return filing for 15+ entities",
            "Conducted internal audit identifying INR 2 Cr in process improvements",
            "Reviewed and implemented IFRS 15 revenue recognition standard",
            "Coordinated tax assessments and resolved notices worth INR 50 Lakh",
        ],
        "fresher_exp_tasks": [
            "Completed CA articleship at {company} covering statutory audit",
            "Assisted in GST return filing and tax compliance during internship",
            "Prepared trial balance and financial statements for small entities",
            "Completed certification in Tally ERP and accounting software",
            "Participated in audit engagement as junior associate",
        ],
    },
    "FinTech / Quant": {
        "jd_skills": ["Python", "algorithmic trading", "data analysis", "financial APIs",
                       "statistics", "machine learning", "quantitative analysis", "SQL",
                       "R", "time series", "risk modelling", "NumPy", "pandas"],
        "jd_responsibilities": [
            "Develop quantitative trading strategies and backtesting frameworks",
            "Build data pipelines for financial market data analysis",
            "Implement statistical models for pricing and risk assessment",
            "Analyze large financial datasets to identify patterns and signals",
            "Develop and maintain financial APIs and data services",
            "Collaborate with traders and portfolio managers on strategy",
            "Monitor and optimize algorithmic trading system performance",
        ],
        "jd_qualifications": [
            "M.Sc/M.Tech in Quantitative Finance, Statistics, CS, or Mathematics",
            "Strong Python and statistics skills",
            "Experience with financial data and time series analysis",
            "Understanding of financial markets and instruments",
        ],
        "resume_skills_pool": [
            "Python", "algorithmic trading", "data analysis", "financial APIs",
            "statistics", "machine learning", "quantitative analysis", "SQL",
            "R", "time series", "risk modelling", "NumPy", "pandas",
            "Matplotlib", "Jupyter", "Bloomberg API", "Zerodha Kite API",
            "options pricing", "Black-Scholes", "Monte Carlo",
            "Excel", "Tableau", "HTML", "PowerPoint",
        ],
        "resume_exp_tasks": [
            "Developed algorithmic trading strategies generating 18% annual returns",
            "Built real-time market data pipeline processing 1M+ ticks/day",
            "Implemented options pricing model using Black-Scholes and Monte Carlo",
            "Created backtesting framework for quantitative trading strategies",
            "Analyzed financial time series data for pattern recognition",
            "Developed risk-adjusted portfolio optimization models",
        ],
        "fresher_exp_tasks": [
            "Built a stock price prediction model using {skill} as thesis project",
            "Completed quantitative finance certification covering options and derivatives",
            "Developed a simple algorithmic trading bot using {skill} and Zerodha API",
            "Analyzed financial datasets using pandas and statistical methods",
            "Interned at {company} in quantitative research team",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PAIR GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def generate_jd(archetype_name: str, archetype: dict) -> str:
    """Generate a realistic JD text from an archetype."""
    company = random.choice(COMPANIES)
    city = random.choice(CITIES)
    exp_range = random.choice(["0-2", "1-3", "2-5", "3-6", "5-8", "5-10", "7-12"])

    # Pick random subset of responsibilities and qualifications
    n_resp = random.randint(3, min(5, len(archetype["jd_responsibilities"])))
    n_qual = random.randint(2, min(4, len(archetype["jd_qualifications"])))
    responsibilities = random.sample(archetype["jd_responsibilities"], n_resp)
    qualifications = random.sample(archetype["jd_qualifications"], n_qual)

    # Pick skills subset
    n_skills = random.randint(5, min(10, len(archetype["jd_skills"])))
    skills = random.sample(archetype["jd_skills"], n_skills)

    jd = f"Job Title: {archetype_name}\n"
    jd += f"Company: {company}\n"
    jd += f"Location: {city}\n"
    jd += f"Experience: {exp_range} years\n\n"
    jd += "About the Role:\n"
    jd += f"We are looking for a talented {archetype_name} to join our team at {company}. "
    jd += f"The ideal candidate will have strong skills in {', '.join(skills[:3])} and "
    jd += f"experience with {', '.join(skills[3:6])}.\n\n"
    jd += "Responsibilities:\n"
    for r in responsibilities:
        jd += f"- {r}\n"
    jd += "\nRequired Skills:\n"
    jd += ", ".join(skills) + "\n\n"
    jd += "Qualifications:\n"
    for q in qualifications:
        jd += f"- {q}\n"
    return jd


def generate_resume(archetype: dict, is_fresher: bool, match_level: str, jd_skills_used: list) -> str:
    """Generate a realistic resume text with controlled keyword alignment."""
    name = random_name()
    city = random.choice(CITIES)
    email = random_email(name)
    phone = random_phone()

    # Determine how many JD skills to include in resume based on match level
    if match_level == "strong":
        skill_fraction = random.uniform(0.6, 0.9)
    elif match_level == "moderate":
        skill_fraction = random.uniform(0.3, 0.55)
    elif match_level == "weak":
        skill_fraction = random.uniform(0.1, 0.28)
    else:  # mismatched
        skill_fraction = random.uniform(0.0, 0.08)

    n_match = max(0, int(len(jd_skills_used) * skill_fraction))
    matched_skills = random.sample(jd_skills_used, min(n_match, len(jd_skills_used)))

    # Add some random skills from pool (including mismatched ones)
    all_skills = archetype["resume_skills_pool"]
    extra = random.sample(all_skills, min(random.randint(2, 5), len(all_skills)))
    resume_skills = list(set(matched_skills + extra))
    random.shuffle(resume_skills)

    resume = f"{name} | {city} | {email} | {phone}\n\n"

    if is_fresher:
        years = 0
        resume += "Professional Summary: "
        summaries = [
            f"Motivated {random.choice(['graduate', 'final-year student', 'fresh graduate'])} "
            f"seeking entry-level opportunities. ",
            f"Recent graduate with training in {', '.join(resume_skills[:3])}. "
            f"Looking for entry-level roles to apply academic knowledge. ",
            f"Enthusiastic fresher with academic projects in {', '.join(resume_skills[:2])}. ",
        ]
        resume += random.choice(summaries)
        resume += f"Skills include {', '.join(resume_skills[:5])}.\n\n"

        resume += "Education:\n"
        degrees = ["B.Tech in Computer Science", "B.Com (Hons)", "BBA in Finance",
                    "B.Sc in Statistics", "MBA (pursuing)", "M.Tech in Data Science",
                    "B.E. in Information Technology", "CA (Intermediate)"]
        resume += f"- {random.choice(degrees)}, {random.choice(['2023', '2024', '2025'])}\n"
        college = random.choice(["University of Delhi", "IIT Bombay", "BITS Pilani",
                                  "Anna University", "Pune University", "VIT Vellore",
                                  "SRM University", "Amity University", "Christ University"])
        resume += f"  {college}\n\n"

        resume += "Projects / Internship:\n"
        tasks = archetype["fresher_exp_tasks"]
        n_tasks = random.randint(2, min(4, len(tasks)))
        for task in random.sample(tasks, n_tasks):
            skill_for_template = random.choice(resume_skills[:5]) if resume_skills else "Python"
            resume += f"- {task.format(skill=skill_for_template, company=random.choice(COMPANIES))}\n"
    else:
        years = random.randint(2, 15)
        resume += f"Professional Summary: Results-driven professional with {years} years of experience. "
        resume += f"Expertise in {', '.join(resume_skills[:4])}. "
        resume += "Proven track record of delivering impactful solutions.\n\n"

        resume += "Experience:\n"
        n_roles = min(random.randint(1, 3), years // 2 + 1)
        for i in range(n_roles):
            company = random.choice(COMPANIES)
            title = random.choice([
                "Senior Analyst", "Associate", "Manager", "Lead", "Consultant",
                "Engineer", "Developer", "Specialist", "Officer", "Executive",
            ])
            resume += f"\n{title} | {company} | {random.randint(1, 4)} years\n"
            tasks = archetype["resume_exp_tasks"]
            n_tasks = random.randint(2, min(3, len(tasks)))
            for task in random.sample(tasks, n_tasks):
                skill_for_template = random.choice(resume_skills[:5]) if resume_skills else "Python"
                resume += f"- {task.format(skill=skill_for_template, company=company)}\n"

        resume += "\nEducation:\n"
        degrees = ["B.Tech in Computer Science", "B.Com", "MBA in Finance", "CA",
                    "M.Sc in Statistics", "B.E. in IT", "BBA", "M.Tech in CS"]
        year = random.randint(2008, 2022)
        resume += f"- {random.choice(degrees)}, {year}\n"

    resume += f"\nSkills: {', '.join(resume_skills)}\n"
    return resume


def compute_score(is_fresher: bool, match_level: str) -> float:
    """Compute a score that respects the fairness correction rules."""
    if is_fresher:
        ranges = {
            "strong": (60, 85),
            "moderate": (35, 60),
            "weak": (15, 35),
            "mismatched": (5, 20),
        }
    else:
        ranges = {
            "strong": (70, 95),
            "moderate": (45, 70),
            "weak": (20, 45),
            "mismatched": (5, 25),
        }
    lo, hi = ranges[match_level]
    return round(random.uniform(lo, hi), 1)


def generate_domain_pairs(
    archetypes: dict,
    total_count: int,
    domain_label: int,
    fresher_fraction: float,
    output_path: Path,
):
    """Generate synthetic pairs for a domain."""
    pairs_per_archetype = total_count // len(archetypes)
    remainder = total_count % len(archetypes)

    # Score distribution targets: 15% Excellent, 25% Good, 30% Moderate, 20% Weak, 10% Poor
    # Map these to match levels that will produce scores in those ranges
    # Strong match -> Excellent/Good scores
    # Moderate match -> Good/Moderate scores
    # Weak match -> Weak scores
    # Mismatched -> Poor scores
    match_level_weights = ["strong"] * 25 + ["moderate"] * 35 + ["weak"] * 25 + ["mismatched"] * 15

    rows = []
    archetype_names = list(archetypes.keys())

    for i, arch_name in enumerate(archetype_names):
        arch = archetypes[arch_name]
        n_pairs = pairs_per_archetype + (1 if i < remainder else 0)
        n_freshers = int(n_pairs * fresher_fraction)

        for j in range(n_pairs):
            is_fresh = j < n_freshers
            match_level = random.choice(match_level_weights)

            jd_text = generate_jd(arch_name, arch)
            # Extract which skills were used in this JD
            jd_lower = jd_text.lower()
            jd_skills_used = [s for s in arch["jd_skills"] if s.lower() in jd_lower]

            resume_text = generate_resume(arch, is_fresh, match_level, jd_skills_used)
            score = compute_score(is_fresh, match_level)

            rows.append({
                "resume_text": resume_text,
                "jd_text": jd_text,
                "ats_score": score,
                "domain_label": domain_label,
            })

    random.shuffle(rows)
    df = pd.DataFrame(rows)

    # Verify score distribution
    print(f"\n  Generated {len(df)} pairs for domain {domain_label}")
    bins = [(85, 100, "Excellent"), (65, 84, "Good"), (45, 64, "Moderate"),
            (25, 44, "Weak"), (0, 24, "Poor")]
    for lo, hi, label in bins:
        count = ((df["ats_score"] >= lo) & (df["ats_score"] <= hi)).sum()
        print(f"    {label:12s} ({lo:3d}-{hi:3d}): {count} ({count/len(df)*100:.1f}%)")

    # Verify fresher stats
    fresher_pattern = r"fresher|entry.level|fresh.graduate|intern|final.year|looking for entry"
    df_freshers = df[df["resume_text"].str.contains(fresher_pattern, case=False, na=False)]
    df_exp = df[~df.index.isin(df_freshers.index)]
    print(f"    Freshers: {len(df_freshers)} ({len(df_freshers)/len(df)*100:.1f}%)")
    print(f"    Fresher mean score:     {df_freshers['ats_score'].mean():.1f}")
    print(f"    Experienced mean score: {df_exp['ats_score'].mean():.1f}")
    print(f"    Gap (exp - fresher):    {df_exp['ats_score'].mean() - df_freshers['ats_score'].mean():.1f}")

    df.to_csv(output_path, index=False)
    print(f"    Saved to: {output_path}")
    return df


def main():
    print("=" * 55)
    print("  PHASE 2 — Generating IT + Finance Supplemental Pairs")
    print("=" * 55)

    # Generate IT supplemental
    print("\n--- IT / Software (400 pairs, domain=0) ---")
    generate_domain_pairs(
        archetypes=IT_ARCHETYPES,
        total_count=400,
        domain_label=0,
        fresher_fraction=0.25,
        output_path=IT_CSV,
    )

    # Generate Finance supplemental
    print("\n--- Finance / Banking (400 pairs, domain=4) ---")
    generate_domain_pairs(
        archetypes=FINANCE_ARCHETYPES,
        total_count=400,
        domain_label=4,
        fresher_fraction=0.25,
        output_path=FINANCE_CSV,
    )

    print("\n" + "=" * 55)
    print("  Phase 2 supplemental generation complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
