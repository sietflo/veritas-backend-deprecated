import io
import re
import json
import joblib
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity

#"Хардкодинг" скілів

SKILLS = [
    # Languages
    "python", "javascript", "typescript", "java", "c++", 
    "c#", "ruby", "php", "go", "rust", 
    "swift", "kotlin", "scala", "r", "perl",
    
    # Web Development (Frontend & Backend)
    "html5", "css3", "sass", "react", "angular", 
    "vue.js", "next.js", "svelte", "nodejs", "express.js", 
    "django", "flask", "fastapi", "spring boot", "laravel", 
    "asp.net", "graphql", "rest api", "webassembly",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", 
    "jenkins", "github actions", "gitlab ci", "terraform", "ansible", 
    "puppet", "chef", "linux", "bash", "powershell",
    
    # Databases & Data Storage
    "mysql", "postgresql", "mongodb", "redis", "sqlite", 
    "oracle", "microsoft sql server", "cassandra", "dynamodb", "neo4j", 
    "elasticsearch", "firebase",
    
    # Data Science, AI & Machine Learning
    "machine learning", "deep learning", "artificial intelligence", "data analytics", "pandas", 
    "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", 
    "apache spark", "hadoop", "tableau", "power bi",
    
    # Mobile & Desktop Development
    "flutter", "react native", "xamarin", "ionic", "electron",
    
    # Cybersecurity & Networking
    "cybersecurity", "penetration testing", "ethical hacking", "wireshark", "splunk", 
    "firewalls", "cryptography", "network security", "tcp/ip", "dns",
    
    # Methodologies, Tools & Practices
    "git", "github", "gitlab", "bitbucket", "agile", 
    "scrum", "kanban", "ci/cd", "devops", "devsecops", 
    "test-driven development", "unit testing", "jira", "confluence",
    
    # Systems, Architecture & UI/UX
    "microservices", "serverless", "system design", "ui/ux design", "figma", 
    "enterprise architecture", "itil", "project management"
]

# Завантажуємо модель TF-IDF один раз при імпорті модуля
try:
    vectorizer = joblib.load('tf-idf.pkl')
except Exception as e:
    print(f"⚠️ Помилка завантаження tf-idf.pkl: {e}. Переконайтеся, що файл лежить поруч.")
    vectorizer = None


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Приймає сирі байти файлу PDF
    та повертає очищений текст.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    # Чистимо специфічні юнікод-символи
                    page_text = page_text.replace('\u2022', '*').replace('\u2013', '-')
                    full_text.append(page_text)
                else:
                    full_text.append("[No readable text found]")
            return "\n\n".join(full_text)
    except Exception as e:
        print(f"Помилка парсингу PDF: {e}")
        return ""


def extract_skills(text: str) -> set:
    """
    Витягує слова-навички з тексту згідно списку вище
    """
    text = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)
    return set(found)


def skill_score(job_text: str, cv_text: str) -> float:
    """
    Функція для обрахунку коефіцієнта навичок які є у кандидата
    порівняно з вакансією для чеснішого фінального обрахунку фінального score
    """
    job_skills = extract_skills(job_text)
    cv_skills = extract_skills(cv_text)
    if not job_skills:
        return 0.0
    return len(job_skills & cv_skills) / len(job_skills)


def calculate_ats_metrics(job_text: str, cv_pdf_bytes: bytes) -> dict:
    """
    ГОЛОВНА ФУНКЦІЯ: Приймає текст вакансії та байти PDF.
    Повертає Python-словник з усіма розрахунками.
    """
    cv_text = extract_pdf_text_from_bytes(cv_pdf_bytes)

    if not cv_text.strip() or not vectorizer:
        return {
            "job_description": job_text.strip(),
            "cv": cv_text.strip(),
            "ats_score": 0.0,
            "missing_skills": [],
            "warning": "Error processing CV or ML model missing."
        }

    # Розрахунок метрик TF-IDF
    job_vector = vectorizer.transform([job_text])
    cv_vector = vectorizer.transform([cv_text])
    tfidf_score = cosine_similarity(cv_vector, job_vector)[0][0]

    # Розрахунок скілів
    s_score = skill_score(job_text, cv_text)
    final_score = round((tfidf_score * 0.5 + s_score * 0.5) * 100, 2)

    # Виявлення прогалин
    job_skills = extract_skills(job_text)
    cv_skills = extract_skills(cv_text)
    missing = sorted(list(job_skills - cv_skills))  # Перетворюємо в list для сумісності з JSON

    # Логіка перевірки на "keyword matching"
    matched = job_skills & cv_skills
    warn = ""
    if final_score >= 75 and matched == job_skills:
        warn = "High keyword overlap detected: CV appears optimized for ATS, but may lack genuine depth."

    return {
        "job_description": job_text.strip(),
        "cv": cv_text.strip(),
        "ats_score": final_score,
        "missing_skills": missing,
        "warning": warn if warn else "No warnings from the system!"
    }
