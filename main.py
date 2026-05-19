import uvicorn
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware

# Імпортуємо твою функцію з файлу ai_coach.py
from ai_coach import run_full_pipeline

app = FastAPI(title="AI Resume Coach API")

# Налаштування CORS, щоб сайт з wuaze міг достукатися до Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "working", "message": "AI Resume Coach API is online!"}


@app.post("/api/analyze")
async def analyze_resume(
        job_description: str = Form(...),
        cv_file: UploadFile = File(...)
):
    try:
        # Зчитуємо файл у байти
        pdf_bytes = await cv_file.read()
        final_output = run_full_pipeline(job_description, pdf_bytes)

        return {
            "status": "success",
            "results": final_output
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Internal error: {str(e)}"
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)