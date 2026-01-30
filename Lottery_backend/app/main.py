from fastapi import FastAPI
from .database import engine
from .models import Base
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lottery Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kaybeecrypto.github.io",  # replace with your GitHub Pages domain root
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}
