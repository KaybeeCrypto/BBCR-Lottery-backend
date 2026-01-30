from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from .database import engine
from .database import SessionLocal
from .database import get_db

from .models import Base
from .models import Round

from sqlalchemy.orm import Session
from .schemas import RoundOut
from typing import Optional
from .schemas import RoundCreate, RoundOut




app = FastAPI(title="Lottery Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kaybeecrypto.github.io",  
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        existing_round = db.query(Round).first()
        if existing_round is None:
            first_round = Round(status="open")
            db.add(first_round)
            db.commit()
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rounds", response_model=RoundOut)
def create_round(payload: RoundCreate, db: Session = Depends(get_db)):
    new_round = Round(status=payload.status)
    db.add(new_round)
    db.commit()
    db.refresh(new_round)  # reload from DB so id/created_at are filled in
    return new_round

@app.get("/rounds/current")
def get_current_round(db: Session = Depends(get_db)):
    round_obj = (
        db.query(Round)
        .order_by(Round.created_at.desc())
        .first()
    )

    if round_obj is None:
        return {"round": None}

    return {"round": RoundOut.model_validate(round_obj)}