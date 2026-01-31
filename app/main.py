from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import engine, SessionLocal, get_db
from .models import Base, Round, AdminConfig
from .schemas import RoundCreate, RoundOut
from .schemas import TokenConfigIn
from fastapi import Header, HTTPException
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
import uuid
app = FastAPI(title="Lottery Backend")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kaybeecrypto.github.io",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Runs once when the server starts
@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        # Ensure at least one round exists
        existing_round = db.query(Round).first()
        if existing_round is None:
            first_round = Round(status="open")
            db.add(first_round)

        # Ensure admin config exists (exactly one)
        admin_config = db.query(AdminConfig).first()
        if admin_config is None:
            admin_config = AdminConfig(round_state="IDLE")
            db.add(admin_config)

        db.commit()
    finally:
        db.close()

@app.get("/admin")
def admin_page():
    return FileResponse("static/admin.html")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rounds", response_model=RoundOut)
def create_round(payload: RoundCreate, db: Session = Depends(get_db)):
    new_round = Round(status=payload.status)
    db.add(new_round)
    db.commit()
    db.refresh(new_round)
    return new_round


@app.get("/rounds/current")
def get_current_round(db: Session = Depends(get_db)):
    # Treat the newest round as the current round
    round_obj = db.query(Round).order_by(Round.id.desc()).first()

    if round_obj is None:
        return {"round": None}

    return {"round": RoundOut.model_validate(round_obj)}

def require_admin(x_admin_secret: str = Header(None)):
    expected = os.getenv("ADMIN_SECRET")

    if expected is None:
        raise HTTPException(status_code=500, detail="Admin secret not configured")

    if x_admin_secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.get("/api/admin/state")
def get_admin_state(
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):

    config = db.query(AdminConfig).first()

    if config is None:
        return {"error": "Admin config not initialized"}

    return {
        "token": {
            "mint_address": config.mint_address,
            "min_hold_amount": config.min_hold_amount,
        },
        "snapshot": {
            "snapshot_id": config.snapshot_id,
            "snapshot_time": config.snapshot_time,
            "snapshot_slot": config.snapshot_slot,
            "eligible_holders": config.eligible_holders,
        },
        "round": {
            "state": config.round_state,
            "commit_deadline": config.commit_deadline,
            "reveal_deadline": config.reveal_deadline,
        },
    }
@app.post("/api/admin/token")
def save_token_config(payload: TokenConfigIn, db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    config = db.query(AdminConfig).first()

    if config is None:
        return {"error": "Admin config not initialized"}

    # Save token configuration
    config.mint_address = payload.mint_address
    config.min_hold_amount = payload.min_hold_amount

    db.commit()

    return {
        "message": "Token configuration saved",
        "mint_address": config.mint_address,
        "min_hold_amount": config.min_hold_amount,
    }

@app.post("/api/admin/holders/preview")
def preview_holders(
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    config = db.query(AdminConfig).first()

    if config is None:
        raise HTTPException(status_code=500, detail="Admin config missing")

    if not config.mint_address or not config.min_hold_amount:
        raise HTTPException(
            status_code=400,
            detail="Token config not set"
        )

    # ---- MOCK DATA (will be replaced with Helius later) ----
    total_holders = 12482
    eligible_holders = 3194
    excluded_lp = 3
    excluded_burn = 1
    # -------------------------------------------------------

    return {
        "token": config.mint_address,
        "min_hold_amount": config.min_hold_amount,
        "total_holders": total_holders,
        "eligible_holders": eligible_holders,
        "excluded": {
            "lp_accounts": excluded_lp,
            "burn_addresses": excluded_burn
        },
        "preview_time": datetime.utcnow().isoformat()
    }
@app.post("/api/admin/snapshot")
def take_snapshot(
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    config = db.query(AdminConfig).first()

    if config is None:
        raise HTTPException(
            status_code=500,
            detail="Admin config missing"
        )

    # Enforce one-time snapshot
    if config.round_state != "IDLE":
        raise HTTPException(
            status_code=400,
            detail="Snapshot already taken or round already started"
        )

    # Token config must exist
    if not config.mint_address or not config.min_hold_amount:
        raise HTTPException(
            status_code=400,
            detail="Token config not set"
        )

    # ---- SNAPSHOT DATA (mocked for now) ----
    snapshot_id = str(uuid.uuid4())
    snapshot_time = datetime.utcnow()
    snapshot_slot = 123456789  # placeholder
    eligible_holders = config.eligible_holders or 0
    # ---------------------------------------

    # Persist snapshot
    config.snapshot_id = snapshot_id
    config.snapshot_time = snapshot_time
    config.snapshot_slot = snapshot_slot
    config.eligible_holders = eligible_holders
    config.round_state = "SNAPSHOT_TAKEN"

    db.commit()

    return {
        "snapshot_id": snapshot_id,
        "snapshot_time": snapshot_time.isoformat(),
        "snapshot_slot": snapshot_slot,
        "eligible_holders": eligible_holders,
        "state": config.round_state
    }
