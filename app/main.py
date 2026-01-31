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
from datetime import datetime, timedelta, timezone
import uuid
import hashlib
import requests

# --- Protocol v1 helpers ---

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def build_canonical(holders):
    holders_sorted = sorted(holders, key=lambda x: x[0])
    return "\n".join(f"{w}:{int(b)}" for w, b in holders_sorted)

def parse_canonical_wallets(canonical: str):
    if not canonical:
        return []
    wallets = []
    for line in canonical.split("\n"):
        if not line.strip():
            continue
        wallet, _ = line.split(":", 1)
        wallets.append(wallet)
    return wallets

# --- UX / API time helpers (important for reliable JS timers) ---

def utcnow() -> datetime:
    # Keep using naive UTC consistently in DB operations (matches your existing code),
    # but always format outbound timestamps as "YYYY-MM-DDTHH:MM:SSZ" to be JS-safe.
    return datetime.utcnow()

def iso_utc_z(dt: datetime):
    if not dt:
        return None
    # Ensure no microseconds + explicit UTC marker.
    # This avoids JS Date parsing bugs and keeps timers reliable.
    return dt.replace(microsecond=0).isoformat() + "Z"

def pretty_utc(dt: datetime):
    if not dt:
        return None
    # Human-readable, audit-friendly string (UTC).
    # Example: "2026-01-31 19:05:49 UTC"
    return dt.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S UTC")

# Minimal safety to prevent "instant rounds" by accident
MIN_PHASE_SECONDS = 10

# --- Helius helpers (DAS getTokenAccounts) ---

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
HELIUS_RPC_URL = "https://mainnet.helius-rpc.com/"

# Basic burn address exclusions (can expand later)
BURN_ADDRESSES = {
    "11111111111111111111111111111111",
}

def helius_get_current_slot() -> int:
    """
    Fetch the current Solana slot using Helius RPC (getSlot).
    """
    if not HELIUS_API_KEY:
        raise HTTPException(status_code=500, detail="HELIUS_API_KEY not configured")

    payload = {
        "jsonrpc": "2.0",
        "id": "slot",
        "method": "getSlot",
        "params": []
    }

    try:
        r = requests.post(
            f"{HELIUS_RPC_URL}?api-key={HELIUS_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Helius RPC failed: {str(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Helius error {r.status_code}: {r.text}")

    data = r.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"Helius RPC error: {data['error']}")

    result = data.get("result")
    if result is None:
        raise HTTPException(status_code=500, detail="Helius returned no slot")

    return int(result)


def helius_get_blockhash_at_slot(slot: int) -> str:
    """
    Fetch the real Solana blockhash for a given slot using Helius RPC.
    """
    if not HELIUS_API_KEY:
        raise HTTPException(status_code=500, detail="HELIUS_API_KEY not configured")

    payload = {
        "jsonrpc": "2.0",
        "id": "blockhash",
        "method": "getBlock",
        "params": [
            slot,
            {
                "encoding": "json",
                "transactionDetails": "none",
                "rewards": False
            }
        ]
    }

    try:
        r = requests.post(
            f"{HELIUS_RPC_URL}?api-key={HELIUS_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Helius RPC failed: {str(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Helius error {r.status_code}: {r.text}")

    data = r.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"Helius RPC error: {data['error']}")

    result = data.get("result")
    if not result or "blockhash" not in result:
        raise HTTPException(
            status_code=500,
            detail=f"No blockhash available for slot {slot}"
        )

    return result["blockhash"]

def helius_get_token_accounts_all(mint: str, limit: int = 1000):
    """
    Fetch ALL token accounts for a given mint using Helius DAS getTokenAccounts with cursor pagination.
    Returns: (last_indexed_slot, token_accounts_list)
    token_accounts_list items look like:
      { address, mint, owner, amount, delegated_amount, frozen, burnt }
    """
    if not HELIUS_API_KEY:
        raise HTTPException(status_code=500, detail="HELIUS_API_KEY not configured")

    cursor = None
    all_accounts = []
    last_indexed_slot = None

    while True:
        params = {"mint": mint, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "getTokenAccounts",
            "params": params,
        }

        try:
            r = requests.post(
                f"{HELIUS_RPC_URL}?api-key={HELIUS_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Helius request failed: {str(e)}")

        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Helius error {r.status_code}: {r.text}")

        data = r.json()
        if "error" in data:
            raise HTTPException(status_code=502, detail=f"Helius RPC error: {data['error']}")

        result = data.get("result") or {}
        if last_indexed_slot is None:
            last_indexed_slot = result.get("last_indexed_slot")

        token_accounts = result.get("token_accounts") or []
        all_accounts.extend(token_accounts)

        cursor = result.get("cursor")
        if not cursor:
            break

        # End-of-pagination safety: if API returns no accounts, stop
        if len(token_accounts) == 0:
            break

    return last_indexed_slot, all_accounts

def aggregate_balances_by_owner(token_accounts):
    """
    Convert token_accounts -> { owner_wallet: total_amount_int }
    """
    balances = {}
    for ta in token_accounts:
        owner = ta.get("owner")
        amount = ta.get("amount", 0)
        if not owner:
            continue
        try:
            amt_int = int(amount)
        except (ValueError, TypeError):
            continue
        if amt_int <= 0:
            continue
        balances[owner] = balances.get(owner, 0) + amt_int
    return balances


app = FastAPI(title="Lottery Backend")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kaybeecrypto.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        if db.query(Round).first() is None:
            db.add(Round(status="open"))

        if db.query(AdminConfig).first() is None:
            db.add(AdminConfig(round_state="IDLE"))

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
def get_admin_state(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()
    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig row missing in DB")

    # UX: always return JS-safe timestamps + human-readable UTC strings
    return {
        "token": {
            "mint_address": config.mint_address,
            "min_hold_amount": config.min_hold_amount,
        },
        "snapshot": {
            "snapshot_id": config.snapshot_id,
            "snapshot_time": iso_utc_z(config.snapshot_time),
            "snapshot_time_utc": pretty_utc(config.snapshot_time),
            "snapshot_slot": config.snapshot_slot,
            "eligible_holders": config.eligible_holders,
        },
        "round": {
            "state": config.round_state,
            "commit_deadline": iso_utc_z(config.commit_deadline),
            "commit_deadline_utc": pretty_utc(config.commit_deadline),
            "reveal_deadline": iso_utc_z(config.reveal_deadline),
            "reveal_deadline_utc": pretty_utc(config.reveal_deadline),
            # Helpful for UI / audits
            "target_slot": getattr(config, "target_slot", None),
            "winner_wallet": getattr(config, "winner_wallet", None),
        },
        "server_time_utc": pretty_utc(utcnow()),
    }

@app.post("/api/admin/token")
def save_token_config(payload: TokenConfigIn, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()
    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig row missing in DB")

    config.mint_address = payload.mint_address
    config.min_hold_amount = payload.min_hold_amount
    db.commit()
    db.refresh(config)

    return {
        "message": "Token configuration saved",
        "mint_address": config.mint_address,
        "min_hold_amount": config.min_hold_amount,
    }


@app.post("/api/admin/holders/preview")
def preview_holders(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()
    if not config or not config.mint_address or config.min_hold_amount is None:
        raise HTTPException(status_code=400, detail="Token config not set")

    last_slot, token_accounts = helius_get_token_accounts_all(config.mint_address, limit=1000)
    balances = aggregate_balances_by_owner(token_accounts)

    total_holders = len(balances)
    excluded_burn = 0
    eligible = 0

    for owner, bal in balances.items():
        if owner in BURN_ADDRESSES:
            excluded_burn += 1
            continue
        if bal >= int(config.min_hold_amount):
            eligible += 1

    # NOTE: LP/program-vault exclusion can be added later once you decide a concrete rule.
    excluded_lp = 0

    return {
        "token": config.mint_address,
        "min_hold_amount": int(config.min_hold_amount),
        "total_holders": total_holders,
        "eligible_holders": eligible,
        "excluded": {
            "lp_accounts": excluded_lp,
            "burn_addresses": excluded_burn
        },
        "last_indexed_slot": last_slot,
        "preview_time": iso_utc_z(utcnow()),
        "preview_time_utc": pretty_utc(utcnow()),
    }

@app.post("/api/admin/snapshot")
def take_snapshot(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()

    if config.round_state != "IDLE":
        raise HTTPException(status_code=400, detail="Snapshot already taken or round already started")

    if not config.mint_address or config.min_hold_amount is None:
        raise HTTPException(status_code=400, detail="Token config not set")

    snapshot_id = str(uuid.uuid4())
    snapshot_time = utcnow()

    # Pull real on-chain token accounts from Helius
    last_slot, token_accounts = helius_get_token_accounts_all(config.mint_address, limit=1000)
    balances = aggregate_balances_by_owner(token_accounts)

    # Build eligible list (Option A: one wallet = one ticket)
    eligible = []
    min_hold = int(config.min_hold_amount)

    for owner, bal in balances.items():
        if owner in BURN_ADDRESSES:
            continue
        if bal >= min_hold:
            eligible.append((owner, bal))

    # Deterministic snapshot commitment
    canonical = build_canonical(eligible)
    snapshot_root = sha256_hex(canonical)
    eligible_holders = len(eligible)

    # Use Helius last_indexed_slot as a conservative snapshot_slot
    snapshot_slot = int(last_slot) if last_slot is not None else 0

    config.snapshot_id = snapshot_id
    config.snapshot_time = snapshot_time
    config.snapshot_slot = snapshot_slot
    config.eligible_holders = eligible_holders
    config.eligible_canonical = canonical
    config.snapshot_root = snapshot_root
    config.round_state = "SNAPSHOT_TAKEN"

    db.commit()

    return {
        "snapshot_id": snapshot_id,
        "snapshot_time": iso_utc_z(snapshot_time),
        "snapshot_time_utc": pretty_utc(snapshot_time),
        "snapshot_slot": snapshot_slot,
        "eligible_holders": eligible_holders,
        "snapshot_root": snapshot_root,
        "state": config.round_state,
    }

# --- Commit / Reveal: updated to support seconds + safe timestamps + UX guardrails ---

@app.post("/api/admin/commit/start")
def start_commit_phase(
    commit_seconds: int = 1800,  # default 30 minutes
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    config = db.query(AdminConfig).first()
    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig row missing in DB")

    if config.round_state != "SNAPSHOT_TAKEN":
        raise HTTPException(status_code=400, detail="Cannot start commit phase in current state")

    # UX guardrail: prevent accidental 0-second or ultra-short rounds
    if commit_seconds is None or int(commit_seconds) < MIN_PHASE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"commit_seconds must be >= {MIN_PHASE_SECONDS}"
        )

    config.commit_deadline = utcnow() + timedelta(seconds=int(commit_seconds))
    config.round_state = "COMMIT"
    db.commit()

    return {
        "state": config.round_state,
        "commit_deadline": iso_utc_z(config.commit_deadline),
        "commit_deadline_utc": pretty_utc(config.commit_deadline),
        "commit_seconds": int(commit_seconds),
    }

@app.post("/api/admin/reveal/start")
def start_reveal_phase(
    reveal_seconds: int = 900,  # default 15 minutes
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    config = db.query(AdminConfig).first()
    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig row missing in DB")

    if config.round_state != "COMMIT":
        raise HTTPException(status_code=400, detail="Cannot start reveal phase in current state")

    if config.commit_deadline is None or utcnow() < config.commit_deadline:
        raise HTTPException(status_code=400, detail="Commit deadline not reached")

    # UX guardrail: prevent accidental instant reveal window
    if reveal_seconds is None or int(reveal_seconds) < MIN_PHASE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"reveal_seconds must be >= {MIN_PHASE_SECONDS}"
        )

    current_slot = helius_get_current_slot()
    slot_offset = 200  # ~ about 1–2 minutes on Solana; safe buffer
    config.target_slot = current_slot + slot_offset

    config.reveal_deadline = utcnow() + timedelta(seconds=int(reveal_seconds))
    config.round_state = "REVEAL"
    db.commit()

    return {
        "state": config.round_state,
        "target_slot": config.target_slot,
        "reveal_deadline": iso_utc_z(config.reveal_deadline),
        "reveal_deadline_utc": pretty_utc(config.reveal_deadline),
        "reveal_seconds": int(reveal_seconds),
    }

@app.post("/api/admin/finalize")
def finalize_winner(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()
    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig row missing in DB")

    if config.round_state != "REVEAL":
        raise HTTPException(status_code=400, detail="Cannot finalize in current state")

    if config.reveal_deadline is None:
        raise HTTPException(status_code=400, detail="Reveal deadline not set")

    if utcnow() < config.reveal_deadline:
        raise HTTPException(status_code=400, detail="Reveal deadline not reached")

    if config.winner_wallet:
        raise HTTPException(status_code=400, detail="Winner already finalized")

    current_slot = helius_get_current_slot()
    if config.target_slot is None:
        raise HTTPException(status_code=400, detail="Target slot not set")

    if current_slot < int(config.target_slot):
        raise HTTPException(
            status_code=400,
            detail=f"Target slot not reached (current_slot={current_slot}, target_slot={config.target_slot})"
        )

    blockhash = helius_get_blockhash_at_slot(config.target_slot)

    seed = f"{blockhash}|{config.snapshot_root}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    number = int(digest, 16)

    eligible_wallets = parse_canonical_wallets(config.eligible_canonical)
    if not eligible_wallets:
        raise HTTPException(status_code=500, detail="No eligible wallets")

    winner_index = number % len(eligible_wallets)
    winner_wallet = eligible_wallets[winner_index]

    config.winner_wallet = winner_wallet
    config.winner_index = winner_index
    config.blockhash = blockhash
    config.round_state = "FINALIZED"

    db.commit()

    return {
        "state": config.round_state,
        "winner_wallet": winner_wallet,
        "blockhash": blockhash,
        "proof": {
            "snapshot_id": config.snapshot_id,
            "snapshot_root": config.snapshot_root,
            "winner_index": winner_index,
            "hash_algorithm": "sha256",
        },
    }

@app.post("/api/admin/round/reset")
def reset_round(
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    config = db.query(AdminConfig).first()

    if not config:
        raise HTTPException(status_code=500, detail="Admin config missing")

    # Clear snapshot + round state
    config.snapshot_id = None
    config.snapshot_time = None
    config.snapshot_slot = None
    config.snapshot_root = None
    config.eligible_canonical = None
    config.eligible_holders = None

    config.commit_deadline = None
    config.reveal_deadline = None
    config.target_slot = None

    config.blockhash = None
    config.winner_wallet = None
    config.winner_index = None

    config.round_state = "IDLE"

    db.commit()

    return {
        "message": "Round reset successfully",
        "round_state": config.round_state
    }
