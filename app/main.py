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
from datetime import datetime, timedelta
import uuid
import hashlib
import requests
from fastapi import Response
import json
from solana.rpc.api import Client
from solana.transaction import Transaction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta


MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"

def load_authority_keypair() -> Keypair:
    raw = os.getenv("SOLANA_AUTHORITY_KEYPAIR_JSON")
    if not raw:
        raise HTTPException(status_code=500, detail="SOLANA_AUTHORITY_KEYPAIR_JSON not configured")
    try:
        arr = json.loads(raw)
        secret = bytes(arr)
        return Keypair.from_bytes(secret)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid authority keypair JSON: {e}")

def solana_client() -> Client:
    rpc_url = os.getenv("SOLANA_RPC_URL")
    if not rpc_url:
        raise HTTPException(status_code=500, detail="SOLANA_RPC_URL not configured")
    return Client(rpc_url)

def send_memo_tx(payload: dict) -> str:
    """
    Sends a Memo transaction signed by the authority wallet.
    Returns tx signature (string). Fees are deducted from authority wallet automatically.
    """
    kp = load_authority_keypair()
    client = solana_client()

    # Compact JSON to keep memo small
    memo_str = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    memo_bytes = memo_str.encode("utf-8")

    program_id = Pubkey.from_string(MEMO_PROGRAM_ID)

    # Memo instruction: no accounts required
    ix = Instruction(
        program_id=program_id,
        accounts=[],
        data=memo_bytes
    )

    tx = Transaction()
    tx.add(ix)

    try:
        resp = client.send_transaction(tx, kp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to send memo tx: {e}")

    # solana-py response shape varies; handle both
    sig = None
    if isinstance(resp, dict):
        sig = resp.get("result")
    else:
        # Some versions return an object-like
        sig = getattr(resp, "value", None) or getattr(resp, "result", None)

    if not sig:
        raise HTTPException(status_code=502, detail=f"Memo tx send returned no signature: {resp}")

    return sig

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

# --- Helius helpers (DAS getTokenAccounts) ---

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
HELIUS_RPC_URL = "https://mainnet.helius-rpc.com/"

from base58 import b58decode
from nacl.bindings import crypto_core_ed25519_is_valid_point

# Common Solana burn / incinerator addresses (expand as you discover more)
BURN_ADDRESSES = {
    "1nc1nerator11111111111111111111111111111111",  # very common burn wallet
    "11111111111111111111111111111111",             # system program (not usually a burn wallet, but harmless)
}

def is_on_curve(pubkey: str) -> bool:
    """
    Returns True for normal user wallets (on-curve ed25519 pubkeys),
    False for PDAs (off-curve).
    """
    try:
        raw = b58decode(pubkey)
        if len(raw) != 32:
            return False
        return bool(crypto_core_ed25519_is_valid_point(raw))
    except Exception:
        return False

def is_excluded_owner(owner: str) -> tuple[bool, str]:
    """
    Returns (excluded?, reason)
    Reasons: "burn" or "lp_program"
    """
    if owner in BURN_ADDRESSES:
        return True, "burn"
    if not is_on_curve(owner):
        # Off-curve owners are usually PDAs: LP vaults, program treasuries, staking vaults, etc.
        return True, "lp_program"
    return False, ""


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

def iso_z(dt):
    if dt is None:
        return None
    # assumes UTC naive datetimes (your code uses utcnow())
    return dt.replace(microsecond=0).isoformat() + "Z"

@app.get("/api/public/state")
def get_public_state(db: Session = Depends(get_db), response: Response = None):
    config = db.query(AdminConfig).first()
    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig missing")


    # Prevent caching (important for GitHub Pages)
    if response is not None:
        response.headers["Cache-Control"] = "no-store"

    return {
        # ---- existing top-level fields (keep for compatibility) ----
        "round_state": config.round_state,
        "commit_deadline": iso_z(config.commit_deadline),
        "reveal_deadline": iso_z(config.reveal_deadline),
        "winner_wallet": config.winner_wallet,

        # ---- snapshot proof ----
        "snapshot": {
            "snapshot_id": config.snapshot_id,
            "snapshot_slot": config.snapshot_slot,
            "snapshot_root": config.snapshot_root,
            "eligible_holders": config.eligible_holders,
            "snapshot_tx": getattr(config, "snapshot_tx", None),
        },

        # ---- reveal phase (no on-chain anchor yet) ----
        "reveal": {
            "target_slot": config.target_slot,
        },

        # ---- finalize proof ----
        "finalize": {
            "winner_wallet": config.winner_wallet,
            "winner_index": config.winner_index,
            "blockhash": config.blockhash,
            "finalize_tx": getattr(config, "finalize_tx", None),
        },
    }

@app.get("/api/public/snapshot/{snapshot_id}/canonical")
def get_snapshot_canonical(snapshot_id: str, db: Session = Depends(get_db)):
    config = db.query(AdminConfig).first()
    if not config or config.snapshot_id != snapshot_id:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    if not config.eligible_canonical:
        raise HTTPException(status_code=404, detail="Canonical snapshot missing")

    return {
        "snapshot_id": snapshot_id,
        "snapshot_root": config.snapshot_root,
        "canonical": config.eligible_canonical,
    }


@app.get("/api/admin/state")
def get_admin_state(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()
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

@app.get("/api/admin/authority/balance")
def authority_balance(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    kp = load_authority_keypair()
    client = solana_client()
    resp = client.get_balance(kp.pubkey())
    lamports = resp["result"]["value"]
    return {
        "authority_pubkey": str(kp.pubkey()),
        "lamports": lamports,
        "sol": lamports / 1_000_000_000,
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
    excluded_lp = 0
    eligible = 0
    min_hold = int(config.min_hold_amount)

    for owner, bal in balances.items():
        excluded, reason = is_excluded_owner(owner)
        if excluded:
            if reason == "burn":
                excluded_burn += 1
            elif reason == "lp_program":
                excluded_lp += 1
            continue

        if bal >= min_hold:
            eligible += 1

    return {
        "token": config.mint_address,
        "min_hold_amount": min_hold,
        "total_holders": total_holders,
        "eligible_holders": eligible,
        "excluded": {
            "lp_accounts": excluded_lp,
            "burn_addresses": excluded_burn
        },
        "last_indexed_slot": last_slot,
        "preview_time": datetime.utcnow().isoformat(),
    }


@app.post("/api/admin/snapshot")
def take_snapshot(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()

    if config.round_state != "IDLE":
        raise HTTPException(status_code=400, detail="Snapshot already taken or round already started")

    if not config.mint_address or config.min_hold_amount is None:
        raise HTTPException(status_code=400, detail="Token config not set")

    snapshot_id = str(uuid.uuid4())
    snapshot_time = datetime.utcnow()

    last_slot, token_accounts = helius_get_token_accounts_all(config.mint_address, limit=1000)
    balances = aggregate_balances_by_owner(token_accounts)

    eligible = []
    min_hold = int(config.min_hold_amount)

    for owner, bal in balances.items():
        excluded, _reason = is_excluded_owner(owner)
        if excluded:
            continue
        if bal >= min_hold:
            eligible.append((owner, bal))


    canonical = build_canonical(eligible)
    snapshot_root = sha256_hex(canonical)
    eligible_holders = len(eligible)

    snapshot_slot = int(last_slot) if last_slot is not None else 0

    config.snapshot_id = snapshot_id
    config.snapshot_time = snapshot_time
    config.snapshot_slot = snapshot_slot
    config.eligible_holders = eligible_holders
    config.eligible_canonical = canonical
    config.snapshot_root = snapshot_root
    config.round_state = "SNAPSHOT_TAKEN"
    config.snapshot_tx_sig = snapshot_tx_sig
    config.snapshot_tx_sig = snapshot_tx_sig

    # Optional: store authority pubkey once
    if not config.authority_pubkey:
        config.authority_pubkey = str(load_authority_keypair().pubkey())

        # Anchor snapshot on-chain (memo tx)
    snap_payload = {
        "p": "commit-lottery-v1",
        "t": "snapshot",
        "snapshot_id": snapshot_id,
        "mint": config.mint_address,
        "min_hold": str(min_hold),
        "last_indexed_slot": str(snapshot_slot),
        "snapshot_root": snapshot_root,
    }
    snapshot_tx_sig = send_memo_tx(snap_payload)

    db.commit()

    return {
        "snapshot_id": snapshot_id,
        "snapshot_time": snapshot_time.isoformat(),
        "snapshot_slot": snapshot_slot,
        "eligible_holders": eligible_holders,
        "snapshot_root": snapshot_root,
        "state": config.round_state,
    }

@app.post("/api/admin/commit/start")
def start_commit_phase(commit_minutes: int = 30, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()
    if config.round_state != "SNAPSHOT_TAKEN":
        raise HTTPException(status_code=400, detail="Cannot start commit phase in current state")

    config.commit_deadline = datetime.utcnow() + timedelta(minutes=commit_minutes)
    config.round_state = "COMMIT"
    db.commit()

    return {
        "state": config.round_state,
        "commit_deadline": config.commit_deadline.replace(microsecond=0).isoformat() + "Z"
,
    }

@app.post("/api/admin/reveal/start")
def start_reveal_phase(reveal_minutes: int = 15, db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()

    if config.round_state != "COMMIT":
        raise HTTPException(status_code=400, detail="Cannot start reveal phase in current state")

    if config.commit_deadline is None or datetime.utcnow() < config.commit_deadline:
        raise HTTPException(status_code=400, detail="Commit deadline not reached")

    current_slot = helius_get_current_slot()
    slot_offset = 200  # ~ about 1–2 minutes on Solana; safe buffer
    config.target_slot = current_slot + slot_offset
    
    reveal_payload = {
        "p": "commit-lottery-v1",
        "t": "reveal_start",
        "snapshot_id": config.snapshot_id,
        "snapshot_root": config.snapshot_root,
        "target_slot": str(config.target_slot),
    }
    reveal_tx_sig = send_memo_tx(reveal_payload)
    config.reveal_tx_sig = reveal_tx_sig

    config.reveal_deadline = datetime.utcnow() + timedelta(minutes=reveal_minutes)
    config.round_state = "REVEAL"
    db.commit()

    return {
        "state": config.round_state,
        "target_slot": config.target_slot,
        "reveal_deadline": config.reveal_deadline.isoformat(),
        "state": config.round_state,
        "target_slot": config.target_slot,
        "reveal_deadline": config.reveal_deadline.isoformat(),
        "reveal_tx_sig": reveal_tx_sig,
    }

@app.post("/api/admin/finalize")
def finalize_winner(db: Session = Depends(get_db), _: None = Depends(require_admin)):
    config = db.query(AdminConfig).first()

    if config.round_state != "REVEAL":
        raise HTTPException(status_code=400, detail="Cannot finalize in current state")

    if datetime.utcnow() < config.reveal_deadline:
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

    finalize_payload = {
        "p": "commit-lottery-v1",
        "t": "finalize",
        "snapshot_id": config.snapshot_id,
        "snapshot_root": config.snapshot_root,
        "target_slot": str(config.target_slot),
        "blockhash": blockhash,
        "winner_index": int(winner_index),
        "winner_wallet": winner_wallet,
    }
    finalize_tx_sig = send_memo_tx(finalize_payload)
    config.finalize_tx_sig = finalize_tx_sig


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
