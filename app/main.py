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
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.instruction import Instruction
from solders.hash import Hash



from base58 import b58decode
from nacl.bindings import crypto_core_ed25519_is_valid_point


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

def get_tx_json(signature: str) -> dict:
    client = solana_client()

    try:
        # jsonParsed makes signer checks easier (accountKeys include signer flags)
        resp = client.get_transaction(signature, encoding="jsonParsed", max_supported_transaction_version=0)
    except TypeError:
        # older solana-py versions may not support max_supported_transaction_version
        resp = client.get_transaction(signature, encoding="jsonParsed")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RPC get_transaction failed: {e}")

    # solana-py shapes vary a bit, handle both common forms
    data = resp if isinstance(resp, dict) else getattr(resp, "value", None) or getattr(resp, "result", None)

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail=f"Unexpected RPC response: {resp}")

    result = data.get("result")
    if not result:
        raise HTTPException(status_code=404, detail=f"Transaction not found on chain: {signature}")

    return result

def tx_signed_by_authority(tx_result: dict, authority_pubkey: str) -> bool:
    """
    Returns True if authority_pubkey is among the required signers.
    Works for jsonParsed where accountKeys are dicts with signer flags,
    and also falls back to header-based signer inference.
    """
    message = (tx_result.get("transaction") or {}).get("message") or {}
    account_keys = message.get("accountKeys") or []

    # jsonParsed: accountKeys are usually [{pubkey, signer, writable}, ...]
    if account_keys and isinstance(account_keys[0], dict):
        for k in account_keys:
            if k.get("pubkey") == authority_pubkey and k.get("signer") is True:
                return True
        return False

    # fallback: plain list of strings + header.numRequiredSignatures
    header = message.get("header") or {}
    n_signers = int(header.get("numRequiredSignatures", 0))
    signer_keys = account_keys[:n_signers] if isinstance(account_keys, list) else []
    return authority_pubkey in signer_keys

def extract_memo_string(tx_result: dict) -> str:
    message = (tx_result.get("transaction") or {}).get("message") or {}
    instructions = message.get("instructions") or []

    for ix in instructions:
        # jsonParsed often has "programId" and "parsed"
        program_id = ix.get("programId")

        # Sometimes "programId" is a dict in jsonParsed
        if isinstance(program_id, dict):
            program_id = program_id.get("toString") or program_id.get("pubkey") or program_id.get("programId")

        if program_id != MEMO_PROGRAM_ID:
            if isinstance(program_id, str):
                pid = program_id
            else:
                pid = str(program_id)

            if pid != MEMO_PROGRAM_ID:
                continue

        # Preferred: parsed memo
        parsed = ix.get("parsed")
        if isinstance(parsed, dict):
            # typical shape: {"type":"memo","info":{"memo":"..."}}
            info = parsed.get("info") or {}
            memo = info.get("memo")
            if isinstance(memo, str) and memo:
                return memo

        # Sometimes parsed is a string
        if isinstance(parsed, str) and parsed:
            return parsed

        # Fallback: decode base58/base64 data field
        data = ix.get("data")
        if isinstance(data, str) and data:
            # RPC "data" for instruction is typically base58 in json encoding.
            # In jsonParsed it can vary. We try base58 first.
            try:
                raw = b58decode(data)
                return raw.decode("utf-8")
            except Exception:
                # if it's not base58, it might be base64
                import base64
                try:
                    raw = base64.b64decode(data)
                    return raw.decode("utf-8")
                except Exception:
                    pass

    raise HTTPException(status_code=404, detail="No memo instruction found in transaction")

def memo_json_from_tx(signature: str) -> dict:
    tx = get_tx_json(signature)
    memo_str = extract_memo_string(tx)

    try:
        payload = json.loads(memo_str)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Memo is not valid JSON: {e}")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Memo JSON is not an object")

    return payload


def memo_matches_expected(memo: dict, expected: dict) -> bool:
    """
    Returns True if memo contains all expected key/value pairs.
    (Memo may include extra keys; that's fine.)
    """
    for k, v in expected.items():
        if str(memo.get(k)) != str(v):
            return False
    return True


def send_memo_tx(payload: dict) -> str:
    kp = load_authority_keypair()
    client = solana_client()

    memo_str = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    memo_bytes = memo_str.encode("utf-8")

    memo_program = Pubkey.from_string(MEMO_PROGRAM_ID)

    ix = Instruction(
        program_id=memo_program,
        accounts=[],
        data=memo_bytes,
    )

    bh_resp = client.get_latest_blockhash()
    if isinstance(bh_resp, dict):
        blockhash_val = bh_resp["result"]["value"]["blockhash"]
    else:
        blockhash_val = bh_resp.value.blockhash

    if isinstance(blockhash_val, Hash):
        recent_blockhash = blockhash_val
    else:
        recent_blockhash = Hash.from_string(str(blockhash_val))

    # IMPORTANT: positional args only
    msg = MessageV0.try_compile(
        kp.pubkey(),
        [ix],
        recent_blockhash,
    )

    tx = VersionedTransaction(msg, [kp])

    try:
        send_resp = client.send_transaction(tx)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to send memo tx: {e}")

    if isinstance(send_resp, dict):
        sig = send_resp.get("result")
    else:
        sig = getattr(send_resp, "value", None) or getattr(send_resp, "result", None)

    if not sig:
        raise HTTPException(status_code=502, detail=f"Memo tx send returned no signature: {send_resp}")

    return str(sig)

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

def recompute_winner(snapshot_root: str, target_slot: int, canonical: str) -> dict:
    """
    Recomputes the winner deterministically.
    Returns { winner_wallet, winner_index, blockhash }
    """

    # Fetch the real blockhash from Solana
    blockhash = helius_get_blockhash_at_slot(target_slot)

    # Recompute the same seed used in finalize
    seed = f"{blockhash}|{snapshot_root}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    number = int(digest, 16)

    wallets = parse_canonical_wallets(canonical)
    if not wallets:
        raise ValueError("No eligible wallets")

    winner_index = number % len(wallets)
    winner_wallet = wallets[winner_index]

    return {
        "winner_wallet": winner_wallet,
        "winner_index": winner_index,
        "blockhash": blockhash,
    }

# --- Helius helpers (DAS getTokenAccounts) ---

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
HELIUS_RPC_URL = "https://mainnet.helius-rpc.com/"

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
        "round_state": config.round_state,
        "commit_deadline": iso_z(config.commit_deadline),
        "reveal_deadline": iso_z(config.reveal_deadline),

        "snapshot": {
            "snapshot_id": config.snapshot_id,
            "snapshot_slot": config.snapshot_slot,
            "snapshot_root": config.snapshot_root,
            "eligible_holders": config.eligible_holders,
            "snapshot_tx_sig": config.snapshot_tx_sig,
        },

        "reveal": {
            "target_slot": config.target_slot,
            "reveal_tx_sig": config.reveal_tx_sig,
        },

        "finalize": {
            "winner_wallet": config.winner_wallet,
            "winner_index": config.winner_index,
            "blockhash": config.blockhash,
            "finalize_tx_sig": config.finalize_tx_sig,
        },

        "authority": {
            "authority_pubkey": config.authority_pubkey,
        }
    }

@app.get("/api/public/verify")
def verify_round(db: Session = Depends(get_db)):
    config = db.query(AdminConfig).first()

    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig missing")

    # Ensure round is finalized
    if config.round_state != "FINALIZED":
        return {
            "valid": False,
            "reason": "Round not finalized yet"
        }

    checks = {}

    # --- 1. Verify snapshot hash ---
    if not config.eligible_canonical:
        checks["snapshot_hash"] = False
    else:
        recomputed_snapshot_root = sha256_hex(config.eligible_canonical)
        checks["snapshot_hash"] = recomputed_snapshot_root == config.snapshot_root


    # --- 2. Verify winner derivation ---
    try:
        recomputed = recompute_winner(
            snapshot_root=config.snapshot_root,
            target_slot=int(config.target_slot),
            canonical=config.eligible_canonical
        )

        checks["winner_wallet"] = recomputed["winner_wallet"] == config.winner_wallet
        checks["winner_index"] = recomputed["winner_index"] == config.winner_index
        checks["blockhash"] = recomputed["blockhash"] == config.blockhash

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "checks": checks
        }

    # --- 3. Verify on-chain memo anchors + signer ---
    onchain = {}

    authority = config.authority_pubkey
    if not authority:
        return {
            "valid": False,
            "reason": "Missing authority_pubkey in DB (cannot verify signer)",
            "checks": checks
        }

    # Snapshot memo
    if not config.snapshot_tx_sig:
        onchain["snapshot_memo"] = False
    else:
        snap_tx = get_tx_json(config.snapshot_tx_sig)
        snap_signed = tx_signed_by_authority(snap_tx, authority)
        snap_memo = memo_json_from_tx(config.snapshot_tx_sig)

        expected_snap = {
            "p": "commit-lottery-v1",
            "t": "snapshot",
            "snapshot_id": config.snapshot_id,
            "snapshot_root": config.snapshot_root,
            "mint": config.mint_address,
            "min_hold": str(int(config.min_hold_amount)) if config.min_hold_amount is not None else None,
            "last_indexed_slot": str(int(config.snapshot_slot)) if config.snapshot_slot is not None else None,
        }

        # remove None expected keys (if any)
        expected_snap = {k: v for k, v in expected_snap.items() if v is not None}

        onchain["snapshot_signed"] = snap_signed
        onchain["snapshot_memo"] = snap_signed and memo_matches_expected(snap_memo, expected_snap)

    # Reveal memo
    if not config.reveal_tx_sig:
        onchain["reveal_memo"] = False
    else:
        rev_tx = get_tx_json(config.reveal_tx_sig)
        rev_signed = tx_signed_by_authority(rev_tx, authority)
        rev_memo = memo_json_from_tx(config.reveal_tx_sig)

        expected_rev = {
            "p": "commit-lottery-v1",
            "t": "reveal_start",
            "snapshot_id": config.snapshot_id,
            "snapshot_root": config.snapshot_root,
            "target_slot": str(int(config.target_slot)) if config.target_slot is not None else None,
        }
        expected_rev = {k: v for k, v in expected_rev.items() if v is not None}

        onchain["reveal_signed"] = rev_signed
        onchain["reveal_memo"] = rev_signed and memo_matches_expected(rev_memo, expected_rev)

    # Finalize memo
    if not config.finalize_tx_sig:
        onchain["finalize_memo"] = False
    else:
        fin_tx = get_tx_json(config.finalize_tx_sig)
        fin_signed = tx_signed_by_authority(fin_tx, authority)
        fin_memo = memo_json_from_tx(config.finalize_tx_sig)

        expected_fin = {
            "p": "commit-lottery-v1",
            "t": "finalize",
            "snapshot_id": config.snapshot_id,
            "snapshot_root": config.snapshot_root,
            "target_slot": str(int(config.target_slot)) if config.target_slot is not None else None,
            "blockhash": config.blockhash,
            "winner_index": str(int(config.winner_index)) if config.winner_index is not None else None,
            "winner_wallet": config.winner_wallet,
        }
        expected_fin = {k: v for k, v in expected_fin.items() if v is not None}

        onchain["finalize_signed"] = fin_signed
        onchain["finalize_memo"] = fin_signed and memo_matches_expected(fin_memo, expected_fin)

    checks["onchain_snapshot"] = bool(onchain.get("snapshot_memo"))
    checks["onchain_reveal"] = bool(onchain.get("reveal_memo"))
    checks["onchain_finalize"] = bool(onchain.get("finalize_memo"))

    # Final verdict
    valid = all(checks.values())

    return {
        "valid": valid,
        "checks": checks,
        "onchain": onchain,
        "computed": recomputed if valid else None,
        "stored": {
            "winner_wallet": config.winner_wallet,
            "winner_index": config.winner_index,
            "blockhash": config.blockhash,
        }
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
    if not config:
        raise HTTPException(status_code=500, detail="AdminConfig missing")

    if config.round_state != "IDLE":
        raise HTTPException(status_code=400, detail="Snapshot already taken or round already started")

    if not config.mint_address or config.min_hold_amount is None:
        raise HTTPException(status_code=400, detail="Token config not set")

    snapshot_id = str(uuid.uuid4())
    snapshot_time = datetime.utcnow()

    # Pull token accounts from Helius
    last_slot, token_accounts = helius_get_token_accounts_all(config.mint_address, limit=1000)
    balances = aggregate_balances_by_owner(token_accounts)

    # Build eligible list
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

    # Anchor snapshot on-chain FIRST
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

    # Persist to DB
    config.snapshot_id = snapshot_id
    config.snapshot_time = snapshot_time
    config.snapshot_slot = snapshot_slot
    config.eligible_holders = eligible_holders
    config.eligible_canonical = canonical
    config.snapshot_root = snapshot_root
    config.snapshot_tx_sig = snapshot_tx_sig
    config.round_state = "SNAPSHOT_TAKEN"

    current_pubkey = str(load_authority_keypair().pubkey())

    if config.authority_pubkey:
        if config.authority_pubkey != current_pubkey:
            raise HTTPException(
                status_code=500,
                detail="Authority pubkey mismatch — DB tampered"
            )
    else:
        config.authority_pubkey = current_pubkey


    db.commit()

    return {
        "snapshot_id": snapshot_id,
        "snapshot_time": snapshot_time.isoformat(),
        "snapshot_slot": snapshot_slot,
        "eligible_holders": eligible_holders,
        "snapshot_root": snapshot_root,
        "snapshot_tx_sig": snapshot_tx_sig,
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