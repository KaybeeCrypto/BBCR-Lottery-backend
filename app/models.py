from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy import Text
from datetime import datetime
from .database import Base


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_id = Column(String, unique=True, index=True)
    status = Column(String)  # scheduled / committed / revealed / paid

    snapshot_hash = Column(String, nullable=True)
    target_slot = Column(Integer, nullable=True)
    blockhash = Column(String, nullable=True)
    winner_wallet = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

class AdminConfig(Base):
    __tablename__ = "admin_config"

    id = Column(Integer, primary_key=True)
    
    # Round info
    snapshot_tx_sig = Column(String, nullable=True)
    reveal_tx_sig = Column(String, nullable=True)
    finalize_tx_sig = Column(String, nullable=True)
    authority_pubkey = Column(String, nullable=True)

    # Token config
    mint_address = Column(String, nullable=True)
    min_hold_amount = Column(Integer, nullable=True)

    # Snapshot info
    snapshot_id = Column(String, nullable=True)
    snapshot_time = Column(DateTime, nullable=True)
    snapshot_slot = Column(Integer, nullable=True)
    eligible_holders = Column(Integer, nullable=True)

    # Lottery state
    round_state = Column(String, default="IDLE")  
    commit_deadline = Column(DateTime, nullable=True)
    reveal_deadline = Column(DateTime, nullable=True)

    # Finalize state
    target_slot = Column(Integer, nullable=True)
    blockhash = Column(String, nullable=True)
    winner_wallet = Column(String, nullable=True)

    

        # --- Protocol v1 additions (deterministic + auditable) ---

    # Cryptographic commitment to snapshot (SHA-256 hex)
    snapshot_root = Column(String, nullable=True)

    # Canonical eligible set (server-side only, not exposed)
    # Format:
    # wallet:balance\nwallet:balance...
    eligible_canonical = Column(Text, nullable=True)

    # Deterministic winner index (for audit/debug)
    winner_index = Column(Integer, nullable=True)


class AdminLog(Base):
    __tablename__ = "admin_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)
    details = Column(String)
