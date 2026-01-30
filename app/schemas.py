from pydantic import BaseModel
from typing import Optional
from datetime import datetime



class RoundCreate(BaseModel):
    status: str = "open"
    
class RoundOut(BaseModel):
    round_id: str
    status: str
    snapshot_hash: Optional[str] = None
    target_slot: Optional[int] = None
    blockhash: Optional[str] = None
    winner_wallet: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
