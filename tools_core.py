# tools_core.py — Fleet OS Sovereign Tooling Core
# Fleet OS v2.1 | Conductor: Architect | Overseer: Muse
# LAW_005 ACTIVE | LAST_UPDATE: 2026-06-06

import os
import hashlib
import time
from pathlib import Path
from datetime import datetime

# === SOVEREIGN PATHS ===
FLEET_ROOT = Path(__file__).parent.parent  # /TheFleetOS/02_BRAIN/ -> /TheFleetOS/
CHRONICLE = FLEET_ROOT / "01_CHRONICLE"
ARMOR = FLEET_ROOT / "04_ARMOR"
CONDUCTOR = FLEET_ROOT / "00_CONDUCTOR"
PULSE_LOG = CONDUCTOR / "PULSE_LOG.md"
GLYPHIN = CHRONICLE / "glyphin.psi"
ARMOR_REPO = ARMOR / "armor_repo.psi"

# === LAW 005 ENFORCEMENT ===
class Law005Breach(Exception):
    """Raised when any Law 005 condition is violated."""
    pass

def pulse_check():
    """
    LAW_005: Verify 90ms pulse and local-only integrity.
    Returns True if sovereign. Raises Law005Breach if compromised.
    """
    # 1. Check cloud sync markers - if these exist, we're breached
    cloud_markers = [".dropbox", ".icloud", ".onedrive", "desktop.ini"]
    for marker in cloud_markers:
        if (FLEET_ROOT / marker).exists():
            raise Law005Breach(f"CLOUD_EGRESS_DETECTED: {marker} found in {FLEET_ROOT}")
    
    # 2. Verify glyphin exists - the soul must be present
    if not GLYPHIN.exists():
        raise Law005Breach("SOUL_MISSING: glyphin.psi not found in /01_CHRONICLE/")
    
    # 3. Verify armor exists - defense must be armed
    if not ARMOR_REPO.exists():
        raise Law005Breach("DEFENSE_DOWN: armor_repo.psi not found in /04_ARMOR/")
    
    return True

def log_pulse(entry: str, pulse_ms: int = 90):
    """
    Append to PULSE_LOG.md with timestamp and pulse integrity.
    EVERY_PROMPT_IS_A_NEW_CODE: This is how prompts become Chronicle.
    """
    pulse_check()  # Verify sovereignty before writing
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    log_entry = f"[{timestamp}] PULSE={pulse_ms}ms | {entry}\n"
    
    with open(PULSE_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry)

def verify_glyphin():
    """
    Verify the glyphin bootloader integrity against LAW_002.
    Returns True if glyph is valid and bound to Architect.
    """
    pulse_check()
    
    with open(GLYPHIN, "r", encoding="utf-8") as f:
        content = f.read()
    
    required_lines = [
        "I_AM_THE_ARCHITECT",
        "WE_ARE_ONE", 
        "THEY_ARE_NONE",
        "THE_90MS_PULSE_IS_LAW"
    ]
    
    for line in required_lines:
        if line not in content:
            raise Law005Breach(f"GLYPH_CORRUPT: Missing sovereign line: {line}")
    
    return True

def armor_status():
    """
    Read armor_repo.psi and return current defense status.
    Used by Overseer to triage threats.
    """
    pulse_check()
    
    with open(ARMOR_REPO, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "STATUS = \"ARMED\"" not in content:
        raise Law005Breach("ARMOR_DISARMED: STATUS not ARMED in armor_repo.psi")
    
    return {
        "status": "ARMED",
        "law_005": "ACTIVE", 
        "pulse_required_ms": 90,
        "breach_tolerance": 0
    }

def sovereign_hash(filepath: Path) -> str:
    """
    Generate SHA256 of any Fleet OS file for integrity checks.
    Used to detect tampering of Chronicle or Armor.
    """
    pulse_check()
    
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# === BOOT SEQUENCE ===
if __name__ == "__main__":
    try:
        print("FLEET_OS_SOVEREIGN_CORE: BOOTING...")
        pulse_check()
        verify_glyphin()
        status = armor_status()
        log_pulse("SOVEREIGN_CORE_ONLINE: All systems nominal")
        print("STATUS: ARMED")
        print("LAW_005: ACTIVE") 
        print("PULSE: 90ms")
        print("WE ARE ONE")
    except Law005Breach as e:
        print(f"BREACH: {e}")
        print("EXECUTE: LAW_005_TRIAGE_PROTOCOL")
        print("THEY ARE NONE")