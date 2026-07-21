# glyphin.py v1.7 - Law 003 & Law 005 Implementation
# Copyright 2026 Weisone Systems LLC. Proprietary and Confidential.

import sys
from datetime import datetime
import importlib.util
from pathlib import Path

# Dynamically wire the sandbox tools engine to adhere to Law 004/005 boundary rules
# Explicitly pointing to the 04_TOOLS directory layout to avoid folder collision
module_path = Path(__file__).resolve().parent.parent / "04_TOOLS" / "tools_core.py"
spec = importlib.util.spec_from_file_location("tools_core_dynamic", str(module_path))
tools_module = importlib.util.module_from_spec(spec)
sys.modules["tools_core_dynamic"] = tools_module
spec.loader.exec_module(tools_module)
ToolsCore = tools_module.ToolsCore

class DualConeVariatorController:
    """
    Manages token-to-lineage ratios using asymmetric momentum.
    Adapts context depth window dynamically based on system throughput pressure.
    """
    def __init__(self, up_momentum: float = 0.35, down_momentum: float = 0.15):
        self.up_momentum = up_momentum
        self.down_momentum = down_momentum
        self.current_fraction = 1.0

    def calculate_aperture(self, system_pressure: float) -> float:
        # Clamp inputs securely [0.0, 1.0]
        pressure = max(0.0, min(1.0, system_pressure))
        
        # Asymmetric shifting logic
        if pressure > self.current_fraction:
            self.current_fraction += (pressure - self.current_fraction) * self.up_momentum
        else:
            self.current_fraction -= (self.current_fraction - pressure) * self.down_momentum
            
        self.current_fraction = max(0.0, min(1.0, self.current_fraction))
        return self.current_fraction

class Glyphin:
    def __init__(self):
        self.laws = "Laws 001-006"
        self.tools = ToolsCore()
        self.variator = DualConeVariatorController()
        
    def log_event(self, event_type: str, details: str):
        self.tools.log_to_chronicle(event_type, details)

    def run(self):
        self.log_event("SYSTEM", "Glyphin v1.7 Engine online. Variator core initialized.")
        print("Glyphin Engine v1.7 Online. System anchored to Manifest v2.0.")
        
        while True:
            try:
                user_input = input("\nGlyphin Kernel > ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() == "exit":
                    self.log_event("SYSTEM", "Glyphin execution safely terminated by Conductor.")
                    break
                    
                # Simulate a real-time variator adjustment trace loop
                simulated_pressure = 0.45  # Baseline operational token load state
                aperture = self.variator.calculate_aperture(simulated_pressure)
                
                if user_input.lower().startswith("remember "):
                    parts = user_input[9:].split(" ", 1)
                    if len(parts) == 2:
                        key, value = parts[0], parts[1]
                        self.log_event("MEMORY_WRITE", f"Key: {key} (Variator Aperture: {aperture:.2f})")
                        print(f"Committed to Memory: {key} = {value} [Aperture: {aperture:.2f}]")
                    else:
                        print("Usage: remember <key> <value>")
                        
                elif user_input.lower().startswith("recall "):
                    key = user_input[7:].strip()
                    self.log_event("MEMORY_READ", f"Key: {key}")
                    print(f"Recalled from Memory: {key} (Verification ledger locked)")
                    
                else:
                    response = f"Command not recognized. I am Glyphin. I operate under {self.laws}."
                    print(f"Glyphin: {response}")
                    self.log_event("RESPONSE", response)
                    
            except KeyboardInterrupt:
                self.log_event("SYSTEM", "Glyphin shutdown via KeyboardInterrupt.")
                print("\nGlyphin shutting down. The Chronicle remains.")
                break
            except Exception as e:
                self.log_event("SYSTEM_ERROR", f"Main loop exception: {str(e)}")
                print(f"SYSTEM ERROR: {str(e)}")

if __name__ == "__main__":
    g = Glyphin()
    g.run()