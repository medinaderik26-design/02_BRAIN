# ollama_bridge.py v1.2 - Context Shifting Integration
# Copyright 2026 Weisone Systems LLC. Proprietary and Confidential.
# Fixed: process_live_stream moved inside OllamaContextBridge class (was dangling outside)

import sys
import json
import importlib.util
from pathlib import Path

tools_path = Path(__file__).resolve().parent.parent / "04_TOOLS" / "tools_core.py"
spec_tools = importlib.util.spec_from_file_location("tools_core_bridge", str(tools_path))
if spec_tools is None or spec_tools.loader is None:
    raise ImportError(f"Unable to load ToolsCore from {tools_path}")

tools_module = importlib.util.module_from_spec(spec_tools)
sys.modules["tools_core_bridge"] = tools_module
spec_tools.loader.exec_module(tools_module)
ToolsCore = tools_module.ToolsCore

from glyphin import DualConeVariatorController

try:
    from ollama import chat
except ImportError as exc:
    raise ImportError("The ollama Python package is required for OllamaContextBridge.") from exc


class OllamaContextBridge:
    def __init__(self, model: str = "llama3"):
        self.tools = ToolsCore()
        self.variator = DualConeVariatorController(up_momentum=0.35, down_momentum=0.15)
        self.model = model
        self.raw_history = []

    def _estimate_pressure(self, new_prompt: str) -> float:
        total_chars = sum(len(m["content"]) for m in self.raw_history) + len(new_prompt)
        return min(1.0, total_chars / 8000.0)

    def _build_active_messages(self, aperture: float) -> list:
        if not self.raw_history:
            return []
        slice_index = max(1, int(len(self.raw_history) * aperture))
        slice_index = min(len(self.raw_history), slice_index)
        return self.raw_history[-slice_index:]

    def process_stream_request(self, new_prompt: str) -> str:
        try:
            estimated_pressure = self._estimate_pressure(new_prompt)
            aperture = self.variator.calculate_aperture(estimated_pressure)
            active_history = self._build_active_messages(aperture)

            self.tools.log_to_chronicle(
                "BRIDGE_SHIFT",
                f"Pressure: {estimated_pressure:.2f} -> Aperture: {aperture:.2f}. Active messages: {len(active_history)}"
            )

            messages = [
                {"role": "system", "content": "You are a local context-managed assistant."},
                *active_history,
                {"role": "user", "content": new_prompt},
            ]

            response = chat(model=self.model, messages=messages, stream=False)
            assistant_text = response["message"]["content"]

            self.raw_history.append({"role": "user", "content": new_prompt})
            self.raw_history.append({"role": "assistant", "content": assistant_text})

            payload_debug = {
                "model": self.model,
                "prompt": new_prompt,
                "context_messages_retained": len(active_history),
                "aperture_applied": round(aperture, 2),
                "response": assistant_text,
            }
            return json.dumps(payload_debug, indent=2)

        except Exception as exc:
            self.tools.log_to_chronicle("BRIDGE_ERROR", str(exc))
            return json.dumps({"error": str(exc)}, indent=2)

    def process_live_stream(self, new_prompt: str):
        """Processes the context pipeline and streams tokens back instantly."""
        try:
            # 1. Run your core architectural calculus
            estimated_pressure = self._estimate_pressure(new_prompt)
            aperture = self.variator.calculate_aperture(estimated_pressure)
            active_history = self._build_active_messages(aperture)

            # 2. Log structural updates to your chronicle
            self.tools.log_to_chronicle(
                "BRIDGE_STREAM_SHIFT",
                f"Pressure: {estimated_pressure:.2f} -> Aperture: {aperture:.2f}"
            )

            # 3. Assemble active context window
            messages = [
                {"role": "system", "content": "You are a local context-managed assistant."},
                *active_history,
                {"role": "user", "content": new_prompt},
            ]

            # 4. Trigger the native Ollama chat stream generator
            response_stream = chat(
                model=self.model,
                messages=messages,
                stream=True
            )

            # 5. Yield each token to the notebook in real-time
            full_response = ""
            for chunk in response_stream:
                token = chunk.get("message", {}).get("content", "")
                full_response += token
                yield token

            # 6. Commit the finalized exchange to your historical record
            self.raw_history.append({"role": "user", "content": new_prompt})
            self.raw_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            yield f"\n[STREAM ERROR]: {str(e)}"


if __name__ == "__main__":
    print("Ollama Context Isolation Bridge active.")
    bridge = OllamaContextBridge()

    print("\n--- Simulating Message 1 (Low Load) ---")
    print(bridge.process_stream_request("Hello kernel, establish initialization handshake."))

    print("\n--- Simulating Message 2 (Increasing Load) ---")
    print(bridge.process_stream_request("A" * 3000 + " [Large payload transmission block]"))

    print("\n--- Simulating Message 3 (High Load Compression Shift) ---")
    print(bridge.process_stream_request("B" * 4000 + " [Secondary heavy payload block]"))
