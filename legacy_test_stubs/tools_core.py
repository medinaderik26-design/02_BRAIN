"""Test-only ToolsCore stub.

Not a product fix. Used if a later import harness needs 04_TOOLS without
changing glyphin.py.
"""


class ToolsCore:
    def __init__(self):
        self.events = []

    def log_to_chronicle(self, event_type: str, details: str) -> None:
        self.events.append((event_type, details))
