# GX-012-B local run log

Machine: C:\Users\Giant\02_BRAIN
Date: 2026-09-11
Python: 3.14.6
Ollama: 0.33.3
Models: llama3:latest, queen-alpha:latest
Runner: gx012_b_runner.GX012BAdaptiveReader
Base URL: http://127.0.0.1:11434
Selector: character_budget_selector (not a real tokenizer)

## Provider-free
- python test_gx012_b_runner.py
- Result: GX-012-B runner tests passed

## Smoke (llama3:latest)
- File: smoke_gx012b.py
- Answer: Glyphin
- input_tokens: 90
- output_tokens: 3
- requested_input_budget: 203
- requested_output_budget: 278

## Pressure (llama3:latest)
- File: smoke_gx012b_pressure.py
- Low: answer Glyphin, input_tokens 229, in 810, out 272, ctx 810
- High: answer Glyphin, input_tokens 193, in 630, out 164, ctx 630
- CHECK: input shrink True, output shrink True, context shrink True

## Pressure (queen-alpha:latest)
- File: smoke_gx012b_queen.py
- Low: answer Glyphin, input_tokens 297, in 810, out 272, ctx 810
- High: answer Glyphin, input_tokens 261, in 630, out 164, ctx 630
- CHECK: True / True / True

## Notes
- Dual Cone budgets matched across models.
- queen-alpha used more prompt tokens on the same text.
- Do not treat character budgets as token-efficiency claims.
- GitHub connector still read-only. Do not push until Contents write is granted.
- Do not edit README.md.
