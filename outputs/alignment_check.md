# Prediction alignment check

- **whisper**: 150 rows, same set as manifest: True, same order: True, duplicates: 0, missing (first 5): none, not in manifest (first 5): none
- **whisper** reference drift vs manifest: 0 rows; language drift: 0 rows; empty model_output: 0 rows; inconsistent model field: 0 rows
- **whisper** distinct row schemas: 1
- **qwen3-asr**: 150 rows, same set as manifest: True, same order: True, duplicates: 0, missing (first 5): none, not in manifest (first 5): none
- **qwen3-asr** reference drift vs manifest: 0 rows; language drift: 0 rows; empty model_output: 0 rows; inconsistent model field: 0 rows
- **qwen3-asr** distinct row schemas: 1
- field sets identical across models: True

**Result: PASS**
