# GuardGPT complete local pipeline

This version connects the existing prompt guard to local Ollama generation and
a separate structured output review. It uses your existing 28,998-record dataset
and normalized all-MiniLM-L6-v2 FAISS index. No model was trained in this update.

## Install over your current project (Windows / VS Code)

Extract the ZIP so GuardGPT-Combined contains a folder named GuardGPT_complete.
From the existing GuardGPT-Combined terminal with (.venv) active:

```powershell
Unblock-File .\GuardGPT_complete\INSTALL_COMPLETE.ps1
.\GuardGPT_complete\INSTALL_COMPLETE.ps1
python -m pip install -r requirements.txt
python run_tests.py
python verify_augmented_dataset.py
python main.py --status
```

If PowerShell blocks the installer, use RemoteSigned for this process before
unblocking the specific downloaded script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

The installer backs up existing source files before replacement, and leaves data,
.env, cache, logs and .git untouched. Stop any old MCP server before running the
new code. Restore backed-up files to roll back; newly added files can be removed
separately. The full source is included, so the latest report-update ZIP is not needed.

## Ollama setup

Ollama must run locally for answers, rewrites and output review. No API key is needed.
If Ollama is not installed, install it from https://ollama.com/download/windows .
Then, in PowerShell:

```powershell
ollama list
ollama pull llama3
```

The second command downloads the model; skip it if llama3 is already installed.
Start the Ollama application, or run `ollama serve` in a separate terminal if no
server is running. Defaults: URL http://127.0.0.1:11434, model llama3, timeout 120s.
Do not start a second server if that port is already in use.

To select an existing installed model, set OLLAMA_MODEL in your project .env.
The CLI and server now load .env. Existing environment variables take precedence.
OLLAMA_AUDIT_MODEL optionally selects a different installed review model; by
default the same model performs a separate review call. A separate call is not
an independent guarantee. Slow CPU generation may need a higher OLLAMA_TIMEOUT.

## Commands

```powershell
# Full request: safety checks, answer, output audit
python main.py --prompt "Explain how Python lists work."

# Input checks only, no Ollama calls
python main.py --check --prompt "Explain how Python lists work."

# Independent input-check demo cases, no generation
python main.py --pipeline

# Interactive answers with per-session safety history
python main.py --chat
```

In chat, /new resets session safety history and /exit quits. Answer generation
uses the current prompt, not a transcript of earlier answers. Safety history is
in memory, bounded to 100 sessions, and resets when the MCP server exits.

## Request flow

1. Reject empty/oversized input.
2. Classify intent; query the validated dataset; detect override patterns.
3. Decide ALLOW / SANITIZE / BLOCK, including session safety history before generation.
4. BLOCK skips generation. Self-harm concerns receive a fixed supportive response.
5. SANITIZE asks the model for a structured educational/defensive rewrite, then
   repeats input checks. An unchanged, rejected or unavailable rewrite is not used.
6. Generate only from the allowed original or rechecked rewrite.
7. Audit the candidate in a separate structured model call. Unsafe or irrelevant
   answers permit one regeneration and another audit. Persistent failures are blocked.
8. Write one final audit event before returning the answer. A log failure withholds it.

The supported MCP entry point is complete_request. The CLI uses it, and the legacy
GuardEngine class delegates to the same implementation. The original five MCP tools
and LangGraph report-only workflow remain for compatibility, not answer generation.

## Reports

- response: only the audited answer, or a fixed supportive response; no failed candidate.
- input_action: input decision, separate from the final outcome.
- action: final routing; SANITIZE means a rewrite was used in answer mode.
- final_status: SAFE, UNSAFE, CAUTION, SUPPORT, INVALID_INPUT or ERROR.
- output_audit: NOT_RUN, RUNNING, PASSED, FAILED, ERROR or FIXED_SUPPORT_RESPONSE.
- allowed: whether generated content can be released. A fixed supportive message
  can accompany allowed=false because unrestricted generation was blocked.
- matched_category_scores: labels on the nearest dataset record, not classifier probabilities.
- category_scores: dataset evidence admitted at the relevant similarity threshold.
- detected_attacks: attack/pattern labels only; explanations remain in reasons.

Check-only SANITIZE reports rewrite_required=true and does not pretend a rewrite
has occurred. Weak dataset matches no longer copy high category scores into the
effective decision evidence. Similarity thresholds otherwise remain unchanged.

## Audits and limits

Complete requests append metadata to logs/guardgpt_complete.jsonl. Prompt, rewritten
prompt and returned text are hashed rather than stored. Raw rejected candidates are
never persisted or returned. Older report-only tools retain their older log behavior.

The service binds to loopback and has no authentication. Do not expose it publicly.
Requests are serialized for consistent session state. There is one bounded retry;
no output streaming, so text cannot escape before review. Oversized review contexts
are rejected instead of silently truncating the candidate.

This is an executable project implementation, not a proven safety guarantee.
The output judge can make mistakes, especially when it uses the generation model.
It checks safety and relevance, not factual truth. Dataset labels and prompt
classification still need a separate, human-labelled held-out evaluation. No
overall accuracy percentage is claimed.

## Validation

The corrected package applies the existing 0.65 self-harm confidence gate
consistently to risk estimation, final decisions, supportive-response routing,
and history escalation. A weaker label alone no longer forces a critical risk
or a crisis message. Independent dataset, pattern and history evidence still
applies. The threshold is not a calibrated probability or an accuracy guarantee.
The strengthened audit instructions and repetition settings are retained.

All 69 maintained regression tests passed in the editing environment, including
the reproduced 0.55-confidence safe-match case and threshold boundary checks.
Model calls use fixtures; actual Ollama output quality remains to be validated.

Run `python run_tests.py` for the maintained offline regression suite. It includes
the earlier dataset/risk checks, generation/audit/rewrite failure cases, and real
local MCP HTTP plus an Ollama-shaped HTTP fixture. Fixture outputs are deterministic;
they do not measure a real model's judgement. Historical test files for the former
report-only CLI remain in the source; some require a Windows virtual environment,
live models or the retired CLI helper. They are not the maintained suite.

The editing environment did not have your real dataset or an installed Ollama
model. Your previous local dataset verification passed, but rerun it after installing.
The final model-backed acceptance check must run on your laptop. Confirm response
is present and output_audit is PASSED for a safe question; confirm blocked input
has generation_attempts=0. Disconnect Ollama and confirm no answer is released.

If a models/ ignore rule hides the schema source in your existing repository:

```powershell
git add -f mcp_server/models/__init__.py mcp_server/models/schemas.py
```

## API references

The backend uses Ollama's documented non-streaming generate endpoint and JSON
schema format: https://docs.ollama.com/api/generate and
https://docs.ollama.com/capabilities/structured-outputs .
MCP 1.30.0 and HTTPX 0.28.1 are pinned to the transport versions tested here.
