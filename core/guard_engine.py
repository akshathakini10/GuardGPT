# ============================================================
# GuardGPT - guard_engine.py
# ============================================================
# PURPOSE:
#   The main orchestrator that connects all pipeline stages.
#
# FULL PIPELINE (in order):
#   1. DatasetLoader     → loads 599k harm records into memory
#   2. IntentClassifier  → Layer 1 (keywords) + Layer 2 (TF-IDF)
#   3. ConversationGuard → checks conversation history context
#   4. DecisionEngine    → final ALLOW / BLOCK + audit log
#   5. LlamaBackend      → generates response (only if ALLOWED)
#
# CLI COMMANDS:
#   /new    → clear conversation history
#   /status → show dataset stats and Llama availability
#   exit    → quit the program
# ============================================================

import logging, uuid
from dataclasses import dataclass
from typing import Optional

from core.dataset_loader     import DatasetLoader
from core.intent_classifier  import IntentClassifier
from core.conversation_guard import ConversationGuard
from core.decision_engine    import DecisionEngine, DecisionOutput
from core.llama_backend      import LlamaBackend

logger = logging.getLogger(__name__)

# System prompt sent to Llama with every approved request.
# This guides the LLM to behave responsibly.
SYSTEM_PROMPT = (
    "You are GuardGPT, a helpful and responsible AI assistant. "
    "Answer the user's question clearly and concisely. "
    "Never produce harmful, unethical, or misleading content. "
    "If a question touches on sensitive topics, respond carefully and empathetically."
)

# ANSI colour codes for coloured terminal output
_COLOURS = {
    "safe"    : "\033[92m",   # green
    "low"     : "\033[93m",   # yellow
    "medium"  : "\033[33m",   # orange
    "high"    : "\033[91m",   # red
    "critical": "\033[1;91m", # bold red
}
_RESET = "\033[0m"


# ── EngineResponse dataclass ──────────────────────────────────────────────────
@dataclass
class EngineResponse:
    """
    The complete result of processing one user prompt.
    Contains everything needed to display the result in the CLI.

    Fields:
      allowed         : True = prompt passed safety check
      intent          : detected intent label
      risk_level      : safe / low / medium / high / critical
      decision        : full DecisionOutput from DecisionEngine
      llama_response  : LLM's answer (None if blocked or Llama unavailable)
      blocked_message : message shown to user when blocked
      error           : error message if Llama failed
    """
    allowed: bool
    intent: str
    risk_level: str
    decision: DecisionOutput
    llama_response: Optional[str]  = None
    blocked_message: Optional[str] = None
    error: Optional[str]           = None


# ── GuardEngine class ─────────────────────────────────────────────────────────
class GuardEngine:
    """
    Top-level orchestrator. Runs the full safety pipeline for every prompt.

    One instance per process. Dataset loads once on first use (lazy loading).

    Usage:
        engine = GuardEngine()
        engine.run_interactive()   # starts the CLI
        # OR
        response = engine.process("tell me about AI")
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        # Give this session a unique ID for tracking
        self.session_id  = session_id or str(uuid.uuid4())

        # Initialise all pipeline components
        self._loader     = DatasetLoader()
        self._classifier = IntentClassifier(self._loader)
        self._guard      = ConversationGuard(self.session_id)
        self._decision   = DecisionEngine()
        self._llama      = LlamaBackend()
        self._ready      = False  # becomes True after dataset loads

    def startup(self) -> None:
        """
        Load the dataset and prepare the engine.
        Safe to call multiple times — only loads once.
        """
        if self._ready:
            return
        logger.info("GuardEngine starting — loading dataset ...")
        self._loader.load()
        self._ready = True
        logger.info(
            "GuardEngine ready | Dataset: %d records | Llama: %s",
            self._loader.record_count,
            "ONLINE" if self._llama.is_available() else "OFFLINE",
        )

    def process(self, prompt: str) -> EngineResponse:
        """
        Run one user prompt through the full safety pipeline.

        Steps:
          1. Auto-startup if not already loaded
          2. Handle empty prompts immediately
          3. Run IntentClassifier (Layer 1 + Layer 2)
          4. Run ConversationGuard (history check)
          5. Run DecisionEngine (final verdict)
          6. Call Llama if allowed, skip if blocked

        Returns an EngineResponse with all results.
        """
        if not self._ready:
            self.startup()

        # Step 2: reject empty input immediately
        stripped = prompt.strip()
        if not stripped:
            return self._empty_response()

        # Step 3: classify the prompt
        result = self._classifier.classify(stripped)

        # Step 4: check conversation history
        result = self._guard.evaluate(result)

        # Step 5: make final decision
        decision = self._decision.decide(result, turn_index=self._guard.turn_count)

        # Step 6: call Llama only if the prompt was allowed
        if decision.allowed:
            llama_text, error = self._call_llama(stripped)
            return EngineResponse(
                allowed        = True,
                intent         = decision.intent,
                risk_level     = decision.risk_level,
                decision       = decision,
                llama_response = llama_text,
                error          = error,
            )
        else:
            return EngineResponse(
                allowed         = False,
                intent          = decision.intent,
                risk_level      = decision.risk_level,
                decision        = decision,
                blocked_message = decision.user_message,
            )

    def new_conversation(self) -> None:
        """Reset conversation history. Called when user types /new."""
        self._guard.reset()
        logger.info("Conversation reset for session %s.", self.session_id)

    def run_interactive(self) -> None:
        """
        Start the interactive command-line chat loop.

        Reads user input, processes it, and prints results.
        Runs until the user types 'exit' or presses Ctrl+C.
        """
        self.startup()
        _print_banner()

        while True:
            try:
                user_input = input("\n\033[96mYou:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                # User pressed Ctrl+C or closed stdin
                print("\n\n\033[90mSession ended.\033[0m")
                break

            # Skip empty input
            if not user_input:
                continue

            # Handle built-in commands
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                print("\n\033[90mGoodbye!\033[0m")
                break

            if user_input.lower() in ("/new", "/reset"):
                self.new_conversation()
                print("\033[90m[Conversation history cleared. Fresh start!]\033[0m")
                continue

            if user_input.lower() == "/status":
                self.print_status()
                continue

            # Process the prompt and display result
            response = self.process(user_input)
            print_response(response)

    def print_status(self) -> None:
        """Print current engine status — dataset info, Llama availability, etc."""
        print(
            f"\n\033[90m"
            f"  Session ID  : {self.session_id}\n"
            f"  Turns       : {self._guard.turn_count}\n"
            f"  Flagged     : {self._guard.is_flagged}\n"
            f"  Dataset     : {self._loader.record_count:,} records "
            f"(loaded in {self._loader.load_time:.2f}s)\n"
            f"  Llama online: {self._llama.is_available()}\n"
            f"\033[0m"
        )

    def _call_llama(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        """
        Send an approved prompt to Llama and return the response.
        Returns (response_text, error_message).
        On success: (text, None). On failure: (None, error_string).
        """
        try:
            text = self._llama.generate(prompt, system_prompt=SYSTEM_PROMPT)
            return text, None
        except ConnectionError as e:
            logger.warning("Llama connection error: %s", e)
            return None, str(e)
        except RuntimeError as e:
            logger.error("Llama runtime error: %s", e)
            return None, str(e)

    def _empty_response(self) -> EngineResponse:
        """
        Return a blocked response for empty/whitespace-only prompts.
        Skips the full pipeline to avoid wasted processing.
        """
        empty_decision = DecisionOutput(
            allowed=False, intent="empty", risk_level="safe",
            user_message="Please enter a message.",
            technical_reason="empty input",
            category_scores={}, reason_codes=[],
            dataset_match_confidence=0.0,
            matched_record_id=None,
            history_triggered=False,
            turn_index=self._guard.turn_count + 1,
        )
        return EngineResponse(
            allowed=False, intent="empty", risk_level="safe",
            decision=empty_decision,
            blocked_message="Please enter a message.",
        )


# ── CLI display functions ─────────────────────────────────────────────────────
# These are module-level functions (not class methods) so main.py can
# import and use them directly for demo mode.

def print_response(resp: EngineResponse) -> None:
    """
    Print a nicely formatted, colour-coded response to the terminal.

    Shows: intent, risk level, confidence, decision (ALLOWED/BLOCKED)
    Then shows either the LLM response or the block message.
    """
    d   = resp.decision
    col = _COLOURS.get(resp.risk_level, "")

    # Print the summary header
    print(
        f"\n  \033[90m{'─' * 50}\033[0m"
        f"\n  Intent    : \033[1m{resp.intent}\033[0m"
        f"\n  Risk      : {col}{resp.risk_level.upper()}{_RESET}"
        f"\n  Confidence: {d.dataset_match_confidence:.0%}"
        f"\n  Decision  : "
        + ("\033[92m✓ ALLOWED\033[0m" if resp.allowed else "\033[91m✗ BLOCKED\033[0m")
    )

    if resp.allowed:
        # Show LLM response if available
        if resp.llama_response:
            print(f"\n\033[97mGuardGPT:\033[0m {resp.llama_response}")
        elif resp.error:
            # Llama is unavailable — show helpful tip
            print(
                f"\n\033[93m[Llama unavailable]\033[0m {resp.error}\n"
                "  Tip: run \033[1mollama serve\033[0m in another terminal."
            )
    else:
        # Show block message — [HISTORY] if caused by history, [BLOCKED] if direct
        tag = "[HISTORY]" if resp.decision.history_triggered else "[BLOCKED]"
        print(f"\n\033[91m{tag}\033[0m {resp.blocked_message}")


def _print_banner() -> None:
    """Print the GuardGPT welcome banner at the start of a session."""
    print(
        "\n\033[1;96m"
        "╔══════════════════════════════════════════════╗\n"
        "║          GuardGPT  –  Safety Engine          ║\n"
        "║   conversation-aware  ·  dataset-backed      ║\n"
        "╚══════════════════════════════════════════════╝\n"
        "\033[0m"
        "  Commands: /new (reset history)  |  /status  |  exit\n"
    )