#!/usr/bin/env python3
"""M5 Inference Engine — Interactive Chat CLI.

Streams responses with:
- <think> block filtering (hidden or dimmed)
- Clean line wrapping
- Model switching via /model command
- System prompt via /system command
"""

import json
import re
import sys
import urllib.request
import urllib.error

API = "http://127.0.0.1:11436"

# ANSI colors
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
YELLOW = "\033[0;33m"
GRAY = "\033[0;90m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def api_get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{API}{path}", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def api_post(path: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def stream_chat(messages: list, model: str, session_id: str) -> str:
    """Send a streaming chat request and print the response, filtering <think> blocks."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 32768,
        "stream": True,
        "session_id": session_id,
        # ── Loop 防止 (greedy decoding は repetition loop の温床) ──
        "temperature": 0.6,           # 軽くサンプル、決定論性は緩める
        "repetition_penalty": 1.05,   # 同じトークン繰返しに mild penalty
    }).encode()

    req = urllib.request.Request(
        f"{API}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    full_text = ""
    in_think = False
    think_buf = ""

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                content = (
                    data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if not content:
                    continue

                full_text += content

                # Filter <think> blocks
                for char in content:
                    if in_think:
                        think_buf += char
                        if think_buf.endswith("</think>"):
                            inner = think_buf[: -len("</think>")]
                            first_line = inner.strip().split("\n")[0][:80]
                            if first_line:
                                sys.stdout.write(f"{DIM}({first_line}...){NC}\n")
                            in_think = False
                            think_buf = ""
                    else:
                        think_buf += char
                        if think_buf.endswith("<think>"):
                            in_think = True
                            think_buf = ""
                        elif len(think_buf) > 7:
                            safe = think_buf[0]
                            think_buf = think_buf[1:]
                            sys.stdout.write(safe)
                            sys.stdout.flush()
                        elif "<think>"[: len(think_buf)] != think_buf:
                            sys.stdout.write(think_buf)
                            sys.stdout.flush()
                            think_buf = ""

            # Flush remaining buffer
            if think_buf and not in_think:
                sys.stdout.write(think_buf)
                sys.stdout.flush()

    except urllib.error.URLError as e:
        print(f"\n{YELLOW}Connection error: {e}{NC}")
    except KeyboardInterrupt:
        pass

    return full_text


def run(args=None) -> int:
    """Entry point for ``m5-infer chat``.

    ``args`` is the argparse namespace from the top-level CLI dispatcher
    (it carries ``model``). When invoked directly via
    ``python -m app.cli.chat``, ``args`` is ``None`` and we fall back to
    ``sys.argv[1]``.
    """
    if args is not None and getattr(args, "model", None):
        model = args.model
    else:
        model = sys.argv[1] if len(sys.argv) > 1 else "default"
    return _chat_loop(model)


def main():
    """Legacy entry point (``python -m app.cli.chat``)."""
    return _chat_loop(sys.argv[1] if len(sys.argv) > 1 else "default")


def _chat_loop(model: str) -> int:
    session_id = f"chat-{id(object()):x}"

    # Check server
    health = api_get("/health")
    if not health:
        print(f"{YELLOW}Server not running. Start it first:{NC}")
        print("  m5-infer start      (or m5-infer)")
        return 1

    # Get current model
    models = api_get("/v1/models")
    current = models.get("data", [{}])[0].get("id", "unknown") if models.get("data") else "unknown"

    # Pull requested model if needed
    if model != "default" and "/" in model and current != model:
        print(f"{YELLOW}Loading model: {model} ...{NC}")
        result = api_post("/v1/models/pull", {"model": model})
        if result.get("status") == "error":
            print(f"{YELLOW}Error: {result.get('error')}{NC}")
            return 1
        print(f"{GREEN}Loaded.{NC}")
        current = model

    print(f"{BOLD}M5 Inference Engine — Chat{NC}")
    print(f"{GRAY}Model:   {current}{NC}")
    print(f"{GRAY}Session: {session_id}{NC}")
    print(f"{GRAY}Commands: /quit  /model <hf-repo>  /system <prompt>  /info{NC}")
    print()

    system_prompt = ""
    history: list[dict] = []  # Accumulated conversation history

    while True:
        try:
            user_input = input(f"{GREEN}You> {NC}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        # Commands
        if user_input.strip() in ("/quit", "/exit"):
            print("Bye!")
            break

        if user_input.strip().startswith("/model "):
            new_model = user_input.strip()[7:].strip()
            print(f"{YELLOW}Switching to: {new_model} ...{NC}")
            result = api_post("/v1/models/pull", {"model": new_model})
            if result.get("status") == "error":
                print(f"{YELLOW}Error: {result.get('error')}{NC}")
            else:
                current = new_model
                model = new_model
                print(f"{GREEN}Now using: {current}{NC}")
            continue

        if user_input.strip().startswith("/system "):
            system_prompt = user_input.strip()[8:]
            history.clear()  # Reset history when system prompt changes
            print(f"{GRAY}System prompt set: {system_prompt[:60]}{'...' if len(system_prompt) > 60 else ''}{NC}")
            continue

        if user_input.strip() == "/clear":
            history.clear()
            print(f"{GRAY}Conversation history cleared.{NC}")
            continue

        if user_input.strip() == "/info":
            info = api_get("/health")
            print(json.dumps(info, indent=2))
            continue

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Build full messages: system prompt + history
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        # Stream response
        sys.stdout.write(f"{CYAN}AI> {NC}")
        sys.stdout.flush()
        response_text = stream_chat(messages, model, session_id)
        print("\n")

        # Add assistant response to history
        if response_text.strip():
            history.append({"role": "assistant", "content": response_text})

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
