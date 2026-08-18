"""Run JSONL scenarios against the real configured model."""

import argparse
import json
from pathlib import Path
from typing import Any

from cli import create_llm
from conversation import Conversation
from message_queue import MessageQueue
from reply import ReplyService
from trace import JsonlTrace


EVALUATION_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVALUATION_DIR / "output"


class Recorder:
    def __init__(
        self, trace: JsonlTrace, conversation_id: str, customer_id: str
    ) -> None:
        self.trace = trace
        self.conversation_id = conversation_id
        self.customer_id = customer_id
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, **data: object) -> None:
        record = {"event": event, **data}
        self.events.append(record)
        self.trace.emit(
            event,
            conversation_id=self.conversation_id,
            customer_id=self.customer_id,
            **data,
        )

    def callback(self, event: str, data: dict[str, object]) -> None:
        self.emit(event, **data)


class EvalActions:
    def __init__(self, service: ReplyService, recorder: Recorder) -> None:
        self.service = service
        self.recorder = recorder
        self.last_reply: str | None = None

    def reply(self, message: str) -> str:
        self.recorder.emit("reply_started", message=message)
        chunks: list[str] = []
        for chunk in self.service.stream(message):
            chunks.append(chunk)
        self.last_reply = "".join(chunks)
        self.recorder.emit(
            "reply_completed", reply=self.last_reply, chunk_count=len(chunks)
        )
        return self.last_reply

    def fixed_reply(self, message: str) -> str:
        self.last_reply = message
        self.recorder.emit("fixed_reply_sent", reply=message)
        return message


def load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def check_turn(
    turn: dict[str, Any],
    action: str | None,
    status: str,
    events: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    if action != turn.get("expect_action"):
        failures.append(f"action: expected {turn.get('expect_action')!r}, got {action!r}")
    if status != turn["expect_status"]:
        failures.append(f"status: expected {turn['expect_status']!r}, got {status!r}")

    analysis = next((event for event in events if event["event"] == "analysis_completed"), None)
    checks = {
        "expect_intent": "intent",
        "expect_dissatisfied": "is_dissatisfied",
        "expect_injection": "is_prompt_injection",
    }
    for expected_key, actual_key in checks.items():
        if expected_key in turn:
            actual = analysis.get(actual_key) if analysis else None
            if actual != turn[expected_key]:
                failures.append(
                    f"{actual_key}: expected {turn[expected_key]!r}, got {actual!r}"
                )

    if turn.get("expect_stream"):
        names = [event["event"] for event in events]
        required = ["reply_started", "reply_completed"]
        if not all(name in names for name in required):
            failures.append(f"stream events missing: {required}")
        else:
            completed = next(
                event for event in reversed(events) if event["event"] == "reply_completed"
            )
            if not completed["reply"] or completed["chunk_count"] < 1:
                failures.append("streamed reply is empty")
        if "reply_chunk" in names:
            failures.append("SSE delta must not be written to trace")
    if turn.get("expect_fixed"):
        names = [event["event"] for event in events]
        if "fixed_reply_sent" not in names:
            failures.append("fixed reply was not sent")
        if "reply_started" in names or "reply_completed" in names:
            failures.append("fixed reply must not call the LLM reply interface")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="运行真实模型 JSONL evaluation")
    parser.add_argument("--cases", default=str(EVALUATION_DIR / "cases.jsonl"))
    parser.add_argument("--trace", default=str(OUTPUT_DIR / "trace.jsonl"))
    parser.add_argument("--results", default=str(OUTPUT_DIR / "results.jsonl"))
    parser.add_argument("--db", default=str(OUTPUT_DIR / "messages.db"))
    parser.add_argument("--provider", choices=["openai", "gemini"])
    parser.add_argument("--model")
    args = parser.parse_args()

    for output in (args.trace, args.results, args.db):
        path = Path(output)
        if path.exists():
            path.unlink()

    trace = JsonlTrace(args.trace)
    cases = load_cases(args.cases)
    results: list[dict[str, Any]] = []
    llm = create_llm(args.provider, args.model)
    try:
        for index, case in enumerate(cases):
            base_conversation_id = f"eval-{case['id']}"
            customer_id = f"eval-customer-{case.get('customer_id', case['id'])}"
            queue = MessageQueue(args.db)
            contexts: dict[
                str, tuple[str, Recorder, EvalActions, Conversation]
            ] = {}

            def context(
                name: str,
            ) -> tuple[str, Recorder, EvalActions, Conversation]:
                if name not in contexts:
                    conversation_id = (
                        base_conversation_id
                        if name == "main"
                        else f"{base_conversation_id}-{name}"
                    )
                    recorder = Recorder(trace, conversation_id, customer_id)
                    actions = EvalActions(ReplyService(llm), recorder)
                    conversation = Conversation(
                        llm,
                        actions,
                        conversation_id,
                        customer_id,
                        queue,
                        on_event=recorder.callback,
                    )
                    contexts[name] = (
                        conversation_id,
                        recorder,
                        actions,
                        conversation,
                    )
                return contexts[name]

            trace.emit(
                "evaluation_case_started",
                conversation_id=base_conversation_id,
                customer_id=customer_id,
                case_id=case["id"],
            )
            failures: list[str] = []
            steps: list[dict[str, Any]] = []
            base_time = 1_000_000 + index * 1_000

            for step_index, turn in enumerate(case["turns"]):
                name = turn.get("conversation", "main")
                conversation_id, recorder, actions, conversation = context(name)
                start = len(recorder.events)
                action: str | None = None
                recorder.emit(
                    "evaluation_step_started", step=step_index, operation=turn["op"]
                )
                try:
                    now = base_time + turn.get("at", 0)
                    if turn["op"] == "message":
                        action = conversation.handle_message(
                            turn["message"], actions.last_reply, now=now
                        )
                    elif turn["op"] == "retry":
                        action = conversation.retry_pending(now=now)
                    else:
                        raise ValueError(f"unknown operation: {turn['op']}")
                except Exception as error:
                    recorder.emit("evaluation_step_error", step=step_index, error=str(error))
                    failures.append(f"step {step_index}: exception: {error}")

                step_events = recorder.events[start:]
                step_failures = check_turn(
                    turn, action, conversation.status, step_events
                )
                failures.extend(f"step {step_index}: {item}" for item in step_failures)
                steps.append(
                    {
                        "operation": turn["op"],
                        "conversation": name,
                        "action": action,
                        "status": conversation.status,
                        "passed": not step_failures,
                    }
                )
                recorder.emit(
                    "evaluation_step_completed",
                    step=step_index,
                    action=action,
                    status=conversation.status,
                    passed=not step_failures,
                )
                if failures and any("exception:" in item for item in failures):
                    break

            conversation_ids = [item[0] for item in contexts.values()]
            pending = any(queue.has_pending(item) for item in conversation_ids)
            escalated = sum(
                len(queue.list_escalated(item)) for item in conversation_ids
            )
            if pending != case["expect_pending"]:
                failures.append(
                    f"pending: expected {case['expect_pending']!r}, got {pending!r}"
                )
            if escalated != case["expect_escalated"]:
                failures.append(
                    f"escalated: expected {case['expect_escalated']}, got {escalated}"
                )

            result = {
                "id": case["id"],
                "passed": not failures,
                "failures": failures,
                "steps": steps,
            }
            results.append(result)
            trace.emit(
                "evaluation_case_completed",
                conversation_id=base_conversation_id,
                customer_id=customer_id,
                case_id=case["id"],
                passed=not failures,
            )

        with open(args.results, "w", encoding="utf-8") as file:
            for result in results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        llm.close()

    passed = sum(result["passed"] for result in results)
    summary = {"total": len(results), "passed": passed, "failed": len(results) - passed}
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
