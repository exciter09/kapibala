"""User-facing streaming CLI."""

import argparse
import os
import select
import sys
from uuid import uuid4

from dotenv import load_dotenv

from conversation import Conversation
from llm_client import LLMClient
from message_queue import MessageQueue
from reply import ReplyService
from trace import JsonlTrace


class StreamingActions:
    def __init__(
        self,
        llm: LLMClient,
        trace: JsonlTrace,
        conversation_id: str,
        customer_id: str,
    ) -> None:
        self.service = ReplyService(llm)
        self.trace = trace
        self.conversation_id = conversation_id
        self.customer_id = customer_id
        self.last_reply: str | None = None

    def reply(self, message: str) -> str:
        self.trace.emit(
            "reply_started",
            conversation_id=self.conversation_id,
            customer_id=self.customer_id,
            message=message,
        )
        print("\nAgent: ", end="", flush=True)
        chunks: list[str] = []
        for chunk in self.service.stream(message):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        print()
        self.last_reply = "".join(chunks)
        self.trace.emit(
            "reply_completed",
            conversation_id=self.conversation_id,
            customer_id=self.customer_id,
            reply=self.last_reply,
            chunk_count=len(chunks),
        )
        return self.last_reply

    def fixed_reply(self, message: str) -> str:
        print(f"\nAgent: {message}")
        self.last_reply = message
        self.trace.emit(
            "fixed_reply_sent",
            conversation_id=self.conversation_id,
            customer_id=self.customer_id,
            reply=message,
        )
        return message


def create_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    load_dotenv()
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    prefix = provider.upper()
    model = model or os.getenv(f"{prefix}_MODEL") or os.getenv("LLM_MODEL")
    if not model:
        raise ValueError(f"missing model: set {prefix}_MODEL or pass --model")
    return LLMClient(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        base_url=os.getenv(f"{prefix}_BASE_URL"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 客户初筛对话 CLI")
    parser.add_argument("--provider", choices=["openai", "gemini"])
    parser.add_argument("--model")
    parser.add_argument("--db", default="messages.db")
    parser.add_argument("--trace", default="trace.jsonl")
    parser.add_argument("--conversation-id", default=f"cli-{uuid4().hex[:8]}")
    parser.add_argument("--customer-id")
    args = parser.parse_args()

    trace = JsonlTrace(args.trace, echo=True)
    conversation_id = args.conversation_id
    customer_id = args.customer_id or conversation_id
    llm = create_llm(args.provider, args.model)
    actions = StreamingActions(llm, trace, conversation_id, customer_id)
    queue = MessageQueue(args.db)
    conversation = Conversation(
        llm,
        actions,
        conversation_id,
        customer_id,
        queue,
        on_event=lambda event, data: trace.emit(
            event,
            conversation_id=conversation_id,
            customer_id=customer_id,
            **data,
        ),
    )
    trace.emit(
        "cli_started",
        conversation_id=conversation_id,
        customer_id=customer_id,
        provider=args.provider or os.getenv("LLM_PROVIDER", "openai"),
        model=llm.model,
    )
    print("输入消息开始对话；命令：/status、/quit")

    try:
        print("你: ", end="", flush=True)
        while True:
            timeout = (
                0.25
                if conversation.status in ("waiting_followup", "waiting_human")
                else None
            )
            readable, _, _ = select.select([sys.stdin], [], [], timeout)
            if not readable:
                received = False
                while human_message := queue.pop_human_message(
                    conversation_id, "admin"
                ):
                    received = True
                    print(f"\nHuman: {human_message['message']}")
                    trace.emit(
                        "human_message_received",
                        conversation_id=conversation_id,
                        customer_id=customer_id,
                        sender="admin",
                        message=human_message["message"],
                    )
                if conversation.retry_pending() or received:
                    print("你: ", end="", flush=True)
                    continue
                continue

            line = sys.stdin.readline()
            if not line:
                break
            message = line.strip()
            if not message:
                print("你: ", end="", flush=True)
                continue
            trace.emit(
                "cli_input",
                conversation_id=conversation_id,
                customer_id=customer_id,
                message=message,
            )
            if message == "/quit":
                break
            if message == "/status":
                trace.emit(
                    "status_requested",
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    status=conversation.status,
                )
                print("你: ", end="", flush=True)
                continue
            conversation.handle_message(message, actions.last_reply)
            print("你: ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    except Exception as error:
        trace.emit(
            "error",
            conversation_id=conversation_id,
            customer_id=customer_id,
            error=str(error),
        )
        raise
    finally:
        trace.emit(
            "cli_stopped",
            conversation_id=conversation_id,
            customer_id=customer_id,
            status=conversation.status,
        )
        llm.close()


if __name__ == "__main__":
    main()
