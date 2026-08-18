"""Human takeover CLI for escalated conversations."""

import argparse
import select
import sys
import time

from message_queue import MessageQueue
from trace import JsonlTrace


def receive_user_messages(
    queue: MessageQueue,
    trace: JsonlTrace,
    conversation_id: str,
    customer_id: str,
) -> bool:
    received = False
    while message := queue.pop_human_message(conversation_id, "user"):
        received = True
        print(f"\nUser: {message['message']}")
        trace.emit(
            "human_message_received",
            conversation_id=conversation_id,
            customer_id=customer_id,
            sender="user",
            message=message["message"],
        )
    return received


def main() -> None:
    parser = argparse.ArgumentParser(description="人工接管 CLI")
    parser.add_argument("conversation_id", nargs="?")
    parser.add_argument("--db", default="messages.db")
    parser.add_argument("--trace", default="admin_trace.jsonl")
    args = parser.parse_args()

    queue = MessageQueue(args.db)
    conversation_id = args.conversation_id
    if not conversation_id:
        waiting = queue.list_conversations("waiting_human")
        if not waiting:
            raise SystemExit("没有等待人工接管的会话")
        print("等待人工接管：" + "、".join(waiting))
        conversation_id = input("输入 conversation id: ").strip()
    if conversation_id not in queue.list_conversations("waiting_human"):
        raise SystemExit("该会话不处于等待人工状态")
    customer_id = queue.get_customer_id(conversation_id) or conversation_id

    trace = JsonlTrace(args.trace, echo=True)
    trace.emit(
        "takeover_started",
        conversation_id=conversation_id,
        customer_id=customer_id,
    )
    print(f"已接管 {conversation_id}；输入 /quit 退出")
    receive_user_messages(queue, trace, conversation_id, customer_id)
    print("Admin: ", end="", flush=True)

    try:
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.25)
            if not readable:
                if receive_user_messages(
                    queue, trace, conversation_id, customer_id
                ):
                    print("Admin: ", end="", flush=True)
                continue
            line = sys.stdin.readline()
            if not line:
                break
            message = line.strip()
            if not message:
                print("Admin: ", end="", flush=True)
                continue
            if message == "/quit":
                break
            queue.send_human_message(
                conversation_id, "admin", message, time.time()
            )
            trace.emit(
                "human_message_sent",
                conversation_id=conversation_id,
                customer_id=customer_id,
                sender="admin",
                message=message,
            )
            print("Admin: ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        trace.emit(
            "takeover_stopped",
            conversation_id=conversation_id,
            customer_id=customer_id,
        )


if __name__ == "__main__":
    main()
