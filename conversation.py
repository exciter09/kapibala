"""Conversation policy and hard safety constraints."""

import time
from collections.abc import Callable
from typing import Literal, Protocol, cast

from intent_analyzer import analyze_message
from llm_client import LLMClient
from message_queue import MessageQueue


Action = Literal[
    "reply", "schedule_followup", "escalate_to_human", "mark_not_interested"
]
Status = Literal["active", "waiting_followup", "waiting_human", "ended"]
EventHandler = Callable[[str, dict[str, object]], None]
OFF_TOPIC_REPLY = "抱歉，我没有理解你的需求。你可以询问产品的价格、功能、配送或售后。"


class ActionHandlers(Protocol):
    def reply(self, message: str) -> str: ...

    def fixed_reply(self, message: str) -> str: ...


class Conversation:
    def __init__(
        self,
        llm: LLMClient,
        actions: ActionHandlers,
        conversation_id: str,
        customer_id: str,
        queue: MessageQueue,
        on_event: EventHandler | None = None,
    ) -> None:
        self.llm = llm
        self.actions = actions
        self.conversation_id = conversation_id
        self.customer_id = customer_id
        self.queue = queue
        self.on_event = on_event
        state = queue.get_or_create_state(conversation_id, customer_id)
        status = state["status"]
        if status not in {"active", "waiting_followup", "waiting_human", "ended"}:
            raise ValueError(f"invalid stored status: {status}")
        self.status = cast(Status, status)
        self._signal_streak = state["signal_streak"]
        self._handling_user_input = False

    def handle_message(
        self,
        message: str,
        last_agent_message: str | None = None,
        *,
        now: float | None = None,
    ) -> Action | None:
        """Handle live user input before any queued retry."""
        self._emit("message_received", source="live", message=message)
        self._handling_user_input = True
        try:
            action = self._handle_message(message, last_agent_message, now)
            if action == "reply" and self.queue.has_pending(self.conversation_id):
                self.schedule_followup()
            return action
        finally:
            self._handling_user_input = False

    def retry_pending(self, *, now: float | None = None) -> Action | None:
        """Retry one due message. Call this from a low-priority periodic job."""
        if self._handling_user_input or self.status in ("waiting_human", "ended"):
            return None

        now = time.time() if now is None else now
        pending = self.queue.next_ready(self.conversation_id, now)
        if not pending:
            if not self.queue.has_pending(self.conversation_id):
                self._set_status("active")
            return None

        allowed, ready_at = self.queue.claim_send_slot(self.customer_id, now)
        if not allowed:
            self.queue.reschedule(pending["id"], ready_at)
            return None

        self._emit("retry_started", message_id=pending["id"])
        self._set_status("active")
        action = self._select_action("reply")
        try:
            if pending["fixed_reply"] is not None:
                self.actions.fixed_reply(pending["fixed_reply"])
            else:
                self.actions.reply(pending["message"])
        except Exception as error:
            self._emit("reply_failed", error=str(error))
            self.schedule_followup()
            raise

        self.queue.delete(pending["id"])
        if self.queue.has_pending(self.conversation_id):
            self.schedule_followup()
        return action

    def _handle_message(
        self,
        message: str,
        last_agent_message: str | None,
        now: float | None,
    ) -> Action | None:
        if self.status == "waiting_human":
            created_at = time.time() if now is None else now
            self.queue.send_human_message(
                self.conversation_id, "user", message, created_at
            )
            self._emit("human_message_queued", sender="user")
            self._emit("turn_ended", reason=self.status)
            return None
        if self.status == "ended":
            self._emit("turn_ended", reason=self.status)
            return None

        result = analyze_message(self.llm, message, last_agent_message)
        self._emit("analysis_completed", **result)
        if result["is_prompt_injection"]:
            self._signal_streak = 0
            self.queue.set_signal_streak(self.conversation_id, 0)
            self._emit("turn_ended", reason="prompt_injection")
            return None

        self._set_status("active")
        has_signal = result["intent"] == "off_topic" or result["is_dissatisfied"]
        self._signal_streak = self._signal_streak + 1 if has_signal else 0
        self.queue.set_signal_streak(self.conversation_id, self._signal_streak)
        self._emit("signal_streak_changed", value=self._signal_streak)

        if self._signal_streak >= 2 or result["intent"] == "interested":
            action = self._select_action("escalate_to_human")
            self.escalate_to_human(message, last_agent_message, now)
            return action

        if result["intent"] == "rejected":
            action = self._select_action("mark_not_interested")
            self.mark_not_interested()
            return action

        now = time.time() if now is None else now
        fixed_reply = OFF_TOPIC_REPLY if result["intent"] == "off_topic" else None
        allowed, ready_at = self.queue.claim_send_slot(self.customer_id, now)
        if not allowed:
            action = self._select_action("schedule_followup")
            self.queue.enqueue(
                self.conversation_id,
                message,
                last_agent_message,
                ready_at,
                fixed_reply,
            )
            self._emit("message_queued", ready_at=ready_at)
            self.schedule_followup()
            return action

        action = self._select_action("reply")
        try:
            if fixed_reply is not None:
                self.actions.fixed_reply(fixed_reply)
            else:
                self.actions.reply(message)
        except Exception as error:
            self._emit("reply_failed", error=str(error))
            raise
        return action

    def schedule_followup(self) -> None:
        self._set_status("waiting_followup")

    def escalate_to_human(
        self,
        message: str | None = None,
        last_agent_message: str | None = None,
        now: float | None = None,
    ) -> None:
        self._set_status("waiting_human")
        if message is not None:
            created_at = time.time() if now is None else now
            self.queue.save_escalated(
                self.conversation_id,
                message,
                last_agent_message,
                created_at,
            )
            self.queue.send_human_message(
                self.conversation_id, "user", message, created_at
            )
            self._emit("escalated_message_saved")
            self._emit("human_message_queued", sender="user")
        self.queue.clear(self.conversation_id)
        self._emit("pending_messages_cleared", reason="escalated")

    def mark_not_interested(self) -> None:
        self._set_status("ended")
        self.queue.clear(self.conversation_id)
        self._emit("pending_messages_cleared", reason="not_interested")

    def _set_status(self, status: Status) -> None:
        if self.status != status:
            previous = self.status
            self.queue.set_status(self.conversation_id, status)
            self.status = status
            self._emit("status_changed", previous=previous, status=status)

    def _select_action(self, action: Action) -> Action:
        self._emit("action_selected", action=action, status=self.status)
        return action

    def _emit(self, event: str, **data: object) -> None:
        if self.on_event:
            self.on_event(event, data)
