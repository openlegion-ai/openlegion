"""Regression test for M3 (Phase 0-5 integration review).

``AskBroker._post_thread`` posted the ask question/answer into the durable
Team Threads store with ``body`` passed positionally (``ThreadStore.post_message``
declares it keyword-only) and assigned the dict returned by ``ensure_dm_thread``
as the ``thread_id``. Both raised ``TypeError`` swallowed by the broker's
non-fatal warning, so no ask Q&A ever landed in the thread store.

These tests wire a real ``ThreadStore(":memory:")`` and assert both the question
and the answer rows are persisted on the pair's DM thread.
"""

import asyncio

from src.host.asks import AskBroker
from src.host.threads import ThreadStore


def _dm_messages(store: ThreadStore, scope_id: str, a: str, b: str) -> list[dict]:
    thread = store.ensure_dm_thread(scope_id, a, b)
    return store.list_messages(thread["id"])


def test_ask_question_and_answer_land_in_thread_store():
    async def _run():
        store = ThreadStore(":memory:")
        broker = AskBroker(thread_store=store)
        record = broker.create(
            "asker", "recipient", "what is the status?", 30, scope_id="team-x"
        )
        # The question must be durably recorded (was silently lost pre-fix).
        msgs = _dm_messages(store, "team-x", "asker", "recipient")
        assert [m["body"] for m in msgs] == ["what is the status?"]
        assert msgs[0]["sender"] == "asker"
        assert msgs[0]["payload"] == {"ask_id": record.ask_id}

        # The answer must also land on the same DM thread.
        result = broker.resolve(record.ask_id, "all green", by="recipient")
        assert result == {"ok": True}
        msgs = _dm_messages(store, "team-x", "asker", "recipient")
        assert [m["body"] for m in msgs] == ["what is the status?", "all green"]
        assert msgs[1]["sender"] == "recipient"
        store.close()

    asyncio.run(_run())


def test_ask_thread_id_is_a_string_not_a_dict():
    async def _run():
        store = ThreadStore(":memory:")
        broker = AskBroker(thread_store=store)
        record = broker.create("a", "b", "q", 30, scope_id="team-y")
        # ensure_dm_thread returns a dict; the record must hold the id string.
        assert isinstance(record.thread_id, str)
        assert record.thread_id == "dm:a:b"
        store.close()

    asyncio.run(_run())
