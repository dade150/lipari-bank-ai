# tests/unit/test_tools.py
import json

import pytest

from lipari_bank_ai.db.models import BankAccount, Transaction
from lipari_bank_ai.llm.opencode_provider import _parse_tool_calls
from lipari_bank_ai.tools.account import AccountTool
from lipari_bank_ai.tools.registry import ToolRegistry
from lipari_bank_ai.tools.transactions import TransactionTool


async def _seed_accounts(session, user_id: str = "u1") -> str:
    acc = BankAccount(
        id="acc-test-001",
        user_id=user_id,
        account_type="checking",
        balance=5230.50,
        currency="EUR",
    )
    session.add(acc)
    await session.flush()
    return acc.id


async def _seed_transactions(session, account_id: str) -> None:
    txs = [
        Transaction(
            id=f"tx-{i}",
            account_id=account_id,
            amount=-45.50 + i,
            description=f"Spesa {i}",
            category="GROCERIES",
        )
        for i in range(5)
    ]
    session.add_all(txs)
    await session.flush()


# ── ToolRegistry ────────────────────────────────────────────────────────────

def test_tool_registry_register_and_get():
    registry = ToolRegistry()

    class FakeTool:
        name = "fake"
        description = "A fake tool"
        parameters_schema = {"type": "object", "properties": {}}
        async def execute(self, params, user_id):  # noqa: ANN001
            return {}

    registry.register(FakeTool())
    assert registry.get("fake") is not None
    assert registry.get("nonexistent") is None


def test_tool_registry_list_tools():
    registry = ToolRegistry()

    class ToolA:
        name = "a"
        description = "Tool A"
        parameters_schema = {}
        async def execute(self, params, user_id):  # noqa: ANN001
            return {}

    class ToolB:
        name = "b"
        description = "Tool B"
        parameters_schema = {}
        async def execute(self, params, user_id):  # noqa: ANN001
            return {}

    registry.register(ToolA())
    registry.register(ToolB())
    assert len(registry.list_tools()) == 2


def test_tool_registry_to_openai_tools():
    registry = ToolRegistry()

    class MyTool:
        name = "my_tool"
        description = "Does something"
        parameters_schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        async def execute(self, params, user_id):  # noqa: ANN001
            return {}

    registry.register(MyTool())
    openai_tools = registry.to_openai_tools()
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "my_tool"


def test_tool_registry_to_prompt_section():
    registry = ToolRegistry()

    class MyTool:
        name = "my_tool"
        description = "Does something"
        parameters_schema = {"type": "object", "properties": {}}
        async def execute(self, params, user_id):  # noqa: ANN001
            return {}

    registry.register(MyTool())
    section = registry.to_prompt_section()
    assert "my_tool" in section
    assert "Does something" in section


# ── AccountTool ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_account_tool_returns_balance(test_session):
    await _seed_accounts(test_session, "u1")
    tool = AccountTool(test_session)

    result = await tool.execute({}, "u1")

    assert result["balance"] == 5230.50
    assert result["currency"] == "EUR"
    assert result["account_type"] == "checking"


@pytest.mark.asyncio
async def test_account_tool_savings(test_session):
    acc = BankAccount(
        id="acc-test-sav",
        user_id="u1",
        account_type="savings",
        balance=15000.00,
        currency="EUR",
    )
    test_session.add(acc)
    await test_session.flush()

    tool = AccountTool(test_session)
    result = await tool.execute({"account_type": "savings"}, "u1")

    assert result["balance"] == 15000.00
    assert result["account_type"] == "savings"


@pytest.mark.asyncio
async def test_account_tool_not_found(test_session):
    tool = AccountTool(test_session)
    result = await tool.execute({}, "nonexistent-user")
    assert "error" in result


# ── TransactionTool ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transaction_tool_returns_recent(test_session):
    acc_id = await _seed_accounts(test_session, "u1")
    await _seed_transactions(test_session, acc_id)

    tool = TransactionTool(test_session)
    result = await tool.execute({"limit": 3}, "u1")

    assert result["count"] == 3
    assert len(result["transactions"]) == 3
    for tx in result["transactions"]:
        assert "amount" in tx
        assert "description" in tx
        assert "category" in tx
        assert "date" in tx


@pytest.mark.asyncio
async def test_transaction_tool_filter_by_category(test_session):
    acc_id = await _seed_accounts(test_session, "u1")
    await _seed_transactions(test_session, acc_id)

    tool = TransactionTool(test_session)
    result = await tool.execute({"category": "GROCERIES"}, "u1")

    assert result["count"] == 5
    for tx in result["transactions"]:
        assert tx["category"] == "GROCERIES"


@pytest.mark.asyncio
async def test_transaction_tool_no_account(test_session):
    tool = TransactionTool(test_session)
    result = await tool.execute({}, "nonexistent-user")
    assert "error" in result


# ── _parse_tool_calls ───────────────────────────────────────────────────────

def test_parse_tool_calls_basic():
    tc = {"name": "get_account_balance", "arguments": {"account_type": "checking"}}
    text = f"Ciao! ```tool\n{json.dumps(tc)}\n```"
    clean, calls = _parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "get_account_balance"
    assert calls[0].arguments == {"account_type": "checking"}
    assert "```tool" not in clean


def test_parse_tool_calls_multiple():
    tc1 = {"name": "get_account_balance", "arguments": {}}
    tc2 = {"name": "get_recent_transactions", "arguments": {"limit": 3}}
    text = (
        f"Primo tool:\n```tool\n{json.dumps(tc1)}\n```\n"
        f"Secondo tool:\n```tool\n{json.dumps(tc2)}\n```"
    )
    clean, calls = _parse_tool_calls(text)

    assert len(calls) == 2
    assert calls[0].name == "get_account_balance"
    assert calls[1].name == "get_recent_transactions"
    assert calls[1].arguments == {"limit": 3}


def test_parse_tool_calls_no_tools():
    text = "Nessun tool qui, solo testo normale."
    clean, calls = _parse_tool_calls(text)

    assert len(calls) == 0
    assert clean == text


def test_parse_tool_calls_invalid_json():
    text = '```tool\nnot valid json\n```'
    clean, calls = _parse_tool_calls(text)

    assert len(calls) == 0
    assert "```tool" not in clean
