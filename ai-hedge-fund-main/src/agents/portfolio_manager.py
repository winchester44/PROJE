import json
import time
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

# ── Debate Engine & Dynamic Weighting ─────────────────────────────────────────
try:
    from src.agents.debate_engine import (
        run_multi_agent_debate,
        GROWTH_AGENTS,
        VALUE_AGENTS,
    )
    _DEBATE_AVAILABLE = True
except ImportError:
    _DEBATE_AVAILABLE = False
    GROWTH_AGENTS = set()
    VALUE_AGENTS = set()


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """Makes final trading decisions and generates orders for multiple tickers"""

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing analyst signals")

        # Find the corresponding risk manager for this portfolio manager
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split('_')[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # Fallback for CLI

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] // current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Compress analyst signals to {sig, conf}
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if not agent.startswith("risk_management_agent") and ticker in signals:
                sig = signals[ticker].get("signal")
                conf = signals[ticker].get("confidence")
                if sig is not None and conf is not None:
                    ticker_signals[agent] = {"sig": sig, "conf": conf}
        signals_by_ticker[ticker] = ticker_signals

    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Generating trading decisions")

    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        agent_id=agent_id,
        state=state,
    )
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()},
                             "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def compute_allowed_actions(
        tickers: list[str],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
) -> dict[str, dict[str, int]]:
    """Compute allowed actions and max quantities for each ticker deterministically."""
    allowed = {}
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {}) or {}
    margin_requirement = float(portfolio.get("margin_requirement", 0.5))
    margin_used = float(portfolio.get("margin_used", 0.0))
    equity = float(portfolio.get("equity", cash))

    for ticker in tickers:
        price = float(current_prices.get(ticker, 0.0))
        pos = positions.get(
            ticker,
            {"long": 0, "long_cost_basis": 0.0, "short": 0, "short_cost_basis": 0.0},
        )
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        max_qty = int(max_shares.get(ticker, 0) or 0)

        # Start with zeros
        actions = {"buy": 0, "sell": 0, "short": 0, "cover": 0, "hold": 0}

        # Long side
        if long_shares > 0:
            actions["sell"] = long_shares
        if cash > 0 and price > 0:
            max_buy_cash = int(cash // price)
            max_buy = max(0, min(max_qty, max_buy_cash))
            if max_buy > 0:
                actions["buy"] = max_buy

        # Short side
        if short_shares > 0:
            actions["cover"] = short_shares
        if price > 0 and max_qty > 0:
            if margin_requirement <= 0.0:
                # If margin requirement is zero or unset, only cap by max_qty
                max_short = max_qty
            else:
                available_margin = max(0.0, (equity / margin_requirement) - margin_used)
                max_short_margin = int(available_margin // price)
                max_short = max(0, min(max_qty, max_short_margin))
            if max_short > 0:
                actions["short"] = max_short

        # Hold always valid
        actions["hold"] = 0

        # Prune zero-capacity actions to reduce tokens, keep hold
        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v

        allowed[ticker] = pruned

    return allowed


def _compact_signals(signals_by_ticker: dict[str, dict]) -> dict[str, dict]:
    """Keep only {agent: {sig, conf}} and drop empty agents."""
    out = {}
    for t, agents in signals_by_ticker.items():
        if not agents:
            out[t] = {}
            continue
        compact = {}
        for agent, payload in agents.items():
            sig = payload.get("sig") or payload.get("signal")
            conf = payload.get("conf") if "conf" in payload else payload.get("confidence")
            if sig is not None and conf is not None:
                compact[agent] = {"sig": sig, "conf": conf}
        out[t] = compact
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MARKET REGIME DETECTION & DYNAMIC WEIGHTING
# ──────────────────────────────────────────────────────────────────────────────

def _detect_market_regime(signals_by_ticker: dict[str, dict]) -> str:
    """
    Sinyal dağılımından piyasa rejimini algılar.
    Returns: 'bull' | 'bear' | 'neutral'
    """
    bullish_count = 0
    bearish_count = 0
    total = 0

    for ticker, agents in signals_by_ticker.items():
        for agent, payload in agents.items():
            sig = payload.get("sig") or payload.get("signal", "")
            total += 1
            if sig == "bullish":
                bullish_count += 1
            elif sig == "bearish":
                bearish_count += 1

    if total == 0:
        return "neutral"

    bull_pct = bullish_count / total
    bear_pct = bearish_count / total

    if bull_pct > 0.55:
        return "bull"
    elif bear_pct > 0.55:
        return "bear"
    return "neutral"


def _apply_regime_weights(
    signals_by_ticker: dict[str, dict],
    regime: str,
) -> dict[str, dict]:
    """
    Piyasa rejimine göre ajanların güven skorlarını dinamik olarak ağırlıklandırır.

    Bull piyasa → büyüme ajanlarına 1.5x çarpan
    Bear piyasa → değer/kontrarian ajanlara 1.5x çarpan
    Nötr        → değişiklik yok
    """
    BOOST_FACTOR = 1.5
    DAMPEN_FACTOR = 0.75

    weighted = {}
    for ticker, agents in signals_by_ticker.items():
        weighted[ticker] = {}
        for agent, payload in agents.items():
            entry = dict(payload)  # kopyala
            conf_key = "conf" if "conf" in entry else "confidence"
            original_conf = entry.get(conf_key, 50)

            if regime == "bull":
                if agent in GROWTH_AGENTS:
                    entry[conf_key] = min(100, int(original_conf * BOOST_FACTOR))
                elif agent in VALUE_AGENTS:
                    entry[conf_key] = max(1, int(original_conf * DAMPEN_FACTOR))
            elif regime == "bear":
                if agent in VALUE_AGENTS:
                    entry[conf_key] = min(100, int(original_conf * BOOST_FACTOR))
                elif agent in GROWTH_AGENTS:
                    entry[conf_key] = max(1, int(original_conf * DAMPEN_FACTOR))

            weighted[ticker][agent] = entry
    return weighted


def generate_trading_decision(
        tickers: list[str],
        signals_by_ticker: dict[str, dict],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
        agent_id: str,
        state: AgentState,
) -> PortfolioManagerOutput:
    """Get decisions from the LLM with deterministic constraints and a minimal prompt."""

    # Deterministic constraints
    allowed_actions_full = compute_allowed_actions(tickers, current_prices, max_shares, portfolio)

    # Pre-fill pure holds to avoid sending them to the LLM at all
    prefilled_decisions: dict[str, PortfolioDecision] = {}
    tickers_for_llm: list[str] = []
    for t in tickers:
        aa = allowed_actions_full.get(t, {"hold": 0})
        # If only 'hold' key exists, there is no trade possible
        if set(aa.keys()) == {"hold"}:
            prefilled_decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=100.0, reasoning="No valid trade available"
            )
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return PortfolioManagerOutput(decisions=prefilled_decisions)

    # Build compact payloads only for tickers sent to LLM
    compact_signals = _compact_signals({t: signals_by_ticker.get(t, {}) for t in tickers_for_llm})
    compact_allowed = {t: allowed_actions_full[t] for t in tickers_for_llm}

    # ── Dinamik Rejim Algılama ve Ağırlıklandırma ─────────────────────────────
    regime = _detect_market_regime(compact_signals)
    if regime != "neutral":
        compact_signals = _apply_regime_weights(compact_signals, regime)

    # ── Opsiyonel Multi-Agent Debate ──────────────────────────────────────────
    debate_context = ""
    if _DEBATE_AVAILABLE and len(tickers_for_llm) > 0:
        try:
            # Debate için tam sinyalleri hazırla
            debate_signals = {
                t: {
                    agent: {
                        "signal": p.get("sig") or p.get("signal", "neutral"),
                        "confidence": p.get("conf") or p.get("confidence", 50),
                        "reasoning": p.get("reasoning", "N/A"),
                    }
                    for agent, p in agents.items()
                }
                for t, agents in compact_signals.items()
            }
            debate_results = run_multi_agent_debate(debate_signals, state, debate_rounds=2)
            debate_parts = []
            for t, res in debate_results.items():
                debate_parts.append(
                    f"{t}: Debate verdict={res['final_signal']} "
                    f"(conf={res['final_confidence']}%) — {res['consensus_summary']}"
                )
            debate_context = "\nDebate Results:\n" + "\n".join(debate_parts)
        except Exception:
            debate_context = ""  # Debate başarısız olursa sessizce atla

    # Minimal prompt template (CoT + Debate eklendi)
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a portfolio manager. Market regime: {regime}.\n\n"
                "DECISION FRAMEWORK (Chain of Thought):\n"
                "Step 1: MACRO — Assess the overall market regime (bull/bear/neutral).\n"
                "Step 2: SECTOR — Consider sector-specific signals from analysts.\n"
                "Step 3: STOCK — Weigh individual analyst signals and debate outcomes.\n"
                "Step 4: ACTION — Pick one allowed action per ticker and quantity ≤ max.\n\n"
                "Inputs per ticker: analyst signals (regime-weighted) and allowed actions.\n"
                "Keep reasoning very concise (max 150 chars). No cash or margin math. Return JSON only."
            ),
            (
                "human",
                "Signals:\n{signals}\n\n"
                "Allowed:\n{allowed}\n\n"
                "{debate_context}\n\n"
                "Format:\n"
                "{{\n"
                '  "decisions": {{\n'
                '    "TICKER": {{"action":"...","quantity":int,"confidence":int,"reasoning":"..."}}\n'
                "  }}\n"
                "}}"
            ),
        ]
    )

    prompt_data = {
        "signals": json.dumps(compact_signals, separators=(",", ":"), ensure_ascii=False),
        "allowed": json.dumps(compact_allowed, separators=(",", ":"), ensure_ascii=False),
        "regime": regime.upper(),
        "debate_context": debate_context,
    }
    prompt = template.invoke(prompt_data)

    # Default factory fills remaining tickers as hold if the LLM fails
    def create_default_portfolio_output():
        # start from prefilled
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=0.0, reasoning="Default decision: hold"
            )
        return PortfolioManagerOutput(decisions=decisions)

    llm_out = call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )

    # Merge prefilled holds with LLM results
    merged = dict(prefilled_decisions)
    merged.update(llm_out.decisions)
    return PortfolioManagerOutput(decisions=merged)
