"""
Multi-Agent Debate Engine (Ajanlar Arası Tartışma Motoru)
==========================================================
Karşıt yatırım felsefelerine sahip ajanların (ör. Warren Buffett vs Cathie Wood)
çıktılarını alıp çaprazlamalı bir tartışma (debate) simüle eder.

Akış:
  1. Her ajan bağımsız analiz çıktısı üretir (signal + reasoning)
  2. Round 1: Her ajan, diğerinin argümanını görür ve kendi görüşünü revize eder
  3. Round 2: Revize edilmiş argümanlar tekrar çaprazlanır (final pozisyon)
  4. Risk Manager ajanı nihai kararı verir
  5. Sonuç: Birleştirilmiş debate transcript + final karar

Kullanım:
    from src.agents.debate_engine import run_multi_agent_debate
    result = run_multi_agent_debate(signals_by_ticker, state)
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.utils.llm import call_llm
from src.graph.state import AgentState


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────

class DebateArgument(BaseModel):
    """Bir ajanın debate turundaki argümanı."""
    agent_name: str = Field(description="Ajan adı (ör. warren_buffett)")
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Güven skoru 0-100")
    argument: str = Field(description="Argüman metni (max 300 karakter)")
    rebuttal: str = Field(description="Karşı argümana yanıt (max 200 karakter)")


class DebateRoundOutput(BaseModel):
    """Debate turundaki revize edilmiş pozisyon."""
    revised_signal: Literal["bullish", "bearish", "neutral"]
    revised_confidence: int = Field(description="Revize edilmiş güven 0-100")
    reasoning: str = Field(description="Neden görüşünü değiştirdin veya korudun?")


class DebateVerdict(BaseModel):
    """Risk Manager'ın nihai kararı."""
    final_signal: Literal["bullish", "bearish", "neutral"]
    final_confidence: int = Field(description="Final güven 0-100")
    consensus_summary: str = Field(description="Tartışma özeti ve karar gerekçesi")


# ──────────────────────────────────────────────────────────────────────────────
# Ajan Grupları (Büyüme vs Değer)
# ──────────────────────────────────────────────────────────────────────────────

# Büyüme odaklı ajanlar (Bull piyasada güçlü)
GROWTH_AGENTS = {
    "cathie_wood_agent", "growth_analyst_agent",
    "peter_lynch_agent", "phil_fisher_agent",
    "stanley_druckenmiller_agent",
}

# Değer/Kontrarian odaklı ajanlar (Bear piyasada güçlü)
VALUE_AGENTS = {
    "warren_buffett_agent", "ben_graham_agent",
    "charlie_munger_agent", "michael_burry_agent",
    "aswath_damodaran_agent", "mohnish_pabrai_agent",
    "nassim_taleb_agent",
}


# ──────────────────────────────────────────────────────────────────────────────
# DEBATE ROUND FONKSIYONU
# ──────────────────────────────────────────────────────────────────────────────

def _run_debate_round(
    agent_name: str,
    agent_signal: dict,
    opponent_name: str,
    opponent_signal: dict,
    round_number: int,
    ticker: str,
    state: AgentState,
) -> DebateRoundOutput:
    """
    Bir ajanı, rakibinin argümanıyla karşılaştırıp revize edilmiş
    pozisyonunu üretmesini ister.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are {agent_name}, a financial analyst in a structured debate.\n"
            "This is Round {round_number} of the debate about {ticker}.\n\n"
            "YOUR ORIGINAL POSITION:\n"
            "Signal: {own_signal} (Confidence: {own_confidence}%)\n"
            "Reasoning: {own_reasoning}\n\n"
            "YOUR OPPONENT ({opponent_name}) ARGUES:\n"
            "Signal: {opp_signal} (Confidence: {opp_confidence}%)\n"
            "Reasoning: {opp_reasoning}\n\n"
            "INSTRUCTIONS:\n"
            "- Consider your opponent's argument carefully\n"
            "- You may adjust your signal and confidence if their argument is compelling\n"
            "- Maintain intellectual honesty — don't change your view without good reason\n"
            "- Keep reasoning concise (max 200 chars)\n"
            "- Return JSON only"
        ),
        (
            "human",
            "Respond with your revised position after considering the opponent's argument."
        ),
    ])

    # Reasoning'i string'e çevir
    own_reasoning = agent_signal.get("reasoning", "")
    if isinstance(own_reasoning, dict):
        own_reasoning = json.dumps(own_reasoning, ensure_ascii=False)[:300]

    opp_reasoning = opponent_signal.get("reasoning", "")
    if isinstance(opp_reasoning, dict):
        opp_reasoning = json.dumps(opp_reasoning, ensure_ascii=False)[:300]

    prompt = template.invoke({
        "agent_name": agent_name.replace("_agent", "").replace("_", " ").title(),
        "round_number": round_number,
        "ticker": ticker,
        "own_signal": agent_signal.get("signal", "neutral"),
        "own_confidence": agent_signal.get("confidence", 50),
        "own_reasoning": own_reasoning,
        "opponent_name": opponent_name.replace("_agent", "").replace("_", " ").title(),
        "opp_signal": opponent_signal.get("signal", "neutral"),
        "opp_confidence": opponent_signal.get("confidence", 50),
        "opp_reasoning": opp_reasoning,
    })

    def default_factory():
        return DebateRoundOutput(
            revised_signal=agent_signal.get("signal", "neutral"),
            revised_confidence=agent_signal.get("confidence", 50),
            reasoning="Debate round failed, maintaining original position."
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=DebateRoundOutput,
        agent_name=f"debate_{agent_name}",
        state=state,
        default_factory=default_factory,
    )


# ──────────────────────────────────────────────────────────────────────────────
# RISK MANAGER FINAL VERDICT
# ──────────────────────────────────────────────────────────────────────────────

def _risk_manager_verdict(
    ticker: str,
    debate_transcript: list[dict],
    state: AgentState,
) -> DebateVerdict:
    """Risk Manager ajanı, tüm debate transcript'ini inceleyip nihai kararı verir."""

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the Risk Manager. You have reviewed a structured debate between "
            "multiple financial analysts about {ticker}.\n\n"
            "Your job is to synthesize all arguments and provide the FINAL trading decision.\n"
            "Consider:\n"
            "- Strength of each analyst's argument\n"
            "- How much they adjusted after hearing counterarguments\n"
            "- Risk/reward balance\n"
            "- Market conditions\n\n"
            "Return a final signal (bullish/bearish/neutral), confidence (0-100), "
            "and a consensus summary. Keep summary under 300 chars. Return JSON only."
        ),
        (
            "human",
            "Debate transcript for {ticker}:\n{transcript}\n\n"
            "Provide your final verdict."
        ),
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "transcript": json.dumps(debate_transcript, ensure_ascii=False, indent=1),
    })

    def default_factory():
        return DebateVerdict(
            final_signal="neutral",
            final_confidence=50,
            consensus_summary="Debate inconclusive, defaulting to neutral."
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=DebateVerdict,
        agent_name="debate_risk_manager",
        state=state,
        default_factory=default_factory,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ANA FONKSİYON: Multi-Agent Debate
# ──────────────────────────────────────────────────────────────────────────────

def run_multi_agent_debate(
    signals_by_ticker: dict[str, dict[str, dict]],
    state: AgentState,
    debate_rounds: int = 2,
) -> dict[str, dict]:
    """
    Tüm ticker'lar için multi-agent debate çalıştırır.

    Parameters
    ----------
    signals_by_ticker : dict
        Her ticker için ajanların sinyallerini içeren sözlük.
        Yapı: {ticker: {agent_name: {"signal": ..., "confidence": ..., "reasoning": ...}}}
    state : AgentState
        LangGraph state (LLM çağrıları için gerekli)
    debate_rounds : int
        Debate tur sayısı (varsayılan: 2)

    Returns
    -------
    dict
        Her ticker için debate sonucu:
        {ticker: {"final_signal": ..., "final_confidence": ..., "consensus_summary": ..., "transcript": [...]}}
    """
    results = {}

    for ticker, agent_signals in signals_by_ticker.items():
        # En az 2 ajan gerekli
        agent_names = list(agent_signals.keys())
        if len(agent_names) < 2:
            results[ticker] = {
                "final_signal": agent_signals.get(agent_names[0], {}).get("signal", "neutral"),
                "final_confidence": agent_signals.get(agent_names[0], {}).get("confidence", 50),
                "consensus_summary": "Tek ajan olduğu için debate yapılamadı.",
                "transcript": [],
            }
            continue

        # Büyüme vs Değer ajanlarını ayır
        growth_agents = [a for a in agent_names if a in GROWTH_AGENTS]
        value_agents = [a for a in agent_names if a in VALUE_AGENTS]
        other_agents = [a for a in agent_names if a not in GROWTH_AGENTS and a not in VALUE_AGENTS]

        # Karşıt çiftler oluştur (büyüme vs değer)
        debate_pairs = []
        min_pairs = min(len(growth_agents), len(value_agents))
        for i in range(min_pairs):
            debate_pairs.append((growth_agents[i], value_agents[i]))

        # Kalan ajanları da ekle (birbirleriyle tartıştır)
        remaining = growth_agents[min_pairs:] + value_agents[min_pairs:] + other_agents
        for i in range(0, len(remaining) - 1, 2):
            debate_pairs.append((remaining[i], remaining[i + 1]))

        if not debate_pairs:
            # Çift oluşturulamadıysa ilk ikiyi eşleştir
            debate_pairs.append((agent_names[0], agent_names[1]))

        # Debate Transcript
        transcript = []

        # Mevcut sinyalleri kopyala (round boyunca güncellenir)
        current_signals = {
            name: dict(sig) for name, sig in agent_signals.items()
        }

        for round_num in range(1, debate_rounds + 1):
            for agent_a, agent_b in debate_pairs:
                sig_a = current_signals.get(agent_a, {"signal": "neutral", "confidence": 50, "reasoning": "N/A"})
                sig_b = current_signals.get(agent_b, {"signal": "neutral", "confidence": 50, "reasoning": "N/A"})

                # Ajan A, Ajan B'nin argümanını görür
                revised_a = _run_debate_round(
                    agent_a, sig_a, agent_b, sig_b, round_num, ticker, state
                )

                # Ajan B, Ajan A'nın argümanını görür
                revised_b = _run_debate_round(
                    agent_b, sig_b, agent_a, sig_a, round_num, ticker, state
                )

                # Güncel sinyalleri revize et
                current_signals[agent_a] = {
                    "signal": revised_a.revised_signal,
                    "confidence": revised_a.revised_confidence,
                    "reasoning": revised_a.reasoning,
                }
                current_signals[agent_b] = {
                    "signal": revised_b.revised_signal,
                    "confidence": revised_b.revised_confidence,
                    "reasoning": revised_b.reasoning,
                }

                transcript.append({
                    "round": round_num,
                    "agent_a": agent_a,
                    "agent_a_revised": current_signals[agent_a],
                    "agent_b": agent_b,
                    "agent_b_revised": current_signals[agent_b],
                })

        # Risk Manager nihai kararı
        verdict = _risk_manager_verdict(ticker, transcript, state)

        results[ticker] = {
            "final_signal": verdict.final_signal,
            "final_confidence": verdict.final_confidence,
            "consensus_summary": verdict.consensus_summary,
            "transcript": transcript,
            "post_debate_signals": current_signals,
        }

    return results
