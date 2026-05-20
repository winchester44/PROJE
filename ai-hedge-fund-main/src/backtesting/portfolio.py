from __future__ import annotations

from typing import Dict, Mapping
from types import MappingProxyType

from .types import PortfolioSnapshot, PositionState, TickerRealizedGains


class Portfolio:
    """Portfolio state management for backtesting operations.

    Encapsulates cash, positions, and margin tracking.
    Supports both long and short positions with proper cost basis tracking
    and realized gains/losses calculation.
    """

    def __init__(
        self,
        *,
        tickers: list[str],
        initial_cash: float,
        margin_requirement: float,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        self._commission_pct = commission_pct    # İşlem başı komisyon oranı (0.001 = %0.1)
        self._slippage_pct = slippage_pct        # Fiyat kayması oranı (0.0005 = %0.05)
        self._total_commissions = 0.0            # Toplam ödenen komisyon
        self._total_slippage_cost = 0.0          # Toplam slippage maliyeti
        self._portfolio: PortfolioSnapshot = {
            "cash": float(initial_cash),
            "margin_used": 0.0,
            "margin_requirement": float(margin_requirement),
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                }
                for ticker in tickers
            },
            "realized_gains": {
                ticker: {"long": 0.0, "short": 0.0}
                for ticker in tickers
            },
        }

    def get_snapshot(self) -> PortfolioSnapshot:
        positions_copy: Dict[str, PositionState] = {
            t: {
                "long": p["long"],
                "short": p["short"],
                "long_cost_basis": p["long_cost_basis"],
                "short_cost_basis": p["short_cost_basis"],
                "short_margin_used": p["short_margin_used"],
            }
            for t, p in self._portfolio["positions"].items()
        }
        gains_copy: Dict[str, TickerRealizedGains] = {
            t: {"long": g["long"], "short": g["short"]}
            for t, g in self._portfolio["realized_gains"].items()
        }
        return {
            "cash": float(self._portfolio["cash"]),
            "margin_used": float(self._portfolio["margin_used"]),
            "margin_requirement": float(self._portfolio["margin_requirement"]),
            "positions": positions_copy,
            "realized_gains": gains_copy,
        }

    def get_cash(self) -> float:
        return float(self._portfolio["cash"])

    def get_margin_used(self) -> float:
        return float(self._portfolio["margin_used"])

    def get_margin_requirement(self) -> float:
        return float(self._portfolio["margin_requirement"])

    def get_positions(self) -> Mapping[str, PositionState]:
        return MappingProxyType(self._portfolio["positions"])  # type: ignore[arg-type]

    def get_realized_gains(self) -> Mapping[str, TickerRealizedGains]:
        return MappingProxyType(self._portfolio["realized_gains"])  # type: ignore[arg-type]

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Slippage uygular: alımda fiyat yukarı, satımda aşağı kayar."""
        if is_buy:
            return price * (1.0 + self._slippage_pct)
        else:
            return price * (1.0 - self._slippage_pct)

    def _deduct_commission(self, trade_value: float) -> float:
        """Komisyon hesaplar ve nakit bakiyeden düşer."""
        commission = trade_value * self._commission_pct
        self._portfolio["cash"] -= commission
        self._total_commissions += commission
        return commission

    def get_total_commissions(self) -> float:
        """Toplam ödenen komisyon miktarı."""
        return self._total_commissions

    def get_total_slippage_cost(self) -> float:
        """Toplam slippage maliyeti."""
        return self._total_slippage_cost

    def apply_long_buy(self, ticker: str, quantity: int, price: float) -> int:
        if quantity <= 0:
            return 0
        quantity = int(quantity)
        position = self._portfolio["positions"][ticker]
        # Slippage: alımda fiyat yukarı kayar
        effective_price = self._apply_slippage(price, is_buy=True)
        slippage_cost = (effective_price - price) * quantity
        self._total_slippage_cost += slippage_cost
        cost = quantity * effective_price
        if cost <= self._portfolio["cash"]:
            old_shares = position["long"]
            old_cost_basis = position["long_cost_basis"]
            total_shares = old_shares + quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_shares
                total_new_cost = cost
                position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["long"] = old_shares + quantity
            self._portfolio["cash"] -= cost
            self._deduct_commission(cost)  # Komisyon düş
            return quantity
        max_quantity = int(self._portfolio["cash"] / effective_price) if effective_price > 0 else 0
        if max_quantity > 0:
            cost = max_quantity * effective_price
            old_shares = position["long"]
            old_cost_basis = position["long_cost_basis"]
            total_shares = old_shares + max_quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_shares
                total_new_cost = cost
                position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["long"] = old_shares + max_quantity
            self._portfolio["cash"] -= cost
            self._deduct_commission(cost)  # Komisyon düş
            return max_quantity
        return 0

    def apply_long_sell(self, ticker: str, quantity: int, price: float) -> int:
        position = self._portfolio["positions"][ticker]
        quantity = min(int(quantity), position["long"]) if quantity > 0 else 0
        if quantity <= 0:
            return 0
        # Slippage: satımda fiyat aşağı kayar
        effective_price = self._apply_slippage(price, is_buy=False)
        slippage_cost = (price - effective_price) * quantity
        self._total_slippage_cost += slippage_cost
        avg_cost = position["long_cost_basis"] if position["long"] > 0 else 0.0
        realized_gain = (effective_price - avg_cost) * quantity
        self._portfolio["realized_gains"][ticker]["long"] += realized_gain
        position["long"] -= quantity
        proceeds = quantity * effective_price
        self._portfolio["cash"] += proceeds
        self._deduct_commission(proceeds)  # Komisyon düş
        if position["long"] == 0:
            position["long_cost_basis"] = 0.0
        return quantity

    def apply_short_open(self, ticker: str, quantity: int, price: float) -> int:
        if quantity <= 0:
            return 0
        quantity = int(quantity)
        position = self._portfolio["positions"][ticker]
        # Slippage: short açarken fiyat aşağı kayar (daha kötü fiyat)
        effective_price = self._apply_slippage(price, is_buy=False)
        slippage_cost = (price - effective_price) * quantity
        self._total_slippage_cost += slippage_cost
        proceeds = effective_price * quantity
        margin_ratio = self._portfolio["margin_requirement"]
        margin_required = proceeds * margin_ratio
        available_cash = max(
            0.0, self._portfolio["cash"] - self._portfolio["margin_used"]
        )
        if margin_required <= available_cash:
            old_short_shares = position["short"]
            old_cost_basis = position["short_cost_basis"]
            total_shares = old_short_shares + quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_short_shares
                total_new_cost = effective_price * quantity
                position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["short"] = old_short_shares + quantity
            position["short_margin_used"] += margin_required
            self._portfolio["margin_used"] += margin_required
            self._portfolio["cash"] += proceeds
            self._portfolio["cash"] -= margin_required
            self._deduct_commission(proceeds)  # Komisyon düş
            return quantity
        max_quantity = int(available_cash / (effective_price * margin_ratio)) if margin_ratio > 0 and effective_price > 0 else 0
        if max_quantity > 0:
            proceeds = effective_price * max_quantity
            margin_required = proceeds * margin_ratio
            old_short_shares = position["short"]
            old_cost_basis = position["short_cost_basis"]
            total_shares = old_short_shares + max_quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_short_shares
                total_new_cost = effective_price * max_quantity
                position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["short"] = old_short_shares + max_quantity
            position["short_margin_used"] += margin_required
            self._portfolio["margin_used"] += margin_required
            self._portfolio["cash"] += proceeds
            self._portfolio["cash"] -= margin_required
            self._deduct_commission(proceeds)  # Komisyon düş
            return max_quantity
        return 0

    def apply_short_cover(self, ticker: str, quantity: int, price: float) -> int:
        position = self._portfolio["positions"][ticker]
        quantity = min(int(quantity), position["short"]) if quantity > 0 else 0
        if quantity <= 0:
            return 0
        # Slippage: cover ederken fiyat yukarı kayar (daha kötü fiyat)
        effective_price = self._apply_slippage(price, is_buy=True)
        slippage_cost = (effective_price - price) * quantity
        self._total_slippage_cost += slippage_cost
        cover_cost = quantity * effective_price
        avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0.0
        realized_gain = (avg_short_price - effective_price) * quantity
        if position["short"] > 0:
            portion = quantity / position["short"]
        else:
            portion = 1.0
        margin_to_release = portion * position["short_margin_used"]
        position["short"] -= quantity
        position["short_margin_used"] -= margin_to_release
        self._portfolio["margin_used"] -= margin_to_release
        self._portfolio["cash"] += margin_to_release
        self._portfolio["cash"] -= cover_cost
        self._deduct_commission(cover_cost)  # Komisyon düş
        self._portfolio["realized_gains"][ticker]["short"] += realized_gain
        if position["short"] == 0:
            position["short_cost_basis"] = 0.0
            position["short_margin_used"] = 0.0
        return quantity

