"""
kabu STATION API ラッパー（三菱UFJ eスマート証券）

必要な環境変数 (.env):
  KABU_STATION_API_BASE_URL      例: http://localhost:18080/kabusapi
  KABU_STATION_API_PASSWORD      APIパスワード
  KABU_STATION_TRADING_PASSWORD  注文パスワード
  KABU_STATION_EXCHANGE          取引所コード (1=東証)
  KABU_STATION_ACCOUNT_TYPE      口座種別 (2=特定)

使い方:
  client = KabuClient()
  client.get_token()              # トークン取得
  board = client.board("7203")    # トヨタの板情報
  positions = client.positions()  # 保有ポジション
  client.order_market_buy("7203", 100)  # 成行買い 100株
"""
from __future__ import annotations
import os
import sys
import time
import logging
import requests
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    KABU_API_BASE_URL, KABU_API_PASSWORD, KABU_TRADING_PASS,
    KABU_EXCHANGE, KABU_ACCOUNT_TYPE,
)

logger = logging.getLogger(__name__)


class KabuAPIError(Exception):
    pass


class KabuClient:
    """kabu STATION REST API クライアント"""

    def __init__(
        self,
        base_url:  str = KABU_API_BASE_URL,
        password:  str = KABU_API_PASSWORD,
        trade_pass: str = KABU_TRADING_PASS,
        exchange:  str = KABU_EXCHANGE,
        account_type: str = KABU_ACCOUNT_TYPE,
    ):
        self.base_url    = base_url.rstrip("/")
        self.password    = password
        self.trade_pass  = trade_pass
        self.exchange    = int(exchange)
        self.account_type = int(account_type)
        self._token: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────
    # 認証
    # ──────────────────────────────────────────────────────────────────

    def get_token(self) -> str:
        """APIトークンを取得・キャッシュする"""
        url = f"{self.base_url}/token"
        resp = requests.post(url, json={"APIPassword": self.password}, timeout=10)
        self._handle_error(resp)
        self._token = resp.json()["Token"]
        logger.info("kabu STATION token acquired")
        return self._token

    @property
    def _headers(self) -> Dict[str, str]:
        if not self._token:
            self.get_token()
        return {"X-API-KEY": self._token, "Content-Type": "application/json"}

    # ──────────────────────────────────────────────────────────────────
    # 市場情報
    # ──────────────────────────────────────────────────────────────────

    def board(self, ticker: str) -> Dict[str, Any]:
        """板情報取得（リアルタイム）"""
        symbol = f"{ticker}@{self.exchange}"
        resp = requests.get(
            f"{self.base_url}/board/{symbol}",
            headers=self._headers, timeout=10
        )
        self._handle_error(resp)
        return resp.json()

    def symbol_info(self, ticker: str) -> Dict[str, Any]:
        """銘柄情報取得"""
        symbol = f"{ticker}@{self.exchange}"
        resp = requests.get(
            f"{self.base_url}/symbol/{symbol}",
            headers=self._headers, timeout=10
        )
        self._handle_error(resp)
        return resp.json()

    def prices_bulk(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """複数銘柄の現在値取得"""
        results = []
        for tk in tickers:
            try:
                b = self.board(tk)
                results.append({
                    "ticker": tk,
                    "price": b.get("CurrentPrice"),
                    "bid":   b.get("BidPrice"),
                    "ask":   b.get("AskPrice"),
                    "volume": b.get("TradingVolume"),
                })
            except Exception as e:
                logger.warning(f"board {tk}: {e}")
            time.sleep(0.1)
        return results

    # ──────────────────────────────────────────────────────────────────
    # ポジション・残高
    # ──────────────────────────────────────────────────────────────────

    def positions(self) -> List[Dict[str, Any]]:
        """保有ポジション一覧"""
        resp = requests.get(
            f"{self.base_url}/positions",
            headers=self._headers, timeout=10
        )
        self._handle_error(resp)
        return resp.json()

    def wallet_cash(self) -> Dict[str, Any]:
        """現金残高"""
        resp = requests.get(
            f"{self.base_url}/wallet/cash",
            headers=self._headers, timeout=10
        )
        self._handle_error(resp)
        return resp.json()

    def wallet_margin(self) -> Dict[str, Any]:
        """信用建余力"""
        resp = requests.get(
            f"{self.base_url}/wallet/margin",
            headers=self._headers, timeout=10
        )
        self._handle_error(resp)
        return resp.json()

    # ──────────────────────────────────────────────────────────────────
    # 注文
    # ──────────────────────────────────────────────────────────────────

    def _send_order(self, payload: dict) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/sendorder",
            json=payload,
            headers=self._headers,
            timeout=15
        )
        self._handle_error(resp)
        return resp.json()

    def order_market_buy(self, ticker: str, qty: int, expire_day: int = 0) -> Dict:
        """成行買い注文"""
        return self._send_order({
            "Password":   self.trade_pass,
            "Symbol":     ticker,
            "Exchange":   self.exchange,
            "SecurityType": 1,          # 1=株式
            "Side":       "2",          # 2=買い
            "CashMargin": 1,            # 1=現物
            "DelivType":  2,            # 2=自動振替
            "AccountType": self.account_type,
            "Qty":        qty,
            "FrontOrderType": 10,       # 10=成行
            "Price":      0,
            "ExpireDay":  expire_day,   # 0=当日
        })

    def order_market_sell(self, ticker: str, qty: int, expire_day: int = 0) -> Dict:
        """成行売り注文"""
        return self._send_order({
            "Password":   self.trade_pass,
            "Symbol":     ticker,
            "Exchange":   self.exchange,
            "SecurityType": 1,
            "Side":       "1",          # 1=売り
            "CashMargin": 1,
            "DelivType":  2,
            "AccountType": self.account_type,
            "Qty":        qty,
            "FrontOrderType": 10,
            "Price":      0,
            "ExpireDay":  expire_day,
        })

    def order_limit_buy(self, ticker: str, qty: int, price: float, expire_day: int = 0) -> Dict:
        """指値買い注文"""
        return self._send_order({
            "Password":   self.trade_pass,
            "Symbol":     ticker,
            "Exchange":   self.exchange,
            "SecurityType": 1,
            "Side":       "2",
            "CashMargin": 1,
            "DelivType":  2,
            "AccountType": self.account_type,
            "Qty":        qty,
            "FrontOrderType": 20,       # 20=指値
            "Price":      price,
            "ExpireDay":  expire_day,
        })

    def order_limit_sell(self, ticker: str, qty: int, price: float, expire_day: int = 0) -> Dict:
        """指値売り注文"""
        return self._send_order({
            "Password":   self.trade_pass,
            "Symbol":     ticker,
            "Exchange":   self.exchange,
            "SecurityType": 1,
            "Side":       "1",
            "CashMargin": 1,
            "DelivType":  2,
            "AccountType": self.account_type,
            "Qty":        qty,
            "FrontOrderType": 20,
            "Price":      price,
            "ExpireDay":  expire_day,
        })

    def cancel_order(self, order_id: str) -> Dict:
        """注文取消"""
        resp = requests.put(
            f"{self.base_url}/cancelorder",
            json={"Password": self.trade_pass, "OrderId": order_id},
            headers=self._headers,
            timeout=10
        )
        self._handle_error(resp)
        return resp.json()

    def orders(self, product: int = 0) -> List[Dict]:
        """注文一覧取得 (product: 0=全部, 1=現物, 2=信用)"""
        resp = requests.get(
            f"{self.base_url}/orders?product={product}",
            headers=self._headers,
            timeout=10
        )
        self._handle_error(resp)
        return resp.json()

    # ──────────────────────────────────────────────────────────────────
    # ユーティリティ
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _handle_error(resp: requests.Response) -> None:
        if resp.status_code not in (200, 201):
            raise KabuAPIError(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            )

    def rebalance(
        self,
        target_weights: Dict[str, float],
        total_budget: float,
        min_trade_unit: int = 100,
        dry_run: bool = True,
    ) -> List[Dict]:
        """
        目標ウエイトに基づくリバランス注文。
        target_weights: {ticker: weight(0~1)}
        total_budget: 総投資額（円）
        min_trade_unit: 単元株数（デフォルト100）
        dry_run=True のとき注文は出さず計画のみ返す。
        """
        # 現在価格の取得
        tickers = list(target_weights.keys())
        prices_data = self.prices_bulk(tickers)
        price_map = {d["ticker"]: d["price"] for d in prices_data if d["price"]}

        orders_plan = []
        for ticker, weight in target_weights.items():
            if ticker not in price_map:
                logger.warning(f"No price for {ticker}")
                continue
            price  = price_map[ticker]
            budget = total_budget * weight
            qty_raw = budget / (price * min_trade_unit)
            qty    = int(qty_raw) * min_trade_unit

            if qty <= 0:
                continue

            plan = {"ticker": ticker, "qty": qty, "price": price, "side": "buy"}
            orders_plan.append(plan)

            if not dry_run:
                self.order_market_buy(ticker, qty)
                time.sleep(0.3)

        return orders_plan
