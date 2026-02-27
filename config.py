"""
共通設定・環境変数ローダー
.envファイルから設定を読み込む。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .envを検索（親ディレクトリも探索）
for p in [Path(__file__).parent, Path(__file__).parent.parent]:
    env_path = p / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        break

# ── kabu STATION API (三菱UFJ eスマート証券) ──────────────────────────
KABU_API_BASE_URL   = os.getenv("KABU_STATION_API_BASE_URL", "http://localhost:18080/kabusapi")
KABU_API_PASSWORD   = os.getenv("KABU_STATION_API_PASSWORD", "")
KABU_TRADING_PASS   = os.getenv("KABU_STATION_TRADING_PASSWORD", "")
KABU_EXCHANGE       = os.getenv("KABU_STATION_EXCHANGE", "1")   # 1=東証
KABU_ACCOUNT_TYPE   = os.getenv("KABU_STATION_ACCOUNT_TYPE", "2")  # 2=特定

# ── データソース ────────────────────────────────────────────────────────
EDINET_API_KEY      = os.getenv("EDINET_API_KEY", "")
FRED_API_KEY        = os.getenv("FRED_API", "")

# ── 基本設定 ────────────────────────────────────────────────────────────
MARKET_TZ = "Asia/Tokyo"
PRICE_SUFFIX = ".T"           # yfinance用 JPX suffix
TOPIX_TICKER = "^N225"        # ベンチマーク（日経225）

# ── ユニバース定義 ──────────────────────────────────────────────────────
JPX_DATA_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JPX_EXCLUDE_TYPES = frozenset([
    "ETF・ETN", "PRO Market",
    "プライム（外国株式）", "スタンダード（外国株式）", "グロース（外国株式）",
    "REIT・ベンチャーファンド・カントリーファンド・インフラファンド",
    "出資証券",
])
