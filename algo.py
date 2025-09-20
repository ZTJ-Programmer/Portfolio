# --- Save as: userstrategies/ict_fvg_bos_choch.py

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy


class ICT_FVG_BOS_CHOCH(IStrategy):
    """
    ICT-style strategy for Freqtrade
    - Detects fractal-1 swing highs/lows
    - BOS (close beyond last swing) & CHoCH (first BOS opposite previous BOS)
    - Wick-to-wick 3-candle Fair Value Gaps (ICT)
    - Entries: Retrace into recent FVG in direction of current BOS bias
    - Exits: Opposite BOS/CHOCH or ROI/stoploss

    Works on any timeframe; start with 5m/15m for testing.
    """

    # === Basic config ===
    timeframe: str = "5m"
    can_short: bool = True               # Set True only if your config/exchange supports shorting
    process_only_new_candles: bool = True
    startup_candle_count: int = 100

    # === Risk / ROI (defaults; tune via hyperopt) ===
    minimal_roi = {
        "0": 0.10,        # take partials quickly by default
    }
    stoploss = -0.02

    # A simple trailing stop to let runners go after first push
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optional plotting hints (freqAI/plotting tools may use this)
    plot_config = {
        "main_plot": {
            "last_sh_price": {"color": "orange"},
            "last_sl_price": {"color": "orange"},
            "bull_fvg_low": {"color": "green"},
            "bull_fvg_high": {"color": "green"},
            "bear_fvg_low": {"color": "red"},
            "bear_fvg_high": {"color": "red"},
        },
        "subplots": {
            "Structure": {
                "bos_bull": {"color": "green"},
                "bos_bear": {"color": "red"},
                "choch_bull": {"color": "green"},
                "choch_bear": {"color": "red"},
                "bias": {"color": "blue"},
            }
        },
    }

    def informative_pairs(self) -> List:
        return []

    # --------- Utilities

    @staticmethod
    def _fractal_swings(df: DataFrame) -> DataFrame:
        # Fractal-1: strict neighbors
        df["swing_high"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
        df["swing_low"]  = (df["low"]  < df["low"].shift(1))  & (df["low"]  < df["low"].shift(-1))

        # Last swing prices (forward-filled)
        df["sh_price"] = np.where(df["swing_high"], df["high"], np.nan)
        df["sl_price"] = np.where(df["swing_low"],  df["low"],  np.nan)

        df["last_sh_price"] = df["sh_price"].ffill()
        df["last_sl_price"] = df["sl_price"].ffill()
        return df

    @staticmethod
    def _bos_choch(df: DataFrame) -> DataFrame:
        # BOS occurs when we close beyond the last swing (using values known at bar open => shift the ref)
        ref_sh = df["last_sh_price"].shift(1)
        ref_sl = df["last_sl_price"].shift(1)

        df["bos_bull"] = (df["close"] > ref_sh) & (df["close"].shift(1) <= ref_sh)
        df["bos_bear"] = (df["close"] < ref_sl) & (df["close"].shift(1) >= ref_sl)

        # Direction stream: 1 for bull BOS, -1 for bear BOS, 0 otherwise
        bos_dir = np.where(df["bos_bull"], 1, np.where(df["bos_bear"], -1, 0)).astype(float)
        df["bos_dir"] = bos_dir

        # Last non-zero BOS dir before current bar
        last_bos_dir = pd.Series(bos_dir).replace(0, np.nan).ffill().shift(1)
        df["last_bos_dir"] = last_bos_dir.fillna(0)

        # CHoCH = BOS in the opposite direction of the previous BOS
        df["choch_bull"] = df["bos_bull"] & (df["last_bos_dir"] == -1)
        df["choch_bear"] = df["bos_bear"] & (df["last_bos_dir"] == 1)

        # Bias = last non-zero BOS dir carried forward (1, -1, or 0 if none yet)
        df["bias"] = pd.Series(bos_dir).replace(0, np.nan).ffill().fillna(0)

        return df

    @staticmethod
    def _fvg(df: DataFrame, max_age_bars: int = 20) -> DataFrame:
        """
        ICT wick-to-wick FVG across 3 candles.
        We store the most recent FVG zone (low/high) and its bar age.
        The zone is anchored on the MIDDLE candle of the 3-candle sequence.
        """
        # Use i-1, i, i+1 with FVG marked at i (the middle bar)
        bull_condition = df["high"].shift(1) < df["low"].shift(-1)
        bear_condition = df["low"].shift(1)  > df["high"].shift(-1)

        df["bull_fvg_low_raw"]  = np.where(bull_condition, df["high"].shift(1), np.nan)
        df["bull_fvg_high_raw"] = np.where(bull_condition, df["low"].shift(-1),  np.nan)

        df["bear_fvg_low_raw"]  = np.where(bear_condition, df["high"].shift(-1), np.nan)
        df["bear_fvg_high_raw"] = np.where(bear_condition, df["low"].shift(1),  np.nan)

        # Track the most recent zone + age
        df["bar_index"] = np.arange(len(df), dtype=float)

        # Bull FVG
        df["bull_fvg_low"]  = df["bull_fvg_low_raw"].ffill()
        df["bull_fvg_high"] = df["bull_fvg_high_raw"].ffill()
        bull_start = np.where(bull_condition, df["bar_index"], np.nan)
        df["bull_fvg_start"] = pd.Series(bull_start).ffill()
        df["bull_fvg_age"]   = df["bar_index"] - df["bull_fvg_start"]
        df["bull_fvg_valid"] = df["bull_fvg_age"] <= max_age_bars

        # Bear FVG
        df["bear_fvg_low"]   = df["bear_fvg_low_raw"].ffill()
        df["bear_fvg_high"]  = df["bear_fvg_high_raw"].ffill()
        bear_start = np.where(bear_condition, df["bar_index"], np.nan)
        df["bear_fvg_start"] = pd.Series(bear_start).ffill()
        df["bear_fvg_age"]   = df["bar_index"] - df["bear_fvg_start"]
        df["bear_fvg_valid"] = df["bear_fvg_age"] <= max_age_bars

        # Sanity: ensure low <= high for zones
        df.loc[df["bull_fvg_low"] > df["bull_fvg_high"], ["bull_fvg_low", "bull_fvg_high"]] = np.nan
        df.loc[df["bear_fvg_low"] > df["bear_fvg_high"], ["bear_fvg_low", "bear_fvg_high"]] = np.nan
        return df

    @staticmethod
    def _inside_zone(price_low: pd.Series, price_high: pd.Series,
                     z_low: pd.Series, z_high: pd.Series) -> pd.Series:
        """
        "Tap" or retrace into a zone (any overlap between bar range and zone).
        """
        return (price_low <= z_high) & (price_high >= z_low)

    # --------- Freqtrade hooks

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df = dataframe.copy()

        # Basic body/volatility measures (optional filters you can use later)
        df["body"] = (df["close"] - df["open"]).abs()
        df["atr_like"] = (df["high"] - df["low"]).rolling(14).mean()

        df = self._fractal_swings(df)
        df = self._bos_choch(df)
        df = self._fvg(df, max_age_bars=20)

        # Session filter example (Killzone-ish): 07:00–12:00 UTC (tune or disable)
        # Works if 'date' column is timezone-aware; if not, adjust to your feed's timezone.
        if "date" in df.columns:
            hours = pd.to_datetime(df["date"]).dt.hour
            df["session_ok"] = (hours >= 7) & (hours <= 12)
        else:
            df["session_ok"] = True

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df = dataframe.copy()

        # Bullish: trade with bullish bias and on retracement into a recent bullish FVG
        long_bias = df["bias"] == 1
        long_retrace = self._inside_zone(df["low"], df["high"], df["bull_fvg_low"], df["bull_fvg_high"]) & df["bull_fvg_valid"]

        # Extra confirmations (optional): displacement flavor – body > ATR-like mean
        long_impulse = df["body"] > (df["atr_like"].fillna(df["body"].median()))

        long_conditions = [
            long_bias,
            long_retrace,
            long_impulse,
            df["session_ok"],
        ]

        df.loc[np.logical_and.reduce(long_conditions), "enter_long"] = 1
        df.loc[np.logical_and.reduce(long_conditions), "enter_tag"] = "ICT: FVG retest (bull)"

        # Bearish (requires shorting enabled): bias bearish + retrace into recent bear FVG
        if self.can_short:
            short_bias = df["bias"] == -1
            short_retrace = self._inside_zone(df["low"], df["high"], df["bear_fvg_low"], df["bear_fvg_high"]) & df["bear_fvg_valid"]
            short_impulse = df["body"] > (df["atr_like"].fillna(df["body"].median()))

            short_conditions = [
                short_bias,
                short_retrace,
                short_impulse,
                df["session_ok"],
            ]

            df.loc[np.logical_and.reduce(short_conditions), "enter_short"] = 1
            df.loc[np.logical_and.reduce(short_conditions), "enter_tag"] = "ICT: FVG retest (bear)"

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df = dataframe.copy()

        # Exit longs on opposite BOS/CHOCH (structure shift against the position)
        df.loc[(df["bos_bear"] | df["choch_bear"]).fillna(False), "exit_long"] = 1
        df.loc[(df["bos_bear"] | df["choch_bear"]).fillna(False), "exit_tag"] = "Opposite BOS/CHOCH"

        # Exit shorts on opposite BOS/CHOCH
        if self.can_short:
            df.loc[(df["bos_bull"] | df["choch_bull"]).fillna(False), "exit_short"] = 1
            df.loc[(df["bos_bull"] | df["choch_bull"]).fillna(False), "exit_tag"] = "Opposite BOS/CHOCH"

        return df