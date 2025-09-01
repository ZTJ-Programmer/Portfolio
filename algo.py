# strategies/multi_ind.py
from jesse.strategies import Strategy
import jesse.indicators as ta
import numpy as np


class MTF_Ultimate(Strategy):
    """
    Multi-Timeframe Ultimate Strategy — QUALITY over QUANTITY edition.

    Changes vs. original:
    • 1D regime gate to remove short bias in mixed regimes.
    • Tighter momentum/trend gates (higher ADX; RSI windows).
    • Chop filter via 15m BB-width% minimum.
    • 5m triggers stricter: pullback+reclaim MUST also have MACD cross (or do 40-bar breakout).
    • Volume confirmation increased to 1.2× SMA.
    • Fee/scalp guard via min ATR% of price.
    • Cooldown increased; hard daily trade cap; fewer TPs (2 instead of 3).
    """

    # ---------------- Parameters ----------------
    def __init__(self):
        super().__init__()

        # Momentum
        self.rsi_period = 14
        # tighter, bias towards mid-trend entries
        self.rsi_lb_long = 42
        self.rsi_ub_short = 58
        self.rsi_overbought = 74
        self.rsi_oversold = 26

        # MAs
        self.fast_ema = 9
        self.slow_ema = 21
        self.setup_sma = 50
        self.trend_ema = 200

        # Bollinger
        self.bb_period = 20
        self.bb_std = 2

        # Trend strength
        self.adx_period = 14
        self.adx_threshold = 16  # was 12

        # Risk / Reward
        self.atr_period = 14
        self.risk_per_trade = 0.012  # was 0.015 — optional: slightly smaller risk as trades are fewer
        self.atr_mult_stop = 1.2
        self.atr_mult_t1 = 1.2
        self.atr_mult_t2 = 2.5
        # removed t3 in execution (kept attr for hyperparams compat)
        self.atr_mult_t3 = 4.0

        # Strict SL bounds
        self.max_stop_atr_mult = 1.3   # max initial stop width (in ATRs)
        self.max_stop_pct = 0.006      # 0.6% of price
        self.min_stop_pct = 0.0015     # 0.15% of price

        # Volume
        self.volume_period = 20
        self.volume_mult = 1.2  # was 1.0

        # Chop filter (15m)
        self.min_bb_width_pct_15m = 0.004  # 0.4% of price min BB width to avoid chop

        # Fee/scalp guard
        self.min_atr_pct = 0.003  # 0.30% of price minimum ATR to consider a trade

        # Cooldown / frequency
        self.cooldown_bars = 6  # was 1
        self._cooldown_until = -1
        self.daily_trade_cap = 8
        self._trades_today = 0
        self._current_day_key = None

        # Caches
        self._ind = {'ind5': None, 'ind15': None, 'ind1h': None, 'ind4h': None, 'ind1d': None}
        self._tf_candles = {'5m': None, '15m': None, '1h': None, '4h': None, '1D': None}
        self._last_ts = {'5m': None, '15m': None, '1h': None, '4h': None, '1D': None}

        # Tail sizes for speed
        self._tail_macd = 80
        self._tail_ma_big = max(self.trend_ema + 10, 220)
        self._tail_bb_atr = max(self.bb_period + self.atr_period + 10, 60)
        self._tail_momentum = max(self.rsi_period + 10, 40)

        # Breakout lookback (exclude current bar)
        self.breakout_lookback = 40

    # ---------------- Jesse plumbing ----------------

    def extra_candles(self):
        return [
            (self.exchange, self.symbol, '5m'),
            (self.exchange, self.symbol, '15m'),
            (self.exchange, self.symbol, '1h'),
            (self.exchange, self.symbol, '4h'),
            (self.exchange, self.symbol, '1D'),  # NEW: daily regime filter
        ]

    # ---------------- Utilities ----------------

    def _get_candles(self, tf: str):
        return self.get_candles(self.exchange, self.symbol, tf)

    @staticmethod
    def _cross_up_vals(prev_a, prev_b, last_a, last_b):
        return (
            prev_a is not None and prev_b is not None and
            last_a is not None and last_b is not None and
            prev_a <= prev_b and last_a > last_b
        )

    @staticmethod
    def _cross_down_vals(prev_a, prev_b, last_a, last_b):
        return (
            prev_a is not None and prev_b is not None and
            last_a is not None and last_b is not None and
            prev_a >= prev_b and last_a < last_b
        )

    @staticmethod
    def _tail(candles, n):
        if candles is None:
            return None
        return candles[-n:] if len(candles) > n else candles

    def _ema(self, candles, period, sequential=False, source_type='close'):
        return ta.ema(candles, period=period, source_type=source_type, sequential=sequential)

    def _sma(self, candles, period, source_type='close', sequential=False):
        return ta.sma(candles, period=period, source_type=source_type, sequential=sequential)

    def _rsi(self, candles, period, sequential=False):
        return ta.rsi(candles, period=period, sequential=sequential)

    def _macd_last(self, candles):
        if candles is None or len(candles) < 35:  # MACD warmup
            return None, None, None
        macd, sig, hist = ta.macd(candles, 12, 26, 9, sequential=False)
        return macd, sig, hist

    def _macd_prev_last(self, candles):
        if candles is None or len(candles) < 36:
            return None, None, None, None, None, None
        last_macd, last_sig, last_hist = self._macd_last(candles)
        prev_macd, prev_sig, prev_hist = self._macd_last(candles[:-1])
        return prev_macd, prev_sig, prev_hist, last_macd, last_sig, last_hist

    def _bb(self, candles, sequential=False):
        return ta.bollinger_bands(
            candles, period=self.bb_period, devup=self.bb_std, devdn=self.bb_std,
            source_type='close', sequential=sequential
        )

    def _atr(self, candles, sequential=False):
        return ta.atr(candles, period=self.atr_period, sequential=sequential)

    def _adx(self, candles, sequential=False):
        return ta.adx(candles, period=self.adx_period, sequential=sequential)

    # ---------------- Indicator builders (per TF, only when new bar) ----------------

    def _update_tf_cache(self, tf: str):
        c = self._get_candles(tf)
        if c is None or len(c) == 0:
            return False
        ts = c[-1, 0]
        changed = (self._last_ts[tf] != ts)
        if changed:
            self._tf_candles[tf] = c
            self._last_ts[tf] = ts
        return changed

    def _build_5m(self):
        c = self._tf_candles['5m']
        if c is None:
            return

        c_ma = self._tail(c, self._tail_ma_big)
        c_bb_atr = self._tail(c, self._tail_bb_atr)
        c_macd = self._tail(c, self._tail_macd)

        ema9 = self._ema(c_ma, self.fast_ema, sequential=False)
        ema21 = self._ema(c_ma, self.slow_ema, sequential=False)

        bb_u, bb_m, bb_l = self._bb(c_bb_atr, sequential=False)
        atr = self._atr(c_bb_atr, sequential=False)

        pm, ps, _, lm, ls, _ = self._macd_prev_last(c_macd)

        vols = c[:, 5]
        vol_sma = float(np.mean(vols[-self.volume_period:])) if len(vols) >= self.volume_period else float(np.mean(vols))

        o, h, l, cl, v = c[-1, 1], c[-1, 2], c[-1, 3], c[-1, 4], c[-1, 5]

        self._ind['ind5'] = dict(
            ema9=ema9, ema21=ema21,
            bb_u=bb_u, bb_m=bb_m, bb_l=bb_l,
            atr=atr,
            macd_prev=pm, macd_sig_prev=ps,
            macd=lm, macd_signal=ls,
            vol=v, vol_sma=vol_sma,
            o=o, h=h, l=l, c=cl
        )

    def _build_15m(self):
        c = self._tf_candles['15m']
        if c is None:
            return

        c_ma = self._tail(c, self._tail_ma_big)
        c_mom = self._tail(c, max(self._tail_momentum, self._tail_macd))

        ema9 = self._ema(c_ma, self.fast_ema, sequential=False)
        ema21 = self._ema(c_ma, self.slow_ema, sequential=False)
        sma50 = self._sma(c_ma, self.setup_sma, sequential=False)
        ema200 = self._ema(c_ma, self.trend_ema, sequential=False)

        rsi = self._rsi(c_mom, self.rsi_period, sequential=False)
        macd, macd_sig, macd_hist = self._macd_last(c_mom)
        adx = self._adx(c_mom, sequential=False)

        o, h, l, cl, v = c[-1, 1], c[-1, 2], c[-1, 3], c[-1, 4], c[-1, 5]

        # Chop filter metric: BB width % of price on 15m
        bb_u15, bb_m15, bb_l15 = self._bb(self._tail(c, self._tail_bb_atr), sequential=False)
        bb_width_pct = (bb_u15 - bb_l15) / cl if (bb_u15 is not None and bb_l15 is not None and cl > 0) else None

        self._ind['ind15'] = dict(
            ema9=ema9, ema21=ema21, sma50=sma50, ema200=ema200,
            rsi=rsi, macd=macd, macd_signal=macd_sig, macd_hist=macd_hist,
            adx=adx, bb_width_pct=bb_width_pct,
            o=o, h=h, l=l, c=cl, vol=v
        )

    def _build_1h(self):
        c = self._tf_candles['1h']
        if c is None:
            return
        c_ma = self._tail(c, self._tail_ma_big)
        ema200 = self._ema(c_ma, self.trend_ema, sequential=False)
        sma50 = self._sma(c_ma, self.setup_sma, sequential=False)
        adx = self._adx(self._tail(c, self._tail_momentum), sequential=False)
        cl = c[-1, 4]
        self._ind['ind1h'] = dict(ema200=ema200, sma50=sma50, adx=adx, c=cl)

    def _build_4h(self):
        c = self._tf_candles['4h']
        if c is None:
            return
        c_ma = self._tail(c, self._tail_ma_big)
        ema200 = self._ema(c_ma, self.trend_ema, sequential=False)
        sma50 = self._sma(c_ma, self.setup_sma, sequential=False)
        adx = self._adx(self._tail(c, self._tail_momentum), sequential=False)
        cl = c[-1, 4]
        self._ind['ind4h'] = dict(ema200=ema200, sma50=sma50, adx=adx, c=cl)

    def _build_1d(self):
        c = self._tf_candles['1D']
        if c is None:
            return
        c_ma = self._tail(c, self._tail_ma_big)
        ema200 = self._ema(c_ma, self.trend_ema, sequential=False)
        sma50 = self._sma(c_ma, self.setup_sma, sequential=False)
        adx = self._adx(self._tail(c, self._tail_momentum), sequential=False)
        cl = c[-1, 4]
        self._ind['ind1d'] = dict(ema200=ema200, sma50=sma50, adx=adx, c=cl)

    def _recompute_changed_timeframes(self):
        changed_5 = self._update_tf_cache('5m')
        changed_15 = self._update_tf_cache('15m')
        changed_1h = self._update_tf_cache('1h')
        changed_4h = self._update_tf_cache('4h')
        changed_1d = self._update_tf_cache('1D')
        if changed_5:
            self._build_5m()
        if changed_15:
            self._build_15m()
        if changed_1h:
            self._build_1h()
        if changed_4h:
            self._build_4h()
        if changed_1d:
            self._build_1d()

    # ---------------- Jesse hooks ----------------

    def before(self):
        self._recompute_changed_timeframes()
        self._manage_trailing()
        self._handle_daily_cap_reset()

    def after(self):
        pass

    # ---------------- Frequency helpers ----------------

    def _handle_daily_cap_reset(self):
        # derive day key from latest 5m candle timestamp
        c5 = self._tf_candles['5m']
        if c5 is None or len(c5) == 0:
            return
        # Use UTC day from candles
        last_ts_ms = c5[-1, 0]
        # day key as integer days since epoch
        day_key = int(last_ts_ms // (24 * 60 * 60 * 1000))
        if day_key != self._current_day_key:
            self._current_day_key = day_key
            self._trades_today = 0

    # ---------------- Regime / Setup ----------------

    def _regime_ok_long(self):
        ind1d, ind1h, ind4h, ind15 = self._ind.get('ind1d'), self._ind.get('ind1h'), self._ind.get('ind4h'), self._ind.get('ind15')

        # Daily must be bullish to allow longs
        cond_1d = ind1d and (ind1d['c'] > ind1d['ema200'] and ind1d['sma50'] > ind1d['ema200'])
        if not cond_1d:
            return False

        cond_1h = ind1h and (ind1h['c'] > ind1h['ema200'] and ind1h['sma50'] > ind1h['ema200'])
        cond_4h = ind4h and (ind4h['c'] > ind4h['ema200'] and ind4h['sma50'] > ind4h['ema200'])
        regime = bool(cond_1h or cond_4h)

        # 15m fallback only if both HTFs are missing, not conflicting
        if not regime and ind15 and (ind1h is None and ind4h is None):
            regime = (ind15['c'] > ind15['ema200']) or (ind15['c'] > ind15['sma50'])

        # ADX from any HTF or 15m
        adx_ok = (
            (ind1h and ind1h['adx'] is not None and ind1h['adx'] >= self.adx_threshold) or
            (ind4h and ind4h['adx'] is not None and ind4h['adx'] >= self.adx_threshold) or
            (ind15 and ind15['adx'] is not None and ind15['adx'] >= self.adx_threshold)
        )
        return bool(regime and adx_ok)

    def _regime_ok_short(self):
        ind1d, ind1h, ind4h, ind15 = self._ind.get('ind1d'), self._ind.get('ind1h'), self._ind.get('ind4h'), self._ind.get('ind15')

        # Daily must be bearish to allow shorts
        cond_1d = ind1d and (ind1d['c'] < ind1d['ema200'] and ind1d['sma50'] < ind1d['ema200'])
        if not cond_1d:
            return False

        cond_1h = ind1h and (ind1h['c'] < ind1h['ema200'] and ind1h['sma50'] < ind1h['ema200'])
        cond_4h = ind4h and (ind4h['c'] < ind4h['ema200'] and ind4h['sma50'] < ind4h['ema200'])
        regime = bool(cond_1h or cond_4h)

        if not regime and ind15 and (ind1h is None and ind4h is None):
            regime = (ind15['c'] < ind15['ema200']) or (ind15['c'] < ind15['sma50'])

        adx_ok = (
            (ind1h and ind1h['adx'] is not None and ind1h['adx'] >= self.adx_threshold) or
            (ind4h and ind4h['adx'] is not None and ind4h['adx'] >= self.adx_threshold) or
            (ind15 and ind15['adx'] is not None and ind15['adx'] >= self.adx_threshold)
        )
        return bool(regime and adx_ok)

    def _setup_ok_long(self):
        ind15 = self._ind['ind15']
        if not ind15:
            return False

        # Chop filter
        if ind15['bb_width_pct'] is None or ind15['bb_width_pct'] < self.min_bb_width_pct_15m:
            return False

        trend_ok = (ind15['ema9'] > ind15['ema21']) and ((ind15['c'] > ind15['ema200']) or (ind15['c'] > ind15['sma50']))
        rsi_ok = (ind15['rsi'] is not None) and (self.rsi_lb_long <= ind15['rsi'] < self.rsi_overbought)
        momentum_ok = (
            (ind15['macd'] is not None and ind15['macd_signal'] is not None and ind15['macd'] >= ind15['macd_signal'])
            and (ind15['adx'] is not None and ind15['adx'] >= (self.adx_threshold - 0))  # no "looser" shortcut
        )
        return trend_ok and rsi_ok and momentum_ok

    def _setup_ok_short(self):
        ind15 = self._ind['ind15']
        if not ind15:
            return False

        # Chop filter
        if ind15['bb_width_pct'] is None or ind15['bb_width_pct'] < self.min_bb_width_pct_15m:
            return False

        trend_ok = (ind15['ema9'] < ind15['ema21']) and ((ind15['c'] < ind15['ema200']) or (ind15['c'] < ind15['sma50']))
        rsi_ok = (ind15['rsi'] is not None) and (self.rsi_oversold < ind15['rsi'] <= self.rsi_ub_short)
        momentum_ok = (
            (ind15['macd'] is not None and ind15['macd_signal'] is not None and ind15['macd'] <= ind15['macd_signal'])
            and (ind15['adx'] is not None and ind15['adx'] >= (self.adx_threshold - 0))
        )
        return trend_ok and rsi_ok and momentum_ok

    # ---------------- Stop & Fee Guards ----------------

    def _atr_fee_guard(self):
        """Reject trades when ATR is too small relative to price -> fees dominate."""
        ind5 = self._ind['ind5']
        if not ind5 or ind5['atr'] is None or ind5['c'] is None or ind5['c'] <= 0:
            return False
        return (ind5['atr'] / ind5['c']) >= self.min_atr_pct

    def _stop_guard_long(self):
        ind5 = self._ind['ind5']
        if not ind5:
            return False
        entry = ind5['c']
        atr = ind5['atr']
        if entry is None or atr is None:
            return False

        stop = max(entry - (atr * self.atr_mult_stop), ind5['bb_l'])
        dist = entry - stop
        max_by_atr = atr * self.max_stop_atr_mult
        max_by_pct = entry * self.max_stop_pct
        min_by_pct = entry * self.min_stop_pct
        return (dist <= min(max_by_atr, max_by_pct)) and (dist >= min_by_pct)

    def _stop_guard_short(self):
        ind5 = self._ind['ind5']
        if not ind5:
            return False
        entry = ind5['c']
        atr = ind5['atr']
        if entry is None or atr is None:
            return False

        stop = min(entry + (atr * self.atr_mult_stop), ind5['bb_u'])
        dist = stop - entry
        max_by_atr = atr * self.max_stop_atr_mult
        max_by_pct = entry * self.max_stop_pct
        min_by_pct = entry * self.min_stop_pct
        return (dist <= min(max_by_atr, max_by_pct)) and (dist >= min_by_pct)

    # ---------------- Triggers (5m) ----------------

    def _trigger_ok_long(self):
        ind5 = self._ind['ind5']
        if not ind5:
            return False

        atr = ind5['atr']
        if atr is None:
            return False
        tol = atr * 0.20  # slightly tighter tolerance

        # Pullback + reclaim MUST also have MACD cross up
        pullback = (
            (ind5['l'] <= ind5['ema9'] + tol) or
            (ind5['l'] <= ind5['ema21'] + tol) or
            (ind5['l'] <= ind5['bb_m'] + tol)
        )
        reclaim = ind5['c'] > max(ind5['ema21'], ind5['bb_m'])

        macd_cross_up = self._cross_up_vals(
            ind5['macd_prev'], ind5['macd_sig_prev'], ind5['macd'], ind5['macd_signal']
        )

        c5 = self._tf_candles['5m']
        if c5 is not None and len(c5) >= (self.breakout_lookback + 1):
            prev_highs = c5[-(self.breakout_lookback + 1):-1, 2]
            hh = float(np.max(prev_highs))
            breakout_long = (ind5['c'] >= hh) and (ind5['c'] > ind5['ema21'])
        else:
            breakout_long = False

        vol_ok = ind5['vol'] >= (ind5['vol_sma'] * self.volume_mult)
        is_green = ind5['c'] > ind5['o']

        confirm_trend = pullback and reclaim and macd_cross_up and (vol_ok or ind5['c'] > ind5['ema9']) and is_green
        confirm_breakout = breakout_long and (vol_ok and macd_cross_up)

        return (confirm_trend or confirm_breakout) and self._stop_guard_long() and self._atr_fee_guard()

    def _trigger_ok_short(self):
        ind5 = self._ind['ind5']
        if not ind5:
            return False

        atr = ind5['atr']
        if atr is None:
            return False
        tol = atr * 0.20

        pullback = (
            (ind5['h'] >= ind5['ema9'] - tol) or
            (ind5['h'] >= ind5['ema21'] - tol) or
            (ind5['h'] >= ind5['bb_m'] - tol)
        )
        reject = ind5['c'] < min(ind5['ema21'], ind5['bb_m'])

        macd_cross_down = self._cross_down_vals(
            ind5['macd_prev'], ind5['macd_sig_prev'], ind5['macd'], ind5['macd_signal']
        )

        c5 = self._tf_candles['5m']
        if c5 is not None and len(c5) >= (self.breakout_lookback + 1):
            prev_lows = c5[-(self.breakout_lookback + 1):-1, 3]
            ll = float(np.min(prev_lows))
            breakout_short = (ind5['c'] <= ll) and (ind5['c'] < ind5['ema21'])
        else:
            breakout_short = False

        vol_ok = ind5['vol'] >= (ind5['vol_sma'] * self.volume_mult)
        is_red = ind5['c'] < ind5['o']

        confirm_trend = pullback and reject and macd_cross_down and (vol_ok or ind5['c'] < ind5['ema9']) and is_red
        confirm_breakout = breakout_short and (vol_ok and macd_cross_down)

        return (confirm_trend or confirm_breakout) and self._stop_guard_short() and self._atr_fee_guard()

    def _under_daily_cap(self):
        return self._trades_today < self.daily_trade_cap

    def should_long(self) -> bool:
        if self._ind['ind5'] is None or self._ind['ind15'] is None:
            return False
        if self.index < self._cooldown_until:
            return False
        if not self._under_daily_cap():
            return False
        return self._regime_ok_long() and self._setup_ok_long() and self._trigger_ok_long()

    def should_short(self) -> bool:
        if self._ind['ind5'] is None or self._ind['ind15'] is None:
            return False
        if self.index < self._cooldown_until:
            return False
        if not self._under_daily_cap():
            return False
        return self._regime_ok_short() and self._setup_ok_short() and self._trigger_ok_short()

    # ---------------- Orders / Risk ----------------

    def _risk_qty_long(self, entry, stop):
        risk_amount = self.balance * self.risk_per_trade
        risk_per_unit = max(entry - stop, entry * 0.001)
        return max(risk_amount / risk_per_unit, 0)

    def _risk_qty_short(self, entry, stop):
        risk_amount = self.balance * self.risk_per_trade
        risk_per_unit = max(stop - entry, entry * 0.001)
        return max(risk_amount / risk_per_unit, 0)

    def go_long(self):
        ind5 = self._ind['ind5']
        entry = ind5['c']
        atr = ind5['atr']

        # match stop guard
        stop = max(entry - (atr * self.atr_mult_stop), ind5['bb_l'])
        min_dist = entry * self.min_stop_pct
        max_dist = min(atr * self.max_stop_atr_mult, entry * self.max_stop_pct)
        dist = entry - stop
        if dist < min_dist:
            stop = entry - min_dist
        elif dist > max_dist:
            stop = entry - max_dist

        qty = self._risk_qty_long(entry, stop)

        t1 = entry + atr * self.atr_mult_t1
        t2 = entry + atr * self.atr_mult_t2

        self.buy = qty, entry
        self.stop_loss = qty, stop
        # fewer exits: 2 TPs to cut fees
        self.take_profit = [
            (qty * 0.50, t1),
            (qty * 0.50, t2),
        ]

        self._trades_today += 1  # count entry as one trade decision

    def go_short(self):
        ind5 = self._ind['ind5']
        entry = ind5['c']
        atr = ind5['atr']

        # match stop guard
        stop = min(entry + (atr * self.atr_mult_stop), ind5['bb_u'])
        min_dist = entry * self.min_stop_pct
        max_dist = min(atr * self.max_stop_atr_mult, entry * self.max_stop_pct)
        dist = stop - entry
        if dist < min_dist:
            stop = entry + min_dist
        elif dist > max_dist:
            stop = entry + max_dist

        qty = self._risk_qty_short(entry, stop)

        t1 = entry - atr * self.atr_mult_t1
        t2 = entry - atr * self.atr_mult_t2

        self.sell = qty, entry
        self.stop_loss = qty, stop
        self.take_profit = [
            (qty * 0.50, t1),
            (qty * 0.50, t2),
        ]

        self._trades_today += 1

    # ---------------- Management ----------------

    def _manage_trailing(self):
        if not self.position or not self.position.is_open or self._ind['ind5'] is None:
            return

        ind5 = self._ind['ind5']
        atr = ind5['atr']
        if atr is None:
            return

        if self.position.is_long:
            if self.close - self.position.entry_price >= 0.8 * atr:
                trail_candidates = [
                    ind5['ema21'],
                    ind5['bb_m'] - 0.25 * atr,
                    self.close - 1.2 * atr,
                    self.position.entry_price,
                ]
                new_stop = max([x for x in trail_candidates if x is not None])
                self.stop_loss = self.position.qty, new_stop

        elif self.position.is_short:
            if self.position.entry_price - self.close >= 0.8 * atr:
                trail_candidates = [
                    ind5['ema21'],
                    ind5['bb_m'] + 0.25 * atr,
                    self.close + 1.2 * atr,
                    self.position.entry_price,
                ]
                new_stop = min([x for x in trail_candidates if x is not None])
                self.stop_loss = self.position.qty, new_stop

    def on_reduced_position(self, order):
        if self.position and self.position.is_open:
            self.stop_loss = self.position.qty, self.position.entry_price

    def on_closed_position(self, order):
        self._cooldown_until = self.index + self.cooldown_bars

    def should_cancel(self) -> bool:
        return False

    # ---------------- Hyperparameters ----------------

    def hyperparameters(self):
        return [
            {'name': 'adx_threshold', 'type': int, 'min': 8, 'max': 24, 'default': self.adx_threshold},
            {'name': 'volume_mult', 'type': float, 'min': 1.0, 'max': 1.6, 'default': self.volume_mult},
            {'name': 'atr_mult_stop', 'type': float, 'min': 1.0, 'max': 1.6, 'default': self.atr_mult_stop},
            {'name': 'atr_mult_t1', 'type': float, 'min': 1.0, 'max': 1.6, 'default': self.atr_mult_t1},
            {'name': 'atr_mult_t2', 'type': float, 'min': 2.0, 'max': 3.5, 'default': self.atr_mult_t2},
            {'name': 'atr_mult_t3', 'type': float, 'min': 3.0, 'max': 5.0, 'default': self.atr_mult_t3},
            {'name': 'max_stop_atr_mult', 'type': float, 'min': 1.1, 'max': 1.6, 'default': self.max_stop_atr_mult},
            {'name': 'max_stop_pct', 'type': float, 'min': 0.004, 'max': 0.008, 'default': self.max_stop_pct},
            {'name': 'min_stop_pct', 'type': float, 'min': 0.001, 'max': 0.002, 'default': self.min_stop_pct},
            {'name': 'min_bb_width_pct_15m', 'type': float, 'min': 0.002, 'max': 0.008, 'default': self.min_bb_width_pct_15m},
            {'name': 'min_atr_pct', 'type': float, 'min': 0.002, 'max': 0.006, 'default': self.min_atr_pct},
            {'name': 'daily_trade_cap', 'type': int, 'min': 3, 'max': 20, 'default': self.daily_trade_cap},
            {'name': 'cooldown_bars', 'type': int, 'min': 2, 'max': 18, 'default': self.cooldown_bars},
            {'name': 'breakout_lookback', 'type': int, 'min': 20, 'max': 60, 'default': self.breakout_lookback},
        ]



"""


Why this should cut trades & fees

The 1D regime filter alone removes most counter-trend shorts during broader uptrends (and vice-versa), which is where your short bias likely came from.

Chop filter + higher ADX kills low-quality churn that racks up fees.

Stricter trigger logic (MACD cross required) and longer lookback breakouts further reduce over-trading.

Cooldown + daily cap give you hard ceilings on trade count even in hyperactive markets.

Two TPs instead of three means fewer exit orders per position.


If you want, I can also add:

A session/clock filter (skip historically illiquid hours for your exchange/pairs).

A max concurrent positions limiter across symbols.

A fee model param (maker/taker) to auto-adjust min_atr_pct.

"""