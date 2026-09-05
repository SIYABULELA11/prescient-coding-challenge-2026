"""
Prescient Coding Challenge 2026 -- your submission.

THIS IS THE ONLY FILE YOU MAY CHANGE.

You implement one function. The harness calls it once per trading day and hands
you a `hist` object holding every observation STRICTLY BEFORE that day. You
return the weights you want to hold for that day.

    generate_weights(hist, prev_weights, params) -> weights

What you get
------------
hist.date                 the day you are allocating for (no data for it yet)
hist.returns              DataFrame [date x asset] of daily returns, decimals
hist.prices               DataFrame [date x asset] of total-return index levels
hist.macro                DataFrame [date x macro feature]
hist.assets               list of the six asset codes, in order
hist.benchmark            Series of benchmark weights
hist.active_weight(w)     total active weight of w -- the number rule 3 tests

prev_weights              what you held yesterday. Trading away from it costs
                          money, so look at it.
params                    the PARAMS dict below, passed straight through

Optional extras, in case you want them: hist.cov() gives an EWMA covariance
matrix and hist.te(w) an ex-ante tracking error. No rule depends on either.

What you must return
--------------------
Six weights (dict, Series or array in hist.assets order) that sum to 1, are all
non-negative, sit within 10% of their benchmark weight, have a total active
weight of no more than 40%, keep total equity at or below 75% and gold at or
below 10%. `make_legal()` below
already does all of that -- you can leave it alone.

Declare every tuneable number in PARAMS. Parameter count is part of the score.

Run `python harness.py` to test on the practice window (calendar 2025), then
`python validate.py` before you submit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesRegressor
except ImportError:  # Safe fallback if the grader omits the optional package.
    ExtraTreesRegressor = None

# --------------------------------------------------------------------------- #
# Every tuneable number lives here. Fewer is better.
# --------------------------------------------------------------------------- #

PARAMS = {
    "target_days": 63,       # predict a slower, cost-compatible relative return
    "train_days":  2016,     # at most eight years in each rolling fit
    "ridge_alpha": 250.0,    # shrink noisy/correlated feature coefficients
    "extra_weight": 0.15,    # nonlinear challenger share after rank calibration
    "extra_trees": 128,      # enough averaging for stable deterministic ranks
    "extra_depth": 4,        # deliberately shallow to resist regime overfit
    "extra_leaf": 63,        # require about one target horizon in every leaf
    "extra_features": 0.50,  # decorrelate trees without hiding most predictors
    "active_weight": 0.18,   # moderate conviction; still well below the 40% cap
    "trade_speed": 0.06,     # slow daily adjustment protects returns after costs
}

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10


# --------------------------------------------------------------------------- #
# <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>
#
# This is your playground. Delete or rewrite anything here. What follows is a
# deliberately naive starting point so you can see the shape of a working
# answer. It is NOT a good answer -- on the practice window it loses to the
# benchmark. Your job is to do better.
#
# Three steps:
#   1. build a signal (here: a plain inverse-volatility tilt, which knows
#      nothing at all about expected return),
#   2. make the weights legal,
#   3. move only part of the way from yesterday, so you do not pay the full
#      trading cost every day.
#
# Steps 2 and 3 are plumbing. Keep them. Step 1 is the actual question, and
# inverse volatility is a poor answer to it: it will always prefer cash and
# bonds, whatever is happening in the world.
#
# Things worth thinking about. Which of these six assets actually diversifies
# the other five? Gold and global equity are both priced in rands -- what does
# that mean when the currency moves? The macro file has a term spread and a
# policy rate in it; what should a steepening curve do to your bond weight? And
# look at the cost table in the README before you trade property daily.
# --------------------------------------------------------------------------- #


_MODEL_CACHE = None
_LAST_DATE = None


def _rolling_zscore(series: pd.Series, days: int = 252) -> pd.Series:
    """Past-only rolling z-score with a stable zero-variance fallback."""
    mean = series.rolling(days, min_periods=days).mean()
    scale = series.rolling(days, min_periods=days).std().replace(0.0, np.nan)
    return (series - mean) / scale


def _feature_frame(returns, prices, macro, benchmark) -> pd.DataFrame:
    """Create a compact feature set with a separate economic role per asset.

    Every row uses information available on or before its own index date. The
    harness passes a frame ending yesterday, so the final row is safe for the
    next decision. Macro observations are aligned as-of from the past; this
    also handles the single absent macro row without backward filling.
    """
    assets = list(returns.columns)
    log_r = np.log1p(returns.clip(lower=-0.999999))
    bm_daily = returns.mul(benchmark.reindex(assets), axis=1).sum(axis=1)
    bm_log = np.log1p(bm_daily.clip(lower=-0.999999))

    mom21 = log_r.rolling(21, min_periods=21).sum()
    mom63 = log_r.rolling(63, min_periods=63).sum()
    mom126 = log_r.rolling(126, min_periods=126).sum()
    mom252 = log_r.rolling(252, min_periods=252).sum()
    mom5 = log_r.rolling(5, min_periods=5).sum()
    bm_mom63 = bm_log.rolling(63, min_periods=63).sum()

    vol21 = returns.rolling(21, min_periods=21).std()
    vol126 = returns.rolling(126, min_periods=126).std()
    vol_ratio = vol21.div(vol126.replace(0.0, np.nan)) - 1.0
    reversal5 = -mom5.div((vol21 * np.sqrt(5)).replace(0.0, np.nan))
    drawdown126 = prices.div(prices.rolling(126, min_periods=126).max()) - 1.0
    trend_quality = mom63.div(
        log_r.abs().rolling(63, min_periods=63).sum().replace(0.0, np.nan)
    )
    rel_mom63 = mom63.sub(bm_mom63, axis=0)

    feature = {}
    for asset in assets:
        # The outputs are asset-specific, but all six market histories are
        # visible to every output so cross-market leadership can be learned.
        feature[f"{asset}_mom21"] = mom21[asset]
        feature[f"{asset}_mom63"] = mom63[asset]
        feature[f"{asset}_mom126"] = mom126[asset]
        feature[f"{asset}_mom252_skip5"] = mom252[asset] - mom5[asset]
        feature[f"{asset}_relmom63"] = rel_mom63[asset]
        feature[f"{asset}_reversal5"] = reversal5[asset]
        feature[f"{asset}_vol21"] = vol21[asset] * np.sqrt(252)
        feature[f"{asset}_vol_ratio"] = vol_ratio[asset]
        feature[f"{asset}_drawdown126"] = drawdown126[asset]
        feature[f"{asset}_trend_quality"] = trend_quality[asset]

    aligned_macro = macro.sort_index().reindex(returns.index, method="ffill")

    def log_change(name: str) -> pd.Series:
        values = aligned_macro[name].clip(lower=1e-12)
        return np.log(values).diff()

    fx = log_change("usdzar")
    dxy = log_change("dxy")
    vix = log_change("vix")
    brent = log_change("brent")
    em = log_change("em_equity")

    fx21 = fx.rolling(21, min_periods=21).sum()
    fx63 = fx.rolling(63, min_periods=63).sum()
    vix_z = _rolling_zscore(np.log(aligned_macro["vix"].clip(lower=1e-12)))
    em63 = em.rolling(63, min_periods=63).sum()
    brent63 = brent.rolling(63, min_periods=63).sum()
    sa10_change21 = aligned_macro["sa_10y"].diff(21)

    feature.update({
        "macro_usdzar_mom21": fx21,
        "macro_usdzar_mom63": fx63,
        "macro_usdzar_vol63": fx.rolling(63, min_periods=63).std() * np.sqrt(252),
        "macro_dxy_mom21": dxy.rolling(21, min_periods=21).sum(),
        "macro_dxy_mom63": dxy.rolling(63, min_periods=63).sum(),
        "macro_vix_z252": vix_z,
        "macro_vix_change21": vix.rolling(21, min_periods=21).sum(),
        "macro_vix_vol63": vix.rolling(63, min_periods=63).std() * np.sqrt(252),
        "macro_brent_mom63": brent63,
        "macro_em_mom21": em.rolling(21, min_periods=21).sum(),
        "macro_em_mom63": em63,
        "macro_us2_change21": aligned_macro["us_2y"].diff(21),
        "macro_us10_change21": aligned_macro["us_10y"].diff(21),
        "macro_sa10_change21": sa10_change21,
        "macro_jibar_change21": aligned_macro["jibar_3m"].diff(21),
        "macro_repo_change21": aligned_macro["sa_repo"].diff(21),
        "macro_us_curve": aligned_macro["us_10y"] - aligned_macro["us_2y"],
        "macro_sa_curve": aligned_macro["sa_10y"] - aligned_macro["jibar_3m"],
        "macro_jibar_repo": aligned_macro["jibar_3m"] - aligned_macro["sa_repo"],
        "risk_breadth63": (mom63[["SA_EQUITY", "GLOBAL_EQUITY", "SA_PROPERTY"]] > 0).mean(axis=1),
        "cross_asset_dispersion63": mom63.std(axis=1),
        "equity_bond_corr63": returns["SA_EQUITY"].rolling(63, min_periods=63).corr(returns["SA_BONDS"]),
    })

    # Decompose ZAR global equity and gold into underlying-asset and currency
    # components using only the supplied USD/ZAR series.
    global_underlying = log_r["GLOBAL_EQUITY"] - fx
    gold_underlying = log_r["GOLD"] - fx
    global_under63 = global_underlying.rolling(63, min_periods=63).sum()
    gold_under63 = gold_underlying.rolling(63, min_periods=63).sum()
    feature["global_underlying_mom63"] = global_under63
    feature["gold_underlying_mom63"] = gold_under63

    # Economically selected nonlinear interactions. We intentionally do not
    # generate every pairwise product: the small effective sample does not
    # support an unrestricted interaction search.
    for asset in assets:
        feature[f"{asset}_momentum_x_stress"] = rel_mom63[asset] * vix_z
        feature[f"{asset}_reversal_x_volchange"] = reversal5[asset] * vol_ratio[asset]

    feature.update({
        "sa_equity_momentum_x_em": rel_mom63["SA_EQUITY"] * em63,
        "sa_equity_momentum_x_brent": rel_mom63["SA_EQUITY"] * brent63,
        "global_momentum_x_fx": rel_mom63["GLOBAL_EQUITY"] * fx63,
        "global_underlying_x_fx": global_under63 * fx63,
        "bond_momentum_x_sa_yield": rel_mom63["SA_BONDS"] * sa10_change21,
        "property_momentum_x_sa_yield": rel_mom63["SA_PROPERTY"] * sa10_change21,
        "gold_underlying_x_fx": gold_under63 * fx63,
        "gold_reversal_x_stress": reversal5["GOLD"] * vix_z,
    })
    return pd.DataFrame(feature, index=returns.index).replace([np.inf, -np.inf], np.nan)


def _future_relative_target(returns, benchmark, days: int) -> pd.DataFrame:
    """Future asset log return minus future benchmark log return."""
    assets = list(returns.columns)
    log_r = np.log1p(returns.clip(lower=-0.999999))
    bm_daily = returns.mul(benchmark.reindex(assets), axis=1).sum(axis=1)
    bm_log = np.log1p(bm_daily.clip(lower=-0.999999))
    future_asset = log_r.rolling(days, min_periods=days).sum().shift(-days)
    future_bm = bm_log.rolling(days, min_periods=days).sum().shift(-days)
    target = future_asset.sub(future_bm, axis=0)
    # Active weights must net to zero, so learn cross-asset differences rather
    # than the common level of expected market return.
    return target.sub(target.mean(axis=1), axis=0)


def _fit_models(hist, params):
    """Fit monthly Ridge and shallow Extra Trees on past-complete labels."""
    target_days = int(params["target_days"])
    train_days = int(params["train_days"])
    warmup = 280
    sample = hist.returns.tail(train_days + warmup + target_days)
    prices = hist.prices.reindex(sample.index)
    x = _feature_frame(sample, prices, hist.macro, hist.benchmark)
    y = _future_relative_target(sample, hist.benchmark, target_days)
    y = y.rank(axis=1, method="average", pct=True)
    y = y.sub(y.mean(axis=1), axis=0)

    valid = x.notna().all(axis=1) & y.notna().all(axis=1)
    x = x.loc[valid].tail(train_days)
    y = y.loc[valid].tail(train_days)
    if len(x) < 500:
        return None

    # Winsorize only with quantiles estimated inside the current training set.
    lower = y.quantile(0.01)
    upper = y.quantile(0.99)
    y_robust = y.clip(lower=lower, upper=upper, axis=1)
    y_centered = y_robust - y_robust.mean()

    x_mean = x.mean()
    x_scale = x.std().replace(0.0, 1.0)
    xv = x.sub(x_mean).div(x_scale).clip(-5.0, 5.0).to_numpy(dtype=float)
    yv = y_centered.to_numpy(dtype=float)
    alpha = float(params["ridge_alpha"])
    gram = xv.T @ xv
    beta = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), xv.T @ yv)

    extra = None
    if ExtraTreesRegressor is not None and float(params["extra_weight"]) > 0.0:
        extra = ExtraTreesRegressor(
            n_estimators=int(params["extra_trees"]),
            max_depth=int(params["extra_depth"]),
            min_samples_leaf=int(params["extra_leaf"]),
            max_features=float(params["extra_features"]),
            random_state=2026,
            n_jobs=1,
        )
        extra.fit(xv, yv)
    return {
        "columns": list(x.columns),
        "mean": x_mean,
        "scale": x_scale,
        "beta": beta,
        "extra": extra,
        "month": (hist.date.year, hist.date.month),
    }


def _fallback_signal(hist) -> pd.Series:
    """Deterministic medium-term relative momentum if a fit is unavailable."""
    momentum = np.log1p(hist.returns.tail(63).clip(lower=-0.999999)).sum()
    return momentum.reindex(hist.assets).fillna(0.0) - momentum.mean()


def _rank_center(values: pd.Series) -> pd.Series:
    """Map six cross-asset observations to stable centered ranks."""
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return values.rank(method="average", pct=True) - values.rank(
        method="average", pct=True
    ).mean()


def build_signal(hist, params) -> pd.Series:
    """Blend stable Ridge ranks with a shallow nonlinear challenger."""
    global _MODEL_CACHE
    month = (hist.date.year, hist.date.month)
    if _MODEL_CACHE is None or _MODEL_CACHE.get("month") != month:
        _MODEL_CACHE = _fit_models(hist, params)
    if _MODEL_CACHE is None:
        return _fallback_signal(hist)

    tail = hist.returns.tail(300)
    current = _feature_frame(
        tail,
        hist.prices.reindex(tail.index),
        hist.macro,
        hist.benchmark,
    ).iloc[-1]
    if not np.isfinite(current.to_numpy(dtype=float)).all():
        return _fallback_signal(hist)
    x = current.reindex(_MODEL_CACHE["columns"])
    x = x.sub(_MODEL_CACHE["mean"]).div(_MODEL_CACHE["scale"]).clip(-5.0, 5.0)
    xv = x.to_numpy(dtype=float)
    ridge_prediction = xv @ _MODEL_CACHE["beta"]
    ridge_score = _rank_center(pd.Series(ridge_prediction, index=hist.assets))

    extra = _MODEL_CACHE.get("extra")
    if extra is None:
        return ridge_score
    extra_prediction = extra.predict(xv.reshape(1, -1))[0]
    extra_score = _rank_center(pd.Series(extra_prediction, index=hist.assets))
    weight = float(params["extra_weight"])
    # Do not rank again after blending. The two inputs are already calibrated
    # centered ranks; preserving their weighted distances lets the challenger
    # soften or strengthen conviction without needing to reverse a full rank.
    return (1.0 - weight) * ridge_score + weight * extra_score


def make_legal(weights: pd.Series, hist) -> pd.Series:
    """Force `weights` to satisfy every rule. You can leave this alone.

    Everything happens in active space -- how far each asset sits from its
    benchmark weight -- because that is how the rules are written.

    The loop is there because the steps interfere: forcing the active weights
    to net to zero (so the portfolio sums to 1) can push an asset back outside
    its band. A few passes settles it. The budget scaling goes last and is safe
    there: shrinking every active weight toward zero cannot breach a band, a
    cap, or non-negativity.
    """
    bm = hist.benchmark
    active = weights.reindex(hist.assets).astype(float) - bm

    for _ in range(50):
        active = active.clip(lower=-ACTIVE_BAND, upper=ACTIVE_BAND)  # rule 2
        active = active.clip(lower=-bm)                              # keeps weights >= 0
        # rule 4: total equity cap. Trim the equity block back, sharing the
        # cut over whichever equity assets still have room to come down.
        eq_excess = (bm[EQUITY] + active[EQUITY]).sum() - EQUITY_CAP
        eq_full = eq_excess > -1e-12
        if eq_excess > 0:
            floor = np.maximum(-ACTIVE_BAND, -bm[EQUITY])
            down = (active[EQUITY] - floor).clip(lower=0)
            if down.sum() > 1e-15:
                active[EQUITY] = active[EQUITY] - eq_excess * down / down.sum()

        active["GOLD"] = min(active["GOLD"], GOLD_CAP - bm["GOLD"])  # rule 5

        excess = active.sum()          # must be zero for weights to sum to 1
        if abs(excess) < 1e-12:
            break
        # give the correction to the assets that have room to absorb it
        room = (ACTIVE_BAND - active) if excess < 0 else (active + bm).clip(lower=0)
        room = room.clip(lower=0)
        if excess < 0 and eq_full:
            room[EQUITY] = 0.0   # equity is at its cap -- top up elsewhere
        if room.sum() <= 1e-15:
            break
        active = active - excess * room / room.sum()

    total = active.abs().sum()                                       # rule 3
    if total > ACTIVE_BUDGET:
        active = active * (ACTIVE_BUDGET / total)

    return bm + active


def generate_weights(hist, prev_weights, params):
    """Return the six portfolio weights to hold on hist.date."""
    global _MODEL_CACHE, _LAST_DATE
    bm = hist.benchmark

    # validate.py runs windows out of chronological order. Never allow a fit
    # cached in a later window to leak into an earlier backtest.
    if _LAST_DATE is not None and hist.date <= _LAST_DATE:
        _MODEL_CACHE = None
    _LAST_DATE = hist.date

    if len(hist.returns) < 550:
        return bm.to_dict()

    # Convert predicted relative returns into a fixed, modest amount of active
    # risk. Cross-sectional scaling prevents forecast magnitude drift from
    # changing the risk budget through time.
    signal = build_signal(hist, params)
    signal = signal.reindex(hist.assets).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal = signal - signal.mean()
    total_signal = signal.abs().sum()
    if total_signal <= 1e-15:
        signal = _fallback_signal(hist)
        total_signal = signal.abs().sum()
    active = float(params["active_weight"]) * signal / max(total_signal, 1e-15)
    target = make_legal(bm + active, hist)

    # Move gradually so short-lived changes do not become expensive trades.
    prev = prev_weights.reindex(hist.assets)
    w = prev + float(params["trade_speed"]) * (target - prev)

    return make_legal(w, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
