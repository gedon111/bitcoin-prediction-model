import requests
import pandas as pd
import numpy as np
import datetime
import webbrowser
import os
import sys
import warnings
from html import escape
import time
import json
import re
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
from bs4 import BeautifulSoup
from icalendar import Calendar as ICalendar

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def fetch_candles(interval, limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching {interval} data: {e}")
        sys.exit(1)

def compute_indicators(df):
    close = df['close']
    low = df['low']
    high = df['high']
    
    # MACD(12, 26, 9)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    MACD = ema_fast - ema_slow
    MACD_signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_hist = MACD - MACD_signal
    
    df['MACD'] = MACD
    df['MACD_signal'] = MACD_signal
    df['MACD_hist'] = MACD_hist
    
    # KDJ(9, 3, 3)
    n = 9
    low_min = low.rolling(window=n, min_periods=1).min()
    high_max = high.rolling(window=n, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    RSV = ((close - low_min) / denom * 100).fillna(50)
    alpha = 1/3
    K = RSV.ewm(alpha=alpha, adjust=False).mean()
    D = K.ewm(alpha=alpha, adjust=False).mean()
    J = 3 * K - 2 * D
    
    df['K'] = K
    df['D'] = D
    df['J'] = J
    
    # ATR(14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    TR = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = TR.ewm(alpha=1/14, adjust=False).mean()
    
    # ATR_200
    df['ATR_200'] = TR.ewm(alpha=1/200, adjust=False).mean()
    
    return df

def compute_smc(df):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_prices = df['open'].values
    volume = df['volume'].values
    atr_200 = df['ATR_200'].values
    
    high_vol = (high - low) >= 2 * atr_200
    parsed_high = np.where(high_vol, low, high)
    parsed_low = np.where(high_vol, high, low)
    
    obs = []
    
    def run_pass(size, level_tag):
        n = len(df)
        trend = 1
        sh_price = None
        sh_bar = None
        sh_crossed = True
        sl_price = None
        sl_bar = None
        sl_crossed = True
        
        for i in range(size + 1, n):
            pivot_bar = i - size
            
            # PIVOT HIGH check
            if high[pivot_bar] > np.max(high[pivot_bar+1:i+1]):
                sh_price = high[pivot_bar]
                sh_bar = pivot_bar
                sh_crossed = False
            
            # PIVOT LOW check
            if low[pivot_bar] < np.min(low[pivot_bar+1:i+1]):
                sl_price = low[pivot_bar]
                sl_bar = pivot_bar
                sl_crossed = False
                
            # BULLISH BOS/CHoCH
            if sh_price is not None and not sh_crossed and close[i-1] <= sh_price < close[i]:
                tag = 'CHoCH' if trend == -1 else 'BOS'
                sh_crossed = True
                trend = 1
                
                seg = parsed_low[sh_bar:i+1]
                ob_idx = sh_bar + np.argmin(seg)
                
                obs.append({
                    'type': 'DEMAND',
                    'top': parsed_high[ob_idx],
                    'bottom': parsed_low[ob_idx],
                    'created_at': i,
                    'ob_bar': int(ob_idx),
                    'level': level_tag,
                    'structure': tag
                })
                
            # BEARISH BOS/CHoCH
            if sl_price is not None and not sl_crossed and close[i-1] >= sl_price > close[i]:
                tag = 'CHoCH' if trend == 1 else 'BOS'
                sl_crossed = True
                trend = -1
                
                seg = parsed_high[sl_bar:i+1]
                ob_idx = sl_bar + np.argmax(seg)
                
                obs.append({
                    'type': 'SUPPLY',
                    'top': parsed_high[ob_idx],
                    'bottom': parsed_low[ob_idx],
                    'created_at': i,
                    'ob_bar': int(ob_idx),
                    'level': level_tag,
                    'structure': tag
                })

    run_pass(5, 'INTERNAL')
    run_pass(50, 'SWING')
    
    n = len(df)
    for ob in obs:
        ob_idx = ob['ob_bar']
        
        # quality_displacement
        disp = False
        for j in range(ob_idx + 1, min(ob_idx + 4, n)):
            if ob['type'] == 'DEMAND':
                if close[j] > open_prices[j] and abs(close[j] - open_prices[j]) >= 1.5 * atr_200[ob_idx]:
                    disp = True; break
            else:
                if close[j] < open_prices[j] and abs(close[j] - open_prices[j]) >= 1.5 * atr_200[ob_idx]:
                    disp = True; break
        ob['quality_displacement'] = disp
        
        # quality_large_bar
        bar_range = high[ob_idx] - low[ob_idx]
        ob['quality_large_bar'] = bar_range >= atr_200[ob_idx]
        
        # quality_fvg
        fvg = False
        for j in range(ob_idx + 1, min(ob_idx + 4, n - 1)):
            if ob['type'] == 'DEMAND':
                if low[j+1] > high[j]:
                    fvg = True; break
            else:
                if high[j+1] < low[j]:
                    fvg = True; break
        ob['quality_fvg'] = fvg
        
        # quality_liquidity_sweep
        prev_start = max(0, ob_idx - 10)
        if ob_idx > 0:
            if ob['type'] == 'DEMAND':
                ob['quality_liquidity_sweep'] = low[ob_idx] <= np.min(low[prev_start:ob_idx])
            else:
                ob['quality_liquidity_sweep'] = high[ob_idx] >= np.max(high[prev_start:ob_idx])
        else:
            ob['quality_liquidity_sweep'] = False
            
        # quality_volume_expansion
        vstart = max(0, ob_idx - 20)
        avg_vol = np.mean(volume[vstart:ob_idx]) if ob_idx > 0 else 0
        volume_good = volume[ob_idx] >= 1.25 * avg_vol
        body = abs(close[ob_idx] - open_prices[ob_idx])
        rng = high[ob_idx] - low[ob_idx]
        impulse_body = (body / rng > 0.6) if rng > 0 else False
        ob['quality_volume_expansion'] = bool(volume_good or impulse_body)
        
        ob['quality'] = sum([
            ob['quality_displacement'],
            ob['quality_large_bar'],
            ob['quality_fvg'],
            ob['quality_liquidity_sweep'],
            ob['quality_volume_expansion']
        ])
        
        mitigated_at = n
        for j in range(ob['created_at'] + 1, n):
            if ob['type'] == 'DEMAND' and close[j] < ob['bottom']:
                mitigated_at = j; break
            elif ob['type'] == 'SUPPLY' and close[j] > ob['top']:
                mitigated_at = j; break
        ob['mitigated_at'] = mitigated_at
        
    return obs

def get_active_obs(obs, current_bar, max_age=500):
    return [ob for ob in obs if ob['created_at'] < current_bar < ob['mitigated_at'] and (current_bar - ob['created_at']) <= max_age]

def _format_dt_from_struct_time(st):
    if not st:
        return None
    try:
        return datetime.datetime(*st[:6]).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None

def _format_dt_from_iso(s):
    if not s:
        return None
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(s2)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(s)

def fetch_x_recent_posts(username, count=10):
    """
    Best-effort X (Twitter) fetch using official API v2.

    Uses env vars:
      - X_BEARER_TOKEN: required for API calls
    """
    token = os.getenv("X_BEARER_TOKEN", "").strip()
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    base = "https://api.twitter.com/2"

    try:
        # Resolve username -> user id
        r1 = requests.get(
            f"{base}/users/by/username/{username}",
            headers=headers,
            timeout=15,
        )
        r1.raise_for_status()
        user_id = r1.json()["data"]["id"]

        # Fetch recent tweets
        r2 = requests.get(
            f"{base}/users/{user_id}/tweets",
            headers=headers,
            params={
                "max_results": min(100, int(count)),
                "tweet.fields": "created_at",
            },
            timeout=15,
        )
        r2.raise_for_status()
        tweets = r2.json().get("data", []) or []

        out = []
        for t in tweets[:count]:
            created_at = t.get("created_at")
            created_txt = _format_dt_from_iso(created_at)
            tid = t.get("id")
            url = f"https://x.com/{username}/status/{tid}" if tid else None
            out.append({
                "title": t.get("text", "").strip() or "(no text)",
                "url": url,
                "datetime": created_txt,
            })
        return out
    except Exception:
        return None

def fetch_rss_entries(feed_url_candidates, limit=10, timeout=20):
    """
    Fetch and normalize RSS/Atom items. Returns list[{title,url,datetime}] or None.
    """
    for feed_url in feed_url_candidates:
        try:
            d = feedparser.parse(feed_url)
            if getattr(d, "bozo", False):
                # bozo indicates parse issues but sometimes still has entries; be conservative.
                pass
            entries = getattr(d, "entries", None) or []
            if not entries:
                continue

            out = []
            for e in entries[:limit]:
                title = e.get("title") or e.get("summary") or "(untitled)"
                link = e.get("link") or e.get("id") or e.get("guid") or None
                dt_txt = None

                # Prefer struct_time for consistent formatting.
                dt_txt = _format_dt_from_struct_time(e.get("published_parsed")) or _format_dt_from_struct_time(e.get("updated_parsed"))
                if not dt_txt:
                    dt_txt = e.get("published") or e.get("updated") or None

                out.append({
                    "title": title,
                    "url": link,
                    "datetime": dt_txt,
                })
            return out
        except Exception:
            continue
    return None

def scrape_latest_headlines(page_url_candidates, link_predicate_regexes, limit=10, timeout=20):
    """
    Generic HTML scrape fallback for “latest headlines”.
    """
    for page_url in page_url_candidates:
        try:
            resp = requests.get(
                page_url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            seen = set()
            out = []
            a_tags = soup.find_all("a", href=True)

            for a in a_tags:
                href = a.get("href", "")
                if not href:
                    continue

                href_abs = href if href.startswith("http") else urljoin(page_url, href)

                if not any(re.search(rx, href_abs, flags=re.I) for rx in link_predicate_regexes):
                    continue

                text = a.get_text(" ", strip=True) or a.get("title") or ""
                if not text:
                    continue

                # De-dupe by URL
                if href_abs in seen:
                    continue
                seen.add(href_abs)

                out.append({
                    "title": text,
                    "url": href_abs,
                    "datetime": None,
                })
                if len(out) >= limit:
                    break

            if out:
                return out
        except Exception:
            continue
    return None

def fetch_financialjuice_news(limit=10):
    # Best-effort: official X API if token exists.
    x_items = fetch_x_recent_posts("financialjuice", count=limit)
    if x_items:
        return x_items

    rss_candidates = [
        "https://www.financialjuice.com/feed.ashx?xy=rss",
        # Cloudflare bypass sometimes works on client setups.
        "https://r.jina.ai/https://www.financialjuice.com/feed.ashx?xy=rss",
    ]
    rss_items = fetch_rss_entries(rss_candidates, limit=limit)
    if rss_items:
        return rss_items

    # Fallback: scrape their website “News” links.
    scrape_items = scrape_latest_headlines(
        page_url_candidates=["https://www.financialjuice.com/"],
        link_predicate_regexes=[r"/News/|feed\.ashx\?xy=rss|/news/i"],
        limit=limit,
    )
    return scrape_items

def fetch_zerohedge_news(limit=10):
    x_items = fetch_x_recent_posts("zerohedge", count=limit)
    if x_items:
        return x_items

    rss_candidates = [
        "https://www.zerohedge.com/rss.xml",
        "https://www.zerohedge.com/feed",
        "https://www.zerohedge.com/blog/feed",
        "http://feeds.feedburner.com/zerohedge/feed",
    ]
    rss_items = fetch_rss_entries(rss_candidates, limit=limit)
    if rss_items:
        return rss_items

    scrape_items = scrape_latest_headlines(
        page_url_candidates=["https://www.zerohedge.com/"],
        link_predicate_regexes=[r"/articles/|/article/|/blog/"],
        limit=limit,
    )
    return scrape_items

def fetch_unusual_whales_news(limit=10):
    # Handle likely username variants.
    x_items = fetch_x_recent_posts("unusualwhales", count=limit) or fetch_x_recent_posts("unusual_whales", count=limit)
    if x_items:
        return x_items

    rss_candidates = [
        "https://media.rss.com/unusualwhales/feed.xml",
        "https://rss.com/podcasts/unusualwhales/feed.xml",
    ]
    rss_items = fetch_rss_entries(rss_candidates, limit=limit)
    if rss_items:
        return rss_items

    scrape_items = scrape_latest_headlines(
        page_url_candidates=["https://unusualwhales.com/news-feed", "https://unusualwhales.com/"],
        link_predicate_regexes=[r"unusualwhales\.com/|/podcasts/|/news-feed/"],
        limit=limit,
    )
    return scrape_items

def _extract_forexfactory_weekly_ics_url():
    """
    ForexFactory updates the versioned ICS URL occasionally.
    We try to discover the current version, falling back to the base URL.
    """
    calendar_page = "https://www.forexfactory.com/calendar"
    fallback = "https://nfs.faireconomy.media/ff_calendar_thisweek.ics"

    try:
        resp = requests.get(
            calendar_page,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()

        m = re.search(
            r"(https?://nfs\.faireconomy\.media/ff_calendar_thisweek\.ics\?version=[a-zA-Z0-9]+)",
            resp.text,
        )
        if m:
            return m.group(1)
    except Exception:
        pass

    return fallback

def fetch_forexfactory_economic_news(limit_max_items=200):
    """
    Returns high-impact events from ForexFactory ICS.

    Rules (per user):
      - Major = High impact only
      - USD + United States events: include high-impact events for the current month
      - Other events: include high-impact events for today + next 7 days
    """
    ics_url = _extract_forexfactory_weekly_ics_url()
    try:
        resp = requests.get(
            ics_url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        ics_text = resp.text
    except Exception:
        return []

    cal = ICalendar.from_ical(ics_text)

    now = datetime.datetime.now()
    today = now.date()
    end_week = today + datetime.timedelta(days=7)

    start_month = datetime.date(now.year, now.month, 1)
    if now.month == 12:
        next_month = datetime.date(now.year + 1, 1, 1)
    else:
        next_month = datetime.date(now.year, now.month + 1, 1)
    end_month = next_month - datetime.timedelta(days=1)

    def is_high_impact(text_upper):
        # Try common patterns, but keep regex conservative.
        return (
            bool(re.search(r"\bHIGH\b", text_upper)) or
            "HIGH IMPACT" in text_upper or
            "IMPACT: HIGH" in text_upper
        )

    def is_usa(text_upper):
        return (
            "USD" in text_upper or
            "UNITED STATES" in text_upper or
            bool(re.search(r"\bUS\b", text_upper))
        )

    out = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        summary = str(component.get("SUMMARY", "") or "").strip()
        description = str(component.get("DESCRIPTION", "") or "").strip()
        text_upper = f"{summary} {description}".upper()

        if not is_high_impact(text_upper):
            continue

        dtstart = component.get("DTSTART")
        dt = None
        if dtstart is not None:
            dt = getattr(dtstart, "dt", dtstart)

        if dt is None:
            continue

        if isinstance(dt, datetime.datetime):
            dt_date = dt.date()
            dt_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            # date-only event
            dt_date = dt
            dt_str = str(dt_date)

        usa = is_usa(text_upper)
        if usa:
            if not (start_month <= dt_date <= end_month):
                continue
        else:
            if not (today <= dt_date <= end_week):
                continue

        # Best-effort link extraction from description text.
        url = None
        murl = re.search(r"https?://[^\s\"'>]+", description)
        if murl:
            url = murl.group(0)

        out.append({
            "title": summary or "(untitled)",
            "url": url,
            "datetime": dt_str,
        })

    out.sort(key=lambda x: x.get("datetime") or "")
    return out[:limit_max_items]

def fetch_all_news_concurrent(cache_ttl_seconds=900):
    """
    Fetch brand news + ForexFactory economic releases concurrently with a local cache.
    Cache is keyed per-source to reduce rate limiting and latency.
    """
    os.makedirs("reports", exist_ok=True)
    cache_path = os.path.abspath(os.path.join("reports", "news_cache.json"))

    def load_cache():
        try:
            if not os.path.exists(cache_path):
                return {}
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def save_cache(cache_obj):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    cache = load_cache()
    now_ts = time.time()

    sources = {
        "financialjuice": fetch_financialjuice_news,
        "zerohedge": fetch_zerohedge_news,
        "unusual_whales": fetch_unusual_whales_news,
        "forexfactory": fetch_forexfactory_economic_news,
    }

    results = {}
    to_fetch = {}

    for key, fetch_fn in sources.items():
        cached = cache.get(key)
        fetched_at = cached.get("fetched_at") if isinstance(cached, dict) else None
        items = cached.get("items") if isinstance(cached, dict) else None

        if isinstance(items, list) and fetched_at is not None and (now_ts - float(fetched_at) < cache_ttl_seconds):
            results[key] = items
        else:
            to_fetch[key] = fetch_fn

    if to_fetch:
        max_workers = min(8, max(1, len(to_fetch)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {}
            for key, fetch_fn in to_fetch.items():
                # Economic feed needs more items to satisfy the “month” rule.
                if key == "forexfactory":
                    fut = ex.submit(fetch_fn, limit_max_items=500)
                else:
                    fut = ex.submit(fetch_fn, limit=10)
                future_map[fut] = key

            fetched_keys = set()
            for fut in as_completed(future_map):
                key = future_map[fut]
                fetched_keys.add(key)
                try:
                    results[key] = fut.result() or []
                except Exception:
                    results[key] = []

                cache[key] = {
                    "fetched_at": now_ts,
                    "items": results[key],
                }

            # Persist cache only if we successfully fetched anything.
            save_cache(cache)

    return (
        results.get("financialjuice", []) or [],
        results.get("zerohedge", []) or [],
        results.get("unusual_whales", []) or [],
        results.get("forexfactory", []) or [],
    )

def generate_html(bias, confidence, s1, s2, s3, s4, s1_val, kdj_vals, atr_vals, 
                  df_4h, df_1d, df_1h,
                  active_1d, active_4h, active_1h,
                  type_1d, type_4h, type_1h,
                  news_financialjuice, news_zerohedge, news_unusual_whales, news_forexfactory):
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    close_1d = df_1d['close'].iloc[-1]
    close_4h = df_4h['close'].iloc[-1]
    close_1h = df_1h['close'].iloc[-1]
    
    active_1d = sorted(active_1d, key=lambda x: abs((x['top'] + x['bottom']) / 2 - close_1d))
    active_4h = sorted(active_4h, key=lambda x: abs((x['top'] + x['bottom']) / 2 - close_4h))
    active_1h = sorted(active_1h, key=lambda x: abs((x['top'] + x['bottom']) / 2 - close_1h))
    
    if bias == 'BULLISH':
        bias_color = 'var(--bull)'
        summary = "Indicators and order block alignment suggest upward momentum. Price is likely to seek higher levels tomorrow."
    elif bias == 'BEARISH':
        bias_color = 'var(--bear)'
        summary = "Bearish pressure from multiple timeframes suggests downside risk. Price may seek lower levels tomorrow."
    else:
        bias_color = 'var(--neutral)'
        summary = "Conflicting signals across timeframes. No clear directional edge. Caution advised."
        
    def get_signal_pill(s):
        if s > 0: return '<span class="signal-pill bull">▲ BULL</span>'
        if s < 0: return '<span class="signal-pill bear">▼ BEAR</span>'
        return '<span class="signal-pill neutral">— NEUT</span>'
        
    def render_ob_column(title, active_obs):
        html = f'<div class="ob-column"><div class="ob-header-pill">{title}</div>'
        if not active_obs:
            html += '<div class="empty-state">— No active OBs —</div>'
        else:
            for ob in active_obs:
                type_color = 'var(--bull)' if ob['type'] == 'DEMAND' else 'var(--bear)'
                html += f'<div class="ob-card stagger-item" style="border-left-color: {type_color}">'
                
                type_class = 'bull-bg' if ob['type'] == 'DEMAND' else 'bear-bg'
                html += f'<div class="ob-top-row">'
                html += f'<span class="ob-type-pill {type_class}">{ob["type"]}</span>'
                html += f'<span class="ob-level-pill">{ob["level"]}</span>'
                html += '</div>'
                
                html += f'<div class="ob-price">${ob["top"]:,.2f} &mdash; ${ob["bottom"]:,.2f}</div>'
                
                struct_color = 'var(--neutral)' if ob['structure'] == 'CHoCH' else 'var(--accent)'
                html += f'<div class="ob-struct" style="color: {struct_color}">{ob["structure"]}</div>'
                
                stars_html = ''.join(['<span class="star-filled">★</span>' for _ in range(ob['quality'])])
                stars_html += ''.join(['<span class="star-empty">☆</span>' for _ in range(5 - ob['quality'])])
                html += f'<div class="ob-quality"><span style="color: {type_color}">{stars_html}</span> <span class="ob-score-num">({ob["quality"]}/5)</span></div>'
                
                badges = []
                if ob['quality_displacement']: badges.append('DISP')
                if ob['quality_large_bar']: badges.append('LARGE')
                if ob['quality_fvg']: badges.append('FVG')
                if ob['quality_liquidity_sweep']: badges.append('LIQ')
                if ob['quality_volume_expansion']: badges.append('VOL')
                
                if badges:
                    html += '<div class="ob-badges">'
                    for b in badges:
                        html += f'<span class="ob-badge">{b}</span>'
                    html += '</div>'
                html += '</div>'
        html += '</div>'
        return html
        
    def render_mtf_col(tf, t_type, price):
        if t_type == 'DEMAND':
            text = '↑ DEMAND'
            color = 'var(--bull)'
        elif t_type == 'SUPPLY':
            text = '↓ SUPPLY'
            color = 'var(--bear)'
        else:
            text = '— NONE'
            color = 'var(--muted)'
            
        p_str = f"${price:,.2f} Mid" if t_type else "No OBs"
        return f'''
        <div class="mtf-col">
            <div class="mtf-tf">{tf}</div>
            <div class="mtf-type" style="color: {color}">{text}</div>
            <div class="mtf-price">{p_str}</div>
        </div>
        '''
        
    demands = [t for t in [type_1d, type_4h, type_1h] if t == 'DEMAND']
    supplies = [t for t in [type_1d, type_4h, type_1h] if t == 'SUPPLY']
    
    if len(demands) == 3:
        align_text = '✓ FULL MTF ALIGNMENT'
        align_color = 'var(--bull)'
    elif len(supplies) == 3:
        align_text = '✓ FULL MTF ALIGNMENT'
        align_color = 'var(--bear)'
    elif len(demands) == 2 or len(supplies) == 2:
        align_text = '~ PARTIAL ALIGNMENT (2/3)'
        align_color = 'var(--neutral)'
    else:
        align_text = '✕ NO ALIGNMENT'
        align_color = 'var(--muted)'

    def render_news_items(items, max_items=10):
        if not items:
            return '<div class="empty-state">— No items —</div>'

        out = []
        for it in items[:max_items]:
            title = escape(str(it.get("title", "(untitled)")))
            url = it.get("url")
            dt_txt = it.get("datetime")
            dt_txt = escape(str(dt_txt)) if dt_txt else ""

            if url:
                safe_url = escape(str(url))
                title_html = (
                    f'<a class="news-link" href="{safe_url}" target="_blank" '
                    f'rel="noopener noreferrer">{title}</a>'
                )
            else:
                title_html = f'<div class="news-title">{title}</div>'

            time_html = f'<div class="news-time">{dt_txt}</div>' if dt_txt else ""
            out.append(f'<div class="news-item">{title_html}{time_html}</div>')
        return "".join(out)

    news_financialjuice_html = render_news_items(news_financialjuice)
    news_zerohedge_html = render_news_items(news_zerohedge)
    news_unusual_whales_html = render_news_items(news_unusual_whales)
    # Economic tab can contain multiple “high impact” events across the month.
    news_forexfactory_html = render_news_items(news_forexfactory, max_items=500)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Next-Day Bias Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@400;700&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #080b12;
            --surface: #0d1118;
            --card: #121926;
            --border: #1c2a3a;
            --bull: #00e676;
            --bear: #ff1744;
            --neutral: #ffab00;
            --accent: #00b4ff;
            --text: #dce3f0;
            --muted: #4f6080;
            --grid: #141e2e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'DM Sans', sans-serif;
            min-height: 100vh;
            line-height: 1.45;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background-image: 
              repeating-linear-gradient(0deg, transparent, transparent 39px, var(--grid) 39px, var(--grid) 40px),
              repeating-linear-gradient(90deg, transparent, transparent 39px, var(--grid) 39px, var(--grid) 40px);
        }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 26px 24px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 32px; }}
        .header-left {{ display: flex; flex-direction: column; gap: 4px; }}
        .header-title {{ font-family: 'Orbitron', sans-serif; font-size: 18px; color: var(--accent); font-weight: 700; }}
        .header-subtext {{ font-family: 'DM Sans', sans-serif; color: var(--muted); font-size: 12px; }}
        .header-time {{ font-family: 'JetBrains Mono', monospace; color: var(--muted); font-size: 12px; }}
        .bias-hero {{ text-align: center; margin-bottom: 40px; }}
        .bias-label {{ font-family: 'Orbitron', sans-serif; letter-spacing: 0.3em; color: var(--muted); font-size: 11px; text-transform: uppercase; margin-bottom: 16px; font-weight: 700; }}
        .bias-badge {{
            display: inline-block; font-family: 'Orbitron', sans-serif; font-weight: 900;
            font-size: clamp(48px, 8vw, 96px); color: {bias_color}; text-shadow: 0 0 40px {bias_color}99;
            border: 2px solid {bias_color}4d; padding: 24px 48px; border-radius: 4px;
            background: {bias_color}0d; animation: pulse-glow 2s ease-in-out infinite;
        }}
        @keyframes pulse-glow {{ 0%, 100% {{ box-shadow: 0 0 20px {bias_color}4d; }} 50% {{ box-shadow: 0 0 60px {bias_color}cc, 0 0 100px {bias_color}4d; }} }}
        .conf-label {{ font-family: 'Orbitron', sans-serif; font-size: 11px; letter-spacing: 0.2em; color: var(--muted); margin-top: 32px; font-weight: 700; }}
        .conf-bar {{ display: flex; justify-content: center; gap: 6px; margin-top: 12px; }}
        .conf-segment {{ width: 48px; height: 6px; border-radius: 3px; }}
        .conf-filled {{ background: {bias_color}; }}
        .conf-empty {{ background: var(--border); }}
        .conf-text {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; margin-top: 12px; color: var(--muted); }}
        .bias-summary {{ margin: 20px auto 0; max-width: 480px; font-size: 15px; line-height: 1.6; }}
        .section-title {{ font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 13px; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; margin-bottom: 16px; }}
        .section-note {{ font-family: 'DM Sans', sans-serif; font-size: 12px; color: var(--muted); margin-top: -12px; margin-bottom: 16px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-bottom: 24px; }}
        .ob-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
        .indicator-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 18px; opacity: 1; animation: none; }}
        .ic-1 {{ animation-delay: 0.3s; }} .ic-2 {{ animation-delay: 0.4s; }} .ic-3 {{ animation-delay: 0.5s; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
        .card-title {{ font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 11px; letter-spacing: 0.25em; color: var(--muted); text-transform: uppercase; }}
        .signal-pill {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 11px; padding: 3px 10px; border-radius: 100px; display: inline-flex; }}
        .signal-pill.bull {{ color: var(--bull); background: #00e6761f; border: 1px solid #00e67666; }}
        .signal-pill.bear {{ color: var(--bear); background: #ff17441f; border: 1px solid #ff174466; }}
        .signal-pill.neutral {{ color: var(--neutral); background: #ffab001f; border: 1px solid #ffab0066; }}
        .val-large {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 28px; color: var(--text); line-height: 1.2; }}
        .val-sub {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--muted); margin-top: 4px; }}
        .val-meta {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 12px; color: var(--muted); margin-top: 16px; }}
        .ob-panel {{ opacity: 1; animation: none; }}
        .ob-header-pill {{ display: inline-block; font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 13px; background: #00b4ff1a; border: 1px solid #00b4ff4d; padding: 6px 16px; border-radius: 4px; margin-bottom: 16px; color: var(--accent); }}
        .empty-state {{ text-align: center; color: var(--muted); font-size: 13px; font-style: italic; padding: 20px 0; }}
        .ob-card {{ background: var(--card); border-radius: 0 6px 6px 0; border: 1px solid var(--border); border-left-width: 3px; padding: 14px 16px; margin-bottom: 10px; }}
        .ob-top-row {{ display: flex; justify-content: space-between; align-items: center; }}
        .ob-type-pill {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 10px; text-transform: uppercase; border-radius: 100px; padding: 2px 8px; }}
        .bull-bg {{ color: var(--bull); background: #00e6761f; }} .bear-bg {{ color: var(--bear); background: #ff17441f; }}
        .ob-level-pill {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 10px; color: var(--muted); background: #1c2a3a80; border-radius: 100px; padding: 2px 8px; }}
        .ob-price {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 15px; color: var(--text); margin: 8px 0; }}
        .ob-struct {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 11px; }}
        .ob-quality {{ margin-top: 8px; font-size: 14px; }}
        .star-filled {{ font-size: 14px; }} .star-empty {{ font-size: 14px; color: #4f608066; }}
        .ob-score-num {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 11px; color: var(--muted); }}
        .ob-badges {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }}
        .ob-badge {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 9px; color: var(--accent); background: #00b4ff14; border: 1px solid #00b4ff40; padding: 1px 6px; border-radius: 100px; }}
        .mtf-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; opacity: 1; animation: none; }}
        .mtf-row {{ display: flex; text-align: center; }}
        .mtf-col {{ flex: 1; padding: 0 16px; border-right: 1px solid var(--border); }}
        .mtf-col:last-child {{ border-right: none; }}
        .mtf-tf {{ font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 12px; color: var(--accent); }}
        .mtf-type {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 14px; margin-top: 8px; }}
        .mtf-price {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 11px; color: var(--muted); margin-top: 4px; }}
        .mtf-align {{ font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 13px; text-align: center; margin-top: 24px; padding-top: 24px; border-top: 1px solid var(--border); color: {align_color}; }}
        .footer {{ border-top: 1px solid var(--border); padding-top: 24px; margin-top: 16px; text-align: center; color: var(--muted); font-size: 12px; opacity: 0; animation: fadeUp 0.4s ease forwards; animation-delay: 0.75s; }}
        @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(16px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        /* Top-level tabs */
        .top-tabs-bar {{
            position: sticky;
            top: 0;
            z-index: 30;
            margin: 18px 0 22px;
            padding: 12px 0 12px;
            background: rgba(8, 11, 18, 0.78);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
        }}
        .top-tab-buttons {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .top-tab-btn {{
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.02em;
            color: var(--muted);
            background: #00b4ff0d;
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 999px;
            cursor: pointer;
            transition: transform 160ms ease, background 160ms ease, border-color 160ms ease, color 160ms ease;
        }}
        .top-tab-btn:hover {{ transform: translateY(-1px); border-color: #2a415f; }}
        .top-tab-btn.active {{
            color: var(--accent);
            background: #00b4ff1a;
            border-color: #00b4ff66;
        }}

        .top-tab-pane {{ display: none; }}
        .top-tab-pane.active {{
            display: block;
            animation: panelIn 240ms ease-out;
        }}
        @keyframes panelIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .stagger-item {{
            opacity: 0;
            transform: translateY(10px);
        }}
        .top-tab-pane.active .stagger-item {{
            animation: itemIn 260ms ease-out forwards;
        }}
        @keyframes itemIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* News tabs */
        .news-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 32px;
            opacity: 0;
            animation: fadeUp 0.4s ease forwards;
            animation-delay: 0.6s;
        }}
        .news-tab-buttons {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }}
        .news-tab-btn {{
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 11px;
            color: var(--muted);
            background: #00b4ff0d;
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 999px;
            cursor: pointer;
        }}
        .news-tab-btn.active {{
            color: var(--accent);
            background: #00b4ff1a;
            border-color: #00b4ff66;
        }}
        .news-tab-pane {{ display: none; }}
        .news-tab-pane.active {{
            display: block;
            animation: newsPanelIn 200ms ease-out;
        }}
        @keyframes newsPanelIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .news-item {{
            padding: 10px 0;
            border-top: 1px solid var(--border);
        }}
        .news-item:first-child {{ border-top: none; padding-top: 0; }}
        .news-title {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 13px; color: var(--text); line-height: 1.4; }}
        .news-time {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 11px; color: var(--muted); margin-top: 6px; }}
        .news-link {{ text-decoration: none; color: var(--text); }}
        .news-link:hover {{ color: var(--accent); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header section-block" style="animation: fadeUp 0.4s ease forwards; opacity:0; animation-delay: 0s;">
            <div class="header-left">
                <div class="header-title">₿ BTC/USDT</div>
                <div class="header-subtext">Next-Day Directional Bias</div>
            </div>
            <div class="header-time">{timestamp}</div>
        </div>
        
        <div class="bias-hero section-block" style="animation: fadeUp 0.4s ease forwards; opacity:0; animation-delay: 0.15s;">
            <div class="bias-label">TOMORROW'S BIAS</div>
            <div class="bias-badge">{bias}</div>
            <div class="conf-label">SIGNAL CONFIDENCE</div>
            <div class="conf-bar">
                {''.join(['<div class="conf-segment conf-filled"></div>' for _ in range(confidence)])}
                {''.join(['<div class="conf-segment conf-empty"></div>' for _ in range(5 - confidence)])}
            </div>
            <div class="conf-text">{confidence} / 5 signals confirm</div>
            <div class="bias-summary">{summary}</div>
        </div>
        
        <div class="top-tabs-bar">
            <div class="top-tab-buttons">
                <button class="top-tab-btn active" type="button" data-tab="tab_indicators">Indicators</button>
                <button class="top-tab-btn" type="button" data-tab="tab_orderblocks">Order Blocks</button>
                <button class="top-tab-btn" type="button" data-tab="tab_news">News</button>
            </div>
        </div>

        <div class="top-tab-pane active" id="tab_indicators">
            <div class="section-title">INDICATOR SIGNALS</div>
            <div class="grid-3">
                <div class="indicator-card stagger-item ic-1">
                    <div class="card-header">
                        <div class="card-title">MACD</div>
                        {get_signal_pill(s1)}
                    </div>
                    <div class="val-large">{'+' if s1_val > 0 else ''}{s1_val:,.1f}</div>
                    <div class="val-meta">Hist trend: {'↑' if s1 == 1 else '↓' if s1 == -1 else '-'}</div>
                </div>
                <div class="indicator-card stagger-item ic-2">
                    <div class="card-header">
                        <div class="card-title">KDJ</div>
                        {get_signal_pill(s2)}
                    </div>
                    <div class="val-sub" style="font-size: 14px; margin-top: 0; font-weight:600; color:var(--text);">K: {kdj_vals[0]:.1f}<br>D: {kdj_vals[1]:.1f}<br>J: {kdj_vals[2]:.1f}</div>
                    <div class="val-meta">Zone: {'Overbought' if kdj_vals[2] > 80 else 'Oversold' if kdj_vals[2] < 20 else 'Neutral'}</div>
                </div>
                <div class="indicator-card stagger-item ic-3">
                    <div class="card-header">
                        <div class="card-title">ATR</div>
                        <span class="signal-pill neutral">— NEUT</span>
                    </div>
                    <div class="val-large">{atr_vals[0]:.1f}</div>
                    <div class="val-sub">ATR/200 ratio: {atr_vals[1]:.2f}</div>
                    <div class="val-meta">Regime: {'Low Vol' if atr_vals[1] < 0.8 else 'High Vol'}</div>
                </div>
            </div>
            
            <div class="mtf-card">
                <div class="card-title" style="margin-bottom: 16px;">MTF CONFLUENCE</div>
                <div class="mtf-row">
                    {render_mtf_col("1D", type_1d, sum([o['top']+o['bottom'] for o in active_1d if o['type']==type_1d])/2/len([o for o in active_1d if o['type']==type_1d]) if type_1d and [o for o in active_1d if o['type']==type_1d] else 0)}
                    {render_mtf_col("4H", type_4h, sum([o['top']+o['bottom'] for o in active_4h if o['type']==type_4h])/2/len([o for o in active_4h if o['type']==type_4h]) if type_4h and [o for o in active_4h if o['type']==type_4h] else 0)}
                    {render_mtf_col("1H", type_1h, sum([o['top']+o['bottom'] for o in active_1h if o['type']==type_1h])/2/len([o for o in active_1h if o['type']==type_1h]) if type_1h and [o for o in active_1h if o['type']==type_1h] else 0)}
                </div>
                <div class="mtf-align">{align_text}</div>
            </div>
        </div>

        <div class="top-tab-pane" id="tab_orderblocks">
            <div class="section-title">ACTIVE ORDER BLOCKS — MULTI-TIMEFRAME</div>
            <div class="section-note">All unmitigated OBs shown. Quality scored 0–5.</div>
            <div class="ob-grid ob-panel">
                {render_ob_column("1D", active_1d)}
                {render_ob_column("4H", active_4h)}
                {render_ob_column("1H", active_1h)}
            </div>
        </div>

        <div class="top-tab-pane" id="tab_news">
            <div class="section-title">NEWS</div>
            <div class="news-card">
                <div class="news-tab-buttons">
                    <button class="news-tab-btn active" type="button" data-tab="news_financialjuice">FinancialJuice</button>
                    <button class="news-tab-btn" type="button" data-tab="news_zerohedge">ZeroHedge</button>
                    <button class="news-tab-btn" type="button" data-tab="news_unusual_whales">Unusual Whales</button>
                    <button class="news-tab-btn" type="button" data-tab="news_forexfactory">ForexFactory (Economic)</button>
                </div>

                <div class="news-tab-pane active" id="news_financialjuice">
                    {news_financialjuice_html}
                </div>
                <div class="news-tab-pane" id="news_zerohedge">
                    {news_zerohedge_html}
                </div>
                <div class="news-tab-pane" id="news_unusual_whales">
                    {news_unusual_whales_html}
                </div>
                <div class="news-tab-pane" id="news_forexfactory">
                    {news_forexfactory_html}
                </div>
            </div>
        </div>

        <script>
            (function () {{
                const btns = document.querySelectorAll('.top-tab-btn');
                const panes = document.querySelectorAll('.top-tab-pane');
                function setActive(tabId) {{
                    btns.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
                    panes.forEach(p => p.classList.toggle('active', p.id === tabId));

                    const activePane = document.getElementById(tabId);
                    if (activePane) {{
                        const items = activePane.querySelectorAll('.stagger-item');
                        items.forEach((el, idx) => {{
                            el.style.animationDelay = (idx * 70) + 'ms';
                        }});
                    }}
                }}
                btns.forEach(b => b.addEventListener('click', () => setActive(b.dataset.tab)));
                const initial = document.querySelector('.top-tab-btn.active');
                if (initial) setActive(initial.dataset.tab);
            }})();
        </script>

        <script>
            (function () {{
                const btns = document.querySelectorAll('.news-tab-btn');
                const panes = document.querySelectorAll('.news-tab-pane');
                const setActive = (tabId) => {{
                    btns.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
                    panes.forEach(p => p.classList.toggle('active', p.id === tabId));
                }};
                btns.forEach(b => b.addEventListener('click', () => setActive(b.dataset.tab)));
            }})();
        </script>
        
        <div class="footer">
            Data: Binance Public API (BTCUSDT) &middot; Generated {timestamp} &middot; This is not financial advice. For educational purposes only.
        </div>
    </div>
</body>
</html>
"""
    return html

def main():
    print("[1/7] Fetching candles...")
    df_1d = fetch_candles("1d")
    df_4h = fetch_candles("4h")
    df_1h = fetch_candles("1h")
    
    print("[2/7] Computing indicators...")
    df_1d = compute_indicators(df_1d)
    df_4h = compute_indicators(df_4h)
    df_1h = compute_indicators(df_1h)
    
    print("[3/7] Detecting order blocks...")
    obs_1d = compute_smc(df_1d)
    obs_4h = compute_smc(df_4h)
    obs_1h = compute_smc(df_1h)
    
    print("[4/7] Filtering active OBs...")
    last_idx_1d = len(df_1d) - 1
    last_idx_4h = len(df_4h) - 1
    last_idx_1h = len(df_1h) - 1
    
    active_1d = get_active_obs(obs_1d, last_idx_1d)
    active_4h = get_active_obs(obs_4h, last_idx_4h)
    active_1h = get_active_obs(obs_1h, last_idx_1h)
    
    print("[5/7] Evaluating bias...")
    close_4h = df_4h['close'].iloc[-1]
    MACD_hist = df_4h['MACD_hist'].iloc[-1]
    MACD_hist_prev = df_4h['MACD_hist'].iloc[-2]
    K = df_4h['K'].iloc[-1]
    D = df_4h['D'].iloc[-1]
    J = df_4h['J'].iloc[-1]
    ATR = df_4h['ATR'].iloc[-1]
    ATR_200 = df_4h['ATR_200'].iloc[-1]
    
    s1 = 1 if (MACD_hist > 0 and MACD_hist > MACD_hist_prev) else -1 if (MACD_hist < 0 and MACD_hist < MACD_hist_prev) else 0
    s2 = 1 if (K > D and J > 50) else -1 if (K < D and J < 50) else 0
    
    if active_4h:
        nearest_ob = min(active_4h, key=lambda x: abs((x['top'] + x['bottom']) / 2 - close_4h))
        s3 = 1 if nearest_ob['type'] == 'DEMAND' else -1
    else:
        s3 = 0
        
    def get_nearest_type(obs_list, close_price):
        if not obs_list: return None
        nearest = min(obs_list, key=lambda x: abs((x['top'] + x['bottom']) / 2 - close_price))
        return nearest['type']
        
    type_1d = get_nearest_type(active_1d, df_1d['close'].iloc[-1])
    type_4h = get_nearest_type(active_4h, df_4h['close'].iloc[-1])
    type_1h = get_nearest_type(active_1h, df_1h['close'].iloc[-1])
    
    types = [t for t in [type_1d, type_4h, type_1h] if t]
    demands = types.count('DEMAND')
    supplies = types.count('SUPPLY')
    
    if demands == 3: s4 = 2
    elif supplies == 3: s4 = -2
    elif demands == 2: s4 = 1
    elif supplies == 2: s4 = -1
    else: s4 = 0
    
    atr_ratio = ATR / ATR_200 if ATR_200 > 0 else 1
    raw_score = s1 + s2 + s3 + s4
    effective_score = raw_score * 0.5 if atr_ratio < 0.8 else raw_score
    
    if effective_score >= 2: bias = 'BULLISH'
    elif effective_score <= -2: bias = 'BEARISH'
    else: bias = 'NEUTRAL'
    
    if bias == 'NEUTRAL':
        confidence = max(1, min(5, 5 - abs(raw_score)))
    else:
        target = 1 if bias == 'BULLISH' else -1
        conf_count = sum([1 for s in [s1, s2, s3, np.sign(s4)] if np.sign(s) == target])
        confidence = max(1, min(5, conf_count))
        
    print("[6/7] Fetching news & economics...")
    news_financialjuice, news_zerohedge, news_unusual_whales, news_forexfactory = fetch_all_news_concurrent()

    print("[7/7] Generating HTML report...")
    html_out = generate_html(bias, confidence, s1, s2, s3, s4, MACD_hist, (K, D, J), (ATR, atr_ratio), 
                             df_4h, df_1d, df_1h, active_1d, active_4h, active_1h, type_1d, type_4h, type_1h,
                             news_financialjuice, news_zerohedge, news_unusual_whales, news_forexfactory)
    
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.abspath(os.path.join("reports", "btc_report.html"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_out)
        
    print("✓ Done.")
    print(f"✓ Report generated: {report_path}")
    print(f"✓ Bias: {bias} (Confidence: {confidence}/5)")
    print(f"✓ Opening in Google Chrome...")
    try:
        # Try standard 64-bit Chrome path
        webbrowser.get('C:/Program Files/Google/Chrome/Application/chrome.exe %s').open('file://' + report_path)
    except Exception:
        try:
            # Try 32-bit Chrome path
            webbrowser.get('C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s').open('file://' + report_path)
        except Exception:
            # Fallback to default browser
            webbrowser.open('file://' + report_path)

if __name__ == "__main__":
    main()
