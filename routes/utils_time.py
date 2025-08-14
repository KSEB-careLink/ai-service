# routes/utils_time.py
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import re
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc

MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")

def ensure_utc_series(s: pd.Series) -> pd.Series:
    """
    입력 Series를 안전하게 'UTC tz-aware'로 표준화.
    - tz-aware 값: UTC로 tz_convert
    - naive 값: UTC로 tz_localize
    - 혼합/문자열: 우선 to_datetime으로 파싱한 뒤 위 규칙 적용
    """
    # 1) 우선 파싱 (tz 정보가 있으면 유지, 없으면 naive)
    s_parsed = pd.to_datetime(s, errors="coerce")  # utc=False (중요: tz 가진 값이 섞여있어도 에러 X)

    # 2) 전부 tz-aware(같은 tz)면 바로 convert
    try:
        tz = s_parsed.dt.tz
    except AttributeError:
        # 전부 NaT인 경우 등
        return s_parsed

    if tz is not None:
        # 이미 tz-aware Series → UTC로 변환
        return s_parsed.dt.tz_convert(UTC)

    # 3) tz가 없는 경우(naive) → UTC로 localize
    return s_parsed.dt.tz_localize(UTC)


def kst_month_window_utc(month_str: str) -> tuple[datetime, datetime]:
    """
    month_str='YYYY-MM'을 KST 기준 월 경계로 해석하여
    [start_utc, end_utc) (UTC tz-aware) 튜플 반환.
    """
    if not MONTH_PATTERN.match(month_str):
        raise ValueError("month must be YYYY-MM")

    year = int(month_str[:4])
    month = int(month_str[5:7])

    start_kst = datetime(year, month, 1, 0, 0, 0, tzinfo=KST)

    # 다음달 1일(KST)
    if month == 12:
        end_kst = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=KST)
    else:
        end_kst = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=KST)

    # UTC로 변환 (반열린 구간)
    start_utc = start_kst.astimezone(UTC)
    end_utc = end_kst.astimezone(UTC)
    return start_utc, end_utc
