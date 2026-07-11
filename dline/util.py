def subtract_months(dt, months):
    year = dt.year - (months // 12)
    month = dt.month - (months % 12)
    if month <= 0:
        year -= 1
        month += 12
    # 处理天数溢出，例如 3 月 31 日减去一个月变为 2 月份，会被修正为 2 月 28/29 日
    day = min(dt.day, [31, 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return dt.replace(year=year, month=month, day=day)