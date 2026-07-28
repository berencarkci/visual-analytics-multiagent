import sys; sys.path.insert(0, "agents")
import pandas as pd
from transforms import apply_transform
from schemas import ChartRecommendation, Transform
from data_analyst import _compute_stats
from messages import TransformPlan

df = pd.DataFrame({"cat": ["6","5e","6","5e","5e"], "amount": [1.0,2,3,4,5],
                   "active": [True,False,True,True,False],
                   "score": [3.0,None,4.5,None,2.0]})

# 1) gruplanmamış agregasyon skaler dönmeli
rec = ChartRecommendation(chart_type="bar", x_axis="amount", y_axis=None,
                          transform=Transform(agg="mean"), reason="-", insight="-")
out, x, y, notes = apply_transform(df, rec)
print("single-value:", out.to_dict("records"), x, y)   # metric / mean(amount)=3.0

# 2) kategorik kolonda coercion yerine sayım
plan = TransformPlan(transform=Transform(), target_columns=["cat"])
s = _compute_stats("distribution_stats", df, df, "cat", None, plan)
print("cat stats focus:", s.get("focus"), s.get("counts"))   # category_counts, {'5e': 3, '6': 2}

# 3) boolean çökmemeli
s = _compute_stats("distribution_stats", df, df, "active", None, plan)
print("bool stats:", {k: s[k] for k in ("focus","counts") if k in s} or "numeric path ok")

# 4) korelasyonda NaN ifşası + düşük-n uyarısı
plan = TransformPlan(transform=Transform(), target_columns=["amount","score"])
s = _compute_stats("correlation", df, df, "amount", "score", plan)
print("corr:", s.get("n"), s.get("n_rows_dropped_missing"), s.get("caution") is not None)