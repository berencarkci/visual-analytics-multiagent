"""SFT training data generator.
Builds data/sft_train.jsonl from three sources:
  - template: questions derived from answer templates, so the target is correct by construction
  - handwritten:vague/free form questions with hand chosen targets
  - failure_targeted: correct answers to observed model failures, maintained by hand in evaluation/failure_examples.py
Design rules baked in:
  - Targets embody the guardrails (pie only <=5 categories, box for discrete x relationships, histogram without groupby) instead of stating them as prose.
  - Insight fields are pointer style: they describe what the chart will show and never contain numbers the model cannot compute from the schema alone
  - Short system prompt, no few shots: format knowledge should move into the weights, the long frozen baseline prompt stays untouched for baseline measurement.
  - Every target is validated through the pydantic ChartRecommendation schema,and the whole question set passes the benchmark contamination check before the file is written.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, "agents")
from check_contamination import check_contamination
from data_ingestion import load_table, profile_table, schema_summary
from failure_examples import FAILURE_EXAMPLES
from schemas import ChartRecommendation
from prompts import SFT_SYSTEM
random.seed(42)
# Dataset registry (verified column names):
DATASETS = {
    "retail": {"path": "data/retail_sales_superstore.csv", "date": "order_date", "metrics": ["sales", "profit", "quantity"], "small_cats": ["category", "region", "segment", "ship_mode"], "large_cats": ["sub_category", "state"]},
    "mall": {"path": "data/customer_analytics_mall.csv", "date": None, "metrics": ["age", "annual_income_k_usd", "spending_score"], "small_cats": ["gender"], "large_cats": []},
    "energy": {"path": "data/energy_consumption_hourly.csv", "date": "date", "metrics": ["appliances", "lights"], "small_cats": [], "large_cats": []}}
_PRETTY = {"annual_income_k_usd": "annual income", "spending_score": "spending score", "sub_category": "sub-category", "ship_mode": "ship mode", "order_date": "order date", "appliances": "appliance consumption", "lights": "light consumption", "t1": "kitchen temperature", "t2": "living room temperature", "t3": "laundry room temperature", "rh_1": "kitchen humidity", "rh_2": "living room humidity"}
def pretty(col: str) -> str:
    return _PRETTY.get(col, col.replace("_", " "))
def target(chart, x, y, groupby=None, agg=None, filter=None, sort=None, limit=None, reason="", insight=""):
    return {"chart_type": chart, "x_axis": x, "y_axis": y, "transform": {"groupby": groupby, "agg": agg, "filter": filter, "sort": sort, "limit": limit}, "reason": reason, "insight": insight}
#################################
# Template banks
def build_template_examples() -> list[dict]:
    ex: list[dict] = []
    # The intent each bank below produces. Agent format training data needs it (the Data Analyst and Visualization prompts both take the intent), and it is known by construction here, the banks are the intent taxonomy.
    bank = {"intent": "comparison"}
    def add(ds, q, t):
        ex.append({"dataset": ds, "question": q, "target": t, "source": "template", "intent": bank["intent"]})
    bank["intent"] = "trend"
    # trend (business metrics, sum)
    for ds in ("retail", "energy"):
        d = DATASETS[ds]
        for metric in d["metrics"][:3]:
            for period, expr in [("monthly", "month"), ("weekly", "week"), ("quarterly", "quarter")]:
                gb = f"{expr}({d['date']})"
                t = target("line", d["date"], metric, groupby=gb, agg="sum", sort="date_asc",
                           reason=f"A line chart shows how {pretty(metric)} develops over time.",
                           insight=f"The chart shows the {period} course of total {pretty(metric)} and where it peaks.")
                for q in [f"Trace the {period} evolution of total {pretty(metric)}.",
                          f"Show the {period} trend of {pretty(metric)}.",
                          f"Plot how {pretty(metric)} moved {period} through the data."]:
                    add(ds, q, t)
    bank["intent"] = "trend"
    #trend (indoor climate, mean)
    for col in ["t1", "t2", "rh_1"]:
        t = target("line", "date", col, groupby="day(date)", agg="mean", sort="date_asc",
                   reason=f"A daily line shows how {pretty(col)} developed.",
                   insight=f"The chart shows the daily course of average {pretty(col)}.")
        for q in [f"Trace how average {pretty(col)} developed day by day.",
                  f"Show the daily course of {pretty(col)}."]:
            add("energy", q, t)
    bank["intent"] = "comparison"
    # comparison
    for metric in DATASETS["retail"]["metrics"]:
        for cat in DATASETS["retail"]["small_cats"]:
            for agg, agg_word in [("sum", "total"), ("mean", "average")]:
                t = target("bar", cat, metric, groupby=cat, agg=agg, sort="value_desc",
                           reason=f"Bars compare {agg_word} {pretty(metric)} across {pretty(cat)} groups.",
                           insight=f"The chart shows which {pretty(cat)} leads in {agg_word} {pretty(metric)}.")
                for q in [f"Rank the {pretty(cat)} groups by {agg_word} {pretty(metric)}.",
                          f"Where is {agg_word} {pretty(metric)} strongest among the {pretty(cat)} groups?",
                          f"Line the {pretty(cat)} groups up by their {agg_word} {pretty(metric)}."]:
                    add("retail", q, t)
    for metric in DATASETS["mall"]["metrics"][1:]:
        t = target("bar", "gender", metric, groupby="gender", agg="mean",
                   reason=f"Bars compare average {pretty(metric)} between the two groups.",
                   insight=f"The chart shows whether average {pretty(metric)} differs by gender.")
        for q in [f"Put the two genders side by side on average {pretty(metric)}.",
                  f"Do men and women differ in average {pretty(metric)}?"]:
            add("mall", q, t)
    t = target("bar", "date", "appliances", groupby="day_of_week(date)", agg="mean",
               reason="Seven weekday groups compare naturally as bars.",
               insight="The chart shows which weekdays average higher appliance consumption.")
    for q in ["Which weekday has the highest average appliance consumption?",
              "Compare average appliance usage by weekday."]:
        add("energy", q, t)
    for metric in ["spending_score", "annual_income_k_usd"]:
        t = target("bar", "age", metric, groupby="bins(age)", agg="mean",
                   reason="Age bins turn a numeric axis into comparable groups.",
                   insight=f"The chart shows how average {pretty(metric)} changes across age bands.")
        for q in [f"Break customers into age bands and rank them by average {pretty(metric)}.",
                  f"How does average {pretty(metric)} differ across age bands?"]:
            add("mall", q, t)
    for col in ["appliances", "lights"]:
        t = target("bar", "date", col, groupby="weekend_flag(date)", agg="mean",
                   reason="Two bars contrast weekend and weekday averages directly.",
                   insight=f"The chart shows whether {pretty(col)} runs higher on weekends or weekdays.")
        for q in [f"Contrast weekend and weekday average {pretty(col)}.",
                  f"Is {pretty(col)} higher on weekends than on weekdays?"]:
            add("energy", q, t)
        t2 = target("bar", "date", col, groupby="hour_of_day(date)", agg="mean",
                    reason="Hourly averages reveal the daily usage rhythm.",
                    insight=f"The chart shows which hours of the day average the highest {pretty(col)}.")
        add("energy", f"Map the daily rhythm of {pretty(col)} by hour.", t2)
    bank["intent"] = "composition"
    # composition (pie only for <=5 categories; large cats -> bar)
    for metric in ["sales", "quantity"]:
        for cat in DATASETS["retail"]["small_cats"]:
            t = target("pie", cat, metric, groupby=cat, agg="sum",
                       reason=f"A pie shows each {pretty(cat)}'s share of a small whole.",
                       insight=f"The chart shows how total {pretty(metric)} splits across {pretty(cat)} groups.")
            for q in [f"Break total {pretty(metric)} down into {pretty(cat)} slices.",
                      f"Show the {pretty(metric)} composition by {pretty(cat)}.",
                      f"How does each {pretty(cat)} slice the total {pretty(metric)} pie?"]:
                add("retail", q, t)
        for cat in DATASETS["retail"]["large_cats"]:
            t = target("bar", cat, metric, groupby=cat, agg="sum", sort="value_desc",
                       reason=f"With many {pretty(cat)} groups a pie is unreadable; sorted bars still show shares.",
                       insight=f"The chart shows each {pretty(cat)}'s contribution to total {pretty(metric)}.")
            add("retail", f"How is total {pretty(metric)} distributed across {pretty(cat)} groups?", t)
    bank["intent"] = "relationship"
    # relationship (scatter for continuous pairs)
    pairs = [("mall", "age", "annual_income_k_usd"), ("mall", "age", "spending_score"),
             ("mall", "annual_income_k_usd", "spending_score"),
             ("retail", "sales", "profit"),
             ("energy", "t1", "appliances"), ("energy", "t2", "appliances"),
             ("energy", "rh_1", "appliances"), ("energy", "t1", "t2"),
             ("energy", "rh_1", "rh_2"), ("energy", "t3", "appliances")]
    for ds, a, b in pairs:
        t = target("scatter", a, b,
                   reason="A scatter plot shows how two numeric variables move together.",
                   insight=f"The chart shows whether {pretty(a)} and {pretty(b)} are related.")
        for q in [f"Do {pretty(a)} and {pretty(b)} move together?",
                  f"How does {pretty(b)} vary with {pretty(a)}?",
                  f"Chart {pretty(a)} against {pretty(b)} and see if they track each other."]:
            add(ds, q, t)
    bank["intent"] = "distribution"
    # distribution (histogram, no groupby)
    dist_cols = [("retail", "sales"), ("retail", "profit"),
                 ("mall", "age"), ("mall", "annual_income_k_usd"), ("mall", "spending_score"),
                 ("energy", "appliances"), ("energy", "lights"),
                 ("energy", "t1"), ("energy", "t2"), ("energy", "rh_1")]
    for ds, col in dist_cols:
        t = target("histogram", col, None,
                   reason="A histogram bins the raw values to show their spread.",
                   insight=f"The chart shows how {pretty(col)} values are spread across their range.")
        for q in [f"Show the value profile of {pretty(col)}.",
                  f"Bin the {pretty(col)} values and show their shape.",
                  f"Show a histogram of {pretty(col)}.",
                  f"What does the spread of {pretty(col)} look like?"]:
            add(ds, q, t)
    bank["intent"] = "filter_aggregation"
    # filter_aggregation
    for n in (5, 10):
        for cat in ["state", "sub_category", "city"]:
            for metric in ["sales", "profit"]:
                t = target("bar", cat, metric, groupby=cat, agg="sum",
                           sort="value_desc", limit=n,
                           reason=f"Sorted bars limited to the top {n} keep the leaders readable.",
                           insight=f"The chart shows the top {n} {pretty(cat)} groups by total {pretty(metric)}.")
                add("retail", f"Limit the view to the {n} strongest {pretty(cat)} groups by {pretty(metric)}.", t)
    for year in (2016, 2017, 2018):
        t = target("bar", "category", "sales", groupby="category", agg="sum",
                   filter=f"year(order_date) == {year}",
                   reason="Bars compare the categories within the filtered year.",
                   insight=f"The chart shows how categories compare on total sales in {year} alone.")
        add("retail", f"Compare total sales by category for {year} only.", t)
    for seg in ["Consumer", "Corporate"]:
        t = target("bar", "category", "sales", groupby="category", agg="sum",
                   filter=f"segment == '{seg}'",
                   reason="Bars compare the categories within the filtered segment.",
                   insight=f"The chart shows how categories compare on sales within the {seg} segment alone.")
        add("retail", f"Looking only at the {seg} segment, rank the categories by total sales.", t)
    for reg in ["West", "East"]:
        t = target("bar", "sub_category", "profit", groupby="sub_category", agg="sum",
                   sort="value_desc", limit=10, filter=f"region == '{reg}'",
                   reason="Sorted, limited bars keep the filtered ranking readable.",
                   insight=f"The chart shows the strongest sub-categories by profit within the {reg} region.")
        add("retail", f"Within the {reg} region only, show the 10 strongest sub-categories by profit.", t)
    bank["intent"] = "distribution"
    # grouped distributions (box: how a numeric spreads across groups)
    box_combos = [("retail", "segment", "sales"), ("retail", "region", "profit"), ("retail", "category", "profit"), ("mall", "gender", "spending_score"), ("mall", "gender", "age")]
    for ds, cat, num in box_combos:
        t = target("box", cat, num,
                   reason=f"Grouped boxes compare the full {pretty(num)} distribution per {pretty(cat)}.",
                   insight=f"The chart shows how the spread of {pretty(num)} differs across {pretty(cat)} groups.")
        for q in [f"How does the {pretty(num)} distribution differ across {pretty(cat)} groups?",
                  f"Compare the spread of {pretty(num)} per {pretty(cat)}."]:
            add(ds, q, t)
    bank["intent"] = "comparison"
    # counting (count / count_distinct targets)
    for cat in DATASETS["retail"]["small_cats"] + DATASETS["retail"]["large_cats"]:
        t = target("bar", cat, "order_id", groupby=cat, agg="count", sort="value_desc",
                   reason=f"Counting rows per {pretty(cat)} answers a how-many question.",
                   insight=f"The chart shows how order volume splits across {pretty(cat)} groups.")
        for q in [f"How many order lines does each {pretty(cat)} account for?",
                  f"Count the order lines per {pretty(cat)}."]:
            add("retail", q, t)
    for cat in ["region", "segment", "category"]:
        t = target("bar", cat, "customer_id", groupby=cat, agg="count_distinct", sort="value_desc",
                   reason=f"Distinct customer counts avoid double-counting repeat buyers.",
                   insight=f"The chart shows how many unique customers each {pretty(cat)} serves.")
        for q in [f"How many unique customers does each {pretty(cat)} serve?",
                  f"Count the distinct customers per {pretty(cat)}."]:
            add("retail", q, t)
    bank["intent"] = "filter_aggregation"
    # filtered time series (filter + time groupby: the combination the model dropped filters on, every filter bank above is categorical + bar)
    for cat_val, cat_col in [("Technology", "category"), ("Furniture", "category"), ("Consumer", "segment"), ("West", "region")]:
        for metric in ["sales", "profit"]:
            t = target("line", "order_date", metric, groupby="month(order_date)",
                       agg="sum", sort="date_asc", filter=f"{cat_col} == '{cat_val}'",
                       reason=f"A monthly line restricted to {cat_val} shows that slice's own course over time.",
                       insight=f"The chart shows how monthly {pretty(metric)} developed for {cat_val}.")
            for q in [f"Show the monthly {pretty(metric)} course for {cat_val} only.",
                      f"Restricted to {cat_val}, how did monthly {pretty(metric)} move?"]:
                add("retail", q, t)
    bank["intent"] = "filter_aggregation"
    # combined filters (two conditions at once)
    for cat_val, year in [("Technology", 2018), ("Furniture", 2017), ("Office Supplies", 2016)]:
        t = target("line", "order_date", "sales", groupby="month(order_date)", agg="sum",
                   sort="date_asc", filter=f"category == '{cat_val}' and year(order_date) == {year}",
                   reason="Both conditions belong in the filter; the groupby then splits the remaining rows by month.",
                   insight=f"The chart shows the monthly sales course for {cat_val} within {year}.")
        for q in [f"Monthly sales for {cat_val} within {year} only.",
                  f"Narrow the data to {cat_val} in {year} and show the monthly sales course."]:
            add("retail", q, t)
    for reg, seg in [("West", "Consumer"), ("East", "Corporate")]:
        t = target("bar", "category", "sales", groupby="category", agg="sum",
                   sort="value_desc", filter=f"region == '{reg}' and segment == '{seg}'",
                   reason="Both restrictions go into the filter, then the categories are compared inside that slice.",
                   insight=f"The chart compares categories within the {seg} segment of the {reg} region.")
        add("retail", f"Inside the {reg} region, {seg} segment only, rank the categories by sales.", t)
    bank["intent"] = "anomaly"
    # anomaly (a line over time; the groupby granularity follows the wording of the question, and bar is never an anomaly answer)
    _UNITS = [("days", "day", "daily"), ("weeks", "week", "weekly"), ("months", "month", "monthly")]
    for ds, metric in [("energy", "appliances"), ("energy", "lights"), ("retail", "sales"), ("retail", "profit")]:
        d = DATASETS[ds]
        for plural, expr, adverb in _UNITS:
            t = target("line", d["date"], metric, groupby=f"{expr}({d['date']})",
                       agg="sum", sort="date_asc",
                       reason=f"A {adverb} line puts every period on one axis, so the ones breaking from the pattern stand out.",
                       insight=f"The chart makes {plural} with unusual total {pretty(metric)} visible.")
            for q in [f"Flag the {plural} where total {pretty(metric)} spiked beyond the norm.",
                      f"Identify the {plural} whose {pretty(metric)} ran abnormally high.",
                      f"Which {plural} had {pretty(metric)} totals that look out of line?"]:
                add(ds, q, t)
    return ex
#################################
# Handwritten vague / free form examples:
def build_handwritten_examples() -> list[dict]:
    H = [
        ("retail", "Does anything stand out in this data?",
         target("line", "order_date", "sales", groupby="month(order_date)", agg="sum", sort="date_asc",
                reason="Without a specific focus, the sales timeline is the most informative overview.",
                insight="The chart gives an overview of sales over time where unusual periods stand out.")),
        ("retail", "Give me a quick overview of the business.",
         target("bar", "category", "sales", groupby="category", agg="sum", sort="value_desc",
                reason="Total sales by category is the broadest single overview of the business.",
                insight="The chart shows which product categories drive the business.")),
        ("retail", "Where does the money come from?",
         target("bar", "region", "sales", groupby="region", agg="sum", sort="value_desc",
                reason="Regional totals answer where revenue originates.",
                insight="The chart shows which regions contribute most of the revenue.")),
        ("retail", "Anything worrying in the numbers?",
         target("line", "order_date", "profit", groupby="month(order_date)", agg="sum", sort="date_asc",
                reason="A monthly profit line exposes declines or unusual dips.",
                insight="The chart makes weak or declining profit periods visible.")),
        ("mall", "What do our customers look like?",
         target("histogram", "age", None,
                reason="The age distribution is the most basic customer profile view.",
                insight="The chart shows which age ranges dominate the customer base.")),
        ("mall", "Is there something interesting about our customers?",
         target("scatter", "annual_income_k_usd", "spending_score",
                reason="The income-spending relationship is the most revealing customer view.",
                insight="The chart shows whether higher income goes together with higher spending.")),
        ("mall", "Who spends the most here?",
         target("bar", "gender", "spending_score", groupby="gender", agg="mean",
                reason="Average spending per group answers who spends more.",
                insight="The chart shows which customer group has the higher average spending score.")),
        ("energy", "Is our energy usage normal?",
         target("line", "date", "appliances", groupby="day(date)", agg="sum", sort="date_asc",
                reason="A daily consumption line exposes abnormal days.",
                insight="The chart makes unusually high or low consumption days visible.")),
        ("energy", "When do we use the most electricity?",
         target("bar", "date", "appliances", groupby="hour_of_day(date)", agg="mean",
                reason="Hourly averages show the daily usage rhythm.",
                insight="The chart shows which hours of the day consume the most on average.")),
        ("energy", "Summarize the consumption pattern for me.",
         target("line", "date", "appliances", groupby="day(date)", agg="sum", sort="date_asc",
                reason="The daily line is the most complete single summary of consumption.",
                insight="The chart summarizes how consumption evolved day by day.")),
        ("retail", "Which products should we worry about?",
         target("bar", "sub_category", "profit", groupby="sub_category", agg="sum", sort="value_desc",
                reason="Sorted profit totals expose the weakest sub-categories at the bottom.",
                insight="The chart shows which sub-categories earn the least or lose money.")),
        ("mall", "Split the customers into income levels for me.",
         target("histogram", "annual_income_k_usd", None,
                reason="A histogram naturally bins customers into income levels.",
                insight="The chart shows how many customers fall into each income range.")),
        ("retail", "How healthy is the business right now?",
         target("line", "order_date", "profit", groupby="month(order_date)", agg="sum", sort="date_asc",
                reason="Monthly profit over time is the most direct health signal.",
                insight="The chart shows whether profit has been growing, flat or declining.")),
        ("retail", "I have five minutes, show me the one chart that matters.",
         target("line", "order_date", "sales", groupby="month(order_date)", agg="sum", sort="date_asc",
                reason="The sales timeline is the single most informative default view.",
                insight="The chart compresses the whole sales history into one view.")),
        ("retail", "Where are we losing money?",
         target("bar", "sub_category", "profit", groupby="sub_category", agg="sum", sort="value_desc",
                reason="A sorted profit ranking pushes the loss-makers to the visible bottom.",
                insight="The chart shows which sub-categories drag profit down.")),
        ("retail", "Which shipping option do people actually pick?",
         target("bar", "ship_mode", "order_id", groupby="ship_mode", agg="count",
                reason="Counting orders per mode answers which option is used most.",
                insight="The chart shows how order volume splits across shipping modes.")),
        ("retail", "Are big discounts a good idea for us?",
         target("box", "discount", "profit",
                reason="Discount takes a few discrete levels; boxes show the profit spread per level.",
                insight="The chart shows how profit behaves as the discount level grows.")),
        ("mall", "Should we treat younger and older shoppers differently?",
         target("bar", "age", "spending_score", groupby="bins(age)", agg="mean",
                reason="Age bands make the spending comparison across ages concrete.",
                insight="The chart shows whether spending habits shift with age.")),
        ("mall", "Draw me a picture of our typical shopper's wallet.",
         target("histogram", "annual_income_k_usd", None,
                reason="The income histogram is the wallet profile of the customer base.",
                insight="The chart shows which income ranges most customers fall into.")),
        ("energy", "Is the house wasteful at night?",
         target("bar", "date", "appliances", groupby="hour_of_day(date)", agg="mean",
                reason="Hourly averages expose consumption in the night hours.",
                insight="The chart shows how nighttime hours compare to the rest of the day.")),
        ("energy", "Did anything strange happen this spring?",
         target("line", "date", "appliances", groupby="day(date)", agg="sum", sort="date_asc",
                reason="A daily line makes strange days visible at a glance.",
                insight="The chart makes unusual consumption days stand out on the timeline.")),
        ("energy", "Does the kitchen heat up with the cooking hours?",
         target("bar", "date", "t1", groupby="hour_of_day(date)", agg="mean",
                reason="Hourly kitchen temperature averages track the cooking rhythm.",
                insight="The chart shows which hours the kitchen runs warmest.")),
        ("retail", "Rank the states, best to worst, for me.",
         target("bar", "state", "sales", groupby="state", agg="sum", sort="value_desc",
                reason="With 49 states, sorted bars keep the ranking readable.",
                insight="The chart ranks every state by its total sales.")),
        ("mall", "Do the rich actually spend more here?",
         target("scatter", "annual_income_k_usd", "spending_score",
                reason="A scatter answers whether income and spending move together.",
                insight="The chart shows whether higher income maps to higher spending scores.")),
    ]
    # a second phrasing for a subset, to teach phrasing robustness
    H += [
        ("retail", "What jumps out at you here?", H[0][2]),
        ("mall", "Tell me something I do not know about the shoppers.", H[5][2]),
        ("energy", "Any weird days in the usage data?", H[7][2]),
    ]
    # intent per handwritten example, in list order (vague questions still map onto the taxonomy, the agents need it even when the user did not say it)
    intents = ["trend", "comparison", "comparison", "trend", "distribution",
               "relationship", "comparison", "anomaly", "comparison", "trend",
               "comparison", "distribution", "trend", "trend", "comparison",
               "comparison", "relationship", "comparison", "distribution",
               "comparison", "anomaly", "comparison", "comparison", "relationship",
               "trend", "relationship", "anomaly"]
    assert len(intents) == len(H), f"{len(intents)} intents for {len(H)} examples"
    return [{"dataset": d, "question": q, "target": t, "source": "handwritten", "intent": intent}
            for (d, q, t), intent in zip(H, intents)]
#################################
# Assembly:
def main() -> int:
    examples = build_template_examples() + build_handwritten_examples()
    examples += [dict(e, source="failure_targeted") for e in FAILURE_EXAMPLES]
    # deduplicate identical questions
    seen, unique = set(), []
    for e in examples:
        key = (e["dataset"], e["question"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(e)
    examples = unique
    # schema texts once per dataset (same summaries the model sees at inference)
    schemas = {}
    for ds, cfg in DATASETS.items():
        df = load_table(cfg["path"])
        schemas[ds] = schema_summary(profile_table(df, ds))
    # validate every target through the real pydantic schema
    for e in examples:
        ChartRecommendation(**e["target"]) # raises on any schema drift
    # contamination check before writing
    hits = check_contamination([e["question"] for e in examples])
    if hits:
        print(f"ABORT: {len(hits)} overlap(s) with the benchmark — fix these first:")
        for h in hits:
            print(f"  [{h['kind']} {h['score']}] {h['benchmark_id']}  <->  {h['sft_question'][:70]}")
        return 1
    random.shuffle(examples)
    out = Path("data/sft_train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for e in examples:
            rec = {"messages": [
                       {"role": "system", "content": SFT_SYSTEM},
                       {"role": "user", "content": f"{schemas[e['dataset']]}\n\nQuestion: {e['question']}"},
                       {"role": "assistant", "content": json.dumps(e["target"])}],
                   "meta": {"source": e["source"], "dataset": e["dataset"], "intent": e["intent"]}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    counts = {}
    for e in examples:
        counts[e["source"]] = counts.get(e["source"], 0) + 1
    total = len(examples)
    print(f"wrote {total} examples -> {out}")
    for src, n in sorted(counts.items()):
        print(f"  {src:16} {n:4}  ({100*n/total:.0f}%)")
    print("contamination check: clean")
    return 0
if __name__ == "__main__":
    sys.exit(main())
#################################