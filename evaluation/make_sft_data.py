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

    bank["intent"] = "filter_aggregation"
    # Lure phrasings (dev-failure driven). 
    # Every filter question above marks the subset explicitly ("only", "within", "restricted to", "top N"). 
    # The dev split showed the model misses filters named as a plain noun modifier: the surface verb (compare / evolve / track / how much) pulls it to comparison or trend while "of the X category" / "for Y customers" silently defines a filter. 
    # These examples teach the structural cue, not the keyword.
    for cat_val, cat_col, noun in [("Furniture", "category", "category"), ("Office Supplies", "category", "category"), ("Home Office", "segment", "segment"), ("East", "region", "region")]:
        for metric in ["profit", "quantity"]:
            t = target("line", "order_date", metric, groupby="month(order_date)",
                       agg="sum", sort="date_asc", filter=f"{cat_col} == '{cat_val}'",
                       reason=f"The {noun} named in the question is a filter; the monthly line then tracks that slice alone.",
                       insight=f"The chart shows the monthly course of total {pretty(metric)} for {cat_val}.")
            for q in [f"Compare the monthly total {pretty(metric)} of the {cat_val} {noun}.",
                      f"How did {pretty(metric)} for the {cat_val} {noun} evolve month by month?",
                      f"Track the monthly {pretty(metric)} of the {cat_val} {noun} over time."]:
                add("retail", q, t)
                
    # contrast twins: the same lure verbs with no modifier stay comparison/trend.
    # Adjacent pairs teach the discriminating feature (the modifier), not the verb.
    bank["intent"] = "comparison"
    for metric in ["profit", "quantity"]:
        t = target("bar", "category", metric, groupby="category", agg="sum", sort="value_desc",
                   reason="No subset is named, so the categories are compared over the whole data.",
                   insight=f"The chart compares total {pretty(metric)} across all categories.")
        add("retail", f"Compare the total {pretty(metric)} across the categories.", t)
    
    bank["intent"] = "trend"
    for metric in ["profit", "quantity"]:
        t = target("line", "order_date", metric, groupby="month(order_date)", agg="sum",
                   sort="date_asc",
                   reason="No subset is named, so the whole data is tracked over time.",
                   insight=f"The chart shows how overall {pretty(metric)} evolved month by month.")
        add("retail", f"How did overall {pretty(metric)} evolve month by month?", t)
    
    bank["intent"] = "filter_aggregation"
    # numeric condition modifiers ("older than", "younger than") on mall
    for cond, phrase in [("age > 40", "older than 40"), ("age < 30", "younger than 30")]:
        for metric in ["spending_score", "annual_income_k_usd"]:
            t = target("bar", "gender", metric, groupby="gender", agg="mean", filter=cond,
                       reason="The age condition is a filter; averages are then compared inside that subset.",
                       insight=f"The chart compares average {pretty(metric)} by gender among customers {phrase}.")
            for q in [f"Split by gender, what do customers {phrase} average on {pretty(metric)}?",
                      f"Among customers {phrase}, compare the average {pretty(metric)} by gender."]:
                add("mall", q, t)
    
    # implicit year restriction ("during YEAR" with no "only")
    for year in (2016, 2017):
        t = target("bar", "category", "profit", groupby="category", agg="sum",
                   filter=f"year(order_date) == {year}",
                   reason="Naming a year restricts the data to it even without the word only.",
                   insight=f"The chart shows what each category earned during {year}.")
        add("retail", f"What did each category earn during {year}?", t)
    
    bank["intent"] = "comparison"
    # derived measures: the question asks for a quantity that is not a column,
    # so y_axis carries the expression and the engine materialises it first
    for cat in ["category", "segment", "region", "ship_mode", "state"]:
        t = target("bar", cat, "days_between(order_date, ship_date)", groupby=cat,
                   agg="mean", sort="value_desc",
                   reason=f"Delivery time is the gap between the two date columns, averaged per {pretty(cat)}.",
                   insight=f"The chart shows which {pretty(cat)} groups wait longest between order and shipment.")
        for q in [f"Which {pretty(cat)} waits longest between order and shipment?",
                  f"Compare average delivery time across {pretty(cat)} groups.",
                  f"How does shipping delay differ by {pretty(cat)}?"]:
            add("retail", q, t)
    for cat in ["sub_category", "category", "region", "segment"]:
        t = target("bar", cat, "ratio(profit, sales)", groupby=cat, agg="mean",
                   sort="value_asc",
                   reason="Margin is profit over sales; ascending order puts the loss makers first.",
                   insight=f"The chart shows which {pretty(cat)} groups earn the thinnest margin on each sale.")
        for q in [f"Which {pretty(cat)} groups have the weakest profit margin?",
                  f"Rank {pretty(cat)} groups from worst to best profit margin.",
                  f"Where is the profit margin thinnest across {pretty(cat)} groups?"]:
            add("retail", q, t)
    for cat in ["sub_category", "category"]:
        t = target("bar", cat, "ratio(sales, quantity)", groupby=cat, agg="mean",
                   sort="value_desc",
                   reason="Revenue per unit is sales divided by quantity.",
                   insight=f"The chart shows which {pretty(cat)} groups bring in the most revenue per unit sold.")
        for q in [f"Which {pretty(cat)} groups earn the most per unit sold?",
                  f"Compare revenue per unit across {pretty(cat)} groups."]:
            add("retail", q, t)
    for cat in ["region", "category"]:
        t = target("bar", cat, "diff(sales, profit)", groupby=cat, agg="sum",
                   sort="value_desc",
                   reason="The gap between revenue and profit is the cost side of each group.",
                   insight=f"The chart shows how much of each {pretty(cat)} group's revenue does not reach profit.")
        add("retail", f"How large is the gap between sales and profit for each {pretty(cat)}?", t)
    
    # diff expansion: the probe showed diff was never learned, it had 2 examples, and the project's own finding is that a structural behaviour needs ~20+. Same target family, many phrasings and group columns.
    for cat in ["sub_category", "segment", "ship_mode", "region", "category"]:
        t = target("bar", cat, "diff(sales, profit)", groupby=cat, agg="sum", sort="value_desc",
                   reason="Sales minus profit is the cost side; diff materialises it per group.",
                   insight=f"The chart shows how much revenue each {pretty(cat)} group spends before reaching profit.")
        for q in [f"How much of each {pretty(cat)} group's sales never becomes profit?",
                  f"Show the gap between sales and profit for every {pretty(cat)} group.",
                  f"Which {pretty(cat)} groups have the widest spread between sales and profit?"]:
            add("retail", q, t)
    
    bank["intent"] = "trend"
    for period, expr in [("monthly", "month"), ("quarterly", "quarter")]:
        t = target("line", "order_date", "diff(sales, profit)", groupby=f"{expr}(order_date)",
                   agg="sum", sort="date_asc",
                   reason=f"A {period} line of the sales-profit gap shows whether costs are growing.",
                   insight=f"The chart shows how the {period} gap between sales and profit developed.")
        for q in [f"Track the {period} gap between sales and profit.",
                  f"How did the difference between sales and profit develop {period}?"]:
            add("retail", q, t)
    
    bank["intent"] = "comparison"
    t = target("bar", "date", "diff(appliances, lights)", groupby="day_of_week(date)", agg="mean",
               reason="The appliance-light gap per weekday is a derived difference measure.",
               insight="The chart shows on which weekdays appliances outdraw the lights the most.")
    for q in ["How big is the gap between appliance and light consumption per weekday?",
              "Compare the difference between appliance and light usage across weekdays."]:
        add("energy", q, t)

    bank["intent"] = "trend"
    for period, expr in [("monthly", "month"), ("quarterly", "quarter"), ("weekly", "week")]:
        t = target("line", "order_date", "days_between(order_date, ship_date)",
                   groupby=f"{expr}(order_date)", agg="mean", sort="date_asc",
                   reason=f"A {period} line shows whether delivery time is drifting.",
                   insight=f"The chart shows how average delivery time developed {period}.")
        for q in [f"Is delivery time getting better or worse {period}?",
                  f"Trace the {period} course of average delivery time.",
                  f"Has shipping speed changed over time, {period}?"]:
            add("retail", q, t)
    for period, expr in [("monthly", "month"), ("quarterly", "quarter")]:
        t = target("line", "order_date", "ratio(profit, sales)", groupby=f"{expr}(order_date)",
                   agg="mean", sort="date_asc",
                   reason=f"A {period} margin line shows whether profitability is eroding.",
                   insight=f"The chart shows how the {period} profit margin developed.")
        for q in [f"Is our profit margin eroding {period}?",
                  f"Trace the {period} profit margin over time."]:
            add("retail", q, t)

    bank["intent"] = "trend"
    # descending time order: "most recent first" phrasing
    for ds, metric in [("retail", "sales"), ("retail", "profit"), ("retail", "quantity"),
                       ("energy", "appliances"), ("energy", "lights")]:
        d = DATASETS[ds]
        for expr, unit in [("month", "months"), ("week", "weeks")]:
            t = target("bar", d["date"], metric, groupby=f"{expr}({d['date']})", agg="sum",
                       sort="date_desc", limit=6,
                       reason="The most recent periods come first when the question asks for the latest figures.",
                       insight=f"The chart shows the six most recent {unit} of total {pretty(metric)}.")
            for q in [f"Show the six most recent {unit} of total {pretty(metric)}, newest first.",
                      f"What do the latest {unit} look like for {pretty(metric)}?"]:
                add(ds, q, t)

    bank["intent"] = "filter_aggregation"
    # ascending value order: "lowest/worst" phrasing
    for cat, metric in [("sub_category", "profit"), ("state", "profit"), ("city", "sales"),
                        ("sub_category", "sales"), ("state", "quantity"), ("city", "profit")]:
        for n in (5, 10):
            t = target("bar", cat, metric, groupby=cat, agg="sum", sort="value_asc", limit=n,
                       reason=f"Ascending order surfaces the weakest {pretty(cat)} groups first.",
                       insight=f"The chart shows the {n} weakest {pretty(cat)} groups by total {pretty(metric)}.")
            for q in [f"Which {n} {pretty(cat)} groups perform worst on {pretty(metric)}?",
                      f"Show the bottom {n} {pretty(cat)} groups by total {pretty(metric)}."]:
                add("retail", q, t)
    
    # "losing money" semantics: the probe failure was the wrong sort
    # direction for loss questions, losing means the lowest totals, ascending.
    for cat in ["sub_category", "state", "city"]:
        t = target("bar", cat, "profit", groupby=cat, agg="sum", sort="value_asc", limit=10,
                   reason="Losing money means the lowest profit totals, so the sort is ascending.",
                   insight=f"The chart surfaces the {pretty(cat)} groups that lose the most money.")
        for q in [f"Which {pretty(cat)} groups are losing us money?",
                  f"Show the {pretty(cat)} groups where we bleed the most profit."]:
            add("retail", q, t)
    
    bank["intent"] = "comparison"
    for cat in ["state", "ship_mode"]:
        t = target("bar", cat, "ratio(profit, sales)", groupby=cat, agg="mean", sort="value_asc",
                   reason="Margin is profit over sales; ascending order puts the thinnest margins first.",
                   insight=f"The chart shows which {pretty(cat)} groups convert sales into profit worst.")
        for q in [f"Which {pretty(cat)} groups convert their sales into profit the worst?",
                  f"Rank the {pretty(cat)} groups by profit margin, thinnest first."]:
            add("retail", q, t)

    bank["intent"] = "anomaly"
    # anomaly (a line over time, the groupby granularity follows the wording of the question, and bar is never an anomaly answer)
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
    
    # soft anomaly wording (dev-failure driven). 
    # The bank above uses hard markers (spiked, abnormally, out of line), the dev split showed soft wording ("strange periods", "does anything stand out in how X behaves") slides to trend. 
    # Also covers sensor columns (t1, rh_1), where the miss was observed on readings style phrasing.
    soft = [("energy", "appliances", "sum"), ("energy", "lights", "sum"), ("energy", "t1", "mean"), ("energy", "rh_1", "mean"), ("retail", "sales", "sum"), ("retail", "profit", "sum")]
    for ds, metric, agg in soft:
        d = DATASETS[ds]
        t = target("line", d["date"], metric, groupby=f"day({d['date']})", agg=agg,
                   sort="date_asc",
                   reason="Soft wording still asks for outliers; a daily line makes odd periods visible.",
                   insight=f"The chart exposes days where {pretty(metric)} breaks from its usual pattern.")
        for q in [f"Did the {pretty(metric)} readings go through any strange stretches?",
                  f"Does anything look odd in how {pretty(metric)} behaved over the days?",
                  f"Point out stretches where {pretty(metric)} broke from its normal rhythm."]:
            add(ds, q, t)
    return ex
#################################

# Handwritten vague/free form examples:
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

# Intent-only examples: consumed only by the Supervisor format in make_agent_sft_data.py. 
# These cover lure patterns whose full chart target is either awkward or would require filter mechanics the engine does not support (e.g. an hour-of-day filter) but the intent label is unambiguous, and the Supervisor is the agent that was failing on exactly these patterns.
# They never enter data/sft_train.jsonl and are contamination checked separately.
def build_intent_only_examples() -> list[dict]:
    Q = [
        # time-of-day subset named as a modifier -> filter_aggregation
        ("energy", "How much power do the lights draw during the night hours?", "filter_aggregation"),
        ("energy", "What is the average appliance load in the evening?", "filter_aggregation"),
        ("energy", "How much energy goes to the appliances in the morning hours?", "filter_aggregation"),
        ("energy", "Looking only at weekend days, how high does appliance consumption run?", "filter_aggregation"),
        # subset named as a noun phrase -> filter_aggregation
        ("retail", "How did sales develop for the home office crowd?", "filter_aggregation"),
        ("retail", "What did the western stores bring in per month?", "filter_aggregation"),
        ("mall", "How do the over-50 customers score on spending?", "filter_aggregation"),
        # soft / readings-style anomaly wording -> anomaly
        ("energy", "Do the temperature sensors show any suspicious stretches?", "anomaly"),
        ("energy", "Is there anything odd about how the humidity readings behave?", "anomaly"),
        ("energy", "Anything in the sensor data that should not be there?", "anomaly"),
        ("retail", "Did any stretch of orders look out of the ordinary?", "anomaly"),
        # contrast anchor: no subset, no anomaly cue -> plain trend
        ("retail", "How are things going overall, month by month?", "trend"),
    ]
    return [{"dataset": d, "question": q, "intent": i, "source": "intent_only"}
            for d, q, i in Q]
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