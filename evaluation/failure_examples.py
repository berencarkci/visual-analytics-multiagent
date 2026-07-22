"""Failure targeted SFT examples.

Every entry here is the correct answer to a question where the untrained model was observed to fail (smoke test, live Space testing). 

Entry format:
    {"dataset": <key>, "question": <text>, "target": {chart_type, x_axis,
     y_axis, transform{groupby, agg, filter, sort, limit}, reason, insight}}

Rules to respect when adding:
- x_axis is always the source column (never the derived label).
- insight must be pointer style (describes what the chart shows), never a number the model could not compute from the schema alone.
- Do not copy benchmark questions verbatim, rephrase the failing question if it came from the benchmark.
"""

FAILURE_EXAMPLES = [
    # share of one category must not filter to that category
    {"dataset": "retail",
     "question": "What is Kentucky's share of total sales among all states?",
     "target": {"chart_type": "bar", "x_axis": "state", "y_axis": "sales",
                "transform": {"groupby": "state", "agg": "sum", "filter": None,
                              "sort": "value_desc", "limit": None},
                "reason": "A share needs all groups; with 49 states a bar chart stays readable where a pie would not.",
                "insight": "The chart shows each state's contribution to total sales, so Kentucky's share can be read against the rest."}},

    # composition of a metric uses sum, not count
    {"dataset": "retail",
     "question": "What is the profit composition by category?",
     "target": {"chart_type": "pie", "x_axis": "category", "y_axis": "profit",
                "transform": {"groupby": "category", "agg": "sum", "filter": None,
                              "sort": None, "limit": None},
                "reason": "A pie chart shows each category's share of a whole; three categories keep it readable.",
                "insight": "The chart shows how total profit splits across the three product categories."}},

    # relationship on a discrete numeric x -> box, not scatter
    {"dataset": "retail",
     "question": "How does profit relate to the discount level of an order line?",
     "target": {"chart_type": "box", "x_axis": "discount", "y_axis": "profit",
                "transform": {"groupby": None, "agg": None, "filter": None,
                              "sort": None, "limit": None},
                "reason": "Discount takes only a few discrete values, so grouped boxes show the profit spread per level better than an overplotted scatter.",
                "insight": "The chart shows how the profit distribution shifts across discount levels."}},

    # days of the week means day_of_week, not every calendar day
    {"dataset": "energy",
     "question": "Does average appliance use differ by weekday name?",
     "target": {"chart_type": "bar", "x_axis": "date", "y_axis": "appliances",
                "transform": {"groupby": "day_of_week(date)", "agg": "mean",
                              "filter": None, "sort": None, "limit": None},
                "reason": "Seven weekday groups compare naturally as bars.",
                "insight": "The chart shows which weekdays tend to have higher average appliance consumption."}},

    # categorical vs numeric 'correlation' -> grouped box, not Pearson
    {"dataset": "retail",
     "question": "Is there a connection between ship mode and order quantity?",
     "target": {"chart_type": "box", "x_axis": "ship_mode", "y_axis": "quantity",
                "transform": {"groupby": None, "agg": None, "filter": None,
                              "sort": None, "limit": None},
                "reason": "Ship mode is categorical, so a correlation coefficient does not apply; grouped boxes compare the quantity distributions instead.",
                "insight": "The chart shows whether order quantities differ across shipping modes."}},

    # histogram bins raw values itself, no groupby
    {"dataset": "mall",
     "question": "Show how annual income is distributed across our customers.",
     "target": {"chart_type": "histogram", "x_axis": "annual_income_k_usd", "y_axis": None,
                "transform": {"groupby": None, "agg": None, "filter": None,
                              "sort": None, "limit": None},
                "reason": "A histogram bins the raw numeric values itself; no aggregation is needed.",
                "insight": "The chart shows how customer incomes are spread across their range."}},

    # trend question: x-axis must be the time column, not the measured value
    {"dataset": "energy",
     "question": "What is the trend of appliance consumption?",
     "target": {"chart_type": "line", "x_axis": "date", "y_axis": "appliances",
                "transform": {"groupby": "month(date)", "agg": "sum", "filter": None,
                              "sort": "date_asc", "limit": None},
                "reason": "A trend is read over time, so the date column is the x-axis and the measured value is aggregated on y.",
                "insight": "The chart shows how appliance consumption develops over time."}},
]