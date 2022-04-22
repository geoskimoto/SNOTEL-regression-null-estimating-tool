#!./.venv/bin/python
# -*- coding: utf-8 -*-

from dash import html


tidbits = [
    html.Summary("Useful Tidbits"),
    html.P(
        [
            html.Ul(
                [
                    html.Li(
                        "Tool can be used to estimate missing or bad data for "
                        "any sensor on any SNOTEL or SNOLITE station.",
                        className="my-1",
                    ),
                    html.Li(
                        "Use the NRCS IMAP "
                        "(https://www.nrcs.usda.gov/wps/portal/wcc/home/quicklinks/imap) "
                        "to find suitable predictor stations based on proximity, "
                        "elevation, aspect, etc. for the response station of interest.",
                        className="my-1",
                    ),
                    html.Li(
                        "When evaluating models, pay attention to the Root Mean "
                        "Square Error (RMSE) between the training and test data sets.",
                        className="my-1",
                    ),
                    html.Ul(
                        [
                            html.Li(
                                "As a general rule you want to minimize the error "
                                "of both and they should be similar in value."
                            ),
                            html.Li("RMSE test > RMSE train => OVERFITTING"),
                            html.Li("RMSE test < RMSE train => UNDERFITTING"),
                            html.Li(
                                "Better to have an underfitting than an overfitting model."
                            ),
                        ],
                        className="my-1",
                    ),
                    html.Li(
                        "When choosing a regression model type to make real world "
                        "estimates, stick to linear models (linear, ridge, lasso, "
                        "and huber).  These models are easy to explain and defend.  "
                        "Additionally, the more advanced models have major drawbacks "
                        "(primarily overfitting) that require either tinkering with "
                        "the length of the training dataset or fine tuning parameters "
                        "that are not available in this app.",
                        className="my-1",
                    ),
                    html.Li(
                        "Author: Nicholas Steele, Snow Survey Hydrologist, "
                        "USDA - NRCS, nick.steele@usda.gov",
                        className="my-1",
                    ),
                ]
            )
        ]
    ),
]


if __name__ == "__main__":
    print("I do nothing, just non dynamic components that take up space...")
