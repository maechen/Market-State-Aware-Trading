"""
Walk-forward fold configuration for SPY GHMM labeling.

Defines 5y/1y/1y train/validation/test windows for multiple folds.
"""


FOLDS: list[dict[str, str]] = [
    {
        "name": "fold1",
        "train_start": "2007-01-01",
        "train_end": "2011-12-31",
        "val_start": "2012-01-01",
        "val_end": "2012-12-31",
        "test_start": "2013-01-01",
        "test_end": "2013-12-31",
    },
    {
        "name": "fold2",
        "train_start": "2008-01-01",
        "train_end": "2012-12-31",
        "val_start": "2013-01-01",
        "val_end": "2013-12-31",
        "test_start": "2014-01-01",
        "test_end": "2014-12-31",
    },
    {
        "name": "fold3",
        "train_start": "2009-01-01",
        "train_end": "2013-12-31",
        "val_start": "2014-01-01",
        "val_end": "2014-12-31",
        "test_start": "2015-01-01",
        "test_end": "2015-12-31",
    },
    {
        "name": "fold4",
        "train_start": "2010-01-01",
        "train_end": "2014-12-31",
        "val_start": "2015-01-01",
        "val_end": "2015-12-31",
        "test_start": "2016-01-01",
        "test_end": "2016-12-31",
    },
    {
        "name": "fold5",
        "train_start": "2011-01-01",
        "train_end": "2015-12-31",
        "val_start": "2016-01-01",
        "val_end": "2016-12-31",
        "test_start": "2017-01-01",
        "test_end": "2017-12-31",
    },
    {
        "name": "fold6",
        "train_start": "2012-01-01",
        "train_end": "2016-12-31",
        "val_start": "2017-01-01",
        "val_end": "2017-12-31",
        "test_start": "2018-01-01",
        "test_end": "2018-12-31",
    },
    {
        "name": "fold7",
        "train_start": "2013-01-01",
        "train_end": "2017-12-31",
        "val_start": "2018-01-01",
        "val_end": "2018-12-31",
        "test_start": "2019-01-01",
        "test_end": "2019-12-31",
    },
    {
        "name": "fold8",
        "train_start": "2014-01-01",
        "train_end": "2018-12-31",
        "val_start": "2019-01-01",
        "val_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2020-12-31",
    },
]

