from datetime import timedelta

import numpy as np
import pandas as pd

from base.logging_config import get_logger

logger = get_logger(__file__)


# noinspection PyArgumentList
def presplit_data(user_item_interaction_data: pd.DataFrame,
                  uid_column: str,
                  date_column: str,
                  num_min: int = 3,
                  test_size_days: int = 14,
                  sort=True,
                  ):
    """
    Split data into train and test set.

    Parameters
    ----------
    user_item_interaction_data :
        Dataframe of shape: user_id, item_id, interaction_date, is_converted(optional)
    uid_column:
        Unique identifier for the customers.
    date_column:
        Column name containing interaction time
    num_min:
        Minimal number of interactions (transactions or clicks) for a customer to be included in the dataset
        (interactions can be both in train and test sets)
    test_size_days:
        Number of days that should be in the test set. The rest will be in the training set.
    sort:
        Sort the dataset by date before splitting in train/test set,  thus having a test set that is succeeding
        the train set

    Returns
    -------
    train_set:
        Pandas dataframe of all training interactions.
    test_set:
        Pandas dataframe of all testing interactions.
    """

    np.random.seed(11)
    if num_min > 0:
        user_item_interaction_data = user_item_interaction_data[
            user_item_interaction_data[uid_column].map(
                user_item_interaction_data[uid_column].value_counts()
            ) >= num_min
        ]

    if sort:
        user_item_interaction_data.sort_values(by=[date_column],
                                               axis=0,
                                               inplace=True)
        # Split into train & test sets
        # most_recent_date = datetime.strptime(max(user_item_interaction_data[date_column]), '%Y-%m-%d')
        # limit_date = datetime.strftime(most_recent_date - timedelta(days=test_size_days), format='%Y-%m-%d')
        most_recent_date = max(user_item_interaction_data[date_column])
        limit_date = most_recent_date - timedelta(days=test_size_days)
        train_set = user_item_interaction_data[user_item_interaction_data[date_column] <= limit_date]
        test_set = user_item_interaction_data[user_item_interaction_data[date_column] > limit_date]

    else:
        # most_recent_date = datetime.strptime(max(user_item_interaction_data[date_column]), '%Y-%m-%d')
        # oldest_date = datetime.strptime(min(user_item_interaction_data[date_column]), '%Y-%m-%d')
        most_recent_date = max(user_item_interaction_data[date_column])
        oldest_date = min(user_item_interaction_data[date_column])
        total_days = timedelta(days=(most_recent_date - oldest_date).days)  # To be tested
        test_size = test_size_days / total_days.days
        test_set = user_item_interaction_data.sample(frac=test_size, random_state=200)
        train_set = user_item_interaction_data.drop(test_set.index)

    # Keep only users in train set
    ctm_list = train_set[uid_column].unique()
    test_set = test_set[test_set[uid_column].isin(ctm_list)]
    return train_set, test_set

