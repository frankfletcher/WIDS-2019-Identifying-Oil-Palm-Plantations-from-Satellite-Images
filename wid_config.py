from typing import Tuple
from IPython.display import Markdown, display
import logging

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.WARNING,
)

from dotenv import load_dotenv
import os

load_dotenv()

import pandas as pd
import numpy as np

np.random.seed(42)
rng = np.random.default_rng(42)

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# class Singleton:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(Singleton, cls).__new__(cls)
#         return cls._instance

#     def __init__(self):
#         if not hasattr(self, "initialized"):  # Prevent reinitialization
#             self.initialized = True


class WIDConfig:
    """Configuration class for managing dataset paths and loading data.

    This class initializes paths for training and testing datasets, loads the data,
    and provides methods for data manipulation and display.
    """

    def __init__(self, verbose=False, model_name=None):
        """Initializes WIDConfig with default paths and loads the data.

        Sets up the data paths and loads the training and holdout datasets.
        """
        super().__init__()
        self.data_path = Path("./data")
        self.data_dir = self.data_path / "train_images"
        self.train_csv = self.data_path / "traininglabels.csv"
        self.test_csv = self.data_path / "testlabels.csv"
        self.holdout_csv = self.data_path / "holdout.csv"
        self.submission_csv = self.data_path / "sample_submission.csv"
        self.downsample_majority_class = False  # 1000
        self.random_state = 42
        self.score_threshold = 1.0
        self.num_workers = 4
        self.num_classes = 1
        self.df, self.df_holdout = self.load_data(verbose)
        self.model_name = model_name

        # read and store the experiment ID in a file
        self.EXP_ID = int(os.getenv("EXP_ID"))
        with open(".env", "w") as f:
            f.write(f"EXP_ID={self.EXP_ID + 1}")

    def load_data(
        self,
        verbose=False,
        train_csv=None,
        test_csv=None,
        holdout_csv=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads training and testing data from CSV files.

        This method reads the training and testing labels, concatenates them,
        and performs data cleaning and oversampling.

        Args:
            verbose (bool): If True, displays data distributions during loading.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training
            DataFrame and the holdout DataFrame.
        """
        if not train_csv:
            train_csv = self.train_csv
        if not test_csv:
            test_csv = self.test_csv
        if not holdout_csv:
            holdout_csv = self.data_path / "holdout.csv"

        print(train_csv, test_csv, holdout_csv)
        # load "test" data that we'll concat with train
        df_test = pd.read_csv(test_csv)

        # add the path to a column
        df_test["base_path"] = self.data_path / "leaderboard_test_data"

        # fix the file names to remove the years
        df_test["image_id"] = df_test.image_id.str[:9].astype(str) + ".jpg"

        # dropping the years creates duplicates which we must remove
        df_test = df_test.drop_duplicates(subset=["image_id"], keep=False)

        # drop where score is less than the threshold
        df_test = df_test.drop(df_test[df_test.score < self.score_threshold].index)

        if verbose:
            self._display_loaded_dataframe(
                "### Test Data", df_test, "### Test Data Distribution"
            )
        # load "train" data
        df = pd.read_csv(train_csv)
        df["image_id"] = df.image_id.str[:9].astype(str) + ".jpg"
        df = df.drop_duplicates(subset=["image_id"], keep=False)
        df["base_path"] = self.data_path / "train_images"

        # drop where score is less than the threshold
        df = df.drop(df[df.score < self.score_threshold].index)

        df = pd.concat([df, df_test], ignore_index=True)

        # the data files contain a lot of missing images so we drop them
        df = self.drop_missing_images(df)

        if verbose:
            self._display_loaded_dataframe(
                "### Train Data", df, "### Train Data Distribution"
            )

        # oversample the minority class to balance the target
        df = self.random_oversample(df, "has_oilpalm")

        df_holdout = pd.read_csv(holdout_csv)
        df_holdout["base_path"] = self.data_path / "leaderboard_holdout_data"
        df_holdout = self.drop_missing_images(df_holdout)

        if verbose:
            self._display_loaded_dataframe(
                "### Train Data after Oversampling",
                df,
                "### Train Data Distribution after Oversampling",
            )
            self._display_loaded_dataframe(
                "### Holdout Data", df_holdout, "### Holdout Data Distribution"
            )

        return df, df_holdout

    def random_oversample(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Oversamples the minority class in the dataset.

        This method increases the representation of the minority class by
        duplicating samples to balance the dataset.

        Assumes the target column is binary (0 or 1) and the minority class is 1.

        Args:
            df (pd.DataFrame): The DataFrame to be oversampled.
            target_column (str): The name of the target column.

        Returns:
            pd.DataFrame: The oversampled DataFrame.
        """
        # oversample the minority class
        df_target_true = df[df[target_column] == 1]
        df_target_false = df[df[target_column] == 0]

        if self.downsample_majority_class:
            df_target_false = df_target_false.sample(
                n=self.downsample_majority_class, random_state=self.random_state
            )

        if len(df_target_true) >= len(df_target_false):
            logging.warning(
                f"No oversampling needed . False: {len(df_target_false)} True: {len(df_target_true)}"
            )
            df = pd.concat([df_target_true, df_target_false], ignore_index=True)
            df = df.sample(frac=1, random_state=self.random_state).reset_index(
                drop=True
            )
            return df

        df_target_true = df_target_true.sample(
            n=len(df_target_false), random_state=self.random_state, replace=True
        )
        df = pd.concat([df_target_false, df_target_true], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        return df

    def _display_loaded_dataframe(self, heading_df_name, df, heading_data_distribution):
        """Displays the loaded DataFrame and its distribution.

        This method shows the first few rows of the DataFrame and visualizes the
        distribution of the target variable.

        Args:
            heading_df_name (str): The title for the DataFrame display.
            df (pd.DataFrame): The DataFrame to display.
            heading_data_distribution (str): The title for the data distribution
            visualization.
        """
        display(Markdown(heading_df_name))
        display(df.head())
        display(Markdown(heading_data_distribution))
        sns.countplot(df, x="has_oilpalm")
        plt.show()

    def drop_missing_images(self, df, verbose=False):
        """Removes entries from the DataFrame where the image files do not exist.

        This method checks for the existence of image files and drops any rows corresponding to missing images.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame with missing images removed.
        """
        n_removed = 0
        for index, row in df.iterrows():
            if not os.path.exists(row.base_path / row.image_id):
                logging.debug(f"dropping: {row.image_id}")
                df.drop(index, inplace=True)
                n_removed += 1

        logging.warning(f"dropped {n_removed} missing images")

        return df

    def display(self):
        """Displays the configuration settings of the WIDConfig instance.

        This method presents the current configuration settings in a readable format.
        """
        display(Markdown("### Configuration"))
        display(Markdown(f"**Data Path**: {self.data_path}"))
        display(Markdown(f"**Data Directory**: {self.data_dir}"))
        display(Markdown(f"**Train CSV**: {self.train_csv}"))
        display(Markdown(f"**Test CSV**: {self.test_csv}"))
        display(Markdown(f"**Submission CSV**: {self.submission_csv}"))

    def __str__(self):
        """Returns a string representation of the WIDConfig instance.

        This method provides a summary of the configuration settings in a formatted string.

        Returns:
            str: A string representation of the WIDConfig instance.
        """
        return (
            f"WIDConfig(data_path={self.data_path}, "
            + "data_dir={self.data_dir}, "
            + "train_csv={self.train_csv}, "
            + "test_csv={self.test_csv}, "
            + "submission_csv={self.submission_csv}, "
            + "device={self.device}, "
            + "batch_size={self.batch_size}, "
            + "num_workers={self.num_workers}, "
            + "num_classes={self.num_classes}, "
            + "num_epochs={self.num_epochs}, "
            + "lr={self.lr}, model={self.model}, "
            + "criterion={self.criterion}, "
            + "optimizer={self.optimizer}, "
            + "transforms={self.transforms}, "
            + "train_transforms={self.train_transforms})"
        )
