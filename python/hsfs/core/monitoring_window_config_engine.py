#
#   Copyright 2024 Hopsworks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, TypeVar, Union

from hsfs import feature_group, feature_view, util
from hsfs.client.exceptions import RestAPIError
from hsfs.core import monitoring_window_config as mwc
from hsfs.core import statistics_engine
from hsfs.core.feature_descriptive_statistics import FeatureDescriptiveStatistics
from hsfs.training_dataset_split import TrainingDatasetSplit


class MonitoringWindowConfigEngine:
    _MAX_TIME_RANGE_LENGTH = 12

    def __init__(self, **kwargs):
        # No need to initialize anything
        pass

    def _init_statistics_engine(self, feature_store_id: int, entity_type: str):
        self._statistics_engine = statistics_engine.StatisticsEngine(
            feature_store_id=feature_store_id,
            entity_type=entity_type,
        )

    def validate_monitoring_window_config(
        self,
        time_offset: Optional[str] = None,
        window_length: Optional[str] = None,
        training_dataset_version: Optional[int] = None,
        row_percentage: Optional[float] = None,
    ) -> "mwc.WindowConfigType":
        if isinstance(training_dataset_version, int):
            if any(
                [
                    time_offset is not None,
                    window_length is not None,
                    row_percentage is not None,
                ]
            ):
                raise ValueError(
                    "If training_dataset_version is set, no other window parameters can be set."
                )
            return mwc.WindowConfigType.TRAINING_DATASET

        if isinstance(time_offset, str):
            return mwc.WindowConfigType.ROLLING_TIME

        if isinstance(window_length, str):
            raise ValueError("window_length can only be set if time_offset is set.")

        return mwc.WindowConfigType.ALL_TIME

    def build_monitoring_window_config(
        self,
        id: Optional[int] = None,
        window_config_type: Optional[Union["mwc.WindowConfigType", str]] = None,
        time_offset: Optional[str] = None,
        window_length: Optional[str] = None,
        training_dataset_version: Optional[int] = None,
        specific_value: Optional[Union[int, float]] = None,
        row_percentage: Optional[float] = None,
    ) -> "mwc.MonitoringWindowConfig":
        """Builds a monitoring window config.

        Args:
            window_config_type: str, required
                Type of the window config, can be either
                `ROLLING_TIME`,`SPECIFIC_VALUE`,`TRAINING_DATASET`.
            time_offset: str, optional
                monitoring window start time is computed as "now - time_offset".
            window_length: str, optional
                monitoring window end time is computed as
                    "now - time_offset + window_length".
            training_dataset_version: int, optional
                Specific id of an entity that has fixed statistics.
            specific_value: float, optional
                Specific value instead of a statistics computed on data.
            row_percentage: float, optional
                Percentage of rows to be used for statistics computation.
            id: int, optional
                Id of the monitoring window config in hopsworks.

        Returns:
            MonitoringWindowConfig The monitoring window configuration.
        """

        detected_window_config_type = self.validate_monitoring_window_config(
            time_offset=time_offset,
            window_length=window_length,
            training_dataset_version=training_dataset_version,
            row_percentage=row_percentage,
        )

        if (
            isinstance(window_config_type, str)
            and window_config_type != detected_window_config_type
        ):
            raise ValueError(
                "The window_config_type parameter does not match the window parameters set."
            )

        if (
            window_config_type
            in [mwc.WindowConfigType.ROLLING_TIME, mwc.WindowConfigType.ALL_TIME]
            and row_percentage is None
        ):
            row_percentage = 1.0

        return mwc.MonitoringWindowConfig(
            id=id,
            window_config_type=detected_window_config_type,
            time_offset=time_offset,
            window_length=window_length,
            training_dataset_version=training_dataset_version,
            specific_value=specific_value,
            row_percentage=row_percentage,
        )

    def time_range_str_to_time_delta(
        self, time_range: str, field_name: Optional[str] = "time_offset"
    ) -> timedelta:
        # sanitize input
        value_error_message = f"Invalid {field_name} format: {time_range}. Use format: 1w2d3h for 1 week, 2 days and 3 hours."
        if (
            len(time_range) > self._MAX_TIME_RANGE_LENGTH
            or re.search(r"([^dwh\d]+)", time_range) is not None
        ):
            raise ValueError(value_error_message)

        matches = re.search(
            # r"^(?!$)(?:.*(?P<week>\d+)w)?(?:.*(?P<day>\d+)d)?(?:.*(?P<hour>\d+)h)?$",
            r"(?:(?P<week>\d+w)()|(?P<day>\d+d)()|(?P<hour>\d+h)())+",
            time_range,
        )
        if matches is None:
            raise ValueError(value_error_message)

        weeks = (
            int(matches.group("week").replace("w", ""))
            if matches.group("week") is not None
            else 0
        )
        days = (
            int(matches.group("day").replace("d", ""))
            if matches.group("day") is not None
            else 0
        )
        hours = (
            int(matches.group("hour").replace("h", ""))
            if matches.group("hour") is not None
            else 0
        )

        return timedelta(weeks=weeks, days=days, hours=hours)

    def get_window_start_end_times(
        self,
        monitoring_window_config: "mwc.MonitoringWindowConfig",
    ) -> Tuple[Optional[int], int]:
        end_time = datetime.now()
        if monitoring_window_config.window_config_type not in [
            mwc.WindowConfigType.ROLLING_TIME,
            mwc.WindowConfigType.ALL_TIME,
        ]:
            return (
                None,
                self.round_and_convert_event_time(event_time=end_time),
            )

        if monitoring_window_config.time_offset is not None:
            time_offset = self.time_range_str_to_time_delta(
                monitoring_window_config.time_offset
            )
            start_time = datetime.now() - time_offset
        else:
            # case where time_offset is None and window_length is None
            return (
                None,
                self.round_and_convert_event_time(event_time=end_time),
            )

        if monitoring_window_config.window_length is not None:
            window_length = self.time_range_str_to_time_delta(
                monitoring_window_config.window_length
            )
            end_time = (
                start_time + window_length
                if start_time + window_length < end_time
                else end_time
            )

        return (
            self.round_and_convert_event_time(event_time=start_time),
            self.round_and_convert_event_time(event_time=end_time),
        )

    def run_single_window_monitoring(
        self,
        entity: Union[feature_group.FeatureGroup, "feature_view.FeatureView"],
        monitoring_window_config: "mwc.MonitoringWindowConfig",
        feature_names: List[str],
    ) -> List[FeatureDescriptiveStatistics]:
        """Fetch the entity data based on monitoring window configuration and compute statistics.

        Args:
            entity: FeatureStore: Feature store to fetch the entity to monitor.
            monitoring_window_config: MonitoringWindowConfig: Monitoring window config.
            feature_name: str: Name of the feature to monitor.

        Returns:
            [FeatureDescriptiveStatistics, List[FeatureDescriptiveStatitics]]: List of Descriptive statistics.
        """
        self._init_statistics_engine(entity._feature_store_id, entity.ENTITY_TYPE)
        (
            start_time,
            end_time,
        ) = self.get_window_start_end_times(
            monitoring_window_config=monitoring_window_config,
        )

        registered_stats = None  # no stats by default value

        if (
            monitoring_window_config.window_config_type
            == mwc.WindowConfigType.TRAINING_DATASET
            and isinstance(entity, feature_view.FeatureView)
        ):
            td_meta = entity._feature_view_engine._get_training_dataset_metadata(
                entity,
                training_dataset_version=monitoring_window_config.training_dataset_version,
            )

            # split untransformed and transformed features
            after_transf_features, before_transf_features = [], []
            if td_meta.transformation_functions is None:
                after_transf_features = feature_names
            else:
                for feat_name in feature_names:
                    if feat_name in td_meta.transformation_functions.keys():
                        before_transf_features.append(feat_name)
                    else:
                        after_transf_features.append(feat_name)

            if after_transf_features:
                registered_stats = entity.get_training_dataset_statistics(
                    training_dataset_version=monitoring_window_config.training_dataset_version,
                    before_transformation=False,
                    feature_names=after_transf_features,
                )

                if (
                    registered_stats.feature_descriptive_statistics is None
                    and registered_stats.split_statistics is not None
                ):
                    # if td splits, we use the train set statistics
                    for split in registered_stats.split_statistics:
                        if split.name == TrainingDatasetSplit.TRAIN:
                            registered_stats = split
                assert (
                    registered_stats.feature_descriptive_statistics
                ), "Registered statistics must have feature descriptive statistics"

            if before_transf_features:
                before_transf_stats = entity.get_training_dataset_statistics(
                    training_dataset_version=monitoring_window_config.training_dataset_version,
                    before_transformation=True,
                    feature_names=before_transf_features,
                )
                assert before_transf_stats.feature_descriptive_statistics, "Registered statistics before transformations must have feature descriptive statistics"

                if registered_stats is None:
                    registered_stats = before_transf_stats
                else:
                    registered_stats.feature_descriptive_statistics.extend(
                        before_transf_stats.feature_descriptive_statistics
                    )
        else:
            # Check if statistics already exists
            registered_stats = self._statistics_engine.get_by_time_window(
                metadata_instance=entity,
                start_commit_time=start_time,
                end_commit_time=end_time,
                feature_names=feature_names,
                row_percentage=monitoring_window_config.row_percentage,
            )

        if registered_stats is None:  # if statistics don't exist
            # TODO: What happens if window is TRAINING_DATASET and the TD statistics were not computed???

            # Fetch the actual data for which to compute statistics based on row_percentage and time window
            entity_feature_df = self.fetch_entity_data_in_monitoring_window(
                entity=entity,
                feature_names=feature_names,
                start_time=start_time,
                end_time=end_time,
                row_percentage=monitoring_window_config.row_percentage,
            )

            # Compute statistics on the feature dataframe
            registered_stats = (
                self._statistics_engine.compute_and_save_monitoring_statistics(
                    entity,
                    feature_dataframe=entity_feature_df,
                    window_start_commit_time=start_time,
                    window_end_commit_time=end_time,
                    row_percentage=monitoring_window_config.row_percentage,
                    feature_name=feature_names,
                )
            )

        assert (
            registered_stats.feature_descriptive_statistics is not None
        ), "statistics should contain the feature descriptive statistics"

        return registered_stats.feature_descriptive_statistics

    def fetch_entity_data_in_monitoring_window(
        self,
        entity: Union["feature_group.FeatureGroup", "feature_view.FeatureView"],
        feature_names: List[str],
        start_time: Optional[int],
        end_time: Optional[int],
        row_percentage: float,
    ) -> TypeVar("pyspark.sql.DataFrame"):
        """Fetch the entity data based on time window and row percentage.

        Args:
            entity: Union[FeatureGroup, FeatureView]: Entity to monitor.
            feature_name: str: Name of the feature to monitor.
            start_time: int: Window start commit or event time
            end_time: int: Window end commit or event time
            row_percentage: fraction of rows to include [0, 1.0]

        Returns:
            `pyspark.sql.DataFrame`. A Spark DataFrame with the entity data
        """
        try:
            if isinstance(entity, feature_group.FeatureGroup):
                entity_df = self.fetch_feature_group_data(
                    entity=entity,
                    feature_names=feature_names,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                entity_df = self.fetch_feature_view_data(
                    entity=entity,
                    feature_names=feature_names,
                    start_time=start_time,
                    end_time=end_time,
                )

            if row_percentage < 1.0:
                entity_df = entity_df.sample(fraction=row_percentage)

        except RestAPIError as e:
            if (
                e.response.json().get("errorCode", "") == 270118
                and e.response.status_code == 404
            ):
                # creating an empty Spark DataFrame requires a valid schema [SchemaType(name, type)]
                # and takes unnecessary time and resources. We can return an empty pandas dataframe instead
                # because the computation of statistics will be discarded in statistics_engine (len(df.head()) == 0)
                import pandas as pd

                return pd.DataFrame(columns=feature_names)
            else:
                raise e

        return entity_df

    def fetch_feature_view_data(
        self,
        entity: "feature_view.FeatureView",
        feature_names: Optional[List[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> TypeVar("pyspark.sql.DataFrame"):
        """Fetch the feature view data based on time window and row percentage.

        Args:
            entity: FeatureView: Feature view to monitor.
            feature_name: str: Name of the feature to monitor.
            start_time: int: Window start commit or event time.
            end_time: int: Window end commit or event time.

        Returns:
            `pyspark.sql.DataFrame`. A Spark DataFrame with the entity data
        """

        # TODO: This fails for FV on non-time-travel FGs.
        entity_df = entity.query.as_of(
            exclude_until=start_time, wallclock_time=end_time
        ).read()

        if feature_names:
            entity_df = entity_df.select(feature_names)

        return entity_df

    def fetch_feature_group_data(
        self,
        entity: "feature_group.FeatureGroup",
        feature_names: Optional[List[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> TypeVar("pyspark.sql.Dataframe"):
        """Fetch the feature group data based on time window.

        Args:
            entity: FeatureGroup: Feature group to monitor.
            feature_name: str: Name of the feature to monitor.
            start_time: int: Window start commit time.
            end_time: int: Window end commit time.
        """
        if feature_names:
            pre_df = entity.select(features=feature_names)
        else:
            pre_df = entity

        full_df = pre_df.as_of(exclude_until=start_time, wallclock_time=end_time).read()

        return full_df

    def round_and_convert_event_time(self, event_time: datetime) -> Optional[int]:
        """Round event time to the latest hour and convert to timestamp.

        Args:
            event_time: datetime: Event time to round and convert.

        Returns:
            datetime: Rounded and converted event time.
        """
        return util.convert_event_time_to_timestamp(event_time)
