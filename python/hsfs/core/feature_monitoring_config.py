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

import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import humps
from hsfs import util
from hsfs.client.exceptions import FeatureStoreException
from hsfs.core import (
    feature_monitoring_config_engine,
    feature_monitoring_result_engine,
    job_api,
    monitoring_window_config_engine,
)
from hsfs.core import monitoring_window_config as mwc
from hsfs.core.feature_monitoring_result import FeatureMonitoringResult
from hsfs.core.feature_statistics_config import FeatureStatisticsConfig
from hsfs.core.job_schedule import JobSchedule
from hsfs.core.statistics_comparison_config import StatisticsComparisonConfig


MAX_LENGTH_NAME = 63
MAX_LENGTH_DESCRIPTION = 2000


class FeatureMonitoringType(str, Enum):
    SCHEDULED_STATISTICS = "SCHEDULED_STATISTICS"  # stats computation on a schedule
    STATISTICS_COMPARISON = "STATISTICS_COMPARISON"  # stats computation and comparison
    PROBABILITY_DENSITY_FUNCTION = "PROBABILITY_DENSITY_FUNCTION"  # data distributions

    @classmethod
    def list_str(cls) -> List[str]:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list(cls) -> List["FeatureMonitoringType"]:
        return list(map(lambda c: c, cls))

    @classmethod
    def from_str(cls, value: str) -> "FeatureMonitoringType":
        if value in cls.list_str():
            return cls(value)
        else:
            raise ValueError(
                f"Invalid value {value} for FeatureMonitoringType, allowed values are {cls.list_str()}"
            )

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, self.__class__):
            return self is other
        return False

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


class FeatureMonitoringConfig:
    def __init__(
        self,
        feature_store_id: int,
        feature_monitoring_type: Union[FeatureMonitoringType, str],
        name: str,
        feature_statistics_configs: Union[
            List[FeatureStatisticsConfig], List[Dict[str, Any]]
        ],
        job_name: Optional[str] = None,
        detection_window_config: Optional[
            Union[mwc.MonitoringWindowConfig, Dict[str, Any]]
        ] = None,
        reference_window_config: Optional[
            Union[mwc.MonitoringWindowConfig, Dict[str, Any]]
        ] = None,
        job_schedule: Optional[Union[Dict[str, Any], JobSchedule]] = None,
        description: Optional[str] = None,
        id: Optional[int] = None,
        feature_group_id: Optional[int] = None,
        feature_view_name: Optional[str] = None,
        feature_view_version: Optional[int] = None,
        valid_feature_names: Optional[List[str]] = None,  # for validation purposes only
        href: Optional[str] = None,
        **kwargs,
    ):
        self.name = name
        self._id = id
        self._href = href
        self.description = description
        self._feature_store_id = feature_store_id
        self._feature_group_id = feature_group_id
        self._feature_view_name = feature_view_name
        self._feature_view_version = feature_view_version
        self._job_name = job_name
        self._feature_monitoring_type = (
            feature_monitoring_type
            if isinstance(feature_monitoring_type, FeatureMonitoringType)
            else FeatureMonitoringType(feature_monitoring_type)
        )

        self._feature_monitoring_config_engine = (
            feature_monitoring_config_engine.FeatureMonitoringConfigEngine(
                feature_store_id=feature_store_id,
                feature_group_id=feature_group_id,
                feature_view_name=feature_view_name,
                feature_view_version=feature_view_version,
            )
        )
        self._monitoring_window_config_engine = (
            monitoring_window_config_engine.MonitoringWindowConfigEngine()
        )
        self._feature_monitoring_result_engine = (
            feature_monitoring_result_engine.FeatureMonitoringResultEngine(
                feature_store_id=feature_store_id,
                feature_group_id=feature_group_id,
                feature_view_name=feature_view_name,
                feature_view_version=feature_view_version,
            )
        )
        self._job_api = job_api.JobApi()

        self.detection_window_config = detection_window_config
        self.reference_window_config = reference_window_config
        self.feature_statistics_configs = self._parse_feature_statistics_configs(
            feature_statistics_configs
        )
        self.job_schedule = job_schedule
        self._valid_feature_names = valid_feature_names

    def _parse_feature_statistics_configs(
        self,
        feature_statistics_configs: Union[
            List[FeatureStatisticsConfig], List[Dict[str, Any]]
        ],
    ) -> List[FeatureStatisticsConfig]:
        fs_configs = []
        for fs_config in feature_statistics_configs:
            fs_configs.append(
                fs_config
                if isinstance(fs_config, FeatureStatisticsConfig)
                else FeatureStatisticsConfig.from_response_json(fs_config)
            )
        return fs_configs

    @classmethod
    def from_response_json(cls, json_dict):
        json_decamelized = humps.decamelize(json_dict)
        if "count" in json_decamelized:
            if json_decamelized["count"] == 0:
                return []
            return [cls(**config) for config in json_decamelized["items"]]
        else:
            return cls(**json_decamelized)

    def to_dict(self):
        the_dict = {
            "id": self._id,
            "featureStoreId": self._feature_store_id,
            "name": self._name,
            "description": self._description,
            "jobName": self._job_name,
            "featureMonitoringType": self._feature_monitoring_type,
            "detectionWindowConfig": self._detection_window_config.to_dict(),
            "featureStatisticsConfigs": [
                fsc.to_dict() for fsc in self._feature_statistics_configs
            ],
        }

        # job schedule
        if isinstance(self._job_schedule, JobSchedule):
            the_dict["jobSchedule"] = self._job_schedule.to_dict()

        # entity id
        if self._feature_group_id is not None:
            the_dict["featureGroupId"] = self._feature_group_id
        elif (
            self._feature_view_name is not None
            and self._feature_view_version is not None
        ):
            the_dict["featureViewName"] = self._feature_view_name
            the_dict["featureViewVersion"] = self._feature_view_version

        if self._feature_monitoring_type == FeatureMonitoringType.SCHEDULED_STATISTICS:
            return the_dict

        reference_window_config = (
            self._reference_window_config.to_dict()
            if self._reference_window_config is not None
            else None
        )
        if reference_window_config is not None:
            the_dict["referenceWindowConfig"] = reference_window_config

        return the_dict

    def json(self) -> str:
        return json.dumps(self, cls=util.FeatureStoreEncoder)

    def __str__(self):
        return self.json()

    def __repr__(self):
        return f"FeatureMonitoringConfig({self._name!r}, {self._feature_monitoring_type!r})"

    def with_detection_window(
        self,
        time_offset: Optional[str] = None,
        window_length: Optional[str] = None,
        row_percentage: Optional[float] = None,
    ) -> "FeatureMonitoringConfig":
        """Sets the detection window of data to compute statistics on.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Compute statistics on a regular basis
            fg.create_scheduled_statistics(
                name="regular_stats",
                cron_expression="0 0 12 ? * * *",
            ).with_detection_window(
                time_offset="1d",
                window_length="1d",
                row_percentage=0.1,
            ).save()
            # Compute and compare statistics
            fg.create_feature_monitoring(
                name="regular_stats",
                feature_name="my_feature",
                cron_expression="0 0 12 ? * * *",
            ).with_detection_window(
                time_offset="1d",
                window_length="1d",
                row_percentage=0.1,
            ).with_reference_window(...).compare_on(...).save()
            ```
        # Arguments
            time_offset: The time offset from the current time to the start of the time window.
            window_length: The length of the time window.
            row_percentage: The fraction of rows to use when computing the statistics [0, 1.0].
        # Returns
            `FeatureMonitoringConfig`. The updated FeatureMonitoringConfig object.
        """
        # Setter is using the engine class to perform input validation.
        self.detection_window_config = {
            "window_config_type": (
                mwc.WindowConfigType.ROLLING_TIME
                if time_offset or window_length
                else mwc.WindowConfigType.ALL_TIME
            ),
            "time_offset": time_offset,
            "window_length": window_length,
            "row_percentage": row_percentage,
        }

        return self

    def with_reference_window(
        self,
        time_offset: Optional[str] = None,
        window_length: Optional[str] = None,
        row_percentage: Optional[float] = None,
    ) -> "FeatureMonitoringConfig":
        """Sets the reference window of data to compute statistics on.
        See also `with_reference_value(...)` and `with_reference_training_dataset(...)` for other reference options.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Setup feature monitoring and a detection window
            my_monitoring_config = fg.create_feature_monitoring(...).with_detection_window(...)
            # Statistics computed on a rolling time window, e.g. same day last week
            my_monitoring_config.with_reference_window(
                time_offset="1w",
                window_length="1d",
            ).compare_on(...).save()
            ```

        !!! warning "Provide a comparison configuration"
            You must provide a comparison configuration via `compare_on()` before saving the feature monitoring config.

        # Arguments
            time_offset: The time offset from the current time to the start of the time window.
            window_length: The length of the time window.
            row_percentage: The percentage of rows to use when computing the statistics. Defaults to 20%.
        # Returns
            `FeatureMonitoringConfig`. The updated FeatureMonitoringConfig object.
        """
        if self.detection_window_config is None:
            self.with_detection_window()  # use default detection window

        # Setter is using the engine class to perform input validation and build monitoring window config object.
        self.reference_window_config = {
            "time_offset": time_offset,
            "window_length": window_length,
            "row_percentage": row_percentage,
        }

        return self

    def with_reference_training_dataset(
        self,
        training_dataset_version: Optional[int] = None,
    ) -> "FeatureMonitoringConfig":
        """Sets the reference training dataset to compare statistics with.
        See also `with_reference_value(...)` and `with_reference_window(...)` for other reference options.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Setup feature monitoring and a detection window
            my_monitoring_config = fg.create_feature_monitoring(...).with_detection_window(...)
            # Only for feature views: Compare to the statistics computed for one of your training datasets
            # particularly useful if it has been used to train a model currently in production
            my_monitoring_config.with_reference_training_dataset(
                training_dataset_version=3,
            ).compare_on(...).save()
            ```

        !!! warning "Provide a comparison configuration"
            You must provide a comparison configuration via `compare_on()` before saving the feature monitoring config.

        # Arguments
            training_dataset_version: The version of the training dataset to use as reference.
        # Returns
            `FeatureMonitoringConfig`. The updated FeatureMonitoringConfig object.
        """
        if self.detection_window_config is None:
            self.with_detection_window()  # use default detection window

        self.reference_window_config = {
            "training_dataset_version": training_dataset_version,
        }

        return self

    def compare_on(
        self,
        feature_name: str,
        metric: str,
        threshold: Optional[float],
        strict: Optional[bool] = False,
        relative: Optional[bool] = False,
        specific_value: Optional[Union[int, float]] = None,
    ) -> "FeatureMonitoringConfig":
        """Sets the statistics comparison criteria for feature monitoring with a reference.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Setup feature monitoring, a detection window and a reference window
            my_monitoring_config = fg.create_feature_monitoring(
                ...
            ).with_detection_window(...).with_reference_window(...)
            # Choose a metric and set a threshold for the difference
            # e.g compare the relative mean of detection and reference window
            my_monitoring_config.compare_on(
                feature_name="my_feature",
                metric="mean",
                threshold=1.0,
                relative=True,
            ).save()
            ```

        !!! note
            Detection window and reference window/value/training_dataset must be set prior to comparison configuration.

        # Arguments
            feature_name: Name of the feature to monitor.
            metric: The metric to use for comparison. Different metric are available for different feature type.
            threshold: The threshold to apply to the difference to potentially trigger an alert.
            strict: Whether to use a strict comparison (e.g. > or <) or a non-strict comparison (e.g. >= or <=).
            relative: Whether to use a relative comparison (e.g. relative mean) or an absolute comparison (e.g. absolute mean).
            specific_value: A float value to use as reference.
        # Returns
            `FeatureMonitoringConfig`. The updated FeatureMonitoringConfig object.
        """
        if self.detection_window_config is None:
            self.with_detection_window()  # use default detection window

        if self.reference_window_config is None and specific_value is None:
            raise ValueError("Reference window is required for statistics comparisons.")

        self._feature_monitoring_config_engine.validate_statistics_metric(metric)

        sc_config = StatisticsComparisonConfig(
            metric=metric,
            threshold=threshold,
            relative=relative,
            strict=strict,
            specific_value=specific_value,
        )

        fs_config = self._with_feature_statistics_config(feature_name=feature_name)
        self._with_statistics_comparison_config(
            feature_statistics_config=fs_config,
            statistics_comparison_config=sc_config,
        )

        return self

    def save(self):
        """Saves the feature monitoring configuration.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Setup feature monitoring and a detection window
            my_monitoring_config = fg.create_scheduled_statistics(
                name="my_monitoring_config",
            ).save()
            ```
        # Returns
            `FeatureMonitoringConfig`. The saved FeatureMonitoringConfig object.
        """
        registered_config = self._feature_monitoring_config_engine.save(self)
        self.detection_window_config = registered_config._detection_window_config
        self.job_schedule = registered_config._job_schedule
        self.feature_statistics_configs = registered_config._feature_statistics_configs
        self._job_name = registered_config._job_name
        self._id = registered_config._id

        if self._feature_monitoring_type != FeatureMonitoringType.SCHEDULED_STATISTICS:
            self.reference_window_config = registered_config._reference_window_config

        return self

    def update(self):
        """Updates allowed fields of the saved feature monitoring configuration.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Fetch registered config by name
            my_monitoring_config = fg.get_feature_monitoring_configs(name="my_monitoring_config")
            # Update the percentage of rows to use when computing the statistics
            my_monitoring_config.detection_window.row_percentage = 10
            my_monitoring_config.update()
            ```
        # Returns
            `FeatureMonitoringConfig`. The updated FeatureMonitoringConfig object.
        """
        return self._feature_monitoring_config_engine.update(self)

    def run_once(self):
        """Trigger the feature monitoring job once which computes and compares statistics on the detection and reference windows.
        !!! example
            ```python3
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Fetch registered config by name
            my_monitoring_config = fg.get_feature_monitoring_configs(name="my_monitoring_config")
            # Trigger the feature monitoring job once
            my_monitoring_config.run_job()
            ```

        !!! info
            The feature monitoring job will be triggered asynchronously and the method will return immediately.
            Calling this method does not affect the ongoing schedule.

        # Raises
            `FeatureStoreException`: If the feature monitoring config has not been saved.
        # Returns
            `Job`. A handle for the job computing the statistics.
        """
        if not self._id or not self._job_name:
            raise FeatureStoreException(
                "Feature monitoring config must be registered via `.save()` before computing statistics."
            )

        return self._feature_monitoring_config_engine.trigger_monitoring_job(
            job_name=self.job_name
        )

    def get_job(self):
        """Get the feature monitoring job which computes and compares statistics on the detection and reference windows.
        !!! example
            ```python3
            # Fetch registered config by name via feature group or feature view
            my_monitoring_config = fg.get_feature_monitoring_configs(name="my_monitoring_config")
            # Get the job which computes statistics on detection and reference window
            job = my_monitoring_config.get_job()
            # Print job history and ongoing executions
            job.executions
            ```
        # Raises
            `FeatureStoreException`: If the feature monitoring config has not been saved.
        # Returns
            `Job`. A handle for the job computing the statistics.
        """
        if not self._id or not self._job_name:
            raise FeatureStoreException(
                "Feature monitoring config must be registered via `.save()` before fetching"
                "the associated job."
            )

        return self._feature_monitoring_config_engine.get_monitoring_job(
            job_name=self.job_name
        )

    def delete(self):
        """Deletes the feature monitoring configuration.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Fetch registered config by name
            my_monitoring_config = fg.get_feature_monitoring_configs(name="my_monitoring_config")
            # Delete the feature monitoring config
            my_monitoring_config.delete()
            ```
        # Raises
            `FeatureStoreException`: If the feature monitoring config has not been saved.
        """
        if not self._id:
            raise FeatureStoreException(
                "Feature monitoring config must be registered via `.save()` before deleting."
            )

        self._feature_monitoring_config_engine.delete(config_id=self._id)

    def disable(self):
        """Disables the schedule of the feature monitoring job.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Fetch registered config by name
            my_monitoring_config = fg.get_feature_monitoring_configs(name="my_monitoring_config")
            # Disable the feature monitoring config
            my_monitoring_config.disable()
            ```
        # Raises
            `FeatureStoreException`: If the feature monitoring config has not been saved.
        """
        self._update_schedule(enabled=False)

    def enable(self):
        """Enables the schedule of the feature monitoring job.
        The scheduler can be configured via the `job_schedule` property.
        !!! example
            ```python
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Fetch registered config by name
            my_monitoring_config = fg.get_feature_monitoring_configs(name="my_monitoring_config")
            # Enable the feature monitoring config
            my_monitoring_config.enable()
            ```
        # Raises
            `FeatureStoreException`: If the feature monitoring config has not been saved.
        """
        self._update_schedule(enabled=True)

    def get_history(
        self,
        start_time: Union[datetime, date, str, int, None] = None,
        end_time: Union[datetime, date, str, int, None] = None,
        with_statistics: bool = True,
    ) -> List["FeatureMonitoringResult"]:
        """
        Fetch the history of the computed statistics and comparison results for this configuration.
        !!! example
            ```python3
            # Fetch your feature group or feature view
            fg = fs.get_feature_group(name="my_feature_group", version=1)
            # Fetch registered config by name
            my_monitoring_config = fg.get_feature_monitoring_configs(name="my_monitoring_config")
            # Fetch the history of the computed statistics for this configuration
            history = my_monitoring_config.get_history(
                start_time="2021-01-01",
                end_time="2021-01-31",
            )
            ```
        # Args:
            start_time: The start time of the time range to fetch the history for.
            end_time: The end time of the time range to fetch the history for.
            with_statistics: Whether to include the computed statistics in the results.
        # Raises
            `FeatureStoreException`: If the feature monitoring config has not been saved.
        """
        if not self._id:
            raise FeatureStoreException(
                "Feature monitoring config must be registered via `.save()` before fetching"
                "the associated history."
            )
        return self._feature_monitoring_result_engine.fetch_all_feature_monitoring_results_by_config_id(
            config_id=self._id,
            start_time=start_time,
            end_time=end_time,
            with_statistics=with_statistics,
        )

    def get_feature_names(self) -> List[str]:
        return [
            fs_config.feature_name for fs_config in self._feature_statistics_configs
        ]

    def _with_feature_statistics_config(
        self, feature_name: str
    ) -> FeatureStatisticsConfig:
        if self.feature_statistics_configs is not None:
            for fsc in self.feature_statistics_configs:
                if fsc.feature_name == feature_name:
                    return fsc

        fs_config = FeatureStatisticsConfig(
            feature_name=feature_name,
            statistics_comparison_configs=[],
        )
        if self.feature_statistics_configs is None:
            self.feature_statistics_configs = [fs_config]
        else:
            self.feature_statistics_configs.append(fs_config)

        return fs_config

    def _with_statistics_comparison_config(
        self,
        feature_statistics_config: FeatureStatisticsConfig,
        statistics_comparison_config: StatisticsComparisonConfig,
    ) -> StatisticsComparisonConfig:
        if feature_statistics_config.statistics_comparison_configs is None:
            raise ValueError("Statistics comparison configurations cannot be updated")

        for sc_config in feature_statistics_config.statistics_comparison_configs:
            if sc_config == statistics_comparison_config:
                raise ValueError(
                    "A similar statistics comparison configuration already exists."
                )

        feature_statistics_config.statistics_comparison_configs.append(
            statistics_comparison_config
        )
        return statistics_comparison_config

    def _update_schedule(self, enabled):
        if self._job_schedule is None:
            raise FeatureStoreException("No scheduler found for monitoring job")
        job_schedule = JobSchedule(
            id=self._job_schedule.id,
            start_date_time=self._job_schedule.start_date_time,
            cron_expression=self._job_schedule.cron_expression,
            end_time=self._job_schedule.end_date_time,
            enabled=enabled,
        )
        self.job_schedule = self._job_api.create_or_update_schedule_job(
            self._job_name, job_schedule.to_dict()
        )
        return self._job_schedule

    @property
    def id(self) -> Optional[int]:
        """Id of the feature monitoring configuration."""
        return self._id

    @property
    def feature_store_id(self) -> int:
        """Id of the Feature Store."""
        return self._feature_store_id

    @property
    def feature_group_id(self) -> Optional[int]:
        """Id of the Feature Group to which this feature monitoring configuration is attached."""
        return self._feature_group_id

    @property
    def feature_view_name(self) -> Optional[str]:
        """Name of the Feature View to which this feature monitoring configuration is attached."""
        return self._feature_view_name

    @property
    def feature_view_version(self) -> Optional[int]:
        """Version of the Feature View to which this feature monitoring configuration is attached."""
        return self._feature_view_version

    @property
    def name(self) -> str:
        """The name of the feature monitoring config.
        A Feature Group or Feature View cannot have multiple feature monitoring configurations with the same name. The name of
        a feature monitoring configuration is limited to 63 characters.

        !!! info "This property is read-only once the feature monitoring configuration has been saved."
        """
        return self._name

    @name.setter
    def name(self, name: str):
        if hasattr(self, "_id"):
            raise AttributeError("The name of a registered config is read-only.")
        elif not isinstance(name, str):
            raise TypeError("name must be of type str")
        if len(name) > MAX_LENGTH_NAME:
            raise ValueError("name must be less than {MAX_LENGTH_NAME} characters.")
        self._name = name

    @property
    def description(self) -> Optional[str]:
        """Description of the feature monitoring configuration."""
        return self._description

    @description.setter
    def description(self, description: Optional[str]):
        if not isinstance(description, str) and description is not None:
            raise TypeError("description must be of type str")
        elif isinstance(description, str) and len(description) > MAX_LENGTH_DESCRIPTION:
            raise ValueError(
                "description must be less than {MAX_LENGTH_DESCRIPTION} characters"
            )
        self._description = description

    @property
    def job_name(self) -> Optional[str]:
        """Name of the feature monitoring job."""
        return self._job_name

    @property
    def enabled(self) -> bool:
        """Controls whether or not this config is spawning new feature monitoring jobs.
        This field belongs to the scheduler configuration but is made transparent to the user for convenience.
        """
        return self.job_schedule.enabled

    @enabled.setter
    def enabled(self, enabled: bool):
        """
        Controls whether or not this config is spawning new feature monitoring jobs.
        This field belongs to the scheduler configuration but is made transparent to the user for convenience.
        """
        self.job_schedule.enabled = enabled

    @property
    def feature_monitoring_type(self) -> FeatureMonitoringType:
        """The type of feature monitoring to perform. Used for internal validation.
        Options are:
            - SCHEDULED_STATISTICS if no reference window (and, therefore, comparison config) is provided
            - STATISTICS_COMPARISON if a reference window (and, therefore, comparison config) is provided.

        !!! info "This property is read-only."
        """
        return self._feature_monitoring_type

    @property
    def job_schedule(self) -> JobSchedule:
        """Schedule of the feature monitoring job.
        This field belongs to the job configuration but is made transparent to the user for convenience.
        """
        return self._job_schedule

    @job_schedule.setter
    def job_schedule(self, job_schedule: Union[JobSchedule, Dict[str, Any]]):
        if isinstance(job_schedule, JobSchedule):
            self._job_schedule = job_schedule
        elif isinstance(job_schedule, dict):
            self._job_schedule = JobSchedule(**job_schedule)
        else:
            raise TypeError("job_schedule must be of type JobScheduler, dict or None")

    @property
    def detection_window_config(self) -> mwc.MonitoringWindowConfig:
        """Configuration for the detection window."""
        return self._detection_window_config

    @detection_window_config.setter
    def detection_window_config(
        self,
        detection_window_config: Optional[
            Union[mwc.MonitoringWindowConfig, Dict[str, Any]]
        ],
    ):
        if isinstance(detection_window_config, mwc.MonitoringWindowConfig):
            self._detection_window_config = detection_window_config
        elif isinstance(detection_window_config, dict):
            self._detection_window_config = (
                self._monitoring_window_config_engine.build_monitoring_window_config(
                    **detection_window_config
                )
            )
        elif detection_window_config is None:
            self._detection_window_config = detection_window_config
        else:
            raise TypeError(
                "detection_window_config must be of type MonitoringWindowConfig, dict or None"
            )

    @property
    def reference_window_config(self) -> mwc.MonitoringWindowConfig:
        """Configuration for the reference window."""
        return self._reference_window_config

    @reference_window_config.setter
    def reference_window_config(
        self,
        reference_window_config: Optional[
            Union[mwc.MonitoringWindowConfig, Dict[str, Any]]
        ] = None,
    ):
        """Sets the reference window for monitoring."""
        if (
            self._feature_monitoring_type == FeatureMonitoringType.SCHEDULED_STATISTICS
            and reference_window_config is not None
        ):
            raise AttributeError(
                "reference_window_config is only available for feature monitoring"
                " not for scheduled statistics. Use `create_feature_monitoring()` instead."
            )
        if isinstance(reference_window_config, mwc.MonitoringWindowConfig):
            self._reference_window_config = reference_window_config
        elif isinstance(reference_window_config, dict):
            self._reference_window_config = (
                self._monitoring_window_config_engine.build_monitoring_window_config(
                    **reference_window_config
                )
            )
        elif reference_window_config is None:
            self._reference_window_config = None
        else:
            raise TypeError(
                "reference_window_config must be of type MonitoringWindowConfig, dict or None"
            )

    @property
    def feature_statistics_configs(self) -> List[FeatureStatisticsConfig]:
        """Configurations for the computation (and comparison) of feature statistics"""
        return self._feature_statistics_configs

    @feature_statistics_configs.setter
    def feature_statistics_configs(
        self, feature_statistics_configs: List[FeatureStatisticsConfig]
    ):
        self._feature_monitoring_config_engine.validate_feature_statistics_configs(
            feature_statistics_configs,
            self._feature_monitoring_type == FeatureMonitoringType.SCHEDULED_STATISTICS,
        )
        self._feature_statistics_configs = feature_statistics_configs
