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

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

from hsfs import util
from hsfs.core import feature_monitoring_config as fmc
from hsfs.core.feature_descriptive_statistics import FeatureDescriptiveStatistics
from hsfs.core.feature_monitoring_config_api import FeatureMonitoringConfigApi
from hsfs.core.feature_monitoring_result import FeatureMonitoringResult
from hsfs.core.feature_monitoring_result_api import FeatureMonitoringResultApi
from hsfs.core.feature_statistics_config import FeatureStatisticsConfig
from hsfs.core.feature_statistics_result import FeatureStatisticsResult
from hsfs.core.job_api import JobApi
from hsfs.core.statistics_comparison_config import StatisticsComparisonConfig
from hsfs.core.statistics_comparison_result import StatisticsComparisonResult


class FeatureMonitoringResultEngine:
    """Logic and helper methods to deal with results from a feature monitoring job.

    Attributes:
        feature_store_id: int. Id of the respective Feature Store.
        feature_group_id: int. Id of the feature group, if monitoring a feature group.
        feature_view_name: str. Name of the feature view, if monitoring a feature view.
        feature_view_version: int. Version of the feature view, if monitoring a feature view.
    """

    def __init__(
        self,
        feature_store_id: int,
        feature_group_id: Optional[int] = None,
        feature_view_name: Optional[str] = None,
        feature_view_version: Optional[int] = None,
        **kwargs,
    ):
        if feature_group_id is None:
            assert feature_view_name is not None
            assert feature_view_version is not None

        self._feature_store_id = feature_store_id
        self._feature_group_id = feature_group_id
        self._feature_view_name = feature_view_name
        self._feature_view_version = feature_view_version

        self._feature_monitoring_result_api = FeatureMonitoringResultApi(
            feature_store_id=feature_store_id,
            feature_group_id=feature_group_id,
            feature_view_name=feature_view_name,
            feature_view_version=feature_view_version,
        )
        self._feature_monitoring_config_api = FeatureMonitoringConfigApi(
            feature_store_id=feature_store_id,
            feature_group_id=feature_group_id,
            feature_view_name=feature_view_name,
            feature_view_version=feature_view_version,
        )
        self._job_api = JobApi()

    # CRUD

    def save(
        self,
        result: FeatureMonitoringResult,
    ) -> FeatureMonitoringResult:
        """Save feature monitoring result.

        Args:
            config_id: int. Id of the feature monitoring configuration.
            execution_id: int. Id of the job execution.
            detection_statistics: FeatureDescriptiveStatistics. Statistics computed from the detection data.
            reference_statistics: Optional[Union[FeatureDescriptiveStatistics, int, float]]. Statistics computed from the reference data.
                Defaults to None if no reference is provided.
            shift_detected: bool. Whether a shift is detected between the detection and reference window.
                It is used to decide whether to trigger an alert.
            difference: Optional[float]. Difference between detection statistics and reference statistics.

        Returns:
            FeatureMonitoringResult. Saved Feature monitoring result.
        """
        return self._feature_monitoring_result_api.create(
            result,
        )

    def save_with_exception(
        self,
        feature_monitoring_config_id: int,
        job_name: str,
    ) -> "FeatureMonitoringResult":
        """Save feature monitoring result with raised_exception flag.

        Args:
            config_id: int. Id of the feature monitoring configuration.
            job_name: str. Name of the monitoring job.

        Returns:
            FeatureMonitoringResult. Saved Feature monitoring result.
        """
        return self.save(
            result=self._build_feature_monitoring_result(
                feature_monitoring_config_id=feature_monitoring_config_id,
                feature_statistics_results=None,
                job_name=job_name,
                raised_exception=True,
            ),
        )

    def get(
        self,
        config_id: Optional[int] = None,
        config_name: Optional[str] = None,
        start_time: Optional[Union[str, int, datetime, date]] = None,
        end_time: Optional[Union[str, int, datetime, date]] = None,
        with_statistics: bool = True,
    ) -> List[FeatureMonitoringResult]:
        """Convenience method to fetch feature monitoring results from an entity.

        Args:
            config_id: Optional[int]. Id of the feature monitoring configuration.
                Defaults to None if config_name is provided.
            config_name: Optional[str]. Name of the feature monitoring configuration.
                Defaults to None if config_id is provided.
            start_time: Optional[Union[str, int, datetime, date]].
                Query results with monitoring time greater than or equal to start_time.
            end_time: Optional[Union[str, int, datetime, date]].
                Query results with monitoring time less than or equal to end_time.
            with_statistics: bool. Whether to include the statistics attached to the results.
                Defaults to True. Set to False to fetch only monitoring metadata.

        Returns:
            List[FeatureMonitoringResult]. List of feature monitoring results.
        """
        if all([config_id is None, config_name is None]):
            raise ValueError(
                "Either config_id or config_name must be provided to fetch feature monitoring results."
            )
        elif all([config_id is not None, config_name is not None]):
            raise ValueError(
                "Only one of config_id or config_name can be provided to fetch feature monitoring results."
            )
        elif config_name is not None and isinstance(config_name, str):
            config = self._feature_monitoring_config_api.get_by_name(config_name)
            if not isinstance(config, fmc.FeatureMonitoringConfig):
                raise ValueError(
                    f"Feature monitoring configuration with name {config_name} does not exist."
                )
            config_id = config._id
        elif config_name is not None:
            raise TypeError(
                f"config_name must be of type str. Got {type(config_name)}."
            )
        elif config_id is not None and not isinstance(config_id, int):
            raise TypeError(f"config_id must be of type int. Got {type(config_id)}.")

        return self.get_by_config_id(
            config_id=config_id,
            start_time=start_time,
            end_time=end_time,
            with_statistics=with_statistics,
        )

    def get_by_config_id(
        self,
        config_id: int,
        start_time: Union[str, int, datetime, date, None] = None,
        end_time: Union[str, int, datetime, date, None] = None,
        with_statistics: bool = False,
    ) -> List[FeatureMonitoringResult]:
        """Fetch all feature monitoring results by config id.

        Args:
            config_id: int. Id of the feature monitoring configuration.
            start_time: Union[str, int, datetime, date, None].
                Query results with monitoring time greater than or equal to start_time.
            end_time: Union[str, int, datetime, date, None].
                Query results with monitoring time less than or equal to end_time.
            with_statistics: bool.
                Whether to include the statistics attached to the results or not

        Returns:
            List[FeatureMonitoringResult]. List of feature monitoring results.
        """

        query_params = self._build_query_params(
            start_time=start_time,
            end_time=end_time,
            with_statistics=with_statistics,
        )

        return self._feature_monitoring_result_api.get_by_config_id(
            config_id=config_id,
            query_params=query_params,
        )

    # operations

    def run_and_save_statistics_comparison(
        self,
        fm_config: "fmc.FeatureMonitoringConfig",
        detection_statistics: List[FeatureDescriptiveStatistics],
        reference_statistics: Optional[List[FeatureDescriptiveStatistics]] = None,
    ) -> FeatureMonitoringResult:
        """Run and upload statistics comparison between detection and reference stats.

        Args:
            fm_config: FeatureMonitoringConfig. Feature monitoring configuration.
            detection_statistics: List[FeatureDescriptiveStatistics]. Computed statistics from detection data.
            reference_statistics: Optional[List[FeatureDescriptiveStatistics]]]. Computed statistics from reference data.

        Returns:
            Union[FeatureMonitoringResult, List[FeatureMonitoringResult]]. Feature monitoring result
        """

        # validate fds
        self._validate_detection_and_reference_statistics(
            detection_statistics=detection_statistics,
            reference_statistics=reference_statistics,
        )

        # create fds dicts
        detection_stats_dict, empty_detection_window = {}, False
        reference_stats_dict, empty_reference_window = None, None
        for det_fds in detection_statistics:
            detection_stats_dict[det_fds.feature_name] = det_fds
            if self._is_monitoring_window_empty(det_fds):
                empty_detection_window = True
        if reference_statistics is not None:
            reference_stats_dict, empty_reference_window = {}, False
            for ref_fds in reference_statistics:
                reference_stats_dict[ref_fds.feature_name] = ref_fds
                if self._is_monitoring_window_empty(ref_fds):
                    empty_reference_window = True

        # build fs results
        fs_results = []
        for fs_config in fm_config.feature_statistics_configs:
            fs_result = self._run_feature_statistics_comparisons(
                fs_config,
                detection_stats_dict[fs_config.feature_name],
                (
                    reference_stats_dict[fs_config.feature_name]
                    if reference_stats_dict is not None
                    else None
                ),
            )
            fs_results.append(fs_result)

        # build fm result
        assert fm_config.id is not None
        fm_result = self._build_feature_monitoring_result(
            feature_monitoring_config_id=fm_config.id,
            feature_statistics_results=fs_results,
            empty_detection_window=empty_detection_window,
            empty_reference_window=empty_reference_window,
        )

        # save and return
        return self.save(fm_result)

    def _validate_detection_and_reference_statistics(
        self,
        detection_statistics: List[FeatureDescriptiveStatistics],
        reference_statistics: Optional[List[FeatureDescriptiveStatistics]],
    ):
        if reference_statistics is None:
            return

        mismatch_msg = "Detection feature statistics must contain a reference feature statistics for the same feature and feature type."
        assert len(reference_statistics) >= len(detection_statistics), mismatch_msg

        det_stats_feat_names, ref_stats_feat_names = set(), set()
        det_stats_set, ref_stats_set = set(), set()
        for det_fds, ref_fds in zip(detection_statistics, reference_statistics):
            det_stats_feat_names.add(det_fds.feature_name)
            ref_stats_feat_names.add(ref_fds.feature_name)
            det_stats_set.add((det_fds.feature_name, det_fds.feature_type))
            ref_stats_set.add((ref_fds.feature_name, ref_fds.feature_type))

        assert det_stats_feat_names.issubset(ref_stats_feat_names), mismatch_msg

        # check if detection and reference statistics contain the same features and feature types.
        # statistics computed on empty data will have feature type None, which can falsify the equality.
        # we shouldn't raise an exception in that case, so we just ignore it by checking the count statistic.

        assert (
            det_stats_set.issubset(ref_stats_set)
            or detection_statistics[0].count == 0
            or reference_statistics[0].count == 0
        ), mismatch_msg

    def _run_feature_statistics_comparisons(
        self,
        fs_config: FeatureStatisticsConfig,
        detection_statistics: FeatureDescriptiveStatistics,
        reference_statistics: Optional[FeatureDescriptiveStatistics],
    ) -> FeatureStatisticsResult:
        sc_results = None
        if fs_config.statistics_comparison_configs is not None:
            sc_results = []
            for sc_config in fs_config.statistics_comparison_configs:
                difference, shift_detected = self._compute_difference_and_shift(
                    sc_config=sc_config,
                    detection_statistics=detection_statistics,
                    reference_statistics=reference_statistics,
                )
                assert sc_config.id is not None
                if difference is not None:
                    # if difference can be computed, add result
                    scr = StatisticsComparisonResult(
                        sc_config.id,
                        shift_detected=shift_detected,
                        difference=difference,
                    )
                    scr._metric = sc_config.metric  # set temporary metric field
                    sc_results.append(scr)

        return self._build_feature_statistics_result(
            feature_name=fs_config.feature_name,
            detection_statistics=detection_statistics,
            reference_statistics=reference_statistics,
            statistics_comparison_results=sc_results if sc_results else None,
        )

    def _compute_difference_and_shift(
        self,
        sc_config: StatisticsComparisonConfig,
        detection_statistics: FeatureDescriptiveStatistics,
        reference_statistics: Optional[FeatureDescriptiveStatistics] = None,
    ) -> Tuple[Optional[float], bool]:
        """Compute the difference and detect shift between the reference and detection statistics.

        Args:
            fm_config: FeatureMonitoringConfig. Feature monitoring configuration.
            detection_statistics: FeatureDescriptiveStatistics. Computed statistics from detection data.
            reference_statistics: Union[FeatureDescriptiveStatistics, int, float].
                Computed statistics from reference data, or a specific value to use as reference.

        Returns:
            `(float, bool)`. The difference between the reference and detection statistics,
                             and whether shift was detected or not
        """
        difference = self._compute_difference_between_stats(
            detection_statistics=detection_statistics,
            reference_statistics=reference_statistics,
            metric=sc_config.metric,
            relative=sc_config.relative,
            specific_value=sc_config.specific_value,
        )
        if difference is None or sc_config.threshold is None:
            # if no difference can be computed or threshold no provided, no shift detected
            return difference, False

        if sc_config.strict:
            shift_detected = True if difference >= sc_config.threshold else False
        else:
            shift_detected = True if difference > sc_config.threshold else False
        return difference, shift_detected

    def _compute_difference_between_stats(
        self,
        detection_statistics: FeatureDescriptiveStatistics,
        metric: str,
        relative: bool = False,
        reference_statistics: Optional[FeatureDescriptiveStatistics] = None,
        specific_value: Optional[Union[int, float]] = None,
    ) -> Optional[float]:
        """Compute the difference between the reference and detection statistics.

        Args:
            detection_statistics: `FeatureDescriptiveStatistics`. Computed statistics from detection data.
            reference_statistics: `Optional[FeatureDescriptiveStatistics]`.
                Computed statistics from reference data, or a specific value to use as reference
            metric: `str`. The metric to compute the difference for.
            relative: `bool`. Whether to compute the relative difference or not.
            specific_value: `Optional[Union[int, float]]`. A specific value to use as reference.

        Returns:
            `Optional[float]`. The difference between the reference and detection statistics, or None if there
                               are no values to compare
        """
        if reference_statistics is None and specific_value is None:
            return None  # no difference can be computed

        metric_lower = metric.lower()  # ensure lower case

        if metric_lower != "count":
            # on empty data, only the count metric can be used to compute the difference
            if self._is_monitoring_window_empty(detection_statistics):
                # if det stats on empty data, no difference can be computed
                return None
            if specific_value is None and self._is_monitoring_window_empty(
                reference_statistics
            ):
                # if ref stats on empty data, no difference can be computed unless a specific value is provided
                return None

        # otherwise, both detection and reference value can be obtained
        detection_value = detection_statistics.get_value(metric_lower)
        reference_value = (
            specific_value
            if specific_value is not None
            else reference_statistics.get_value(metric_lower)
        )
        return self._compute_difference_between_specific_values(
            detection_value, reference_value, relative
        )

    def _compute_difference_between_specific_values(
        self,
        detection_value: Union[int, float],
        reference_value: Union[int, float],
        relative: bool = False,
    ) -> float:
        """Compute the difference between a reference and detection value.

        Args:
            detection_value: Union[int, float]. The detection value.
            reference_value: Union[int, float]. The reference value
            relative: `bool`. Whether to compute the relative difference or not.

        Returns:
            `float`. The difference between the reference and detection values.
        """

        diff = abs(detection_value - reference_value)
        if relative:
            if reference_value == 0:
                return float("inf")
            else:
                return diff / reference_value
        else:
            return diff

    # builders

    def _build_feature_monitoring_result(
        self,
        feature_monitoring_config_id: int,
        feature_statistics_results: Optional[List[FeatureStatisticsResult]],
        empty_detection_window: bool = False,
        empty_reference_window: Optional[bool] = False,
        raised_exception: bool = False,
        execution_id: Optional[int] = None,
        job_name: Optional[str] = None,
    ) -> FeatureMonitoringResult:
        """Build feature monitoring result.

        Args:
            config_id: int. Id of the feature monitoring configuration.
            execution_id: int. Id of the job execution.
            detection_statistics: FeatureDescriptiveStatistics. Statistics computed from the detection data.
            reference_statistics: Optional[Union[FeatureDescriptiveStatistics, int, float]]. Statistics computed from the reference data.
                Defaults to None if no reference is provided.
            shift_detected: bool. Whether a shift is detected between the detection and reference window.
                It is used to decide whether to trigger an alert.
            difference: Optional[float]. Difference between detection statistics and reference statistics.

        Returns:
            FeatureMonitoringResult. Saved Feature monitoring result.
        """
        monitoring_time = round(
            util.convert_event_time_to_timestamp(datetime.now()), -3
        )
        if execution_id is None and job_name is not None:
            execution_id = self._get_monitoring_job_execution_id(job_name)
        else:
            execution_id = 0

        # get shifted features
        shifted_feature_names = set()
        if feature_statistics_results is not None:
            for fsr in feature_statistics_results:
                if fsr.shifted_metric_names:
                    shifted_feature_names.add(fsr.feature_name)

        return FeatureMonitoringResult(
            feature_store_id=self._feature_store_id,
            feature_monitoring_config_id=feature_monitoring_config_id,
            execution_id=execution_id,
            feature_statistics_results=feature_statistics_results,
            monitoring_time=monitoring_time,
            raised_exception=raised_exception,
            shifted_feature_names=shifted_feature_names,
            empty_detection_window=empty_detection_window,
            empty_reference_window=empty_reference_window,
        )

    def _build_feature_statistics_result(
        self,
        feature_name: str,
        detection_statistics: FeatureDescriptiveStatistics,
        reference_statistics: Optional[FeatureDescriptiveStatistics],
        statistics_comparison_results: Optional[
            List[StatisticsComparisonResult]
        ] = None,
    ) -> FeatureStatisticsResult:
        detection_statistics_id = detection_statistics.id
        reference_statistics_id = (
            reference_statistics.id
            if isinstance(reference_statistics, FeatureDescriptiveStatistics)
            else None
        )

        # get shifted metrics
        shifted_metric_names = set()
        if statistics_comparison_results is not None:
            for scr in statistics_comparison_results:
                if scr.shift_detected:
                    # _metric here is just a temporary field not included in the scr DTO
                    shifted_metric_names.add(scr._metric)

        return FeatureStatisticsResult(
            feature_name=feature_name,
            statistics_comparison_results=statistics_comparison_results,
            detection_statistics_id=detection_statistics_id,
            reference_statistics_id=reference_statistics_id,
            shifted_metric_names=shifted_metric_names,
        )

    def _build_query_params(
        self,
        start_time: Union[str, int, datetime, date, None],
        end_time: Union[str, int, datetime, date, None],
        with_statistics: bool,
    ) -> Dict[str, Union[str, List[str]]]:
        """Build query parameters for feature monitoring result API calls.

        Args:
            start_time: Union[str, int, datetime, date, None].
                Query results with monitoring time greater than or equal to start_time.
            end_time: Union[str, int, datetime, date, None].
                Query results with monitoring time less than or equal to end_time.
            with_statistics: bool.
                Whether to include the statistics attached to the results or not

        Returns:
            Dict[str, Union[str, List[str]]]. Query parameters.
        """

        query_params = {"sort_by": "monitoring_time:desc"}

        filter_by = []
        if start_time:
            timestamp_start_time = util.convert_event_time_to_timestamp(start_time)
            filter_by.append(f"monitoring_time_gte:{timestamp_start_time}")
        if end_time:
            timestamp_end_time = util.convert_event_time_to_timestamp(end_time)
            filter_by.append(f"monitoring_time_lte:{timestamp_end_time}")
        if len(filter_by) > 0:
            query_params["filter_by"] = filter_by

        if with_statistics:
            query_params["expand"] = "statistics"

        return query_params

    def _get_monitoring_job_execution_id(
        self,
        job_name: str,
    ) -> int:
        """Get the execution id of the last execution of the monitoring job.

        The last execution is assumed to be the current execution.
        The id defaults to 0 if no execution is found.

        Args:
            job_name: str. Name of the monitoring job.

        Returns:
            int. Id of the last execution of the monitoring job.
                It is assumed to be the current execution.
        """
        execution = self._job_api.last_execution(self._job_api.get(name=job_name))
        return (
            execution[0]._id
            if isinstance(execution, list) and len(execution) > 0
            else 0
        )

    def _is_monitoring_window_empty(
        self,
        monitoring_window_statistics: Optional[FeatureDescriptiveStatistics] = None,
    ) -> Optional[bool]:
        """Check if the monitoring window is empty.

        Args:
            monitoring_window_statistics: FeatureDescriptiveStatistics. Statistics computed for the monitoring window.

        Returns:
            bool. Whether the monitoring window is empty or not.
        """
        if monitoring_window_statistics is None:
            return None
        elif monitoring_window_statistics.count == 0:
            return True
        else:
            return False
