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
from typing import List, Optional, Set, Union

import humps
from hsfs import util
from hsfs.core.feature_descriptive_statistics import FeatureDescriptiveStatistics
from hsfs.core.statistics_comparison_result import StatisticsComparisonResult


class FeatureStatisticsResult:
    def __init__(
        self,
        feature_name: str,
        shifted_metric_names: Optional[Union[Set[str], List[str]]] = None,
        statistics_comparison_results: Optional[
            Union[List[StatisticsComparisonResult], List[dict]]
        ] = None,
        detection_statistics_id: Optional[int] = None,
        reference_statistics_id: Optional[int] = None,
        detection_statistics: Optional[
            Union[FeatureDescriptiveStatistics, dict]
        ] = None,
        reference_statistics: Optional[
            Union[FeatureDescriptiveStatistics, dict]
        ] = None,
        id: Optional[int] = None,
        href: Optional[str] = None,
        **kwargs,
    ):
        self._id = id
        self._href = href
        self._feature_name = feature_name
        self._shifted_metric_names = self._parse_shifted_metric_names(
            shifted_metric_names
        )
        self._statistics_comparison_results = self._parse_statistics_comparison_result(
            statistics_comparison_results
        )
        self._detection_statistics_id = detection_statistics_id
        self._reference_statistics_id = reference_statistics_id
        self._detection_statistics = self._parse_descriptive_statistics(
            detection_statistics
        )
        self._reference_statistics = self._parse_descriptive_statistics(
            reference_statistics
        )

    def _parse_descriptive_statistics(
        self,
        statistics: Optional[Union[FeatureDescriptiveStatistics, dict]],
    ) -> Optional[FeatureDescriptiveStatistics]:
        if statistics is None:
            return None
        return (
            statistics
            if isinstance(statistics, FeatureDescriptiveStatistics)
            else FeatureDescriptiveStatistics.from_response_json(statistics)
        )

    def _parse_statistics_comparison_result(
        self,
        statistics_comparison_results: Optional[
            List[Union[StatisticsComparisonResult, dict]]
        ] = None,
    ) -> Optional[List[StatisticsComparisonResult]]:
        if statistics_comparison_results is None:
            return None
        sc_results = []
        for sc_result in statistics_comparison_results:
            sc_results.append(
                sc_result
                if isinstance(sc_result, StatisticsComparisonResult)
                else StatisticsComparisonResult.from_response_json(sc_result)
            )
        return sc_results

    def _parse_shifted_metric_names(self, shifted_metric_names) -> Optional[Set[str]]:
        return (
            set(shifted_metric_names)
            if isinstance(shifted_metric_names, list)
            else shifted_metric_names
        )

    @classmethod
    def from_response_json(cls, json_dict):
        json_decamelized = humps.decamelize(json_dict)
        if "count" in json_decamelized:
            if json_decamelized["count"] == 0:
                return []
            return [cls(**result) for result in json_decamelized["items"]]
        else:
            return cls(**json_decamelized)

    def to_dict(self):
        the_dict = {
            "id": self._id,
            "featureName": self._feature_name,
        }
        if self._statistics_comparison_results:
            the_dict["statisticsComparisonResults"] = [
                sc_result.to_dict() for sc_result in self._statistics_comparison_results
            ]
        if self._detection_statistics_id is not None:
            the_dict["detectionStatisticsId"] = self._detection_statistics_id
        if self._reference_statistics_id is not None:
            the_dict["referenceStatisticsId"] = self._reference_statistics_id
        if self._detection_statistics is not None:
            the_dict["detectionStatistics"] = self._detection_statistics.to_dict()
        if self._reference_statistics is not None:
            the_dict["referenceStatistics"] = self._reference_statistics.to_dict()
        if self._shifted_metric_names:
            the_dict["shiftedMetricNames"] = list(self._shifted_metric_names)

        return the_dict

    def json(self) -> str:
        return json.dumps(self, cls=util.FeatureStoreEncoder)

    def __str__(self):
        return self.json()

    def __repr__(self) -> str:
        return f"FeatureStatisticsResult({self._feature_name!r})"

    @property
    def id(self) -> Optional[int]:
        """Id of the feature monitoring result."""
        return self._id

    @property
    def feature_name(self) -> str:
        """Feature name"""
        return self._feature_name

    @property
    def detection_statistics_id(self) -> Optional[int]:
        """Id of the feature descriptive statistics computed on the detection window."""
        return self._detection_statistics_id

    @property
    def reference_statistics_id(self) -> Optional[int]:
        """Id of the feature descriptive statistics computed on the reference window."""
        return self._reference_statistics_id

    @property
    def detection_statistics(self) -> Optional[FeatureDescriptiveStatistics]:
        """Feature descriptive statistics computed on the detection window."""
        return self._detection_statistics

    @property
    def reference_statistics(self) -> Optional[FeatureDescriptiveStatistics]:
        """Feature descriptive statistics computed on the reference window."""
        return self._reference_statistics

    @property
    def shifted_metric_names(self) -> Optional[Set[str]]:
        return self._shifted_metric_names

    @property
    def statistics_comparison_results(
        self,
    ) -> Optional[List[StatisticsComparisonResult]]:
        return self._statistics_comparison_results
