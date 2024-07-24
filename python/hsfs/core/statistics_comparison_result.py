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
from typing import Optional

import humps
from hsfs import util


class StatisticsComparisonResult:
    def __init__(
        self,
        statistics_comparison_config_id: int,
        shift_detected: Optional[bool] = None,
        difference: Optional[float] = None,
        id: Optional[int] = None,
        feature_statistics_result_id: Optional[int] = None,
        href: Optional[str] = None,
        **kwargs,
    ):
        self._id = id
        self._feature_statistics_result_id = feature_statistics_result_id
        self._statistics_comparison_config_id = statistics_comparison_config_id
        self._shift_detected = shift_detected
        self._difference = difference
        self._href = href

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
        return {
            "id": self._id,
            "featureStatisticsResultId": self._feature_statistics_result_id,
            "statisticsComparisonConfigId": self._statistics_comparison_config_id,
            "shiftDetected": self._shift_detected,
            "difference": self._difference,
        }

    def json(self) -> str:
        return json.dumps(self, cls=util.FeatureStoreEncoder)

    def __str__(self):
        return self.json()

    def __repr__(self) -> str:
        return f"StatisticsComparisonResult({self._metric!r})"

    @property
    def id(self) -> Optional[int]:
        """Id of the feature monitoring result."""
        return self._id

    @property
    def feature_statistics_result_id(self) -> Optional[int]:
        """Id of the feature statistics result containing this comparison result."""
        return self._feature_statistics_result_id

    @property
    def statistics_comparison_config_id(self) -> int:
        return self._statistics_comparison_config_id

    @property
    def shift_detected(self) -> Optional[bool]:
        """Id of the feature descriptive statistics computed on the detection window."""
        return self._shift_detected

    @property
    def difference(self) -> Optional[float]:
        """Id of the feature descriptive statistics computed on the reference window."""
        return self._difference

    @property
    def metric(self) -> str:
        return self._metric

    @metric.setter
    def metric(self, metric: str):
        # helping field, not used in the DTO
        self._metric = metric
