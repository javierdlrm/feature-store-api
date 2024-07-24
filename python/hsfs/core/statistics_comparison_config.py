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
from typing import Optional, Union

import humps
from hsfs import util


class StatisticsComparisonConfig:
    def __init__(
        self,
        metric: str,
        threshold: Optional[Union[float, int]] = None,
        relative: Optional[bool] = False,
        strict: Optional[bool] = False,
        specific_value: Optional[Union[int, float]] = None,
        id: Optional[int] = None,
        href: Optional[str] = None,
        **kwargs,
    ):
        self._metric = metric.upper()
        self._threshold = threshold
        self._relative = relative
        self._strict = strict
        self._specific_value = specific_value
        self._id = id

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
        return {
            "id": self._id,
            "metric": self._metric,
            "threshold": self._threshold,
            "relative": self._relative,
            "strict": self._strict,
            "specificValue": self._specific_value,
        }

    def json(self) -> str:
        return json.dumps(self, cls=util.FeatureStoreEncoder)

    def __str__(self):
        return self.json()

    def __repr__(self):
        return f"StatisticsComparisonConfig({self._metric!r})"

    def __eq__(self, other):
        if not isinstance(other, StatisticsComparisonConfig):
            return False

        return (
            self._metric,
            self._threshold,
            self._relative,
            self._strict,
            self._specific_value,
        ) == (
            other._metric,
            other._threshold,
            other._relative,
            other._strict,
            other._specific_value,
        )

    @property
    def id(self) -> Optional[int]:
        """Id of the feature monitoring configuration."""
        return self._id

    @property
    def metric(self) -> str:
        """Id of the Feature Group to which this feature monitoring configuration is attached."""
        return self._metric

    @property
    def threshold(self) -> Optional[Union[float, int]]:
        """Name of the Feature View to which this feature monitoring configuration is attached."""
        return self._threshold

    @property
    def relative(self) -> Optional[bool]:
        return self._relative

    @property
    def strict(self) -> Optional[bool]:
        return self._strict

    @property
    def specific_value(self) -> Optional[Union[int, float]]:
        return self._specific_value
