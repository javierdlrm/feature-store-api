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
from typing import Any, Dict, List, Optional, Union

import humps
from hsfs import util
from hsfs.core.statistics_comparison_config import StatisticsComparisonConfig


class FeatureStatisticsConfig:
    def __init__(
        self,
        feature_name: str,
        statistics_comparison_configs: Optional[
            Union[List[StatisticsComparisonConfig], List[Dict[str, Any]]]
        ] = None,
        id: Optional[int] = None,
        href: Optional[str] = None,
        **kwargs,
    ):
        self._feature_name = feature_name
        self._statistics_comparison_configs = self._parse_statistics_comparison_configs(
            statistics_comparison_configs
        )
        self._id = id
        self._href = href

    def _parse_statistics_comparison_configs(
        self,
        statistics_comparison_configs: Optional[
            Union[List[StatisticsComparisonConfig], List[Dict[str, Any]]]
        ],
    ) -> Optional[List[StatisticsComparisonConfig]]:
        if statistics_comparison_configs is None:
            return None
        sc_configs = []
        for sc_config in statistics_comparison_configs:
            sc_configs.append(
                sc_config
                if isinstance(sc_config, StatisticsComparisonConfig)
                else StatisticsComparisonConfig.from_response_json(sc_config)
            )
        return sc_configs

    @classmethod
    def from_response_json(cls, json_dict) -> FeatureStatisticsConfig:
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
            "featureName": self._feature_name,
        }
        if self._statistics_comparison_configs is not None:
            the_dict["statisticsComparisonConfigs"] = [
                sc_config.to_dict() for sc_config in self._statistics_comparison_configs
            ]
        return the_dict

    def json(self) -> str:
        return json.dumps(self, cls=util.FeatureStoreEncoder)

    def __str__(self):
        return self.json()

    def __repr__(self):
        return f"FeatureStatisticsConfig({self._feature_name!r})"

    @property
    def id(self) -> Optional[int]:
        """Id of the feature monitoring configuration."""
        return self._id

    @property
    def feature_name(self) -> str:
        return self._feature_name

    @property
    def statistics_comparison_configs(
        self,
    ) -> Optional[List[StatisticsComparisonConfig]]:
        return self._statistics_comparison_configs
