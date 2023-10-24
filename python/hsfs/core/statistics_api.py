#
#   Copyright 2020 Logical Clocks AB
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

from typing import Optional, List, Union

from hsfs import client, statistics, feature_view
from hsfs.core import job


class StatisticsApi:
    def __init__(self, feature_store_id, entity_type):
        """Statistics endpoint for `trainingdatasets` and `featuregroups` resource.

        :param feature_store_id: id of the respective featurestore
        :type feature_store_id: int
        :param entity_type: "trainingdatasets" or "featuregroups"
        :type entity_type: str
        """
        self._feature_store_id = feature_store_id
        self._entity_type = entity_type  # TODO: Support FV

    def post(
        self, metadata_instance, stats, training_dataset_version
    ) -> Optional[statistics.Statistics]:
        _client = client.get_instance()
        path_params = self.get_path(metadata_instance, training_dataset_version)

        headers = {"content-type": "application/json"}
        stats = statistics.Statistics.from_response_json(
            _client._send_request(
                "POST", path_params, headers=headers, data=stats.json()
            )
        )
        return self._extract_single_stats(stats)

    def get(
        self,
        metadata_instance,
        computation_time=None,
        start_time=None,
        end_time=None,
        is_event_window=None,
        feature_names=None,
        row_percentage=None,
        for_transformation=None,
        transformed_with_version=None,
        training_dataset_version=None,
    ) -> Optional[statistics.Statistics]:
        """Get single statistics of an entity.

        :param metadata_instance: metadata object of the instance to get statistics of
        :type metadata_instance: TrainingDataset, FeatureGroup, FeatureView
        :param computation_time: Time at which statistics where computed
        :type computation_time: int
        :param start_time: Window start time
        :type start_time: int
        :param end_time: Window end time
        :type end_time: int
        :param is_event_window: Whether window times are event times or commit times
        :type is_event_window: bool
        :param feature_names: List of feature names of which statistics are retrieved
        :type feature_names: List[str]
        :param row_percentage: Percentage of feature values used during statistics computation
        :type row_percentage: float
        :param for_transformation: Whether the statistics were computed for transformation functions or not
        :type for_transformation: bool
        :param transformed_with_version: Training dataset version whose transformation functions were applied before computing statistics
        :type transformed_with_version: int
        :param training_dataset_version: Version of the training dataset on which statistics were computed
        :type training_dataset_version: int
        """
        # get statistics by entity + filters + sorts, including the feature descriptive statistics
        _client = client.get_instance()
        path_params = self.get_path(metadata_instance, training_dataset_version)

        # single statistics
        offset, limit = 0, 1

        headers = {"content-type": "application/json"}
        query_params = self._build_get_query_params(
            computation_time=computation_time,
            start_time=start_time,
            end_time=end_time,
            filter_eq_times=True,
            is_event_window=is_event_window,
            feature_names=feature_names,
            row_percentage=row_percentage,
            for_transformation=for_transformation,
            transformed_with_version=transformed_with_version,
            training_dataset_version=training_dataset_version,
            # retrieve only one entity statistics, including the feature descriptive statistics
            offset=offset,
            limit=limit,
            with_content=True,
        )

        print(
            "[statistics_api] get all",
            "- with filters: "
            + (
                ",".join(query_params["filter_by"])
                if "filter_by" in query_params
                else "none"
            )
            + " , with sorts: "
            + (
                ",".join(query_params["sort_by"])
                if "sort_by" in query_params
                else "none"
            )
            + " , with features: "
            + (",".join(feature_names) if feature_names is not None else "none"),
        )

        # response is either a single item or not found exception
        stats = statistics.Statistics.from_response_json(
            _client._send_request("GET", path_params, query_params, headers=headers)
        )
        return self._extract_single_stats(stats)

    def get_all(
        self,
        metadata_instance,
        computation_time=None,
        start_time=None,
        end_time=None,
        is_event_window=None,
        feature_names=None,
        row_percentage=None,
        for_transformation=None,
        transformed_with_version=None,
        training_dataset_version=None,
    ) -> Optional[List[statistics.Statistics]]:
        """Get all statistics of an entity.

        :param metadata_instance: metadata object of the instance to get statistics of
        :type metadata_instance: TrainingDataset, FeatureGroup, FeatureView
        :param computation_time: Time at which statistics where computed
        :type computation_time: int
        :param start_time: Window start time
        :type start_time: int
        :param end_time: Window end time
        :type end_time: int
        :param is_event_window: Whether window times are event times or commit times
        :type is_event_window: bool
        :param feature_names: List of feature names of which statistics are retrieved
        :type feature_names: List[str]
        :param row_percentage: Percentage of feature values used during statistics computation
        :type row_percentage: float
        :param for_transformation: Whether the statistics were computed for transformation functions or not
        :type for_transformation: bool
        :param transformed_with_version: Training dataset version whose transformation functions were applied before computing statistics
        :type transformed_with_version: int
        :param training_dataset_version: Version of the training dataset on which statistics were computed
        :type training_dataset_version: int
        """
        # get all statistics by entity + filters + sorts, without the feature descriptive statistics
        _client = client.get_instance()
        path_params = self.get_path(metadata_instance, training_dataset_version)

        # multiple statistics
        offset, limit = 0, None

        headers = {"content-type": "application/json"}
        query_params = self._build_get_query_params(
            computation_time=computation_time,
            start_time=start_time,
            end_time=end_time,
            filter_eq_times=False,
            is_event_window=is_event_window,
            feature_names=feature_names,
            row_percentage=row_percentage,
            for_transformation=for_transformation,
            transformed_with_version=transformed_with_version,
            training_dataset_version=training_dataset_version,
            # retrieve all entity statistics, excluding feature descriptive statistics
            offset=offset,
            limit=limit,
            with_content=False,
        )

        print(
            "[statistics_api] get all",
            "- with filters: "
            + (
                ",".join(query_params["filter_by"])
                if "filter_by" in query_params
                else "none"
            )
            + " , with sorts: "
            + (
                ",".join(query_params["sort_by"])
                if "sort_by" in query_params
                else "none"
            )
            + " , with features: "
            + (",".join(feature_names) if feature_names is not None else "none"),
        )

        # response is either a single item or not found exception
        return statistics.Statistics.from_response_json(
            _client._send_request("GET", path_params, query_params, headers=headers)
        )

    def compute(self, metadata_instance, training_dataset_version=None) -> job.Job:
        """Compute statistics for an entity.

        :param metadata_instance: metadata object of the instance to compute statistics for
        :type metadata_instance: TrainingDataset, FeatureGroup, FeatureView
        :param training_dataset_version: version of the training dataset metadata object
        :type training_dataset_version: int
        """
        _client = client.get_instance()
        path_params = self.get_path(metadata_instance, training_dataset_version) + [
            "compute"
        ]
        return job.Job.from_response_json(_client._send_request("POST", path_params))

    def get_path(self, metadata_instance, training_dataset_version=None) -> list:
        """Get statistics path.

        :param metadata_instance: metadata object of the instance to compute statistics for
        :type metadata_instance: TrainingDataset, FeatureGroup
        :param training_dataset_version: version of the training dataset metadata object
        :type training_dataset_version: int
        """
        _client = client.get_instance()
        if isinstance(metadata_instance, feature_view.FeatureView):
            path = [
                "project",
                _client._project_id,
                "featurestores",
                self._feature_store_id,
                "featureview",
                metadata_instance.name,
                "version",
                metadata_instance.version,
            ]
            if training_dataset_version is not None:
                path += [
                    "trainingdatasets",
                    "version",
                    training_dataset_version,
                ]
            return path + ["statistics"]
        else:
            return [
                "project",
                _client._project_id,
                "featurestores",
                self._feature_store_id,
                self._entity_type,
                metadata_instance.id,
                "statistics",
            ]

    def _extract_single_stats(self, stats) -> Optional[statistics.Statistics]:
        return stats[0] if isinstance(stats, list) else stats

    def _build_get_query_params(
        self,
        computation_time=None,
        start_time=None,
        end_time=None,
        filter_eq_times=None,
        is_event_window=None,
        feature_names=None,
        row_percentage=None,
        for_transformation=None,
        transformed_with_version=None,
        training_dataset_version=None,
        offset=0,
        limit=None,
        with_content=False,
    ) -> dict:
        """Build query parameters for statistics requests.

        :param computation_time: Time at which statistics where computed
        :type computation_time: int
        :param start_time: Window start time
        :type start_time: int
        :param end_time: Window end time
        :type end_time: int
        :param is_event_window: Whether window times are event times or commit times
        :type is_event_window: bool
        :param feature_names: List of feature names of which statistics are retrieved
        :type feature_names: List[str]
        :param row_percentage: Percentage of feature values used during statistics computation
        :type row_percentage: float
        :param for_transformation: Whether the statistics were computed for transformation functions or not
        :type for_transformation: bool
        :param transformed_with_version: Training dataset version whose transformation functions were applied before computing statistics
        :type transformed_with_version: int
        :param training_dataset_version: Version of the training dataset on which statistics were computed
        :type training_dataset_version: int
        :param offset: Offset for pagination queries
        :type offset: int
        :param limit: Limit for pagination queries
        :type limit: int
        :param with_content: Whether include feature descriptive statistics in the response or not
        :type with_content: bool
        """
        query_params: dict[str, Union[int, str, List[str]]] = {"offset": offset}
        if limit is not None:
            query_params["limit"] = limit
        if with_content:
            query_params["fields"] = "content"

        sorts: List[str] = []
        filters: List[str] = []

        # filters and sorts

        # commit time
        if computation_time is not None:
            sorts.append("commit_time:desc")
            filters.append("commit_time_ltoeq:" + str(computation_time))
        elif (start_time is None and end_time is None) or is_event_window:
            # if not commit window, sort by commit
            sorts.append("commit_time:desc")

        # window times
        if start_time is not None:
            col_name = (
                "window_start_event_time"
                if is_event_window
                else "window_start_commit_time"
            )
            sorts.append(col_name + ":asc")
            filter_name = col_name + ("_eq" if filter_eq_times else "_gtoeq")
            filters.append(filter_name + ":" + str(start_time))
        if end_time is not None:
            col_name = (
                "window_end_event_time" if is_event_window else "window_end_commit_time"
            )
            sorts.append(col_name + ":desc")
            filter_name = col_name + ("_eq" if filter_eq_times else "_ltoeq")
            filters.append(filter_name + ":" + str(end_time))

        # row percentage
        if row_percentage is not None:
            filters.append("row_percentage_eq:" + str(row_percentage))

        # for transformation
        if for_transformation is not None:
            filters.append("for_transformation_eq:" + str(for_transformation))

        # feature names
        if feature_names is not None:
            query_params["feature_names"] = feature_names

        # others
        if is_event_window is not None:
            query_params["is_event_window"] = is_event_window
        if transformed_with_version is not None:
            query_params["transformed_with_version"] = transformed_with_version
        if training_dataset_version is not None:
            query_params["training_dataset_version"] = training_dataset_version

        if sorts:
            query_params["sort_by"] = sorts
        if filters:
            query_params["filter_by"] = filters

        return query_params
