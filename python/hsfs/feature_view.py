#
#   Copyright 2022 Logical Clocks AB
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

import json
import warnings
from datetime import datetime, date
from typing import Optional, Union, List, Dict, Any, TypeVar
from hsfs.client.exceptions import FeatureStoreException
from hsfs.core import feature_monitoring_config as fmc
from hsfs.core import feature_monitoring_result as fmr

import humps
import copy
import pandas as pd
import numpy as np

from hsfs import (
    util,
    training_dataset_feature,
    storage_connector,
    training_dataset,
)
from hsfs.constructor import query, filter
from hsfs.core import (
    feature_view_engine,
    transformation_function_engine,
    vector_server,
    statistics_engine,
    feature_monitoring_config_engine,
    feature_monitoring_result_engine,
)
from hsfs.transformation_function import TransformationFunction
from hsfs.statistics_config import StatisticsConfig
from hsfs.core.feature_view_api import FeatureViewApi
from hsfs.training_dataset_split import TrainingDatasetSplit


class FeatureView:
    ENTITY_TYPE = "featureview"

    def __init__(
        self,
        name: str,
        query,
        featurestore_id,
        id=None,
        version: Optional[int] = None,
        description: Optional[str] = "",
        labels: Optional[List[str]] = [],
        transformation_functions: Optional[Dict[str, TransformationFunction]] = {},
        featurestore_name=None,
    ):
        self._name = name
        self._id = id
        self._query = query
        self._featurestore_id = featurestore_id
        self._feature_store_id = featurestore_id  # for consistency with feature group
        self._feature_store_name = featurestore_name
        self._version = version
        self._description = description
        self._labels = labels
        self._transformation_functions = (
            {
                ft_name: copy.deepcopy(transformation_functions[ft_name])
                for ft_name in transformation_functions
            }
            if transformation_functions
            else transformation_functions
        )
        self._features = []
        self._feature_view_engine = feature_view_engine.FeatureViewEngine(
            featurestore_id
        )
        self._transformation_function_engine = (
            transformation_function_engine.TransformationFunctionEngine(featurestore_id)
        )
        self._single_vector_server = None
        self._batch_vectors_server = None
        self._batch_scoring_server = None

        self._statistics_engine = statistics_engine.StatisticsEngine(
            featurestore_id, self.ENTITY_TYPE
        )

        if self._id:
            self._init_feature_monitoring_engine()

    def delete(self):
        """Delete current feature view, all associated metadata and training data.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # delete a feature view
            feature_view.delete()
            ```

        !!! danger "Potentially dangerous operation"
            This operation drops all metadata associated with **this version** of the
            feature view **and** related training dataset **and** materialized data in HopsFS.

        # Raises
            `hsfs.client.exceptions.RestAPIError`.
        """
        warnings.warn(
            "All jobs associated to feature view `{}`, version `{}` will be removed.".format(
                self._name, self._version
            ),
            util.JobWarning,
        )
        self._feature_view_engine.delete(self.name, self.version)

    @staticmethod
    def clean(feature_store_id: int, feature_view_name: str, feature_view_version: str):
        """
        Delete the feature view and all associated metadata and training data.
        This can delete corrupted feature view which cannot be retrieved due to a corrupted query for example.

        !!! example
            ```python
            # delete a feature view and all associated metadata
            feature_view.clean(
                feature_store_id=1,
                feature_view_name='feature_view_name',
                feature_view_version=1
            )
            ```

        !!! danger "Potentially dangerous operation"
            This operation drops all metadata associated with **this version** of the
            feature view **and** related training dataset **and** materialized data in HopsFS.

        # Arguments
            feature_store_id: int. Id of feature store.
            feature_view_name: str. Name of feature view.
            feature_view_version: str. Version of feature view.

        # Raises
            `hsfs.client.exceptions.RestAPIError`.
        """
        if not isinstance(feature_store_id, int):
            raise ValueError("`feature_store_id` should be an integer.")
        FeatureViewApi(feature_store_id).delete_by_name_version(
            feature_view_name, feature_view_version
        )

    def update(self):
        """Update the description of the feature view.

        !!! example "Update the feature view with a new description."
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            feature_view.description = "new description"
            feature_view.update()

            # Description is updated in the metadata. Below should return "new description".
            fs.get_feature_view("feature_view_name", 1).description
            ```

        # Returns
            `FeatureView` Updated feature view.

        # Raises
            `hsfs.client.exceptions.RestAPIError`.
        """
        return self._feature_view_engine.update(self)

    def init_serving(
        self,
        training_dataset_version: Optional[int] = None,
        external: Optional[bool] = None,
        options: Optional[dict] = None,
    ):
        """Initialise feature view to retrieve feature vector from online feature store.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # initialise feature view to retrieve a feature vector
            feature_view.init_serving(training_dataset_version=1)
            ```

        # Arguments
            training_dataset_version: int, optional. Default to be 1. Transformation statistics
                are fetched from training dataset and applied to the feature vector.
            external: boolean, optional. If set to True, the connection to the
                online feature store is established using the same host as
                for the `host` parameter in the [`hsfs.connection()`](connection_api.md#connection) method.
                If set to False, the online feature store storage connector is used which relies on the private IP.
                Defaults to True if connection to Hopsworks is established from external environment (e.g AWS
                Sagemaker or Google Colab), otherwise to False.
            options: Additional options as key/value pairs for configuring online serving engine.
                * key: kwargs of SqlAlchemy engine creation (See: https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine).
                  For example: `{"pool_size": 10}`
        """

        if training_dataset_version is None:
            training_dataset_version = 1
            warnings.warn(
                "No training dataset version was provided to initialise serving. Defaulting to version 1.",
                util.VersionWarning,
            )

        # initiate single vector server
        self._single_vector_server = vector_server.VectorServer(
            self._featurestore_id, self._features, training_dataset_version
        )
        self._single_vector_server.init_serving(self, False, external, options=options)

        # initiate batch vector server
        self._batch_vectors_server = vector_server.VectorServer(
            self._featurestore_id, self._features, training_dataset_version
        )
        self._batch_vectors_server.init_serving(self, True, external, options=options)

        # initiate batch scoring server
        self.init_batch_scoring(training_dataset_version)

    def init_batch_scoring(
        self,
        training_dataset_version: Optional[int] = None,
    ):
        """Initialise feature view to retrieve feature vector from offline feature store.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # initialise feature view to retrieve feature vector from offline feature store
            feature_view.init_batch_scoring(training_dataset_version=1)

            # get batch data
            batch_data = feature_view.get_batch_data(...)
            ```

        # Arguments
            training_dataset_version: int, optional. Default to be None. Transformation statistics
                are fetched from training dataset and applied to the feature vector.
        """

        self._batch_scoring_server = vector_server.VectorServer(
            self._featurestore_id, self._features, training_dataset_version
        )
        self._batch_scoring_server.init_batch_scoring(self)

    def get_batch_query(
        self,
        start_time: Optional[Union[str, int, datetime, date]] = None,
        end_time: Optional[Union[str, int, datetime, date]] = None,
    ):
        """Get a query string of the batch query.

        !!! example "Batch query for the last 24 hours"
            ```python
                # get feature store instance
                fs = ...

                # get feature view instance
                feature_view = fs.get_feature_view(...)

                # set up dates
                import datetime
                start_date = (datetime.datetime.now() - datetime.timedelta(hours=24))
                end_date = (datetime.datetime.now())

                # get a query string of batch query
                query_str = feature_view.get_batch_query(
                    start_time=start_date,
                    end_time=end_date
                )
                # print query string
                print(query_str)
            ```

        # Arguments
            start_time: Start event time for the batch query, inclusive. Optional. Strings should be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`,
                `%Y-%m-%d %H:%M:%S`, or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            end_time: End event time for the batch query, exclusive. Optional. Strings should be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`,
                `%Y-%m-%d %H:%M:%S`, or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.

        # Returns
            `str`: batch query
        """
        return self._feature_view_engine.get_batch_query_string(
            self,
            start_time,
            end_time,
            training_dataset_version=(
                self._batch_scoring_server.training_dataset_version
                if self._batch_scoring_server
                else None
            ),
        )

    def get_feature_vector(
        self,
        entry: Dict[str, Any],
        passed_features: Optional[Dict[str, Any]] = {},
        external: Optional[bool] = None,
    ):
        """Returns assembled feature vector from online feature store.
            Call [`feature_view.init_serving`](#init_serving) before this method if the following configurations are needed.
              1. The training dataset version of the transformation statistics
              2. Additional configurations of online serving engine
        !!! warning "Missing primary key entries"
            If the provided primary key `entry` can't be found in one or more of the feature groups
            used by this feature view the call to this method will raise an exception.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get a feature vector
            feature_view.get_feature_vector(
                entry = {"pk1": 1, "pk2": 2}
            )
            ```

        !!! example "Get feature vector with user-supplied features"
            ```python
            # get feature store instance
            fs = ...
            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # the application provides a feature value 'app_attr'
            app_attr = ...

            # get a feature vector
            feature_view.get_feature_vector(
                entry = {"pk1": 1, "pk2": 2},
                passed_features = { "app_feature" : app_attr }
            )
            ```

        # Arguments
            entry: dictionary of feature group primary key and values provided by serving application.
            passed_features: dictionary of feature values provided by the application at runtime.
                They can replace features values fetched from the feature store as well as
                providing feature values which are not available in the feature store.
            external: boolean, optional. If set to True, the connection to the
                online feature store is established using the same host as
                for the `host` parameter in the [`hsfs.connection()`](connection_api.md#connection) method.
                If set to False, the online feature store storage connector is used
                which relies on the private IP. Defaults to True if connection to Hopsworks is established from
                external environment (e.g AWS Sagemaker or Google Colab), otherwise to False.

        # Returns
            `list` List of feature values related to provided primary keys, ordered according to positions of this
            features in the feature view query.

        # Raises
            `Exception`. When primary key entry cannot be found in one or more of the feature groups used by this
                feature view.
        """
        if self._single_vector_server is None:
            self.init_serving(external=external)
        return self._single_vector_server.get_feature_vector(entry, passed_features)

    def get_feature_vectors(
        self,
        entry: List[Dict[str, Any]],
        passed_features: Optional[List[Dict[str, Any]]] = {},
        external: Optional[bool] = None,
    ):
        """Returns assembled feature vectors in batches from online feature store.
            Call [`feature_view.init_serving`](#init_serving) before this method if the following configurations are needed.
              1. The training dataset version of the transformation statistics
              2. Additional configurations of online serving engine
        !!! warning "Missing primary key entries"
            If any of the provided primary key elements in `entry` can't be found in any
            of the feature groups, no feature vector for that primary key value will be
            returned.
            If it can be found in at least one but not all feature groups used by
            this feature view the call to this method will raise an exception.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get assembled serving vectors
            feature_view.get_feature_vectors(
                entry = [
                    {"pk1": 1, "pk2": 2},
                    {"pk1": 3, "pk2": 4},
                    {"pk1": 5, "pk2": 6}
                ]
            )
            ```

        # Arguments
            entry: a list of dictionary of feature group primary key and values provided by serving application.
            passed_features: a list of dictionary of feature values provided by the application at runtime.
                They can replace features values fetched from the feature store as well as
                providing feature values which are not available in the feature store.
            external: boolean, optional. If set to True, the connection to the
                online feature store is established using the same host as
                for the `host` parameter in the [`hsfs.connection()`](connection_api.md#connection) method.
                If set to False, the online feature store storage connector is used
                which relies on the private IP. Defaults to True if connection to Hopsworks is established from
                external environment (e.g AWS Sagemaker or Google Colab), otherwise to False.

        # Returns
            `List[list]` List of lists of feature values related to provided primary keys, ordered according
                to positions of this features in the feature view query.

        # Raises
            `Exception`. When primary key entry cannot be found in one or more of the feature groups used by this
                feature view.
        """
        if self._batch_vectors_server is None:
            self.init_serving(external=external)
        return self._batch_vectors_server.get_feature_vectors(entry, passed_features)

    def get_batch_data(
        self,
        start_time: Optional[Union[str, int, datetime, date]] = None,
        end_time: Optional[Union[str, int, datetime, date]] = None,
        read_options=None,
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """Get a batch of data from an event time interval from the offline feature store.

        !!! example "Batch data for the last 24 hours"
            ```python
                # get feature store instance
                fs = ...

                # get feature view instance
                feature_view = fs.get_feature_view(...)

                # set up dates
                import datetime
                start_date = (datetime.datetime.now() - datetime.timedelta(hours=24))
                end_date = (datetime.datetime.now())

                # get a batch of data
                df = feature_view.get_batch_data(
                    start_time=start_date,
                    end_time=end_date
                )
            ```

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            start_time: Start event time for the batch query, inclusive. Optional. Strings should be
                formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            end_time: End event time for the batch query, exclusive. Optional. Strings should be
                formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            read_options: User provided read options. Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            `DataFrame`: A dataframe
        """

        if self._batch_scoring_server is None:
            self.init_batch_scoring()

        return self._feature_view_engine.get_batch_data(
            self,
            start_time,
            end_time,
            self._batch_scoring_server.training_dataset_version,
            self._batch_scoring_server._transformation_functions,
            read_options,
            spine,
        )

    def add_tag(self, name: str, value):
        """Attach a tag to a feature view.

        A tag consists of a name and value pair.
        Tag names are unique identifiers across the whole cluster.
        The value of a tag can be any valid json - primitives, arrays or json objects.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # attach a tag to a feature view
            feature_view.add_tag(name="tag_schema", value={"key", "value"})
            ```

        # Arguments
            name: Name of the tag to be added.
            value: Value of the tag to be added.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to add the tag.
        """
        return self._feature_view_engine.add_tag(self, name, value)

    def get_tag(self, name: str):
        """Get the tags of a feature view.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get a tag of a feature view
            name = feature_view.get_tag('tag_name')
            ```

        # Arguments
            name: Name of the tag to get.

        # Returns
            tag value

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to retrieve the tag.
        """
        return self._feature_view_engine.get_tag(self, name)

    def get_tags(self):
        """Returns all tags attached to a training dataset.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get tags
            list_tags = feature_view.get_tags()
            ```

        # Returns
            `Dict[str, obj]` of tags.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to retrieve the tags.
        """
        return self._feature_view_engine.get_tags(self)

    def get_parent_feature_groups(self):
        """Get the parents of this feature view, based on explicit provenance.
        Parents are feature groups or external feature groups. These feature
        groups can be accessible, deleted or inaccessible.
        For deleted and inaccessible feature groups, only a minimal information is
        returned.

        # Returns
            `Links`: Object containing the section of provenance graph requested.
        """
        return self._feature_view_engine.get_parent_feature_groups(self)

    def delete_tag(self, name: str):
        """Delete a tag attached to a feature view.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # delete a tag
            feature_view.delete_tag('name_of_tag')
            ```

        # Arguments
            name: Name of the tag to be removed.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to delete the tag.
        """
        return self._feature_view_engine.delete_tag(self, name)

    def create_training_data(
        self,
        start_time: Optional[Union[str, int, datetime, date]] = "",
        end_time: Optional[Union[str, int, datetime, date]] = "",
        storage_connector: Optional[storage_connector.StorageConnector] = None,
        location: Optional[str] = "",
        description: Optional[str] = "",
        extra_filter: Optional[Union[filter.Filter, filter.Logic]] = None,
        data_format: Optional[str] = "parquet",
        coalesce: Optional[bool] = False,
        seed: Optional[int] = None,
        statistics_config: Optional[Union[StatisticsConfig, bool, dict]] = None,
        write_options: Optional[Dict[Any, Any]] = {},
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """Create the metadata for a training dataset and save the corresponding training data into `location`.
        The training data can be retrieved by calling `feature_view.get_training_data`.

        !!! example "Create training dataset"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # create a training dataset
            version, job = feature_view.create_training_data(
                description='Description of a dataset',
                data_format='csv',
                # async creation in order not to wait till finish of the job
                write_options={"wait_for_job": False}
            )
            ```

        !!! example "Create training data specifying date range  with dates as strings"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            start_time = "2022-01-01 00:00:00"
            end_time = "2022-06-06 23:59:59"

            # create a training dataset
            version, job = feature_view.create_training_data(
                start_time=start_time,
                end_time=end_time,
                description='Description of a dataset',
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format='csv'
            )

            # When we want to read the training data, we need to supply the training data version returned by the create_training_data method:
            X_train, X_test, y_train, y_test = feature_view.get_training_data(version)
            ```

        !!! example "Create training data specifying date range  with dates as datetime objects"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            from datetime import datetime
            date_format = "%Y-%m-%d %H:%M:%S"

            start_time = datetime.strptime("2022-01-01 00:00:00", date_format)
            end_time = datetime.strptime("2022-06-06 23:59:59", date_format)

            # create a training dataset
            version, job = feature_view.create_training_data(
                start_time=start_time,
                end_time=end_time,
                description='Description of a dataset',
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format='csv'
            )
            ```

        !!! example "Write training dataset to external storage"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get storage connector instance
            external_storage_connector = fs.get_storage_connector("storage_connector_name")

            # create a train-test split dataset
            version, job = feature_view.create_training_data(
                start_time=...,
                end_time=...,
                storage_connector = external_storage_connector,
                description=...,
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format=...
            )
            ```

        !!! info "Data Formats"
            The feature store currently supports the following data formats for
            training datasets:

            1. tfrecord
            2. csv
            3. tsv
            4. parquet
            5. avro
            6. orc

            Currently not supported petastorm, hdf5 and npy file formats.

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            start_time: Start event time for the training dataset query, inclusive. Optional. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            end_time: End event time for the training dataset query, exclusive. Optional. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            storage_connector: Storage connector defining the sink location for the
                training dataset, defaults to `None`, and materializes training dataset
                on HopsFS.
            location: Path to complement the sink storage connector with, e.g if the
                storage connector points to an S3 bucket, this path can be used to
                define a sub-directory inside the bucket to place the training dataset.
                Defaults to `""`, saving the training dataset at the root defined by the
                storage connector.
            description: A string describing the contents of the training dataset to
                improve discoverability for Data Scientists, defaults to empty string
                `""`.
            extra_filter: Additional filters to be attached to the training dataset.
                The filters will be also applied in `get_batch_data`.
            data_format: The data format used to save the training dataset,
                defaults to `"parquet"`-format.
            coalesce: If true the training dataset data will be coalesced into
                a single partition before writing. The resulting training dataset
                will be a single file per split. Default False.
            seed: Optionally, define a seed to create the random splits with, in order
                to guarantee reproducability, defaults to `None`.
            statistics_config: A configuration object, or a dictionary with keys
                "`enabled`" to generally enable descriptive statistics computation for
                this feature group, `"correlations`" to turn on feature correlation
                computation and `"histograms"` to compute feature value frequencies. The
                values should be booleans indicating the setting. To fully turn off
                statistics computation pass `statistics_config=False`. Defaults to
                `None` and will compute only descriptive statistics.
            write_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                When using the `python` engine, write_options can contain the
                following entries:
                * key `spark` and value an object of type
                [hsfs.core.job_configuration.JobConfiguration](../job_configuration)
                  to configure the Hopsworks Job used to compute the training dataset.
                * key `wait_for_job` and value `True` or `False` to configure
                  whether or not to the save call should return only
                  after the Hopsworks Job has finished. By default it waits.
                Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            (td_version, `Job`): Tuple of training dataset version and job.
                When using the `python` engine, it returns the Hopsworks Job
                that was launched to create the training dataset.
        """
        td = training_dataset.TrainingDataset(
            name=self.name,
            version=None,
            event_start_time=start_time,
            event_end_time=end_time,
            description=description,
            data_format=data_format,
            storage_connector=storage_connector,
            location=location,
            featurestore_id=self._featurestore_id,
            splits={},
            seed=seed,
            statistics_config=statistics_config,
            coalesce=coalesce,
            extra_filter=extra_filter,
        )
        # td_job is used only if the python engine is used
        td, td_job = self._feature_view_engine.create_training_dataset(
            self, td, write_options, spine
        )
        warnings.warn(
            "Incremented version to `{}`.".format(td.version),
            util.VersionWarning,
        )

        return td.version, td_job

    def create_train_test_split(
        self,
        test_size: Optional[float] = None,
        train_start: Optional[Union[str, int, datetime, date]] = "",
        train_end: Optional[Union[str, int, datetime, date]] = "",
        test_start: Optional[Union[str, int, datetime, date]] = "",
        test_end: Optional[Union[str, int, datetime, date]] = "",
        storage_connector: Optional[storage_connector.StorageConnector] = None,
        location: Optional[str] = "",
        description: Optional[str] = "",
        extra_filter: Optional[Union[filter.Filter, filter.Logic]] = None,
        data_format: Optional[str] = "parquet",
        coalesce: Optional[bool] = False,
        seed: Optional[int] = None,
        statistics_config: Optional[Union[StatisticsConfig, bool, dict]] = None,
        write_options: Optional[Dict[Any, Any]] = {},
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """Create the metadata for a training dataset and save the corresponding training data into `location`.
        The training data is split into train and test set at random or according to time ranges.
        The training data can be retrieved by calling `feature_view.get_train_test_split`.

        !!! example "Create random splits"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # create a train-test split dataset
            version, job = feature_view.create_train_test_split(
                test_size=0.2,
                description='Description of a dataset',
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format='csv'
            )
            ```

        !!! example "Create time series splits by specifying date as string"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            train_start = "2022-01-01 00:00:00"
            train_end = "2022-06-06 23:59:59"
            test_start = "2022-06-07 00:00:00"
            test_end = "2022-12-25 23:59:59"

            # create a train-test split dataset
            version, job = feature_view.create_train_test_split(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                description='Description of a dataset',
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format='csv'
            )
            ```

        !!! example "Create time series splits by specifying date as datetime object"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            from datetime import datetime
            date_format = "%Y-%m-%d %H:%M:%S"

            train_start = datetime.strptime("2022-01-01 00:00:00", date_format)
            train_end = datetime.strptime("2022-06-06 23:59:59", date_format)
            test_start = datetime.strptime("2022-06-07 00:00:00", date_format)
            test_end = datetime.strptime("2022-12-25 23:59:59" , date_format)

            # create a train-test split dataset
            version, job = feature_view.create_train_test_split(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                description='Description of a dataset',
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format='csv'
            )
            ```

        !!! example "Write training dataset to external storage"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get storage connector instance
            external_storage_connector = fs.get_storage_connector("storage_connector_name")

            # create a train-test split dataset
            version, job = feature_view.create_train_test_split(
                train_start=...,
                train_end=...,
                test_start=...,
                test_end=...,
                storage_connector = external_storage_connector,
                description=...,
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format=...
            )
            ```

        !!! info "Data Formats"
            The feature store currently supports the following data formats for
            training datasets:

            1. tfrecord
            2. csv
            3. tsv
            4. parquet
            5. avro
            6. orc

            Currently not supported petastorm, hdf5 and npy file formats.

        !!! warning "Warning, the following code will fail because category column contains sparse values and training dataset may not have all values available in test split."
            ```python
            import pandas as pd

            df = pd.DataFrame({
                'category_col':['category_a','category_b','category_c','category_d'],
                'numeric_col': [40,10,60,40]
            })

            feature_group = fs.get_or_create_feature_group(
                name='feature_group_name',
                version=1,
                primary_key=['category_col']
            )

            feature_group.insert(df)

            label_encoder = fs.get_transformation_function(name='label_encoder')

            feature_view = fs.create_feature_view(
                name='feature_view_name',
                query=feature_group.select_all(),
                transformation_functions={'category_col':label_encoder}
            )

            feature_view.create_train_test_split(
                test_size=0.5
            )
            # Output: KeyError: 'category_c'
            ```

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            test_size: size of test set.
            train_start: Start event time for the train split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            train_end: End event time for the train split query, exclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_start: Start event time for the test split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_end: End event time for the test split query, exclusive. Strings should
                be  formatted in one of the following ormats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            storage_connector: Storage connector defining the sink location for the
                training dataset, defaults to `None`, and materializes training dataset
                on HopsFS.
            location: Path to complement the sink storage connector with, e.g if the
                storage connector points to an S3 bucket, this path can be used to
                define a sub-directory inside the bucket to place the training dataset.
                Defaults to `""`, saving the training dataset at the root defined by the
                storage connector.
            description: A string describing the contents of the training dataset to
                improve discoverability for Data Scientists, defaults to empty string
                `""`.
            extra_filter: Additional filters to be attached to the training dataset.
                The filters will be also applied in `get_batch_data`.
            data_format: The data format used to save the training dataset,
                defaults to `"parquet"`-format.
            coalesce: If true the training dataset data will be coalesced into
                a single partition before writing. The resulting training dataset
                will be a single file per split. Default False.
            seed: Optionally, define a seed to create the random splits with, in order
                to guarantee reproducability, defaults to `None`.
            statistics_config: A configuration object, or a dictionary with keys
                "`enabled`" to generally enable descriptive statistics computation for
                this feature group, `"correlations`" to turn on feature correlation
                computation and `"histograms"` to compute feature value frequencies. The
                values should be booleans indicating the setting. To fully turn off
                statistics computation pass `statistics_config=False`. Defaults to
                `None` and will compute only descriptive statistics.
            write_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                When using the `python` engine, write_options can contain the
                following entries:
                * key `spark` and value an object of type
                [hsfs.core.job_configuration.JobConfiguration](../job_configuration)
                  to configure the Hopsworks Job used to compute the training dataset.
                * key `wait_for_job` and value `True` or `False` to configure
                  whether or not to the save call should return only
                  after the Hopsworks Job has finished. By default it waits.
                Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            (td_version, `Job`): Tuple of training dataset version and job.
                When using the `python` engine, it returns the Hopsworks Job
                that was launched to create the training dataset.
        """

        self._validate_train_test_split(
            test_size=test_size, train_end=train_end, test_start=test_start
        )
        td = training_dataset.TrainingDataset(
            name=self.name,
            version=None,
            test_size=test_size,
            time_split_size=2,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            description=description,
            data_format=data_format,
            storage_connector=storage_connector,
            location=location,
            featurestore_id=self._featurestore_id,
            splits={},
            seed=seed,
            statistics_config=statistics_config,
            coalesce=coalesce,
            extra_filter=extra_filter,
        )
        # td_job is used only if the python engine is used
        td, td_job = self._feature_view_engine.create_training_dataset(
            self, td, write_options, spine
        )
        warnings.warn(
            "Incremented version to `{}`.".format(td.version),
            util.VersionWarning,
        )

        return td.version, td_job

    def create_train_validation_test_split(
        self,
        validation_size: Optional[float] = None,
        test_size: Optional[float] = None,
        train_start: Optional[Union[str, int, datetime, date]] = "",
        train_end: Optional[Union[str, int, datetime, date]] = "",
        validation_start: Optional[Union[str, int, datetime, date]] = "",
        validation_end: Optional[Union[str, int, datetime, date]] = "",
        test_start: Optional[Union[str, int, datetime, date]] = "",
        test_end: Optional[Union[str, int, datetime, date]] = "",
        storage_connector: Optional[storage_connector.StorageConnector] = None,
        location: Optional[str] = "",
        description: Optional[str] = "",
        extra_filter: Optional[Union[filter.Filter, filter.Logic]] = None,
        data_format: Optional[str] = "parquet",
        coalesce: Optional[bool] = False,
        seed: Optional[int] = None,
        statistics_config: Optional[Union[StatisticsConfig, bool, dict]] = None,
        write_options: Optional[Dict[Any, Any]] = {},
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """Create the metadata for a training dataset and save the corresponding training data into `location`.
        The training data is split into train, validation, and test set at random or according to time range.
        The training data can be retrieved by calling `feature_view.get_train_validation_test_split`.

        !!! example "Create random splits"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # create a train-validation-test split dataset
            version, job = feature_view.create_train_validation_test_split(
                validation_size=0.3,
                test_size=0.2,
                description='Description of a dataset',
                data_format='csv'
            )
            ```

        !!! example "Create time series splits by specifying date as string"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            train_start = "2022-01-01 00:00:00"
            train_end = "2022-06-01 23:59:59"
            validation_start = "2022-06-02 00:00:00"
            validation_end = "2022-07-01 23:59:59"
            test_start = "2022-07-02 00:00:00"
            test_end = "2022-08-01 23:59:59"

            # create a train-validation-test split dataset
            version, job = feature_view.create_train_validation_test_split(
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
                description='Description of a dataset',
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format='csv'
            )
            ```

        !!! example "Create time series splits by specifying date as datetime object"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            from datetime import datetime
            date_format = "%Y-%m-%d %H:%M:%S"

            train_start = datetime.strptime("2022-01-01 00:00:00", date_format)
            train_end = datetime.strptime("2022-06-06 23:59:59", date_format)
            validation_start = datetime.strptime("2022-06-02 00:00:00", date_format)
            validation_end = datetime.strptime("2022-07-01 23:59:59", date_format)
            test_start = datetime.strptime("2022-06-07 00:00:00", date_format)
            test_end = datetime.strptime("2022-12-25 23:59:59", date_format)

            # create a train-validation-test split dataset
            version, job = feature_view.create_train_validation_test_split(
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
                description='Description of a dataset',
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format='csv'
            )
            ```

        !!! example "Write training dataset to external storage"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get storage connector instance
            external_storage_connector = fs.get_storage_connector("storage_connector_name")

            # create a train-validation-test split dataset
            version, job = feature_view.create_train_validation_test_split(
                train_start=...,
                train_end=...,
                validation_start=...,
                validation_end=...,
                test_start=...,
                test_end=...,
                description=...,
                storage_connector = external_storage_connector,
                # you can have different data formats such as csv, tsv, tfrecord, parquet and others
                data_format=...
            )
            ```

        !!! info "Data Formats"
            The feature store currently supports the following data formats for
            training datasets:

            1. tfrecord
            2. csv
            3. tsv
            4. parquet
            5. avro
            6. orc

            Currently not supported petastorm, hdf5 and npy file formats.

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            validation_size: size of validation set.
            test_size: size of test set.
            train_start: Start event time for the train split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            train_end: End event time for the train split query, exclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            validation_start: Start event time for the validation split query, inclusive. Strings
                should be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            validation_end: End event time for the validation split query, exclusive. Strings
                should be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_start: Start event time for the test split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_end: End event time for the test split query, exclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            storage_connector: Storage connector defining the sink location for the
                training dataset, defaults to `None`, and materializes training dataset
                on HopsFS.
            location: Path to complement the sink storage connector with, e.g if the
                storage connector points to an S3 bucket, this path can be used to
                define a sub-directory inside the bucket to place the training dataset.
                Defaults to `""`, saving the training dataset at the root defined by the
                storage connector.
            description: A string describing the contents of the training dataset to
                improve discoverability for Data Scientists, defaults to empty string
                `""`.
            extra_filter: Additional filters to be attached to the training dataset.
                The filters will be also applied in `get_batch_data`.
            data_format: The data format used to save the training dataset,
                defaults to `"parquet"`-format.
            coalesce: If true the training dataset data will be coalesced into
                a single partition before writing. The resulting training dataset
                will be a single file per split. Default False.
            seed: Optionally, define a seed to create the random splits with, in order
                to guarantee reproducability, defaults to `None`.
            statistics_config: A configuration object, or a dictionary with keys
                "`enabled`" to generally enable descriptive statistics computation for
                this feature group, `"correlations`" to turn on feature correlation
                computation and `"histograms"` to compute feature value frequencies. The
                values should be booleans indicating the setting. To fully turn off
                statistics computation pass `statistics_config=False`. Defaults to
                `None` and will compute only descriptive statistics.
            write_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                When using the `python` engine, write_options can contain the
                following entries:
                * key `spark` and value an object of type
                [hsfs.core.job_configuration.JobConfiguration](../job_configuration)
                  to configure the Hopsworks Job used to compute the training dataset.
                * key `wait_for_job` and value `True` or `False` to configure
                  whether or not to the save call should return only
                  after the Hopsworks Job has finished. By default it waits.
                Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            (td_version, `Job`): Tuple of training dataset version and job.
                When using the `python` engine, it returns the Hopsworks Job
                that was launched to create the training dataset.
        """

        self._validate_train_validation_test_split(
            validation_size=validation_size,
            test_size=test_size,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
        )
        td = training_dataset.TrainingDataset(
            name=self.name,
            version=None,
            validation_size=validation_size,
            test_size=test_size,
            time_split_size=3,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
            description=description,
            data_format=data_format,
            storage_connector=storage_connector,
            location=location,
            featurestore_id=self._featurestore_id,
            splits={},
            seed=seed,
            statistics_config=statistics_config,
            coalesce=coalesce,
            extra_filter=extra_filter,
        )
        # td_job is used only if the python engine is used
        td, td_job = self._feature_view_engine.create_training_dataset(
            self, td, write_options, spine
        )
        warnings.warn(
            "Incremented version to `{}`.".format(td.version),
            util.VersionWarning,
        )

        return td.version, td_job

    def recreate_training_dataset(
        self,
        training_dataset_version: int,
        write_options: Optional[Dict[Any, Any]] = None,
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """
        Recreate a training dataset.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # recreate a training dataset that has been deleted
            feature_view.recreate_training_dataset(training_dataset_version=1)
            ```

        !!! info
            If a materialised training data has deleted. Use `recreate_training_dataset()` to
            recreate the training data.

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            training_dataset_version: training dataset version
            read_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                When using the `python` engine, write_options can contain the
                following entries:
                * key `spark` and value an object of type
                [hsfs.core.job_configuration.JobConfiguration](../job_configuration)
                  to configure the Hopsworks Job used to compute the training dataset.
                * key `wait_for_job` and value `True` or `False` to configure
                  whether or not to the save call should return only
                  after the Hopsworks Job has finished. By default it waits.
                Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            `Job`: When using the `python` engine, it returns the Hopsworks Job
                that was launched to create the training dataset.
        """
        td, td_job = self._feature_view_engine.recreate_training_dataset(
            self, training_dataset_version, write_options, spine
        )
        return td_job

    def training_data(
        self,
        start_time: Optional[Union[str, int, datetime, date]] = None,
        end_time: Optional[Union[str, int, datetime, date]] = None,
        description: Optional[str] = "",
        extra_filter: Optional[Union[filter.Filter, filter.Logic]] = None,
        statistics_config: Optional[Union[StatisticsConfig, bool, dict]] = None,
        read_options: Optional[Dict[Any, Any]] = None,
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """
        Create the metadata for a training dataset and get the corresponding training data from the offline feature store.
        This returns the training data in memory and does not materialise data in storage.
        The training data can be recreated by calling `feature_view.get_training_data` with the metadata created.

        !!! example "Create random splits"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get training data
            features_df, labels_df  = feature_view.training_data(
                description='Descriprion of a dataset',
            )
            ```

        !!! example "Create time-series based splits"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up a date
            start_time = "2022-05-01 00:00:00"
            end_time = "2022-06-04 23:59:59"
            # you can also pass dates as datetime objects

            # get training data
            features_df, labels_df = feature_view.training_data(
                start_time=start_time,
                end_time=end_time,
                description='Description of a dataset'
            )
            ```

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            start_time: Start event time for the training dataset query, inclusive. Strings should
            be formatted in one of the following
                formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            end_time: End event time for the training dataset query, exclusive. Strings should be
            formatted in one of the following
                formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            description: A string describing the contents of the training dataset to
                improve discoverability for Data Scientists, defaults to empty string
                `""`.
            extra_filter: Additional filters to be attached to the training dataset.
                The filters will be also applied in `get_batch_data`.
            statistics_config: A configuration object, or a dictionary with keys
                "`enabled`" to generally enable descriptive statistics computation for
                this feature group, `"correlations`" to turn on feature correlation
                computation and `"histograms"` to compute feature value frequencies. The
                values should be booleans indicating the setting. To fully turn off
                statistics computation pass `statistics_config=False`. Defaults to
                `None` and will compute only descriptive statistics.
            read_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                When using the `python` engine, write_options can contain the
                following entries:
                * key `"hive_config"` to pass a dictionary of hive or tez configurations.
                  For example: `{"hive_config": {"hive.tez.cpu.vcores": 2, "tez.grouping.split-count": "3"}}`
                * key `spark` and value an object of type
                [hsfs.core.job_configuration.JobConfiguration](../job_configuration)
                  to configure the Hopsworks Job used to compute the training dataset.
                Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            (X, y): Tuple of dataframe of features and labels. If there are no labels, y returns `None`.
        """
        td = training_dataset.TrainingDataset(
            name=self.name,
            version=None,
            splits={},
            event_start_time=start_time,
            event_end_time=end_time,
            description=description,
            storage_connector=None,
            featurestore_id=self._featurestore_id,
            data_format="tsv",
            location="",
            statistics_config=statistics_config,
            training_dataset_type=training_dataset.TrainingDataset.IN_MEMORY,
            extra_filter=extra_filter,
        )
        td, df = self._feature_view_engine.get_training_data(
            self, read_options, training_dataset_obj=td, spine=spine
        )
        warnings.warn(
            "Incremented version to `{}`.".format(td.version),
            util.VersionWarning,
        )
        return df

    def train_test_split(
        self,
        test_size: Optional[float] = None,
        train_start: Optional[Union[str, int, datetime, date]] = "",
        train_end: Optional[Union[str, int, datetime, date]] = "",
        test_start: Optional[Union[str, int, datetime, date]] = "",
        test_end: Optional[Union[str, int, datetime, date]] = "",
        description: Optional[str] = "",
        extra_filter: Optional[Union[filter.Filter, filter.Logic]] = None,
        statistics_config: Optional[Union[StatisticsConfig, bool, dict]] = None,
        read_options: Optional[Dict[Any, Any]] = None,
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """
        Create the metadata for a training dataset and get the corresponding training data from the offline feature store.
        This returns the training data in memory and does not materialise data in storage.
        The training data is split into train and test set at random or according to time ranges.
        The training data can be recreated by calling `feature_view.get_train_test_split` with the metadata created.

        !!! example "Create random train/test splits"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get training data
            X_train, X_test, y_train, y_test = feature_view.train_test_split(
                test_size=0.2
            )
            ```

        !!! example "Create time-series train/test splits"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            train_start = "2022-05-01 00:00:00"
            train_end = "2022-06-04 23:59:59"
            test_start = "2022-07-01 00:00:00"
            test_end= "2022-08-04 23:59:59"
            # you can also pass dates as datetime objects

            # get training data
            X_train, X_test, y_train, y_test = feature_view.train_test_split(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                description='Description of a dataset'
            )
            ```

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            test_size: size of test set. Should be between 0 and 1.
            train_start: Start event time for the train split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`.
            train_end: End event time for the train split query, exclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_start: Start event time for the test split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_end: End event time for the test split query, exclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            description: A string describing the contents of the training dataset to
                improve discoverability for Data Scientists, defaults to empty string
                `""`.
            extra_filter: Additional filters to be attached to the training dataset.
                The filters will be also applied in `get_batch_data`.
            statistics_config: A configuration object, or a dictionary with keys
                "`enabled`" to generally enable descriptive statistics computation for
                this feature group, `"correlations`" to turn on feature correlation
                computation and `"histograms"` to compute feature value frequencies. The
                values should be booleans indicating the setting. To fully turn off
                statistics computation pass `statistics_config=False`. Defaults to
                `None` and will compute only descriptive statistics.
            read_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                When using the `python` engine, write_options can contain the
                following entries:
                * key `"hive_config"` to pass a dictionary of hive or tez configurations.
                  For example: `{"hive_config": {"hive.tez.cpu.vcores": 2, "tez.grouping.split-count": "3"}}`
                * key `spark` and value an object of type
                [hsfs.core.job_configuration.JobConfiguration](../job_configuration)
                  to configure the Hopsworks Job used to compute the training dataset.
                Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            (X_train, X_test, y_train, y_test):
                Tuple of dataframe of features and labels
        """
        self._validate_train_test_split(
            test_size=test_size, train_end=train_end, test_start=test_start
        )
        td = training_dataset.TrainingDataset(
            name=self.name,
            version=None,
            splits={},
            test_size=test_size,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            time_split_size=2,
            description=description,
            storage_connector=None,
            featurestore_id=self._featurestore_id,
            data_format="tsv",
            location="",
            statistics_config=statistics_config,
            training_dataset_type=training_dataset.TrainingDataset.IN_MEMORY,
            extra_filter=extra_filter,
        )
        td, df = self._feature_view_engine.get_training_data(
            self,
            read_options,
            training_dataset_obj=td,
            splits=[TrainingDatasetSplit.TRAIN, TrainingDatasetSplit.TEST],
            spine=spine,
        )
        warnings.warn(
            "Incremented version to `{}`.".format(td.version),
            util.VersionWarning,
        )
        return df

    @staticmethod
    def _validate_train_test_split(test_size, train_end, test_start):
        if not ((test_size and 0 < test_size < 1) or (train_end or test_start)):
            raise ValueError(
                "Invalid split input."
                " You should specify either `test_size` or (`train_end` or `test_start`)."
                " `test_size` should be between 0 and 1 if specified."
            )

    def train_validation_test_split(
        self,
        validation_size: Optional[float] = None,
        test_size: Optional[float] = None,
        train_start: Optional[Union[str, int, datetime, date]] = "",
        train_end: Optional[Union[str, int, datetime, date]] = "",
        validation_start: Optional[Union[str, int, datetime, date]] = "",
        validation_end: Optional[Union[str, int, datetime, date]] = "",
        test_start: Optional[Union[str, int, datetime, date]] = "",
        test_end: Optional[Union[str, int, datetime, date]] = "",
        description: Optional[str] = "",
        extra_filter: Optional[Union[filter.Filter, filter.Logic]] = None,
        statistics_config: Optional[Union[StatisticsConfig, bool, dict]] = None,
        read_options: Optional[Dict[Any, Any]] = None,
        spine: Optional[
            Union[
                pd.DataFrame,
                TypeVar("pyspark.sql.DataFrame"),  # noqa: F821
                TypeVar("pyspark.RDD"),  # noqa: F821
                np.ndarray,
                List[list],
                TypeVar("SpineGroup"),
            ]
        ] = None,
    ):
        """
        Create the metadata for a training dataset and get the corresponding training data from the offline feature store.
        This returns the training data in memory and does not materialise data in storage.
        The training data is split into train, validation, and test set at random or according to time ranges.
        The training data can be recreated by calling `feature_view.get_train_validation_test_split` with the metadata created.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get training data
            X_train, X_val, X_test, y_train, y_val, y_test = feature_view.train_validation_test_split(
                validation_size=0.3,
                test_size=0.2
            )
            ```

        !!! example "Time Series split"
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # set up dates
            start_time_train = '2017-01-01 00:00:01'
            end_time_train = '2018-02-01 23:59:59'

            start_time_val = '2018-02-02 23:59:59'
            end_time_val = '2019-02-01 23:59:59'

            start_time_test = '2019-02-02 23:59:59'
            end_time_test = '2020-02-01 23:59:59'
            # you can also pass dates as datetime objects

            # get training data
            X_train, X_val, X_test, y_train, y_val, y_test = feature_view.train_validation_test_split(
                train_start=start_time_train,
                train_end=end_time_train,
                validation_start=start_time_val,
                validation_end=end_time_val,
                test_start=start_time_test,
                test_end=end_time_test
            )
            ```

        !!! warning "Spine Groups/Dataframes"
            Spine groups and dataframes are currently only supported with the Spark engine and
            Spark dataframes.

        # Arguments
            validation_size: size of validation set. Should be between 0 and 1.
            test_size: size of test set. Should be between 0 and 1.
            train_start: Start event time for the train split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            train_end: End event time for the train split query, exclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            validation_start: Start event time for the validation split query, inclusive. Strings
                should be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            validation_end: End event time for the validation split query, exclusive. Strings
                should be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_start: Start event time for the test split query, inclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            test_end: End event time for the test split query, exclusive. Strings should
                be formatted in one of the following formats `%Y-%m-%d`, `%Y-%m-%d %H`, `%Y-%m-%d %H:%M`, `%Y-%m-%d %H:%M:%S`,
                or `%Y-%m-%d %H:%M:%S.%f`. Int, i.e Unix Epoch should be in seconds.
            description: A string describing the contents of the training dataset to
                improve discoverability for Data Scientists, defaults to empty string
                `""`.
            extra_filter: Additional filters to be attached to the training dataset.
                The filters will be also applied in `get_batch_data`.
            statistics_config: A configuration object, or a dictionary with keys
                "`enabled`" to generally enable descriptive statistics computation for
                this feature group, `"correlations`" to turn on feature correlation
                computation and `"histograms"` to compute feature value frequencies. The
                values should be booleans indicating the setting. To fully turn off
                statistics computation pass `statistics_config=False`. Defaults to
                `None` and will compute only descriptive statistics.
            read_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                When using the `python` engine, write_options can contain the
                following entries:
                * key `"hive_config"` to pass a dictionary of hive or tez configurations.
                  For example: `{"hive_config": {"hive.tez.cpu.vcores": 2, "tez.grouping.split-count": "3"}}`
                * key `spark` and value an object of type
                [hsfs.core.job_configuration.JobConfiguration](../job_configuration)
                  to configure the Hopsworks Job used to compute the training dataset.
                Defaults to `{}`.
            spine: Spine dataframe with primary key, event time and
                label column to use for point in time join when fetching features. Defaults to `None` and is only required
                when feature view was created with spine group in the feature query.
                It is possible to directly pass a spine group instead of a dataframe to overwrite the left side of the
                feature join, however, the same features as in the original feature group that is being replaced need to
                be available in the spine group.

        # Returns
            (X_train, X_val, X_test, y_train, y_val, y_test):
                Tuple of dataframe of features and labels
        """

        self._validate_train_validation_test_split(
            validation_size=validation_size,
            test_size=test_size,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
        )
        td = training_dataset.TrainingDataset(
            name=self.name,
            version=None,
            splits={},
            validation_size=validation_size,
            test_size=test_size,
            time_split_size=3,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
            description=description,
            storage_connector=None,
            featurestore_id=self._featurestore_id,
            data_format="tsv",
            location="",
            statistics_config=statistics_config,
            training_dataset_type=training_dataset.TrainingDataset.IN_MEMORY,
            extra_filter=extra_filter,
        )
        td, df = self._feature_view_engine.get_training_data(
            self,
            read_options,
            training_dataset_obj=td,
            splits=[
                TrainingDatasetSplit.TRAIN,
                TrainingDatasetSplit.VALIDATION,
                TrainingDatasetSplit.TEST,
            ],
            spine=spine,
        )
        warnings.warn(
            "Incremented version to `{}`.".format(td.version),
            util.VersionWarning,
        )
        return df

    @staticmethod
    def _validate_train_validation_test_split(
        validation_size,
        test_size,
        train_end,
        validation_start,
        validation_end,
        test_start,
    ):
        if not (
            (validation_size and 0 < validation_size < 1)
            and (test_size and 0 < test_size < 1)
            and (validation_size + test_size < 1)
            or ((train_end or validation_start) and (validation_end or test_start))
        ):
            raise ValueError(
                "Invalid split input."
                " You should specify either (`validation_size` and `test_size`) or ((`train_end` or `validation_start`) and (`validation_end` or `test_start`))."
                "`validation_size`, `test_size` and sum of `validationSize` and `testSize` should be between 0 and 1 if specified."
            )

    def get_training_data(
        self,
        training_dataset_version,
        read_options: Optional[Dict[Any, Any]] = None,
    ):
        """
        Get training data created by `feature_view.create_training_data`
        or `feature_view.training_data`.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get training data
            features_df, labels_df = feature_view.get_training_data(training_dataset_version=1)
            ```

        !!! warning "External Storage Support"
            Reading training data that was written to external storage using a Storage
            Connector other than S3 can currently not be read using HSFS APIs with
            Python as Engine, instead you will have to use the storage's native client.

        # Arguments
            training_dataset_version: training dataset version
            read_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                For python engine:
                * key `"hive_config"` to pass a dictionary of hive or tez configurations.
                  For example: `{"hive_config": {"hive.tez.cpu.vcores": 2, "tez.grouping.split-count": "3"}}`
                Defaults to `{}`.

        # Returns
            (X, y): Tuple of dataframe of features and labels
        """
        td, df = self._feature_view_engine.get_training_data(
            self, read_options, training_dataset_version=training_dataset_version
        )
        return df

    def get_train_test_split(
        self,
        training_dataset_version,
        read_options: Optional[Dict[Any, Any]] = None,
    ):
        """
        Get training data created by `feature_view.create_train_test_split`
        or `feature_view.train_test_split`.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get training data
            X_train, X_test, y_train, y_test = feature_view.get_train_test_split(training_dataset_version=1)
            ```

        # Arguments
            training_dataset_version: training dataset version
            read_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                For python engine:
                * key `"hive_config"` to pass a dictionary of hive or tez configurations.
                  For example: `{"hive_config": {"hive.tez.cpu.vcores": 2, "tez.grouping.split-count": "3"}}`
                Defaults to `{}`.

        # Returns
            (X_train, X_test, y_train, y_test):
                Tuple of dataframe of features and labels
        """
        td, df = self._feature_view_engine.get_training_data(
            self,
            read_options,
            training_dataset_version=training_dataset_version,
            splits=[TrainingDatasetSplit.TRAIN, TrainingDatasetSplit.TEST],
        )
        return df

    def get_train_validation_test_split(
        self,
        training_dataset_version,
        read_options: Optional[Dict[Any, Any]] = None,
    ):
        """
        Get training data created by `feature_view.create_train_validation_test_split`
        or `feature_view.train_validation_test_split`.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get training data
            X_train, X_val, X_test, y_train, y_val, y_test = feature_view.get_train_validation_test_splits(training_dataset_version=1)
            ```

        # Arguments
            training_dataset_version: training dataset version
            read_options: Additional options as key/value pairs to pass to the execution engine.
                For spark engine: Dictionary of read options for Spark.
                For python engine:
                * key `"hive_config"` to pass a dictionary of hive or tez configurations.
                  For example: `{"hive_config": {"hive.tez.cpu.vcores": 2, "tez.grouping.split-count": "3"}}`
                Defaults to `{}`.

        # Returns
            (X_train, X_val, X_test, y_train, y_val, y_test):
                Tuple of dataframe of features and labels
        """
        td, df = self._feature_view_engine.get_training_data(
            self,
            read_options,
            training_dataset_version=training_dataset_version,
            splits=[
                TrainingDatasetSplit.TRAIN,
                TrainingDatasetSplit.VALIDATION,
                TrainingDatasetSplit.TEST,
            ],
        )
        return df

    def add_training_dataset_tag(self, training_dataset_version: int, name: str, value):
        """Attach a tag to a training dataset.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature feature view instance
            feature_view = fs.get_feature_view(...)

            # attach a tag to a training dataset
            feature_view.add_training_dataset_tag(
                training_dataset_version=1,
                name="tag_schema",
                value={"key", "value"}
            )
            ```

        # Arguments
            training_dataset_version: training dataset version
            name: Name of the tag to be added.
            value: Value of the tag to be added.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to add the tag.
        """
        return self._feature_view_engine.add_tag(
            self, name, value, training_dataset_version=training_dataset_version
        )

    def get_training_dataset_tag(self, training_dataset_version: int, name: str):
        """Get the tags of a training dataset.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get a training dataset tag
            tag_str = feature_view.get_training_dataset_tag(
                training_dataset_version=1,
                 name="tag_schema"
            )
            ```

        # Arguments
            training_dataset_version: training dataset version
            name: Name of the tag to get.

        # Returns
            tag value

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to retrieve the tag.
        """
        return self._feature_view_engine.get_tag(
            self, name, training_dataset_version=training_dataset_version
        )

    def get_training_dataset_tags(self, training_dataset_version: int):
        """Returns all tags attached to a training dataset.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # get a training dataset tags
            list_tags = feature_view.get_training_dataset_tags(
                training_dataset_version=1
            )
            ```

        # Returns
            `Dict[str, obj]` of tags.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to retrieve the tags.
        """
        return self._feature_view_engine.get_tags(
            self, training_dataset_version=training_dataset_version
        )

    def delete_training_dataset_tag(self, training_dataset_version: int, name: str):
        """Delete a tag attached to a training dataset.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # delete training dataset tag
            feature_view.delete_training_dataset_tag(
                training_dataset_version=1,
                name='name_of_dataset'
            )
            ```

        # Arguments
            training_dataset_version: training dataset version
            name: Name of the tag to be removed.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to delete the tag.
        """
        return self._feature_view_engine.delete_tag(
            self, name, training_dataset_version=training_dataset_version
        )

    def purge_training_data(self, training_dataset_version: int):
        """Delete a training dataset (data only).

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # purge training data
            feature_view.purge_training_data(training_dataset_version=1)
            ```

        # Arguments
            training_dataset_version: Version of the training dataset to be removed.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to delete the training dataset.
        """
        self._feature_view_engine.delete_training_dataset_only(
            self, training_data_version=training_dataset_version
        )

    def purge_all_training_data(self):
        """Delete all training datasets (data only).

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # purge all training data
            feature_view.purge_all_training_data()
            ```

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to delete the training datasets.
        """
        self._feature_view_engine.delete_training_dataset_only(self)

    def delete_training_dataset(self, training_dataset_version: int):
        """Delete a training dataset. This will delete both metadata and training data.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # delete a training dataset
            feature_view.delete_training_dataset(
                training_dataset_version=1
            )
            ```

        # Arguments
            training_dataset_version: Version of the training dataset to be removed.

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to delete the training dataset.
        """
        self._feature_view_engine.delete_training_data(
            self, training_data_version=training_dataset_version
        )

    def delete_all_training_datasets(self):
        """Delete all training datasets. This will delete both metadata and training data.

        !!! example
            ```python
            # get feature store instance
            fs = ...

            # get feature view instance
            feature_view = fs.get_feature_view(...)

            # delete all training datasets
            feature_view.delete_all_training_datasets()
            ```

        # Raises
            `hsfs.client.exceptions.RestAPIError` in case the backend fails to delete the training datasets.
        """
        self._feature_view_engine.delete_training_data(self)

    def _get_feature_monitoring_configs(
        self,
        name: Optional[str] = None,
        feature_name: Optional[str] = None,
        config_id: Optional[int] = None,
    ) -> Union[
        "fmc.FeatureMonitoringConfig", List["fmc.FeatureMonitoringConfig"], None
    ]:
        """Fetch feature monitoring configs attached to the feature view.

        If no arguments is provided the method will return all feature monitoring configs
        attached to the feature view, meaning all feature monitoring configs that are attach
        to a feature in the feature view. If you wish to fetch a single config, provide the
        its name. If you wish to fetch all configs attached to a particular feature, provide
        the feature name.

        !!! example
            ```python3
            # fetch your feature view
            fv = fs.get_feature_view(name="my_feature_view", version=1)

            # fetch all feature monitoring configs attached to the feature view
            fm_configs = fv._get_feature_monitoring_configs()

            # fetch a single feature monitoring config by name
            fm_config = fv._get_feature_monitoring_configs(name="my_config")

            # fetch all feature monitoring configs attached to a particular feature
            fm_configs = fv._get_feature_monitoring_configs(feature_name="my_feature")

            # fetch a single feature monitoring config with a particular id
            fm_config = fv._get_feature_monitoring_configs(config_id=1)
            ```

        # Arguments
            name: If provided fetch only the feature monitoring config with the given name.
                Defaults to None.
            feature_name: If provided, fetch only configs attached to a particular feature.
                Defaults to None.
            config_id: If provided, fetch only the feature monitoring config with the given id.
                Defaults to None.

        # Raises
            `hsfs.client.exceptions.RestAPIError`.
            `hsfs.client.exceptions.FeatureStoreException`.
            ValueError: if both name and feature_name are provided.
            TypeError: if name or feature_name are not string or None.

        # Return
            Union[`FeatureMonitoringConfig`, List[`FeatureMonitoringConfig`], None]
                A list of feature monitoring configs. If name provided,
                returns either a single config or None if not found.
        """
        # TODO: Should this filter out scheduled statistics only configs?
        if not self._id:
            raise FeatureStoreException(
                "Only Feature Group registered with Hopsworks can fetch feature monitoring configurations."
            )

        return self._fm_config_engine.get_feature_monitoring_configs(
            name=name,
            feature_name=feature_name,
            config_id=config_id,
        )

    def _get_feature_monitoring_history(
        self,
        config_name: Optional[str] = None,
        config_id: Optional[int] = None,
        start_time: Optional[Union[int, str, datetime, date]] = None,
        end_time: Optional[Union[int, str, datetime, date]] = None,
        with_statistics: Optional[bool] = True,
    ) -> List["fmr.FeatureMonitoringResult"]:
        """Fetch feature monitoring history for a given feature monitoring config.

        !!! example
            ```python3
            # fetch your feature view
            fv = fs.get_feature_view(name="my_feature_group", version=1)

            # fetch feature monitoring history for a given feature monitoring config
            fm_history = fv._get_feature_monitoring_history(
                config_name="my_config",
                start_time="2020-01-01",
            )

            # or use the config id
            fm_history = fv._get_feature_monitoring_history(
                config_id=1,
                start_time=datetime.now() - timedelta(weeks=2),
                end_time=datetime.now() - timedelta(weeks=1),
                with_statistics=False,
            )
            ```

        # Arguments
            config_name: The name of the feature monitoring config to fetch history for.
                Defaults to None.
            config_id: The id of the feature monitoring config to fetch history for.
                Defaults to None.
            start_date: The start date of the feature monitoring history to fetch.
                Defaults to None.
            end_date: The end date of the feature monitoring history to fetch.
                Defaults to None.
            with_statistics: Whether to include statistics in the feature monitoring history.
                Defaults to True. If False, only metadata about the monitoring will be fetched.

        # Raises
            `hsfs.client.exceptions.RestAPIError`.
            `hsfs.client.exceptions.FeatureStoreException`.
            ValueError: if both config_name and config_id are provided.
            TypeError: if config_name or config_id are not respectively string, int or None.

        # Return
            List[`FeatureMonitoringResult`]
                A list of feature monitoring results containing the monitoring metadata
                as well as the computed statistics for the detection and reference window
                if requested.
        """
        if not self._id:
            raise FeatureStoreException(
                "Only Feature View registered with Hopsworks can fetch feature monitoring history."
            )

        return self._fm_result_engine.get_feature_monitoring_results(
            config_name=config_name,
            config_id=config_id,
            start_time=start_time,
            end_time=end_time,
            with_statistics=with_statistics,
        )

    def _enable_statistics_monitoring(
        self,
        name: str,
        job_frequency: str = "DAILY",
        feature_name: Optional[str] = None,
        description: Optional[str] = None,
        start_date_time: Optional[Union[int, str, datetime, date, pd.Timestamp]] = None,
        is_event_time: Optional[bool] = True,
        training_dataset_version: Optional[int] = None,
    ) -> "fmc.FeatureMonitoringConfig":
        """Run a job to compute statistics on snapshot of feature data on a schedule.

        !!! experimental
            Public API is subject to change, this feature is not suitable for production use-cases.

        !!! example
            ```python3
            # fetch feature view
            fv = fs.get_feature_view(name="my_feature_view", version=1)

            # enable statistics monitoring
            my_config = fv._enable_statistics_monitoring(
                name="my_config",
                job_frequency="DAILY",
                start_date_time="2021-01-01 00:00:00",
                description="my description",
            ).with_detection_window(
                # Statistics computed on 10% of the last week of data
                time_offset="1w",
                row_percentage=0.1,
            ).save()
            ```

        # Arguments
            name: Name of the feature monitoring configuration.
                name must be unique for all configurations attached to the feature view.
            feature_name: Name of the feature to monitor. If not specified, statistics
                will be computed for all features.
            job_frequency: Frequency at which to compute the statistics for the feature.
                Options are "HOURLY", "DAILY", "WEEKLY", "MONTHLY", defaults to "DAILY".
            description: Description of the feature monitoring configuration.
            start_date_time: Start date and time from which to start computing statistics.
            is_event_time: If true, use event time to compute statistics.
                Defaults to False.
            training_dataset_version: The version of the dataset to use
                to fetch statistics for the transformation function. If provided, the
                statistics are computed after applying the transformation function.
                Otherwise, the statistics are computed on the raw data. Defaults to None.

        # Raises
            `hsfs.client.exceptions.FeatureStoreException`.

        # Return
            `FeatureMonitoringConfig` Configuration with minimal information about the feature monitoring.
                Additional information are required before feature monitoring is enabled.
        """
        if not self._id:
            raise FeatureStoreException(
                "Only Feature Group registered with Hopsworks can enable scheduled statistics monitoring."
            )

        return self._fm_config_engine._build_default_statistics_monitoring_config(
            name=name,
            feature_name=feature_name,
            description=description,
            job_frequency=job_frequency,
            start_date_time=start_date_time,
            is_event_time=is_event_time,
            training_dataset_version=training_dataset_version,
            valid_feature_names=[feat.name for feat in self._features],
        )

    def _enable_feature_monitoring(
        self,
        name: str,
        feature_name: str,
        job_frequency: str = "DAILY",
        description: Optional[str] = None,
        start_date_time: Optional[Union[int, str, datetime, date, pd.Timestamp]] = None,
        is_event_time: Optional[bool] = True,
        training_dataset_version: Optional[int] = None,
    ) -> "fmc.FeatureMonitoringConfig":
        """Enable feature monitoring to compare statistics on snapshots of feature data over time.

        !!! experimental
            Public API is subject to change, this feature is not suitable for production use-cases.

        !!! example
            ```python3
            # fetch feature view
            fg = fs.get_feature_view(name="my_feature_view", version=1)

            # enable feature monitoring
            my_config = fg._enable_feature_monitoring(
                name="my_monitoring_config",
                feature_name="my_feature",
                job_frequency="DAILY",
                description="my monitoring config description",
            ).with_detection_window(
                # Data inserted in the last day
                time_offset="1d",
                window_length="1d",
            ).with_reference_window(
                # compare to a given value
                specific_value=0.5,
            ).compare_on(
                metric="mean",
                threshold=0.5,
            ).save()
            ```

        # Arguments
            name: Name of the feature monitoring configuration.
                name must be unique for all configurations attached to the feature group.
            feature_name: Name of the feature to monitor.
            job_frequency: Frequency at which to compute the statistics for the feature.
                Options are "HOURLY", "DAILY", "WEEKLY", "MONTHLY", defaults to "DAILY".
            description: Description of the feature monitoring configuration.
            start_date_time: Start date and time from which to start computing statistics.
            is_event_time: If true, use event time to compute statistics.
                Defaults to False.
            training_dataset_version: The version of the dataset to use
                to fetch statistics for the transformation function. If provided, the
                statistics are computed after applying the transformation function.
                Otherwise, the statistics are computed on the raw data. Defaults to None

        # Raises
            `hsfs.client.exceptions.FeatureStoreException`.

        # Return
            `FeatureMonitoringConfig` Configuration with minimal information about the feature monitoring.
                Additional information are required before feature monitoring is enabled.
        """
        if not self._id:
            raise FeatureStoreException(
                "Only Feature Group registered with Hopsworks can enable feature monitoring."
            )

        return self._fm_config_engine._build_default_feature_monitoring_config(
            name=name,
            feature_name=feature_name,
            description=description,
            job_frequency=job_frequency,
            start_date_time=start_date_time,
            is_event_time=is_event_time,
            training_dataset_version=training_dataset_version,
            valid_feature_names=[feat.name for feat in self._features],
        )

    @classmethod
    def from_response_json(cls, json_dict):
        json_decamelized = humps.decamelize(json_dict)
        fv = cls(
            id=json_decamelized.get("id", None),
            name=json_decamelized["name"],
            query=query.Query.from_response_json(json_decamelized["query"]),
            featurestore_id=json_decamelized["featurestore_id"],
            version=json_decamelized.get("version", None),
            description=json_decamelized.get("description", None),
            featurestore_name=json_decamelized.get("featurestore_name", None),
        )
        features = json_decamelized.get("features", [])
        if features:
            features = [
                training_dataset_feature.TrainingDatasetFeature.from_response_json(
                    feature
                )
                for feature in features
            ]
        fv.schema = features
        fv.labels = [feature.name for feature in features if feature.label]
        return fv

    def update_from_response_json(self, json_dict):
        other = self.from_response_json(json_dict)
        for key in [
            "name",
            "description",
            "id",
            "query",
            "featurestore_id",
            "version",
            "labels",
            "schema",
        ]:
            self._update_attribute_if_present(self, other, key)
        self._init_feature_monitoring_engine()
        return self

    @staticmethod
    def _update_attribute_if_present(this, new, key):
        if getattr(new, key):
            setattr(this, key, getattr(new, key))

    def _init_feature_monitoring_engine(self):
        self._fm_config_engine = (
            feature_monitoring_config_engine.FeatureMonitoringConfigEngine(
                feature_store_id=self._featurestore_id,
                feature_view_id=self._id,
                feature_view_name=self._name,
                feature_view_version=self._version,
            )
        )
        self._fm_result_engine = (
            feature_monitoring_result_engine.FeatureMonitoringResultEngine(
                feature_store_id=self._featurestore_id,
                feature_view_id=self._id,
                feature_view_name=self._name,
                feature_view_version=self._version,
            )
        )

    def json(self):
        return json.dumps(self, cls=util.FeatureStoreEncoder)

    def to_dict(self):
        return {
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "query": self._query,
            "features": self._features,
            "type": "featureViewDTO",
        }

    @property
    def id(self):
        """Feature view id."""
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def featurestore_id(self):
        """Feature store id."""
        return self._featurestore_id

    @featurestore_id.setter
    def featurestore_id(self, id):
        self._featurestore_id = id

    @property
    def feature_store_name(self):
        """Name of the feature store in which the feature group is located."""
        return self._feature_store_name

    @property
    def name(self):
        """Name of the feature view."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def version(self):
        """Version number of the feature view."""
        return self._version

    @version.setter
    def version(self, version):
        self._version = version

    @property
    def labels(self):
        """The labels/prediction feature of the feature view.

        Can be a composite of multiple features.
        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = [lb.lower() for lb in labels]

    @property
    def description(self):
        """Description of the feature view."""
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @property
    def query(self) -> "query.Query":
        """Query of the feature view."""
        return self._query

    @query.setter
    def query(self, query_obj: "query.Query"):
        self._query = query_obj

    @property
    def transformation_functions(self):
        """Get transformation functions."""
        return self._transformation_functions

    @transformation_functions.setter
    def transformation_functions(self, transformation_functions):
        self._transformation_functions = transformation_functions

    @property
    def schema(self):
        """Feature view schema."""
        return self._features

    @schema.setter
    def schema(self, features):
        self._features = features

    @property
    def primary_keys(self):
        """Set of primary key names that is used as keys in input dict object for `get_serving_vector` method."""
        _vector_server = self._single_vector_server or self._batch_vectors_server
        if _vector_server:
            return _vector_server.serving_keys
        else:
            _vector_server = vector_server.VectorServer(
                self._featurestore_id, self._features
            )
            _vector_server.init_prepared_statement(self, False, False)
            return _vector_server.serving_keys

    @property
    def statistics(
        self,
        start_time,
        end_time,
        is_event_time: Optional[bool] = False,
        feature_name: Optional[str] = None,
        row_percentage: Optional[float] = None,
        computed_at: Optional[float] = None,
    ):
        """Get the computed statistics for the feature view by time window.
        If the window is based on event times, the statistics computation time (i.e., computed_at parameter) is required.

        # Arguments
            start_time: int: Window start time.
            end_time: int: Window end time.
            is_event_time: bool: Whether start and end times are event times or commit times. This parameter is optional. Default value is False.
            feature_name: str: Name of the feature from which statistics where computed. This parameter is optional.
            row_percentage: float: The fraction of rows to use when computing the statistics [0, 1.0]. This parameter is optional.
            computed_at: float. Timestamp or commit time when statistics where computed.

        # Returns
            Statistics: Feature view statistics
        """
        if is_event_time and computed_at is None:
            raise ValueError(
                "Computation time is required for fetching statistics based on event times"
            )

        return self._statistics_engine.get_by_time_window(
            self,
            start_time=start_time,
            end_time=end_time,
            is_event_time=is_event_time,
            feature_name=feature_name,
            row_percentage=row_percentage,
            computed_at=computed_at,
        )
