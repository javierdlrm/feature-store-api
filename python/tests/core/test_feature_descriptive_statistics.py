#
#   Copyright 2023 Hopsworks AB
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

from hsfs.core.feature_descriptive_statistics import FeatureDescriptiveStatistics
from hsfs.core.feature_monitoring_result import FeatureMonitoringResult


class TestFeatureDescriptiveStatistics:
    def test_from_response_json_fractional(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_fractional_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._id == 52
        assert result._feature_name == "amount"
        assert result._feature_type == "Fractional"
        assert result._count == 4
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._min == 1
        assert result._max == 2
        assert result._sum == 3
        assert result._mean == 5.1
        assert result._stddev == 6.1
        assert result._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10
        assert result._extended_statistics is None

    def test_from_response_json_integral(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_integral_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._id == 52
        assert result._feature_name == "age"
        assert result._feature_type == "Integral"
        assert result._count == 4
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._min == 1
        assert result._max == 2
        assert result._sum == 3
        assert result._mean == 5.1
        assert result._stddev == 6.1
        assert result._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10
        assert result._extended_statistics is None

    def test_from_response_json_string(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_string_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._id == 52
        assert result._feature_name == "first_name"
        assert result._feature_type == "String"
        assert result._count == 4
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10
        assert result._extended_statistics is None

    def test_from_response_json_boolean(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_boolean_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._id == 52
        assert result._feature_name == "is_active"
        assert result._feature_type == "Boolean"
        assert result._count == 4
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10
        assert result._extended_statistics is None

    def test_from_response_json_for_empty_data(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_for_empty_data"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._id == 52
        assert result._feature_name == "amount"
        assert result._count == 0
        assert result._feature_type is None
        assert result._completeness is None
        assert result._num_non_null_values is None
        assert result._num_null_values is None
        assert result._approx_num_distinct_values is None
        assert result._min is None
        assert result._max is None
        assert result._sum is None
        assert result._mean is None
        assert result._stddev is None
        assert result._percentiles is None
        assert result._distinctness is None
        assert result._entropy is None
        assert result._uniqueness is None
        assert result._exact_num_distinct_values is None
        assert result._extended_statistics is None

    def test_from_response_for_transformation_functions(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_for_transformation_functions"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._id == 52
        assert result._feature_name == "amount"
        assert result._feature_type is None
        assert result._count is None
        assert result._completeness is None
        assert result._num_non_null_values is None
        assert result._num_null_values is None
        assert result._approx_num_distinct_values is None
        assert result._min is None
        assert result._max is None
        assert result._sum is None
        assert result._mean is None
        assert result._stddev is None
        assert result._percentiles is None
        assert result._distinctness is None
        assert result._entropy is None
        assert result._uniqueness is None
        assert result._exact_num_distinct_values is None
        assert isinstance(result._extended_statistics, dict)
        assert "unique_values" in result._extended_statistics

    def test_from_response_json_with_extended_statistics(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_with_extended_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._id == 52
        assert result._feature_name == "age"
        assert result._feature_type == "Integral"
        assert result._count == 4
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._min == 1
        assert result._max == 2
        assert result._sum == 3
        assert result._mean == 5.1
        assert result._stddev == 6.1
        assert result._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10
        assert isinstance(result._extended_statistics, dict)
        assert "correlations" in result._extended_statistics
        assert "histogram" in result._extended_statistics

    def test_from_deequ_json_fractional(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_deequ_fractional_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_deequ_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._feature_name == "amount"
        assert result._feature_type == "Fractional"
        assert result._count == 15
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._min == 1
        assert result._max == 2
        assert result._sum == 3
        assert result._mean == 5.1
        assert result._stddev == 6.1
        # assert result._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10

    def test_from_deequ_json_integral(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_deequ_integral_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_deequ_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._feature_name == "age"
        assert result._feature_type == "Integral"
        assert result._count == 15
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._min == 1
        assert result._max == 2
        assert result._sum == 3
        assert result._mean == 5.1
        assert result._stddev == 6.1
        # assert result._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10

    def test_from_deequ_json_string(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_deequ_string_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_deequ_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._feature_name == "first_name"
        assert result._feature_type == "String"
        assert result._count == 15
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10

    def test_from_deequ_json_boolean(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_descriptive_statistics"][
            "get_deequ_boolean_feature_statistics"
        ]["response"]

        # Act
        result = FeatureDescriptiveStatistics.from_deequ_json(result_json)

        # Assert
        assert isinstance(result, FeatureDescriptiveStatistics)
        assert result._feature_name == "is_active"
        assert result._feature_type == "Boolean"
        assert result._count == 15
        assert result._completeness == 1
        assert result._num_non_null_values == 7
        assert result._num_null_values == 8
        assert result._approx_num_distinct_values == 9
        assert result._distinctness == 0.9
        assert result._entropy == 0.8
        assert result._uniqueness == 0.7
        assert result._exact_num_distinct_values == 10

    def test_from_response_json_via_fg_fm_result(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_monitoring_result"][
            "get_via_feature_group"
        ]["detection_and_reference_statistics"]["response"]

        # Act
        result = FeatureMonitoringResult.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureMonitoringResult)
        assert result._detection_statistics_id is None
        assert result._reference_statistics_id is None
        assert isinstance(result._detection_statistics, FeatureDescriptiveStatistics)
        assert isinstance(result._reference_statistics, FeatureDescriptiveStatistics)

        det_stats = result._detection_statistics
        assert det_stats._id == 52
        assert result._feature_name == "amount"
        assert det_stats._feature_type == "Fractional"
        assert det_stats._count == 4
        assert det_stats._completeness == 1
        assert det_stats._num_non_null_values == 7
        assert det_stats._num_null_values == 8
        assert det_stats._approx_num_distinct_values == 9
        assert det_stats._min == 1
        assert det_stats._max == 2
        assert det_stats._sum == 3
        assert det_stats._mean == 5.1
        assert det_stats._stddev == 6.1
        # assert det_stats._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert det_stats._distinctness == 0.9
        assert det_stats._entropy == 0.8
        assert det_stats._uniqueness == 0.7
        assert det_stats._exact_num_distinct_values == 10
        assert det_stats._extended_statistics is None

        ref_stats = result._reference_statistics
        assert ref_stats._id == 53
        assert det_stats._feature_name == "amount"
        assert ref_stats._feature_type == "Fractional"
        assert ref_stats._count == 4
        assert ref_stats._completeness == 1
        assert ref_stats._num_non_null_values == 7
        assert ref_stats._num_null_values == 8
        assert ref_stats._approx_num_distinct_values == 9
        assert ref_stats._min == 1
        assert ref_stats._max == 2
        assert ref_stats._sum == 3
        assert ref_stats._mean == 5.1
        assert ref_stats._stddev == 6.1
        # assert ref_stats._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert ref_stats._distinctness == 0.9
        assert ref_stats._entropy == 0.8
        assert ref_stats._uniqueness == 0.7
        assert ref_stats._exact_num_distinct_values == 10
        assert ref_stats._extended_statistics is None

    def test_from_response_json_via_fv_fm_result(self, backend_fixtures):
        # Arrange
        result_json = backend_fixtures["feature_monitoring_result"][
            "get_via_feature_view"
        ]["detection_and_reference_statistics"]["response"]

        # Act
        result = FeatureMonitoringResult.from_response_json(result_json)

        # Assert
        assert isinstance(result, FeatureMonitoringResult)
        assert result._detection_statistics_id is None
        assert result._reference_statistics_id is None
        assert isinstance(result._detection_statistics, FeatureDescriptiveStatistics)
        assert isinstance(result._reference_statistics, FeatureDescriptiveStatistics)

        det_stats = result._detection_statistics
        assert det_stats._id == 52
        assert det_stats._feature_name == "amount"
        assert det_stats._feature_type == "Fractional"
        assert det_stats._count == 4
        assert det_stats._completeness == 1
        assert det_stats._num_non_null_values == 7
        assert det_stats._num_null_values == 8
        assert det_stats._approx_num_distinct_values == 9
        assert det_stats._min == 1
        assert det_stats._max == 2
        assert det_stats._sum == 3
        assert det_stats._mean == 5.1
        assert det_stats._stddev == 6.1
        # assert det_stats._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert det_stats._distinctness == 0.9
        assert det_stats._entropy == 0.8
        assert det_stats._uniqueness == 0.7
        assert det_stats._exact_num_distinct_values == 10
        assert det_stats._extended_statistics is None

        ref_stats = result._reference_statistics
        assert ref_stats._id == 53
        assert det_stats._feature_name == "amount"
        assert ref_stats._feature_type == "Fractional"
        assert ref_stats._count == 4
        assert ref_stats._completeness == 1
        assert ref_stats._num_non_null_values == 7
        assert ref_stats._num_null_values == 8
        assert ref_stats._approx_num_distinct_values == 9
        assert ref_stats._min == 1
        assert ref_stats._max == 2
        assert ref_stats._sum == 3
        assert ref_stats._mean == 5.1
        assert ref_stats._stddev == 6.1
        # assert ref_stats._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert ref_stats._distinctness == 0.9
        assert ref_stats._entropy == 0.8
        assert ref_stats._uniqueness == 0.7
        assert ref_stats._exact_num_distinct_values == 10
        assert ref_stats._extended_statistics is None

    def test_from_response_json_via_fm_result_list_with_statistics(
        self, backend_fixtures
    ):
        # Arrange
        result_json = backend_fixtures["feature_monitoring_result"][
            "get_list_with_statistics"
        ]["detection_and_reference_statistics"]["response"]

        # Act
        result_list = FeatureMonitoringResult.from_response_json(result_json)
        result = result_list[0]

        # Assert
        assert isinstance(result_list, list)
        assert len(result_list) == 1
        assert isinstance(result, FeatureMonitoringResult)
        assert result._detection_statistics_id is None
        assert result._reference_statistics_id is None
        assert isinstance(result._detection_statistics, FeatureDescriptiveStatistics)
        assert isinstance(result._reference_statistics, FeatureDescriptiveStatistics)

        det_stats = result._detection_statistics
        assert det_stats._id == 52
        assert det_stats._feature_name == "amount"
        assert det_stats._feature_type == "Fractional"
        assert det_stats._count == 4
        assert det_stats._completeness == 1
        assert det_stats._num_non_null_values == 7
        assert det_stats._num_null_values == 8
        assert det_stats._approx_num_distinct_values == 9
        assert det_stats._min == 1
        assert det_stats._max == 2
        assert det_stats._sum == 3
        assert det_stats._mean == 5.1
        assert det_stats._stddev == 6.1
        # assert det_stats._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert det_stats._distinctness == 0.9
        assert det_stats._entropy == 0.8
        assert det_stats._uniqueness == 0.7
        assert det_stats._exact_num_distinct_values == 10
        assert det_stats._extended_statistics is None

        ref_stats = result._reference_statistics
        assert ref_stats._id == 53
        assert det_stats._feature_name == "amount"
        assert ref_stats._feature_type == "Fractional"
        assert ref_stats._count == 4
        assert ref_stats._completeness == 1
        assert ref_stats._num_non_null_values == 7
        assert ref_stats._num_null_values == 8
        assert ref_stats._approx_num_distinct_values == 9
        assert ref_stats._min == 1
        assert ref_stats._max == 2
        assert ref_stats._sum == 3
        assert ref_stats._mean == 5.1
        assert ref_stats._stddev == 6.1
        # assert ref_stats._percentiles == {"25%": 0.4, "50%": 0.6, "75%": 0.86}  # TODO: Parse deequ approxPercentiles
        assert ref_stats._distinctness == 0.9
        assert ref_stats._entropy == 0.8
        assert ref_stats._uniqueness == 0.7
        assert ref_stats._exact_num_distinct_values == 10
        assert ref_stats._extended_statistics is None
