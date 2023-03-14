# from datasketches import frequent_strings_sketch, frequent_items_error_type
# from datasketches import hll_sketch, hll_union, HLL_4
# from datasketches import kll_floats_sketch

# from pyspark.sql.types import (
#     BooleanType,
#     ByteType,
#     ShortType,
#     IntegerType,
#     LongType,
#     FloatType,
#     DoubleType,
#     DecimalType,
#     TimestampType,
#     DateType,
#     StringType,
# )

# import numpy as np

# from math import sqrt


# class DatasketchesStatistics:
#     def __init__(
#         self,
#         column_names,
#         data_types,
#         approx_distinct,
#         freq_items,
#         histograms,
#         mean,
#         variance,
#         count,
#         weight_sum,
#         num_non_zeros,
#         max_,
#         min_,
#         norm_l2,
#         norm_l1,
#         percentiles,
#     ):
#         self.column_names = column_names
#         self.data_types = data_types

#         self.approx_distinct = approx_distinct
#         self.freq_items = freq_items
#         self.histograms = histograms

#         self.mean = mean
#         self.variance = variance
#         self.count = count
#         self.weight_sum = weight_sum
#         self.num_non_zeros = num_non_zeros
#         self.max_ = max_
#         self.min_ = min_
#         self.norm_l2 = norm_l2
#         self.norm_l1 = norm_l1
#         self.percentiles = percentiles

#     def to_dict(self):
#         return {
#             "column_names": self.column_names,
#             "data_types": self.data_types,
#             "approx_distinct": self.approx_distinct,
#             "frequent_items": self.freq_items,
#             "histograms": self.histograms,
#             "mean": self.mean,
#             "variance": self.variance,
#             "count": self.count,
#             "weight_sum": self.weight_sum,
#             "num_non_zeros": self.num_non_zeros,
#             "max": self.max_,
#             "min": self.min_,
#             "norm L2": self.norm_l2,
#             "norm L1": self.norm_l1,
#             "percentiles": self.percentiles,
#         }


# class DatasketchesConfig:
#     def __init__(
#         self,
#         approx_distinctness=True,
#         frequent_items=True,
#         approx_quantiles=True,
#         freq_item_max=253,  # physical size of the internal hash map managed by this sketch and must be a power of 2
#         freq_item_sketch_lg_max_k=8,  # 1024,
#         treat_infinity_as_nan=True,
#         hll_sketch_lg_k=12,  # parameter that controls size of the sketch and accuracy of estimates
#         hll_sketch_type=HLL_4,
#         kll_sketch_k=200,  # 2048,
#         kll_sketch_shrinking_factor=0.64,
#         kll_histogram_num_buckets=20,
#     ):
#         self.approx_distinctness = approx_distinctness
#         self.frequent_items = frequent_items
#         self.approx_quantiles = approx_quantiles
#         self.freq_item_max = freq_item_max
#         self.freq_item_sketch_lg_max_k = freq_item_sketch_lg_max_k
#         self.treat_infinity_as_nan = treat_infinity_as_nan
#         self.hll_sketch_lg_k = hll_sketch_lg_k
#         self.hll_sketch_type = hll_sketch_type
#         self.kll_sketch_k = kll_sketch_k
#         self.kll_sketch_shrinking_factor = kll_sketch_shrinking_factor
#         self.kll_histogram_num_buckets = kll_histogram_num_buckets

#     def to_dict(self):
#         return {
#             "approx_distinctness": self.approx_distinctness,
#             "frequent_items": self.frequent_items,
#             "approx_quantiles": self.approx_quantiles,
#             "freq_item_max": self.freq_item_max,
#             "freq_item_sketch_lg_max_k": self.freq_item_sketch_lg_max_k,  # 1024,
#             "treat_infinity_as_nan": self.treat_infinity_as_nan,
#             "hll_sketch_lg_k": self.hll_sketch_lg_k,
#             "hll_sketch_type": self.hll_sketch_type,
#             "kll_sketch_k": self.kll_sketch_k,
#             "kll_sketch_shrinking_factor": self.kll_sketch_shrinking_factor,
#             "kll_histogram_num_buckets": self.kll_histogram_num_buckets,
#         }


# # Numerize values based on PySpark types
# class NumerizationHelper:
#     def numerize(value, data_type):
#         if isinstance(data_type, BooleanType):
#             return 1.0 if bool(value) else 0.0  # TODO: np.bool_ ?
#         if isinstance(data_type, ByteType):
#             return int(value)  # np.int8(value)
#         if isinstance(data_type, ShortType):
#             return value  # np.short(value)
#         if isinstance(data_type, IntegerType):
#             return value  # np.intc(value)
#         if isinstance(data_type, LongType):
#             return value  # np.int_(value)
#         if isinstance(data_type, FloatType):
#             return value  # np.single(value)
#         if isinstance(data_type, DoubleType):
#             return value  # np.double(value)
#         if isinstance(data_type, DecimalType):
#             return value  # np.longdouble(value)
#         if isinstance(data_type, TimestampType):
#             return value.timestamp()  # np.timedelta64(str(value))
#         if isinstance(data_type, DateType):
#             return value.timestamp()  # np.datetime64(str(value))
#         if isinstance(data_type, StringType):
#             # TODO: use 64 bit hash to reduce likelihood of hash collisions
#             # XxHash64Function.hash(UTF8String.fromString(instance.getString(index)),
#             #  StringType, 42L).toDouble
#             return hash(str(value))

#         # TODO: use 64 bit hash to reduce likelihood of hash collisions
#         # XxHash64Function.hash(instance.get(index), data_types(index), 42L).toDouble
#         hash(value)


# class MultivariateOnlineSummarizer:
#     def __init__(self, vector_size: int, data_types, config: DatasketchesConfig):
#         # params
#         self.vector_size = vector_size
#         self.data_types = data_types
#         self.config = config

#         # transient fields
#         self._longs_sketches = None
#         self._hll_sketches = None
#         self._kll_sketches = None

#         # serializable fields
#         self._longs_sketches_bin = None
#         self._hll_sketches_bin = None
#         self._kll_sketches_bin = None
#         # - initialized when handling the first row
#         self._n = None
#         self._curr_mean = None
#         self._curr_m2_n = None
#         self._curr_m2 = None
#         self._curr_l1 = None
#         self._curr_weight_sum = None
#         self._nnz = None
#         self._curr_max = None
#         self._curr_min = None
#         # - needs to be initialized here
#         self._total_count = 0
#         self._total_weight_sum = 0
#         self._weight_square_sum = 0

#         if self.config.frequent_items:
#             self._longs_sketches = [
#                 frequent_strings_sketch(self.config.freq_item_sketch_lg_max_k)
#                 for i in range(0, self.vector_size)
#             ]
#         if self.config.approx_distinctness:
#             self._hll_sketches = [
#                 hll_sketch(self.config.hll_sketch_lg_k, self.config.hll_sketch_type)
#                 for i in range(0, self.vector_size)
#             ]

#         if self.config.approx_quantiles:
#             # self._kll_sketches = (0 to vector_size).map( i => new QuantileNonSample[Double](self.config.kllSketchSize, self.config.kllSketchShrinkingFactor))
#             self._kll_sketches = [
#                 kll_floats_sketch(self.config.kll_sketch_k)
#                 for i in range(0, self.vector_size)
#             ]

#     # Methods

#     def add(self, instance, weight=1.0):
#         if weight == 0.0:
#             return

#         size = len(instance)
#         if self._n is None:
#             assert (size > 0, "Vector should have dimension larger than zero.")
#             self._n = size
#             self._curr_mean = np.zeros(self._n).tolist()
#             self._curr_m2_n = np.zeros(self._n).tolist()
#             self._curr_m2 = np.zeros(self._n).tolist()
#             self._curr_l1 = np.zeros(self._n).tolist()
#             self._curr_weight_sum = np.zeros(self._n).tolist()
#             self._nnz = np.zeros(self._n).tolist()
#             self._curr_max = np.full(self._n, np.finfo(np.double).min).tolist()
#             self._curr_min = np.full(self._n, np.finfo(np.double).max).tolist()

#         assert (
#             self._n == size,
#             "Dimensions mismatch when adding new sample."
#             + " Expecting ${n} but got ${size}.",
#         )

#         nans = (
#             [np.NINF, np.PINF, np.NAN]
#             if self.config.treat_infinity_as_nan
#             else [np.NAN]
#         )

#         for idx in range(self._n):
#             instance_value = instance[idx]
#             if instance_value is not None:
#                 value = NumerizationHelper.numerize(
#                     instance_value, self.data_types[idx]
#                 )
#                 if value not in nans:
#                     if self.config.frequent_items:
#                         self._longs_sketches[idx].update(str(value))
#                     if self.config.approx_distinctness:
#                         self._hll_sketches[idx].update(value)
#                     if self.config.approx_quantiles:
#                         self._kll_sketches[idx].update(float(value))
#                     if self._curr_max[idx] < value:
#                         self._curr_max[idx] = value
#                     if self._curr_min[idx] > value:
#                         self._curr_min[idx] = value

#                     prevMean = self._curr_mean[idx]
#                     diff = value - prevMean
#                     self._curr_mean[idx] = prevMean + weight * diff / (
#                         self._curr_weight_sum[idx] + weight
#                     )
#                     self._curr_m2_n[idx] += (
#                         weight * (value - self._curr_mean[idx]) * diff
#                     )
#                     self._curr_m2[idx] += weight * value * value
#                     self._curr_l1[idx] += weight * abs(value)
#                     self._curr_weight_sum[idx] += weight
#                     self._nnz[idx] += 1

#         self._total_weight_sum += weight
#         self._weight_square_sum += weight * weight
#         self._total_count += 1

#     def merge(self, other: "MultivariateOnlineSummarizer"):
#         if self._total_weight_sum != 0.0 and other._total_weight_sum != 0.0:
#             assert (
#                 self._n == other._n,
#                 "Dimensions mismatch when merging with another summarizer. "
#                 + f"Expecting ${self._n} but got ${other._n}.",
#             )
#             self._total_count += other._total_count
#             self._total_weight_sum += other._total_weight_sum
#             self._weight_square_sum += other._weight_square_sum
#             for i in range(self._n):
#                 thisNnz = self._curr_weight_sum[i]
#                 otherNnz = other._curr_weight_sum[i]
#                 totalNnz = thisNnz + otherNnz
#                 totalCnnz = self._nnz[i] + other._nnz[i]
#                 if totalNnz != 0.0:
#                     deltaMean = other._curr_mean[i] - self._curr_mean[i]
#                     # merge mean together
#                     self._curr_mean[i] += deltaMean * otherNnz / totalNnz
#                     # merge m2n together
#                     self._curr_m2_n[i] += (
#                         other._curr_m2_n[i]
#                         + deltaMean * deltaMean * thisNnz * otherNnz / totalNnz
#                     )
#                     # merge m2 together
#                     self._curr_m2[i] += other._curr_m2[i]
#                     # merge l1 together
#                     self._curr_l1[i] += other._curr_l1[i]
#                     # merge max and min
#                     self._curr_max[i] = max(self._curr_max[i], other._curr_max[i])
#                     self._curr_min[i] = min(self._curr_min[i], other._curr_min[i])

#                 self._curr_weight_sum[i] = totalNnz
#                 self._nnz[i] = totalCnnz
#         elif self._total_weight_sum == 0.0 and other._total_weight_sum != 0.0:
#             self._n = other._n
#             self._curr_mean = other._curr_mean.view()
#             self._curr_m2_n = other._curr_m2_n.view()
#             self._curr_m2 = other._curr_m2.view()
#             self._curr_l1 = other._curr_l1.view()
#             self._total_count = other._total_count
#             self._total_weight_sum = other._total_weight_sum
#             self._weight_square_sum = other._weight_square_sum
#             self._curr_weight_sum = other._curr_weight_sum.view()
#             self._nnz = other._nnz.view()
#             self._curr_max = other._curr_max.view()
#             self._curr_min = other._curr_min.view()

#         if self.config.frequent_items:
#             self._longs_sketches = [
#                 sk[0].merge(sk[1])
#                 for sk in zip(self._longs_sketches, other._longs_sketches)
#             ]

#         if self.config.approx_distinctness:

#             def merge_hll_sketches(hll_1, hll_2):
#                 union = hll_union(self.config.hll_sketch_lg_k)
#                 union.update(hll_1)
#                 union.update(hll_2)
#                 union.get_result()

#             self._hll_sketches = [
#                 merge_hll_sketches(hll[0], hll[1])
#                 for hll in zip(self._hll_sketches, other._hll_sketches)
#             ]

#         if self.config.approx_quantiles:
#             self._kll_sketches = [
#                 kll[0].merge(kll[1])
#                 for kll in zip(self._kll_sketches, other._kll_sketches)
#             ]

#     def get_stats(self, column_names, data_types):
#         if self.config.frequent_items:
#             #             hashing_cols_lookups = np.empty(self.vector_size)
#             def map_longs_sketches(sketch):
#                 items = sketch.get_frequent_items(
#                     frequent_items_error_type.NO_FALSE_POSITIVES
#                 )
#                 take_n = min(len(items), self.config.freq_item_max)
#                 items_lookup = list(map(lambda kv: (kv[1], kv[0]), items[:take_n]))
#                 return items_lookup

#             hashing_cols_lookups = list(map(map_longs_sketches, self._longs_sketches))
#             #             for idx in range(len(self._longs_sketches)):
#             #                 hashing_cols_lookups[idx] = map_longs_sketches(self._longs_sketches[idx])
#         else:
#             hashing_cols_lookups = None

#         if self.config.approx_distinctness:
#             distinct_counts = list(
#                 map(lambda sketch: sketch.get_estimate(), self._hll_sketches)
#             )
#         else:
#             distinct_counts = None

#         if self.config.approx_quantiles:
#             #             histogram = np.empty(self.vector_size)
#             def map_histogram(sketch_tupl):
#                 idx, sketch = sketch_tupl
#                 max_v = self.max_[idx]
#                 min_v = self.min_[idx]
#                 step_size = float(max_v - min_v) / self.config.kll_histogram_num_buckets
#                 if step_size == 0.0:
#                     buckets = [0] * self.config.kll_histogram_num_buckets
#                 else:
#                     split_points = [
#                         float(min_v + step_size * (step + 1))
#                         for step in range(self.config.kll_histogram_num_buckets)
#                     ]
#                     buckets = sketch.get_pmf(split_points)
#                 for i in range(20):
#                     _from = float(min_v + step_size * i)
#                     _to = float(min_v + step_size * (i + 1))
#                     buckets[i] = f"${_from}-${_to}"

#             histogram = list(map(map_histogram, enumerate(self._kll_sketches)))
#         else:
#             histogram = None

#         if self.config.approx_quantiles:
#             percentiles = list(
#                 map(
#                     lambda sketch: [
#                         sketch.get_quantile(i / 100.0) for i in range(1, 99)
#                     ],
#                     self._kll_sketches,
#                 )
#             )
#         else:
#             percentiles = None

#         return DatasketchesStatistics(
#             column_names,
#             data_types,
#             distinct_counts,
#             hashing_cols_lookups,
#             histogram,
#             self.mean,
#             self.variance,
#             self.count,
#             self.weight_sum,
#             self.num_non_zeros,
#             self.max_,
#             self.min_,
#             self.norm_l2,
#             self.norm_l1,
#             percentiles,
#         )

#     # Statistics

#     @property
#     def mean(self):
#         assert (
#             self._total_weight_sum > 0,
#             "Nothing has been added to this summarizer.",
#         )
#         return [
#             self._curr_mean[i] * (self._curr_weight_sum[i] / self._total_weight_sum)
#             for i in range(self._n)
#         ]  # TODO: initialize with size self._n

#     @mean.setter
#     def mean(self, mean):
#         self._mean = mean

#     @property
#     def variance(self):
#         assert (
#             self._total_weight_sum > 0,
#             "Nothing has been added to this summarizer.",
#         )

#         realVariance = []  # TODO: initialize with size self._n
#         denominator = self._total_weight_sum - (
#             self._weight_square_sum / self._total_weight_sum
#         )

#         # Sample variance is computed, if the denominator is less than 0, the variance is just 0.
#         if denominator > 0.0:
#             delta_mean = self._curr_mean
#             for i in range(len(self._curr_m2_n)):
#                 # We prevent variance from negative value caused by numerical error.
#                 realVariance.append(
#                     max(
#                         (
#                             self._curr_m2_n[i]
#                             + delta_mean[i]
#                             * delta_mean[i]
#                             * self._curr_weight_sum[i]
#                             * (self._total_weight_sum - self._curr_weight_sum[i])
#                             / self._total_weight_sum
#                         )
#                         / denominator,
#                         0.0,
#                     )
#                 )
#         return realVariance

#     @property
#     def count(self):
#         return self._total_count

#     @property
#     def weight_sum(self):
#         return self._total_weight_sum

#     @property
#     def num_non_zeros(self):
#         assert (self._total_count > 0, "Nothing has been added to this summarizer.")
#         return list(map(float, self._nnz))

#     @property
#     def max_(self):
#         assert (
#             self._total_weight_sum > 0,
#             "Nothing has been added to this summarizer.",
#         )
#         # we don't want nulls to count towards max-value
#         # for i in range(self._n):
#         #     if (self._nnz[i] < self._total_count) and (self._curr_max[i] < 0.0):
#         #         self._curr_max[i] = 0.0
#         return self._curr_max

#     @property
#     def min_(self):
#         assert (
#             self._total_weight_sum > 0,
#             "Nothing has been added to this summarizer.",
#         )
#         # we don't want nulls to count towards min-value
#         # for i in range(self._n):
#         #     if (self._nnz[i] < self._total_count) and (self._curr_min[i] > 0.0):
#         #         self._curr_min[i] = 0.0
#         return self._curr_min

#     @property
#     def norm_l2(self):
#         assert (
#             self._total_weight_sum > 0,
#             "Nothing has been added to this summarizer.",
#         )
#         real_magnitude = []  # TODO: initialize with size self._n
#         for i in range(len(self._curr_m2)):
#             real_magnitude.append(sqrt(self._curr_m2[i]))
#         return real_magnitude

#     @property
#     def norm_l1(self):
#         assert (
#             self._total_weight_sum > 0,
#             "Nothing has been added to this summarizer.",
#         )
#         return self._curr_l1

#     # Serialization / deserialization

#     def to_serializable(self):
#         if self.config.frequent_items:
#             self._longs_sketches_bin = list(
#                 map(lambda sketch: sketch.serialize(), self._longs_sketches)
#             )
#             # self._longs_sketches = None  # it's not serializable
#         if self.config.approx_distinctness:
#             self._hll_sketches_bin = list(
#                 map(lambda sketch: sketch.serialize_updatable(), self._hll_sketches)
#             )
#             # self._hll_sketches = None  # it's not serializable
#         if self.config.approx_quantiles:
#             self._kll_sketches_bin = list(
#                 map(lambda sketch: sketch.serialize(), self._kll_sketches)
#             )
#             # self._kll_sketches = None  # it's not serializable
#         return self

#     def from_serialized(self):
#         if self.config.frequent_items:
#             self._longs_sketches = list(
#                 map(
#                     lambda sketch: frequent_strings_sketch.deserialize(sketch),
#                     self._longs_sketches_bin,
#                 )
#             )
#             # self._longs_sketches_bin = None
#         if self.config.approx_distinctness:
#             self._hll_sketches = list(
#                 map(
#                     lambda sketch: hll_sketch.deserialize(sketch),
#                     self._hll_sketches_bin,
#                 )
#             )
#             # self._hll_sketches_bin = None
#         if self.config.approx_quantiles:
#             self._kll_sketches = list(
#                 map(
#                     lambda sketch: kll_floats_sketch.deserialize(sketch),
#                     self._kll_sketches_bin,
#                 )
#             )
#             # self._kll_sketches_bin = None
#         return self

#     def __getstate__(self):
#         self.to_serializable()
#         state = self.__dict__.copy()
#         if "_longs_sketches" in state:
#             del state["_longs_sketches"]
#         if "_hll_sketches" in state:
#             del state["_hll_sketches"]
#         if "_kll_sketches" in state:
#             del state["_kll_sketches"]
#         return state

#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         self.from_serialized()


# class DatasketchesStatisticsEngine:
#     def __init__(self):
#         pass

#     def partition_stats(self, vector_size, data_types, config):
#         # initialize one MultivariateOnlineSummarizeer per partition
#         summarizer = MultivariateOnlineSummarizer(vector_size, data_types, config)

#         def map_func(partitionData):
#             for row in partitionData:
#                 summarizer.add(row)
#             summarizer.to_serializable()
#             yield summarizer

#         return map_func

#     def reduce_stats(self, col_sketches_1, col_sketches_2):
#         a = col_sketches_1.from_serialized()
#         b = col_sketches_2.from_serialized()
#         a.merge(b)
#         a.to_serializable()
#         return a

#     # def to_deequ_profiles(stats: DatasketchesStatistics):
#     #     import com.amazon.deequ.analyzers.DataTypeInstances.{Boolean, Decimal, Fractional, Integral, String, Unknown}
#     #     import com.amazon.deequ.profiles.{ColumnProfile, NumericColumnProfile, StandardColumnProfile}

#     #     col_profiles = [] # TODO: Initialize using n_cols
#     #     for i in len(stats.num_non_zeros):
#     #         data_type = stats.data_types[i]
#     #         if isinstance(data_type, ByteType) or isinstance(data_type, ShortType) or  isinstance(data_type, IntegerType) or  isinstance(data_type, LongType):
#     #             data_type = Integral
#     #         elif isinstance(data_type, FloatType) or isinstance(data_type, DoubleType):
#     #             data_type = Fractional
#     #         elif isinstance(data_type, DecimalType):
#     #             data_type = Decimal

#     #         # TODO: convert stats.histograms and stats.percentiles to deequ-style histograms/percentiles

#     #         NumericColumnProfile(
#     #             stats.column_names[i],
#     #             (stats.count - stats.num_non_zeros[i]) / stats.count,
#     #             None,
#     #             None,
#     #             None,
#     #             float(stats.approx_distinct[i]),
#     #             data_type,
#     #             False,
#     #             [{}], # new HashMap[String, Long](),
#     #             None,
#     #             None,
#     #             stats.mean[i]
#     #             stats.max_[i],
#     #             stats.min_[i],
#     #             None,
#     #             math.sqrt(stats.variance[i]),
#     #             None,
#     #             None,
#     #             None,
#     #             stats.count,
#     #             stats.data_types[i].typeName)
#     #         case _ =>
#     #         val dataType = stats.dataTypes[i] match {
#     #             case BooleanType => Boolean
#     #             case StringType | TimestampType | DateType | BinaryType => String
#     #             case _ => Unknown
#     #         }
#     #         // TODO: convert stats.frequentItems to deequ-style   histograms
#     #         StandardColumnProfile(
#     #             stats.columnNames[i],
#     #             (stats.count - stats.numNonzeros[i]) / stats.count,
#     #             None,
#     #             None,
#     #             None,
#     #             stats.approxDistinct.get[i].toLong,
#     #             dataType,
#     #             false,
#     #             new HashMap[String, Long](),
#     #             None,
#     #             None,
#     #             Some(stats.count),
#     #             Some(stats.dataTypes[i].typeName))
#     #     }
#     #     columnProfiles.append(profile)
#     #     }
#     #     columnProfiles

#     def compute_statistics(self, df, config=DatasketchesConfig(), use_pandas_udf=False):
#         rows = df.rdd
#         row_size = len(df.schema)
#         column_names, data_types = zip(*[(kv.name, kv.dataType) for kv in df.schema])
#         if not use_pandas_udf:
#             finalSummarizer = rows.mapPartitions(
#                 self.partition_stats(row_size, data_types, config),
#                 preservesPartitioning=True,
#             ).treeReduce(self.reduce_stats)
#         # else:
#         #     finalSummarizer.
#         return finalSummarizer.from_serialized().get_stats(column_names, data_types)
