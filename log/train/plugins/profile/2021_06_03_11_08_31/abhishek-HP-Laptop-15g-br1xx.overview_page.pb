?	G?˵h_j@G?˵h_j@!G?˵h_j@	FK???FK???!FK???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:G?˵h_j@?????M??A?+?V]Tj@YK??q????rEagerKernelExecute 0*	?"??~j@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???nI??!?X??L?B@)ު?PMI??1̟ޣm<@:Preprocessing2U
Iterator::Model::ParallelMapV2?.ޏ?/??!?f????5@)?.ޏ?/??1?f????5@:Preprocessing2F
Iterator::Model{1??*??!4?6?.?B@)4J??%??1?aۍ?0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3?<֌??!??;?O@)ffffff??1K?$Ϙ%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????!??!??q"@)??????1??!??q"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate$??:???!QiY?(@)??x'??1??uc->@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceI.?!????!??\O?@)I.?!????1??\O?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????%ǝ?!?ux?+@)?2?FY?i?1P????*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9FK???I-?=???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????M???????M??!?????M??      ??!       "      ??!       *      ??!       2	?+?V]Tj@?+?V]Tj@!?+?V]Tj@:      ??!       B      ??!       J	K??q????K??q????!K??q????R      ??!       Z	K??q????K??q????!K??q????b      ??!       JCPU_ONLYYFK???b q-?=???X@Y      Y@qe??]?#??"?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 