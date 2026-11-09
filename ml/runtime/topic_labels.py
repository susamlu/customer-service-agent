"""训练集主题标签空间的定义。

出现 >= 100 次的意图保留为独立分区，其余（含空 intent）归入「其他」。
训练（topic 头标签）、检索分区（pipeline / build_vector_index）、离线评测
（retrieval / rerank ablation）共用这一份，避免口径漂移。

单独成文件是为了让不加载模型的脚本能拿常量而不拖入 torch 和
sentence-transformers（pipeline.py 顶层会 import 这两个）。
"""

MIN_COUNT = 100
OTHER = "其他"
