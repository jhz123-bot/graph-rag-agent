"""GraphRAG Agent package."""

__version__ = "0.1.0"

# 说明：为保证轻量脚本（如抽取/训练 demo）可在无 Neo4j 环境导入，
# 这里对重量级模块采用可选导入。
try:
    from graphrag_agent.pipelines.ingestion import DocumentProcessor, FileReader, ChineseTextChunker
    from graphrag_agent.graph import (
        GraphConnectionManager, connection_manager, EntityRelationExtractor, GraphWriter, GraphStructureBuilder,
        EntityMerger, SimilarEntityDetector, EntityDisambiguator, EntityAligner, EntityQualityProcessor,
        ChunkIndexManager, EntityIndexManager,
    )
    from graphrag_agent.community import CommunityDetectorFactory, CommunitySummarizerFactory
    from graphrag_agent.search import LocalSearch, GlobalSearch
    from graphrag_agent.search import LocalSearchTool, GlobalSearchTool, HybridSearchTool, NaiveSearchTool, DeepResearchTool
    from graphrag_agent.cache_manager import (
        CacheManager, MemoryCacheBackend, DiskCacheBackend, HybridCacheBackend,
        SimpleCacheKeyStrategy, ContextAwareCacheKeyStrategy, ContextAndKeywordAwareCacheKeyStrategy,
        VectorSimilarityMatcher,
    )
    from graphrag_agent.evaluation.core import (
        BaseMetric, BaseEvaluator, AnswerEvaluationSample, AnswerEvaluationData, RetrievalEvaluationSample, RetrievalEvaluationData,
    )
    from graphrag_agent.evaluation.evaluators import AnswerEvaluator, GraphRAGRetrievalEvaluator, CompositeGraphRAGEvaluator
except Exception:
    # 允许在依赖不完整时仅使用子模块（如 extractors/training scripts）
    pass

__all__ = ["__version__"]
