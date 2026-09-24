# Retrieval evaluation

`hybrid_search_queries.json` 是固定查询集，用于比较关键词、语义和 RRF 混合检索。评估时应使用相同候选库、Top-K 和人工相关性标签；未标注结果不得计入质量指标。

```bash
kagglekit hybrid-evaluate \
  --queries evaluation/hybrid_search_queries.json \
  --output evaluation/hybrid_search_results.local.json
```

本仓库不提交带个人审阅痕迹或未确认结论的本地结果文件。

`hybrid_search_results.json` 是随仓库提交的**空模板**：只声明 `status = "pending_human_review"`，`rows` 为空，不含任何真实检索结果或人工标签。真实结果请按上面的命令输出到 `hybrid_search_results.local.json`（已被 `.gitignore` 排除），不要写回模板文件。
