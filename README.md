# KaggleKit

KaggleKit 是一个面向 Kaggle Dataset 元数据的轻量工具箱，覆盖候选发现、详情补全、规则过滤、本地检索、混合排序和获取速度测试。默认只处理公开元数据，不下载真实数据集文件。

## 能做什么

- 按标签或查询词发现 Kaggle Dataset。
- 调用 Kaggle REST API 补全标题、描述、标签、作者、更新时间和统计信息。
- 通过 PostgreSQL 全文检索、向量相似度和 RRF 混合排序查找数据集。
- 对 REST 串行、REST 受控并发和 Kaggle CLI 的元数据获取速度进行公平测试。
- 把数据集元数据和少量文件结构信息整理成 Markdown。

## 项目结构

```text
KaggleKit/
├── src/kagglekit/       # 核心代码
├── examples/            # 可直接运行的示例
├── benchmarks/          # 测速说明与固定样本
├── evaluation/          # 检索评估查询集
├── migrations/          # PostgreSQL / pgvector 建表脚本
└── tests/               # 单元测试与脱敏样例数据
```

## 安装

需要 Python 3.11+。检索功能需要 PostgreSQL；语义检索还需要 pgvector。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
```

需要读取表格或 Parquet 文件时：

```bash
python -m pip install -e ".[dataset-markdown,test]"
```

## 配置

复制 `.env.example` 的字段到你自己的环境配置中。不要提交真实 Token 或数据库密码。

```bash
export KAGGLE_API_TOKEN="YOUR_API_KEY"
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DATABASE"
```

当前 REST 元数据接口可匿名访问；若 Kaggle 改变访问策略，再配置 Token。Kaggle CLI 对照测试需另行安装并完成官方认证。

## 快速使用

不写数据库，只检查一页公开元数据：

```bash
kagglekit dataset-sync smoke --source "tag:classification"
```

初始化数据库并同步候选：

```bash
psql "$DATABASE_URL" -f migrations/001_phase1_resources.sql
psql "$DATABASE_URL" -f migrations/002_dataset_sync_state.sql
psql "$DATABASE_URL" -f migrations/003_keyword_search.sql
psql "$DATABASE_URL" -f migrations/004_semantic_search.sql
kagglekit dataset-sync bootstrap --source "query:machine learning"
```

检索：

```bash
kagglekit search keyword --query "housing prices" --limit 10
kagglekit semantic-backfill
kagglekit search semantic --query "中国房价" --limit 10
kagglekit search hybrid \
  --lexical-query "China housing prices" \
  --semantic-query "中国房价" \
  --include-timing
```

## Benchmark

固定同一组 Dataset，只计详情获取阶段，避免候选发现速度干扰结果：

```bash
kagglekit acquisition-benchmark \
  --ref uciml/iris \
  --ref kaggle/sf-salaries \
  --workers 4 \
  --output benchmarks/results/report.json
```

输出包括总耗时、吞吐量、P50/P95、失败数和字段完整度。也可以运行 `examples/run_benchmark.py`。测速结果受网络、限流和本机环境影响，不应把单次结果当作普遍结论。

## 测试

```bash
pytest
```

带 PostgreSQL 的集成测试会在未配置 `TEST_DATABASE_URL` 时跳过。

## 数据与隐私

- 仓库只包含公开 Kaggle 元数据接口的代码、固定查询和少量脱敏测试夹具。
- 不包含 Kaggle 原始大数据、个人配置、真实密钥、运行日志或本机绝对路径。
- `examples/data/` 中的 JSON 仅用于演示响应结构。

