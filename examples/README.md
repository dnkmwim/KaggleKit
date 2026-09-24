# Examples

- `fetch_one_page.py`：匿名获取一页 Kaggle Dataset 公开元数据，不写数据库。
- `run_benchmark.py`：用固定样本比较 REST 串行与 4 并发；安装 Kaggle CLI 后会同时测试 CLI。
- `data/`：少量脱敏 API 响应样例，不是真实大数据。

从仓库根目录运行：

```bash
python examples/fetch_one_page.py
python examples/run_benchmark.py
```

