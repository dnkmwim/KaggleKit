# Benchmarks

这里保存可重复的测速入口和固定样本，不保存个人网络环境下的临时日志。

比较原则：

1. 所有方法使用同一组 Dataset ref。
2. 候选发现不计入详情获取耗时。
3. 同时记录总耗时、吞吐量、P50/P95、失败数和字段完整度。
4. `rest-concurrent` 默认使用 4 个受控并发任务；不要盲目增加，以免触发限流。
5. 实际结果写入已忽略的 `benchmarks/results/`。

运行：

```bash
python examples/run_benchmark.py
```

或使用命令行传入 `benchmarks/sample_refs.txt` 中的固定样本。

## 已有结果

- [`REPORT.md`](./REPORT.md)：数据获取方式与检索方法的完整测评结论。

报告正文是整理后的结论；单次运行的原始输出按上文第 5 条不提交，可用 `REPORT.md` 中的命令复现。

