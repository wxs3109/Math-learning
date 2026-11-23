r"""
第 17 章 · 需要真实 GPU 的实验（llm/ch17-inference-serving.md 的 17.12 节）。
在 Kaggle Notebook（打开 GPU 和 Internet）、Colab 或 RunPod 上运行：

    pip install vllm          # 会顺带装好匹配的 PyTorch；Kaggle 上大约几分钟
    python ch17_gpu_benchmark.py

vLLM 需要计算能力 7.5 以上的 GPU（T4、RTX 20 系及以后、A100、L4、H100……）。
最后会打印一段 JSON，记下来和第 17 章 17.13 节的理论预期对照。

A. 矩阵乘法：批大小从 1 增加到 4096，看 GPU 从"带宽受限"走到"算力受限"（只需要 PyTorch）。
B. vLLM 吞吐：同时提交 1、8、32、128 个请求，看总吞吐和单个请求的速度怎样变化。
C. vLLM 前缀缓存：同样 32 个请求、同样长度，前缀各不相同（不可能命中）vs 共享一个已缓存的前缀，比较 prefill 时间。

所有代码放在 main() 里、由 `if __name__ == "__main__"` 调用：vLLM 会启动子进程，
没有这个保护时子进程重新导入本文件，又跑一遍实验，常见的报错是 "An attempt has been made to start a new process..."。
"""
import json, time


def bench_matmul(torch, results):
    print("\n=== A. W(8192×8192, FP16) × X(8192×B) ===")
    dt = torch.float16
    W = torch.randn(8192, 8192, device="cuda", dtype=dt)
    rows = []
    for B in [1, 4, 16, 64, 256, 1024, 4096]:
        X = torch.randn(8192, B, device="cuda", dtype=dt)
        for _ in range(3): W @ X                                    # 预热
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reps = 50
        start.record()
        for _ in range(reps): W @ X
        end.record(); torch.cuda.synchronize()
        t = start.elapsed_time(end) / reps / 1e3                    # 秒
        tflops = 2 * 8192 * 8192 * B / t / 1e12
        gbs = (W.numel() + X.numel() + 8192 * B) * 2 / t / 1e9
        rows.append({"B": B, "ms": round(t * 1e3, 4), "TFLOPS": round(tflops, 2), "GB/s": round(gbs, 1)})
        print(f"  B={B:>5}  {t * 1e3:8.3f} ms  {tflops:7.2f} TFLOP/s  有效带宽 {gbs:8.1f} GB/s")
    results["matmul"] = rows
    del W, X; torch.cuda.empty_cache()


def bench_throughput(llm, SamplingParams, results):
    print("\n=== B. 并发请求数 vs 吞吐（每个请求生成 256 个 token） ===")
    gen = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)   # ignore_eos：保证每个请求都生成满 256 个
    rows = []
    for n in [1, 8, 32, 128]:
        prompts = [f"请写一段关于第 {i} 个质数的短文。" for i in range(n)]
        llm.generate(prompts[:1], SamplingParams(max_tokens=8), use_tqdm=False)   # 预热
        t0 = time.perf_counter()
        outs = llm.generate(prompts, gen, use_tqdm=False)
        t = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        rows.append({"concurrency": n, "seconds": round(t, 3), "tokens_per_s": round(toks / t, 1),
                     "per_request_tokens_per_s": round(256 / t, 1)})
        print(f"  并发 {n:>4}：总耗时 {t:6.2f}s，总吞吐 {toks / t:8.1f} token/s，单个请求 {256 / t:6.1f} token/s")
    results["vllm_throughput"] = rows


def bench_prefix_cache(llm, SamplingParams, results):
    print("\n=== C. 前缀缓存：32 个请求、每个约 2000 token，只做 prefill（max_tokens=1） ===")
    body = "本产品的每一个部件都经过严格测试，使用前请仔细阅读说明。" * 80
    one = SamplingParams(temperature=0.0, max_tokens=1)
    # 冷：每个请求的前缀开头就不同，分块哈希全都不一样，不可能互相命中。
    #     注意不能用"同一批共享前缀的请求跑第一次"当冷：同一批里先算完的块会被后面的请求命中，冷的那次其实是半热的。
    cold = [f"手册编号 {i:04d}。" + body + f"\n问题 {i}：第 {i} 条注意事项是什么？" for i in range(32)]
    # 热：共享同一个前缀，先用一个请求把它算进缓存，再计时。
    shared = "以下是一份很长的产品手册。" + body
    warm = [shared + f"\n问题 {i}：第 {i} 条注意事项是什么？" for i in range(32)]
    if hasattr(llm, "reset_prefix_cache"): llm.reset_prefix_cache()
    t0 = time.perf_counter(); llm.generate(cold, one, use_tqdm=False); t_cold = time.perf_counter() - t0
    llm.generate(warm[:1], one, use_tqdm=False)                       # 把共享前缀放进缓存
    t0 = time.perf_counter(); llm.generate(warm, one, use_tqdm=False); t_warm = time.perf_counter() - t0
    print(f"  冷（前缀各不相同）：{t_cold:.3f}s")
    print(f"  热（共享前缀已缓存）：{t_warm:.3f}s，快 {t_cold / t_warm:.1f} 倍")
    results["prefix_cache"] = {"cold_s": round(t_cold, 3), "warm_s": round(t_warm, 3),
                               "prompt_tokens": len(llm.get_tokenizer().encode(warm[0]))}


def main():
    import torch
    results = {"gpu": torch.cuda.get_device_name(0),
               "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))}
    print("GPU:", results["gpu"], "计算能力", results["capability"])
    bench_matmul(torch, results)
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("\n没有安装 vllm，跳过 B、C。"); print(json.dumps(results, ensure_ascii=False)); return
    major = torch.cuda.get_device_capability(0)[0]
    llm = LLM(model=MODEL, dtype="bfloat16" if major >= 8 else "half",   # T4 不支持 BF16
              gpu_memory_utilization=0.85, max_model_len=4096, enable_prefix_caching=True, seed=0)
    bench_throughput(llm, SamplingParams, results)
    bench_prefix_cache(llm, SamplingParams, results)
    print("\n=== 结果（JSON），和 17.13 节的理论预期对照 ===")
    print(json.dumps(results, ensure_ascii=False))


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"     # 小模型：T4 也放得下。卡大的话可以换成 Qwen/Qwen2.5-7B-Instruct

if __name__ == "__main__":
    main()
