r"""
第 5、6 章的配套代码（llm/ch05-language-model.md、llm/ch06-self-attention.md）。只依赖 numpy，可以直接运行：python3 ch05_06_lm_attention.py

第 1 部分：字符级 bigram 语言模型 —— "预测下一个 token" 的最小例子
  - 最大似然估计就是数频率；交叉熵 = 平均负对数似然；困惑度 = exp(交叉熵)。
第 2 部分：单头因果自注意力（causal self-attention）的前向计算
  - 演示 softmax(QK^T/√d)V 的每一步形状；
  - 验证因果掩码：改动后面的 token，不影响前面位置的输出；
  - 演示为什么要除以 √d：不除的话打分的方差随 d 线性增长，softmax 会退化成 one-hot。
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 第 1 部分：bigram 语言模型
# ---------------------------------------------------------------------------
text = ("to be or not to be that is the question "
        "whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune ")
train, test = text[:110], text[110:]

vocab = sorted(set(text))                    # 字符级 token 表
stoi = {ch: i for i, ch in enumerate(vocab)}
V = len(vocab)

def encode(s):
    return np.array([stoi[c] for c in s])

# 计数矩阵：counts[a, b] = 训练文本里 "a 后面紧跟 b" 的次数
counts = np.zeros((V, V))
ids = encode(train)
for a, b in zip(ids[:-1], ids[1:]):
    counts[a, b] += 1

def bigram_probs(alpha):
    """加 alpha 平滑后的条件分布 P(b | a)。alpha=0 就是纯最大似然（数频率）。"""
    c = counts + alpha
    return c / c.sum(axis=1, keepdims=True)

def cross_entropy(P, s):
    """平均负对数似然（单位：nat）。这就是语言模型训练时最小化的损失。"""
    ids = encode(s)
    return -np.mean(np.log(P[ids[:-1], ids[1:]]))

print("=== 第 1 部分：bigram 语言模型 ===")
print(f"词表大小 V = {V}，均匀猜测的困惑度 = {V}")
for alpha in [0.01, 0.1, 1.0]:
    P = bigram_probs(alpha)
    tr, te = cross_entropy(P, train), cross_entropy(P, test)
    print(f"平滑 alpha={alpha:<5}: 训练集困惑度 {np.exp(tr):6.2f}   测试集困惑度 {np.exp(te):6.2f}")

# 从模型采样：每一步按 P(下一个字符 | 当前字符) 抽样
P = bigram_probs(0.1)
cur, out = stoi["t"], ["t"]
for _ in range(60):
    cur = rng.choice(V, p=P[cur]); out.append(vocab[cur])
print("采样结果：", "".join(out))

# ---------------------------------------------------------------------------
# 第 2 部分：单头因果自注意力
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # 减去最大值，数值稳定
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_self_attention(X, Wq, Wk, Wv, scale=True):
    """
    X: (T, d_model)，T 个位置的输入向量（token embedding + 位置编码）。
    返回 (T, d_head) 的输出，以及 (T, T) 的注意力权重。
    """
    Q, K, Vv = X @ Wq, X @ Wk, X @ Wv            # 各自 (T, d_head)
    d = Q.shape[1]
    scores = Q @ K.T                             # (T, T)：位置 i 对位置 j 的"相关度"
    if scale:
        scores = scores / np.sqrt(d)
    T = X.shape[0]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # 上三角 = "未来"位置
    scores = np.where(mask, -np.inf, scores)            # 因果掩码：不许看未来
    A = softmax(scores, axis=1)                  # 每一行是一个概率分布
    return A @ Vv, A                             # 输出 = 按权重对 value 加权平均

print("\n=== 第 2 部分：因果自注意力 ===")
T, d_model, d_head = 6, 16, 8
X = rng.normal(size=(T, d_model))
Wq, Wk, Wv = (rng.normal(size=(d_model, d_head)) / np.sqrt(d_model) for _ in range(3))
Y, A = causal_self_attention(X, Wq, Wk, Wv)
print(f"输入 X 形状 {X.shape}，输出 Y 形状 {Y.shape}，注意力矩阵 A 形状 {A.shape}")
print("注意力矩阵（第 i 行 = 位置 i 分给各位置的权重；右上角全是 0）：")
print(np.round(A, 2))
print("每行之和：", np.round(A.sum(axis=1), 6))

# 验证因果性：只改最后一个位置的输入，前面位置的输出应该完全不变
X2 = X.copy(); X2[-1] += 10 * rng.normal(size=d_model)
Y2, _ = causal_self_attention(X2, Wq, Wk, Wv)
print("改动最后一个 token 后，前 T-1 个位置输出的最大变化：",
      f"{np.abs(Y2[:-1] - Y[:-1]).max():.2e}（应为 0）")
print("最后一个位置输出的变化：", f"{np.abs(Y2[-1] - Y[-1]).max():.2f}（应不为 0）")

# 为什么要除以 √d：q·k 是 d 个独立项之和，方差 ≈ d
print("\n为什么要除以 √d（q、k 各分量独立标准正态）：")
for d in [4, 64, 1024]:
    q = rng.normal(size=(2000, d)); k = rng.normal(size=(2000, d))
    raw = (q * k).sum(1)
    print(f"  d={d:<5} 打分方差：不缩放 {raw.var():8.1f}   除以 √d 后 {(raw / np.sqrt(d)).var():5.2f}")

d = 1024
q = rng.normal(size=d); Kmat = rng.normal(size=(T, d))
w_raw = softmax(Kmat @ q); w_scaled = softmax(Kmat @ q / np.sqrt(d))
print(f"  d={d} 时 softmax 的最大权重：不缩放 {w_raw.max():.3f}（几乎 one-hot，梯度消失）"
      f"   缩放后 {w_scaled.max():.3f}")
