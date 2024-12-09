```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def cross_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, V)
    return output, attention_weights

# 示例
if __name__ == "__main__":
    np.random.seed(42)
    seq_len_q = 3
    seq_len_k = 4
    d_k = 5
    d_v = 6

    Q = np.random.rand(seq_len_q, d_k)
    K = np.random.rand(seq_len_k, d_k)
    V = np.random.rand(seq_len_k, d_v)

    print("查询矩阵 Q:\n", Q)
    print("\n键矩阵 K:\n", K)
    print("\n值矩阵 V:\n", V)

    output, attention_weights = cross_attention(Q, K, V)

    print("\n注意力权重:\n", attention_weights)
    print("\n注意力输出:\n", output)
```