```python
import json
import numpy as np
test_files = ["test_embeddings/miniF2F_full.json_sentence.json","test_embeddings/proofnet_sft.json_sentence.json"]
train_files = ["train_embeddings/3k_without_comment.json_sentence.json","train_embeddings/adddata.json_sentence.json"]
def dataloader(files):
    sentences = []
    embeddings = []
    for file in files:
        with open(file,"r",encoding="utf-8") as f:
            datas = json.load(f)
        for key,value in datas.items():
            if value is not None:
                embeddings.append(value)
                sentences.append({"sentence":key,"dataset":file})
    embeddings = np.array(embeddings)
    print(embeddings.shape)
    return sentences,embeddings
test_sentences,test_embeddings = dataloader(test_files)
train_sentences,train_embeddings = dataloader(train_files)
def normalize_vectors(vectors):
    """对向量进行归一化处理"""
    # 计算L2范数
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 防止除以零
    norms[norms == 0] = 1
    # 向量除以模
    normalized_vectors = vectors / norms
    return normalized_vectors
test_embeddings = normalize_vectors(test_embeddings)
train_embeddings = normalize_vectors(train_embeddings)
```

<script src="https://giscus.app/client.js"
        data-repo="InuyashaYang/AIDIY"
        data-repo-id="R_kgDOM1VVTQ"
        data-category="Announcements"
        data-category-id="DIC_kwDOM1VVTc4Ckls_"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="zh-CN"
        crossorigin="anonymous"
        async>
</script>
