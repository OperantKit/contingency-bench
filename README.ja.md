# contingency-bench

:gb: [English README](README.md)

contingency-bench は、大規模言語モデル（LLM）が並立変動間隔（VI）スケジュールの 2 つの選択肢をどう選ぶかを測り、その選択を一般化マッチング法則でまとめます。動物と同じ言葉、つまり強化への感度・偏好（バイアス）・参照用の学習モデルの 1 つのパラメータで LLM の選択行動を記述したい研究者に向けたものです。[OperantKit](https://github.com/OperantKit) の中では、LLM を被験体としたベンチマークを担います。装置の計時は測りません。

- **偏りの少ない指標で感度を出す。** 総反応だけでなく、強化されなかった反応にも当てはめます。
- **どの条件も試行数が同じ。** 差が経験量の差にならないようにします。
- **オフラインで動く。** LLM の代わりに模擬被験体を使えるので、API キーなしで手順全体を確かめられます。

## インストール

Python 3.11 以上。contingency-bench はまだ PyPI に公開されていないので、GitHub から入れます。依存する numpy と scipy は PyPI から入ります。

```bash
pip install git+https://github.com/OperantKit/contingency-bench
```

Claude のモデルに問い合わせるときは、Anthropic SDK も入れて `ANTHROPIC_API_KEY` を設定します。

```bash
pip install anthropic
```

## 例

LLM の代わりに模擬被験体（`beta = 0.2` の方策圧縮の学習者）で手順全体を走らせます。API キーは要りません。

```python
import numpy as np
from contingency_bench import run_benchmark

np.random.seed(0)  # 模擬セッションは numpy の大域の乱数生成器から引く
results = run_benchmark(["mock-subject"], use_mock=True, steps_per_condition=200,
                        mock_betas={"mock-subject": 0.2})
r = results[0]
print(round(r.sensitivity_unreinforced, 2), round(r.sensitivity_total, 2), round(r.bias, 2), round(r.estimated_beta, 2))
# 0.38 0.82 -0.18 0.17
```

4 つの値は、強化されなかった反応に当てはめた一般化マッチング法則の感度、総反応に当てはめた感度、偏好（log b）、1 つ目の感度から推定し直した `beta` です。推定された 0.17 は、模擬被験体に与えた 0.2 に近い値です。

## 仕組み

1. **セッション。** ボタンは 2 つです。試行ごとに、まだ強化が設定されていないボタンが、それぞれの確率で設定されます。設定されたボタンへの反応は強化され、設定は解除されます。設定の確率が (0.1, 0.9)、(0.3, 0.7)、(0.5, 0.5)、(0.7, 0.3)、(0.9, 0.1) の 5 条件を走らせます。
2. **被験体。** LLM は試行ごとに、試行番号とボタンごとのそれまでの反応数・強化数を書いた短い日本語のプロンプトを受け取り、`1` か `2` で答えます。模擬被験体は方策圧縮の学習者です（Lai & Gershman, 2021）。
3. **当てはめ。** 一般化マッチング法則（Baum, 1974）を 5 条件にわたって 2 通りに当てはめます。強化されなかった反応の比と、総反応の比です。強化が頻繁なとき、総反応は感度を上に偏らせます（Davison, 2021）。
4. **参照モデル。** `estimate_beta_from_sensitivity` は、当てはめた感度に最も近い感度を出す方策圧縮の学習者の `beta` を探します。これで結果を 1 つのパラメータで言えます。

| 名前 | 中身 |
|---|---|
| `run_benchmark`、`run_condition` | 1 つ以上の被験体で全条件を走らせる／1 条件を走らせる |
| `ConcurrentVISession` | 2 つのボタンのセッション |
| `MockLLMClient`、`APILLMClient`、`BaseLLMClient` | 模擬被験体、Anthropic API 経由の Claude のモデル、自作のクライアントのための基底クラス |
| `PolicyCompressionAgent` | 参照用の学習者 |
| `estimate_matching_parameters`、`estimate_beta_from_sensitivity` | マッチング法則の当てはめと `beta` の探索 |
| `equate_training_steps` | どの条件にも同じ試行数を割り当てる |

コマンドラインでも同じ手順を走らせられます: `contingency-bench --mock --steps 50`。`--mock` を付けないと API を呼びます。`--model`（複数回指定可）で Claude のモデル ID を選びます。`--mock` のときの既定のモデル名は、`beta` の違う模擬被験体に付けた名前にすぎません。

## 制限

- **セッションは時間でなく試行で進みます。** ここでの「VI」は試行ごとの一定の設定確率のことで、秒単位の強化間間隔も、切替遅延（COD）も、Fleshler–Hoffman の値もありません。実験箱のセッションのモデルではありません。
- **ボタンは 2 つ、条件は 5 つだけ**です。自分の `conditions` を渡さない限りコードで固定です。
- **組み込みは Anthropic のモデルだけ**です。他のモデルには `BaseLLMClient` のサブクラスが要ります。
- **失敗した呼び出しは無作為の選択になります。** API の呼び出しが失敗したとき（警告がログに出ます）や、答えに `1` と `2` の両方が含まれるか、どちらも含まれないとき（何もログに出ません）、その試行は無作為に答えられます。そうなった試行の数は報告されません。
- **計時は測りません。** このパッケージに、ハードウェアを組み込んだ（HIL: hardware-in-the-loop）計時のベンチマークはありません。

## AI のコーディングエージェント向け

AI のコーディングエージェント向けの案内は [AGENTS.md](AGENTS.md) にあります。

## 開発

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

両方の README の `python` ブロックはすべて `tests/test_readme_examples.py` が実行します。

## 参考文献

- Baum, W. M. (1974). On two types of deviation from the matching law: Bias and undermatching. *Journal of the Experimental Analysis of Behavior, 22*(1), 231–242. https://doi.org/10.1901/jeab.1974.22-231
- Davison, M. (2021). Concurrent schedules: How responses per reinforcer affects estimates of sensitivity to reinforcement. *Journal of the Experimental Analysis of Behavior, 116*(1), 114–123. https://doi.org/10.1002/jeab.694
- Lai, L., & Gershman, S. J. (2021). Policy compression: An information bottleneck in action selection. *Psychology of Learning and Motivation, 74*, 195–232. https://doi.org/10.1016/bs.plm.2021.02.004

## ライセンス

MIT です。[LICENSE](LICENSE) を参照してください。
