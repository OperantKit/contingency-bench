# contingency-bench

:gb: [English README](README.md)

OperantKit 実験キットのためのベンチマーク・ランナー: タイミング精度、
LLM-as-subject 性能、マッチング法則パラメータ回復。

## 戦略 A: Python 専用移植

本パッケージはオリジナル Rust + Python 混成ベンチマーク・プロジェクトの
Python 専用移植である。戦略 A（移植計画に基づく）は LLM 行動特性
ランナーをここに残し、Rust hardware-in-the-loop (HIL) host_runner /
Teensy ファームウェアは別パッケージに延期する。そのハンドオフが
届くまで、Rust ワークスペースと Teensy ground-truth ファームウェアは
凍結 v0 アーカイブ内に残る（読み取り専用参照、本配布の一部ではない）。

## 公開 API

```python
from contingency_bench import (
    BaseLLMClient,
    APILLMClient,
    MockLLMClient,
    ConcurrentVISession,
    PolicyCompressionAgent,
    MatchingParameters,
    estimate_matching_parameters,
    estimate_beta_from_sensitivity,
    equate_training_steps,
    run_benchmark,
    run_condition,
)
```

CLI:

```bash
contingency-bench --mock --steps 50 --verbose
contingency-bench --mock --model claude-haiku-4-5-20251001 --model claude-sonnet-5 --steps 100
```

## 30 秒で動く例

```bash
pip install git+https://github.com/OperantKit/contingency-bench
```

LLM の代わりに模擬被験体（`beta = 0.2` の方策圧縮 Q 学習エージェント）でベンチマークを走らせる。API キーは要らない:

```python
import numpy as np
from contingency_bench import run_benchmark

np.random.seed(0)  # 模擬セッションは numpy の大域乱数生成器を使う
results = run_benchmark(["mock-subject"], use_mock=True, steps_per_condition=200,
                        mock_betas={"mock-subject": 0.2})
r = results[0]
print(round(r.sensitivity_unreinforced, 2), round(r.sensitivity_total, 2), round(r.bias, 2), round(r.estimated_beta, 2))
# 0.38 0.82 -0.18 0.17
```

表示される値は、強化されなかった反応から求めた一般化対応法則の感度、全反応から求めた感度、偏り (log b)、感度から逆算した `beta` である。逆算した 0.17 は模擬に使った 0.2 に近い。`use_mock=False` にすると、同じ手続きで各試行を Anthropic API 経由で Claude のモデルに提示する（`pip install 'contingency-bench[llm]'`、`ANTHROPIC_API_KEY`）。

## 自作の 50 行の LLM ベンチマークでは足りない理由

- **偏りの少ない反応指標で対応法則を当てはめる.** 感度は全反応からだけでなく、強化されなかった反応からも推定する（全反応は強化率が高いと推定を上に偏らせる。Davison, 2021）。強化比の異なる 5 条件にわたって当てはめる。
- **経験量をそろえる.** `equate_training_steps` はどの条件にも同じ試行数を与える (Johnson et al., 2022)。感度の差が訓練量の差にならない。
- **結果を解釈するための参照モデル.** `estimate_beta_from_sensitivity` は当てはめた感度を方策圧縮エージェント (Lai & Gershman, 2021) の `beta` に対応づけるので、モデルの対応の度合いを 1 つのパラメータで言える。
- **API なしで動く.** `MockLLMClient` と CLI の `--mock` で、テストや CI 向けにパイプライン全体をオフラインで走らせられる。Anthropic SDK は実際のクライアントを作るときにだけ import する。

## インストール

```bash
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

ライブ API ベンチマーク用（Anthropic SDK、遅延読み込み）:

```bash
.venv/bin/python -m pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...
```

## テスト

```bash
.venv/bin/python -m pytest -q
```

## ローカル依存関係に関する注記

Q-learning エージェントと GML 推定器は、同じ monorepo の姉妹パッケージに
由来するが、まだ PyPI に公開されていない。`contingency-bench` を
スタンドアロン・パッケージとしてインストール可能に保つため、両モジュールは
ここに **インライン化** されている:

- `contingency_bench.policy_compression.PolicyCompressionAgent`
- `contingency_bench.matching.estimate_matching_parameters`
  （`session_analyzer.analytics.matching` のミラー）

upstream モジュールが公開されたら、インライン・コピーは再エクスポートに
置き換えるべき。公開シンボルは同じまま。

## LLM 依存関係に関する注記

v0 ドライバは `litellm` を介して複数モデル・ベンダをサポートしていた。本
移植は新リポジトリ標準の LLM 依存関係に合わせ、単一ベンダの `anthropic`
SDK を採用している。SDK は `contingency_bench.llm_client.APILLMClient.__init__`
で **遅延** インポートされる; モジュール自体のインポートで optional な
`[llm]` extra が必要になることはない。複数ベンダ・サポートの再導入は
フォローアップ。

## 将来の Rust ワークスペース

Rust HIL ハーネス (`host_runner` バイナリ、`protocol`、`scenarios`、
`metrics`) と Teensy 4.1 ground-truth ファームウェアは v0 アーカイブに
存在し、`contingency-rs` ハンドオフと並行して別パッケージに移植される
予定。これらは本 Python 専用リリースでは意図的にスコープ外。

## 参考文献

- Baum, W. M. (1974). On two types of deviation from the matching law: Bias
  and undermatching. *Journal of the Experimental Analysis of Behavior*, 22,
  231-242. https://doi.org/10.1901/jeab.1974.22-231
- Davison, M. (2021). What is the value of the generalized matching law?
  *Journal of the Experimental Analysis of Behavior*, 116, 114-123.
  https://doi.org/10.1002/jeab.694
- Johnson, J. E., Kangas, B. D., Bergman, J., & Rice, K. C. (2022).
  Comparison of fixed- and variable-ratio schedules of reinforcement on
  progressive-ratio responding. *Journal of the Experimental Analysis of
  Behavior*, 117, 320-330. https://doi.org/10.1002/jeab.755
- Lai, L., & Gershman, S. J. (2021). Policy compression: An information
  bottleneck in action selection. *Psychology of Learning and Motivation*, 74,
  195-232. https://doi.org/10.1016/bs.plm.2021.02.004
