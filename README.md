# LoRAILD

## 動作環境
- Debian GNU/Linux version 11
- Python 3.10.6

### Pythonのモジュール
- requirement.txtの通り

## 利用方法
### 学習を行う場合
1. トークナイズを行う。以下を実行すると、元の訓練データを9:1に分割し、9を訓練データ、1を検証データ、元の検証データがテストデータに設定され、さらにRoBERTa用にトークナイズされる。
```
bash runs/preprocess/normal/roberta.sh
```
2. 教師モデルの訓練する。以下を実行すると、教師モデルが得られる。各タスク、学習率で5つずつチェックポイントが出力されるので、好きなものを選ぶ。
```
bash runs/training/hps/LoRAteacher/*.sh
```
3. ハイパーパラメータ探索をしたい場合は、runs/training/hps以下にある2番で紹介したもの以外のbashを実行する。
```
bash runs/training/hps/*.sh
```

4. ハイパーパラメータを自分で設定したい場合は、runs/training/main以下ののbashファイルに書き込む(後述)。
5. 以下を実行すると、設定したハイパーパラメータで、20エポックの学習が行われる。
```
bash runs/training/main/*.sh
```


### バッシュファイルの仕様
#### 動作に必要なもの(confs以下にある各種コンフィグファイル)
- all_conf: 手法間で共通する設定が書かれたファイル(教師モデルのパスなど)
- task_conf: タスク毎に異なる設定がある場合に記述できる場所の予定だったが、使われていないので、特に必要がなければdummy_task.yamlを使用
- method_conf: 手法毎に異なる設定を記述するファイル
- nep_token_conf: neptune.aiのトークンが書かれたファイル。confs/nep_token.yamlに以下のように記述
```
nep_token: YOUR_TOKEN
```

#### 設定できるところ
- lrs: 各タスクでの学習率
- curriculum_lrs: カリキュラム学習を行う際の学習率(カリキュラム学習が始まるとlrsではなくcurriculum_lrsにある学習率が利用される)
- 各種コンフィグファイルのパスを設定

### 手法とフォルダ名の対応
- nokd  : 蒸留なし
- LoRAKD: LoRAの教師モデルを用いた通常の知識蒸留
- RAILKD: $\mathrm{RAIL\text{-}KD_c}$
- RAILKD_l: $\mathrm{RAIL\text{-}KD_l}$
- CurriculumRAILKD: カリキュラム学習を行った $\mathrm{RAIL\text{-}KD_c}$
- CurriculumFullLoRAILD: 提案手法

### yamlファイルの仕様
#### all_LoRA.yaml
- teacher: 教師モデルのタイプ
- teacher_peft: 実際の教師モデルへのパス
    - roberta*: *部分の数字がLoRAのrを表す。

#### all_Merged.yaml
- teacher: 実際の教師モデルへのパス

#### method_conf
- outdir: 実験データやチェックポイントを出力するディレクトリ
- method: 手法
    - 手法とフォルダ名の対応と一致している
- student: 生徒モデルの名前
- dataset_path: トークナイズ済みのデータセットがあるディレクトリ
- batch_size: バッチサイズ
- epoch: 学習を何エポックするか
- device_num: GPUの数
- nep_method: neptune.aiで表示されるmethod
- save_check: チェックポイントを保存するかどうか
- save_teacher: 教師モデルを保存するかどうか。教師モデルの学習のときのみTrueにする
- is_lora: LoRAアダプターがついているか
    - 教師モデルについているか
    - 生徒モデルについているか
- t_name: 教師モデルがLoRAの場合、rはいくつか？all_LoRA.yamlのteacher_peftのキーと一致させる。LoRAでない場合は単にroberta
- model_name: 生徒モデルについて
- full: 生徒モデルがLoRAの場合、LoRA以外も学習対象にするか
- lambdas: 損失関数の重み
- ild_mode: 提案手法でのaverageなどを決めるもの。設定しなければrandomになる
- teacher_layers: 教師モデルと生徒モデルの対応を設定する
    - averageの場合、以下のようにすると、生徒モデルの0と教師モデルの0から3、生徒モデルの1と教師モデルの4から7...のように対応する
        ```
        teacher_layers :
        -
         - 0
         - 1
         - 2
         - 3
        -
         - 4
         - 5
         - 6
         - 7
        -
         - 8
         - 9
         - 10
         - 11
        -
         - 12
         - 13
         - 14
         - 15
        -
         - 16
         - 17
         - 18
         - 19
        -
         - 20
         - 21
         - 22
         - 23
        ```
    - fixedの場合、以下のようにすると生徒モデルの0と教師モデルの3、生徒モデルの1と教師モデルの7...のように対応する。
        ```
        teacher_layers :
        0: 3
        1: 7
        2: 11
        3: 15
        4: 19
        5: 23
        ```
- tags: neptune.aiのsys/tagsに記録される。
- nep_proj: neptune.aiのプロジェクト名を記入
