# MemRift 绀轰緥锛氫粠 run.py / train.py 璺戦€氳锟�?

鏈ず渚嬬敤浜庨獙锟�? MemRift 鏄惁宸插畬鏁撮泦鎴愬埌 FlagScale 璁粌娴佺▼锛坄run.py` 锟�? Megatron backend 锟�? `train_gpt.py` 锟�? `train.py` 锟�? `inject_memrift_if_configured`锛夛拷?

## 鍓嶇疆鏉′欢

1. **CUDA 鎵╁睍**锛氬畨锟�? `flagscale/compress/float_split_stride_pin`锛堣浠撳簱璇存槑锛夛拷?
2. **鍘嬬缉鏉冮噸**锛氫娇锟�? `scripts/memrift_prepare_weight.sh` 锟�? `prepare_weight.py` 鐢熸垚鍘嬬缉鏉冮噸鐩綍锛堝惈 `index.json` 锟�? `*.bin`锛夛拷?
3. **鏁版嵁锟�? tokenizer**锛氬噯澶囧彲鐢ㄧ殑 `data_path` 锟�? `tokenizer_path`锛堜緥锟�? TinyLlama 鏍煎紡锛夛拷?

## 浣跨敤 openassistant-guanaco

**鐩存帴浣跨敤锟�? memrift_demo 鍚屾簮鏁版嵁**锛氬湪閰嶇疆涓叧锟�? mock銆佸皢 `data_path` 璁句负 `timdettmers/openassistant-guanaco`锛岃缁冨惎鍔ㄦ椂浼氳嚜鍔ㄤ粠 HuggingFace 鎷夊彇璇ユ暟鎹泦骞惰浆锟�? Megatron IndexedDataset锟�?.bin/.idx锛夛紝鏃犻渶浜嬪厛锟�? preprocess_data锟�?

```bash
# 浠撳簱鏍圭洰褰曟墽琛岋細浣跨敤 guanaco 閰嶇疆锛坢ock_data: false, data_path: timdettmers/openassistant-guanaco锟�?
python run.py --config-path=examples/memrift/conf --config-name=train_guanaco action=run \
  train.system.memrift_compressed_weight_dir=./memrift_weights/tinyllama_1b_level18
```

鏁版嵁浼氱紦瀛樺湪 `./data/guanaco_flagscale/`锛屼笌 memrift_demo 鐩稿悓鐨勯澶勭悊锛坄data_prepare`锛夊拰 tokenization锛坢ax_length銆乸adding銆乼runcation锛夛拷?

## 浣跨敤锟�? memrift_demo 鐩稿悓鐨勬暟鎹紙鎵嬪姩鍑嗗 JSONL / .bin锟�?

鑻ュ笇锟�?**鎵嬪姩**鍑嗗鏁版嵁鎴栦娇鐢ㄨ嚜瀹氫箟鏉℃暟锛屽彲鎸変互涓嬫楠ゅ噯澶囷細

1. **鐢熸垚锟�? demo 鍚屾簮锟�? JSONL**锛堝湪浠撳簱鏍圭洰褰曟墽琛岋級锟�?
  ```bash
   python scripts/data/prepare_guanaco_for_flagscale.py \
     --dataset timdettmers/openassistant-guanaco \
     --outdir ./data/guanaco \
     --batch_size 8 \
     --max_samples 1000
  ```
   寰楀埌 `./data/guanaco/guanaco_train.jsonl`锛屽唴瀹逛笌 memrift_demo 锟�? `data_prepare()` 閫昏緫涓€鑷达拷?
2. **杞负 Megatron 鏍煎紡锟�?.bin/.idx锟�?**锛氫娇锟�? Megatron-LM 锟�? `preprocess_data.py`锛岀敤锟�? demo 鐩稿悓锟�? tokenizer锛堝 TinyLlama锛夊锟�? JSONL 锟�? tokenize锛屽緱锟�? `*_text_document.bin` 锟�? `*_text_document.idx`銆傚皢璁粌閰嶇疆涓殑 `data_path` 鎸囧悜璇ユ暟鎹墠缂€锛堜緥锟�? `./data/guanaco/guanaco_text_document`锛夛拷?
3. **閰嶇疆绀轰緥**锛氬湪 `run.py` 涓鐩栨暟鎹矾寰勶細
  ```bash
   train.data.data_path=/path/to/guanaco_text_document
  ```
   鏁版嵁閫昏緫锟�? memrift_demo 锟�? `--dataset`銆乣data_prepare`銆乼okenizer 璁剧疆淇濇寔涓€鑷村嵆鍙拷?

## 锟�? run.py 璺戯紙鎺ㄨ崘锟�?

锟�?**浠撳簱鏍圭洰锟�?**锛堝嵆 `FlagScale` 鐩綍锛変笅鎵ц锟�?

```bash
# 鎸囧畾璺緞涓庡皯锟�? iter 鍋氬啋鐑熸祴璇曪紙锟�? 5 锟�? iter锟�?
export MEMRIFT_WEIGHT_DIR=/path/to/memrift_weights/tinyllama_1b_level18
export DATA_PATH=/path/to/your/data
export TOKENIZER_PATH=/path/to/TinyLlama-1.1B-Chat-v1.0
./scripts/run_memrift_smoke.sh
```

浠呮煡鐪嬪皢瑕佹墽琛岀殑鍛戒护锛堜笉鐪熸璺戣缁冿級锟�?

```bash
./scripts/run_memrift_smoke.sh dryrun
```

鑷畾涔夎凯浠ｆ鏁帮紙榛樿 5锛夛細

```bash
TRAIN_ITERS=10 ./scripts/run_memrift_smoke.sh
```

绛変环鐨勬墜鍔ㄥ懡浠わ紙涓嶉€氳繃鑴氭湰锛夛細

```bash
python run.py \
  --config-path=examples/memrift/conf \
  --config-name=train \
  action=run \
  train.trainer.train_iters=5 \
  train.system.memrift_compressed_weight_dir=/path/to/weights \
  train.data.data_path=/path/to/data \
  train.model.tokenizer_path=/path/to/tokenizer
```

## 锟斤拷模锟斤拷 Mock + MemRift锟斤拷权锟斤拷/锟斤拷锟斤拷锟届步锟斤拷

锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷 `train.system` 锟叫撅拷锟窖匡拷锟斤拷 `memrift_weight_async` 锟斤拷 `memrift_act_async`锟斤拷锟斤拷 `memrift_activation_enable` 锟斤拷希锟斤拷锟組emRift v1 要锟斤拷 `tensor_model_parallel_size: 1`锟斤拷锟节仓匡拷锟侥柯贾达拷小锟�

**1) 准锟斤拷压锟斤拷权锟截ｏ拷锟斤拷 GPU锟斤拷锟斤拷模锟斤拷一锟轿ｏ拷**


| 模锟斤拷                   | HuggingFace 示锟斤拷                                 | 锟斤拷锟斤拷锟斤拷锟侥柯�                                |
| --------------------- | ----------------------------------------------- | ---------------------------------------- |
| TinyLlama-1.1B        | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`            | `./memrift_weights/tinyllama_1b_level18` |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct`              | `./memrift_weights/llama32_3b_level18`   |
| Mistral-7B            | `mistralai/Mistral-7B-Instruct-v0.3`锟斤拷锟斤拷 v0.1锟斤拷 | `./memrift_weights/mistral_7b_level18`   |
| Llama-3.1-8B          | `meta-llama/Llama-3.1-8B`                       | `./memrift_weights/llama31_8b_level18`   |


```bash
python -m flagscale.compress.memrift.offline_comp.prepare_weight \
  --model <HF锟津本碉拷路锟斤拷> --outdir <锟较憋拷目录> --level 18
```

**2) 锟斤拷锟斤拷 mock 训锟斤拷锟斤拷锟斤拷锟斤拷压锟斤拷目录锟斤拷 tokenizer 锟斤拷锟斤拷锟斤拷要锟斤拷**

```bash
# TinyLlama 1.1B
python run.py --config-path=examples/memrift/conf --config-name=train_mock action=run \
  train.system.memrift_compressed_weight_dir=./memrift_weights/tinyllama_1b_level18 \
  train.model.tokenizer_path=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  train.model.tokenizer_model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Llama-3.2-3B-Instruct
python run.py --config-path=examples/memrift/conf --config-name=train_llama32_3b_mock action=run \
  train.system.memrift_compressed_weight_dir=./memrift_weights/llama32_3b_level18 \
  train.model.tokenizer_path=meta-llama/Llama-3.2-3B-Instruct \
  train.model.tokenizer_model=meta-llama/Llama-3.2-3B-Instruct

# Mistral-7B锟斤拷锟斤拷 yaml 默锟斤拷一锟铰匡拷省锟斤拷 tokenizer 锟斤拷锟角ｏ拷
python run.py --config-path=examples/memrift/conf --config-name=train_mistral_7b_mock action=run \
  train.system.memrift_compressed_weight_dir=./memrift_weights/mistral_7b_level18

# Llama-3.1-8B
python run.py --config-path=examples/memrift/conf --config-name=train_llama31_8b_mock action=run \
  train.system.memrift_compressed_weight_dir=./memrift_weights/llama31_8b_level18 \
  train.model.tokenizer_path=meta-llama/Llama-3.1-8B \
  train.model.tokenizer_model=meta-llama/Llama-3.1-8B
```

锟斤拷锟斤拷锟侥硷拷位锟矫ｏ拷`conf/train/tinyllama_1b_lora_memrift*.yaml`锟斤拷`llama32_3b_lora_memrift*.yaml`锟斤拷`mistral_7b_lora_memrift*.yaml`锟斤拷`llama31_8b_lora_memrift*.yaml`锟斤拷锟斤拷锟斤拷锟斤拷诤锟� `train_mock.yaml`锟斤拷`train_llama32_3b_mock.yaml`锟斤拷`train_mistral_7b_mock.yaml`锟斤拷`train_llama31_8b_mock.yaml`锟斤拷

## 閰嶇疆璇存槑

- 涓婚厤缃細`conf/train.yaml`锛堟寚锟�? `backend: megatron`銆乣entrypoint: flagscale/train/train_gpt.py`锛夛拷?
- 榛樿鍚堝苟锛歚conf/train/tinyllama_1b_lora_memrift.yaml`锛圡emRift 寮€鍏炽€丩oRA銆佹暟锟�?/妯″瀷绛夛級锟�?
- Runner 浼氭妸 `train.system` / `train.model` / `train.data` / `train.trainer` 灞曞钩涓哄懡浠よ鍙傛暟浼犵粰 `train_gpt.py`锛堝 `--memrift-enable`銆乣--memrift-compressed-weight-dir` 绛夛級锟�?

## 楠岃瘉 MemRift 鏄惁鐢熸晥

- 鏃ュ織涓簲鍑虹幇 MemRift 鐩稿叧鍒濆鍖栵紙锟�? `inject_memrift_if_configured`銆佹潈閲嶉噴鏀句笌 prefetch锛夛拷?
- 鑻ュ紑锟�? `memrift_print_debug: true`锛屼細鏈夋洿璇︾粏鐨勮皟璇曡緭鍑猴拷?
- 璺戝畬璁惧畾锟�? `train_iters` 涓旀棤鎶ラ敊鍗冲彲璁や负鏈ず渚嬮泦鎴愰€氳繃锟�?

## 杩唬鏃堕暱涓庢樉瀛樺嘲鍊煎湪鍝噷锟�?

璁粌锟�? Runner 璋冭捣鍚庯紝**鏍囧噯杈撳嚭**浼氶噸瀹氬悜鍒版棩蹇楁枃浠讹紝鑰屼笉鏄綋鍓嶇粓绔拷?

### 鏃ュ織鏂囦欢浣嶇疆

- 鍗曟満锛歚outputs/memrift_example/logs/host_0_localhost.output`
- 锟�? `no_shared_fs`锛歚outputs/memrift_example/logs/host.output`

鍗筹細`**<experiment.exp_dir>/logs/`** 涓嬪锟�? host 锟�? `.output` 鏂囦欢锟�?

### 杩唬鏃堕暱锛坋lapsed time per iteration锟�?

- **浣嶇疆**锛氫笂锟�? `.output` 鏂囦欢閲岋紝姣忛殧 `**log_interval`** 锟�? iteration 浼氭墦涓€琛岃缁冩棩蹇楋紙榛樿 `log_interval: 10`锛夛拷?
- **瀛楁**锛歚elapsed time per iteration (ms): 1234.5`锛堝崟浣嶏細姣锛夛拷?
- 浠ｇ爜浣嶇疆锛歚flagscale/train/train.py` 锟�? `training_log()`锛岀害 1979锟�?1980 琛岋拷?

绀轰緥涓€琛岋細

```text
iteration       10/      50 | ... | elapsed time per iteration (ms): 1234.5 | ... | loss scale: ...
```

### 鏄惧瓨宄板€硷紙max allocated / max reserved锟�?

- **浣嶇疆**锛氬悓涓€ `.output` 鏂囦欢涓拷?
- **鏃舵満**锛氬湪**绗竴锟�?**杈惧埌 `log_interval` 锟�? iteration 鏃讹紙optimizer 鍒濆鍖栧畬鎴愬悗锛夛紝浼氳皟鐢ㄤ竴锟�? `report_memory()`锛屾墦鍗板綋锟�? rank 鐨勬樉瀛橈拷?
- **瀛楁**锟�?
  - `max allocated`锛氬埌褰撳墠鏃跺埢涓烘锟�?**鍒嗛厤宄帮拷?**锛圡B锟�?
  - `max reserved`锛氬埌褰撳墠鏃跺埢涓烘锟�?**淇濈暀宄帮拷?**锛圡B锟�?

绀轰緥锟�?

```text
[Rank 0] (after 10 iterations) memory (MB) | allocated: 1234.5 | max allocated: 5678.9 | reserved: ... | max reserved: ...
```

- **TensorBoard**锛氳嫢锟�? system 涓紑锟�? `log_memory_to_tensorboard: true`锛屾樉瀛樹細鍐欏叆 TensorBoard锛堝 `mem-max-allocated-bytes`锛夛紝鍙湪 `outputs/memrift_example/tensorboard` 锟�? `tensorboard --logdir=...` 鏌ョ湅锟�?

## 鍐呭瓨 Profiling锛堢湅璁粌鏃舵樉瀛樹娇鐢級

闇€瑕佺郴缁熸煡锟�? TinyLlama MemRift 璁粌杩囩▼涓殑鏄惧瓨鏇茬嚎涓庡嘲鍊兼椂锛屽彲鐢ㄤ笓鐢ㄨ剼鏈紑鍚唴瀛樼浉鍏冲弬鏁帮細

```bash
# 鍦ㄤ粨搴撴牴鐩綍鎵ц锛氬紑锟�? log_memory_to_tensorboard銆乺ecord_memory_history锛屽苟鍐欏唴瀛樺揩锟�?
./scripts/profile_tinyllama_memrift_memory.sh
```

鑴氭湰浼氶粯璁ゅ紑鍚細

- **log_memory_to_tensorboard**锛氭瘡 `tensorboard_log_interval` 灏嗗綋锟�?/宄板€兼樉瀛樺啓锟�? TensorBoard锟�?
- **record_memory_history**锛氭瘡 `log_interval` 锟�? CUDA 鍐呭瓨蹇収鍐欏叆 `memory_snapshot_path`锛堥粯锟�? `memrift_memory_snapshot.pickle`锛夛拷?

**鏌ョ湅鏂瑰紡锟�?**

1. **鏃ュ織**锛歚outputs/memrift_example/logs/host_0_localhost.output` 锟�? `report_memory()` 锟�? `max allocated` / `max reserved` (MB)锟�?
2. **TensorBoard 鏇茬嚎**锛歚tensorboard --logdir=outputs/memrift_example/tensorboard`锛屾煡锟�? `mem-allocated-bytes`銆乣mem-max-allocated-bytes` 绛夛拷?
3. **鍐呭瓨蹇収**锛歚memrift_memory_snapshot.pickle` 鍙敤 PyTorch 瀹樻柟鍙鍖栧伐鍏锋煡鐪嬪垎閰嶅巻鍙诧拷?

濡傞渶鍚屾椂锟�? PyTorch Profiler trace锛堝惈鏃堕棿绾夸笌鏄惧瓨锛夛紝鍙锟�?

```bash
ENABLE_PYTORCH_PROFILER=1 ./scripts/profile_tinyllama_memrift_memory.sh
```

trace 浼氬啓鍏ュ悓涓€ tensorboard 鐩綍锛屽湪 TensorBoard 锟�? PyTorch Profiler 闈㈡澘鏌ョ湅锟�?

## 涓嶄緷锟�? FlagScale 妗嗘灦锛氱嫭绔嬭剼鏈窇 Llama 1.1B + 璁板綍鏄惧瓨宄帮拷?

鑻ュ彧鎯崇敤 **FlagScale 閲嶆瀯锟�? MemRift 浠ｇ爜**锛坄flagscale/compress/memrift`锛夎窇 TinyLlama 1.1B 璁粌骞惰褰曟樉瀛樺嘲鍊硷紝鑰屼笉锟�? `run.py` / Megatron 娴佺▼锛屽彲浣跨敤鐙珛鑴氭湰锟�?

```bash
# 浠撳簱鏍圭洰褰曟墽琛岋紙鑻ユ棤鍘嬬缉鏉冮噸浼氬厛鑷姩鎵ц prepare_weight锟�?
./scripts/run_memrift_standalone_llama1.1b.sh
```

鎴栨墜鍔ㄦ寚瀹氬弬鏁帮細

```bash
# 1锛夊噯澶囧帇缂╂潈閲嶏紙鑻ュ皻鏈噯澶囷級
python -m flagscale.compress.memrift.offline_comp.prepare_weight \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --outdir ./memrift_weights/tinyllama_1b_level18 --level 18

# 2锛夎繍琛岃缁冨苟璁板綍宄帮拷?
python scripts/train_memrift_standalone_llama1.1b.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --compressed_weights ./memrift_weights/tinyllama_1b_level18 \
  --steps 5 --max_length 512 --profile_memory \
  --output_peak ./scripts/standalone_peak_memory.json
```

鍙€夛細`--activation` 寮€鍚縺娲诲帇缂╋紱`--output_peak <path>` 灏嗗嘲锟�? (MB) 鍐欏叆 JSON銆傝剼鏈娇锟�? HuggingFace Transformers + PEFT LoRA锛屼粎渚濊禆 `flagscale.compress.memrift` 锟�? `CompressedParam`銆乣AsyncCompressor`銆乣DecoderLayerWrapper` 绛夛紝涓嶄緷锟�? FlagScale 锟�? train.py / Megatron锟�?

## 鎸囨爣璇勬祴锛堣鍒掓墽琛岋級

- 鎸囨爣瀹氫箟涓庨獙鏀跺叕寮忚 [METRICS.md](METRICS.md)锛堝熀绾夸竴寰� LoRA锛汸PL + BoolQ锛涘帇缂╃巼 r/r0锛涜缁冧笂涓嬫枃 +20% 闃舵涓€绛夛級銆�
- 閲囬泦 LoRA 鍩虹嚎 PPL銆丅oolQ銆佸姞杞借€楁椂鍙婄鐩樺帇缂╃巼锛圝SONL锛夛細

```bash
cd /share/project/mengyc/code/memrift-flagscale
pip install datasets peft  # 鑻ユ湭瀹夎
python scripts/memrift_metrics_matrix.py --models all --output memrift_metrics.jsonl
```

- 璁粌 `seq_length` 鎻愬崌鑷崇害 `1.2 * L0`锛�

```bash
CONFIG_NAME=train_llama32_3b_mock L0=2048 ./scripts/run_train_context_plus20.sh
```

- 智源模型占位：[conf/train/zhiyuan_placeholder.yaml](conf/train/zhiyuan_placeholder.yaml)

- **Llama-3.1-8B**：MemRift+LoRA 与 FlagScale 纯 LoRA 对比（显存峰值、每迭代耗时），结果写入仓库根目录 `output/llama31_8b_compare/`：

```bash
cd /share/project/mengyc/code/memrift-flagscale
bash scripts/run_llama31_8b_memrift_vs_lora_output.sh
# 可选: TRAIN_ITERS=30 SKIP_PREPARE=1 LLAMA31_MODEL=/path/to/local-llama8b
```

汇总：`output/llama31_8b_compare/summary/comparison.json`（含 `max_allocated_mb`、`elapsed_ms_per_iter_mean` 及二者差值/比值）。

