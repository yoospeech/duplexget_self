# Multilingual Duplex Dataset Collector

무료 공개 음성 데이터셋을 다운로드하고, **Stereo 검사 → Sidon 48kHz 복원 → GPU Whisper 단어 정렬**까지 연결하는 프로젝트입니다.

## 옵션별 실행 방법

| 실행 명령 | 동작 | 다운로드 | GPU 모델 실행 |
|---|---|---|---|
| `bash run.sh` | 공개 데이터 다운로드 → Audio Inspection → Stereo 선별 → Sidon(48kHz) → Whisper | O | O |
| `bash run.sh --check` | HF 공개 데이터셋 접근 상태 및 라이선스 메타데이터 검사 | X | X |
| `bash run.sh --dry-run` | 데이터셋별 파일 수와 예상 다운로드 용량 확인 | X | X |
| `bash run.sh --process-existing` | `raw/`에 이미 있는 음성 파일만 검사·전처리 | X | O |
| `bash run.sh --lang id` | 카탈로그에서 인도네시아어로 표시된 공개 데이터셋만 다운로드·처리 | O | O |
| `bash run.sh --device cuda:1` | Whisper를 GPU 1에서 실행 (Sidon은 기본 GPU 0) | O | O |
| `bash run.sh --skip-install` | 의존성 재설치 없이 전체 실행 | O | O |
| `bash run.sh --inspect` | *현재 미지원*: 실행 옵션으로 사용하지 마세요 | — | — |

**실행 예시**

```bash
# 처음에는 다운로드 예상 용량부터 확인
bash run.sh --dry-run --skip-install

# 실제 전체 파이프라인 실행
bash run.sh --skip-install

# 이미 받아둔 Stereo 오디오만 검사하고 처리
bash run.sh --process-existing --skip-install

# 원본 파일의 채널 수와 샘플레이트만 검사 (Sidon/Whisper 실행 안 함)
.venv/bin/python src/inspect_audio.py scan --input raw \
  --out reports/audio_inspection.jsonl

# 다운로드된 파일 없이 HF 저장소 접근 상태만 확인
bash run.sh --check --skip-install

# 언어별 공개 데이터셋 후보 규모
.venv/bin/python src/public_stats.py
```

**중요:** `run.sh`는 현재 기본적으로 공개로 표시된 HF 후보만 다운로드합니다. 승인된 Gated 데이터를 포함해 전체 HF 후보를 확인하려면 아래 별도 명령을 사용하세요. 접근 승인·라이선스 조건을 직접 확인해야 합니다.

```bash
.venv/bin/python src/catalog_tool.py download --all \
  --include-restricted --acknowledge-rights --dry-run

# 승인된 저장소 전체 다운로드 (저장 용량 주의)
.venv/bin/python src/catalog_tool.py download --all \
  --include-restricted --acknowledge-rights --out raw/
```

TAR/ZIP/Parquet 내부 음성 자동 추출은 아직 지원되지 않습니다. 검사 결과에서 Stereo 여부가 확인되기 전까지는 Full-Duplex 학습용으로 확정하지 않습니다.

## 바로 실행하기

```bash
git clone https://github.com/yoospeech/duplexget_self.git
cd duplexget_self
bash run.sh
```

처음 실행하면 Python 가상환경과 패키지를 준비하고, 라이선스 확인 여부를 물어본 후 작업을 진행합니다. **NVIDIA GPU(CUDA), Python 3.10+, FFmpeg가 필요합니다.** Ubuntu에서 FFmpeg가 없다면 `sudo apt-get install ffmpeg`으로 설치하세요.

### 진행 순서

```text
무료 공개 HF 데이터셋 다운로드
            ↓
Audio Inspection (채널·샘플레이트·길이)
            ↓
Stereo(정확히 2채널)만 선별
            ↓
48kHz 미만: Sidon GPU 복원 → 48kHz Stereo
48kHz 이상: 48kHz Stereo 출력
            ↓
Whisper Timestamped (GPU, word-level)
            ↓
processed/free_stereo_48k/ + JSONL 메타데이터
```

### 실행 옵션

| 명령 | 기능 |
|---|---|
| `bash run.sh` | 전체 다운로드 + 검사 + Sidon + Whisper |
| `bash run.sh --check` | 무료 공개 후보의 접근·라이선스 메타데이터 확인 |
| `bash run.sh --dry-run` | 실제 다운로드 전 파일 크기 미리 확인 |
| `bash run.sh --process-existing` | 이미 `raw/`에 받은 음성 처리 |
| `bash run.sh --lang id` | 인도네시아어 데이터셋 위주로 수집 |
| `bash run.sh --device cuda:1` | Whisper에 GPU 1 사용 (Sidon 기본 GPU 0) |
| `bash run.sh --skip-install` | 패키지 자동 설치 생략 |

`python src/public_stats.py`를 실행하면 공개 후보 데이터의 언어별 데이터셋 수와 발표된 시간을 CSV로 확인할 수 있습니다. [데이터셋 대시보드](data/catalog.html) / [100개 카탈로그](data/catalog.json)

---

## 상세 설명

### 1. 다운로드 범위와 이용 권한

100개 후보 중 카탈로그에서 **공개**로 표시된 항목만 다운로드를 시도합니다. 공급자 유료 상품, 비공개 또는 Gated 저장소, 라이선스 메타데이터가 없는 저장소는 기본 제외됩니다. 공개 저장소라는 이유만으로 기업·상업 학습이 허용되는 것은 아닙니다. 각 원본과 포함된 음성 파일의 라이선스를 별도 검토해야 합니다.

다운로드는 `raw/`에 저장하며 재실행할 수 있습니다. WAV, FLAC, MP3, 일부 ZIP/TAR/Parquet 등의 저장소 파일은 내려받을 수 있으나, **압축·Parquet 안의 음성 파일 자동 추출은 아직 구현되지 않았습니다.** 이 파일들은 추출한 후 `bash run.sh --process-existing`을 다시 사용하세요. 대규모 데이터 수집 전에 디스크 공간을 확인하세요.

### 2. Stereo 및 샘플레이트 정책

| 원본 Stereo 샘플레이트 | 처리 | 저장 형식 |
|---|---|---|
| 8 / 16 / 22.05 kHz | Sidon: 내부 16kHz 특징 추출 후 복원 | 48kHz Stereo WAV |
| 24 / 44.1 kHz | Sidon: 내부 16kHz 특징 추출 후 복원 | 48kHz Stereo WAV |
| 48 kHz | Stereo 유지, PCM WAV로 변환 | 48kHz Stereo WAV |
| 96 kHz 등 48kHz 초과 | 일반 다운샘플링 | 48kHz Stereo WAV |
| Mono 또는 3채널 이상 | 처리 대상에서 제외, 검사 기록 유지 | 없음 |

Sidon은 좌우 채널을 **각각** 처리합니다. 원본 고주파 대역이 정확히 복원되는 것은 아니며, 모델이 새로운 음향 특성을 생성할 수 있습니다. 원본 파일은 `raw/`에 보존하고 복원 여부 및 원본 샘플레이트를 `provenance.jsonl`에 기록합니다.

**주의:** 2채널 Stereo가 곧 두 화자의 분리 녹음이라는 뜻은 아닙니다. 이 일괄 수집 모드에서는 `speaker_separation_verified=false`로 표시합니다. 화자별 Full-Duplex 지도학습에 사용할 때는 별도 검증이 필요합니다.

### 3. GPU 및 Annotation

- **Sidon**: CUDA GPU 필수. 48kHz 미만 Stereo에 적용
- **Whisper Timestamped**: CUDA GPU 필수. 내부적으로 전사 입력만 16kHz로 리샘플링
- **FFprobe/FFmpeg**, 파일 검사·다운로드: CPU로 실행 가능 (모델 추론 아님)

Whisper는 기본적으로 두 채널을 별도로 정렬합니다. 채널 0은 `<clip>.json`, 채널 1은 `<clip>.user.json`으로 저장합니다. 후자는 공식 Moshi 학습 코드에 자동 연결되지 않으며, 좌우 화자 역할도 검증되지 않은 상태입니다.

원본 `annotate.py` 방식의 `alignments` 필드를 사용합니다. GPU 환경에 맞춰 CUDA 활성화된 PyTorch/Torchaudio 빌드를 설치해야 합니다. 실행 전 확인:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 4. 데이터 규모와 보고서

`data/catalog.json`에는 100개 수집 후보가 들어 있으며, 현재 그중 30개가 공개로 표기되어 있습니다. 실제 HF 저장소 유효성·권리·중복은 아직 모두 검증되지 않았습니다. 언어별 발표 시간은 실제 확보 시간이나 Stereo 학습 가능 시간을 의미하지 않습니다.

```bash
python src/public_stats.py
bash run.sh --check
bash run.sh --dry-run
```

생성되는 주요 기록은 `reports/free_audio_inspection.jsonl`, `reports/public_language_stats.csv`, `raw/download_results.jsonl`, `processed/free_stereo_48k/train.jsonl`, `provenance.jsonl`입니다.

### 5. Moshi 호환성 및 제한

이 프로젝트는 사용자의 요청에 따라 **48kHz 원본·복원 음성을 24kHz로 내리지 않습니다.** 일반적인 Kyutai Moshi/Mimi 오디오 토크나이저는 24kHz 입력을 전제로 하므로, 현재 48kHz 음성은 그대로 즉시 학습에 넣을 수 없습니다. 추후 학습 입력 어댑터 또는 48kHz 지원 오디오 토크나이저가 필요합니다.

실제 다중 데이터셋 다운로드, Sidon 체크포인트 GPU 추론, Whisper 통합 테스트는 아직 완료되지 않았습니다. 처음에는 소규모 음성 데이터로 실행하여 모델 호환성, 시간 정렬, 화자 분리 문제를 확인하세요.

### 관련 코드

- [전체 실행](run.sh) · [무료 데이터 수집 파이프라인](download_free.sh)
- [Audio Inspection](src/inspect_audio.py) · [Sidon 복원](src/restore_sidon.py)
- [48kHz Stereo 정리](src/prepare_moshi.py) · [Whisper Timestamped](src/annotate.py)
- [공개 데이터 통계](src/public_stats.py)

### 참고 링크

- [Sidon](https://github.com/sarulab-speech/Sidon)
- [Kyutai Moshi Fine-tuning](https://github.com/kyutai-labs/moshi-finetune)
- [Hugging Face download API](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)
