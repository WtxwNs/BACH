#!/usr/bin/env bash
set -euo pipefail

DATA_SETTING=${1:-}
MODE_TYPE=${2:-}
TOKENIZER_MODEL=${3:-}
if [[ -z "$DATA_SETTING" || -z "$MODE_TYPE" || -z "$TOKENIZER_MODEL" ]]; then
    echo "Usage: $0 <setting> <cot|icl_cot> <tokenizer_model> [audio_prompt_modes]" >&2
    exit 1
fi
if [[ "$DATA_SETTING" != dummy ]]; then
    echo "Invalid setting: $DATA_SETTING" >&2
    exit 1
fi
if [[ "$MODE_TYPE" != cot && "$MODE_TYPE" != icl_cot ]]; then
    echo "Invalid mode_type: $MODE_TYPE. Use 'cot' or 'icl_cot'." >&2
    exit 1
fi
read -r -a AUDIO_PROMPT_MODES <<< "${4:-dual inst vocal mixture}"
for mode in "${AUDIO_PROMPT_MODES[@]}"; do
    case "$mode" in
        dual|inst|vocal|mixture) ;;
        *) echo "Invalid audio prompt mode: $mode" >&2; exit 1 ;;
    esac
 done

DATA_ROOT=example
NAME_PREFIX=dummy.msa.xcodec_16k
ORDER=textfirst
JSONL_NAME=jsonl/$NAME_PREFIX.jsonl

common_args=(
    python core/preprocess_data_conditional_xcodec_segment.py
    --input "$DATA_ROOT/$JSONL_NAME"
    --tokenizer-model "$TOKENIZER_MODEL"
    --tokenizer-type MMSentencePieceTokenizer
    --codec-type xcodec --workers 8 --partitions 1
    --instruction "Generate music from the given lyrics segment by segment."
    --instruction-dropout-rate 0.0 --order "$ORDER" --append-eod
    --quantizer-begin 0 --n-quantizer 1
    --use-token-level-interleave --keep-sequential-samples --cot
)

if [[ "$MODE_TYPE" == cot ]]; then
    MMAP_NAME=mmap/${NAME_PREFIX}_stage_1_token_level_interleave_cot_xcodec_$ORDER
    rm -f -- "$DATA_ROOT/jsonl/${NAME_PREFIX}_"*.jsonl
    mkdir -p -- "$DATA_ROOT/$MMAP_NAME"
    "${common_args[@]}" --output-prefix "$DATA_ROOT/$MMAP_NAME"
    rm -f -- "$DATA_ROOT/jsonl/${NAME_PREFIX}_"*.jsonl
    rm -f -- "$DATA_ROOT/${MMAP_NAME}_"*_text_document.bin
    rm -f -- "$DATA_ROOT/${MMAP_NAME}_"*_text_document.idx
else
    MMAP_NAME=mmap/${NAME_PREFIX}_stage_1_token_level_interleave_long_prompt_msa_$ORDER
    rm -f -- "$DATA_ROOT/jsonl/${NAME_PREFIX}_"*.jsonl
    for mode in "${AUDIO_PROMPT_MODES[@]}"; do
        MODE_MMAP_NAME=${MMAP_NAME}_${mode}
        mkdir -p -- "$DATA_ROOT/$MODE_MMAP_NAME"
        "${common_args[@]}" --output-prefix "$DATA_ROOT/$MODE_MMAP_NAME" \
            --use-audio-icl --audio-prompt-mode "$mode" --audio-prompt-len 30
        rm -f -- "$DATA_ROOT/jsonl/${NAME_PREFIX}_"*.jsonl
        rm -f -- "$DATA_ROOT/${MODE_MMAP_NAME}_"*_text_document.bin
        rm -f -- "$DATA_ROOT/${MODE_MMAP_NAME}_"*_text_document.idx
    done
fi
echo "Preprocessing finished for setting '$DATA_SETTING' and mode_type '$MODE_TYPE'."
