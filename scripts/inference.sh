#!/usr/bin/env bash
CKPT_NAME=$1
FILE_NAME=$2
FPS=${3:-30}

# Use system ffmpeg (has libx264) over conda ffmpeg (built with --disable-gpl)
# Unset LD_LIBRARY_PATH so system ffmpeg doesn't load conda's shared libs
FFMPEG="env -u LD_LIBRARY_PATH /usr/bin/ffmpeg"

NAME="${FILE_NAME%.*}"
EXT="${FILE_NAME##*.}"

IMG_PATH=./demo/input_frames/$NAME
OUTPUT_PATH=./demo/output_frames/$NAME

mkdir -p $IMG_PATH
mkdir -p $OUTPUT_PATH

# convert video to frames
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        $FFMPEG -i ./demo/$FILE_NAME -f image2 -vf fps=${FPS}/1 -qscale 0 ${IMG_PATH}/%06d.jpg 
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp ./demo/$FILE_NAME $IMG_PATH/000001.$EXT
        ;;
    *)
        echo "Unknown file type."
        exit 1
        ;;
esac

END_COUNT=$(find "$IMG_PATH" -type f | wc -l)

# inference with smplest_x
PYTHONPATH=../:$PYTHONPATH \
python main/inference_mod.py \
    --num_gpus 1 \
    --file_name ARG_CRO_220001 \
    --ckpt_name smplest_x_h \
    --start 1 \
    --end 1238 \
    --boxes_path ./demo/ARG_CRO_220001_boxes.npy

python main/inference.py \
    --num_gpus 1 \
    --file_name $NAME \
    --ckpt_name $CKPT_NAME \
    --end $END_COUNT \
    --multi_person \


# convert frames to video
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        $FFMPEG -y -f image2 -r ${FPS} -i ${OUTPUT_PATH}/%06d.jpg -vcodec libx264 -crf 18 -pix_fmt yuv420p ./demo/result_${NAME}.mp4
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp $OUTPUT_PATH/000001.$EXT ./demo/result_$FILE_NAME
        ;;
    *)
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    rm -rf ./demo/input_frames
    rm -rf ./demo/output_frames
else
    echo "ffmpeg failed. Output frames preserved at ${OUTPUT_PATH}"
fi

