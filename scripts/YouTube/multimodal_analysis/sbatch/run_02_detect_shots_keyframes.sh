#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=01:00:00
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-15

# Stop the script on any error
set -e


# ================================
# Load environment
# ================================
# This job only ever runs on Roihu's x86_64 CPU partitions ("small"), but
# sbatch by default exports the submitting shell's full environment into
# the job -- including MODULEPATH. If that shell's session started on an
# aarch64 node (e.g. from working on the GPU stages), the inherited
# MODULEPATH points at aarch64 module trees, and `module load` here
# silently resolves aarch64 binaries that then fail with "Exec format
# error" on this (x86_64) node -- observed for rclone. Rebuild MODULEPATH
# from Roihu's own x86_64 init file first so module resolution never
# depends on the submitting shell's architecture.
unset MODULEPATH
while read -r entry; do
    for expanded in $entry; do
        [ -d "$expanded" ] && MODULEPATH="${MODULEPATH:+$MODULEPATH:}$expanded"
    done
done < /appl/modulefiles/.modulepath_init_x86_64
export MODULEPATH

# On Roihu, ffmpeg lives under the gcc/15.2.0 hierarchy (Lmod won't expose
# it otherwise), and loading it after allas knocks allas-tools' rclone off
# PATH -- so gcc+ffmpeg must load first, allas last.
module purge
module load gcc/15.2.0 ffmpeg
module load allas

source /projappl/project_2006782/leyixu/cpu-env/bin/activate

# Move to script directory
cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis


# ================================
# Get (or generate) the video id list, then pick this task's video
# ================================
# Every array task hits this; mkdir is atomic, so it doubles as a lock --
# one task generates the list, the rest wait for the file to appear. Only
# run one array job at a time against a given VIDEO_ID_LIST path.

VIDEO_ID_LIST="video_ids_filtered.txt"
LOCK_DIR="${VIDEO_ID_LIST}.lock"

if [ ! -f "$VIDEO_ID_LIST" ]; then
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "Generating $VIDEO_ID_LIST..."
        python pipeline/02_detect_shots_and_extract_keyframes.py --list_video_ids "$VIDEO_ID_LIST"
        rmdir "$LOCK_DIR"
    else
        echo "Waiting for $VIDEO_ID_LIST (another task is generating it)..."
        while [ ! -f "$VIDEO_ID_LIST" ]; do
            sleep 2
        done
    fi
fi

VIDEO_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$VIDEO_ID_LIST")

if [[ -z "$VIDEO_ID" ]]; then
  echo "No video found for index $SLURM_ARRAY_TASK_ID (list has fewer videos than the array size)"
  exit 0
fi

ALLAS_VIDEO_BUCKET="2006782_youtube_videos"
ALLAS_KEYFRAME_BUCKET="2006782_youtube_videos_keyframes"

VIDEO_TMP_DIR="videos_tmp"
mkdir -p "$VIDEO_TMP_DIR"

VIDEO_PATH="$VIDEO_TMP_DIR/${VIDEO_ID}.mp4"
TAR_FILE="${VIDEO_ID}.tar.gz"


# ================================
# Skip if already processed
# ================================
# s3allas, not allas: the swift "allas:" rclone remote isn't authenticated
# on Roihu ("auth not found"); the project's S3 remote (active by default
# after `module load allas`) is s3allas:.
if rclone lsf "s3allas:$ALLAS_KEYFRAME_BUCKET/$TAR_FILE" 2>/dev/null | grep -q .; then
    echo "Skipping $VIDEO_ID (keyframe tar already in Allas)"
    exit 0
fi


# ================================
# Download source video
# ================================
if [ -f "$VIDEO_PATH" ]; then
    echo "Video already exists locally: $VIDEO_PATH"
else
    echo "Downloading ${VIDEO_ID}.mp4 from Allas to $VIDEO_TMP_DIR..."
    if rclone copy "s3allas:$ALLAS_VIDEO_BUCKET/${VIDEO_ID}.mp4" "$VIDEO_TMP_DIR"; then
        echo "Downloaded ${VIDEO_ID}.mp4 successfully"
    else
        echo "Failed to download ${VIDEO_ID}.mp4 from Allas"
        exit 1
    fi
fi

echo "Processing: $VIDEO_ID"


# ================================
# Detect shots + extract keyframes. The script itself builds the tar shard
# in memory and uploads it straight to Allas (common/keyframe_io.py) -- no
# local tar file is ever created, so there's nothing for this shell script
# to tar or rclone-upload afterward. It exits non-zero on either zero
# keyframes or a failed Allas upload, which `set -e` catches below.
# ================================
python pipeline/02_detect_shots_and_extract_keyframes.py \
    --video_id="$VIDEO_ID" \
    --video_path="$VIDEO_PATH"
echo "Shot detection + keyframe upload finished: $VIDEO_ID"

if [ -f "$VIDEO_PATH" ]; then
    echo "Deleting processed video: $VIDEO_PATH"
    rm -f -- "$VIDEO_PATH"
fi

echo "Finished completely: $VIDEO_ID"
