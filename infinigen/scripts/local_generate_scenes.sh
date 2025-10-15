#!/bin/bash
#
#SBATCH --job-name=render
#SBATCH --output=render.txt
#SBATCH --account=pvl
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=00-01:00:00
#
#SBATCH --chdir=/n/fs/pvl-transobj/layereddepth/infinigen_internal

name=$1
echo folder: $name
num_scenes=50
num_concurrent=500
num_cam=20
num_stuck_at_task=300
max_queued_total=100
max_running_scene=50

/n/fs/pvl-transobj/mamba/envs/infinigen/bin/python3 -m infinigen.datagen.manage_jobs \
    --debug --wandb_mode disabled \
    --output_folder outputs/$name --overwrite \
    --num_scenes $num_scenes \
    --use_existing \
    --pipeline_configs local_256GB indoor_background_configs monocular blender_gt\
    --configs fast_solve singleroom \
    --pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' \
    manage_datagen_jobs.num_concurrent=$num_concurrent jobs_to_launch_next.max_stuck_at_task=$num_stuck_at_task \
    jobs_to_launch_next.max_queued_total=$max_queued_total jobs_to_launch_next.max_running_scene=$max_running_scene \
    slurm_submit_cmd.slurm_account=pvl coarse/slurm_submit_cmd.slurm_account=None \
    iterate_scene_tasks.n_camera_rigs=$num_cam \
    --overrides compose_indoors.restrict_single_supported_roomtype=True camera.spawn_camera_rigs.n_camera_rigs=$num_cam \
    compute_base_views.min_candidates_ratio=1 compose_indoors.terrain_enabled=False \
    keep_cam_pose_proposal.terrain_coverage_range=[0.5,1]