

def job_script(setting, job_name=None, script_address=None, job_output_folder=None):
    text = """#!/bin/bash
#$ -S /bin/bash 
#$ -j Y 
#$ -V 
"""
    text = text + '#$ -o ' + job_output_folder + '\n'
    text = text + '#$ -q ' + setting['cluster_queue'] + '\n'
    text = text + '#$ -N ' + job_name + '\n'
    text = text + '#$ -l h_vmem=' + setting['cluster_memory'] + '\n'
    if setting['cluster_NumberOfCPU'] is not None:
        text = text + '#$ -pe BWA ' + str(setting['cluster_NumberOfCPU']) + '\n'

    if setting['cluster_hostname'] is not None:
        text = text + '#$ -l hostname=' + setting['cluster_hostname'] + '\n'

    # if setting['cluster_Cuda']:

    # text = text + 'export LD_LIBRARY_PATH="/exports/lkeb-hpc/hsokootioskooyi/Program/cudnn73/lib64:$LD_LIBRARY_PATH"' '\n'
    text = text + 'export CUDA_VISIBLE_DEVICES=' + '"' + str(setting['cluster_CUDA_VISIBLE_DEVICES']) + '"' + '\n'
    text = text + 'source ' + setting['cluster_venv'] + '\n'
    text = text + 'module load cuda9' + '\n'

    if setting['cluster_hostname'] is not None:
        text = text + 'echo "on Hostname = $(hostname)"' '\n'
    if setting['cluster_Cuda']:
        text = text + 'echo "on GPU      = $CUDA_VISIBLE_DEVICES"' '\n'
    text = text + 'echo' '\n'
    text = text + 'echo "@ $(date)"' '\n'
    text = text + 'echo' '\n'

    text = text + 'source ~/.bashrc' '\n'

    text = text + 'python ' + script_address + ' --where_to_run Cluster '
    text = text + '\n'
    return text
