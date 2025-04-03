#(c) 2023 ANSYS, Inc. Unauthorized use, distribution, or duplication is prohibited.
import subprocess
import re
import math
import glob
import lumapi
import os
import json
import random
import socket
import time
import warnings

#defaut config
mpirun = 'mpirun'
mpilib = None
fdtd_engine = '/opt/lumerical/v242/bin/fdtd-engine-ompi-lcl'
fdtd_gui = '/opt/lumerical/v242/bin/fdtd-solutions'
pythonpath = '/opt/lumerical/v242/api/python'
python = '/opt/lumerical/v242/python/bin/python'

#read system and user config
config_files = ['%s/lumslurm.config' % os.path.dirname(__file__),
                os.path.expanduser('~/.lumslurm.config')]

for config_file in config_files: 
    try:
        with open(config_file, 'r') as cf:
            config = json.load(cf)
    except FileNotFoundError:
        config = None

    if config:
        for varname in ['mpirun', 'mpilib', 'fdtd_engine', 'fdtd_gui', 'pythonpath', 'python']:
            if varname in config: globals()[varname] = config[varname]

def fdtd_memory_estimate(filename):
    """
    Estimates the memory requirements for an FDTD simulation

    This runs the FDTD solver on a project file to perform
    the memory estimate.

    Parameters:
        filename (string): name of fsp file, possibly including a path

    Returns:
        dict: a dict with fields memory, gridpoints and time_steps.
        Memory field is the memory requirement in bytes
    """

    args = [fdtd_engine, '-mr', filename]
    p = subprocess.run(args, capture_output=True, encoding='utf-8')

    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        raise Exception(fdtd_engine + ' failed with: ' + p.stderr)
    
    report = {}
    for line in p.stdout.split():
        x = line.split('=')
        if len(x) == 2:
            report[x[0]] = float(x[1])

    report['memory'] = report['memory']*1e6

    return report

def partition_info():
    """
    Query slurm for information about partitions

    Returns:
        dict: key is partition name, value is a dict with
        partition details
    """

    args = ['sinfo', '--json']
    p = subprocess.run(args, capture_output=True, encoding='utf-8')

    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        raise Exception('sinfo command failed with: ' + p.stderr)

    info = json.loads(p.stdout)
    partitions = dict()

    if 'sinfo' in info:
        parts = info['sinfo']
        for part in parts:
            specs = {'cpu': part['cpus']['maximum'], 'memory': part['memory']['maximum']*1e6}

            gres = part['gres']['total'].split(":", 2)
            if len(gres) >=2 and gres[0] == 'gpu':
                m = re.search('[0-9]+', gres[-1])
                if m:
                    specs['gpu'] = int(m[0])
                    if len(gres) > 2:
                        specs['gpu model'] = gres[1]
                else:
                    specs['gpu'] = 0
            else:
                specs['gpu'] = 0

            pname = part['partition']['name']
            if pname in partitions:
                for k in ['cpu', 'memory', 'gpu']:
                    if specs[k] > partitions[pname][k]:
                        partitions[pname][k] = specs[k]
                if 'gpu model' in specs:
                    partitions[pname]['gpu model'] = specs['gpu model']
            else:
                partitions[pname] = specs
    else:
        raise Exception('sinfo output not in the expected format.'
			+' You will need to manually specify the number of threads and processes')

    return partitions

def suggest_partitions(fsp_file):
    """
    Suggest slurm partitions for solving fsp_file

    This function selects the slurm paritions with
    nodes with an optimal number of CPUs for the
    solve job. The solve job's memory requirements
    are estimated to determine the number of CPUs
    to use.

    Suggested partitions will all contain nodes with the same
    number of CPU
                    
    Parameters:
        fsp_file (sting): name of the fsp file, possibly
        including the path
                         
    Returns:
        string: comma separated list of slurm partitions.
    """

    req_mem = fdtd_memory_estimate(fsp_file)['memory']
    thread_mid = math.ceil(req_mem/100e6)
    thread_high = math.floor(req_mem/50e6)

    partitions = partition_info()

    potentials = list()
    for partition, info in partitions.items():

        if req_mem > info['memory']: continue
        if info.get('gpu') > 0: continue

        cpu = info['cpu']
        distance = abs(cpu-thread_mid)
        rank = 1 if cpu <= thread_high else 2

        potentials.append({'partition': partition,
                           'rank': rank,
                           'distance': distance,
                           'cpu': cpu})

    potentials.sort(key=lambda val: val['distance'])

    for r in [1, 2]:
        semifinalists = [x for x in potentials if x['rank'] == r]
        if len(semifinalists) > 0:
            lowest_distance = semifinalists[0]['distance']
            finalists = [x for x in semifinalists if x['distance'] == lowest_distance]
            finalists.sort(reverse=True, key=lambda val: val['cpu'])
            highest_cpu = finalists[0]['cpu']
            winners = [x['partition'] for x in finalists if x['cpu'] == highest_cpu]
            return ",".join(winners)

def slurm_license_exists(feature):

    args = ['scontrol', 'show', 'lic']
    p = subprocess.run(args, capture_output=True, encoding='utf-8')
    p.check_returncode()
    return ('LicenseName=%s\n'%feature) in p.stdout

def sbatch(command,
           partition=None,
           nodes=1,
           ntasks_per_node=None,
           cpus_per_task=None,
           gpus_per_node=None,
           dependency=None,
           licenses=None,
           name=None,
           block=False):
    """
    Run a job using slurm's sbatch
    """

    #Remove licenses that Slurm does not know about
    licenses_filtered = []
    for license in licenses.split(','):
        feature, qty = license.split(':',2)
        if slurm_license_exists(feature):
            licenses_filtered.append(license)
        else:
            warnings.warn('Ignored license feature %s that Slurm does not recognize' % feature)

    args = [
            'sbatch',
            '--nodes=%d' % nodes
           ]
    if partition:
        args.append('--partition=%s' % partition)
    if ntasks_per_node:
        args.append('--ntasks-per-node=%d' % ntasks_per_node)
    if cpus_per_task:
        args.append('--cpus-per-task=%d' % cpus_per_task)
    if gpus_per_node:
        args.append('--gpus-per-node=%d' % gpus_per_node)
    if dependency:
        args.append('--dependency=%s' % dependency)
    if licenses_filtered:
        args.append('--licenses=%s' % (','.join(licenses_filtered)))
    if name:
        args.append('--job-name=%s' % name)
    if block:
        args.append('--wait')

    script = '#!/bin/bash -x\n' + command

    p = subprocess.run(args, capture_output=True, input=script, encoding='utf-8')

    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        raise Exception('sbatch command failed with: ' + p.stderr)

    m = re.match('Submitted batch job ([0-9]+)\n', p.stdout)
    if not m: raise Exception('sbatch returned unexpected output')
    
    return m[1]
    
def run_solve(fsp_file,
              partition='auto',
              nodes=1,
              processes_per_node='auto',
              threads_per_process='auto',
              gpus_per_node='auto',
              block=False):
    """
    Run an FDTD solve job using slurm

    Parameters:
        fsp_file (string): name of fsp file, possibly including path
        partition (string, optional): name of slurm partition to run
            job on. Multiple partitions allowed as comma separated list.
            Default value of 'auto' will result in partition being selected
            automatically. See suggest_partition()
        nodes (int, optional): number of nodes for distributed solve.
            Default value of 1 runs solve on a single node
        processes_per_node (int, optional): number of processes to run on each node.
            Default value 'auto' will automatically determine the 
            number of processes based on available CPU and number of 
            threads_per_process (if specified)
        threads_per_process (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU and the number of
            processes_per_node (if specified)
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default is true. Value of false will queue a job
                         
    Returns:
        string: slurm job ID for the solve job
    """

    if partition == 'auto':
        partition = suggest_partitions(fsp_file)

    if gpus_per_node=='auto' or processes_per_node=='auto' or threads_per_process=='auto':
        first_partition = partition.split(",")[0]
        pinfo = partition_info()[first_partition]
        
    if gpus_per_node=='auto':
        gpus_per_node = pinfo['gpu'] if pinfo['gpu'] > 0 else None

    if gpus_per_node and gpus_per_node > 0:
        if nodes > 1:
            warnings.warn('GPU solver does not support more than 1 node. Setting nodes=1')
        nodes = 1

        if processes_per_node == 'auto':
            processes_per_node = 1
        if processes_per_node > 1:
            warnings.warn('GPU solver does not support more than 1 process per node. Setting processes_per_node=1')
        processes_per_node = 1

        if threads_per_process == 'auto':
            threads_per_process = gpus_per_node * pinfo['cpu'] / pinfo['gpu']
        
    else:
        if processes_per_node == 'auto' and threads_per_process == 'auto':
            cpu = pinfo['cpu']
            if nodes == 1:
                processes_per_node = 1
                threads_per_process = cpu
            else:
                processes_per_node = cpu
                threads_per_process = 1
        elif processes_per_node == 'auto':
            cpu = pinfo['cpu']
            processes_per_node = round(cpu/threads_per_process)
        elif threads_per_process == 'auto':
            cpu = pinfo['cpu']
            threads_per_process = round(cpu/processes_per_node)

    args = [
            'FI_EFA_FORK_SAFE=1',
            mpirun,
            '--map-by socket',
            '--bind-to socket',
           ]
    if "intel" not in mpirun:
        args.append('--report-bindings')
    args.append(fdtd_engine)
    args.append('-t %d' % threads_per_process)
    if mpilib: args.insert(0, "LD_LIBRARY_PATH=%s" % mpilib)
    if gpus_per_node:
        args.append('-gpu')
    args.append(fsp_file)

    license_units = math.ceil(nodes*processes_per_node*threads_per_process/32)
    licenses = 'lum_fdtd_solve:%d' % license_units 

    jobid = sbatch(' '.join(args),
                   partition,
                   nodes,
                   processes_per_node,
                   threads_per_process,
                   gpus_per_node=gpus_per_node,
                   licenses=licenses,
                   name="S:%s"%fsp_file,
                   block=block)
    return jobid

def run_lum_script(script_file,
                   fsp_file=None,
                   partition=None,
                   threads='auto',
                   dependency=None,
                   job_name=None,
                   block=False):

    if threads == 'auto':
        if partition: threads = partition_info()[partition]['cpu']
        else: threads = 1

    args = [
            fdtd_gui,
            '-platform offscreen',
            '-threads %d' % threads,
            '-run %s' % script_file,
            '-use-solve',
            '-exit'
           ]

    if(fsp_file): args.append(fsp_file)

    licenses = 'lum_fdtd_solve:%d' % math.ceil(threads/32)
    if not job_name: job_name = fsp_file

    jobid = sbatch(' '.join(args),
                   partition,
                   cpus_per_task=threads,
                   dependency=dependency,
                   licenses=licenses,
                   name=job_name,
                   block=block)
    return jobid

def run_py_script(py_file,
                  data_file=None,
                  partition=None,
                  threads='auto',
                  dependency=None,
                  job_name=None,
                  args = [],
                  block=False):

    if threads == 'auto':
        if partition: threads = partition_info()[partition]['cpu']
        else: threads = 1

    cl = [
            'PYTHONPATH=%s' % pythonpath,
            python,
            py_file,
            str(threads)
           ]

    if(data_file): cl.append(data_file)
    for arg in args: cl.append(str(arg))

    licenses = 'lum_fdtd_solve:%d' % math.ceil(threads/32)
    if not job_name: job_name = data_file

    return sbatch(' '.join(cl),
                  partition,
                  cpus_per_task=threads,
                  dependency=dependency,
                  licenses=licenses,
                  name=job_name,
                  block=block)

def run_py_code(py_code,
                fsp_file=None,
                partition=None,
                threads='auto',
                dependency=None,
                job_name=None,
                block=False):

    if threads == 'auto':
        if partition: threads = partition_info()[partition]['cpu']
        else: threads = 1

    args = [
            'PYTHONPATH=%s' % pythonpath,
            python,
            '-c',
            "$'%s'" % py_code.replace('\n','\\n'),
            str(threads)
           ]

    if(fsp_file): args.append(fsp_file)

    licenses = 'lum_fdtd_solve:%d' % math.ceil(threads/32)
    if not job_name: job_name = fsp_file

    return sbatch(' '.join(args),
                  partition,
                  cpus_per_task=threads,
                  dependency=dependency,
                  licenses=licenses,
                  name=job_name,
                  block=block)

def run_script(script_file=None,
               script_code=None,
               fsp_file=None,
               partition=None,
               threads='auto',
               dependency=None,
               job_name=None,
               block=False):
    """
    Run a script job using slurm

    A script job can be Lumerical script or Python. For Python, code can
    be passed as a string or a file. For Lumerical script, only files
    are supported.

    Parameters:
        script_file (string, optional): Filename of script possibly
            including path. If not specified then it is assumed you
            will supply the script_code parameter. Filename should 
            end with lsf or py extension
        script_code (string, optional): string containing Python code.
            If not specified it is assumes you will supply the
            script_file parameter
        fsp_file (string): name of fsp file, possibly including path.
            The fsp file will be passed as the second command line
            argument for Python. The fsp file will be loaded before
            the lsf script is run
        partition (string, optional): name of slurm partition to run
            job on. Multiple partitions allowed as comma separated list.
            Default value of None will result in slurms's default 
            partition being selected
        threads (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU
        dependency (string, optional): ID or IDs of other slurm job(s) that
            must complete before this job will run. If none supplied then
            there is no dependency
        job_name (string, optional): Name to use for slurm job. This will be
            displayed in squeue output
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default value of false will queue job and return
                         
    Returns:
        string: slurm job ID for the script job
    """
    if script_file:
        if script_file.endswith('.py'):
            return run_py_script(script_file, fsp_file, partition, threads, dependency, job_name, block)
        else:
            return run_lum_script(script_file, fsp_file, partition, threads, dependency, job_name, block)
    else:
        return run_py_code(script_code, fsp_file, partition, threads, dependency, job_name, block)

def run_solve_and_script(fsp_file,
                         script_file=None,
                         script_code=None,
                         solve_partition='auto',
                         solve_nodes=1,
                         solve_processes_per_node='auto',
                         solve_threads_per_process='auto',
                         solve_gpus_per_node=None,
                         script_partition=None,
                         script_threads='auto',
                         job_name=None,
                         block=False):
    """
    Run a solve job, then a script job to process 
    the solve output
    """

    solveid = run_solve(fsp_file,
                        solve_partition,
                        solve_nodes,
                        solve_processes_per_node,
                        solve_threads_per_process,
                        solve_gpus_per_node)

    return run_script(script_file,
                      script_code,
                      fsp_file,
                      script_partition,
                      script_threads,
                      "afterok:"+solveid,
                      "P:%s"%fsp_file,
                      block)

def run_batch(fsp_file_pattern,
              postprocess_script=None,
              postprocess_code=None,
              collect_script=None,
              collect_code=None,
              solve_partition='auto',
              solve_nodes=1,
              solve_processes_per_node='auto',
              solve_threads_per_process='auto',
              solve_gpus_per_node=None,
              script_partition=None,
              script_threads='auto',
              job_name=None,
              block=True):
    """
    Run a set of FDTD solve jobs with optional post-processing scripts

    Post-processing scripts or code are run after every solve job. The
    fsp file is supplied as a command line argument (Python) or loaded
    before script runs (lsf).

    The optional collect script can be run when all solve and
    post-processing script jobs have been completed.

    Solve jobs and script jobs can run on different slurm partitions.

    For solve jobs you can configure distributed solves with nodes>1.
    You can easily set the number of processes per node and number of
    threads per process. If no values provided the function will choose
    good default values.

    Parameters:
        fsp_file_pattern (string): pattern for fsp file name,
            possibly including path. Pattern follows glob syntax
        postprocess_script (string, optional): Filename of post-process
            script possibly including path. Post-process script is run
            after every solve job. Filename should end with lsf or py
            extension.
        postprocess_code (string, optional): string containing Python code
            for post-process script.
        collect_script (string, optional): Filename of collect
            script possibly including path. Collect script is run
            after every solve job and post-process script job has completed.
            Filename should end with lsf or py extension.
        collect_code (string, optional): string containing Python code for
            collect script.
        solve_partition (string, optional): name of slurm partition to run
            solve job on. Multiple partitions allowed as comma separated list.
            Default value of 'auto' will result in partition being selected
            automatically. See suggest_partition()
        solve_nodes (int, optional): number of nodes for distributed solve.
            Default value of 1 runs solve on a single node
        solve_processes_per_node (int, optional): number of processes to run on each node.
            Default value 'auto' will automatically determine the 
            number of processes based on available CPU and number of 
            threads_per_process (if specified)
        solve_threads_per_process (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU and the number of
            processes_per_node (if specified)
        script_partition (string, optional): name of slurm partition to run
            script jobs on. Multiple partitions allowed as comma separated list.
            Default value of None will result in default slurm partition being selected
        script_threads (int, optional): number of threads to use for each script process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU
        job_name (string, optional): Name to use for slurm job. This will be
            displayed in squeue output
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default is true. Value of false will queue a job
                         
    Returns:
        string: slurm job ID for the solve job
    """

    fsp_files = glob.glob(fsp_file_pattern)

    jobids = list()
    for fsp_file in fsp_files:
        jobid = run_solve_and_script(fsp_file,
                                     postprocess_script,
                                     postprocess_code,
                                     solve_partition,
                                     solve_nodes,
                                     solve_processes_per_node,
                                     solve_threads_per_process,
                                     solve_gpus_per_node,
                                     script_partition,
                                     script_threads,
                                     job_name
                                     )
        jobids.append(jobid)

    dependency = 'afterok:' + ':'.join(jobids)
    if not job_name: job_name = fsp_file_pattern

    if collect_script:
        return run_script(script_file=collect_script,
                          partition=script_partition,
                          threads=script_threads,
                          dependency=dependency,
                          job_name="C:%s"%job_name,
                          block=block)
    elif collect_code:
        return run_script(script_code=collect_code,
                          partition=script_partition,
                          threads=script_threads,
                          dependency=dependency,
                          job_name="C:%s"%job_name,
                          block=block)
    else:
        return jobids

def compute_results(fsp_file, sweep_name, threads):
    """
    Helper function to compute all the results needed
    for a sweep.

    Used as post-processing script for sweeps
    """

    print(fsp_file)
    fdtd = lumapi.FDTD(filename = fsp_file,
                       serverArgs = {
                           'use-solve':True,
                           'platform':'offscreen',
                           'threads': '%d'%threads})

    print(sweep_name)
    props = fdtd.getsweep(sweep_name).split('\n')

    for prop in props:
        val = fdtd.getsweep(sweep_name, prop)
        if type(val) == dict and 'result' in val.keys():
            result_path = val['result'].split("::")
            result_name = result_path.pop()
            result_object = "::".join(result_path)
            
            print('%s, %s' % (result_object, result_name))
            fdtd.getresult(result_object, result_name)

    fdtd.save()

def load_sweep(fsp_file, sweep_name, threads):
    """
    Helper function to run the loadsweep command.

    Used as the collection script when running sweeps
    """

    print("loadsweep: %s|%s" % (fsp_file, sweep_name))
    fdtd = lumapi.FDTD(filename = fsp_file,
                       serverArgs = {
                           'use-solve':True,
                           'platform':'offscreen',
                           'threads': '%d'%threads})

    top_sweep = sweep_name.split("::")[0]
    fdtd.loadsweep(top_sweep)
    fdtd.save()

def run_sweep(fsp_file,
              sweep_name,
              solve_partition='auto',
              solve_nodes=1,
              solve_processes_per_node='auto',
              solve_threads_per_process='auto',
              solve_gpus_per_node=None,
              script_partition=None,
              script_threads='auto',
              block=True):
    """
    Run an FDTD sweep or nested sweeps as independent jobs for solves
    and result computation

    Each solve job for a parameter value is run as a slurm job. Solve
    jobs can be distributed or single node.
    
    Results are computed as a post-processing step for each solve job
    in their own slurm job. This greatly accelerates heavy post-
    processing like far field projections
    
    Sweep results are collected by a final slurm job that runs when all
    the solve and post-processing jobs have finished.

    Solve jobs and script jobs can run on different slurm partitions.

    For solve jobs you can configure distributed solves with nodes>1.
    You can easily set the number of processes per node and number of
    threads per process. If no values provided the function will choose
    good default values.

    Parameters:
        fsp_file (string): fsp file name, possibly including path.
        sweep_name (string): name of the sweep in fsp file that you
            will run (solve). For nested sweeps include the full
            hierarchy with :: separators. Eg outersweep::innersweep
        solve_partition (string, optional): name of slurm partition to run
            solve jobs on. Multiple partitions allowed as comma separated list.
            Default value of 'auto' will result in partition being selected
            automatically. See suggest_partition()
        solve_nodes (int, optional): number of nodes for distributed solve.
            Default value of 1 runs solve on a single node
        solve_processes_per_node (int, optional): number of processes to run on each node.
            Default value 'auto' will automatically determine the 
            number of processes based on available CPU and number of 
            threads_per_process (if specified)
        solve_threads_per_process (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU and the number of
            processes_per_node (if specified)
        script_partition (string, optional): name of slurm partition to run
            script jobs on (result computations and data collection).
            Multiple partitions allowed as comma separated list.
            Default value of None will result in default slurm partition being selected
        script_threads (int, optional): number of threads to use for each script process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default is true. Value of false will queue a job
                         
    Returns:
        string: slurm job ID for the solve job
    """

    fdtd = lumapi.FDTD(filename = fsp_file,
                       serverArgs = {'use-solve':True, 'platform':'offscreen'})

    top_sweep = sweep_name.split("::")[0]
    fdtd.savesweep(top_sweep)
    fsp_file_pattern = '%s_%s/*.fsp' % (os.path.splitext(fsp_file)[0], top_sweep) 

    postprocess_code = '\n'.join([
            'import lumslurm',
            'import sys',
            'lumslurm.compute_results(sys.argv[2], "%s", int(sys.argv[1]))' % (sweep_name)
            ])

    collect_code = '\n'.join([
            'import lumslurm',
            'import sys',
            'lumslurm.load_sweep("%s", "%s", int(sys.argv[1]))' % (fsp_file, sweep_name)
            ])

    return run_batch(fsp_file_pattern,
                     postprocess_code=postprocess_code,
                     collect_code=collect_code,
                     solve_partition=solve_partition,
                     solve_nodes=solve_nodes,
                     solve_processes_per_node=solve_processes_per_node,
                     solve_threads_per_process=solve_threads_per_process,
                     solve_gpus_per_node=solve_gpus_per_node,
                     script_partition=script_partition,
                     script_threads=script_threads,
                     job_name=fsp_file,
                     block=block)

def launch_notebook_server(partition=None, threads='auto'):

    if threads == 'auto':
        if partition: threads = partition_info()[partition]['cpu']
        else: threads = 1

    d = random.randint(10, 99)

    script = 'vncserver :%d -geometry 1920x1080\n' % d
    script += 'DISPLAY=:%d jupyter notebook --ip=0.0.0.0\n' % d
    script += 'vncserver -kill :%d\n' % d

    job = sbatch(script,
                 partition,
                 cpus_per_task=threads,
                 name='notebook',
                 block=False
                )

    print('Waiting for server to start...')
    filename = 'slurm-%s.out' % job
    while not os.path.exists(filename):
        time.sleep(1)

    print('Waiting for notebook to start...')
    e = re.compile('\s+http://([\w-]+):8888/\?token\=(\w+)')
    while True:
        with open(filename) as f:
            contents = f.read()
            m = e.search(contents)
            if m:
                host = m[1]
                token = m[2]
                break
        time.sleep(1)

    ip = socket.getaddrinfo(host, 80, proto=socket.IPPROTO_TCP)[0][4][0]
    url = 'http://%s:8888/?token=%s' % (ip, token)
    print('Server running at:')
    print(url)
