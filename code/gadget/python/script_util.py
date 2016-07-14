#!/usr/bin/env python
import os, sys, time, subprocess, fnmatch
import numpy as np
TSLP = 0.1

def execute (jobs, nproc, nt=-1, name="anonymous", sh=False, tsleep=TSLP):
  if int(nt) > 0: os.environ['OMP_NUM_THREADS'] = str(nt)
  tic = time.time()
  procs = list()
  while True:
    _procs = procs[:]
    for p in procs:
      if p.poll() is not None:
        if p.returncode != 0:
          print "Error: Python script exit unexpectedly..."
          sys.exit(1)
        _procs.remove(p)
    procs = _procs
    while len(procs) < nproc and len(jobs) > 0:
        procs.append(subprocess.Popen(jobs.pop(0), shell = sh))
    if len(procs) == 0: break
    time.sleep(tsleep)
  print name + ": %.2f sec. [w/ %d proc.]" % (time.time() - tic, nproc)


def gen_matlab_cmd (script_dir, script_name, args):
  return "cd \'{sd}\'; {sn}({arg})"\
    .format(sd = script_dir, sn = script_name, arg = ', '.join(args))


def wrap_matlab_cmd (cmds, singlecore=False):
  s = "matlab -nosplash -nodisplay "
  if singlecore:
    s += "-singleCompThread "
  job = [s + "-r \"{cmds}; exit\"".format(cmds = cmds)]
  return job


def run_matlab_cmd (cmds, name="anonymous", singlecore=False):
  tic = time.time()
  s = "matlab -nosplash -nodisplay "
  if singlecore:
    s += "-singleCompThread "
  job = [s + "-r \"{cmds}; exit\"".format(cmds = cmds)]
  subprocess.call(job, shell = True)
  print name + ": %.2f sec." % (time.time() - tic)


def make_dir (path):
  if not os.path.exists(path): os.makedirs(path)
  return path

def is_file_valid (path):
  return os.path.isfile(path) and os.stat(path).st_size > 0

def last_file (dir_path):
  fs = os.listdir(dir_path)
  fs.sort()
  return dir_path + '/' + fs[-1]

def all_file (dir_path, fn_pattern):
  ret = []
  for f in os.listdir(dir_path):
    if fnmatch.fnmatch(f, fn_pattern):
      ret.append(dir_path + '/' + f)
  return ret

def sample (array, n, repeat, replace=False, do_sort=True):
  ret = []
  for i in range(0, repeat):
    s = np.random.choice(array, n, replace=replace)
    if do_sort:
      s.sort()
    ret.append(s)
  return ret

def write_matrix (path, matrix, pattern='%03d', delim=' '):
  with open(path, 'w') as f:
    f.writelines([
      delim.join([pattern % c for c in r]) + '\n' for r in matrix])

def read_matrix (path, as_string=True):
  if as_string:
    ret = [line.rstrip().split(' ') for line in open(path, 'r')]
  else:
    ret = [map(float, line.split(' ')) for line in open(path, 'r')]
  return ret

def read_vector (path, as_string=True):
  mat = read_matrix(path, as_string)
  return [x for row in mat for x in row]

def read_matrix_cols (paths, col_indices, delim = ' '):
  ret = []
  for f in paths:
    for line in open(f, 'r'):
      sline = line.split(delim)
      ret.extend([float(sline[c]) for c in col_indices])
  return ret

def median (x):
  x.sort()
  l = len(x)
  if l % 2 == 0:
    return (x[l / 2 - 1] + x[l / 2]) / 2.0
  return x[(l - 1) / 2]
