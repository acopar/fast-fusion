from mpi4py import MPI
import subprocess
import os
import sys

rank = 4

environ = os.environ
environ['OMP_NUM_THREADS'] = '1'
environ['MV2_USE_CUDA'] = '1'
environ['MKL_NUM_THREADS'] = '1'
proc = subprocess.Popen(['mpirun',  '-np', str(rank), #'-mca', 'orte_base_help_aggregate', '0', 
        #'--mca', 'btl', '^openib',
        'python', 'ffusion/factorize.py'] + sys.argv[1:], env=environ)

out, err = proc.communicate()

#print("Spawning")
#comm = MPI.COMM_SELF.Spawn(sys.executable, args=['ffusion/factorize.py'] + sys.argv[1:], maxprocs=4)#

#print("Processes spawned")
#val=42
#sub_comm.bcast(val, MPI.ROOT)#

#common_comm=sub_comm.Merge(False)
#print 'parent in common_comm ', common_comm.Get_rank(), ' of  ',common_comm.Get_size()
#MPI_Intercomm_merge(parentcomm,1,&intracomm);

