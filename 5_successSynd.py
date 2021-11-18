import os, sys, random
import pylab
import matplotlib.pyplot as plt
# from phi.flow import *

if not os.path.isdir('PDE-Control'):
    print("Cloning, PDE-Control repo, this can take a moment")
    os.system("git clone --recursive https://github.com/holl-/PDE-Control.git")
# now we can load the necessary phiflow libraries and helper functions

import sys; sys.path.append('PDE-Control/src')
from shape_utils import load_shapes, distribute_random_shape
from control.pde.incompressible_flow import IncompressibleFluidPDE
from control.control_training import ControlTraining
from control.sequences import StaggeredSequence, RefinedSequence

domain = Domain([64, 64]) # 1D Grid resolution and physical size
step_count = 16 # how many solver steps to perform
dt = 1.0 # Time increment per solver step
example_count = 1000
batch_size = 100
data_path = 'shape-transitions'
pretrain_data_path = 'moving-squares'
shape_library = load_shapes('PDE-Control/notebooks/shapes')

pylab.subplots(1, len(shape_library), figsize=(17, 5))
for t in range(len(shape_library)):
    pylab.subplot(1, len(shape_library), t+1)
    pylab.imshow(shape_library[t], origin='lower')

for scene in Scene.list(data_path): scene.remove()
for _ in range(example_count // batch_size):
    scene = Scene.create(data_path, count=batch_size, copy_calling_script=False)
    print(scene)
    start = distribute_random_shape(domain.resolution, batch_size, shape_library)
    end__ = distribute_random_shape(domain.resolution, batch_size, shape_library)
    [scene.write_sim_frame([start], ['density'], frame=f) for f in range(step_count)]
    scene.write_sim_frame([end__], ['density'], frame=step_count)

for scene in Scene.list(pretrain_data_path): scene.remove()
for scene_index in range(example_count // batch_size):
    scene = Scene.create(pretrain_data_path, count=batch_size, copy_calling_script=False)
    print(scene)
    pos0 = np.random.randint(10, 56, (batch_size, 2)) # start position
    pose = np.random.randint(10, 56, (batch_size, 2)) # end position
    size = np.random.randint(6, 10, (batch_size, 2))
    for frame in range(step_count+1):
        time = frame / float(step_count + 1)
        pos = np.round(pos0 * (1 - time) + pose * time).astype(np.int)
        density = AABox(lower=pos-size//2, upper=pos-size//2+size).value_at(domain.center_points())
        scene.write_sim_frame([density], ['density'], frame=frame)

supervised_checkpoints = {}
for n in [2, 4, 8, 16]:
    app = ControlTraining(n, IncompressibleFluidPDE(domain, dt),
    datapath=pretrain_data_path, val_range=val_range, train_range=train_range, trace_to_channel=lambda _: 'density',
    obs_loss_frames=[n//2], trainable_networks=['OP%d ' % n],
    sequence_class=None).prepare()
    for i in range(1000):
        app.progress() # Run Optimization for one batch
        supervised_checkpoints['OP%d ' % n] = app.save_model()

app = ControlTraining(1, IncompressibleFluidPDE(domain, dt),
datapath=pretrain_data_path, val_range=val_range, train_range=train_range,trace_to_channel=lambda _: 'density',
obs_loss_frames=[1], trainable_networks=['CFE']).prepare()
for i in range(1000):
    app.progress() # Run Optimization for one batch
supervised_checkpoints['CFE'] = app.save_model()

staggered_app = ControlTraining(step_count, IncompressibleFluidPDE(domain, dt),
datapath=data_path, val_range=val_range, train_range=train_range, trace_to_channel=lambda _: 'density', obs_loss_frames=[step_count], trainable_networks=['CFE', 'OP2','OP4', 'OP8', 'OP16'],sequence_class=StaggeredSequence, learning_rate=5e-4).prepare()

staggered_app.load_checkpoints(supervised_checkpoints)
for i in range(1000):
    staggered_app.progress() # run staggered Optimization for one batch
staggered_checkpoint = staggered_app.save_model()

states = staggered_app.infer_all_frames(test_range)

batches = [0,1,2]
pylab.subplots(len(batches), 10, sharey='row', sharex='col', figsize=(14, 6))
pylab.tight_layout(w_pad=0)
# solutions
for i, batch in enumerate(batches):
    for t in range(9):
        pylab.subplot(len(batches), 10, t + 1 + i * 10)
        pylab.title('t=%d ' % (t * 2))
        pylab.imshow(states[t * 2].density.data[batch, ..., 0], origin='lower')
        pylab.show()

# add targets
testset = BatchReader(Dataset.load(staggered_app.data_path,test_range), staggered_app._channel_struct)[test_range]
for i, batch in enumerate(batches):
    pylab.subplot(len(batches), 10, i * 10 + 10)
    pylab.title('target')
    pylab.imshow(testset[1][i,...,0], origin='lower')
    pylab.show()

errors = []
for batch in enumerate(test_range):
    initial = np.mean( np.abs( states[0].density.data[batch, ..., 0] - testset[1][batch,...,0] ))
    solution = np.mean( np.abs( states[16].density.data[batch, ..., 0] - testset[1][batch,...,0] ))
    errors.append( solution/initial )
print("Relative MAE: "+format(np.mean(errors)))