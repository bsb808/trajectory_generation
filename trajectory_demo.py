# https://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/Manual-Paths.html

# For use with ipython3 --pylab 
# Requires building from source b/c of mismatched numpy versions
#from klampt.model import trajectory
from trajectory import trajectory
import importlib
importlib.reload(trajectory)

# traj = trajectory.Trajectory()
# #... set up traj
# traj2 = trajectory.HermiteTrajectory()
# traj2.makeSpline(traj)
speed = 0.1
#milestones = [[0,0,0],[1,0,0],[2, 1,0],[ 2,-1.5,0],[0.5,5,0]]
milestones = [[0,0,0], [1,0,0], [1,1,0], [2,1,0], [2,-1,0], [-5, -1, 0]]

times = [0]
for i in range(1,len(milestones)):
    dist = np.sqrt(np.sum((np.array(milestones[i])-np.array(milestones[i-1]))**2))
    times.append(times[-1] + dist/speed)

traj = trajectory.Trajectory(times=times, milestones=milestones)
#prints milestones 0-5
print(0,":",traj.eval(0))
print(1,":",traj.eval(1))


# from klampt import vis

# vis.add("point",[0,0,0])
# vis.animate("point",traj)
# vis.add("traj",traj)
#vis.spin(float('inf'))   #show the visualization forever

traj2 = trajectory.HermiteTrajectory()
traj2.makeSpline(traj, preventOvershoot=False)

traj_timed = traj2

# print(traj_timed.endTime())
traj_timed.eval(1)
traj_timed.deriv(1)

# Plot in 2D
wpts = []
for m in milestones:
    wpts.append(m[:2])

tpts = []
tvels = []
tt = linspace(0,traj_timed.endTime(),200)
for t in tt:
    tpts.append(traj_timed.eval(t)[:2])
    tvels.append(traj_timed.deriv(t)[:2])
tpts = array(tpts)
tvels = array(tvels)

figure(1)
clf()
wpts = array(wpts)
plot(wpts[:,0],wpts[:,1],'ro')
plot(tpts[:,0],tpts[:,1],'b-')
xlabel('x')
ylabel('y')

# Plot 2D positions and velocities
figure(2)
clf()
subplot(3,1,1)
plot(tt,tpts[:,0],'b-')
plot(tt,tpts[:,1],'g-')
xlabel('t')
ylabel('position')
legend(['x','y'])

subplot(3,1,2)
plot(tt,tvels[:,0],'b-')
plot(tt,tvels[:,1],'g-')
xlabel('t')
ylabel('velocity')
legend(['x','y'])

subplot(3,1,3)
vels = [np.linalg.norm(v) for v in tvels]
plot(tt,vels)
xlabel('t')
ylabel('velocity')

# Quiver plot of position and velocity at N points along the trajectory
N = 10
tt = linspace(0,traj_timed.endTime(),N)
tpts = []
tvels = []
for t in tt:
    tpts.append(traj_timed.eval(t))
    tvels.append(traj_timed.deriv(t))
tpts = array(tpts)
tvels = array(tvels)
figure(1)
quiver(tpts[:,0],tpts[:,1],tvels[:,0],tvels[:,1])
xlabel('x')
ylabel('y')

show()