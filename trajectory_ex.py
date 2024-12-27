# https://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/Manual-Paths.html

from klampt.model import trajectory

# traj = trajectory.Trajectory()
# #... set up traj
# traj2 = trajectory.HermiteTrajectory()
# traj2.makeSpline(traj)

milestones = [[0,0,0],[0.02,0,0],[1,0,0],[2,0,1],[2.2,0,1.5],[3,0,1],[4,0,-0.3]]
traj = trajectory.Trajectory(milestones=milestones)
#prints milestones 0-5
print(0,":",traj.eval(0))
print(1,":",traj.eval(1))


from klampt import vis

# vis.add("point",[0,0,0])
# vis.animate("point",traj)
# vis.add("traj",traj)
#vis.spin(float('inf'))   #show the visualization forever

traj2 = trajectory.HermiteTrajectory()
traj2.makeSpline(traj)

# vis.add("point",[0,0,0])
# vis.animate("point",traj2)
# vis.add("traj2",traj2)
# vis.spin(float('inf'))
traj_timed = trajectory.path_to_trajectory(traj,vmax=2,amax=4)
#next, try this line instead
#traj_timed = trajectory.path_to_trajectory(traj,timing='sqrt-L2',speed='limited',vmax=2,amax=4)
#or this line
#traj_timed = trajectory.path_to_trajectory(traj2.discretize(0.1),timing='sqrt-L2',speed=0.3)
vis.add("point",[0,0,0])
vis.animate("point",traj_timed)
vis.spin(float('inf'))