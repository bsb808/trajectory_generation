"""Classes for loading, saving, evaluating, and operating on trajectories.

* For piecewise-linear interpolation in cartesian space, use :class:`~klampt.model.trajectory.Trajectory`.
* For piecewise-linear interpolation on a robot, use :class:`~klampt.model.trajectory.RobotTrajectory`.
* For Hermite interpolation in cartesian space, use :class:`~klampt.model.trajectory.HermiteTrajectory`.

"""

import bisect

# from ..math import so3,se3,vectorops
from . import vectorops
from . import spline
# from ..math.geodesic import *
import warnings
# from ..robotsim import RobotModel,RobotModelLink
# from .subrobot import SubRobotModel
from typing import Iterable,Optional,Union,Sequence,List,Tuple,Callable
from .typing import Vector3,Vector,Rotation,RigidTransform
MetricType = Callable[[Vector,Vector],float]

class Trajectory:
    """A basic piecewise-linear trajectory class, which can be overloaded
    to provide different functionality.  A plain Trajectory interpolates
    in Cartesian space.

    (To interpolate for a robot, use RobotTrajectory. To perform
    Hermite interpolation, use HermiteTrajectory)

    Attributes:
        times (list of floats): a list of times at which the milestones are met.
        milestones (list of Configs): a list of milestones that are interpolated.

    """
        
    def __init__(self,
            times: Optional[List[float]] = None,
            milestones: Optional[List[Vector]] = None
        ):
        """Args:
            times (list of floats, optional): if provided, initializes the
                self.times attribute.  If milestones is provided, a uniform
                timing is set.  Otherwise self.times is empty.
            milestones (list of Configs, optional): if provided, initializes
                the self.milestones attribute.  Otherwise milestones is empty.

        Does not perform error checking.  The caller must be sure that
        the lists have the same size, the times are non-decreasing, and the configs
        are equally-sized (you can call checkValid() for this).
        """
        if milestones is None:
            milestones = []
        if times is None:
            times = list(range(len(milestones)))
        self.times = times
        self.milestones = milestones


    def startTime(self) -> float:
        """Returns the initial time."""
        try: return self.times[0]
        except IndexError: return 0.0
        
    def endTime(self) -> float:
        """Returns the final time."""
        try: return self.times[-1]
        except IndexError: return 0.0

    def duration(self) -> float:
        """Returns the duration of the trajectory."""
        return self.endTime()-self.startTime()

    def checkValid(self) -> None:
        """Checks whether this is a valid trajectory, raises a
        ValueError if not."""
        if len(self.times) != len(self.milestones):
            raise ValueError("Times and milestones are not the same length")
        if len(self.times)==0:
            raise ValueError("Trajectory is empty")
        for (tprev,t) in zip(self.times[:-1],self.times[1:]):
            if tprev > t:
                raise ValueError("Timing is not sorted")
        n = len(self.milestones[0])
        for q in self.milestones:
            if len(q) != n:
                raise ValueError("Invalid milestone size")
        return

    def getSegment(self, t: float, endBehavior: str = 'halt')  -> Tuple[int,float]:
        """Returns the index and interpolation parameter for the
        segment at time t. 

        Running time is O(log n) time where n is the number of segments.

        Args:
            t (float): The time at which to evaluate the segment
            endBehavior (str): If 'loop' then the trajectory loops forever.  

        Returns:
            (index,param) giving the segment index and interpolation
            parameter.  index < 0 indicates that the time is before the first
            milestone and/or there is only 1 milestone.
        """
        if len(self.times)==0:
            raise ValueError("Empty trajectory")
        if len(self.times)==1:
            return (-1,0)
        if t > self.times[-1]:
            if endBehavior == 'loop':
                try:
                    t = t % self.times[-1]
                except ZeroDivisionError:
                    t = 0
            else:
                return (len(self.milestones)-1,0)
        if t >= self.times[-1]:
            return (len(self.milestones)-1,0)
        if t <= self.times[0]:
            return (-1,0)
        i = bisect.bisect_right(self.times,t)
        p=i-1
        assert i > 0 and i < len(self.times),"Invalid time index "+str(t)+" in "+str(self.times)
        u=(t-self.times[p])/(self.times[i]-self.times[p])
        if i==0:
            if endBehavior == 'loop':
                t = t + self.times[-1]
                p = -2
                u=(t-self.times[p])/(self.times[-1]-self.times[p])
            else:
                return (-1,0)
        assert u >= 0 and u <= 1
        return (p,u)
    
    def eval(self, t: float, endBehavior: str = 'halt') -> Vector:
        """Evaluates the trajectory using piecewise linear
        interpolation. 

        Args:
            t (float): The time at which to evaluate the segment
            endBehavior (str): If 'loop' then the trajectory loops forever.  

        Returns:
            The configuration at time t
        """
        return self.eval_state(t,endBehavior)

    def deriv(self, t: float, endBehavior: str = 'halt') -> Vector:
        """Evaluates the trajectory velocity using piecewise linear
        interpolation. 

        Args:
            t (float): The time at which to evaluate the segment
            endBehavior (str): If 'loop' then the trajectory loops forever.  

        Returns:
            The velocity (derivative) at time t
        """
        return self.deriv_state(t,endBehavior)

    def waypoint(self, state: Vector) -> Vector:
        """Returns the primary configuration corresponding to the given state.

        This is usually the same as ``state`` but for some trajectories,
        specifically Hermite curves, the state and configuration are not
        identically the same.
        """
        return state

    def eval_state(self, t: float, endBehavior: str = 'halt') -> Vector:
        """Internal eval, used on the underlying state representation"""
        i,u = self.getSegment(t,endBehavior)
        if i<0: return self.milestones[0]
        elif i+1>=len(self.milestones): return self.milestones[-1]
        #linear interpolate between milestones[i] and milestones[i+1]
        return self.interpolate_state(self.milestones[i],self.milestones[i+1],u,self.times[i+1]-self.times[i])

    def deriv_state(self, t: float, endBehavior: str = 'halt') -> Vector:
        """Internal deriv, used on the underlying state representation"""
        i,u = self.getSegment(t,endBehavior)
        if i<0: return [0.0]*len(self.milestones[0])
        elif i+1>=len(self.milestones): return [0.0]*len(self.milestones[-1])
        return self.difference_state(self.milestones[i+1],self.milestones[i],u,self.times[i+1]-self.times[i])

    def interpolate_state(self, a: Vector, b: Vector, u: float, dt: float) -> Vector:
        """Can override this to implement non-cartesian spaces.
        Interpolates along the geodesic from a to b.  dt is the 
        duration of the segment from a to b"""
        return vectorops.interpolate(a,b,u)
    
    def difference_state(self, a: Vector, b: Vector, u: float, dt: float) -> Vector:
        """Subclasses can override this to implement non-Cartesian
        spaces.  Returns the time derivative along the geodesic from b to
        a, with time domain [0,dt].  In cartesian spaces, this is (a-b)/dt.
        
        Args:
            a (vector): the end point of the segment
            b (vector): the start point of the segment.
            u (float): the evaluation point of the derivative along the
                segment, with 0 indicating b and 1 indicating a
            dt (float): the duration of the segment from b to a.
        """
        return vectorops.mul(vectorops.sub(a,b),1.0/dt)
    
    def concat(self,
            suffix: 'Trajectory',
            relative: bool = False,
            jumpPolicy: str = 'strict'
        ) -> 'Trajectory':
        """Returns a new trajectory with another trajectory
        concatenated onto self.

        Args:
            suffix (Trajectory): the suffix trajectory
            relative (bool):  If True, then the suffix's time domain is shifted
                so that self.times[-1] is added on before concatenation.
            jumpPolicy (str):  If the suffix starts exactly at the existing trajectory's
                end time, then jumpPolicy is checked.  Can be:

                - 'strict': the suffix's first milestone has to be equal to the
                  existing trajectory's last milestone. Otherwise an exception
                  is raised.
                - 'blend': the existing trajectory's last milestone is
                  discarded.
                - 'jump': a discontinuity is added to the trajectory.

        """
        if self.__class__ is not suffix.__class__:
            raise ValueError("Can only concatenate like Trajectory classes: %s != %s"%(self.__class__.__name__,suffix.__class__.__name__))
        if not relative or len(self.times)==0:
            offset = 0
        else:
            offset = self.times[-1]
        if len(self.times)!=0:
            if suffix.times[0]+offset < self.times[-1]:
                raise ValueError("Invalid concatenation, suffix startTime precedes endTime")
            if suffix.times[0]+offset == self.times[-1]:
                #keyframe exactly equal; skip the first milestone
                #check equality with last milestone
                if jumpPolicy=='strict' and suffix.milestones[0] != self.milestones[-1]:
                    print("Suffix start:",suffix.milestones[0])
                    print("Self end:",self.milestones[-1])
                    raise ValueError("Concatenation would cause a jump in configuration")
                if jumpPolicy=='strict' or (jumpPolicy=='blend' and suffix.milestones[0] != self.milestones[-1]):
                    #discard last milestone of self
                    times = self.times[:-1] + [t+offset for t in suffix.times]
                    milestones = self.milestones[:-1] + suffix.milestones
                    return self.constructor()(times,milestones)
        times = self.times + [t+offset for t in suffix.times]
        milestones = self.milestones + suffix.milestones
        return self.constructor()(times,milestones)

    def insert(self, time: float) -> int:
        """Inserts a milestone and keyframe at the given time.  Returns the index of the new
        milestone, or if a milestone already exists, then it returns that milestone index.

        If the path is empty, the milestone is set to an empty list [].
        """
        if len(self.times) == 0:
            self.times = [time]
            self.milestones = [[]]
            return 0
        if time <= self.times[0]:
            if time < self.times[0]:
                self.times.insert(0,time)
                self.milestones.insert(0,self.milestones[0][:])
            return 0
        elif time >= self.times[-1]:
            if time > self.times[-1]:
                self.times.append(time)
                self.milestones.append(self.milestones[-1][:])
            return len(self.times)-1
        else:
            i,u = self.getSegment(time)
            assert i >= 0,"getSegment returned -1? something must be wrong with the times"
            if u == 0:
                return i
            elif u == 1:
                return i+1
            else:
                q = self.interpolate_state(self.milestones[i],self.milestones[i+1],u,self.times[i+1]-self.times[i])
                self.times.insert(i,time)
                self.milestones.insert(i,q)
                return i

    def split(self, time: float) -> Tuple['Trajectory','Trajectory']:
        """Returns a pair of trajectories obtained from splitting this
        one at the given time.

        Returns:
            A pair (prefix,suffix) satisfying prefix.endTime()==time,
            suffix.startTime()==time, and
            prefix.milestones[-1]==suffix.milestones[0]
        """
        if time <= self.times[0]:
            #split before start of trajectory
            return self.constructor()([time],[self.milestones[0]]),self.constructor()([time]+self.times,[self.milestones[0]]+self.milestones)
        elif time >= self.times[-1]:
            #split after end of trajectory
            return self.constructor()(self.times+[time],self.milestones+[self.milestones[-1]]),self.constructor()([time],[self.milestones[-1]])
        i,u = self.getSegment(time)
        assert i >= 0,"getSegment returned -1? something must be wrong with the times"
        #split in middle of trajectory
        splitpt = self.interpolate_state(self.milestones[i],self.milestones[i+1],u,self.times[i+1]-self.times[i])
        front = self.constructor()(self.times[:i+1],self.milestones[:i+1])
        back = self.constructor()(self.times[i+1:],self.milestones[i+1:])
        if u > 0:
            front.times.append(time)
            front.milestones.append(splitpt)
        if u < 1:
            back.times = [time] + back.times
            back.milestones = [splitpt] + back.milestones
        return (front,back)

    def before(self, time: float) -> 'Trajectory':
        """Returns the part of the trajectory before the given time"""
        return self.split(time)[0]

    def after(self, time: float) -> 'Trajectory':
        """Returns the part of the trajectory after the given time"""
        return self.split(time)[1]

    def splice(self,
            suffix: 'Trajectory',
            time: List[float] = None,
            relative: bool = False,
            jumpPolicy: str = 'strict'
        ) -> 'Trajectory':
        """Returns a path such that the suffix is spliced in at some time

        Args:
            suffix (Trajectory): the trajectory to splice in
            time (float, optional): determines when the splice occurs.
                The suffix is spliced in at the suffix's start time if time=None,
                or the given time if specified.
            jumpPolicy (str): if 'strict', then it is required that
                suffix(t0)=path(t0) where t0 is the absolute start time
                of the suffix.

        """
        offset = 0
        if time is None:
            time = suffix.times[0]
        if relative and len(self.times) > 0:
            offset = self.times[-1]
        time = time+offset
        before = self.before(time)
        return before.concat(suffix,relative,jumpPolicy)

    def constructor(self) -> Callable[[List,List],'Trajectory']:
        """Returns a "standard" constructor for the split / concat
        routines.  The result should be a function that takes two
        arguments: a list of times and a list of milestones."""
        return Trajectory

    def length(self, metric: Optional[MetricType] = None) -> float:
        """Returns the arc-length of the trajectory, according to the given
        metric.

        If metric = None, uses the "natural" metric for this trajectory,
        which is usually Euclidean.  Otherwise it is a function f(a,b)
        from configurations to nonnegative numbers.
        """
        if metric is None:
            metric = vectorops.distance
        return sum(metric(a,b) for a,b in zip(self.milestones[:-1],self.milestones[1:]))

    def discretize_state(self, dt: float) -> 'Trajectory':
        """Returns a copy of this but with uniformly defined milestones at
        resolution dt.  Start and goal are maintained exactly"""
        assert dt > 0,"dt must be positive"
        t = self.times[0]
        new_milestones = [self.milestones[0][:]]
        new_times = [self.times[0]]
        #TODO: (T/dt) log n time, can be done in (T/dt) time
        while t+dt < self.times[-1]:
            t += dt
            new_times.append(t)
            new_milestones.append(self.eval_state(t))
        if abs(t-self.times[-1]) > 1e-6:
            new_times.append(self.times[-1])
            new_milestones.append(self.milestones[-1][:])
        else:
            new_times[-1] = self.times[-1]
            new_milestones[-1] = self.milestones[-1][:]
        return self.constructor()(new_times,new_milestones)

    def discretize(self, dt: float) -> 'Trajectory':
        """Returns a trajectory, uniformly discretized at resolution dt, and
        with state-space the same as its configuration space. Similar to
        discretize, but if the state space is of higher dimension (e.g.,
        Hermite trajectories) this projects to a piecewise linear trajectory.
        """
        return self.discretize_state(dt)

    def remesh(self, newtimes: Iterable[float], tol: float=1e-6) -> Tuple['Trajectory',List[int]]:
        """Returns a path that has milestones at the times given in newtimes,
        as well as the current milestone times. 

        Args:
            newtimes: an iterable over floats.  It does not need to be sorted. 
            tol (float, optional): a parameter specifying how closely the
                returned path must interpolate the original path.  Old
                milestones will be dropped if they are not needed to follow the
                path within this tolerance.

        The end behavior is assumed to be 'halt'.

        Returns:
            A tuple (path,newtimeidx) where path is the remeshed path, and
            newtimeidx is a list of time indices satisfying
            ``path.times[newtimeidx[i]] = newtimes[i]``.
        """
        sorter = [(t,-1-i) for (i,t) in enumerate(self.times)]  + [(t,i) for (i,t) in enumerate(newtimes)]
        sorter = sorted(sorter)
        res = self.constructor()(None,None)
        res.times.append(sorter[0][0])
        res.milestones.append(self.milestones[0])
        #maybe a constant first section
        resindices = []
        i = 0
        while sorter[i][0] < self.startTime():
            if sorter[i][1] >= 0:
                resindices.append(0)
            i += 1
        if i != 0:
            res.times.append(self.startTime())
            res.milestones.append(self.milestones[0])
        firstold = 0
        lastold = 0
        while i < len(sorter):
            #check if we should add this
            t,idx = sorter[i]
            i+=1
            if idx >= 0:  #new time
                if t == res.times[-1]:
                    resindices.append(len(res.times)-1)
                    continue
                #it's a new mesh point, add it and check whether previous old milestones should be added
                if self.times[lastold] == t:
                    #matched the last old mesh point, no need to call eval_state()
                    newx = self.milestones[lastold]
                else:
                    newx = self.eval_state(t)
                res.times.append(t)
                res.milestones.append(newx)
                for j in range(firstold,lastold):
                    if self.times[j] == t:
                        continue
                    x = res.eval_state(self.times[j])
                    if vectorops.norm(self.difference_state(x,self.milestones[j],1.0,1.0)) > tol:
                        #add it
                        res.times[-1] = self.times[j]
                        res.milestones[-1] = self.milestones[j]
                        res.times.append(t)
                        res.milestones.append(newx)
                resindices.append(len(res.times)-1)
                firstold = lastold+1
            else:
                #mark the range of old milestones to add
                lastold = -idx-1
        for j in range(firstold,lastold):
            res.times.append(self.times[j])
            res.milestones.append(self.milestones[j])
        #sanity check
        for i in range(len(res.times)-1):
            assert res.times[i] < res.times[i+1]
        for i,idx in enumerate(resindices):
            assert newtimes[i] == res.times[idx],"Resindices mismatch? {} should index {} to {}".format(resindices,newtimes,res.times)
        return (res,resindices)

    def extractDofs(self,dofs:List[int]) -> 'Trajectory':
        """Extracts a trajectory just over the given DOFs.

        Args:
            dofs (list of int): the indices to extract.

        Returns:
            A copy of this trajectory but only over the given DOFs.
        """
        if len(self.times)==0:
            return self.constructor()
        n = len(self.milestones[0])
        for d in dofs:
            if abs(d) >= n:
                raise ValueError("Invalid dof")
        return self.constructor()([t for t in self.times],[[m[j] for j in dofs] for m in self.milestones])

    def stackDofs(self, trajs: List['Trajectory'], strict: bool = True) -> None:
        """Stacks the degrees of freedom of multiple trajectories together.
        The result is contained in self.

        All evaluations are assumed to take place with the 'halt' endBehavior.

        Args:
            trajs (list or tuple of Trajectory): the trajectories to stack
            strict (bool, optional): if True, will warn if the classes of the
                trajectories do not match self.
        """
        if not isinstance(trajs,(list,tuple)):
            raise ValueError("Trajectory.stackDofs takes in a list of trajectories as input")
        warned = not strict
        for traj in trajs:
            if traj.__class__ != self.__class__:
                if not warned:
                    warnings.warn("Trajectory.stackDofs is merging trajectories of different classes?")
                    warned = True
        alltimes = set()
        for traj in trajs:
            for t in traj.times:
                alltimes.add(t)
        self.times = sorted(alltimes)
        stacktrajs = [traj.remesh(self.times) for traj in trajs]
        for traj in stacktrajs:
            assert len(traj.milestones) == len(self.times)
        self.milestones = []
        for i,t in enumerate(self.times):
            self.milestones.append(sum([list(traj.milestones[i]) for traj in stacktrajs],[]))



class HermiteTrajectory(Trajectory):
    """A trajectory that performs cubic interpolation between prescribed
    segment endpoints and velocities. 

    The milestones (states) are given in phase space (x,dx).
    
    ``eval(t)`` returns the primary configuration x, and ``deriv(t)``
    returns the velocity dx.  To get acceleration, use ``accel(t)``.  To get
    the state space (x,dx), use ``eval_state(t)``.

    Args:
        times (list of float, optional): the knot points
        milestones (list of lists, optional): the milestones met at the knot
            points.
        dmilestones (list of lists, optional): the velocities (derivatives
            w.r.t time) at each knot point.  

    Possible constructor options are:

    - HermiteTrajectory(): empty trajectory
    - HermiteTrajectory(times,milestones): milestones contains 
      2N-D lists consisting of the concatenation of a point and its outgoing 
      velocity.
    - HermiteTrajectory(times,milestones,dmilestones):
      milestones and dmilestones each contain N-D lists defining the points and
      outgoing velocities.

    Note: the curve is assumed to be smooth. To make a non-smooth curve,
    duplicate the knot point and milestone, but set a different velocity
    at the copy.
    """
    def __init__(self,
            times: Optional[List[float]] = None,
            milestones: Optional[List[Vector]] = None,
            dmilestones: Optional[List[Vector]] = None
        ):
        if dmilestones is None:
            Trajectory.__init__(self,times,milestones)
        else:
            assert milestones != None
            #interpret as config/velocity
            self.times = times
            self.milestones = [q+dq for (q,dq) in zip(milestones,dmilestones)]

    def makeSpline(self,
            waypointTrajectory: Trajectory,
            preventOvershoot: bool = True,
            loop: bool = False
        ) -> None:
        """Computes natural velocities for a standard configuration-
        space Trajectory to make it smoother."""
        if loop and waypointTrajectory.milestones[-1] != waypointTrajectory.milestones[0]:
            raise ValueError("Asking for a loop trajectory but the endpoints don't match up")
        velocities = []
        t = waypointTrajectory
        d = len(t.milestones[0])
        if len(t.milestones)==1:
            velocities.append([0]*d)
        elif len(t.milestones)==2:
            if loop:
                v = [0]*d
            else:
                s = (1.0/(t.times[1]-t.times[0]) if (t.times[1]-t.times[0]) != 0 else 0)
                v = vectorops.mul(vectorops.sub(t.milestones[1],t.milestones[0]),s) 
            velocities.append(v)
            velocities.append(v)
        else:
            third = 1.0/3.0
            N = len(waypointTrajectory.milestones)
            if loop:
                timeiter = zip([-2]+list(range(N-1)),range(0,N),list(range(1,N))+[1])
            else:
                timeiter = zip(range(0,N-2),range(1,N-1),range(2,N))
            for p,i,n in timeiter:
                if p < 0:
                    dtp = t.times[-1] - t.times[-2]
                else:
                    dtp = t.times[i] - t.times[p]
                if n <= i:
                    dtn = t.times[1]-t.times[0]
                else:
                    dtn = t.times[n]-t.times[i]
                assert dtp >= 0 and dtn >= 0
                s = (1.0/(dtp+dtn) if (dtp+dtn) != 0 else 0)
                v = vectorops.mul(vectorops.sub(t.milestones[n],t.milestones[p]),s)
                if preventOvershoot:
                    for j,(x,a,b) in enumerate(zip(t.milestones[i],t.milestones[p],t.milestones[n])):
                        if x <= min(a,b):
                            v[j] = 0.0
                        elif x >= max(a,b):
                            v[j] = 0.0
                        elif v[j] < 0 and x - v[j]*third*dtp >= a:
                            v[j] = 3.0/dtp*(x-a)
                        elif v[j] > 0 and x - v[j]*third*dtp <= a:
                            v[j] = 3.0/dtp*(x-a)
                        elif v[j] < 0 and x + v[j]*third*dtn < b:
                            v[j] = 3.0/dtn*(b-x)
                        elif v[j] > 0 and x + v[j]*third*dtn > b:
                            v[j] = 3.0/dtn*(b-x)
                        
                velocities.append(v)
            if not loop:
                #start velocity as quadratic
                x2 = vectorops.madd(t.milestones[1],velocities[0],-third*(t.times[1]-t.times[0]))
                x1 = vectorops.madd(x2,vectorops.sub(t.milestones[1],t.milestones[0]),-third)
                v0 = vectorops.mul(vectorops.sub(x1,t.milestones[0]),3.0/(t.times[1]-t.times[0]))
                #terminal velocity as quadratic
                xn_2 = vectorops.madd(t.milestones[-2],velocities[-1],third*(t.times[-1]-t.times[-2]))
                xn_1 = vectorops.madd(xn_2,vectorops.sub(t.milestones[-1],t.milestones[-2]),third)
                vn = vectorops.mul(vectorops.sub(t.milestones[-1],xn_1),3.0/(t.times[-1]-t.times[-2]))
                velocities = [v0]+velocities+[vn]
        self.__init__(waypointTrajectory.times[:],waypointTrajectory.milestones,velocities)

 
    def waypoint(self,state):
        return state[:len(state)//2]

    def eval_state(self,t,endBehavior='halt'):
        """Returns the (configuration,velocity) state at time t."""
        return Trajectory.eval_state(self,t,endBehavior)

    def eval(self,t,endBehavior='halt'):
        """Returns just the configuration component of the result"""
        res = Trajectory.eval_state(self,t,endBehavior)
        return res[:len(res)//2]
    
    def deriv(self,t,endBehavior='halt'):
        """Returns just the velocity component of the result"""
        res = Trajectory.eval_state(self,t,endBehavior)
        return res[len(res)//2:]

    def eval_accel(self,t,endBehavior='halt') -> Vector:
        """Returns just the acceleration component of the derivative"""
        res = Trajectory.deriv_state(self,t,endBehavior)
        return res[len(res)//2:]

    def interpolate_state(self,a,b,u,dt):
        assert len(a)==len(b)
        x1,v1 = a[:len(a)//2],vectorops.mul(a[len(a)//2:],dt)
        x2,v2 = b[:len(b)//2],vectorops.mul(b[len(b)//2:],dt)
        x = spline.hermite_eval(x1,v1,x2,v2,u)
        dx = vectorops.mul(spline.hermite_deriv(x1,v1,x2,v2,u),1.0/dt)
        return x+dx
    
    def difference_state(self,a,b,u,dt):
        assert len(a)==len(b)
        x1,v1 = a[:len(a)//2],vectorops.mul(a[len(a)//2:],dt)
        x2,v2 = b[:len(b)//2],vectorops.mul(b[len(b)//2:],dt)
        dx = vectorops.mul(spline.hermite_deriv(x1,v1,x2,v2,u,order=1),1.0/dt)
        ddx = vectorops.mul(spline.hermite_deriv(x1,v1,x2,v2,u,order=2),1.0/pow(dt,2))
        return dx+ddx

    def discretize(self,dt):
        """Creates a discretized piecewise linear Trajectory in config space
        that approximates this curve with resolution dt.
        """
        res = self.discretize_state(dt)
        n = len(res.milestones[0])//2
        return Trajectory(res.times,[m[:n] for m in res.milestones])

    def length(self) -> float:
        """Returns an upper bound on length given by the Bezier property. 
        Faster than calculating the true length.  To retrieve an approximation
        of true length, use self.discretize(dt).length().
        """
        n = len(self.milestones[0])//2
        third = 1.0/3.0
        def distance(x,y):
            cp0 = x[:n]
            cp1 = vectorops.madd(cp0,x[n:],third)
            cp3 = y[:n]
            cp2 = vectorops.madd(cp3,y[n:],-third)
            return third*vectorops.norm(x[n:]) + vectorops.distance(cp1,cp2) + third*vectorops.norm(y[n:])
        return Trajectory.length(self,distance)

    def checkValid(self):
        Trajectory.checkValid(self)
        for m in self.milestones:
            if len(m)%2 != 0:
                raise ValueError("Milestone length isn't even?: {}".format(len(m)))

    def extractDofs(self,dofs) -> 'HermiteTrajectory':
        """Extracts a trajectory just over the given DOFs.

        Args:
            dofs (list of int): the (primary) indices to extract. Each entry
            must be < len(milestones[0])/2.

        Returns:
            A copy of this trajectory but only over the given DOFs.
        """
        if len(self.times)==0:
            return self.constructor()
        n = len(self.milestones[0])//2
        for d in dofs:
            if abs(d) >= n:
                raise ValueError("Invalid dof")
        return self.constructor()([t for t in self.times],[[m[j] for j in dofs] + [m[n+j] for j in dofs] for m in self.milestones])

    def stackDofs(self,trajs,strict=True) -> None:
        """Stacks the degrees of freedom of multiple trajectories together.
        The result is contained in self.

        All evaluations are assumed to take place with the 'halt' endBehavior.

        Args:
            trajs (list or tuple of HermiteTrajectory): the trajectories to 
                stack
            strict (bool, optional): ignored. Will always warn for invalid
                classes.
        """
        if not isinstance(trajs,(list,tuple)):
            raise ValueError("HermiteTrajectory.stackDofs takes in a list of trajectories as input")
        for traj in trajs:
            if not isinstance(traj,HermiteTrajectory):
                raise ValueError("Can't stack non-HermiteTrajectory objects into a HermiteTrajectory")
        alltimes = set()
        for traj in trajs:
            for t in traj.times:
                alltimes.add(t)
        self.times = sorted(alltimes)
        stacktrajs = [traj.remesh(self.times) for traj in trajs]
        for traj in stacktrajs:
            assert len(traj.milestones) == len(self.times)
        self.milestones = []
        for i,t in enumerate(self.times):
            q = []
            v = []
            for traj in stacktrajs:
                n = len(traj.milestones[i])//2
                q += list(traj.milestones[i][:n])
                v += list(traj.milestones[i][n:])
            self.milestones.append(q + v)

    def constructor(self):
        return HermiteTrajectory
