
from scipy.linalg import expm as expM
from scipy import pi
import ckbot.logcal as L

c = L.cluster()
c.populate(1)
motors = [motor[1] for motor in c.items()]

def seToSE( x ):
    """
    Convert a twist (a rigid velocity, element of se(3)) to a rigid
    motion (an element of SE(3))

    INPUT:
        x -- 6 sequence
    OUTPUT:
        result -- 4 x 4

    """
    x = asarray(x,dtype=float)
    if x.shape != (6,):
        raise ValueError("shape must be (6,); got %s" % str(x.shape))
    #
    return expM(screw(x))

def screw( v ):
    """
    Convert a 6-vector to a screw matrix

    The function is vectorized, such that:
    INPUT:
        v -- N... x 6 -- input vectors
    OUTPUT:
        N... x 4 x 4
    """
    v = asarray(v)
    z = zeros_like(v[0,...])
    return array([
            [ z, -v[...,5], v[...,4], v[...,0] ],
            [ v[...,5],    z,-v[...,3], v[...,1] ],
            [-v[...,4],    v[...,3], z, v[...,2] ],
            [ z,                 z,                z, z] ])

def unscrew( S ):
    """
    Convert a screw matrix to a 6-vector
    
    The function is vectorized, such that:
    INPUT:
        S -- N... x 4 x 4 -- input screws
    OUTPUT:
        N... x 6
    
    This is the "safe" function -- it tests for screwness first.
    Use unscrew_UNSAFE(S) to skip this check
    """
    S = asarray(S)
    assert allclose(S[...,:3,:3].transpose(0,1),-S[...,:3,:3]),"S[...,:3,:3] is skew"
    assert allclose(S[...,3,:],0),"Bottom row is 0"
    return unscrew_UNSAFE(S)

def jacobian_cdas( func, scl, lint=0.8, tol=1e-12, eps = 1e-30, withScl = False ):
    """Compute Jacobian of a function based on auto-scaled central differences.
    
    INPUTS:
        func -- callable -- K-vector valued function of a D-dimensional vector
        scl -- D -- vector of maximal scales allowed for central differences
        lint -- float -- linearity threshold, in range 0 to 1. 0 disables
            auto-scaling; 1 requires completely linear behavior from func
        tol -- float -- minimal step allowed
        eps -- float -- infinitesimal; must be much smaller than smallest change in
            func over a change of tol in the domain.
        withScl -- bool -- return scales together with Jacobian
    
    OUTPUTS: jacobian function
        jFun: x --> J (for withScale=False)
        jFun: x --> J,s (for withScale=True)

        x -- D -- input point
        J -- K x D -- Jacobian of func at x
        s -- D -- scales at which Jacobian holds around x
    """
    scl = abs(asarray(scl).flatten())
    N = len(scl)
    lint = abs(lint)
    def centDiffJacAutoScl( arg ):
        """
        Algorithm: use the value of the function at the center point
            to test linearity of the function. Linearity is tested by
            taking dy+ and dy- for each dx, and ensuring that they
            satisfy lint<|dy+|/|dy-|<1/lint
        """
        x0 = asarray(arg).flatten()
        y0 = func(x0)
        s = scl.copy()
        #print "Jac at ",x0
        idx = slice(None)
        dyp = empty((len(s),len(y0)),x0.dtype)
        dyn = empty_like(dyp)
        while True:
            #print "Jac iter ",s
            d0 = diag(s)
            dyp[idx,:] = [ func(x0+dx)-y0 for dx in d0[idx,:] ]
            dypc = dyp.conj()
            dyn[idx,:] = [ func(x0-dx)-y0 for dx in d0[idx,:] ]
            dync = dyn.conj()
            dp = sum(dyp * dypc,axis=1)
            dn = sum(dyn * dync,axis=1)
            nul = (dp == 0) | (dn == 0)
            if any(nul):
                s[nul] *= 1.5
                continue
            rat = dp/(dn+eps)
            nl = ((rat<lint) | (rat>(1.0/lint)))
            # If no linearity violations found --> done
            if ~any(nl):
                break
            # otherwise -- decrease steps
            idx, = nl.flatten().nonzero()
            s[idx] *= 0.75
            # Don't allow steps smaller than tol
            s[idx[s[idx]<tol]] = tol
            if all(s[idx]<tol):
                break
        res = ((dyp-dyn)/(2*s[:,newaxis])).T
        if withScl:
            return res, s
        return res
    return centDiffJacAutoScl


class Arm( object ):
    def __init__(self):
        # arm geometry to draw
        # size of the cubic shape with pyramid along x axis
        # sizes are given in diameter along axis (radius to corner of square)
        d_x = 1.0
        d_y = 1.0
        d_z = 1.0
        # length of pyramid
        d = 0.2
        penta = asarray([
            [ 0,     d,   d_x,    d_x,      d, 0],
            [ 0, d_y/2, d_y/2, -d_y/2, -d_y/2, 0],
            [ 0,     0,     0,      0,      0, 0],
            [ 1,     1,     1,      1,      1, 1],
        ]).T
        sqr = asarray([
            [     d,     d,      d,      d,     d,   d_x,   d_x,    d_x,    d_x,   d_x ],
            [ d_y/2,     0, -d_y/2,      0, d_y/2, d_y/2,     0, -d_y/2,      0, d_y/2 ],
            [     0, d_z/2,      0, -d_z/2,     0,     0, d_z/2,      0, -d_z/2,     0 ],
            [     1,     1,      1,      1,     1,     1,     1,      1,      1,     1 ],
        ]).T
        geom = concatenate([
            penta, penta[:,[0,2,1,3]], sqr,
        ], axis=0)
        self.geom = [asarray([[0,0,0,1]]).T ]

        tw = [
            asarray([0, 0,  0, 0, 0, 1]),
            asarray([0, 20, 0, 1, 0, 0]),
            asarray([0, -34, 0, -1, 0, 0]),
        ]
        tw = tw[:3]

        transforms = [
           [[ 0, 0, 6.8, 0],
            [ 0, 6.8, 0, 0],
            [ 24.5, 0, 0, 0],
            [ 0, 0, 0, 1]],
           [[ 8.8, 0, 0, 3],
            [ 0, 6.8, 0, 0],
            [ 0, 0, 6.8, 20],
            [ 0, 0, 0, 1]],
           [[ -8.8, 0, 0, 3],
            [ 0, 6.8, 0, 0],
            [ 0, 0, 6.8, 34],
            [ 0, 0, 0, 1]],
        ]
        transforms = transforms[:3]
        for transformer in transforms:
            self.geom.append(dot(transformer, geom.T))
        print self.geom

        '''
        # link lengths
        self.ll = asarray([1,3,3,3,3])

        tw = []
        LL = 0
        our_twists = asarray([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
        ])
        for n,ll in enumerate(self.ll):
            self.geom.append(
                ( asarray([ll,1,1,1])*geom+[LL,0,0,0] ).T
            )
            #w = asarray([0,(n+1) % 2, n % 2])
            w = our_twists[n]
            v = -cross(w,[LL,0,0])
            tw.append( concatenate([v,w],0) )
            LL += ll
            print tw
        '''
        self.tw = asarray(tw)
        #self.tool = asarray([LL,0,0,1]).T
        self.tool = asarray([-8.5, 0.0, 20+14+28+7.0, 1.0]).T
        # overwrite method with jacobian function
        self.getToolJac = jacobian_cdas(
            self.getTool, ones(self.tw.shape[0])*0.05
        )

    def at( self, ang ):
        """
        Compute the rigid transformations for a 3 segment arm
        at the specified angles
        """
        ang = asarray(ang)[:,newaxis]
        tw = ang * self.tw
        A = [identity(4)]
        for twi in tw:
            M = seToSE(twi)
            A.append(dot(A[-1],M))
        return A

    def getTool( self, ang ):
        """
        Get "tool tip" position in world coordinates
        """
        M = self.at(ang)[-1]
        return dot(M, self.tool)

    def getToolJac( self, ang ):
        """
        Get "tool tip" Jacobian by numerical approximation

        NOTE: implementation is a placeholder. This method is overwritten
        dynamically by __init__() to point to a jacobian_cdas() function
        """
        raise RuntimeError("uninitialized method called")

    def plotIJ( self, ang, axI=0, axJ=1 ):
        """
        Display the specified axes of the arm at the specified set of angles
        """
        A = self.at(ang)
        for a,g in zip(A, self.geom):
            ng = dot(a,g)
            plot( ng[axI,:], ng[axJ,:], '.-' )
        tp = dot(a, self.tool)
        plot( tp[axI], tp[axJ], 'hk' )
        plot( tp[axI], tp[axJ], '.y' )


    def plot3D( self, ang ):
        #ax = [-90,90,-90,90]
        ax = [-40,40,-40,40]
        subplot(2,2,1)
        self.plotIJ(ang,0,1)
        axis('equal')
        #axis(ax)
        grid(1)
        xlabel('X'); ylabel('Y')
        subplot(2,2,2)
        self.plotIJ(ang,1,2)
        axis('equal')
        #axis(ax)
        grid(1)
        xlabel('Y'); ylabel('Z')
        subplot(2,2,3)
        self.plotIJ(ang,0,2)
        axis('equal')
        #axis(ax)
        grid(1)
        xlabel('X'); ylabel('Z')

# Clip value to lower and upper bounds
def clip(value, lower, upper):
    return min(max(value, lower), upper)

def center(value, lower, upper):
    while value > upper:
        value = value - upper
    while value < lower:
        value = value - lower

def set_motor_ang(motor, ang):
    fractional_angle = center(ang, -pi, pi) / pi
    if ang < 0:
        pos = -fractional_angle * 1023 + 1024
    else:
        pos = fractional_angle * 1023
    motor.pna.mem_write_fast(motor.mcu.goal_position, int(round(pos)))

def example():
    """
    Run an example of a robot arm

    This can be steered via inverse Jacobian, or positioned.
    """
    a = Arm()
    f = gcf()
    # ang = [0,0,0,0,0,0]
    ang = [0,0,0]
    while 1:
        f.set(visible=0)
        clf()
        a.plot3D(ang)
        f.set(visible=1)
        draw()
        print "Angles: ",ang
        d = input("direction as list / angles as tuple?>")
        if type(d) == list:
            Jt = a.getToolJac(ang)
            ang = ang + dot(pinv(Jt)[:,:len(d)],d)
        else:
            ang = d

        for a, m in zip(ang, motors):
            set_motor_ang(m, a)
