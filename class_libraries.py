
import numpy as np

## get gradient of f to state x and input u
def prtl_f_prtl_x_and_u( theta_B, omega_B, a_B, x_mB, grvty, dt ):
    temp = a_B[0] * np.sin(theta_B) - a_B[1] * np.cos(theta_B) - grvty * np.cos(theta_B)
    
    return np.array([[            1, dt, dt**2/2,    0,              0 ],
                     [            0,  1,      dt,    0,              0 ],
                     [   omega_B**2,  0,       0, temp, 2*x_mB*omega_B ],
                     [            0,  0,       0,    1,             dt ]], dtype=float )


'''
member function:
    __init__( self, S_DIM, A_DIM, C_LR, A_LR, A_UPDATE_STEPS, C_UPDATE_STEPS )
    implmnt(  )
'''
class ILQR(object):

    def __init__( self, itr_num, dim_x, dim_u, w_x, w_omega, grvty, dt ):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.itr_num = itr_num
        self.w_x = w_x
        self.w_omega = w_omega
        self.grvty = grvty
        self.dt = dt
        
        self.F_set = np.zeros( ( itr_num, dim_x, dim_x + dim_u ), dtype=float ) # dim(F): n * (n+m)
        self.f_set = np.zeros( ( itr_num, dim_x, 1 ), dtype=float ) # dim(f): n * 1 
        self.K_set = np.zeros( ( itr_num, dim_u, dim_x ), dtype=float ) # dim(K): m * n
        self.k_set = np.zeros( ( itr_num, dim_u, 1 ), dtype=float ) # dim(k): m * 1
        self.V_set = np.zeros( ( itr_num, dim_x, dim_x ), dtype=float ) # dim(V): n * n
        self.V_set[-1,0,0] = w_x
        self.v_set = np.zeros( ( itr_num, dim_x, 1 ), dtype=float ) # dim(v): n * 1
        self.Q_set = np.zeros( ( itr_num, dim_x + dim_u, dim_x + dim_u ), dtype=float ) # dim(Q): (n+m) * (n+m)
        self.q_set = np.zeros( ( itr_num, dim_x + dim_u, 1 ), dtype=float ) # dim(q): (n+m) * 1
        
               
    def implmnt( self,  theta_B, omega_B, a_B, x_mB, v_mB, a_mB ):
    
        for i in range( self.itr_num-2, -1, -1 ):
            
            self.F_set[i] = prtl_f_prtl_x_and_u( theta_B[i], omega_B[i], a_B[i], x_mB[i], self.grvty, self.dt  ) 
            self.f_set[i] = np.array([ [x_mB[i+1]], [v_mB[i+1]], [a_mB[i+1]], [theta_B[i+1]]]) - self.F_set[i].dot( np.array([ [x_mB[i]], [v_mB[i]], [a_mB[i]], [theta_B[i]], [omega_B[i]] ]) )
            
            self.Q_set[i] = self.F_set[i].T.dot( self.V_set[i+1].dot( self.F_set[i] ) )
            self.Q_set[i,0,0] = self.Q_set[i,0,0] + self.w_x # specialized for this case
            self.Q_set[ i, self.dim_x, self.dim_x ] = self.Q_set[ i, self.dim_x, self.dim_x ] + self.w_omega # specialized for this case
            self.q_set[i] = self.F_set[i].T.dot( self.V_set[i+1].dot( self.f_set[i] ) ) + self.F_set[i].T.dot( self.v_set[i+1] )
            
            self.K_set[i] = - self.Q_set[ i, self.dim_x:, :self.dim_x ] / ( self.Q_set[ i, self.dim_x:, self.dim_x: ] + 1e-9 )
            self.k_set[i] = - self.q_set[ i, self.dim_x:, : ] / ( self.Q_set[ i, self.dim_x:, self.dim_x: ] + 1e-9 )            
            
            self.V_set[i] = ( self.Q_set[ i,:self.dim_x, :self.dim_x ]
                          + self.Q_set[ i, :self.dim_x, self.dim_x: ].dot( self.K_set[i] )
                          + self.K_set[i].T.dot( self.Q_set[ i, self.dim_x: , :self.dim_x ] )
                          + self.K_set[i].T.dot( self.Q_set[ i, self.dim_x: , self.dim_x: ].dot( self.K_set[i] ) ) 
                          )
            
            self.v_set[i] = ( self.q_set[ i, :self.dim_x, :]
                          + self.Q_set[ i, :self.dim_x, self.dim_x: ].dot( self.k_set[i] )
                          + self.K_set[i].T.dot( self.q_set[ i, self.dim_x: , : ] )
                          + self.K_set[i].T.dot( self.Q_set[ i, self.dim_x: , self.dim_x: ].dot( self.k_set[i] ) ) 
                          )
        
        return self.K_set, self.k_set


