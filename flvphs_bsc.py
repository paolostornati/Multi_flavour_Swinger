#!/usr/bin/env python
# coding: utf-8

# In[830]:


import scipy         
#import datetime
from scipy.linalg import expm,det
import scipy.sparse as sparse
from quspin.operators import hamiltonian, commutator, exp_op # Hamiltonians and operators
from quspin.basis import tensor_basis, spin_basis_1d # bases
import numpy as np # general math functions
import matplotlib.pyplot as plt # plotting library
#from numba import jit
from scipy.integrate import solve_ivp,odeint
#from functools import reuce
import sys


# In[831]:


class constructor_LGT:
    
    def __init__(self,N,L,g,m_1,m_2,pen,bc):
        
        
        self.no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
        self.l = 1
        self.N = N #number of lattice sites
        self.L = L #number of links
        self.g = g #coupling constant
        self.m_1 = m_1 #mass of the 1st specie
        self.m_2 = m_2 #mass of the 1st specie
        self.pen = pen #lagrange multiplier in the penalty term
        self.bc = bc # 0 - non-twisted bc, 1 - twisted bc
        
        self.basis = self.initialise_basis()
        
        self.hE_coeff_ar = self.hE_coeff() #Taylor coeff. of (-1)^E
        self.Proj_even_coeff_ar = self.Proj_even_coeff() #Taylor coeff. of P_1(n = even)
        self.Proj_odd_coeff_ar = self.Proj_odd_coeff() #Taylor coeff. of P_1(n = odd)
        self.Proj_0_even_coeff_ar = self.Proj_0_even_coeff() #Taylor coeff. of P_1(n = even)
        self.Proj_0_odd_coeff_ar = self.Proj_0_odd_coeff() #Taylor coeff. of P_0(n = odd)
        
        self.Gauss_law_ar = [] #Gauss laws
        self.Proj_ar = [] #Projectors on 1
        self.Proj_0_ar = [] #Projectors on 0
        
        self.h_E_ar = [] #(-1)^E
        self.h_U_ar = [] #U
        self.h_U_dag_ar = [] #U^dagger
        self.interaction_1_ar = [] #\sigma^+ U \sigma^- + h.c.

        for i in range(N):
            self.Gauss_law_ar.append(self.Gauss_law(i))
            self.Proj_ar.append(self.Proj(i))
            self.Proj_0_ar.append(self.Proj_0(i))
            self.h_E_ar.append(self.h_E(i))
            self.h_U_ar.append(self.h_U(i))
            self.h_U_dag_ar.append(self.h_U_dag(i))
            self.interaction_1_ar.append(self.interaction_1(i))
            
            
    def initialise_basis(self):
        basis_link =spin_basis_1d(L=self.L,S = str(self.l)) #spin - d on the links
        basis_aux =spin_basis_1d(L=self.N,S = "1/2",pauli = -1) #spin - 1/2 on the sites
        basis=tensor_basis(basis_link,basis_aux)      
        
        return basis
    
    
    
    def check_hermitian(self,a):
        
        norm = scipy.sparse.linalg.norm(a-a.T.conj())
        
        if norm <= 1e-3:
            return("True")
        
        else:
            print("The norm is not zero:",norm)
            return("False")
            
    
        



        
    def hE_coeff(self):
        A = np.zeros((int(2*self.l)+1,int(2*self.l)+1), dtype = complex)
        for i in range(int(2*self.l)+1):
            for j in range(int(2*self.l)+1):
                A[i][j] = (-self.l+i)**j
        a = np.zeros(int(2*self.l)+1,dtype = complex)
        for i in range(int(2*self.l)+1):
            a[i] = (-1)**(-self.l+i)
        solution = scipy.linalg.solve(A,a)
        x = np.zeros(int(2*3/2)+1, dtype = complex)
        for i in range(int(2*self.l)+1):
            x[i] = solution[i]

        return x
    
    def Proj_even_coeff(self):

        B = np.zeros((int(4*self.l)+2,int(4*self.l)+2))
        for i in range(int(4*self.l)+2):
            for j in range(int(4*self.l)+2):
                B[i][j] = (-int(2*self.l)+i-1)**j
        b = np.zeros(int(4*self.l)+2)
        b[int(2*self.l)+2] = 1.
        solution = scipy.linalg.solve(B,b)
        y = np.zeros(int(4*3/2)+2)
        for i in range(int(4*self.l)+2):
            y[i] = solution[i]
        return y
        
    def Proj_odd_coeff(self):
        C = np.zeros((int(4*self.l)+2,int(4*self.l)+2))
        for i in range(int(4*self.l)+2):
            for j in range(int(4*self.l)+2):
                C[i][j] = (-int(2*self.l)+i+1)**j
        c = np.zeros(int(4*self.l)+2)
        c[int(2*self.l)] = 1.
        solution = scipy.linalg.solve(C,c)
        z = np.zeros(int(4*3/2)+2)
        for i in range(int(4*self.l)+2):
            z[i] = solution[i]

        return z

    def Proj_0_even_coeff(self):
        D = np.zeros((int(4*self.l)+2,int(4*self.l)+2))
        for i in range(int(4*self.l)+2):
            for j in range(int(4*self.l)+2):
                D[i][j] = (-int(2*self.l)+i-1)**j
        d = np.zeros(int(4*self.l)+2)
        d[int(2*self.l)+1] = 1.
        solution = scipy.linalg.solve(D,d)
        yy = np.zeros(int(4*3/2)+2)
        for i in range(int(4*self.l)+2):
            yy[i] = solution[i]
        return yy
    

    def Proj_0_odd_coeff(self):
        E = np.zeros((int(4*self.l)+2,int(4*self.l)+2))
        for i in range(int(4*self.l)+2):
            for j in range(int(4*self.l)+2):
                E[i][j] = (-int(2*self.l)+i+1)**j
        e = np.zeros(int(4*self.l)+2)
        e[int(2*self.l)-1] = 1.
        solution = scipy.linalg.solve(E,e)
        zz = np.zeros(int(4*3/2)+2)
        for i in range(int(4*self.l)+2):
            zz[i] = solution[i]
        return zz
    

    def Gauss_law(self,n):

        if n%2 == 0:
            liste = [[1.,n]]
            liste_m = [[-1.,(n-1)%self.L]]
            liste_p5 = [[-0.5,n]]

            gauss_law_map = [

                ["z|",liste],
                ["z|",liste_m],
                ["|I",liste_p5],
                ["|z",liste_p5],
            ]

            gauss_law = hamiltonian(gauss_law_map,dynamic_list=[],basis=self.basis,**self.no_checks)

        else:

            liste = [[1.,n]]
            liste2 = [[2.,n]]
            liste_m = [[-1.,n-1]]
            liste_p5 = [[-0.5,n]]

            gauss_law_map = [

                ["z|",liste],
                ["z|",liste_m],
                ["I|",liste2],
                ["|I",liste_p5],
                ["|z",liste_p5],
            ]

            gauss_law = hamiltonian(gauss_law_map,dynamic_list=[],basis=self.basis,**self.no_checks)

        return gauss_law.tocsc()
    
    

    def Proj(self,n):
        Proj = sparse.csc_matrix((self.basis.Ns,self.basis.Ns))
        g = self.Gauss_law_ar[n]
        if n%2 == 0:
            for j in range(int(4*self.l)+2):
                Proj += self.Proj_even_coeff_ar[j]*g**j
        else:
            for j in range(int(4*self.l)+2):
                Proj += self.Proj_odd_coeff_ar[j]*g**j
        return Proj

    def Proj_0(self,n):
        Proj = sparse.csc_matrix((self.basis.Ns,self.basis.Ns))
        g = self.Gauss_law_ar[n]
        if n%2 == 0:
            for j in range(int(4*self.l)+2):
                Proj += self.Proj_0_even_coeff_ar[j]*g**j
        else:
            for j in range(int(4*self.l)+2):
                Proj += self.Proj_0_odd_coeff_ar[j]*g**j
        return Proj
    
    
    
    def h_E(self,n):

        const_term = [[self.hE_coeff_ar[0],n]]
        linear_term = [[self.hE_coeff_ar[1],n]]
        quadratic_term = [[self.hE_coeff_ar[2],n,n]]

        prefactor_E = [
                ["I|", const_term],
                ["z|", linear_term],
                ["zz|", quadratic_term],

            ]
        hE = hamiltonian(prefactor_E,dynamic_list = [],basis=self.basis,**self.no_checks)

        return hE.tocsc()
    
    
    
    def h_U(self,n):

        op_p_val = [[1.,n]]

        op_p_map = [
            ["+|",op_p_val],
        ]

        hU = hamiltonian(op_p_map,dynamic_list = [],basis=self.basis,**self.no_checks)

        return hU.tocsc()
 
    def h_U_dag(self,n):

        op_p_val = [[1.,n]]

        op_p_map = [
            ["-|",op_p_val],
        ]

        hU = hamiltonian(op_p_map,dynamic_list = [],basis=self.basis,**self.no_checks)

        return hU.tocsc()
    
    

    def interaction_1(self,n):

        interaction_term = [[1.,n,n,(n+1)%self.L]]

        int_1_map = [
            ["+|+-",interaction_term],
            ["-|-+",interaction_term],

        ]
        hint = hamiltonian(int_1_map,dynamic_list = [],basis=self.basis,**self.no_checks)

        return hint.tocsc()
 

        
    def hamiltonian_matrix(self):
        
        kin_energy = [[0.5*self.g**2,i,i] for i in range(self.L)]
        mass_term_diff = [[0.5*(self.m_1-self.m_2)*(-1)**i,i] for i in range(self.N)]
        mass_term_2 = [[2*self.m_2*(-1)**i,i] for i in range(self.L)]


        hamiltonian_map = [

            ["zz|",kin_energy],

            ["|z",mass_term_diff],
            ["|I",mass_term_diff],

            ["z|",mass_term_2],
        ]

        Ham_part_1 = hamiltonian(hamiltonian_map,dynamic_list = [],basis=self.basis,**self.no_checks)
        ham_part_1_matrix = Ham_part_1.tocsc()  # E^2 energy term + mass term
        
        
        interaction_ham_1 = 0 #hopping term for the first specie
        interaction_ham_2 = 0 #hopping term for the second specie


        for i in range(self.L-1):
            interaction_ham_1 += 1/(2*np.sqrt(self.l*(self.l+1)))*(self.h_E_ar[(i-1)%self.L]@self.interaction_1_ar[i])
            interaction_ham_2 += 1/(2*np.sqrt(self.l*(self.l+1)))*self.h_E_ar[i+1]@(self.Proj_ar[i]@self.h_U_ar[i]@self.Proj_ar[i+1]+self.Proj_ar[i+1]@self.h_U_dag_ar[i]@self.Proj_ar[i])

        interaction_ham_1 += 1/(2*np.sqrt(self.l*(self.l+1)))*(self.h_E_ar[self.L-2]@self.interaction_1_ar[self.L-1])
        interaction_ham_2 += (-1)**self.bc*1/(2*np.sqrt(self.l*(self.l+1)))*self.h_E_ar[0]@(self.Proj_ar[self.L-1]@self.h_U_ar[self.L-1]@self.Proj_ar[0]+self.Proj_ar[0]@self.h_U_dag_ar[self.L-1]@self.Proj_ar[self.L-1])

        Hamiltonian_two_flavours = ham_part_1_matrix + interaction_ham_1 + interaction_ham_2 - self.N/2*(self.m_1+self.m_2)*sparse.identity(self.basis.Ns)
        
        G = 0 #penalty term due to the Gauss laws

        if self.pen != 0:
            for i in range(self.N):
                g_ham = self.Gauss_law_ar[i]
                G += (g_ham*(g_ham-sparse.identity(self.basis.Ns)))**2


        return Hamiltonian_two_flavours + self.pen*G
    
    
    def electric_field_matrix(self,n):
        
        op_e_val = [[1.,n]]

        op_e_map = [
            ["z|",op_e_val],
        ]

        hEl = hamiltonian(op_e_map,dynamic_list = [],basis=self.basis,**self.no_checks)

        return hEl.tocsc()
    
    def fermion_number_matrix(self,n,specie):
        
        number_1_val = [[0.5,n]]

        number_1_map = [

            ["|z",number_1_val],
            ["|I",number_1_val],
        ]

        ham_number = hamiltonian(number_1_map,dynamic_list = [],basis=self.basis,**self.no_checks)  
        
        ham_number_m = ham_number.tocsc()
        
        if specie == 1:
            ham_number_m = self.Gauss_law(n)
            
        return (-1)**n*ham_number_m
    
    
    
    
    


# In[832]:


def sort_eigenval_and_eigenvec(x,y):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
        (y[:,i], y[:,swap]) = (y[:,swap], y[:,i])
    return x,y


# In[833]:



def main(N,L,g,m_1,m_2,pen,bc,Nst):


    LGT_object = constructor_LGT(N,L,g,m_1,m_2,pen,bc)
    eigenval, eigenvec = scipy.sparse.linalg.eigsh(LGT_object.hamiltonian_matrix(),k = Nst,which = "SA")
    
    eigenval_s, eigenvec_s = sort_eigenval_and_eigenvec(eigenval, eigenvec)
        
    psi_ground = eigenvec_s[:,0]
    
    
    electric_field = []
    mass_term_1 = []
    mass_term_2 = []
    

        
    for i in range(N):
        
        electric_field.append(np.real(np.conj(psi_ground)@LGT_object.electric_field_matrix(i)@psi_ground))
        mass_term_1.append(np.real(np.conj(psi_ground)@LGT_object.fermion_number_matrix(i,0)@psi_ground))
        mass_term_2.append(np.real(np.conj(psi_ground)@LGT_object.fermion_number_matrix(i,1)@psi_ground))
                
        
        
    f = open("twflph_N="+str(N)+"_m_1="+str(m_1)+"_m_2="+str(m_2)+"_g="+str(g)+"_bc="+str(bc)+".txt", "w")
    
    for l in range(N):
        f.write(str(l)+" "+str(electric_field[l])+" "+str(mass_term_1[l])+" "+str(mass_term_2[l])+"\n")
        
    f.close()

        
        
    pass
    
    
    
    



# In[ ]:


main(int(sys.argv[1]),int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]))
print("Done!")