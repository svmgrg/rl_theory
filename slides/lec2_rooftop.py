import numpy as np
import matplotlib.pyplot as plt

import pdb

def calc_P_pred(P, pi):
    P_pred = np.array([[P[0, 0] * pi[0] + P[1, 0] * pi[1],
                        P[0, 1] * pi[0] + P[1, 1] * pi[1]],
                       [P[2, 0] * pi[2],
                        P[2, 1] * pi[2]]])
    return P_pred

def calc_V_Q(Q):
    V_Q = np.array([[max(Q[0, 0], Q[1, 0])],
                    [Q[2, 0]]])
    return V_Q

def calc_pi_Q(Q):
    pi = np.zeros(3)
    
    if Q[0, 0] > Q[1, 0]:
        action = 0
    else:
        action = 1

    pi[action] = 1 # which action to take in state 1
    pi[2] = 1 # always take action 1 in state 2

    return pi

def calc_r_pi(reward, pi):
    r_pi = np.array([[reward[0, 0] * pi[0] + reward[1, 0] * pi[1]],
                     [reward[2, 0] * pi[2]]])
    return r_pi

gamma = 0.9
I = np.eye(2)

reward = np.array([[0.1],
                   [0.5],
                   [0.2]])

P = np.array([[1, 0],
              [0.5, 0.5],
              [0.2, 0.8]])

pi = np.array([0.5, 0.5, 1])

#----------------------------------------------------------------------
# Contour Plot
#----------------------------------------------------------------------
plt.figure(figsize=(5, 5))
num_partitions = 11
x = np.linspace(0, 1, num_partitions)
y = np.linspace(0, 1, num_partitions)
X, Y = np.meshgrid(x, y)
Z = np.zeros((num_partitions, num_partitions))

for i in range(num_partitions):
    for j in range(num_partitions):
        pi = np.array([X[i, j], Y[i, j], 1])
        if pi[0] + pi[1] != 1:
            continue
        P_pred = calc_P_pred(P, pi)
        r_pi = calc_r_pi(reward, pi)
        V = (1 - gamma) * np.matmul(np.linalg.inv(I - gamma * P_pred),
                                    r_pi)
        Z[i, j] = V[0][0]
        print(V[0][0])
        plt.scatter(pi[0], pi[1], s=np.exp(V[0]*10)*10, color='b')
        plt.text(pi[0]+0.03, pi[1]+0.03, '{:0.2f}'.format(V[0][0]))
        
plt.scatter(0, 0, s=0, color='b', label=r'$V^\pi(s_1)$')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel(r'$\pi(a_1|s_1)$')
plt.ylabel(r'$\pi(a_2|s_1)$')
plt.legend()
plt.savefig('v_pi.pdf', dpi=300)

#----------------------------------------------------------------------
# Q--value iteration
#----------------------------------------------------------------------
plt.figure()
Q = np.zeros((3, 1))
Q_star = np.array([[0.29357516],
                   [0.31508201],
                   [0.27398612]])

Q_list = []
e_list = []

for i in range(1, 100):
    Q_list.append(np.abs(Q_star - Q).max())
    e_list.append(np.exp(-1 * (1 - gamma) * i))
    Q = (1 - gamma) * reward + gamma * np.matmul(P, calc_V_Q(Q))

plt.plot(Q_list, label=r'$||Q^{(k)} - Q^*||_\infty$')
plt.plot(e_list, label=r'$\exp(-(1 - \gamma)k)$')
plt.ylim(top=0.35)
plt.legend()
plt.xlabel('Iteration ' + r'$k$')
plt.ylabel('Magnitude')
plt.savefig('value_iter.pdf', dpi=300)
print(Q)
