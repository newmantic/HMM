# HMM


A Hidden Markov Model is a statistical model that represents systems where the underlying states are hidden (i.e., not directly observable) but can be inferred through observable sequences of events.


An HMM is defined by the following components:
N: The number of hidden states. Let the set of hidden states be denoted by S = {S_1, S_2, ..., S_N}.
M: The number of distinct observation symbols per state. Let the set of observation symbols be denoted by V = {v_1, v_2, ..., v_M}.
A: The state transition probability matrix, where each element A[i][j] represents the probability of transitioning from state S_i to state S_j. Formally, A = {a_ij}, where a_ij = P(q_{t+1} = S_j | q_t = S_i), with 1 <= i, j <= N and 1 <= t <= T.
B: The observation probability matrix, where each element B[j][k] represents the probability of observing symbol v_k given the system is in state S_j. Formally, B = {b_jk}, where b_jk = P(O_t = v_k | q_t = S_j), with 1 <= j <= N and 1 <= k <= M.
pi: The initial state distribution vector, where pi[i] represents the probability that the system starts in state S_i. Formally, pi = {pi_i}, where pi_i = P(q_1 = S_i), with 1 <= i <= N.


An HMM is fully specified by the following triplet:
lambda = (A, B, pi)
where:
A = {a_ij} is the state transition matrix.
B = {b_jk} is the observation probability matrix.
pi = {pi_i} is the initial state distribution.

Given an observation sequence O = {O_1, O_2, ..., O_T}, where each O_t is one of the symbols from V, the goal of an HMM is to model the joint probability distribution P(O, Q | lambda), where Q = {q_1, q_2, ..., q_T} is the sequence of hidden states corresponding to the observations.

The three central problems of HMMs are: 1) Evaluation 2) Decoding 3) Learning

Evaluation Problem: Given the model lambda = (A, B, pi) and an observation sequence O, compute the probability P(O | lambda). This is solved using the Forward Algorithm.

Forward probability alpha_t(i) is defined as:
alpha_t(i) = P(O_1, O_2, ..., O_t, q_t = S_i | lambda)

This can be computed recursively as:
alpha_1(i) = pi_i * b_i(O_1)
alpha_{t+1}(j) = [sum_{i=1}^N alpha_t(i) * a_ij] * b_j(O_{t+1})


Decoding Problem: Given the model lambda = (A, B, pi) and an observation sequence O, find the most likely sequence of hidden states Q*. This is solved using the Viterbi Algorithm.

Viterbi probability delta_t(i) is defined as:
delta_t(i) = max_{q_1,q_2,...,q_{t-1}} P(q_1, q_2, ..., q_t = S_i, O_1, O_2, ..., O_t | lambda)

This can be computed recursively as:
delta_1(i) = pi_i * b_i(O_1)
delta_{t+1}(j) = max_{i=1}^N [delta_t(i) * a_ij] * b_j(O_{t+1})


Learning Problem: Given the model lambda = (A, B, pi) and an observation sequence O, adjust the model parameters (A, B, pi) to maximize P(O | lambda). This is solved using the Baum-Welch Algorithm or the Expectation-Maximization (EM) Algorithm.

Baum-Welch Algorithm 
The Baum-Welch algorithm iteratively estimates the parameters A, B, and pi by maximizing the likelihood of the observation sequence.
Define:
gamma_t(i) = P(q_t = S_i | O, lambda) = alpha_t(i) * beta_t(i) / P(O | lambda)
xi_t(i, j) = P(q_t = S_i, q_{t+1} = S_j | O, lambda)

The parameters are updated as follows:
pi'_i = gamma_1(i)
a'ij = [sum{t=1}^{T-1} xi_t(i, j)] / [sum_{t=1}^{T-1} gamma_t(i)]
b'jk = [sum{t=1}^T s.t. O_t = v_k gamma_t(j)] / [sum_{t=1}^T gamma_t(j)]
