import torch as T
import numpy as np
import pdb

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

class BlackScholesEnv():
    def __init__(self, params):
        self.params = params
        self.spaces = {
            't_space' : np.arange(params["Ndt"]),
            'S_space' : np.linspace(params['S0'] * np.exp(-6 * params["sigma"] * np.sqrt(params['T'])), 
                                    params['S0'] * np.exp(6 * params["sigma"] * np.sqrt(params['T'])), 31),
            'alpha_space' : np.linspace(-params["max_alpha"], params["max_alpha"], 31),
            'B_space' : np.linspace(params["B0"]-params["S0"]*params["max_alpha"],
                                    params["B0"]+params["S0"]*params["max_alpha"], 31)
            }

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
 
    def __repr__(self):
        return f"BlackScholesEnv({self.params})"

    def reset(self, Nsims=1):
        S0 = self.params["S0"]*T.ones(Nsims).to(self.device)
        B0 = self.params["B0"]*T.ones(Nsims).to(self.device)
        alpha = T.zeros(Nsims).to(self.device)
        
        return S0, alpha, B0

    def random_reset( self, time, Nsims=1):
        if time == self.spaces["t_space"][0]:
            S0, alpha_m1, B0 = self.reset(Nsims)
        else:
            S0 = self.params["S0"] * \
                    T.exp((self.params['mu']- .5 *self.params["sigma"]**2)* self.params['T'] + \
                    self.params["sigma"] * np.sqrt(self.params['T'])*T.randn(size = (Nsims,), device=self.device))
            alpha_m1 = -self.params["max_alpha"] + 2*self.params["max_alpha"] * \
                        T.rand(size=(Nsims,), device=self.device)
            B0 = -(self.params["S0"]*alpha_m1) + 0.75*T.randn(size=(Nsims,), device=self.device)

        return S0, alpha_m1, B0

    def step(self, S_t, alpha_tm1, B_t, alpha_t):
        Nsims = S_t.shape[0]

        dt = self.params["T"]/self.params["Ndt"]
        sqrt_dt = np.sqrt(dt)

        # Price modification according to Geometric Brownian Motion
        S_tp1 = S_t * T.exp((self.params["r"] - 0.5 * self.params["sigma"]**2) * dt +\
             self.params["sigma"] * sqrt_dt * T.randn(Nsims, device=self.device))


        # compute the bank account cash-flow
        B_tplus = B_t \
                - (alpha_t-alpha_tm1)*S_t \
                - T.abs(alpha_t-alpha_tm1)*self.params["epsilon"]

        # interest rate on the bank account
        B_tp1 = B_tplus * np.exp(self.params["r"]*dt)

        # Reward calculation (wealth at time t+1 - wealth at time t) 
        r = B_tp1 + alpha_t * S_tp1 - (B_t + alpha_tm1* S_t)

        return S_tp1, alpha_t, B_tp1, -r

    # payoff of the option
    def option_price(self, S):
        return T.maximum(self.params["K"]- S, T.zeros(1, device=self.device))

    # settlement of the option
    def settlement (self, S_T, alpha_Tm1, B_T):
        return T.abs(alpha_Tm1)*self.params["epsilon"] + self.option_price(S_T)

   