import torch
from torch.autograd import Variable

def generate_linear_points(self, num_points, ranges):
    grids = [torch.linspace(r[0], r[1], num_points).to(self.device) for r in ranges]
    return torch.stack(torch.meshgrid(*grids, indexing='ij')).reshape(len(ranges), -1)


def generate_random_points(self, num_points, ranges):
    # grids = [torch.Tensor(num_points).to(self.device).uniform_(r[0], r[1]) for r in ranges]
    # return torch.stack(torch.meshgrid(*grids, indexing='ij')).reshape(len(ranges), -1)
    grids = [torch.Tensor(num_points ** (len(ranges))).to(self.device).uniform_(r[0], r[1]) for r in ranges]
    return torch.stack(grids).reshape(len(ranges), -1)


def l(self, N): return int(self.linear_mult * N)


def make_distributed_points(self):
    x_range = [0, 1]
    y_range = [0, 1]
    t_range = [0, self.T * self.t_max]

    # --- Initial Condition ---
    with torch.no_grad():
        x_linear, y_linear = self.generate_linear_points(self.l(self.N_IC), [x_range, y_range])
        x_random, y_random = self.generate_random_points(self.N_IC, [x_range, y_range])

    self.x_IC = Variable(torch.cat((x_linear, x_random)), requires_grad=True).to(self.device)
    self.y_IC = Variable(torch.cat((y_linear, y_random)), requires_grad=True).to(self.device)
    self.t_IC = Variable(torch.zeros_like(self.x_IC), requires_grad=True).to(self.device)
    if self.NN_params['input_size'] != 3:
        self.beta_IC = Variable(self.beta * torch.ones_like(self.x_IC), requires_grad=True).to(self.device)
    self.Initial_conditions(self.x_IC, self.y_IC)

    # --- PDE Points ---
    x_linear, y_linear, t_linear = self.generate_linear_points(self.l(self.N_PDE), [x_range, y_range, t_range])
    x_random, y_random, t_random = self.generate_random_points(self.N_PDE, [x_range, y_range, t_range])

    try:
        self.x_PDE
    except AttributeError:
        self.x_PDE = torch.Tensor([]).to(self.device)
        self.y_PDE = torch.Tensor([]).to(self.device)
        self.t_PDE = torch.Tensor([]).to(self.device)

    x = Variable(torch.cat((x_random, self.x_PDE[self.N_PDE ** 3:])), requires_grad=True).to(self.device)
    y = Variable(torch.cat((y_random, self.y_PDE[self.N_PDE ** 3:])), requires_grad=True).to(self.device)
    t = Variable(torch.cat((t_random, self.t_PDE[self.N_PDE ** 3:])), requires_grad=True).to(self.device)
    if self.NN_params['input_size'] == 3:
        conv, div, corr = self.compute_PDE(x, y, t)
    else:
        beta = Variable(self.beta * torch.ones_like(x), requires_grad=True).to(self.device)
        conv, div, corr = self.compute_PDE(x, y, t, beta)

    pde_dist = self.weights[0] * torch.where(conv.abs() > 0.01, 1, 0) + self.weights[1] * torch.where(div.abs() > 0.01,
                                                                                                      1, 0) + \
               self.weights[2] * torch.where(corr.abs() > 0.01, 1, 0)
    # pde_dist = self.weights[0] * conv.abs() + self.weights[1] * div.abs() + self.weights[2] * corr.abs()
    sampled_indices_pde = torch.multinomial(pde_dist / pde_dist.sum(), self.N_PDE ** 3, replacement=True)

    self.x_PDE = Variable(torch.cat((x_linear, x[sampled_indices_pde])), requires_grad=True)
    self.y_PDE = Variable(torch.cat((y_linear, y[sampled_indices_pde])), requires_grad=True)
    self.t_PDE = Variable(torch.cat((t_linear, t[sampled_indices_pde])), requires_grad=True)
    if self.NN_params['input_size'] != 3:
        self.beta_PDE = Variable(self.beta * torch.ones_like(self.x_PDE), requires_grad=True).to(self.device)

    #  --- Boundary Conditions ---
    with torch.no_grad():
        x = torch.linspace(0, 1, self.N_BC).to(self.device)
        y = torch.linspace(0, 1, self.N_BC).to(self.device)
        t = torch.linspace(0, self.T * self.t_max, self.N_BC).to(self.device)
        c_condition_linear = cnd.form_boundaries([x, y, t], self.ones, self.zeros)

        x = torch.Tensor(self.N_BC).to(self.device).uniform_(0, 1)
        y = torch.Tensor(self.N_BC).to(self.device).uniform_(0, 1)
        t = torch.Tensor(self.N_BC).to(self.device).uniform_(0, self.T * self.t_max)
        c_condition_random = cnd.form_boundaries([x, y, t], self.ones, self.zeros)

        self.x_BC = Variable(torch.cat((c_condition_linear[:, 0], c_condition_random[:, 0])), requires_grad=True)
        self.y_BC = Variable(torch.cat((c_condition_linear[:, 1], c_condition_random[:, 1])), requires_grad=True)
        self.t_BC = Variable(torch.cat((c_condition_linear[:, 2], c_condition_random[:, 2])), requires_grad=True)

        if self.NN_params['input_size'] == 3:
            self.c, self.p, self.u = self.Boundary_conditions(self.x_BC, self.y_BC, self.t_BC, self.beta)
        else:
            self.beta_BC = Variable(self.beta * torch.ones_like(self.x_BC), requires_grad=True)
            self.c, self.p, self.u = self.Boundary_conditions(self.x_BC, self.y_BC, self.t_BC, self.beta_BC)

        self.where_c_tb = (self.y_BC == 1) | (self.y_BC == 0)
        self.where_c_in = (self.x_BC == 0) & ((self.y_BC - 1 / 2).abs().round(decimals=5) <= self.zeta / 2)
        self.where_c_out = (self.x_BC == 0) & ((self.y_BC - 1 / 2).abs().round(decimals=5) > self.zeta / 2)

        self.x_lr = Variable(self.x_BC[self.where_c_out], requires_grad=True).to(self.device)
        self.y_lr = Variable(self.y_BC[self.where_c_out], requires_grad=True).to(self.device)
        self.t_lr = Variable(self.t_BC[self.where_c_out], requires_grad=True).to(self.device)

        self.x_tb = Variable(self.x_BC[self.where_c_tb], requires_grad=True).to(self.device)
        self.y_tb = Variable(self.y_BC[self.where_c_tb], requires_grad=True).to(self.device)
        self.t_tb = Variable(self.t_BC[self.where_c_tb], requires_grad=True).to(self.device)

        if self.NN_params['input_size'] != 3:
            self.beta_lr = Variable(self.beta * torch.ones_like(self.x_lr), requires_grad=True)
            self.beta_tb = Variable(self.beta * torch.ones_like(self.x_tb), requires_grad=True)
