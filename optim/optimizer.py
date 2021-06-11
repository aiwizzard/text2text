import torch.optim as optim


class ScheduledOptimizer:
    def __init__(
        self, optimizer: optim.Optimizer, factor=2, model_dim=2048, warmup=4000
    ):
        super(ScheduledOptimizer, self).__init__()
        self.optimizer = optimizer
        self.factor = factor
        self.model_dim = model_dim
        self.warmup = warmup
        self.n_steps = 0
        self.learning_rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_learning_rate()
        self.optimizer.step()

    def _update_learning_rate(self):
        self.n_steps += 1
        learning_rate = self.factor * (
            self.model_dim ** -0.5
            * min(self.n_steps ** -0.5, self.n_steps * self.warmup ** -1.5)
        )
        for p in self.optimizer.param_groups:
            p["lr"] = learning_rate
        self.learning_rate = learning_rate

    def load(self, opt_state_dict, parameters):
        self.load_state_dict(opt_state_dict)
        self.load_parameters(parameters)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, obj):
        self.optimizer.load_state_dict(obj)

    def parameters(self):
        return {
            "warmup": self.warmup,
            "n_steps": self.n_steps,
            "factor": self.factor,
            "learning_rate": self.learning_rate,
        }

    def load_parameters(self, obj):
        self.warmup = obj["warmup"]
        self.n_steps = obj["n_steps"]
        self.factor = obj["factor"]
        self.learning_rate = obj["learning_rate"]
