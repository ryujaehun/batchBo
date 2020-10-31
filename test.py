from bayes_opt import BayesianOptimization


def black_box_function(x, y):
    return -(-x ** 2 - (y - 1) ** 2 + 1)

optimizer = BayesianOptimization(
    f=None,
    pbounds={'x': (-2, 4), 'y': (-3, 3)},
    verbose=2,
    random_state=1,
    n_restarts_optimizer=5,
    batch_size=4
)
from bayes_opt import UtilityFunction

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

for _ in range(10):
    next_point = optimizer.suggest(utility)

    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)
    print(target)

print(target, next_point)
