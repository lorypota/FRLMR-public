import numpy as np


class CMDPEnv:
    def __init__(
        self,
        graph,
        demand_vectors,
        lambdas,
        gamma,
        station_params,
        failure_cost_coef=0.0,
    ):
        """
        Args:
            graph: NetworkX graph representing the MSS network.
            demand_vectors: Pre-generated demand vectors.
            lambdas: Dict mapping category -> [lambda_morning, lambda_evening].
                Only constrained categories appear in this dict.
                This is a mutable reference; the training loop updates it.
            gamma: Rebalancing cost parameter.
            station_params: Dict mapping station category -> dict with keys:
                'chi':               (unused in CMDP)
                'phi':               rebalancing cost multiplier for gamma
                'evening_target':    target occupancy when next_rebalancing_hour == 23
                'evening_threshold': tolerance around evening target
                'morning_target':    target occupancy when next_rebalancing_hour != 23
                'morning_threshold': tolerance around morning target
        """
        self.G = graph
        self.demand_vectors = demand_vectors
        self.num_days = len(demand_vectors)
        self.num_stations = len(list(self.G.nodes))
        self.hour = 0
        self.day = 0
        self.next_rebalancing_hour = 11
        self.lambdas = lambdas
        self.gamma = gamma
        self.csi = 0.3
        self.station_params = station_params
        self.failure_cost_coef = failure_cost_coef

    @property
    def current_period(self):
        """Period that was just simulated (0=morning, 1=evening).

        Must be called after get_state() / step(), which flips
        next_rebalancing_hour.  If next_rebalancing_hour == 23 the morning
        period was just processed; if 11 the evening period was.
        """
        return 0 if self.next_rebalancing_hour == 23 else 1

    def get_state(self):
        state = np.zeros((self.num_stations, 2), dtype=np.int64)
        failures = [0] * self.num_stations

        time = 0 if self.next_rebalancing_hour == 11 else 1
        while self.hour <= self.next_rebalancing_hour:
            for i in range(self.num_stations):
                n_bikes = self.G.nodes[i]["bikes"]

                demand_list = self.demand_vectors[self.day][i][self.hour]
                for demand_change in demand_list:
                    n_bikes += demand_change
                    if n_bikes < 0:
                        n_bikes = 0
                        failures[i] += 1
                    elif n_bikes > 100:
                        n_bikes = 100

                self.G.nodes[i]["bikes"] = n_bikes

                state[i] = [self.G.nodes[i]["bikes"], time]

            self.hour += 1

        self.next_rebalancing_hour = 23 if self.hour == 12 else 11
        if self.next_rebalancing_hour == 11:
            self.day += 1
            self.hour = 0
            if self.day == self.num_days:
                self.day = 0

        return state, failures

    def compute_reward(self, action, failures, mu):
        base_rewards = np.zeros(self.num_stations)
        reb_costs = np.zeros(self.num_stations)
        current_period = self.current_period

        for i in range(self.num_stations):
            cat = self.G.nodes[i]["station"]
            p = self.station_params[cat]

            rebalancing_penalty = 1 if action[i] != 0 else 0
            reb_cost = self.gamma * p["phi"] * rebalancing_penalty

            base_rewards[i] -= self.failure_cost_coef * failures[i]
            base_rewards[i] -= reb_cost
            reb_costs[i] = reb_cost

            if self.next_rebalancing_hour == 23:
                target = p["evening_target"]
                threshold = p["evening_threshold"]
            else:
                target = p["morning_target"]
                threshold = p["morning_threshold"]

            deviation = abs(mu[i] - target)
            if deviation > threshold:
                base_rewards[i] -= self.csi * (deviation - threshold)

        # Lagrangian penalty on top of base reward
        rewards = base_rewards.copy()
        for i in range(self.num_stations):
            cat = self.G.nodes[i]["station"]
            if cat in self.lambdas:
                rewards[i] -= self.lambdas[cat][current_period] * failures[i]

        return rewards, base_rewards, reb_costs

    def reset(self):
        self.hour = 0
        self.day = 0
        self.next_rebalancing_hour = 11

        state = np.zeros((self.num_stations, 2), dtype=np.int64)
        for i in range(self.num_stations):
            state[i] = [self.G.nodes[i]["bikes"], 0]

        return state

    def step(self, action):  # action is a (num_stations, 1) vector
        mu = np.zeros(self.num_stations, dtype=np.int64)
        for i in range(self.num_stations):
            self.G.nodes[i]["bikes"] += action[i]
            mu[i] = self.G.nodes[i]["bikes"]

        state, failures = self.get_state()
        reward, base_reward, reb_costs = self.compute_reward(action, failures, mu)

        return state, reward, base_reward, failures, reb_costs
