from __future__ import annotations

import random

import numpy as np
from scipy.stats import poisson

OCCUPANCY_BIN_SIZE = 0.01
ZONE_ACTIONS = (
    -0.30,
    -0.25,
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
)


def generate_separate_event_demand(node_list, num_days, demand_params, time_slots):
    all_days_demand_vectors = []
    transformed_demand_vectors = []

    for _day in range(num_days):
        daily_arrivals = []
        daily_departures = []
        for category_count, category_params in zip(
            node_list, demand_params, strict=True
        ):
            category_arrivals = np.zeros((category_count, 24), dtype=np.int64)
            category_departures = np.zeros((category_count, 24), dtype=np.int64)
            for params, (start, end) in zip(category_params, time_slots, strict=True):
                lambda_arrivals, lambda_departures = params
                category_arrivals[:, start:end] = poisson.rvs(
                    lambda_arrivals, size=(category_count, end - start)
                )
                category_departures[:, start:end] = poisson.rvs(
                    lambda_departures, size=(category_count, end - start)
                )
            daily_arrivals.append(category_arrivals)
            daily_departures.append(category_departures)

        daily_arrivals = np.vstack(daily_arrivals)
        daily_departures = np.vstack(daily_departures)
        all_days_demand_vectors.append(
            {
                "arrivals": daily_arrivals,
                "departures": daily_departures,
            }
        )

        transformed_day = []
        for zone in range(daily_arrivals.shape[0]):
            transformed_zone = []
            for hour in range(24):
                events = [1] * int(daily_arrivals[zone, hour])
                events.extend([-1] * int(daily_departures[zone, hour]))
                if events:
                    np.random.shuffle(events)
                else:
                    events = [0]
                transformed_zone.append(events)
            transformed_day.append(transformed_zone)
        transformed_demand_vectors.append(transformed_day)

    return all_days_demand_vectors, transformed_demand_vectors


def available_zone_actions(state) -> tuple[float, ...]:
    occupancy = round(float(state[0]), 2)
    return tuple(
        action
        for action in ZONE_ACTIONS
        if 0.0 <= round(occupancy + action, 2) <= 1.0
    )


def occupancy_bin(bikes: int, capacity: int) -> float:
    if capacity <= 0:
        return 0.0
    occupancy = min(1.0, max(0.0, bikes / capacity))
    return round(round(occupancy / OCCUPANCY_BIN_SIZE) * OCCUPANCY_BIN_SIZE, 2)


class ZoneRebalancingAgent:
    def __init__(self, category, epsilon=1, epsilon_decay=8.25e-7, min_epsilon=0.01):
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.q_table = {}
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.category = category

    def set_epsilon(self, value):
        self.epsilon = value

    def get_q_value(self, state, action):
        key = ((round(float(state[0]), 2), int(state[1])), float(action))
        return self.q_table.get(key, 0)

    def update_q_table(self, state, action, reward, next_state):
        actions = available_zone_actions(next_state)
        max_q_next = max(self.get_q_value(next_state, a) for a in actions)
        q_current = self.get_q_value(state, action)
        q_new = q_current + self.learning_rate * (
            reward + self.discount_factor * max_q_next - q_current
        )
        self.q_table[((round(float(state[0]), 2), int(state[1])), float(action))] = (
            q_new
        )

    def decide_action(self, state):
        actions = available_zone_actions(state)
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.get_q_value(state, action) for action in actions]
        return actions[int(np.argmax(q_values))]

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)


class ZoneCMDPEnv:
    def __init__(
        self,
        graph,
        demand_vectors,
        lambdas,
        gamma,
        station_params,
        failure_cost_coef=0.0,
    ):
        self.G = graph
        self.demand_vectors = demand_vectors
        self.num_days = len(demand_vectors)
        self.num_zones = len(list(self.G.nodes))
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
        return 0 if self.next_rebalancing_hour == 23 else 1

    def _state_for_zone(self, zone, time):
        node = self.G.nodes[zone]
        return [occupancy_bin(node["bikes"], node["capacity"]), time]

    def get_state(self):
        state = np.zeros((self.num_zones, 2), dtype=float)
        failures = [0] * self.num_zones

        time = 0 if self.next_rebalancing_hour == 11 else 1
        while self.hour <= self.next_rebalancing_hour:
            for zone in range(self.num_zones):
                node = self.G.nodes[zone]
                n_bikes = node["bikes"]
                capacity = node["capacity"]

                demand_list = self.demand_vectors[self.day][zone][self.hour]
                for demand_change in demand_list:
                    n_bikes += demand_change
                    if n_bikes < 0:
                        n_bikes = 0
                        failures[zone] += 1

                node["bikes"] = min(capacity, n_bikes)
                state[zone] = self._state_for_zone(zone, time)

            self.hour += 1

        self.next_rebalancing_hour = 23 if self.hour == 12 else 11
        if self.next_rebalancing_hour == 11:
            self.day += 1
            self.hour = 0
            if self.day == self.num_days:
                self.day = 0

        return state, failures

    def compute_reward(self, actions, failures, post_action_occupancy, bike_deltas):
        base_rewards = np.zeros(self.num_zones)
        reb_costs = np.zeros(self.num_zones)
        current_period = self.current_period

        for zone in range(self.num_zones):
            cat = self.G.nodes[zone]["station"]
            p = self.station_params[cat]

            rebalancing_penalty = 1 if bike_deltas[zone] != 0 else 0
            reb_cost = self.gamma * p["phi"] * rebalancing_penalty

            base_rewards[zone] -= self.failure_cost_coef * failures[zone]
            base_rewards[zone] -= reb_cost
            reb_costs[zone] = reb_cost

            if self.next_rebalancing_hour == 23:
                target = p["evening_target"] / 100
                threshold = p["evening_threshold"] / 100
            else:
                target = p["morning_target"] / 100
                threshold = p["morning_threshold"] / 100

            deviation = abs(post_action_occupancy[zone] - target)
            if deviation > threshold:
                base_rewards[zone] -= self.csi * (deviation - threshold) * 100

        rewards = base_rewards.copy()
        for zone in range(self.num_zones):
            cat = self.G.nodes[zone]["station"]
            if cat in self.lambdas:
                rewards[zone] -= self.lambdas[cat][current_period] * failures[zone]

        return rewards, base_rewards, reb_costs

    def reset(self):
        self.hour = 0
        self.day = 0
        self.next_rebalancing_hour = 11

        state = np.zeros((self.num_zones, 2), dtype=float)
        for zone in range(self.num_zones):
            state[zone] = self._state_for_zone(zone, 0)
        return state

    def step(self, actions):
        post_action_occupancy = np.zeros(self.num_zones, dtype=float)
        bike_deltas = np.zeros(self.num_zones, dtype=np.int64)

        for zone in range(self.num_zones):
            node = self.G.nodes[zone]
            capacity = node["capacity"]
            before = node["bikes"]
            delta = int(round(float(actions[zone]) * capacity))
            after = min(capacity, max(0, before + delta))
            node["bikes"] = after
            bike_deltas[zone] = after - before
            post_action_occupancy[zone] = after / capacity if capacity > 0 else 0.0

        state, failures = self.get_state()
        reward, base_reward, reb_costs = self.compute_reward(
            actions, failures, post_action_occupancy, bike_deltas
        )

        return state, reward, base_reward, failures, reb_costs
