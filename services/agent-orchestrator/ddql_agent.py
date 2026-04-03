"""Double Deep Q-Learning execution agent.

Learns optimal order timing/sizing from market microstructure signals.
Pure-python implementation with optional numpy acceleration.

State vector (6-dim):
    [mid_price_norm, spread_rel, inventory_norm, momentum_5, vol_20, pnl_norm]

Action space (5):
    0 = hold, 1 = buy_aggressive, 2 = buy_passive,
    3 = sell_aggressive, 4 = sell_passive

Architecture:
    2-layer MLP (6 → 32 → 32 → 5) with ReLU activations.
    Full backpropagation through all layers via mini-batch SGD.
    Double DQN: online network selects actions, target network evaluates.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class DDQLAgent:
    """Double DQN with online + target networks and experience replay."""

    N_STATES = 6
    N_ACTIONS = 5

    def __init__(
        self,
        agent_id: str,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 2000,
        batch_size: int = 32,
        target_update_freq: int = 50,
    ) -> None:
        self.agent_id = agent_id
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.replay_buffer: deque[Transition] = deque(maxlen=buffer_size)

        # 2-layer MLP: N_STATES → 32 → 32 → N_ACTIONS
        self._init_weights()

    # ─── Weight initialization ───────────────────────────────────────────────

    @staticmethod
    def _he_init(rows: int, fan_in: int) -> list[list[float]]:
        """He (Kaiming) initialization for ReLU layers."""
        scale = math.sqrt(2.0 / fan_in)
        return [[random.gauss(0, scale) for _ in range(fan_in)]
                for _ in range(rows)]

    def _init_weights(self) -> None:
        # Online network
        self.W1 = self._he_init(32, self.N_STATES)
        self.b1 = [0.0] * 32
        self.W2 = self._he_init(32, 32)
        self.b2 = [0.0] * 32
        self.W3 = self._he_init(self.N_ACTIONS, 32)
        self.b3 = [0.0] * self.N_ACTIONS
        # Target network (copy of online)
        self._copy_to_target()

    def _copy_to_target(self) -> None:
        self.tW1 = copy.deepcopy(self.W1)
        self.tb1 = self.b1[:]
        self.tW2 = copy.deepcopy(self.W2)
        self.tb2 = self.b2[:]
        self.tW3 = copy.deepcopy(self.W3)
        self.tb3 = self.b3[:]

    # ─── Forward pass helpers ────────────────────────────────────────────────

    @staticmethod
    def _relu(x: list[float]) -> list[float]:
        return [max(0.0, v) for v in x]

    @staticmethod
    def _relu_deriv(x: list[float]) -> list[float]:
        """Derivative of ReLU (step function at pre-activation values)."""
        return [1.0 if v > 0 else 0.0 for v in x]

    @staticmethod
    def _linear(W: list[list[float]], b: list[float],
                x: list[float]) -> list[float]:
        return [sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
                for i in range(len(W))]

    def _forward_with_cache(
        self, W1: list[list[float]], b1: list[float],
        W2: list[list[float]], b2: list[float],
        W3: list[list[float]], b3: list[float],
        state: list[float],
    ) -> tuple[list[float], dict]:
        """Forward pass returning output and intermediate activations."""
        z1 = self._linear(W1, b1, state)
        h1 = self._relu(z1)
        z2 = self._linear(W2, b2, h1)
        h2 = self._relu(z2)
        out = self._linear(W3, b3, h2)
        cache = {"state": state, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        return out, cache

    def _forward(
        self, W1: list[list[float]], b1: list[float],
        W2: list[list[float]], b2: list[float],
        W3: list[list[float]], b3: list[float],
        state: list[float],
    ) -> list[float]:
        h1 = self._relu(self._linear(W1, b1, state))
        h2 = self._relu(self._linear(W2, b2, h1))
        return self._linear(W3, b3, h2)

    def predict(self, state: list[float]) -> list[float]:
        return self._forward(
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, state)

    def predict_target(self, state: list[float]) -> list[float]:
        return self._forward(
            self.tW1, self.tb1, self.tW2, self.tb2, self.tW3, self.tb3, state)

    # ─── Action selection ────────────────────────────────────────────────────

    def select_action(self, state: list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.N_ACTIONS - 1)
        q = self.predict(state)
        return q.index(max(q))

    def store(self, transition: Transition) -> None:
        self.replay_buffer.append(transition)

    # ─── State encoding ─────────────────────────────────────────────────────

    def encode_state(
        self,
        mid_price: float,
        spread: float,
        inventory: int,
        price_history: list[float],
        cash: float,
    ) -> list[float]:
        norm_price = mid_price / 100.0
        norm_spread = spread / mid_price if mid_price > 0 else 0.0
        norm_inv = max(-1.0, min(1.0, inventory / 100.0))

        # 5-tick momentum (guarded against zero/missing prices)
        if len(price_history) >= 5 and price_history[-5] > 0:
            momentum = mid_price / price_history[-5] - 1.0
        else:
            momentum = 0.0

        # 20-tick realized volatility
        if len(price_history) >= 20:
            start = max(1, len(price_history) - 20)
            rets = []
            for i in range(start, len(price_history)):
                if price_history[i - 1] > 0 and price_history[i] > 0:
                    rets.append(
                        math.log(price_history[i] / price_history[i - 1]))
            if len(rets) > 1:
                mean_r = sum(rets) / len(rets)
                vol = (sum((r - mean_r) ** 2 for r in rets)
                       / len(rets)) ** 0.5
            else:
                vol = 0.0
        else:
            vol = 0.0

        norm_cash = cash / 1_000_000.0
        return [norm_price, norm_spread, norm_inv, momentum, vol, norm_cash]

    # ─── Training (full backprop through all layers) ─────────────────────────

    def train_step(self) -> None:
        """Mini-batch SGD with full backpropagation through all layers."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(list(self.replay_buffer), self.batch_size)

        # Accumulate gradients across batch
        dW1 = [[0.0] * self.N_STATES for _ in range(32)]
        db1 = [0.0] * 32
        dW2 = [[0.0] * 32 for _ in range(32)]
        db2 = [0.0] * 32
        dW3 = [[0.0] * 32 for _ in range(self.N_ACTIONS)]
        db3 = [0.0] * self.N_ACTIONS

        for t in batch:
            # Double DQN: online selects, target evaluates
            q_online_next = self.predict(t.next_state)
            best_action = q_online_next.index(max(q_online_next))
            q_target_val = self.predict_target(t.next_state)[best_action]
            target_q = t.reward + (
                0.0 if t.done else self.gamma * q_target_val)

            # Forward pass with cache
            q_current, cache = self._forward_with_cache(
                self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, t.state)

            error = target_q - q_current[t.action]

            # ── Backprop layer 3 (output) ────────────────────────────────
            # dL/dW3[a][j] = -error * h2[j]  (only for the taken action)
            # dL/db3[a]    = -error
            d_out = [0.0] * self.N_ACTIONS
            d_out[t.action] = error  # positive = increase Q for this action

            for j in range(32):
                dW3[t.action][j] += error * cache["h2"][j]
            db3[t.action] += error

            # ── Backprop layer 2 (hidden) ────────────────────────────────
            # d_h2[j] = sum_a(d_out[a] * W3[a][j])  (only action a matters)
            relu2 = self._relu_deriv(cache["z2"])
            d_h2 = [d_out[t.action] * self.W3[t.action][j] * relu2[j]
                     for j in range(32)]

            for i in range(32):
                for j in range(32):
                    dW2[i][j] += d_h2[i] * cache["h1"][j]
                db2[i] += d_h2[i]

            # ── Backprop layer 1 (hidden) ────────────────────────────────
            relu1 = self._relu_deriv(cache["z1"])
            d_h1 = [sum(d_h2[k] * self.W2[k][j] for k in range(32))
                     * relu1[j]
                     for j in range(32)]

            for i in range(32):
                for j in range(self.N_STATES):
                    dW1[i][j] += d_h1[i] * cache["state"][j]
                db1[i] += d_h1[i]

        # ── Apply gradients (SGD, averaged over batch) ───────────────────
        scale = self.lr / self.batch_size
        for i in range(32):
            for j in range(self.N_STATES):
                self.W1[i][j] += scale * dW1[i][j]
            self.b1[i] += scale * db1[i]

        for i in range(32):
            for j in range(32):
                self.W2[i][j] += scale * dW2[i][j]
            self.b2[i] += scale * db2[i]

        for i in range(self.N_ACTIONS):
            for j in range(32):
                self.W3[i][j] += scale * dW3[i][j]
            self.b3[i] += scale * db3[i]

        # ── Decay exploration, update target network periodically ────────
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._copy_to_target()

    # ─── Reward computation ──────────────────────────────────────────────────

    @staticmethod
    def compute_reward(
        pnl_delta: float,
        spread_captured: float,
        inventory: int,
        inventory_penalty_per_unit: float = 0.001,
    ) -> float:
        """Reward = realized PnL + spread captured - inventory holding cost.

        The penalty scales with absolute inventory to discourage large
        directional positions.
        """
        inventory_cost = abs(inventory) * inventory_penalty_per_unit
        return pnl_delta + spread_captured - inventory_cost

    # ─── Action → order translation ──────────────────────────────────────────

    def act(
        self, state: list[float], mid_price: float, spread: float
    ) -> dict | None:
        """Select action via epsilon-greedy policy and return an order dict."""
        action = self.select_action(state)
        half_spread = spread / 2.0

        if action == 0:
            return None
        elif action == 1:  # buy aggressive (cross spread)
            return {"side": "buy",
                    "price": mid_price + half_spread, "quantity": 5}
        elif action == 2:  # buy passive (inside spread)
            return {"side": "buy",
                    "price": mid_price - half_spread * 0.5, "quantity": 5}
        elif action == 3:  # sell aggressive
            return {"side": "sell",
                    "price": mid_price - half_spread, "quantity": 5}
        elif action == 4:  # sell passive
            return {"side": "sell",
                    "price": mid_price + half_spread * 0.5, "quantity": 5}
        return None
