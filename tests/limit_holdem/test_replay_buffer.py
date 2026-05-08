import numpy as np
from NashOrPass.limit_holdem.buffers.replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_sample():
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.push(
            obs=np.zeros(72),
            action=0,
            reward=1.0,
            next_obs=np.ones(72),
            done=False,
            legal_actions=[0, 1],
            next_legal_actions=[2, 3],
        )

    batch = buffer.sample(3)

    assert batch["obs"].shape == (3, 72)
    assert batch["actions"].shape == (3,)
    assert len(buffer) == 5