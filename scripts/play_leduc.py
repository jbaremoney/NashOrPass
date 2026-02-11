import time
from pettingzoo.classic import leduc_holdem_v4

def main():
    env = leduc_holdem_v4.env(render_mode="human")
    env.reset(seed=42)

    step_i = 0

    # Import pygame AFTER env is created (so deps are definitely installed/loaded)
    import pygame

    pygame.init()

    for agent in env.agent_iter():
        # --- handle window close immediately ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("QUIT received — closing env.")
                env.close()
                return

        obs, reward, termination, truncation, info = env.last()

        # Render every step so the screen updates
        env.render()

        if termination or truncation:
            action = None
        else:
            mask = obs["action_mask"]
            action = env.action_space(agent).sample(mask=mask)  # mask is supported by Gymnasium spaces :contentReference[oaicite:2]{index=2}

        print(f"[{step_i:04d}] agent={agent} r={reward} term={termination} trunc={truncation} action={action}", flush=True)
        step_i += 1

        env.step(action)
        time.sleep(0.25)  # slow enough to see motion

    print("Game finished. Closing.")
    env.close()

if __name__ == "__main__":
    main()
