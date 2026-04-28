from NashOrPass.agents.models.leduc.MDP import LeducSimpleMDP
from NashOrPass.environment.leduc.simple.models.State import MDPState


def initial_states():
    cards = ["Jh", "Jd", "Qh", "Qd", "Kh", "Kd"]
    states = []

    for hero_card in cards:
        for villain_card in cards:
            if villain_card == hero_card:
                continue

            for hero_position in ["btn", "bb"]:
                to_act = 0 if hero_position == "btn" else 1
                states.append(
                    MDPState(
                        hero_card=hero_card,
                        villain_card=villain_card,
                        round_stage="preflop",
                        flop_card="none",
                        action_facing="none",
                        position=hero_position,
                        bb_amnt=1,
                        to_act=to_act,
                        pot=2,
                        checked_alr=False,
                    )
                )

    return states

def initial_hero_states(villain_policy="uniform"):
    roots = initial_states()
    hero_states = []

    for s in roots:
        if s.to_act == 0:
            hero_states.append(s)
        else:
            outs = LeducSimpleMDP.env_transition_dist(s, villain_policy)
            for _, s_next, _, done in outs:
                if not done:
                    hero_states.append(s_next)

    return hero_states

def generate_reachable_states(villain_policy="uniform"):
    seen = set()
    stack = initial_hero_states(villain_policy)

    while stack:
        s = stack.pop()

        if s in seen:
            continue
        seen.add(s)

        if s.to_act != 0:
            continue

        for a in LeducSimpleMDP.legal_actions_from_mdp(s):
            outs = LeducSimpleMDP.action_outcomes(s, a, villain_policy)

            for _, s_next, _, done in outs:
                if not done and s_next not in seen:
                    stack.append(s_next)

    return seen


def value_iteration(villain_policy="uniform", gamma=1.0, theta=1e-8, max_iters=1000):
    states = generate_reachable_states(villain_policy)
    V = {s: 0.0 for s in states}
    policy = {}

    for it in range(max_iters):
        delta = 0.0

        for s in states:
            legal = LeducSimpleMDP.legal_actions_from_mdp(s)

            q_values = {}

            for a in legal:
                q = 0.0
                outs = LeducSimpleMDP.action_outcomes(s, a, villain_policy)

                for p, s_next, r, done in outs:

                    q += p * r if done else p * (r + gamma * V[s_next])

                q_values[a] = q

            best_a = max(q_values, key=q_values.get)
            best_v = q_values[best_a]

            delta = max(delta, abs(V[s] - best_v))
            V[s] = best_v
            policy[s] = best_a

        print(f"iter={it}, delta={delta}")

        if delta < theta:
            break

    return V, policy

def q_values_for_state(s, V, villain_policy="uniform", gamma=1.0):
    qs = {}
    for a in LeducSimpleMDP.legal_actions_from_mdp(s):
        q = 0.0
        outs = LeducSimpleMDP.action_outcomes(s, a, villain_policy)

        for p, s_next, r, done in outs:
            q += p * r if done else p * (r + gamma * V[s_next])

        qs[a] = q
    return qs


if __name__ == "__main__":
    V, policy = value_iteration()

    roots = initial_states()

    for root in roots:
        print("\n" + "=" * 70)
        print("HAND ROOT")
        print("state:", root.to_tuple())
        print(f"hero_card={root.hero_card}, villain_card={root.villain_card}, hero_position={root.hero_position}")
        print("=" * 70)

        if root.to_act == 0:
            print("Hero acts first preflop.")
            print("Legal actions:", LeducSimpleMDP.legal_actions_from_mdp(root))
            print("Best action:", policy[root])
            print("Q-values:", q_values_for_state(root, V))

        else:
            print("Hero is BB, so villain/button acts first preflop.")
            print("Villain policy: uniform over legal actions")
            print("Outcomes after villain's first action:")

            outs = LeducSimpleMDP.env_transition_dist(root, villain_policy="uniform")

            total_prob = 0.0

            for i, (p, s_next, r, done) in enumerate(outs, start=1):
                total_prob += p

                print("\n" + "-" * 50)
                print(f"Branch {i}")
                print(f"Probability: {p:.4f}")
                print("Resulting state:", s_next.to_tuple())
                print(f"Immediate reward: {r}")
                print(f"Terminal hand? {done}")

                if done:
                    print("Outcome: hand ended before hero acted.")
                else:
                    print("Outcome: next hero decision state.")
                    print("Legal actions:", LeducSimpleMDP.legal_actions_from_mdp(s_next))
                    print("Best action:", policy[s_next])
                    print("Q-values:", q_values_for_state(s_next, V))

            print("\nTotal branch probability:", total_prob)