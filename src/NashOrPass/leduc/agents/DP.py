from pathlib import Path
import argparse
import pickle

from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP
from NashOrPass.leduc.env.models.State import MDPState


SUPPORTED_VILLAIN_POLICIES = [
    "uniform",
    "check_call",
    "tight",
    "aggressive",
    "always_raise",
    "always_fold",
    "rank_aware_tight",
    "rank_aware_aggressive",
]

def save_policy(policy, filename="dp_policy.pkl"):
    """
    Save a deterministic DP policy to:

        NashOrPass/leduc/agents/saved_policies/<filename>

    The saved policy maps:

        state.to_tuple() -> action

    instead of storing raw MDPState objects.
    """
    agents_dir = Path(__file__).resolve().parent
    save_dir = agents_dir / "saved_policies"
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename

    serializable_policy = {
        state.to_tuple(): action
        for state, action in policy.items()
    }

    with open(path, "wb") as f:
        pickle.dump(serializable_policy, f)

    print(f"Saved policy to {path}")


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
    """
    Return the initial states where hero is to act.

    If hero is button, hero acts immediately.

    If hero is big blind, villain/button acts first, so we advance the MDP
    through villain's first action distribution and collect the resulting
    hero decision states.
    """
    roots = initial_states()
    hero_states = []

    for s in roots:
        if s.to_act == 0:
            hero_states.append(s)
        else:
            outs = LeducSimpleMDP.env_transition_dist(
                s,
                villain_policy=villain_policy,
            )

            for _, s_next, _, done in outs:
                if not done:
                    hero_states.append(s_next)

    return hero_states


def generate_reachable_states(villain_policy="uniform"):
    """
    Generate all reachable hero decision states under a fixed villain policy.

    This creates the single-agent MDP induced by fixing the opponent policy.
    The only decision states we store are states where hero is to act.
    """
    seen = set()
    stack = initial_hero_states(villain_policy=villain_policy)

    while stack:
        s = stack.pop()

        if s in seen:
            continue

        seen.add(s)

        if s.to_act != 0:
            continue

        legal = LeducSimpleMDP.legal_actions_from_mdp(s)

        for action in legal:
            outs = LeducSimpleMDP.action_outcomes(
                s,
                action,
                villain_policy=villain_policy,
            )

            for _, s_next, _, done in outs:
                if not done and s_next not in seen:
                    stack.append(s_next)

    return seen


def q_values_for_state(s, V, villain_policy="uniform", gamma=1.0):
    """
    Compute Q(s, a) for every legal hero action using the current V table.
    """
    q_values = {}

    for action in LeducSimpleMDP.legal_actions_from_mdp(s):
        q = 0.0

        outs = LeducSimpleMDP.action_outcomes(
            s,
            action,
            villain_policy=villain_policy,
        )

        for p, s_next, r, done in outs:
            if done:
                q += p * r
            else:
                q += p * (r + gamma * V[s_next])

        q_values[action] = q

    return q_values


def value_iteration(
    villain_policy="uniform",
    state_space_policy="uniform",
    gamma=1.0,
    theta=1e-8,
    max_iters=1000,
    verbose=True,
):
    """
    Compute a deterministic best-response policy for hero against a fixed
    villain policy.

    villain_policy:
        The opponent policy used inside the Bellman backups.

    state_space_policy:
        The opponent policy used only to generate the set of hero states.

    For normal same-opponent solving, these can be the same.

    For cross-play experiments, use state_space_policy="uniform" so that
    all saved policies are defined on a common broad state space.
    """
    states = generate_reachable_states(villain_policy=state_space_policy)

    V = {s: 0.0 for s in states}
    policy = {}

    for it in range(max_iters):
        delta = 0.0

        for s in states:
            q_values = {}

            for action in LeducSimpleMDP.legal_actions_from_mdp(s):
                q = 0.0

                outs = LeducSimpleMDP.action_outcomes(
                    s,
                    action,
                    villain_policy=villain_policy,
                )

                for p, s_next, r, done in outs:
                    if done:
                        q += p * r
                    else:
                        if s_next not in V:
                            raise KeyError(
                                "Next state missing from value table.\n"
                                f"Current state: {s.to_tuple()}\n"
                                f"Action: {action}\n"
                                f"Next state: {s_next.to_tuple()}\n"
                                f"Try using state_space_policy='uniform'."
                            )

                        q += p * (r + gamma * V[s_next])

                q_values[action] = q

            best_action = max(q_values, key=q_values.get)
            best_value = q_values[best_action]

            delta = max(delta, abs(V[s] - best_value))

            V[s] = best_value
            policy[s] = best_action

        if verbose:
            print(f"iter={it}, delta={delta}")

        if delta < theta:
            if verbose:
                print(f"Converged after {it + 1} iterations.")
            break

    return V, policy


def print_root_diagnostics(V, policy, villain_policy="uniform", gamma=1.0):
    """
    Print the optimal action and Q-values from each initial root state.
    Useful for sanity-checking the solved policy.
    """
    roots = initial_states()

    for root in roots:
        print("\n" + "=" * 70)
        print("HAND ROOT")
        print("state:", root.to_tuple())
        print(
            f"hero_card={root.hero_card}, "
            f"villain_card={root.villain_card}, "
            f"hero_position={root.hero_position}"
        )
        print("=" * 70)

        if root.to_act == 0:
            print("Hero acts first preflop.")
            print("Legal actions:", LeducSimpleMDP.legal_actions_from_mdp(root))
            print("Best action:", policy[root])
            print(
                "Q-values:",
                q_values_for_state(
                    root,
                    V,
                    villain_policy=villain_policy,
                    gamma=gamma,
                ),
            )

        else:
            print("Hero is BB, so villain/button acts first preflop.")
            print(f"Villain policy: {villain_policy}")
            print("Outcomes after villain's first action:")

            outs = LeducSimpleMDP.env_transition_dist(
                root,
                villain_policy=villain_policy,
            )

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
                    print(
                        "Q-values:",
                        q_values_for_state(
                            s_next,
                            V,
                            villain_policy=villain_policy,
                            gamma=gamma,
                        ),
                    )

            print("\nTotal branch probability:", total_prob)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--villain-policy", type=str, default="uniform")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--save-filename",
        type=str,
        default=None,
        help="Filename under src/NashOrPass/leduc/agents/saved_policies",
    )

    args = parser.parse_args()

    if args.save_filename is None:
        args.save_filename = f"dp_policy_vs_{args.villain_policy}.pkl"

    V, policy = value_iteration(
        villain_policy=args.villain_policy,
        gamma=args.gamma,
    )

    print("=" * 70)
    print("DP Best Response")
    print("=" * 70)
    print(f"Villain policy:        {args.villain_policy}")
    print(f"Gamma:                 {args.gamma}")
    print(f"Number of states:      {len(V)}")
    print(f"Number of hero states: {len(policy)}")
    print("-" * 70)

    roots = initial_states()
    root_values = [V[root] for root in roots if root in V]

    if root_values:
        print(f"Average root value:    {sum(root_values) / len(root_values):.6f}")
        print(f"Min root value:        {min(root_values):.6f}")
        print(f"Max root value:        {max(root_values):.6f}")

    save_policy(policy, filename=args.save_filename)


if __name__ == "__main__":
    main()
