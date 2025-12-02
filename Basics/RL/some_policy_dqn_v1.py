def select_action(state, policy="epsilon_greedy"):
    global steps_done
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # Q-values
    with torch.no_grad():
        q_values = policy_net(state_t).squeeze()

    # -----------------------------------------
    # 1) ε-GREEDY (stardard)
    # -----------------------------------------
    if policy == "epsilon_greedy":
        eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1.0 * steps_done / eps_decay)
        steps_done += 1
        if random.random() > eps_threshold:
            return q_values.argmax().item()
        else:
            return random.randrange(n_actions)

    # -----------------------------------------
    # 2) BOLTZMANN / SOFTMAX POLICY
    # -----------------------------------------
    if policy == "boltzmann":
        tau = 1.0  # temperature; The smaller the number, the more deterministic it is.
        probs = torch.softmax(q_values / tau, dim=0).numpy()
        return np.random.choice(np.arange(n_actions), p=probs)

    # -----------------------------------------
    # 3) ε-GREEDY INVERTED (favors rare action/fraud)
    # -----------------------------------------
    if policy == "epsilon_inverse":
        # When Q indicates legal class, force explore fraud.
        eps = 0.1
        best_action = q_values.argmax().item()

        # Discovering who is a "fraud"
        fraud_action = 1  # IF FraudEnv defines action 1 as fraud.
        legal_action = 0  # action 0 as legal

        if random.random() < eps:
            return fraud_action
        else:
            return best_action

    # -----------------------------------------
    # 4) UCB (Upper Confidence Bound)
    # -----------------------------------------
    if policy == "ucb":
        #Initialize counters
        if not hasattr(select_action, "counts"):
            select_action.counts = np.zeros(n_actions)
            select_action.total = 0

        c = 2  # exploitation force
        ucb_scores = []

        for a in range(n_actions):
            if select_action.counts[a] == 0:
                ucb_scores.append(float("inf"))
            else:
                bonus = c * np.sqrt(np.log(select_action.total + 1) / select_action.counts[a])
                ucb_scores.append(q_values[a].item() + bonus)

        action = int(np.argmax(ucb_scores))
        select_action.counts[action] += 1
        select_action.total += 1
        return action

    # -----------------------------------------
    # (fallback)
    # -----------------------------------------
    return q_values.argmax().item()
