import matplotlib

matplotlib.use('TkAgg')

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from tensorflow.keras.models import load_model

# --- AI PIPELINE INITIALIZATION ---
print("INITIALIZING AI ENGINE: Loading weights and datasets...")
try:
    model = load_model('proposed_model.h5')
    live_data = np.load('grid_data.npy')
    print("AI ENGINE ONLINE: Deep Learning Pipeline Active.")
except Exception as e:
    print(f"CRITICAL PIPELINE FAILURE: {e}")
    exit()

# --- DEMO CONFIGURATION ---
NUM_GENERATORS = 3
NUM_SUBSTATIONS = 6
NUM_CONSUMERS = 10
TIME_STEP = 60

SPIKE_THRESHOLD = 0.32  # Set lower for hyper-sensitive demo mode


def create_grid():
    G = nx.DiGraph()
    generators = [f"G{i}" for i in range(NUM_GENERATORS)]
    substations = [f"S{i}" for i in range(NUM_SUBSTATIONS)]
    consumers = [f"C{i}" for i in range(NUM_CONSUMERS)]

    for g in generators: G.add_node(g, type="generator", capacity=random.randint(250, 400))
    for s in substations: G.add_node(s, type="substation")

    # NEW: Assign random priority tiers (1 to 5) to ALL consumers
    for c in consumers: G.add_node(c, type="consumer", priority=random.randint(1, 5))

    for g in generators:
        for s in random.sample(substations, k=3):
            G.add_edge(g, s, capacity=random.randint(100, 200), flow=0)
    for s in substations:
        others = [x for x in substations if x != s]
        for o in random.sample(others, k=2):
            G.add_edge(s, o, capacity=random.randint(80, 150), flow=0)
    for c in consumers:
        for s in random.sample(substations, k=2):
            G.add_edge(s, c, capacity=random.randint(50, 100), flow=0)

    return G, generators, substations, consumers


def layered_layout(generators, substations, consumers):
    pos = {}
    g_space, s_space, c_space = 10, 7, 4
    g_offset = -(len(generators) - 1) * g_space / 2
    s_offset = -(len(substations) - 1) * s_space / 2
    c_offset = -(len(consumers) - 1) * c_space / 2
    for i, g in enumerate(generators): pos[g] = (g_offset + i * g_space, 10)
    for i, s in enumerate(substations): pos[s] = (s_offset + i * s_space, 5)
    for i, c in enumerate(consumers): pos[c] = (c_offset + i * c_space, 0)
    return pos


def distribute_energy(G, generators, consumers, demands):
    total_supply = sum(G.nodes[g]["capacity"] for g in generators)
    total_demand = sum(demands[c] for c in consumers)
    allocation = {}
    if total_supply >= total_demand:
        for c in consumers: allocation[c] = demands[c]
    else:
        weighted_sum = sum(demands[c] * G.nodes[c]["priority"] for c in consumers)
        for c in consumers:
            if weighted_sum == 0:
                allocation[c] = 0
            else:
                share = total_supply * (demands[c] * G.nodes[c]["priority"]) / weighted_sum
                allocation[c] = min(demands[c], int(share))

    for u, v in G.edges(): G[u][v]["flow"] = 0
    temp = G.copy()
    temp.add_node("SuperSource")
    temp.add_node("SuperSink")
    for g in generators: temp.add_edge("SuperSource", g, capacity=G.nodes[g]["capacity"])
    for c in consumers: temp.add_edge(c, "SuperSink", capacity=allocation[c])
    _, flow_dict = nx.maximum_flow(temp, "SuperSource", "SuperSink")
    for u in flow_dict:
        for v in flow_dict[u]:
            if G.has_edge(u, v): G[u][v]["flow"] = flow_dict[u][v]
    return allocation


G, generators, substations, consumers = create_grid()
pos = layered_layout(generators, substations, consumers)
fig, ax = plt.subplots(figsize=(16, 9))
fig.canvas.manager.set_window_title('SGMS: Dynamic Priority Routing')


def animate(frame):
    global TIME_STEP
    ax.clear()

    if TIME_STEP >= len(live_data):
        TIME_STEP = 60

    window = live_data[TIME_STEP - 60: TIME_STEP, :]
    tensor_input = np.expand_dims(window, axis=0)

    raw_prediction = model.predict(tensor_input, verbose=0)
    predicted_value = float(raw_prediction[0][0])

    print(f"Step {TIME_STEP} | Raw AI Output: {predicted_value:.4f}")

    demands = {c: random.randint(20, 50) for c in consumers}
    critical_node = None  # Initialize empty for rendering logic

    if predicted_value > SPIKE_THRESHOLD:
        # --- THE DYNAMIC ROUTING ALGORITHM ---
        # 1. Calculate Severity Score for all nodes (Demand * Priority)
        severity_scores = {c: demands[c] * G.nodes[c]["priority"] for c in consumers}

        # 2. Identify the node in the most critical danger
        critical_node = max(severity_scores, key=severity_scores.get)
        critical_priority = G.nodes[critical_node]['priority']

        ai_status = f"AI WARNING: ANOMALY (Val: {predicted_value:.2f})\nREROUTING TO CRITICAL NODE: {critical_node} (Pri: {critical_priority})"
        ai_color = "red"

        # 3. Dynamically route emergency power to the critical node
        scaled_demand = int(predicted_value * 200) if predicted_value < 1.0 else int(predicted_value)
        demands[critical_node] = max(150, scaled_demand)

        for u, v in G.edges():
            if v == critical_node:
                G[u][v]["capacity"] = 350
            elif G.nodes[v]["type"] == "consumer":
                G[u][v]["capacity"] = 100
    else:
        ai_status = f"AI MONITOR | LIVE INFERENCE: {predicted_value:.2f}"
        ai_color = "green"
        for u, v in G.edges():
            if G.nodes[v]["type"] == "consumer":
                G[u][v]["capacity"] = 100

    alloc = distribute_energy(G, generators, consumers, demands)
    flows = [G[u][v]["flow"] for u, v in G.edges()]
    widths = [f / 15 + 1 for f in flows]

    sizes, colors = [], []
    for n in G.nodes():
        typ = G.nodes[n]["type"]
        if typ == "generator":
            sizes.append(1400); colors.append("#ff4d4d")
        elif typ == "substation":
            sizes.append(800); colors.append("#ffa64d")
        else:
            sizes.append(400)
            # Highlight the dynamically chosen node
            if n == critical_node and predicted_value > SPIKE_THRESHOLD:
                colors.append("yellow")
            else:
                colors.append("#4da6ff")

    nx.draw(G, pos, ax=ax, width=widths, node_size=sizes, node_color=colors, with_labels=True, font_weight='bold')

    for n, (x, y) in pos.items():
        node = G.nodes[n]
        if node["type"] == "generator":
            txt = f"Cap:{node['capacity']}"
        elif node["type"] == "substation":
            incoming = sum(G[u][n]["flow"] for u in G.predecessors(n))
            txt = f"\nFlow:{incoming}"
        else:
            incoming = sum(G[u][n]["flow"] for u in G.predecessors(n))
            priority_val = G.nodes[n]['priority']
            # Output the Priority on the screen so the HOD can see the math
            txt = f"\nPri:{priority_val}\nReq:{demands[n]}\nGot:{incoming}"
        ax.text(x, y - 1.2, txt, fontsize=9, ha="center", bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_title(f"LIVE DATA INTEGRATION: LSTM PREDICTIVE ROUTING\nStep: {TIME_STEP}", fontsize=14, pad=20)
    ax.text(0.5, 0.95, ai_status, transform=ax.transAxes, fontsize=12, ha='center', va='center',
            color='white', fontweight='bold', bbox=dict(facecolor=ai_color, alpha=0.8, pad=10))

    TIME_STEP += 1


ani = animation.FuncAnimation(fig, animate, frames=60, interval=1000)
plt.show()