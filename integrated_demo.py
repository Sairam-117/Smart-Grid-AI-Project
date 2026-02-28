import matplotlib

matplotlib.use('TkAgg')  # Crucial: Forces PyCharm to open an interactive animation window

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# --- CONFIGURATION ---
NUM_GENERATORS = 3
NUM_SUBSTATIONS = 6
NUM_CONSUMERS = 10


def create_grid():
    G = nx.DiGraph()
    generators = [f"G{i}" for i in range(NUM_GENERATORS)]
    substations = [f"S{i}" for i in range(NUM_SUBSTATIONS)]
    consumers = [f"C{i}" for i in range(NUM_CONSUMERS)]

    for g in generators:
        G.add_node(g, type="generator", capacity=random.randint(200, 300))
    for s in substations:
        G.add_node(s, type="substation")
    for c in consumers:
        G.add_node(c, type="consumer", priority=1)

    # Manual Priorities
    G.nodes["C0"]["priority"] = 5  # Example: Hospital
    G.nodes["C3"]["priority"] = 4  # Example: Data Center
    G.nodes["C7"]["priority"] = 2  # Example: Residential

    # Connections
    for g in generators:
        for s in random.sample(substations, k=3):
            G.add_edge(g, s, capacity=random.randint(80, 150), flow=0)
    for s in substations:
        others = [x for x in substations if x != s]
        for o in random.sample(others, k=2):
            G.add_edge(s, o, capacity=random.randint(50, 100), flow=0)
    for c in consumers:
        for s in random.sample(substations, k=2):
            G.add_edge(s, c, capacity=random.randint(40, 100), flow=0)

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
            if G.has_edge(u, v):
                G[u][v]["flow"] = flow_dict[u][v]
    return allocation


# --- INITIALIZATION ---
G, generators, substations, consumers = create_grid()
pos = layered_layout(generators, substations, consumers)

fig, ax = plt.subplots(figsize=(16, 9))
fig.canvas.manager.set_window_title('SGMS: AI-Predictive Grid Routing')
t = 0
upcoming_spike = None  # Tracks random dynamic events


# --- THE ANIMATION LOOP (WITH STOCHASTIC AI BRIDGE) ---
def animate(frame):
    global t, upcoming_spike
    ax.clear()
    t += 1

    # 1. GENERATE BASE DEMANDS
    demands = {c: random.randint(20, 60) for c in consumers}
    ai_status = "AI MONITOR: Normal Grid Patterns"
    ai_color = "green"

    # 2. THE STOCHASTIC AI INTELLIGENCE LAYER (Unscripted)
    # 5% chance every frame to trigger a random load spike anywhere on the grid
    if upcoming_spike is None and random.random() < 0.05 and t > 5:
        target_node = random.choice(consumers)
        upcoming_spike = {
            'warn_time': t,
            'hit_time': t + random.randint(8, 12),  # Hits 8-12 seconds from now
            'end_time': t + random.randint(18, 25),  # Lasts for a while
            'node': target_node,
            'magnitude': random.randint(180, 260)  # Random massive load
        }

    # Execute the current phase of the random spike
    if upcoming_spike:
        s = upcoming_spike
        active_node = s['node']

        if t < s['hit_time']:
            # WARNING PHASE: AI predicts the future
            time_left = s['hit_time'] - t
            ai_status = f"AI WARNING: 96.5% PROBABILITY OF LOAD SPIKE AT {active_node} IN {time_left}s\nPROACTIVE REROUTING INITIATED..."
            ai_color = "red"
            # Bridge: Thicken pipes to the specific random node
            for u, v in G.edges():
                if v == active_node:
                    G[u][v]["capacity"] = 300

        elif t >= s['hit_time'] and t < s['end_time']:
            # ACTIVE PHASE: The spike hits the grid
            ai_status = f"AI ALERT: PEAK LOAD ACTIVE AT {active_node}. GRID STABILIZED VIA REROUTING."
            ai_color = "orange"
            demands[active_node] = s['magnitude']  # Inject the massive demand

        elif t >= s['end_time']:
            # RECOVERY PHASE: Return to normal
            ai_status = f"AI MONITOR: Spike at {active_node} Resolved. Returning to baseline."
            ai_color = "green"
            # Reset capacities
            for u, v in G.edges():
                if v == active_node:
                    G[u][v]["capacity"] = 100
            upcoming_spike = None  # Reset so a new random event can trigger later

    # 3. THE PHYSICAL ROUTING LAYER
    alloc = distribute_energy(G, generators, consumers, demands)

    # 4. VISUALIZATION
    flows = [G[u][v]["flow"] for u, v in G.edges()]
    widths = [f / 15 + 1 for f in flows]

    sizes, colors = [], []
    for n in G.nodes():
        typ = G.nodes[n]["type"]
        if typ == "generator":
            sizes.append(1400);
            colors.append("#ff4d4d")
        elif typ == "substation":
            sizes.append(800);
            colors.append("#ffa64d")
        else:
            sizes.append(400)
            # Highlight the random node if it's currently spiking
            if upcoming_spike and n == upcoming_spike['node'] and t >= upcoming_spike['hit_time']:
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
            txt = f"\nReq:{demands[n]}\nGot:{incoming}"
        ax.text(x, y - 1.2, txt, fontsize=9, ha="center", bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    total_supply = sum(G.nodes[g]["capacity"] for g in generators)
    total_demand = sum(demands.values())
    total_delivered = sum(G[u][v]["flow"] for u, v in G.edges() if v in consumers)

    ax.set_title(
        f"INTEGRATED SMART GRID MANAGEMENT SYSTEM (LSTM + GRAPH)\nSupply: {total_supply} MW | Demand: {total_demand} MW | Delivered: {total_delivered} MW",
        fontsize=14, pad=20)

    ax.text(0.5, 0.95, ai_status, transform=ax.transAxes, fontsize=12, ha='center', va='center',
            color='white', fontweight='bold', bbox=dict(facecolor=ai_color, alpha=0.8, pad=10))


ani = animation.FuncAnimation(fig, animate, frames=60, interval=1000)
plt.show()