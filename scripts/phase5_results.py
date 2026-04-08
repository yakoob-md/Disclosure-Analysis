import os
import pickle
from collections import Counter

def main():
    print("="*50)
    print("PHASE 5 VALIDATION: KNOWLEDGE GRAPH TOPOLOGY")
    print("="*50)

    kg_file = "graphs/knowledge_graph.pkl"
    if not os.path.exists(kg_file):
        print("KG not built yet. Provide neo4j validation logic or run phase 5.")
        return

    with open(kg_file, 'rb') as f:
        KG = pickle.load(f)

    print("\n=== GRAPH SIZE ===")
    print(f"Total Nodes: {KG.number_of_nodes():,}")
    print(f"Total Edges: {KG.number_of_edges():,}")

    print("\n=== NODE TYPES ===")
    n_types = Counter([data.get('node_type','?') for _, data in KG.nodes(data=True)])
    for k, v in n_types.items():
        print(f"  {k:15s}: {v:,}")

    print("\n=== EDGE TYPES ===")
    e_types = Counter([data.get('relation','?') for _, _, data in KG.edges(data=True)])
    for k, v in e_types.items():
        print(f"  {k:20s}: {v:,}")

    print("\n=== SAMPLE CONNECTIONS ===")
    count = 0
    for u, v, data in KG.edges(data=True):
        if data.get('relation') == 'SENT':
            print(f"> Sender: {u}  --[SENT]-->  Email: {v}")
            count += 1
            if count >= 3: break

    with open('results/phase5/summary.txt', 'w') as f:
        f.write(f"Phase 5\nNodes: {KG.number_of_nodes()}\nEdges: {KG.number_of_edges()}\n")
    print("\nReport saved to results/phase5/summary.txt")

if __name__ == "__main__":
    main()
