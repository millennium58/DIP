def ida_star_algorithm(start_node, end_node, adjacency_list, H):
    def search(node, g, threshold):
        h = H[node]
        f = g + h
        if f > threshold:
            return f
        if node == end_node:
            return 'Found'
        min_val = float('inf')
        next_threshold = float('inf')

        for neighbor, cost in adjacency_list[node]:
            if neighbor not in closed_list:
                closed_list.add(neighbor)
                parent[neighbor] = node
                result = search(neighbor, g + cost, threshold)
                if result == 'Found':
                    return 'Found'
                if result < next_threshold:
                    next_threshold = result

        return next_threshold

    threshold = H[start_node]
    parent = {}
    while True:
        closed_list = set([start_node])
        result = search(start_node, 0, threshold)
        if result == 'Found':
            reconst_path = []
            n = end_node
            while n != start_node:
                reconst_path.append(n)
                n = parent[n]
            reconst_path.append(start_node)
            reconst_path.reverse()
            print(f'Path is found: {reconst_path}')
            return reconst_path
        if result == float('inf'):
            print('Path does not exist')
            return None
        threshold = result

if __name__ == '__main__':
    start_node = 'A'
    end_node = 'D'

    adjacency_list = {
        'A': [('B', 1), ('C', 3)],
        'B': [('D', 5)],
        'C': [('D', 12)],
        'D': []
    }

    H = {
        'A': 1,
        'B': 1,
        'C': 1,
        'D': 1
    }
    path = ida_star_algorithm(start_node, end_node, adjacency_list, H)
    if path:
        print(f'Path is found: {path}')
