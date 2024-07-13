def calc_heuristic(x,y):
    return x+y

def calc_best_node(n):
    return min(1000,n)

def a_star_algorithm(start_node, stop_node, adjacency_list, H):
    open_list = set([start_node])
    closed_list = set([])

    g = {}
    g[start_node] = 0

    parents = {}
    parents[start_node] = start_node

    while open_list:
        n = None

        for v in open_list:
            if n is None or g[v] + H[v] < g[n] + H[n]:
                n = v

        if n is None:
            print('Path does not exist!')
            return None

        if n == stop_node:
            reconst_path = []

            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
                best_node=calc_best_node(g[n]+H[n])

            reconst_path.append(start_node)
            reconst_path.reverse()

            print('Path found: {}'.format(reconst_path))
            print(f'best node is: {}')
            return reconst_path

        for (m, weight) in adjacency_list.get(n, []):
            if m not in open_list and m not in closed_list:
                open_list.add(m)
                parents[m] = n

                g[m] = calc_heuristic(g[n],weight)
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n

                    if m in closed_list:
                        closed_list.remove(m)
                        open_list.add(m)

        open_list.remove(n)
        closed_list.add(n)

    print('Path does not exist!')
    return None

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

    path = a_star_algorithm(start_node, end_node, adjacency_list, H)
    if path:
        print(f'Path is: {path}')
