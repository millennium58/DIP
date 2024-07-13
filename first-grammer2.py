productions = {}
first_set = {}

def calculate_first(non_terminal):
    if non_terminal in first_set:
        return first_set[non_terminal]
    result = set()
    for production in productions[non_terminal]:
        strings = production.split('|')
        for str in strings:
            if 'id' in str:
                result.add('id')
            first_char = str.strip()[0]
            if first_char.isupper():
                first_of_symbol = calculate_first(first_char)
                result.update(first_of_symbol)
                if '#' not in first_of_symbol:
                    break
            else:
                result.add(first_char)
    first_set[non_terminal] = result
    return result

if __name__ == "__main__":
    productions['X'] = ['Tns|Rm|id']
    productions['T'] = ['q|#']
    productions['S'] = ['p|#']
    productions['R'] = ['om|ST']

    for non_terminal in productions.keys():
        calculate_first(non_terminal)

    for non_terminal, first_set in first_set.items():
        print(f'FIRST({non_terminal}) = {{ {", ".join(first_set)} }}')
=