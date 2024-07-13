if __name__ == '__main__':
    t = int(input())
    while t > 0:
        foo=True
        string = 'a*b+c*'
        x = input()

        if 'b' not in x:
            print('invalid')
        elif x == 'b':
            print('valid')
        else:
            valid = True
            for i in range(len(x) - 1):
                if x[i] == 'b' and x[i + 1] == 'a'  or x[i]=='c' and x[i+1]=='b' or x[i]=='c' and x[i+1]=='a':
                    print('invalid')
                    foo=False
                    break;


            if foo:
                print('valid')

        t -= 1


