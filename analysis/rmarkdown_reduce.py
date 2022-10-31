f = open(r'C:\Users\jbos1\Desktop\Projects\Kaggle\spaceship-titanic\analysis\exploration.Rmd', 'r')
lines = f.readlines()
for i in range(len(lines)):
    if lines[i][0] == '#':
        if (i > 0) and (lines[i-1][0:5] == '```{r'):
            continue
        if lines[i][1] != '#':
            print('\n', end='')
        print(f'{i:4}: {lines[i]}', end='')