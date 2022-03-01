pentagonal = []
def f(n):
	return int(n * (3 * n - 1) / 2)

for i in range(1,200):
	pentagonal += [f(i)]

print(pentagonal)
minimum = 10000
for i in range(1,len(pentagonal)):
    for j in range(i):
            sum = pentagonal[i] + pentagonal[j]
            dif = pentagonal[i] - pentagonal[j]
            print("i=",pentagonal[i],"j=",pentagonal[j],sum,dif)
            if sum in pentagonal and dif in pentagonal:
                print("cool")
            if dif < minimum:
                minimum = dif
print(minimum)