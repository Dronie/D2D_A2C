import math

class prob:
    def probability(self, n, c):
            form = []
            for i in range(0, n):
                form.append( math.log((c-i) / c) )
            print(math.pow(math.e, sum(form)))



