from statsmodels.stats.proportion import proportion_confint as binofit
import sys

acc = float(sys.argv[1])

# 0.01 confidence intervals
easy_interval = binofit(acc * 400, 400, 0.01)
print "0.01 interval = ", easy_interval
strict_interval = binofit(acc * 400, 400, 0.001)
print "0.001 interval = ", strict_interval
