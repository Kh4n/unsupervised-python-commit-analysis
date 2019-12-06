import matplotlib.pyplot as plt

with open("larg2_output", 'r') as f:
    nums = [float(k) for k in f.readlines()]

plt.plot(nums)
plt.xlabel("commit")
plt.ylabel("reconstruction error")
plt.show()