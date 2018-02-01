from os.path import basename
import os

print(os.path.splitext(basename("/a/b/c.txt"))[0])

# now you can call it directly with basename
print(basename("/a/b/c.txt"))


from sklearn.model_selection import train_test_split

a = [1,2,3]
b= [4,5,6]
c= [7,8,9]

split = train_test_split(a,b,c,test_size=1)

print(split)
