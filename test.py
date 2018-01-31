from os.path import basename
import os

print(os.path.splitext(basename("/a/b/c.txt"))[0])

# now you can call it directly with basename
print(basename("/a/b/c.txt"))
