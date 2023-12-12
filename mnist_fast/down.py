import requests
url = "https://s3.amazonaws.com/img-datasets/mnist.npz"
myfile = requests.get(url)
open('mnist.npz', 'wb').write(myfile.content)
