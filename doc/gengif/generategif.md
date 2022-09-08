# GIF Generation

In order to generate .gif file which shows the changing process of the Simulation, we create two Notebooks to reach the goal.

## save_plot.ipynb

The following snippet of code is used to list the concequences of a simulation which are store in the numpy.npy format.

```python
# list the ".npy" files
def GetFileList(dir, fileList): 
    if os.path.isfile(dir): 
        print("input must be a dir!")
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            fileList.append(s)
    fileList = [t for t in fileList if '.npy' in t]
    fileList.remove("final.npy")
    list_final = [str(i+1) + ".npy" for i in range(len(fileList))]
    list_final.append("final.npy")
    return list_final

npy_list = GetFileList('./iterations/', [])
print(npy_list)
```

By using the **continuous_model** ,we just plot the mesh of the simulation. The plots show the changing of the mesh during the simulation.

```python
# Define several parameters
Ms = 3.84e5

for file in npy_list:
    arr = np.load('./iterations/' + file)
    nx = arr.shape[0]
    ny = arr.shape[1]
    nz = arr.shape[2]


    my_cool_mesh = RectangularMesh(nx=nx, ny=ny, nz=nz, units=5e-9)
    my_cool_m = m_Field(my_cool_mesh, Ms, arr)
    
    save_name = file.split(".")[0]
    field = plot_field(my_cool_mesh, my_cool_m, 'z', value=10e-9, save_path="./plots/", save_name=save_name)
```

## visualization.ipynb

The **save_plot.ipynb** file is just saved as a json-like file.So we can use json method to load the data stream of it.

```python
js = json.load(open('save_plot.ipynb','r',encoding='utf-8'))
```

By solving the data, we have found that the plots are stored in a base64 stream in the notebook under 'image/png' attribute path. So we create a class to quickly locate and read the data stream.

```python
from typing import List
class JsonPathFinder:
    def __init__(self, json_content, mode='key'):
        self.data = json_content
        self.mode = mode

    def iter_node(self, rows, road_step, target):
        if isinstance(rows, dict):
            key_value_iter = (x for x in rows.items())
        elif isinstance(rows, list):
            key_value_iter = (x for x in enumerate(rows))
        else:
            return
        for key, value in key_value_iter:
            current_path = road_step.copy()
            current_path.append(key)
            if self.mode == 'key':
                check = key
            else:
                check = value
            if check == target:
                yield current_path
            if isinstance(value, (dict, list)):
                yield from self.iter_node(value, current_path, target)

    def find_one(self, target: str) -> list:
        path_iter = self.iter_node(self.data, [], target)
        for path in path_iter:
            return path
        return []

    def find_all(self, target) -> List[list]:
        path_iter = self.iter_node(self.data, [], target)
        return list(path_iter)
```

Now we can get the base64 stream of the plots and convert to .png format saved.

```python
print('Start the Search by Key...')
finder = JsonPathFinder(js)
path_list = finder.find_all('image/png')
print('Search is completed')

for i in range(len(path_list)):
    content = js[path_list[i][0]][path_list[i][1]]
    image = content[path_list[i][2]][path_list[i][3]][path_list[i][4]][path_list[i][5]]
    image_data = base64.b64decode(image)
    
    image_url = "./pics/" + str(i + 1) + ".png"
    
    with open(image_url, 'wb') as f:
        f.write(image_data)

print("Finished saving!")
```

After all of that, a gif can be created by using **imageio** package.

```python
path = "./pics/"
pathes = []


# list the ".npy" files
def GetFileList(dir, fileList): 
    if os.path.isfile(dir): 
        print("please input a dir!")
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            fileList.append(s)
    fileList = [t for t in fileList if '.png' in t]
    list_final = [str(i+1) + ".png" for i in range(len(fileList))]
    return list_final

file_list = GetFileList("./pics/", [])
print(file_list)


images=[]
for pic in file_list:
    images.append(imageio.imread("./pics/" + pic))
 
imageio.mimsave('./pics/plane.gif',images,duration=0.15)
```