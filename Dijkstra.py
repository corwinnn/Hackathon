import xml.etree.ElementTree as ET
import dijkstra
from haversine import haversine # distance on Earth surface

root = ET.parse('roads_fixed.xml').getroot()

xs = []
ys = []

edges = {}
mapa = {}
reverse_mapa = {}

i = 0
for v in root.findall('N'):
    
    xs.append(float(v.get('Lat')))
    ys.append(float(v.get('Long')))
    can_reach = []
    for f in v.findall('L/TN'):
        ssl = 60
        if f.get('SL') != None:
            ssl = int(f.get('SL'))
        can_reach.append([(float(f.get('Lat')), float(f.get('Long'))), ('time', (f.get('SCH') != None)), ('sl', ssl)])
                         
    mapa[(float(v.get('Lat')), float(v.get('Long')))] = i
    reverse_mapa[i] = (float(v.get('Lat')), float(v.get('Long')))
    i += 1
    edges.update([((float(v.get('Lat')), float(v.get('Long'))), can_reach)])


# Расстояние в километрах
def distance(x, y):
    return  haversine(x, y)
# Расстояние в минутах
def time_dist(x, y):
    return (haversine(x, y[0]) / float(y[2][1]))*60

def get_dist_graph():
    graph = dijkstra.Graph()

    for i in edges:
        for j in edges[i]:
            graph.add_edge(i, j[0], distance(i, j[0]))
    
            
    return graph

def get_time_graph(time = False):
    graph = dijkstra.Graph()

    for i in edges:
        for j in edges[i]:
            if(time and j[1][1]):
                continue
            graph.add_edge(i, j[0], time_dist(i, j))
    
    return graph

d_graph = get_dist_graph()
full_t_graph = get_time_graph()
half_t_graph = get_time_graph(True)

def get_dist(point, another_point, cur_time=12*60):
    dijk_d = dijkstra.DijkstraSPF(d_graph, point)
    dijk_ft = dijkstra.DijkstraSPF(full_t_graph, point)
    dijk_ht = dijkstra.DijkstraSPF(half_t_graph, point)
    timie = 0
    if(cur_time < 8*60 or cur_time > 20*60):
        timie = dijk_ht.get_distance(another_point)
    else:
        timie = dijk_ft.get_distance(another_point)
    return (dijk_d.get_distance(another_point), timie)

print(get_dist(reverse_mapa[0], reverse_mapa[7]))