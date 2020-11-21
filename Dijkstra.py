import xml.etree.ElementTree as ET
import dijkstra
from haversine import haversine  # distance on Earth surface


class Dijkstra:
    def __init__(self):
        self.root = ET.parse("roads_fixed.xml").getroot()

        self.xs = []
        self.ys = []

        self.edges = {}
        self.mapa = {}
        self.reverse_mapa = {}
        self.prep()
        self.d_graph = self.get_dist_graph()
        self.full_t_graph = self.get_time_graph()
        self.half_t_graph = self.get_time_graph(True)

    def prep(self):

        i = 0
        for v in self.root.findall("N"):

            self.xs.append(float(v.get("Lat")))
            self.ys.append(float(v.get("Long")))
            can_reach = []
            for f in v.findall("L/TN"):
                ssl = 60
                if f.get("SL") is not None:
                    ssl = int(f.get("SL"))
                can_reach.append(
                    [
                        (float(f.get("Lat")), float(f.get("Long"))),
                        ("time", (f.get("SCH") != None)),
                        ("sl", ssl),
                    ]
                )

            self.mapa[(float(v.get("Lat")), float(v.get("Long")))] = i
            self.reverse_mapa[i] = (float(v.get("Lat")), float(v.get("Long")))
            i += 1
            self.edges.update(
                [((float(v.get("Lat")), float(v.get("Long"))), can_reach)]
            )

    # Расстояние в километрах
    def distance(self, x, y):
        return haversine(x, y)

    # Расстояние в минутах
    def time_dist(self, x, y):
        return (haversine(x, y[0]) / float(y[2][1])) * 60

    def get_dist_graph(self):
        graph = dijkstra.Graph()

        for i in self.edges:
            for j in self.edges[i]:
                graph.add_edge(i, j[0], self.distance(i, j[0]))

        return graph

    def get_time_graph(self, time=False):
        graph = dijkstra.Graph()

        for i in self.edges:
            for j in self.edges[i]:
                if time and j[1][1]:
                    continue
                graph.add_edge(i, j[0], self.time_dist(i, j))

        return graph

    def get_dist(self, point, another_point, cur_time=12 * 60):
        dijk_d = dijkstra.DijkstraSPF(self.d_graph, point)
        dijk_ft = dijkstra.DijkstraSPF(self.full_t_graph, point)
        dijk_ht = dijkstra.DijkstraSPF(self.half_t_graph, point)
        timie = 0
        if cur_time < 8 * 60 or cur_time > 20 * 60:
            timie = dijk_ht.get_distance(another_point)
        else:
            timie = dijk_ft.get_distance(another_point)
        return (dijk_d.get_distance(another_point), timie)


some = Dijkstra()

print(some.get_dist(some.reverse_mapa[0], some.reverse_mapa[7]))
