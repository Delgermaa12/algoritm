from flask import Flask, request, jsonify, render_template
import geopandas as gpd
import math, time, psutil, os, heapq
from collections import deque, defaultdict

app = Flask(__name__)


class RoadNetworkGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)
        self.node_coords_to_id = {}
        self.next_node_id = 0

    def add_node(self, lon, lat):
        if (lon, lat) not in self.node_coords_to_id:
            node_id = self.next_node_id
            self.nodes[node_id] = (lon, lat)
            self.node_coords_to_id[(lon, lat)] = node_id
            self.next_node_id += 1
            return node_id
        return self.node_coords_to_id[(lon, lat)]

    def add_edge(self, node1, node2, weight, road_data=None):
        self.edges[node1].append((node2, weight, road_data or {}))
        self.edges[node2].append((node1, weight, road_data or {}))


def calculate_distance(c1, c2):
    R = 6371
    lon1, lat1 = c1
    lon2, lat2 = c2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RoadNetworkAnalyzer:
    def __init__(self, shapefile_path):
        try:
            self.gdf = gpd.read_file(shapefile_path)
            self.graph = RoadNetworkGraph()
            self.build_graph()
            print("Граф амжилттай бүтээгдлээ")
        except Exception as e:
            print(f"Shapefile уншихад алдаа гарлаа: {e}")
            # Жишээ өгөгдөл үүсгэх
            self.graph = RoadNetworkGraph()
            self._create_sample_data()

    def _create_sample_data(self):
        """Жишээ өгөгдөл үүсгэх"""
        print("Жишээ өгөгдөл үүсгэж байна...")
        coords = [
            (106.915, 47.920), (106.916, 47.921), (106.917, 47.919),
            (106.918, 47.922), (106.919, 47.920), (106.920, 47.923)
        ]

        for i in range(len(coords)):
            self.graph.add_node(*coords[i])

        for i in range(len(coords) - 1):
            dist = calculate_distance(coords[i], coords[i + 1])
            self.graph.add_edge(i, i + 1, dist, {'name': f'Жишээ зам {i + 1}'})

    def build_graph(self):
        print("Граф боловсруулж байна...")
        for _, road in self.gdf.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                for i in range(len(coords) - 1):
                    node1 = self.graph.add_node(*coords[i])
                    node2 = self.graph.add_node(*coords[i + 1])
                    dist = calculate_distance(coords[i], coords[i + 1])
                    road_data = {'name': road.get('name', 'unknown')}
                    self.graph.add_edge(node1, node2, dist, road_data)
        print(f"Граф үүссэн: {len(self.graph.nodes)} орой, {sum(len(v) for v in self.graph.edges.values())} ирмэг")

    def find_nearest_node(self, lon, lat):
        best, min_d = None, float("inf")
        for nid, (x, y) in self.graph.nodes.items():
            d = calculate_distance((lon, lat), (x, y))
            if d < min_d:
                min_d, best = d, nid
        return best

    def bfs(self, start, end):
        try:
            s = self.find_nearest_node(*start)
            e = self.find_nearest_node(*end)

            if s is None or e is None:
                return self._error('BFS', 'Эхлэх эсвэл төгсгөлийн цэг олдсонгүй')

            visited, queue = set([s]), deque([[s]])
            t0, mem0 = time.time(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            while queue:
                path = queue.popleft()
                node = path[-1]
                if node == e:
                    coords = [self.graph.nodes[n] for n in path]
                    return self._result('BFS', coords, time.time() - t0, mem0)
                for nb, _, _ in self.graph.edges[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(path + [nb])
            return self._error('BFS', 'Зам олдсонгүй')
        except Exception as e:
            return self._error('BFS', f'Алдаа: {str(e)}')

    def dijkstra(self, start, end):
        try:
            s = self.find_nearest_node(*start)
            e = self.find_nearest_node(*end)

            if s is None or e is None:
                return self._error('Dijkstra', 'Эхлэх эсвэл төгсгөлийн цэг олдсонгүй')

            dist = {n: float("inf") for n in self.graph.nodes}
            prev = {}
            dist[s] = 0
            pq = [(0, s)]
            t0, mem0 = time.time(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            while pq:
                d, node = heapq.heappop(pq)
                if node == e:
                    path = []
                    while node in prev:
                        path.append(node)
                        node = prev[node]
                    path.append(s)
                    path.reverse()
                    coords = [self.graph.nodes[n] for n in path]
                    return self._result('Dijkstra', coords, time.time() - t0, mem0)

                for nb, w, _ in self.graph.edges[node]:
                    nd = d + w
                    if nd < dist[nb]:
                        dist[nb] = nd
                        prev[nb] = node
                        heapq.heappush(pq, (nd, nb))
            return self._error('Dijkstra', 'Зам олдсонгүй')
        except Exception as e:
            return self._error('Dijkstra', f'Алдаа: {str(e)}')

    def dfs(self, start, end):
        try:
            s = self.find_nearest_node(*start)
            e = self.find_nearest_node(*end)

            if s is None or e is None:
                return self._error('DFS', 'Эхлэх эсвэл төгсгөлийн цэг олдсонгүй')

            visited = set()
            stack = [(s, [s])]
            t0, mem0 = time.time(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            while stack:
                node, path = stack.pop()
                if node == e:
                    coords = [self.graph.nodes[n] for n in path]
                    return self._result('DFS', coords, time.time() - t0, mem0)

                if node not in visited:
                    visited.add(node)
                    for nb, _, _ in self.graph.edges[node]:
                        if nb not in visited:
                            stack.append((nb, path + [nb]))

            return self._error('DFS', 'Зам олдсонгүй')
        except Exception as e:
            return self._error('DFS', f'Алдаа: {str(e)}')

    def _result(self, algo, path, t, mem0):
        mem1 = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        dist = sum(calculate_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
        return {
            'algorithm': algo,
            'path': path,
            'distance': dist,
            'execution_time': t,
            'memory_used': mem1 - mem0,
            'status': 'success'
        }

    def _error(self, algo, message='Зам олдсонгүй'):
        return {
            'algorithm': algo,
            'error': message,
            'status': 'error'
        }


shapefile_path = "data/gis_osm_roads_free_1.shp"
if not os.path.exists(shapefile_path):
    print(f"Анхаар: {shapefile_path} файл олдсонгүй. Жишээ өгөгдөл ашиглаж байна.")

analyzer = RoadNetworkAnalyzer(shapefile_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/bfs", methods=["POST"])
def bfs_api():
    try:
        data = request.get_json()
        if not data or 'start' not in data or 'end' not in data:
            return jsonify({'error': 'Эхлэх болон төгсгөлийн цэг шаардлагатай'}), 400
        res = analyzer.bfs(data["start"], data["end"])
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': f'Алдаа: {str(e)}'}), 500


@app.route("/api/dijkstra", methods=["POST"])
def dijkstra_api():
    try:
        data = request.get_json()
        if not data or 'start' not in data or 'end' not in data:
            return jsonify({'error': 'Эхлэх болон төгсгөлийн цэг шаардлагатай'}), 400
        res = analyzer.dijkstra(data["start"], data["end"])
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': f'Алдаа: {str(e)}'}), 500


@app.route("/api/dfs", methods=["POST"])
def dfs_api():
    try:
        data = request.get_json()
        if not data or 'start' not in data or 'end' not in data:
            return jsonify({'error': 'Эхлэх болон төгсгөлийн цэг шаардлагатай'}), 400
        res = analyzer.dfs(data["start"], data["end"])
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': f'Алдаа: {str(e)}'}), 500


@app.route("/api/info")
def info():
    return jsonify({
        "nodes": len(analyzer.graph.nodes),
        "edges": sum(len(v) for v in analyzer.graph.edges.values()),
        "status": "active"
    })


if __name__ == "__main__":
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        print("Templates хавтас үүсгэгдлээ")

    app.run(debug=True, port=5000)